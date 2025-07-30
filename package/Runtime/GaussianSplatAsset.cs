// SPDX-License-Identifier: MIT

using System;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;
using System.Collections.Generic;
using Unity.Collections;

using UnityEngine.Experimental.Rendering;

namespace GaussianSplatting.Runtime
{
    public class GaussianSplatAsset : ScriptableObject, IGaussianSplatAsset
    {
        public const int kCurrentVersion = 2023_10_20;
        public const int kChunkSize = 256;
        public const int kTextureWidth = 2048; // allows up to 32M splats on desktop GPU (2k width x 16k height)
        public const int kMaxSplats = 8_600_000; // mostly due to 2GB GPU buffer size limit when exporting a splat (2GB / 248B is just over 8.6M)

        [SerializeField] int m_FormatVersion;
        [SerializeField] int m_SplatCount;
        [SerializeField] Vector3 m_BoundsMin;
        [SerializeField] Vector3 m_BoundsMax;
        [SerializeField] Hash128 m_DataHash;

        public int formatVersion => m_FormatVersion;
        public int splatCount => m_SplatCount;
        public Vector3 boundsMin => m_BoundsMin;
        public Vector3 boundsMax => m_BoundsMax;
        public Hash128 dataHash => m_DataHash;

        public bool isGaMeS_asset => m_AlphaData != null && m_ScaleData != null;

        // Match VECTOR_FMT_* in HLSL
        public enum VectorFormat
        {
            Float32, // 12 bytes: 32F.32F.32F
            Norm16, // 6 bytes: 16.16.16
            Norm11, // 4 bytes: 11.10.11
            Norm6   // 2 bytes: 6.5.5
        }

        public static int GetVectorSize(VectorFormat fmt)
        {
            return fmt switch
            {
                VectorFormat.Float32 => 12,
                VectorFormat.Norm16 => 6,
                VectorFormat.Norm11 => 4,
                VectorFormat.Norm6 => 2,
                _ => throw new ArgumentOutOfRangeException(nameof(fmt), fmt, null)
            };
        }

        public enum ColorFormat
        {
            Float32x4,
            Float16x4,
            Norm8x4,
            BC7,
        }
        public static int GetColorSize(ColorFormat fmt)
        {
            return fmt switch
            {
                ColorFormat.Float32x4 => 16,
                ColorFormat.Float16x4 => 8,
                ColorFormat.Norm8x4 => 4,
                ColorFormat.BC7 => 1,
                _ => throw new ArgumentOutOfRangeException(nameof(fmt), fmt, null)
            };
        }

        public enum SHFormat
        {
            Float32,
            Float16,
            Norm11,
            Norm6,
            Cluster64k,
            Cluster32k,
            Cluster16k,
            Cluster8k,
            Cluster4k,
        }

        public struct SHTableItemFloat32
        {
            public float3 sh1, sh2, sh3, sh4, sh5, sh6, sh7, sh8, sh9, shA, shB, shC, shD, shE, shF;
            public float3 shPadding; // pad to multiple of 16 bytes
        }
        public struct SHTableItemFloat16
        {
            public half3 sh1, sh2, sh3, sh4, sh5, sh6, sh7, sh8, sh9, shA, shB, shC, shD, shE, shF;
            public half3 shPadding; // pad to multiple of 16 bytes
        }
        public struct SHTableItemNorm11
        {
            public uint sh1, sh2, sh3, sh4, sh5, sh6, sh7, sh8, sh9, shA, shB, shC, shD, shE, shF;
        }
        public struct SHTableItemNorm6
        {
            public ushort sh1, sh2, sh3, sh4, sh5, sh6, sh7, sh8, sh9, shA, shB, shC, shD, shE, shF;
            public ushort shPadding; // pad to multiple of 4 bytes
        }


        public void Initialize(int splats, VectorFormat formatPos, VectorFormat formatScale, ColorFormat formatColor, SHFormat formatSh, Vector3 bMin, Vector3 bMax, CameraInfo[] cameraInfos, string pointCloudPath)
        {
            m_SplatCount = splats;
            m_FormatVersion = kCurrentVersion;
            m_PosFormat = formatPos;
            m_ScaleFormat = formatScale;
            m_ColorFormat = formatColor;
            m_SHFormat = formatSh;
            m_Cameras = cameraInfos;
            m_BoundsMin = bMin;
            m_BoundsMax = bMax;
            m_PointCloudPath = pointCloudPath;
        }

        Hash128 ComputeHashFromNativeArray(NativeArray<uint> data)
        {
            byte[] byteArray = new byte[data.Length * sizeof(uint)];
            System.Buffer.BlockCopy(data.ToArray(), 0, byteArray, 0, byteArray.Length);
            return Hash128.Compute(Convert.ToBase64String(byteArray)); // use string overload
        }

        Hash128 ComputeHash(byte[] data)
        {
            return Hash128.Compute(data);
        }

        public void SetDataHash(Hash128 hash)
        {
            m_DataHash = hash;
        }

        public void SetAssetFiles(TextAsset dataChunk, TextAsset dataPos, TextAsset dataOther, TextAsset dataColor, TextAsset dataSh, TextAsset dataAlpha = null, TextAsset dataScale = null)
        {
            m_ChunkData = dataChunk;
            m_PosData = dataPos;
            m_AlphaData = dataAlpha;
            m_ScaleData = dataScale;
            m_OtherData = dataOther;
            m_ColorData = dataColor;
            m_SHData = dataSh;
        }

        public static int GetOtherSizeNoSHIndex(VectorFormat scaleFormat)
        {
            return 4 + GetVectorSize(scaleFormat);
        }

        public static int GetSHCount(SHFormat fmt, int splatCount)
        {
            return fmt switch
            {
                SHFormat.Float32 => splatCount,
                SHFormat.Float16 => splatCount,
                SHFormat.Norm11 => splatCount,
                SHFormat.Norm6 => splatCount,
                SHFormat.Cluster64k => 64 * 1024,
                SHFormat.Cluster32k => 32 * 1024,
                SHFormat.Cluster16k => 16 * 1024,
                SHFormat.Cluster8k => 8 * 1024,
                SHFormat.Cluster4k => 4 * 1024,
                _ => throw new ArgumentOutOfRangeException(nameof(fmt), fmt, null)
            };
        }

        public static (int, int) CalcTextureSize(int splatCount)
        {
            int width = kTextureWidth;
            int height = math.max(1, (splatCount + width - 1) / width);
            // our swizzle tiles are 16x16, so make texture multiple of that height
            int blockHeight = 16;
            height = (height + blockHeight - 1) / blockHeight * blockHeight;
            return (width, height);
        }

        public static GraphicsFormat ColorFormatToGraphics(ColorFormat format)
        {
            return format switch
            {
                ColorFormat.Float32x4 => GraphicsFormat.R32G32B32A32_SFloat,
                ColorFormat.Float16x4 => GraphicsFormat.R16G16B16A16_SFloat,
                ColorFormat.Norm8x4 => GraphicsFormat.R8G8B8A8_UNorm,
                ColorFormat.BC7 => GraphicsFormat.RGBA_BC7_UNorm,
                _ => throw new ArgumentOutOfRangeException(nameof(format), format, null)
            };
        }

        public static long CalcPosDataSize(int splatCount, VectorFormat formatPos)
        {
            return splatCount * GetVectorSize(formatPos);
        }
        public static long CalcOtherDataSize(int splatCount, VectorFormat formatScale)
        {
            return splatCount * GetOtherSizeNoSHIndex(formatScale);
        }
        public static long CalcColorDataSize(int splatCount, ColorFormat formatColor)
        {
            var (width, height) = CalcTextureSize(splatCount);
            return width * height * GetColorSize(formatColor);
        }
        public static long CalcSHDataSize(int splatCount, SHFormat formatSh)
        {
            int shCount = GetSHCount(formatSh, splatCount);
            return formatSh switch
            {
                SHFormat.Float32 => shCount * UnsafeUtility.SizeOf<SHTableItemFloat32>(),
                SHFormat.Float16 => shCount * UnsafeUtility.SizeOf<SHTableItemFloat16>(),
                SHFormat.Norm11 => shCount * UnsafeUtility.SizeOf<SHTableItemNorm11>(),
                SHFormat.Norm6 => shCount * UnsafeUtility.SizeOf<SHTableItemNorm6>(),
                _ => shCount * UnsafeUtility.SizeOf<SHTableItemFloat16>() + splatCount * 2
            };
        }
        public static long CalcChunkDataSize(int splatCount)
        {
            int chunkCount = (splatCount + kChunkSize - 1) / kChunkSize;
            return chunkCount * UnsafeUtility.SizeOf<ChunkInfo>();
        }

        [SerializeField] VectorFormat m_PosFormat = VectorFormat.Norm11;
        [SerializeField] VectorFormat m_ScaleFormat = VectorFormat.Norm11;
        [SerializeField] SHFormat m_SHFormat = SHFormat.Norm11;
        [SerializeField] ColorFormat m_ColorFormat;

        [SerializeField] TextAsset m_PosData;

        [SerializeField] TextAsset m_AlphaData;
        [SerializeField] TextAsset m_ScaleData;
        [SerializeField] TextAsset m_ColorData;
        [SerializeField] TextAsset m_OtherData;
        [SerializeField] TextAsset m_SHData;
        // Chunk data is optional (if data formats are fully lossless then there's no chunking)
        [SerializeField] TextAsset m_ChunkData;

        [SerializeField] CameraInfo[] m_Cameras;


        private GaussianSplatData m_SplatPosData;
        private GaussianSplatData m_SplatColorData;
        private GaussianSplatData m_SplatOtherData;
        private GaussianSplatData m_SplatSHData;
        private GaussianSplatData m_SplatChunkData;
        private GaussianSplatData m_SplatAlphaData;
        private GaussianSplatData m_SplatScaleData;


        [SerializeField] List<List<List<float>>> m_alphas;
        [SerializeField] string m_PointCloudPath;

        public VectorFormat posFormat => m_PosFormat;
        public VectorFormat scaleFormat => m_ScaleFormat;
        public SHFormat shFormat => m_SHFormat;
        public ColorFormat colorFormat => m_ColorFormat;

        public TextAsset alphaData => m_AlphaData;
        public TextAsset scaleData => m_ScaleData;
        public string pointCloudPath => m_PointCloudPath;

        public IGaussianSplatData posData
        {
            get
            {
                if (m_SplatPosData == null)
                    m_SplatPosData = new GaussianSplatData(m_PosData);
                return m_SplatPosData;
            }
        }
        public IGaussianSplatData colorData
        {
            get
            {
                if (m_SplatColorData == null)
                    m_SplatColorData = new GaussianSplatData(m_ColorData);
                return m_SplatColorData;
            }
        }
        public IGaussianSplatData otherData
        {
            get
            {
                if (m_SplatOtherData == null)
                    m_SplatOtherData = new GaussianSplatData(m_OtherData);
                return m_SplatOtherData;
            }
        }
        public IGaussianSplatData shData
        {
            get
            {
                if (m_SplatSHData == null)
                    m_SplatSHData = new GaussianSplatData(m_SHData);
                return m_SplatSHData;
            }
        }
        public IGaussianSplatData chunkData
        {
            get
            {
                if (m_SplatChunkData == null)
                    m_SplatChunkData = new GaussianSplatData(m_ChunkData);
                return m_SplatChunkData;
            }
        }



        public void Dispose()
        {
            // Dispose of any resources if needed
        }


        public CameraInfo[] cameras => m_Cameras;

        public List<List<List<float>>> alphas => m_alphas;

        public struct ChunkInfo
        {
            public uint colR, colG, colB, colA;
            public float2 posX, posY, posZ;
            public uint sclX, sclY, sclZ;
            public uint shR, shG, shB;
        }

        [Serializable]
        public struct CameraInfo
        {
            public Vector3 pos;
            public Vector3 axisX, axisY, axisZ;
            public float fov;
        }

        public class GaussianSplatData : IGaussianSplatData
        {
            public TextAsset m_Asset;

            public GaussianSplatData(TextAsset asset)
            {
                m_Asset = asset;
            }

            public long dataSize => m_Asset.dataSize;

            public byte[] bytes => m_Asset.bytes;

            public NativeArray<T> GetData<T>() where T : struct
            {
                return m_Asset.GetData<T>();
            }
        }
    }
}
