// SPDX-License-Identifier: MIT

using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Experimental.Rendering;


namespace GaussianSplatting.Runtime
{
    public class GaussianSplatRuntimeAsset : IGaussianSplatAsset
    {
        public const int kCurrentVersion = 2023_10_20;
        public const int kChunkSize = 256;
        public const int kTextureWidth = 2048; // allows up to 32M splats on desktop GPU (2k width x 16k height)
        public const int kMaxSplats = 8_600_000; // mostly due to 2GB GPU buffer size limit when exporting a splat (2GB / 248B is just over 8.6M)

        string m_Name;
        int m_FormatVersion;
        int m_SplatCount;
        Vector3 m_BoundsMin;
        Vector3 m_BoundsMax;
        Hash128 m_DataHash;

        public string name => m_Name;
        public int formatVersion => m_FormatVersion;
        public int splatCount => m_SplatCount;
        public Vector3 boundsMin => m_BoundsMin;
        public Vector3 boundsMax => m_BoundsMax;
        public Hash128 dataHash => m_DataHash;

        public void Initialize(string name, int splats, GaussianSplatAsset.VectorFormat formatPos, GaussianSplatAsset.VectorFormat formatScale, GaussianSplatAsset.ColorFormat formatColor, GaussianSplatAsset.SHFormat formatSh, Vector3 bMin, Vector3 bMax, GaussianSplatAsset.CameraInfo[] cameraInfos, string pointCloudPath)
        {
            m_Name = name;
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

        public void SetDataHash(Hash128 hash)
        {
            m_DataHash = hash;
        }

        public void setAssetData(NativeArray<byte> dataChunk, NativeArray<byte> dataPos, NativeArray<byte> dataOther, NativeArray<byte> dataColor, NativeArray<byte> dataSh, TextAsset dataAlpha, TextAsset dataScale)
        {
            m_SplatChunkData = new GaussianSplatData(dataChunk);
            m_SplatPosData = new GaussianSplatData(dataPos);
            m_SplatOtherData = new GaussianSplatData(dataOther);
            m_SplatColorData = new GaussianSplatData(dataColor);
            m_SplatSHData = new GaussianSplatData(dataSh);

            m_SplatAlphaData = dataAlpha;
            m_SplatScaleData = dataScale;
        }

        public void Dispose()
        {
            m_SplatPosData?.Dispose();
            m_SplatColorData?.Dispose();
            m_SplatOtherData?.Dispose();
            m_SplatSHData?.Dispose();
            m_SplatChunkData?.Dispose();
        }

        GaussianSplatAsset.VectorFormat m_PosFormat = GaussianSplatAsset.VectorFormat.Norm11;
        GaussianSplatAsset.VectorFormat m_ScaleFormat = GaussianSplatAsset.VectorFormat.Norm11;
        GaussianSplatAsset.SHFormat m_SHFormat = GaussianSplatAsset.SHFormat.Norm11;
        GaussianSplatAsset.ColorFormat m_ColorFormat;

        GaussianSplatData m_SplatPosData;
        GaussianSplatData m_SplatColorData;
        GaussianSplatData m_SplatOtherData;
        GaussianSplatData m_SplatSHData;
        // Chunk data is optional (if data formats are fully lossless then there's no chunking)
        GaussianSplatData m_SplatChunkData;

        TextAsset m_SplatAlphaData;
        TextAsset m_SplatScaleData;

        string m_PointCloudPath;

        GaussianSplatAsset.CameraInfo[] m_Cameras;

        public GaussianSplatAsset.VectorFormat posFormat => m_PosFormat;
        public GaussianSplatAsset.VectorFormat scaleFormat => m_ScaleFormat;
        public GaussianSplatAsset.SHFormat shFormat => m_SHFormat;
        public GaussianSplatAsset.ColorFormat colorFormat => m_ColorFormat;

        public IGaussianSplatData posData => m_SplatPosData;
        public IGaussianSplatData colorData => m_SplatColorData;
        public IGaussianSplatData otherData => m_SplatOtherData;
        public IGaussianSplatData shData => m_SplatSHData;
        public IGaussianSplatData chunkData => m_SplatChunkData;

        public TextAsset alphaData => m_SplatAlphaData;
        public TextAsset scaleData => m_SplatScaleData;
        public string pointCloudPath => m_PointCloudPath;
        public GaussianSplatAsset.CameraInfo[] cameras => m_Cameras;

        public class GaussianSplatData : IGaussianSplatData
        {
            public NativeArray<byte> data;

            public GaussianSplatData(NativeArray<byte> data)
            {
                this.data = data;
            }

            public long dataSize => data.Length;

            public byte[] bytes => data.ToArray();


            public NativeArray<T> GetData<T>() where T : struct
            {
                return data.Reinterpret<T>(UnsafeUtility.SizeOf<byte>());
            }

            public void Dispose()
            {
                if (data.IsCreated)
                    data.Dispose();
            }
        }
    }
}