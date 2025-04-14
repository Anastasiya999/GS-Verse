using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using GaussianSplatting.Runtime;
using GaussianSplatting.Shared;
using Unity.Mathematics;
using Unity.Collections;
using System;
using Unity.Jobs;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Burst;
using System.IO;

public class SplatAccessor : MonoBehaviour
{

    private GaussianSplatRenderer _splatRenderer;
    private MeshFilter _meshFilter;
    private Mesh _mesh;


    void Start()
    {

        var faceVertices = ProcessModifiedMesh(gameObject);
        _splatRenderer = GameObject.FindGameObjectWithTag("SplatRenderer").GetComponent<GaussianSplatRenderer>();
        if (_splatRenderer.asset.alphaData != null)
        {
            // Get the raw bytes from the TextAsset
            byte[] fileBytes = _splatRenderer.asset.alphaData.bytes;
            byte[] fileScaleBytes = _splatRenderer.asset.scaleData.bytes;
            var decodedAlphas = DecodeAlphasv2(fileBytes, faceVertices.Count / 3);
            var decodedScales = DecodeScales(fileScaleBytes, _splatRenderer.asset.splatCount);


            var xyzValues = SplatMathUtils.ToNativeArray(SplatMathUtils.CalculateXYZ(faceVertices, 5, decodedAlphas));
            var (rotations, scalings) = SplatMathUtils.GenerateRotationsAndScalesNative(faceVertices, decodedScales, 5, Allocator.TempJob);

            NativeArray<InputSplatRuntimeData> inputSplatsData = new(_splatRenderer.asset.splatCount, Allocator.TempJob);

            var job = new CreateAssetDataJob()
            {
                m_InputPos = xyzValues,
                m_InputRot = rotations,
                m_InputScale = scalings,
                m_Output = inputSplatsData
            };
            job.Schedule(xyzValues.Length, 8192).Complete();


            CreateAsset(inputSplatsData);


            rotations.Dispose();
            scalings.Dispose();
            inputSplatsData.Dispose();
            xyzValues.Dispose();

        }
        else
        {
            Debug.LogError("posData is missing in GaussianSplatAsset!");
        }
    }



    public List<Vector3> ProcessModifiedMesh(GameObject gameObject)
    {

        MeshFilter meshFilter = gameObject.GetComponent<MeshFilter>();

        Mesh mesh = meshFilter.sharedMesh;
        Vector3[] vertices = mesh.vertices;

        // 2. Modify vertex positions (example: move all vertices up by 0.5 units)
        for (int i = 0; i < vertices.Length; i++)
        {
            vertices[i] += Vector3.up * 0.5f;
        }


        mesh.vertices = vertices;
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();

        return SplatMathUtils.GetMeshFaceVertices(gameObject);
    }

    public static void CompareFirst100Values(NativeArray<uint> posDataArray1, NativeArray<uint> posDataArray2)
    {
        // Ensure both arrays have the same size
        if (posDataArray1.Length != posDataArray2.Length)
        {
            Debug.LogError("Arrays have different sizes! Cannot compare.");
            return;
        }

        // Determine the maximum number of elements to compare (100 or the length of the shortest array)
        int compareLength = 100;

        // Iterate through both arrays and compare the first 100 elements
        for (int i = 0; i < compareLength; i++)
        {
            uint value1 = posDataArray1[i];
            uint value2 = posDataArray2[i];

            if (value1 != value2)
            {
                uint difference = value1 ^ value2; // XOR to see how they differ (bitwise)
                Debug.Log($"Difference at Index {i}:");
                Debug.Log($"Array 1 Value: {value1}, Array 2 Value: {value2}");
                Debug.Log($"Difference (XOR): {difference} (in binary: {System.Convert.ToString(difference, 2).PadLeft(32, '0')})");
            }
            else
            {
                // Log when values are the same
                Debug.Log($"Index {i} is the same in both arrays: Value = {value1}");
            }
        }
    }
    unsafe void CreateAsset(NativeArray<InputSplatRuntimeData> splatPosData)
    {


        var m_FormatPos = GaussianSplatAsset.VectorFormat.Norm11;
        float3 boundsMin, boundsMax;
        var boundsJob = new CalcBoundsJob
        {
            m_BoundsMin = &boundsMin,
            m_BoundsMax = &boundsMax,
            m_SplatPosData = splatPosData
        };
        boundsJob.Schedule().Complete();

        ReorderMorton(splatPosData, boundsMin, boundsMax);
        LinearizeData(splatPosData);
        var chunks = CreateChunkData(splatPosData, _splatRenderer.asset.chunkData.GetData<GaussianSplatAsset.ChunkInfo>());

        int dataLen = splatPosData.Length * GaussianSplatAsset.GetVectorSize(m_FormatPos);
        dataLen = NextMultipleOf(dataLen, 8);
        NativeArray<byte> data = new(dataLen, Allocator.TempJob);

        CreatePositionsDataJob job = new CreatePositionsDataJob
        {
            m_Input = splatPosData,
            m_Format = m_FormatPos,
            m_FormatSize = GaussianSplatAsset.GetVectorSize(m_FormatPos),
            m_Output = data
        };
        job.Schedule(splatPosData.Length, 8192).Complete();

        NativeArray<byte> otherData = CreateOtherData(splatPosData);

        var convertedPos = ReinterpretByteArrayAsUint(data);
        var convertedOther = ReinterpretByteArrayAsUint(otherData);
        Debug.Log($"was created? {_splatRenderer.asset.GetRawChunkDataCreated()}");

        _splatRenderer.asset.rawPosData = convertedPos;
        _splatRenderer.asset.rawOtherData = convertedOther;
        _splatRenderer.asset.rawChunkData = _splatRenderer.asset.chunkData.GetData<GaussianSplatAsset.ChunkInfo>();
        _splatRenderer.asset.SetRawChunkDataCreated(true);
        _splatRenderer.asset.rawChunkData = chunks;

        Debug.Log($"chunk data size {_splatRenderer.asset.rawChunkData.Length}");

        //chunks.Dispose();
        data.Dispose();


    }

    NativeArray<byte> CreateOtherData(NativeArray<InputSplatRuntimeData> inputSplats)
    {
        var m_FormatScale = GaussianSplatAsset.VectorFormat.Norm11;
        int formatSize = GaussianSplatAsset.GetOtherSizeNoSHIndex(m_FormatScale); // 4 + 4
        int dataLen = inputSplats.Length * formatSize;

        dataLen = NextMultipleOf(dataLen, 8); // serialized as ulong
        NativeArray<byte> data = new(dataLen, Allocator.TempJob);


        CreateOtherDataJob job = new CreateOtherDataJob
        {
            m_Input = inputSplats,
            m_ScaleFormat = m_FormatScale,
            m_FormatSize = formatSize,
            m_Output = data
        };
        job.Schedule(inputSplats.Length, 8192).Complete();

        return data;
    }


    struct CreateOtherDataJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<InputSplatRuntimeData> m_Input;
        public GaussianSplatAsset.VectorFormat m_ScaleFormat;
        public int m_FormatSize;
        [NativeDisableParallelForRestriction] public NativeArray<byte> m_Output;
        const int ROTATION_SIZE = 4;
        const int SCALE_SIZE = 4;
        const int SH_INDEX_OFFSET = 8; // 4 + 4

        public unsafe void Execute(int index)
        {
            byte* outputPtr = (byte*)m_Output.GetUnsafePtr() + index * m_FormatSize;

            // rotation: 4 bytes
            {
                Quaternion rotQ = m_Input[index].rot;
                float4 rot = new float4(rotQ.x, rotQ.y, rotQ.z, rotQ.w);
                uint enc = EncodeQuatToNorm10(rot);
                *(uint*)outputPtr = enc;
                outputPtr += 4;
            }

            // scale: 6, 4 or 2 bytes
            EmitEncodedVector(m_Input[index].scale, outputPtr, m_ScaleFormat, index);
            outputPtr += GaussianSplatAsset.GetVectorSize(m_ScaleFormat);

        }
    }

    static void LinearizeData(NativeArray<InputSplatRuntimeData> splatData)
    {
        LinearizeDataJob job = new LinearizeDataJob();
        job.splatData = splatData;
        job.Schedule(splatData.Length, 4096).Complete();
    }
    [BurstCompile]
    struct LinearizeDataJob : IJobParallelFor
    {
        public NativeArray<InputSplatRuntimeData> splatData;
        public void Execute(int index)
        {
            var splat = splatData[index];

            // rot
            var q = splat.rot;
            var qq = GaussianUtils.NormalizeSwizzleRotation(new float4(q.x, q.y, q.z, q.w));
            qq = GaussianUtils.PackSmallest3Rotation(qq);
            splat.rot = new Quaternion(qq.x, qq.y, qq.z, qq.w);

            // scale
            splat.scale = GaussianUtils.LinearScale(splat.scale);

            splatData[index] = splat;
        }
    }



    public static NativeArray<uint> ReinterpretByteArrayAsUint(NativeArray<byte> byteData)
    {
        int dataLength = byteData.Length;
        int count = dataLength / 4;
        NativeArray<uint> result = new NativeArray<uint>(count, Allocator.Persistent);

        byte[] buffer = new byte[4];

        for (int i = 0; i < count; i++)
        {
            int baseIndex = i * 4;
            buffer[0] = byteData[baseIndex];
            buffer[1] = byteData[baseIndex + 1];
            buffer[2] = byteData[baseIndex + 2];
            buffer[3] = byteData[baseIndex + 3];

            result[i] = BitConverter.ToUInt32(buffer, 0);
        }

        return result;
    }

    static void ReorderMorton(NativeArray<InputSplatRuntimeData> splatData, float3 boundsMin, float3 boundsMax)
    {
        ReorderMortonJob order = new ReorderMortonJob
        {
            m_SplatData = splatData,
            m_BoundsMin = boundsMin,
            m_InvBoundsSize = 1.0f / (boundsMax - boundsMin),
            m_Order = new NativeArray<(ulong, int)>(splatData.Length, Allocator.TempJob)
        };
        order.Schedule(splatData.Length, 4096).Complete();
        order.m_Order.Sort(new OrderComparer());

        NativeArray<InputSplatRuntimeData> copy = new(order.m_SplatData, Allocator.TempJob);
        for (int i = 0; i < copy.Length; ++i)
            order.m_SplatData[i] = copy[order.m_Order[i].Item2];
        copy.Dispose();

        order.m_Order.Dispose();
    }

    [BurstCompile]
    struct CalcBoundsJob : IJob
    {
        [NativeDisableUnsafePtrRestriction] public unsafe float3* m_BoundsMin;
        [NativeDisableUnsafePtrRestriction] public unsafe float3* m_BoundsMax;
        [ReadOnly] public NativeArray<InputSplatRuntimeData> m_SplatPosData;

        public unsafe void Execute()
        {
            float3 boundsMin = float.PositiveInfinity;
            float3 boundsMax = float.NegativeInfinity;

            for (int i = 0; i < m_SplatPosData.Length; ++i)
            {
                float3 pos = m_SplatPosData[i].pos;
                boundsMin = math.min(boundsMin, pos);
                boundsMax = math.max(boundsMax, pos);
            }
            *m_BoundsMin = boundsMin;
            *m_BoundsMax = boundsMax;
        }
    }

    [BurstCompile]
    struct ReorderMortonJob : IJobParallelFor
    {
        const float kScaler = (float)((1 << 21) - 1);
        public float3 m_BoundsMin;
        public float3 m_InvBoundsSize;
        [ReadOnly] public NativeArray<InputSplatRuntimeData> m_SplatData;
        public NativeArray<(ulong, int)> m_Order;

        public void Execute(int index)
        {
            float3 pos = ((float3)m_SplatData[index].pos - m_BoundsMin) * m_InvBoundsSize * kScaler;
            uint3 ipos = (uint3)pos;
            ulong code = GaussianUtils.MortonEncode3(ipos);
            m_Order[index] = (code, index);
        }
    }

    struct OrderComparer : IComparer<(ulong, int)>
    {
        public int Compare((ulong, int) a, (ulong, int) b)
        {
            if (a.Item1 < b.Item1) return -1;
            if (a.Item1 > b.Item1) return +1;
            return a.Item2 - b.Item2;
        }
    }


    static uint EncodeQuatToNorm10(float4 v) // 32 bits: 10.10.10.2
    {
        return (uint)(v.x * 1023.5f) | ((uint)(v.y * 1023.5f) << 10) | ((uint)(v.z * 1023.5f) << 20) | ((uint)(v.w * 3.5f) << 30);
    }


    struct CreatePositionsDataJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<InputSplatRuntimeData> m_Input;
        public GaussianSplatAsset.VectorFormat m_Format;
        public int m_FormatSize;
        [NativeDisableParallelForRestriction] public NativeArray<byte> m_Output;

        public unsafe void Execute(int index)
        {
            byte* outputPtr = (byte*)m_Output.GetUnsafePtr() + index * m_FormatSize;

            EmitEncodedVector(m_Input[index].pos, outputPtr, m_Format, index);

        }
    }

    struct CreateAssetDataJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<Quaternion> m_InputRot;
        [ReadOnly] public NativeArray<Vector3> m_InputScale;
        [ReadOnly] public NativeArray<Vector3> m_InputPos;

        [NativeDisableParallelForRestriction] public NativeArray<InputSplatRuntimeData> m_Output;

        public unsafe void Execute(int index)
        {

            m_Output[index] = new InputSplatRuntimeData
            {
                pos = m_InputPos[index],
                scale = m_InputScale[index],
                rot = m_InputRot[index]
            };
        }
    }


    static NativeArray<GaussianSplatAsset.ChunkInfo> CreateChunkData(NativeArray<InputSplatRuntimeData> splatData, NativeArray<GaussianSplatAsset.ChunkInfo> prevChunks)
    {
        int chunkCount = (splatData.Length + GaussianSplatAsset.kChunkSize - 1) / GaussianSplatAsset.kChunkSize;
        CalcChunkDataJob job = new CalcChunkDataJob
        {
            splatPositions = splatData,
            chunks = new(chunkCount, Allocator.TempJob),
            prevChunks = prevChunks
        };

        job.Schedule(chunkCount, 8).Complete();

        return job.chunks;
    }

    public static NativeArray<float3> ConvertToFloat3Array(NativeArray<Vector3> input, Allocator allocator)
    {
        var output = new NativeArray<float3>(input.Length, allocator);
        for (int i = 0; i < input.Length; i++)
        {
            output[i] = input[i];
        }
        return output;
    }

    [BurstCompile]
    struct CalcChunkDataJob : IJobParallelFor
    {
        [NativeDisableParallelForRestriction] public NativeArray<InputSplatRuntimeData> splatPositions;
        [ReadOnly] public NativeArray<GaussianSplatAsset.ChunkInfo> prevChunks;
        public NativeArray<GaussianSplatAsset.ChunkInfo> chunks;

        public void Execute(int chunkIdx)
        {
            GaussianSplatAsset.ChunkInfo prevInfo = prevChunks[chunkIdx];

            float3 chunkMinpos = float.PositiveInfinity;
            float3 chunkMinscl = float.PositiveInfinity;
            float3 chunkMaxpos = float.NegativeInfinity;
            float3 chunkMaxscl = float.NegativeInfinity;

            int splatBegin = math.min(chunkIdx * GaussianSplatAsset.kChunkSize, splatPositions.Length);
            int splatEnd = math.min((chunkIdx + 1) * GaussianSplatAsset.kChunkSize, splatPositions.Length);

            // calculate data bounds inside the chunk
            for (int i = splatBegin; i < splatEnd; ++i)
            {
                InputSplatRuntimeData s = splatPositions[i];

                // transform scale to be more uniformly distributed
                s.scale = math.pow(s.scale, 1.0f / 8.0f);

                splatPositions[i] = s;

                chunkMinpos = math.min(chunkMinpos, s.pos);
                chunkMinscl = math.min(chunkMinscl, s.scale);
                chunkMaxpos = math.max(chunkMaxpos, s.pos);
                chunkMaxscl = math.max(chunkMaxscl, s.scale);

            }

            // make sure bounds are not zero
            chunkMaxpos = math.max(chunkMaxpos, chunkMinpos + 1.0e-5f);
            chunkMaxscl = math.max(chunkMaxscl, chunkMinscl + 1.0e-5f);

            // store chunk info
            GaussianSplatAsset.ChunkInfo info = default;
            info.posX = new float2(chunkMinpos.x, chunkMaxpos.x);
            info.posY = new float2(chunkMinpos.y, chunkMaxpos.y);
            info.posZ = new float2(chunkMinpos.z, chunkMaxpos.z);
            info.sclX = math.f32tof16(chunkMinscl.x) | (math.f32tof16(chunkMaxscl.x) << 16);
            info.sclY = math.f32tof16(chunkMinscl.y) | (math.f32tof16(chunkMaxscl.y) << 16);
            info.sclZ = math.f32tof16(chunkMinscl.z) | (math.f32tof16(chunkMaxscl.z) << 16);
            // Reuse color data from previous chunk
            info.colR = prevInfo.colR;
            info.colG = prevInfo.colG;
            info.colB = prevInfo.colB;
            info.colA = prevInfo.colA;

            // Reuse SH coefficients from previous chunk
            info.shR = prevInfo.shR;
            info.shG = prevInfo.shG;
            info.shB = prevInfo.shB;


            chunks[chunkIdx] = info;

            // adjust data to be 0..1 within chunk bounds
            for (int i = splatBegin; i < splatEnd; ++i)
            {
                InputSplatRuntimeData s = splatPositions[i];
                s.pos = ((float3)s.pos - chunkMinpos) / (chunkMaxpos - chunkMinpos);
                s.scale = ((float3)s.scale - chunkMinscl) / (chunkMaxscl - chunkMinscl);

                splatPositions[i] = s;
            }

        }
    }

    public struct InputSplatRuntimeData
    {
        public Vector3 pos;
        public Vector3 scale;
        public Quaternion rot;
    }
    static unsafe void EmitEncodedVector(float3 v, byte* outputPtr, GaussianSplatAsset.VectorFormat format, int index)
    {
        switch (format)
        {
            case GaussianSplatAsset.VectorFormat.Float32:
                {
                    *(float*)outputPtr = v.x;
                    *(float*)(outputPtr + 4) = v.y;
                    *(float*)(outputPtr + 8) = v.z;
                }
                break;
            case GaussianSplatAsset.VectorFormat.Norm11:
                {
                    uint enc = EncodeFloat3ToNorm11(math.saturate(v));
                    *(uint*)outputPtr = enc;
                }
                break;

        }
    }

    static uint EncodeFloat3ToNorm11(float3 v) // 32 bits: 11.10.11
    {
        return (uint)(v.x * 2047.5f) | ((uint)(v.y * 1023.5f) << 11) | ((uint)(v.z * 2047.5f) << 21);
    }
    public NativeArray<uint> ConvertPositions(NativeArray<float3> positions)
    {

        NativeArray<uint> encodedPositions = new NativeArray<uint>(positions.Length, Allocator.TempJob);

        for (int i = 0; i < positions.Length; i++)
        {
            encodedPositions[i] = EncodeFloat3ToNorm11(math.saturate(positions[i]));
        }

        for (int i = 0; i < 10 && i < encodedPositions.Length; i++)
        {
            Debug.Log($"Encoded Position {i}: {encodedPositions[i]}");
            Debug.Log($"Original Position {i}: {positions[i]}, Saturated: {math.saturate(positions[i])}");
        }

        return encodedPositions;
    }
    List<List<List<float>>> DecodeAlphas(byte[] fileBytes, int numFaces)
    {
        int vectorSize = GaussianSplatAsset.GetVectorSize(_splatRenderer.asset.posFormat);
        int numPoints = numFaces * 5;  // Assuming 5 points per triangle

        // Ensure we have enough data in fileBytes to decode the alphas
        if (fileBytes.Length < numPoints * vectorSize)
        {
            Debug.LogError($"Insufficient data: expected {numPoints * vectorSize} bytes, but got {fileBytes.Length} bytes.");

        }

        byte[] buffer = new byte[vectorSize];
        List<List<List<float>>> decodedAlphas = new List<List<List<float>>>();

        for (int i = 0; i < numFaces; i++)
        {
            List<List<float>> faceAlphas = new List<List<float>>();
            for (int j = 0; j < 5; j++) // 5 points per triangle
            {
                int offset = (i * 5 + j) * vectorSize;
                if (offset + vectorSize > fileBytes.Length)
                {
                    Debug.LogError($"Offset and length are out of bounds for the array. Offset: {offset}, VectorSize: {vectorSize}, FileLength: {fileBytes.Length}");
                    break;
                }

                // Copy the relevant bytes into the buffer
                System.Buffer.BlockCopy(fileBytes, offset, buffer, 0, vectorSize);

                // Decode the packed uint to float3
                uint packed = System.BitConverter.ToUInt32(buffer, 0);
                float3 pos = DecodeNorm11ToFloat3(packed);

                // Add the decoded float3 to the List
                faceAlphas.Add(new List<float> { pos.x, pos.y, pos.z });
            }
            decodedAlphas.Add(faceAlphas);
        }

        return decodedAlphas;
    }

    List<List<List<float>>> DecodeAlphasv2(byte[] fileBytes, int numFaces)
    {
        int vectorSize = 12;
        int numPoints = numFaces * 5;  // Assuming 5 points per triangle

        // Ensure we have enough data in fileBytes to decode the alphas
        if (fileBytes.Length < numPoints * vectorSize)
        {
            Debug.LogError($"Insufficient data: expected {numPoints * vectorSize} bytes, but got {fileBytes.Length} bytes.");

        }

        byte[] buffer = new byte[vectorSize];
        List<List<List<float>>> decodedAlphas = new List<List<List<float>>>();

        for (int i = 0; i < numFaces; i++)
        {
            List<List<float>> faceAlphas = new List<List<float>>();
            for (int j = 0; j < 5; j++) // 5 points per triangle
            {
                int offset = (i * 5 + j) * vectorSize;
                if (offset + vectorSize > fileBytes.Length)
                {
                    Debug.LogError($"Offset and length are out of bounds for the array. Offset: {offset}, VectorSize: {vectorSize}, FileLength: {fileBytes.Length}");
                    break;
                }

                float x = System.BitConverter.ToSingle(fileBytes, offset);
                float y = System.BitConverter.ToSingle(fileBytes, offset + 4);
                float z = System.BitConverter.ToSingle(fileBytes, offset + 8);

                faceAlphas.Add(new List<float> { x, y, z });

                // Add the decoded float3 to the List

            }
            decodedAlphas.Add(faceAlphas);
        }

        return decodedAlphas;
    }

    List<float> DecodeScales(byte[] fileBytes, int numberOfSplats)
    {
        int vectorSize = GaussianSplatAsset.GetVectorSize(_splatRenderer.asset.posFormat);

        // Ensure we have enough data in fileBytes to decode the alphas
        if (fileBytes.Length < numberOfSplats * vectorSize)
        {
            Debug.LogError($"Insufficient data: expected {numberOfSplats * vectorSize} bytes, but got {fileBytes.Length} bytes.");

        }

        byte[] buffer = new byte[vectorSize];
        List<float> decodedScales = new List<float>();

        for (int i = 0; i < numberOfSplats; i++)
        {
            Buffer.BlockCopy(fileBytes, i * vectorSize, buffer, 0, vectorSize);

            // Convert to float
            float scale = System.BitConverter.ToSingle(buffer, 0);
            decodedScales.Add(scale);
        }

        return decodedScales;
    }

    void AccessFirstFaceVertices()
    {
        if (_mesh == null || _mesh.vertexCount == 0)
        {
            Debug.LogWarning("No mesh data available");
            return;
        }

        // Get all vertices and triangles
        Vector3[] vertices = _mesh.vertices;
        int[] triangles = _mesh.triangles;

        if (triangles.Length >= 6) // At least one complete triangle
        {
            // Get the first triangle's vertices
            Vector3 v1 = vertices[triangles[0]];
            Vector3 v2 = vertices[triangles[1]];
            Vector3 v3 = vertices[triangles[2]];

            Debug.Log($"First face vertices:\n{v1}\n{v2}\n{v3}");

            Vector3 v4 = vertices[triangles[3]];
            Vector3 v5 = vertices[triangles[4]];
            Vector3 v6 = vertices[triangles[5]];

            Debug.Log($"Second face vertices:\n{v4}\n{v5}\n{v6}");
        }
        else
        {
            Debug.LogWarning("Mesh doesn't contain complete triangles");
        }
    }

    static int NextMultipleOf(int size, int multipleOf)
    {
        return (size + multipleOf - 1) / multipleOf * multipleOf;
    }

    static float3 DecodeVectorSafe(byte[] data, GaussianSplatAsset.VectorFormat format)
    {
        switch (format)
        {
            case GaussianSplatAsset.VectorFormat.Float32:
                return new float3(
                    System.BitConverter.ToSingle(data, 0),
                    System.BitConverter.ToSingle(data, 4),
                    System.BitConverter.ToSingle(data, 8));

            case GaussianSplatAsset.VectorFormat.Norm11:
                uint packed = System.BitConverter.ToUInt32(data, 0);
                return DecodeNorm11ToFloat3(packed);

            // Add other format cases as needed
            default:
                throw new NotImplementedException();
        }
    }

    static float3 DecodeNorm11ToFloat3(uint encoded)
    {
        // Bit layout: 11 bits X, 10 bits Y, 11 bits Z (total 32 bits)

        // Masks:
        const uint maskX = 0x7FF;    // 11 bits (0x7FF = 2047)
        const uint maskY = 0x3FF;    // 10 bits (0x3FF = 1023)
        const uint maskZ = 0x7FF;    // 11 bits (0x7FF = 2047)

        // Extract and normalize components
        float x = (float)(encoded & maskX) / maskX;           // X: bits 0-10
        float y = (float)((encoded >> 11) & maskY) / maskY;    // Y: bits 11-20
        float z = (float)((encoded >> 21) & maskZ) / maskZ;    // Z: bits 21-31

        return new float3(x, y, z);
    }


    // Update is called once per frame
    void Update()
    {
        var faceVertices = ProcessModifiedMesh(gameObject);
        _splatRenderer = GameObject.FindGameObjectWithTag("SplatRenderer").GetComponent<GaussianSplatRenderer>();
        if (_splatRenderer.asset.alphaData != null)
        {
            // Get the raw bytes from the TextAsset
            byte[] fileBytes = _splatRenderer.asset.alphaData.bytes;
            byte[] fileScaleBytes = _splatRenderer.asset.scaleData.bytes;
            var decodedAlphas = DecodeAlphasv2(fileBytes, faceVertices.Count / 3);
            var decodedScales = DecodeScales(fileScaleBytes, _splatRenderer.asset.splatCount);


            var xyzValues = SplatMathUtils.ToNativeArray(SplatMathUtils.CalculateXYZ(faceVertices, 5, decodedAlphas));
            var (rotations, scalings) = SplatMathUtils.GenerateRotationsAndScalesNative(faceVertices, decodedScales, 5, Allocator.TempJob);

            NativeArray<InputSplatRuntimeData> inputSplatsData = new(_splatRenderer.asset.splatCount, Allocator.TempJob);

            var job = new CreateAssetDataJob()
            {
                m_InputPos = xyzValues,
                m_InputRot = rotations,
                m_InputScale = scalings,
                m_Output = inputSplatsData
            };
            job.Schedule(xyzValues.Length, 8192).Complete();


            CreateAsset(inputSplatsData);


            rotations.Dispose();
            scalings.Dispose();
            inputSplatsData.Dispose();
            xyzValues.Dispose();

        }
        else
        {
            Debug.LogError("posData is missing in GaussianSplatAsset!");
        }
    }
}
