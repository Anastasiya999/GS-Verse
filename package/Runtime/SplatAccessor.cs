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
    private bool needsAssetUpdate = false;

    Mesh deformingMesh;
    public float springForce = 20f;
    float uniformScale = 1f;
    public float damping = 5f;

    public bool isClicked = false;
    private bool isCreatingAsset = false;

    float frameCount = 0f;

    NativeArray<float3> originalVertices;
    NativeArray<float3> transformedOriginalVertices;
    NativeArray<float3> displacedVertices;
    NativeArray<float3> transformedDisplacedVertices;
    NativeArray<float3> vertexVelocities;
    NativeArray<float3> vertexVelocitiesGS;
    NativeArray<float3> verticesGS;
    NativeArray<int> triangles;
    List<List<List<float>>> decodedAlphas;

    NativeArray<float3> decodedAlphasNative;
    List<float> decodedScales;
    NativeArray<float> decodedScalesNative;

    private JobHandle createAssetJobHandle;
    private NativeArray<InputSplatRuntimeData> inputSplatsData;
    private NativeArray<float3> xyzValues;
    private NativeArray<quaternion> rotations;
    private NativeArray<float3> scalings;
    private NativeArray<float3> faceVertices;
    private bool isJobScheduled = false;

    void Start()
    {
        _splatRenderer = GameObject.FindGameObjectWithTag("SplatRenderer").GetComponent<GaussianSplatRenderer>();
        if (_splatRenderer.asset.alphaData != null)
        {

            byte[] fileBytes = _splatRenderer.asset.alphaData.bytes;
            byte[] fileScaleBytes = _splatRenderer.asset.scaleData.bytes;
            decodedAlphasNative = DecodeAlphasToNativeFloat3(fileBytes, _splatRenderer.asset.splatCount / 5, 5, Allocator.Persistent);
            decodedScalesNative = DecodeScalesToNative(fileScaleBytes, _splatRenderer.asset.splatCount, Allocator.Persistent);

        }
        else
        {
            Debug.LogError("posData is missing in GaussianSplatAsset!");
        }
        GetComponent<MeshRenderer>().enabled = false; // Hide the mesh
        uniformScale = transform.localScale.x;
        deformingMesh = GetComponent<MeshFilter>().mesh;

        var verts = deformingMesh.vertices;
        var tris = deformingMesh.triangles;

        int vertexCount = verts.Length;
        int triangleCount = tris.Length;

        originalVertices = new NativeArray<float3>(vertexCount, Allocator.Persistent);
        displacedVertices = new NativeArray<float3>(vertexCount, Allocator.Persistent);
        vertexVelocities = new NativeArray<float3>(vertexCount, Allocator.Persistent);

        //TODO: refactor - don't use additional vertices
        transformedOriginalVertices = new NativeArray<float3>(vertexCount, Allocator.Persistent);
        transformedDisplacedVertices = new NativeArray<float3>(vertexCount, Allocator.Persistent);
        vertexVelocitiesGS = new NativeArray<float3>(vertexCount, Allocator.Persistent);

        verticesGS = new NativeArray<float3>(vertexCount, Allocator.Persistent);
        triangles = new NativeArray<int>(triangleCount, Allocator.Persistent);

        for (int i = 0; i < vertexCount; i++)
        {
            float3 v = verts[i];
            originalVertices[i] = v;
            transformedOriginalVertices[i] = TransformVertex(v);
            displacedVertices[i] = v;
            transformedDisplacedVertices[i] = TransformVertex(v);
            verticesGS[i] = v;
        }

        for (int i = 0; i < triangleCount; i++)
        {
            triangles[i] = tris[i];
        }
        inputSplatsData = new NativeArray<InputSplatRuntimeData>(_splatRenderer.asset.splatCount, Allocator.Persistent);



    }

    public static Vector3 TransformVertex(Vector3 v)
    {
        return new Vector3(
                    v.x,
                    -v.z,
                    v.y
                );
    }

    void Update()
    {
        if (isJobScheduled)
        {
            if (createAssetJobHandle.IsCompleted)
            {
                createAssetJobHandle.Complete();
                CreateAsset(inputSplatsData);

                // Dispose arrays
                xyzValues.Dispose();
                rotations.Dispose();
                scalings.Dispose();
                faceVertices.Dispose();

                isJobScheduled = false;
            }
            return;
        }

        frameCount++;
        var dis = isClicked ? 1f : 0f;
        bool mouseDown = Input.GetMouseButton(0);
        if (mouseDown)
        {
            for (int i = 0; i < displacedVertices.Length; i++)
            {
                UpdateVertex(i, dis);
            }
        }
        else
        {
            ReturnToOriginalShape();
        }

        deformingMesh.vertices = ConvertToVector3Array(displacedVertices);
        deformingMesh.RecalculateNormals();
        // Prepare NativeArrays
        faceVertices = SplatMathUtils.GetMeshFaceVerticesNative(gameObject, transformedDisplacedVertices, triangles, Allocator.Persistent);
        xyzValues = CreateXYZData(decodedAlphasNative, faceVertices, _splatRenderer.asset.splatCount / 5, 5);
        (rotations, scalings) = CreateScaleRotationData(faceVertices, decodedScalesNative, 5);

        var job = new CreateAssetDataJob()
        {
            m_InputPos = xyzValues,
            m_InputRot = rotations,
            m_InputScale = scalings,
            m_Output = inputSplatsData
        };

        createAssetJobHandle = job.Schedule(xyzValues.Length, 8192);
        isJobScheduled = true;
    }
    void OnDestroy()
    {
        if (isJobScheduled)
        {
            createAssetJobHandle.Complete();
            isJobScheduled = false;
        }

        Debug.Log("destroyed");
        _splatRenderer.asset.SetRawChunkDataCreated(false);

        if (inputSplatsData.IsCreated)
            inputSplatsData.Dispose();
    }

    public static Vector3[] ConvertToVector3Array(NativeArray<float3> nativeArray)
    {
        Vector3[] result = new Vector3[nativeArray.Length];
        for (int i = 0; i < nativeArray.Length; i++)
        {
            float3 f = nativeArray[i];
            result[i] = new Vector3(f.x, f.y, f.z);
        }
        return result;
    }


    public void SetClickState(bool clicked)
    {
        isClicked = clicked;
    }

    void ReturnToOriginalShape()
    {
        for (int i = 0; i < transformedDisplacedVertices.Length; i++)
        {
            transformedDisplacedVertices[i] = Vector3.Lerp(transformedDisplacedVertices[i], transformedOriginalVertices[i], Time.deltaTime * 5f);
        }
    }

    void UpdateVertex(int i, float dis)
    {


        Vector3 velocity = vertexVelocities[i];
        Vector3 displacement = displacedVertices[i] - originalVertices[i];
        displacement *= uniformScale;
        velocity -= displacement * springForce * Time.deltaTime;
        velocity *= 1f - damping * Time.deltaTime;
        vertexVelocities[i] = velocity;

        Vector3 velocityGS = vertexVelocitiesGS[i];
        Vector3 displacementGS = transformedDisplacedVertices[i] - transformedOriginalVertices[i];
        displacementGS *= uniformScale;
        velocityGS -= displacementGS * springForce * Time.deltaTime;
        velocityGS *= 1f - damping * Time.deltaTime;
        vertexVelocitiesGS[i] = velocityGS;

        displacedVertices[i] += (float3)velocity * (Time.deltaTime / uniformScale);
        //TODO: make sure it also works
        // transformedDisplacedVertices[i] += (float3)velocity * (Time.deltaTime / uniformScale);
        transformedDisplacedVertices[i] += (float3)velocityGS * (Time.deltaTime / uniformScale) + (float3)new Vector3(-transformedOriginalVertices[i].x * (Time.deltaTime / uniformScale) * dis, transformedOriginalVertices[i].y * (Time.deltaTime / uniformScale) * 1.5f * dis, transformedOriginalVertices[i].z * (Time.deltaTime / uniformScale) * dis);

    }
    [BurstCompile]
    struct AddDeformingForceJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float3> displacedVertices;
        [ReadOnly] public NativeArray<float3> transformedDisplacedVertices;

        public NativeArray<float3> vertexVelocities;
        public NativeArray<float3> vertexVelocitiesGS;

        [ReadOnly] public Vector3 pointLocal;
        [ReadOnly] public Vector3 force;
        [ReadOnly] public float uniformScale;
        [ReadOnly] public float deltaTime;

        public void Execute(int i)
        {
            Vector3 pointToVertex = displacedVertices[i] - (float3)pointLocal;
            Vector3 pointToVertexGS = transformedDisplacedVertices[i] - (float3)TransformVertex(pointLocal);

            pointToVertex *= uniformScale;
            pointToVertexGS *= uniformScale;

            float attenuation = 1f / (1f + pointToVertex.sqrMagnitude);
            float attenuationGS = 1f / (1f + pointToVertexGS.sqrMagnitude);

            Vector3 appliedForce = force * attenuation * deltaTime;
            Vector3 appliedForceGS = force * deltaTime;

            vertexVelocities[i] += (float3)appliedForce;
            vertexVelocitiesGS[i] += (float3)appliedForceGS;
        }
    }


    public void AddDeformingForce(Vector3 point, Vector3 force)
    {


        Vector3 pointLocal = transform.InverseTransformPoint(point);

        var job = new AddDeformingForceJob
        {
            displacedVertices = displacedVertices,
            transformedDisplacedVertices = transformedDisplacedVertices,
            vertexVelocities = vertexVelocities,
            vertexVelocitiesGS = vertexVelocitiesGS,
            pointLocal = pointLocal,
            force = force,
            uniformScale = uniformScale,
            deltaTime = Time.deltaTime
        };

        JobHandle handle = job.Schedule(displacedVertices.Length, 64);
        handle.Complete();
    }

    public void AddDeformingForce(Vector3 point, float force)
    {
        for (int i = 0; i < displacedVertices.Length; i++)
        {
            AddForceToVertex(i, point, force);
        }
        Debug.DrawLine(Camera.main.transform.position, point);
    }

    void AddForceToVertex(int i, Vector3 point, float force)
    {
        point = transform.InverseTransformPoint(point);
        Vector3 pointToVertex = displacedVertices[i] - (float3)point;
        Vector3 pointToVertexGS = transformedDisplacedVertices[i] - (float3)TransformVertex(point);

        pointToVertex *= uniformScale;
        float attenuatedForce = force / (1f + pointToVertex.sqrMagnitude);
        float velocity = attenuatedForce * Time.deltaTime;
        vertexVelocities[i] += (float3)pointToVertex.normalized * velocity;

        pointToVertexGS *= uniformScale;
        float attenuatedForceGS = force / (1f + pointToVertexGS.sqrMagnitude);
        float velocityGS = attenuatedForceGS * Time.deltaTime;
        vertexVelocitiesGS[i] += (float3)pointToVertexGS.normalized * velocityGS;
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


        ReorderMorton(splatPosData, boundsMin, boundsMax, _splatRenderer);


        LinearizeData(splatPosData);


        var chunks = CreateChunkData(splatPosData, _splatRenderer.asset.chunkData.GetData<GaussianSplatAsset.ChunkInfo>());
        NativeArray<uint> data = CreatePosDataUint(splatPosData, m_FormatPos);
        NativeArray<uint> otherData = CreateOtherDataUint(splatPosData);

        _splatRenderer.asset.SetIsSorting(true);
        _splatRenderer.asset.rawPosData = data;
        _splatRenderer.asset.rawOtherData = otherData;


        _splatRenderer.asset.rawChunkData = _splatRenderer.asset.chunkData.GetData<GaussianSplatAsset.ChunkInfo>();
        if (frameCount == 1)
        {
            _splatRenderer.asset.SetRawChunkDataCreated(true);
        }



        _splatRenderer.asset.rawChunkData = chunks;
        _splatRenderer.asset.SetDataHash(new Hash128((uint)_splatRenderer.asset.splatCount, (uint)(_splatRenderer.asset.formatVersion + frameCount), 0, 0));

    }

    Hash128 ComputeHashFromNativeArray(NativeArray<uint> data)
    {
        byte[] byteArray = new byte[data.Length * sizeof(uint)];
        System.Buffer.BlockCopy(data.ToArray(), 0, byteArray, 0, byteArray.Length);
        return Hash128.Compute(Convert.ToBase64String(byteArray));
    }




    NativeArray<uint> CreateOtherDataUint(NativeArray<InputSplatRuntimeData> inputSplats)
    {
        const int uintsPerSplat = 2; // 1 uint for rotation (Norm10) + 1 for scale (Norm11)

        NativeArray<uint> data = new(inputSplats.Length * uintsPerSplat, Allocator.Persistent);

        CreateOtherDataUintJob job = new CreateOtherDataUintJob
        {
            m_Input = inputSplats,
            m_Output = data
        };

        job.Schedule(inputSplats.Length, 64).Complete();
        return data;
    }

    NativeArray<uint> CreatePosDataUint(NativeArray<InputSplatRuntimeData> inputSplats, GaussianSplatAsset.VectorFormat m_FormatPos)
    {
        int dataLen = inputSplats.Length * GaussianSplatAsset.GetVectorSize(m_FormatPos);
        dataLen = NextMultipleOf(dataLen, 8);
        NativeArray<uint> data = new(dataLen / 4, Allocator.Persistent);

        CreatePositionsUintDataJob job = new CreatePositionsUintDataJob
        {
            m_Input = inputSplats,
            m_Format = m_FormatPos,
            m_FormatSize = GaussianSplatAsset.GetVectorSize(m_FormatPos),
            m_Output = data
        };

        job.Schedule(inputSplats.Length, 8192).Complete();

        return data;
    }


    NativeArray<float3> CreateXYZData(NativeArray<float3> alphas, NativeArray<float3> vertices, int numTriangles, int numPtsEachTriangle)
    {

        NativeArray<float3> data = new(numTriangles * numPtsEachTriangle, Allocator.Persistent);


        CreateXYZDataJob job = new CreateXYZDataJob
        {
            m_Alphas = alphas,
            m_Vertices = vertices,
            m_Output = data,
            m_numberPtsPerTriangle = numPtsEachTriangle
        };
        job.Schedule(numTriangles * numPtsEachTriangle, 8192).Complete();

        return data;
    }


    public static (NativeArray<quaternion> rotations, NativeArray<float3> scalings) CreateScaleRotationData(NativeArray<float3> vertices, NativeArray<float> scales, int numPtsEachTriangle)
    {

        int numTriangles = vertices.Length / 3;
        int totalPoints = numTriangles * numPtsEachTriangle;

        NativeArray<quaternion> rotations = new NativeArray<quaternion>(totalPoints, Allocator.Persistent);
        NativeArray<float3> scalings = new NativeArray<float3>(totalPoints, Allocator.Persistent);


        CreateRotationsAndScalesJob job = new CreateRotationsAndScalesJob
        {
            m_Vertices = vertices,
            m_Scales = scales,
            m_numPtsEachTriangle = numPtsEachTriangle,
            Rotations = rotations,
            Scalings = scalings
        };
        job.Schedule(totalPoints, 9192).Complete();

        return (rotations, scalings);
    }




    [BurstCompile]
    struct CreateXYZDataJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float3> m_Alphas;
        [ReadOnly] public NativeArray<float3> m_Vertices;
        public int m_numberPtsPerTriangle;
        [NativeDisableParallelForRestriction] public NativeArray<float3> m_Output;


        public unsafe void Execute(int index)
        {
            int triangleIndex = index / m_numberPtsPerTriangle;
            int pointIndex = index % m_numberPtsPerTriangle;

            int v0Idx = triangleIndex * 3;
            float3 v0 = m_Vertices[v0Idx];
            float3 v1 = m_Vertices[v0Idx + 1];
            float3 v2 = m_Vertices[v0Idx + 2];

            int alphaIndex = triangleIndex * m_numberPtsPerTriangle + pointIndex;
            float3 alpha = m_Alphas[alphaIndex];
            float3 point = alpha.x * v0 + alpha.y * v1 + alpha.z * v2;
            m_Output[index] = point;
        }
    }

    [BurstCompile]
    public struct CreateRotationsAndScalesJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float3> m_Vertices;
        [ReadOnly] public NativeArray<float> m_Scales;
        public int m_numPtsEachTriangle;

        [WriteOnly] public NativeArray<quaternion> Rotations;
        [WriteOnly] public NativeArray<float3> Scalings;

        public void Execute(int index)
        {

            int triangleIndex = index / m_numPtsEachTriangle;
            int pointIndex = index % m_numPtsEachTriangle;

            int v0Idx = triangleIndex * 3;
            float3 v0 = m_Vertices[v0Idx];
            float3 v1 = m_Vertices[v0Idx + 1];
            float3 v2 = m_Vertices[v0Idx + 2];

            float3 normal = math.normalize(math.cross(v1 - v0, v2 - v0));
            float3 centroid = (v0 + v1 + v2) / 3.0f;
            float3 basis1 = math.normalize(v1 - centroid);

            float3 v2Init = v2 - centroid;
            float3 basis2 = math.normalize(
                v2Init - math.dot(v2Init, normal) * normal - math.dot(v2Init, basis1) * basis1
            );

            float s1 = math.length(v1 - centroid) / 2.0f;
            float s2 = math.dot(v2Init, basis2) / 2.0f;
            float s0 = 1e-8f;

            float scaleFactor = m_Scales[index];
            float x = math.log(math.max(0, scaleFactor * s0) + s0);
            float y = math.log(math.max(0, scaleFactor * s1) + s0);
            float z = math.log(math.max(0, scaleFactor * s2) + s0);

            float3 scaleVec = new float3(x, y, z);

            // Emulate Unity's Matrix4x4.SetColumn + .rotation behavior
            float4 col0 = new float4(normal, 0f);  // x-axis
            float4 col1 = new float4(basis1, 0f);  // y-axis
            float4 col2 = new float4(basis2, 0f);  // z-axis

            float3x3 rotMatrix = new float3x3(
                new float3(col0.x, col0.y, col0.z),
                new float3(col1.x, col1.y, col1.z),
                new float3(col2.x, col2.y, col2.z)
            );

            quaternion q = quaternion.LookRotationSafe(rotMatrix.c2, rotMatrix.c1); // basis2 (z), basis1 (y)
            quaternion reordered = new quaternion(q.value.w, q.value.x, q.value.y, q.value.z); // match Unity style

            Rotations[index] = reordered;
            Scalings[index] = scaleVec;
        }
    }

    public void MarkAssetDirty()
    {
        needsAssetUpdate = true;
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


    static void ReorderMorton(NativeArray<InputSplatRuntimeData> splatData, float3 boundsMin, float3 boundsMax, GaussianSplatRenderer gs)
    {

        ReorderMortonJob order = new ReorderMortonJob
        {
            m_SplatData = splatData,
            m_BoundsMin = boundsMin,
            m_InvBoundsSize = 1.0f / (boundsMax - boundsMin),
            m_Order = new NativeArray<(ulong, int)>(splatData.Length, Allocator.Persistent)
        };
        order.Schedule(splatData.Length, 4096).Complete();
        order.m_Order.Sort(new OrderComparer());




        NativeArray<InputSplatRuntimeData> copy = new(order.m_SplatData, Allocator.Persistent);
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

    [BurstCompile]
    struct CreateAssetDataJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<quaternion> m_InputRot;
        [ReadOnly] public NativeArray<float3> m_InputScale;
        [ReadOnly] public NativeArray<float3> m_InputPos;

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
            chunks = new(chunkCount, Allocator.Persistent),
            prevChunks = prevChunks
        };

        job.Schedule(chunkCount, 8).Complete();

        return job.chunks;
    }

    [BurstCompile]
    struct CreatePositionsUintDataJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<InputSplatRuntimeData> m_Input;
        public GaussianSplatAsset.VectorFormat m_Format;
        public int m_FormatSize; // should be 4
        [NativeDisableParallelForRestriction] public NativeArray<uint> m_Output;

        public void Execute(int index)
        {
            float3 pos = m_Input[index].pos;

            // Encode using your custom 11-10-11 format
            uint encoded = EncodeFloat3ToNorm11(math.saturate(pos));

            m_Output[index] = encoded;
        }


    }

    [BurstCompile]
    struct CreateOtherDataUintJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<InputSplatRuntimeData> m_Input;
        [NativeDisableParallelForRestriction] public NativeArray<uint> m_Output;

        public void Execute(int index)
        {
            var input = m_Input[index];

            // Rotation as float4 → Norm10.10.10.2 (1 uint)
            Quaternion rotQ = input.rot;
            float4 rot = new float4(rotQ.x, rotQ.y, rotQ.z, rotQ.w);
            uint rotEncoded = EncodeQuatToNorm10(rot);
            m_Output[index * 2] = rotEncoded;

            // Scale as float3 → Norm11.10.11 (1 uint)
            uint scaleEncoded = EncodeFloat3ToNorm11(math.saturate(input.scale));
            m_Output[index * 2 + 1] = scaleEncoded;
        }
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

    static uint EncodeFloat3ToNorm11(float3 v) // 32 bits: 11.10.11
    {
        return (uint)(v.x * 2047.5f) | ((uint)(v.y * 1023.5f) << 11) | ((uint)(v.z * 2047.5f) << 21);
    }



    public static NativeArray<float3> DecodeAlphasToNativeFloat3(byte[] fileBytes, int numFaces, int numPointsPerTriangle, Allocator allocator)
    {
        int floatsPerPoint = 3; // each float3 = 3 floats
        int bytesPerPoint = floatsPerPoint * 4; // 3 floats * 4 bytes = 12 bytes
        int totalPoints = numFaces * numPointsPerTriangle;
        int expectedBytes = totalPoints * bytesPerPoint;

        if (fileBytes.Length < expectedBytes)
        {
            Debug.LogError($"Insufficient data: expected {expectedBytes} bytes, but got {fileBytes.Length} bytes.");
            return default;
        }

        NativeArray<float3> alphas = new NativeArray<float3>(totalPoints, allocator);

        for (int i = 0; i < totalPoints; i++)
        {
            int offset = i * bytesPerPoint;

            float x = BitConverter.ToSingle(fileBytes, offset);
            float y = BitConverter.ToSingle(fileBytes, offset + 4);
            float z = BitConverter.ToSingle(fileBytes, offset + 8);

            alphas[i] = new float3(x, y, z);

        }

        return alphas;
    }

    NativeArray<float> DecodeScalesToNative(byte[] fileBytes, int numberOfSplats, Allocator allocator)
    {
        int vectorSize = GaussianSplatAsset.GetVectorSize(_splatRenderer.asset.posFormat);
        int requiredLength = numberOfSplats * vectorSize;

        if (fileBytes.Length < requiredLength)
        {
            Debug.LogError($"Insufficient data: expected {requiredLength} bytes, but got {fileBytes.Length} bytes.");
            return default;
        }

        NativeArray<float> decodedScales = new NativeArray<float>(numberOfSplats, allocator, NativeArrayOptions.UninitializedMemory);

        for (int i = 0; i < numberOfSplats; i++)
        {
            int offset = i * vectorSize;
            float scale = BitConverter.ToSingle(fileBytes, offset);
            decodedScales[i] = scale;
        }

        return decodedScales;
    }

    static int NextMultipleOf(int size, int multipleOf)
    {
        return (size + multipleOf - 1) / multipleOf * multipleOf;
    }





}
