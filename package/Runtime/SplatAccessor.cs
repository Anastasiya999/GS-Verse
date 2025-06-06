using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using GaussianSplatting.Runtime;
using GaussianSplatting.Shared;
using GaussianSplatting.Runtime.Utils;
using Unity.Mathematics;
using Unity.Collections;
using System;
using Unity.Jobs;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Burst;
using System.IO;

namespace GaussianSplatting.Runtime
{
    public class SplatAccessor : MonoBehaviour
    {

        private GaussianSplatRenderer _splatRenderer;
        private GaussianSplatRenderer _splatRendererBackground;
        private MeshFilter _meshFilter;
        private Mesh _mesh;
        private bool needsAssetUpdate = false;

        Mesh deformingMesh;
        public float springForce = 20f;
        float uniformScale = 1f;
        public float damping = 5f;

        public int numberPtsPerTriangle = 3;

        public GameObject boundingBoxObject;


        [HideInInspector]
        public string assetFilePath;

        private bool isClicked = false;

        float frameCount = 0f;

        private NativeArray<int> selectedVertexIndices;

        NativeArray<int> originalTriangleIndices;


        NativeArray<float3> originalVertices;
        NativeArray<float3> displacedVertices;
        NativeArray<float3> vertexVelocities;

        NativeArray<int> triangles;
        List<List<List<float>>> decodedAlphas;
        NativeArray<float3> decodedAlphasNative;

        List<float> decodedScales;
        NativeArray<float> decodedScalesNative;

        private JobHandle createAssetJobHandle;

        private NativeArray<InputSplatData> inputSplatsData;

        private NativeArray<InputSplatData> runTimeInputSplatsData;

        private NativeArray<float3> xyzValues;
        private NativeArray<quaternion> rotations;

        private NativeArray<float3> scalings;

        private NativeArray<float3> faceVertices;


        private NativeArray<float3> bgXyzValues;
        private NativeArray<quaternion> bgRotations;
        private NativeArray<float3> bgScalings;
        private NativeArray<float3> bgFaceVertices;
        NativeArray<int> backgroundTriangleIndices;
        private NativeArray<InputSplatData> backgroundInputSplatsData;
        private NativeArray<int> selectedBackgroundVertexIndices;
        private JobHandle createAssetBgJobHandle;

        private bool isCreateAssetJobActive = false;


        private bool isSegmented = false;


        private GaussianSplatRuntimeAssetCreator creator = null;

        bool IsSelectionMode() => boundingBoxObject != null;

        void Start()
        {
            _splatRenderer = GameObject.FindGameObjectWithTag("SplatRenderer").GetComponent<GaussianSplatRenderer>();

            if (_splatRenderer.asset.alphaData != null)
            {

                byte[] fileBytes = _splatRenderer.asset.alphaData.bytes;
                byte[] fileScaleBytes = _splatRenderer.asset.scaleData.bytes;
                decodedAlphasNative = DecodeAlphasToNativeFloat3(fileBytes, _splatRenderer.asset.splatCount / numberPtsPerTriangle, numberPtsPerTriangle, Allocator.Persistent);
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
            triangles = new NativeArray<int>(triangleCount, Allocator.Persistent);

            for (int i = 0; i < vertexCount; i++)
            {
                float3 v = verts[i];
                originalVertices[i] = v;
                displacedVertices[i] = v;
            }

            for (int i = 0; i < triangleCount; i++)
            {
                triangles[i] = tris[i];
            }

            creator = new GaussianSplatRuntimeAssetCreator();
            runTimeInputSplatsData = creator.CreateAsset(_splatRenderer.asset.pointCloudPath);

            if (IsSelectionMode())
            {
                _splatRendererBackground = GameObject.FindGameObjectWithTag("SplatBackground").GetComponent<GaussianSplatRenderer>();

                Transform meshTransform = transform;
                Bounds bounds = GetWorldBounds(boundingBoxObject);
                HashSet<int> vertexSet = new HashSet<int>();
                List<int> originalTriangleIndicesList = new List<int>();

                List<int> backgroundTriangleIndicesList = new List<int>();
                HashSet<int> backgroundVertexSet = new HashSet<int>();

                for (int i = 0; i < triangles.Length; i += 3)
                {
                    int i0 = triangles[i];
                    int i1 = triangles[i + 1];
                    int i2 = triangles[i + 2];

                    Vector3 v0 = meshTransform.TransformPoint(originalVertices[i0]);
                    Vector3 v1 = meshTransform.TransformPoint(originalVertices[i1]);
                    Vector3 v2 = meshTransform.TransformPoint(originalVertices[i2]);

                    // Check if any vertex of the triangle is inside the bounding box
                    if (bounds.Contains(v0) || bounds.Contains(v1) || bounds.Contains(v2))
                    {
                        vertexSet.Add(i0);
                        vertexSet.Add(i1);
                        vertexSet.Add(i2);
                        originalTriangleIndicesList.Add(i / 3);
                    }
                    else
                    {

                        backgroundVertexSet.Add(i0);
                        backgroundVertexSet.Add(i1);
                        backgroundVertexSet.Add(i2);
                        backgroundTriangleIndicesList.Add(i / 3);
                    }
                }

                selectedVertexIndices = new NativeArray<int>(vertexSet.Count, Allocator.Persistent);
                originalTriangleIndices = new NativeArray<int>(originalTriangleIndicesList.ToArray(), Allocator.Persistent);

                selectedBackgroundVertexIndices = new NativeArray<int>(backgroundVertexSet.Count, Allocator.Persistent);
                backgroundTriangleIndices = new NativeArray<int>(backgroundTriangleIndicesList.ToArray(), Allocator.Persistent);

                int idx = 0;
                foreach (int i in vertexSet)
                {
                    selectedVertexIndices[idx++] = i;
                }
                idx = 0;
                foreach (int i in backgroundVertexSet)
                {
                    selectedBackgroundVertexIndices[idx++] = i;
                }


                inputSplatsData = new NativeArray<InputSplatData>(originalTriangleIndices.Length * numberPtsPerTriangle, Allocator.Persistent);
                faceVertices = SplatMathUtils.GetMeshFaceSelectedVerticesNative(displacedVertices, triangles, originalTriangleIndices, Allocator.Persistent);
                xyzValues = CreateXYZDataSelected(decodedAlphasNative, faceVertices, originalTriangleIndices, numberPtsPerTriangle);
                (rotations, scalings) = CreateScaleRotationDataSelected(faceVertices, decodedScalesNative, originalTriangleIndices, numberPtsPerTriangle);

                var job = new CreateAssetDataJobSelected()
                {
                    m_InputPos = xyzValues,
                    m_InputRot = rotations,
                    m_InputScale = scalings,
                    m_Output = inputSplatsData,
                    m_PrevOutput = runTimeInputSplatsData,
                    m_originalTriangleIndices = originalTriangleIndices,
                    m_numberPtsPerTriangle = numberPtsPerTriangle

                };

                createAssetJobHandle = job.Schedule(originalTriangleIndices.Length * numberPtsPerTriangle, 8192);
                createAssetJobHandle.Complete();
                CreateAsset();

                backgroundInputSplatsData = new NativeArray<InputSplatData>(backgroundTriangleIndices.Length * numberPtsPerTriangle, Allocator.Persistent);
                bgFaceVertices = SplatMathUtils.GetMeshFaceSelectedVerticesNative(displacedVertices, triangles, backgroundTriangleIndices, Allocator.Persistent);
                bgXyzValues = CreateXYZDataSelected(decodedAlphasNative, bgFaceVertices, backgroundTriangleIndices, numberPtsPerTriangle);
                (bgRotations, bgScalings) = CreateScaleRotationDataSelected(bgFaceVertices, decodedScalesNative, backgroundTriangleIndices, numberPtsPerTriangle);

                var jobBg = new CreateAssetDataJobSelected()
                {
                    m_InputPos = bgXyzValues,
                    m_InputRot = bgRotations,
                    m_InputScale = bgScalings,
                    m_Output = backgroundInputSplatsData,
                    m_PrevOutput = runTimeInputSplatsData,
                    m_originalTriangleIndices = backgroundTriangleIndices,
                    m_numberPtsPerTriangle = numberPtsPerTriangle

                };

                createAssetBgJobHandle = jobBg.Schedule(backgroundTriangleIndices.Length * numberPtsPerTriangle, 8192);
                createAssetBgJobHandle.Complete();
                if (creator != null)
                {
                    var newAsset = creator.CreateAsset("new asset background", backgroundInputSplatsData, _splatRenderer.asset.alphaData, _splatRenderer.asset.scaleData, _splatRenderer.asset.pointCloudPath);
                    _splatRendererBackground.InjectAsset(newAsset);
                }

                ConfigureForegroundBackground(_splatRenderer, _splatRendererBackground);

            }
            else
            {

                faceVertices = SplatMathUtils.GetMeshFaceVerticesNative(gameObject, displacedVertices, triangles, Allocator.Persistent);
                xyzValues = CreateXYZData(decodedAlphasNative, faceVertices, _splatRenderer.asset.splatCount / numberPtsPerTriangle, numberPtsPerTriangle);
                (rotations, scalings) = CreateScaleRotationData(faceVertices, decodedScalesNative, numberPtsPerTriangle);
                inputSplatsData = new NativeArray<InputSplatData>(_splatRenderer.asset.splatCount, Allocator.Persistent);


                var job = new CreateAssetDataJob()
                {
                    m_InputPos = xyzValues,
                    m_InputRot = rotations,
                    m_InputScale = scalings,
                    m_PrevOutput = runTimeInputSplatsData,
                    m_Output = inputSplatsData
                };

                createAssetJobHandle = job.Schedule(xyzValues.Length, 8192);
                createAssetJobHandle.Complete();
                CreateAsset();

            }

        }

        void Update()
        {

            if (isCreateAssetJobActive)
            {
                if (createAssetJobHandle.IsCompleted)
                {
                    createAssetJobHandle.Complete();
                    CreateAsset();

                    // Dispose arrays
                    xyzValues.Dispose();
                    rotations.Dispose();
                    scalings.Dispose();
                    faceVertices.Dispose();

                    isCreateAssetJobActive = false;
                }
                return;
            }

            frameCount++;
            var dis = isClicked ? 1f : 0f;
            bool mouseDown = Input.GetMouseButton(0);
            if (mouseDown)
            {

                if (IsSelectionMode())
                {
                    var springJob = new VertexSpringJobSelected
                    {
                        deltaTime = Time.deltaTime,
                        springForce = springForce,
                        damping = damping,
                        uniformScale = uniformScale,
                        dis = dis,

                        displacedVertices = displacedVertices,
                        originalVertices = originalVertices,
                        vertexVelocities = vertexVelocities,

                        selectedVertexIndices = selectedVertexIndices
                    };
                    JobHandle handle = springJob.Schedule(selectedVertexIndices.Length, 64);
                    handle.Complete();
                }
                else
                {
                    var springJob = new VertexSpringJob
                    {
                        deltaTime = Time.deltaTime,
                        springForce = springForce,
                        damping = damping,
                        uniformScale = uniformScale,
                        dis = dis,

                        displacedVertices = displacedVertices,
                        originalVertices = originalVertices,
                        vertexVelocities = vertexVelocities,

                    };
                    JobHandle handle = springJob.Schedule(displacedVertices.Length, 64);
                    handle.Complete();
                }

            }
            else
            {


                if (IsSelectionMode())
                {
                    ReturnToOriginalShapeSelected(selectedVertexIndices);
                }
                else
                {
                    ReturnToOriginalShape();
                }



            }

            deformingMesh.SetVertices(displacedVertices);

            if (!isCreateAssetJobActive)
            {
                if (IsSelectionMode())
                {

                    faceVertices = SplatMathUtils.GetMeshFaceSelectedVerticesNative(displacedVertices, triangles, originalTriangleIndices, Allocator.Persistent);
                    xyzValues = CreateXYZDataSelected(decodedAlphasNative, faceVertices, originalTriangleIndices, numberPtsPerTriangle);
                    (rotations, scalings) = CreateScaleRotationDataSelected(faceVertices, decodedScalesNative, originalTriangleIndices, numberPtsPerTriangle);

                    var job = new CreateAssetDataJobSelected()
                    {
                        m_InputPos = xyzValues,
                        m_InputRot = rotations,
                        m_InputScale = scalings,
                        m_Output = inputSplatsData,
                        m_PrevOutput = runTimeInputSplatsData,
                        m_originalTriangleIndices = originalTriangleIndices,
                        m_numberPtsPerTriangle = numberPtsPerTriangle

                    };

                    createAssetJobHandle = job.Schedule(originalTriangleIndices.Length * numberPtsPerTriangle, 8192);
                }
                else
                {
                    faceVertices = SplatMathUtils.GetMeshFaceVerticesNative(gameObject, displacedVertices, triangles, Allocator.Persistent);
                    xyzValues = CreateXYZData(decodedAlphasNative, faceVertices, _splatRenderer.asset.splatCount / numberPtsPerTriangle, numberPtsPerTriangle);
                    (rotations, scalings) = CreateScaleRotationData(faceVertices, decodedScalesNative, numberPtsPerTriangle);

                    var job = new CreateAssetDataJob()
                    {
                        m_InputPos = xyzValues,
                        m_InputRot = rotations,
                        m_InputScale = scalings,
                        m_Output = inputSplatsData,
                        m_PrevOutput = runTimeInputSplatsData,

                    };
                    createAssetJobHandle = job.Schedule(xyzValues.Length, 8192);
                }


                isCreateAssetJobActive = true;
            }



        }

        private NativeArray<InputSplatData> CopyTo(NativeArray<InputSplatData> source)
        {
            NativeArray<InputSplatData> copy = new NativeArray<InputSplatData>(source.Length, Allocator.Persistent);
            NativeArray<InputSplatData>.Copy(source, copy);
            return copy;
        }

        void OnDestroy()
        {

            if (isCreateAssetJobActive)
            {
                createAssetJobHandle.Complete();
                isCreateAssetJobActive = false;
            }



            if (inputSplatsData.IsCreated)
                inputSplatsData.Dispose();
            if (runTimeInputSplatsData.IsCreated)
                runTimeInputSplatsData.Dispose();
            if (backgroundInputSplatsData.IsCreated)
                backgroundInputSplatsData.Dispose();
            DisposeIfCreated(ref originalVertices);
            DisposeIfCreated(ref displacedVertices);
            DisposeIfCreated(ref vertexVelocities);

            DisposeIfCreated(ref triangles);

            DisposeIfCreated(ref decodedAlphasNative);
            DisposeIfCreated(ref decodedScalesNative);

            DisposeIfCreated(ref inputSplatsData);
            DisposeIfCreated(ref runTimeInputSplatsData);

            DisposeIfCreated(ref faceVertices);
            DisposeIfCreated(ref xyzValues);
            DisposeIfCreated(ref rotations);
            DisposeIfCreated(ref scalings);

            DisposeIfCreated(ref originalTriangleIndices);
            DisposeIfCreated(ref selectedVertexIndices);

            DisposeIfCreated(ref bgFaceVertices);
            DisposeIfCreated(ref bgRotations);
            DisposeIfCreated(ref bgScalings);
            DisposeIfCreated(ref bgXyzValues);
            DisposeIfCreated(ref backgroundTriangleIndices);
            DisposeIfCreated(ref selectedBackgroundVertexIndices);
        }

        public static void SetRenderOrder(GaussianSplatRenderer renderer, int order)
        {
            if (renderer != null)
                renderer.m_RenderOrder = order;
        }

        public static void ConfigureForegroundBackground(GaussianSplatRenderer foreground, GaussianSplatRenderer background)
        {
            SetRenderOrder(background, 1);
            SetRenderOrder(foreground, 2);
        }

        private void DisposeIfCreated<T>(ref NativeArray<T> array) where T : struct
        {
            if (array.IsCreated)
            {
                array.Dispose();
            }
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
            for (int i = 0; i < displacedVertices.Length; i++)
            {
                displacedVertices[i] = Vector3.Lerp(displacedVertices[i], originalVertices[i], Time.deltaTime * 5f);
            }


            deformingMesh.SetVertices(displacedVertices);
        }

        void ReturnToOriginalShapeSelected(NativeArray<int> selectedVertexIndices)
        {
            for (int i = 0; i < selectedVertexIndices.Length; i++)
            {
                int index = selectedVertexIndices[i];
                displacedVertices[index] = Vector3.Lerp(displacedVertices[index], originalVertices[index], Time.deltaTime * 5f);
            }

            deformingMesh.SetVertices(displacedVertices);
        }

        [BurstCompile]
        public struct VertexSpringJob : IJobParallelFor
        {
            public float deltaTime;
            public float springForce;
            public float damping;
            public float uniformScale;
            public float dis;

            public NativeArray<float3> displacedVertices;
            public NativeArray<float3> originalVertices;
            public NativeArray<float3> vertexVelocities;

            public void Execute(int i)
            {

                float3 velocity = vertexVelocities[i];
                float3 displacement = (displacedVertices[i] - originalVertices[i]) * uniformScale;
                velocity -= displacement * springForce * deltaTime;
                velocity *= 1f - damping * deltaTime;
                vertexVelocities[i] = velocity;

                displacedVertices[i] += velocity * (deltaTime / uniformScale);

            }
        }

        [BurstCompile]
        public struct VertexSpringJobSelected : IJobParallelFor
        {
            public float deltaTime;
            public float springForce;
            public float damping;
            public float uniformScale;
            public float dis;

            [NativeDisableParallelForRestriction] public NativeArray<float3> displacedVertices;
            [NativeDisableParallelForRestriction] public NativeArray<float3> originalVertices;
            [NativeDisableParallelForRestriction] public NativeArray<float3> vertexVelocities;

            [ReadOnly] public NativeArray<int> selectedVertexIndices;

            public void Execute(int index)
            {
                //SELECTED
                int i = selectedVertexIndices[index];

                float3 velocity = vertexVelocities[i];
                float3 displacement = (displacedVertices[i] - originalVertices[i]) * uniformScale;
                velocity -= displacement * springForce * deltaTime;
                velocity *= 1f - damping * deltaTime;
                vertexVelocities[i] = velocity;

                displacedVertices[i] += velocity * (deltaTime / uniformScale);


            }
        }

        [BurstCompile]
        struct AddDeformingForceJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float3> displacedVertices;

            public NativeArray<float3> vertexVelocities;

            [ReadOnly] public Vector3 pointLocal;
            [ReadOnly] public Vector3 force;
            [ReadOnly] public float uniformScale;
            [ReadOnly] public float deltaTime;

            public void Execute(int i)
            {
                Vector3 pointToVertex = displacedVertices[i] - (float3)pointLocal;
                pointToVertex *= uniformScale;

                float attenuation = 1f / (1f + pointToVertex.sqrMagnitude);
                Vector3 appliedForce = force * attenuation * deltaTime;
                vertexVelocities[i] += (float3)appliedForce;

            }
        }

        [BurstCompile]
        struct AddDeformingForceJobSelected : IJobParallelFor
        {

            [NativeDisableParallelForRestriction] public NativeArray<float3> displacedVertices;

            [NativeDisableParallelForRestriction] public NativeArray<float3> vertexVelocities;

            [ReadOnly] public NativeArray<int> selectedVertexIndices;

            [ReadOnly] public Vector3 pointLocal;
            [ReadOnly] public Vector3 force;
            [ReadOnly] public float uniformScale;
            [ReadOnly] public float deltaTime;

            public void Execute(int index)
            {
                int i = selectedVertexIndices[index];
                Vector3 pointToVertex = displacedVertices[i] - (float3)pointLocal;
                pointToVertex *= uniformScale;

                float attenuation = 1f / (1f + pointToVertex.sqrMagnitude);
                Vector3 appliedForce = force * attenuation * deltaTime;
                vertexVelocities[i] += (float3)appliedForce;


            }
        }



        public void AddDeformingForce(Vector3 point, Vector3 force)
        {


            Vector3 pointLocal = transform.InverseTransformPoint(point);

            if (IsSelectionMode())
            {
                var job = new AddDeformingForceJobSelected
                {
                    displacedVertices = displacedVertices,
                    vertexVelocities = vertexVelocities,
                    selectedVertexIndices = selectedVertexIndices,
                    pointLocal = pointLocal,
                    force = force,
                    uniformScale = uniformScale,
                    deltaTime = Time.deltaTime
                };

                JobHandle handle = job.Schedule(selectedVertexIndices.Length, 64);
                handle.Complete();
            }
            else
            {
                var job = new AddDeformingForceJob
                {
                    displacedVertices = displacedVertices,
                    vertexVelocities = vertexVelocities,
                    pointLocal = pointLocal,
                    force = force,
                    uniformScale = uniformScale,
                    deltaTime = Time.deltaTime
                };

                JobHandle handle = job.Schedule(displacedVertices.Length, 64);
                handle.Complete();
            }


        }



        unsafe void CreateAsset()
        {
            if (creator != null)
            {
                var newAsset = creator.CreateAsset("new asset", inputSplatsData, _splatRenderer.asset.alphaData, _splatRenderer.asset.scaleData, _splatRenderer.asset.pointCloudPath);
                _splatRenderer.InjectAsset(newAsset);

            }
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

        NativeArray<float3> CreateXYZDataSelected(NativeArray<float3> alphas, NativeArray<float3> selectedVertices, NativeArray<int> originalTriangleIndices, int numPtsEachTriangle)
        {

            int numTriangles = selectedVertices.Length / 3;
            int totalPoints = numTriangles * numPtsEachTriangle;
            NativeArray<float3> data = new(totalPoints, Allocator.Persistent);


            CreateXYZDataJobSelected job = new CreateXYZDataJobSelected
            {
                m_Alphas = alphas,
                m_Vertices = selectedVertices,
                m_Output = data,
                m_numberPtsPerTriangle = numPtsEachTriangle,
                m_originalTriangleIndices = originalTriangleIndices,
            };
            job.Schedule(totalPoints, 8192).Complete();

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


        public static (NativeArray<quaternion> rotations, NativeArray<float3> scalings) CreateScaleRotationDataSelected(NativeArray<float3> selectedVertices, NativeArray<float> scales, NativeArray<int> originalTriangleIndices, int numPtsEachTriangle)
        {

            int numTriangles = selectedVertices.Length / 3;
            int totalPoints = numTriangles * numPtsEachTriangle;

            NativeArray<quaternion> rotations = new NativeArray<quaternion>(totalPoints, Allocator.Persistent);
            NativeArray<float3> scalings = new NativeArray<float3>(totalPoints, Allocator.Persistent);


            CreateRotationsAndScalesJobSelected job = new CreateRotationsAndScalesJobSelected
            {
                m_Vertices = selectedVertices,
                m_Scales = scales,
                m_numPtsEachTriangle = numPtsEachTriangle,
                Rotations = rotations,
                Scalings = scalings,
                m_originalTriangleIndices = originalTriangleIndices
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

        struct CreateXYZDataJobSelected : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float3> m_Alphas;
            [ReadOnly] public NativeArray<float3> m_Vertices;
            public int m_numberPtsPerTriangle;
            [NativeDisableParallelForRestriction] public NativeArray<float3> m_Output;

            [ReadOnly] public NativeArray<int> m_originalTriangleIndices;


            public unsafe void Execute(int index)
            {
                int triIndex = index / m_numberPtsPerTriangle;
                int pointIndex = index % m_numberPtsPerTriangle;

                int v0Idx = triIndex * 3;

                float3 v0 = m_Vertices[v0Idx];
                float3 v1 = m_Vertices[v0Idx + 1];
                float3 v2 = m_Vertices[v0Idx + 2];

                int originalTriIndex = m_originalTriangleIndices[triIndex];
                int alphaIndex = originalTriIndex * m_numberPtsPerTriangle + pointIndex;

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

        [BurstCompile]
        public struct CreateRotationsAndScalesJobSelected : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float3> m_Vertices;
            [ReadOnly] public NativeArray<float> m_Scales;
            public int m_numPtsEachTriangle;

            [WriteOnly] public NativeArray<quaternion> Rotations;
            [WriteOnly] public NativeArray<float3> Scalings;

            [ReadOnly] public NativeArray<int> m_originalTriangleIndices;

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

                int originalTriIndex = m_originalTriangleIndices[triangleIndex];
                int scaleIndex = originalTriIndex * m_numPtsEachTriangle + pointIndex;

                float scaleFactor = m_Scales[scaleIndex];
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
            [ReadOnly] public NativeArray<InputSplatData> m_PrevOutput;
            [NativeDisableParallelForRestriction] public NativeArray<InputSplatData> m_Output;

            public unsafe void Execute(int i)
            {

                m_Output[i] = new InputSplatData
                {
                    pos = m_InputPos[i],
                    scale = m_InputScale[i],
                    rot = m_InputRot[i],

                    nor = m_PrevOutput[i].nor,
                    dc0 = m_PrevOutput[i].dc0,
                    sh1 = m_PrevOutput[i].sh1,
                    sh2 = m_PrevOutput[i].sh2,
                    sh3 = m_PrevOutput[i].sh3,
                    sh4 = m_PrevOutput[i].sh4,
                    sh5 = m_PrevOutput[i].sh5,
                    sh6 = m_PrevOutput[i].sh6,
                    sh7 = m_PrevOutput[i].sh7,
                    sh8 = m_PrevOutput[i].sh8,
                    sh9 = m_PrevOutput[i].sh9,
                    shA = m_PrevOutput[i].shA,
                    shB = m_PrevOutput[i].shB,
                    shC = m_PrevOutput[i].shC,
                    shD = m_PrevOutput[i].shD,
                    shE = m_PrevOutput[i].shE,
                    shF = m_PrevOutput[i].shF,
                    opacity = m_PrevOutput[i].opacity
                };
            }
        }


        [BurstCompile]
        struct CreateAssetDataJobSelected : IJobParallelFor
        {
            [ReadOnly] public NativeArray<quaternion> m_InputRot;
            [ReadOnly] public NativeArray<float3> m_InputScale;
            [ReadOnly] public NativeArray<float3> m_InputPos;
            [ReadOnly] public NativeArray<InputSplatData> m_PrevOutput;
            [NativeDisableParallelForRestriction] public NativeArray<InputSplatData> m_Output;

            public int m_numberPtsPerTriangle;

            [ReadOnly] public NativeArray<int> m_originalTriangleIndices;


            public unsafe void Execute(int i)
            {

                int triIndex = i / m_numberPtsPerTriangle;
                int pointIndex = i % m_numberPtsPerTriangle;

                int originalTriIndex = m_originalTriangleIndices[triIndex];
                int index = originalTriIndex * m_numberPtsPerTriangle + pointIndex;


                m_Output[i] = new InputSplatData
                {
                    pos = m_InputPos[i],
                    scale = m_InputScale[i],
                    rot = m_InputRot[i],

                    nor = m_PrevOutput[index].nor,
                    dc0 = m_PrevOutput[index].dc0,
                    sh1 = m_PrevOutput[index].sh1,
                    sh2 = m_PrevOutput[index].sh2,
                    sh3 = m_PrevOutput[index].sh3,
                    sh4 = m_PrevOutput[index].sh4,
                    sh5 = m_PrevOutput[index].sh5,
                    sh6 = m_PrevOutput[index].sh6,
                    sh7 = m_PrevOutput[index].sh7,
                    sh8 = m_PrevOutput[index].sh8,
                    sh9 = m_PrevOutput[index].sh9,
                    shA = m_PrevOutput[index].shA,
                    shB = m_PrevOutput[index].shB,
                    shC = m_PrevOutput[index].shC,
                    shD = m_PrevOutput[index].shD,
                    shE = m_PrevOutput[index].shE,
                    shF = m_PrevOutput[index].shF,
                    opacity = m_PrevOutput[index].opacity
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

        Bounds GetWorldBounds(GameObject go)
        {
            Collider col = go.GetComponent<Collider>();
            if (col != null)
                return col.bounds;

            // No collider? Use transform info as an approximate AABB
            Vector3 center = go.transform.position;
            Vector3 size = Vector3.Scale(go.transform.localScale, Vector3.one);
            return new Bounds(center, size);
        }


    }
}