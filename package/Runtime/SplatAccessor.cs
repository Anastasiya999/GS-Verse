using System.Collections.Generic;
using UnityEngine;
using GaussianSplatting.Shared;
using GaussianSplatting.Runtime.Utils;
using Unity.Mathematics;
using Unity.Collections;
using System;
using Unity.Jobs;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Burst;

namespace GaussianSplatting.Runtime
{

    public class SplatAccessor : MonoBehaviour
    {

        private GaussianSplatRenderer _splatRenderer;

        Mesh deformingMesh;
        public float springForce = 20f;
        public bool blenderMesh = false;
        float uniformScale = 1f;
        public float damping = 5f;


        private int numberPtsPerTriangle = 3;

        public GameObject boundingBoxObject;

        private bool isClicked = false;


        private NativeArray<int> selectedVertexIndices;
        private NativeArray<float> selectedVertexWeights;

        NativeArray<int> originalTriangleIndices;


        NativeArray<float3> originalVertices;
        NativeArray<float3> displacedVertices;
        NativeArray<float3> vertexVelocities;

        NativeArray<int> triangles;
        NativeArray<float3> decodedAlphasNative;
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


        private GaussianSplatRuntimeAssetCreator creator = null;
        private GaussianGaMeSSplatAsset gaussianGaMeSSplatAsset = null;

        bool IsSelectionMode() => boundingBoxObject != null;

        // Keep track of cleanup actions to run if initialization fails halfway.
        private readonly List<Action> _deferredCleanup = new List<Action>();
        private bool _initializedSuccessfully = false;

        // runtime markers
        private GameObject topMarker;
        private GameObject bottomMarker;
        [Header("Visualization")]
        public Color topColor = Color.green;
        public Color bottomColor = Color.red;
        public float gizmoSphereRadius = 0.02f;
        public Vector3 minPointLocal;
        public Vector3 maxPointLocal;
        public Vector3 minPointWorld;
        public Vector3 maxPointWorld;
        private bool needsRebuild = true;
        [SerializeField] private float returnFinishEpsilon = 1e-4f;



        void Start()
        {

            try
            {
                InitializeSafely();
                _initializedSuccessfully = true;
            }
            catch (Exception ex)
            {
                Debug.LogError($"GaussianSplatInitializer: Initialization failed: {ex.Message}\n{ex.StackTrace}");
                RunDeferredCleanup();
            }
        }

        GaussianSplatRenderer FindWithChildTag(GameObject root, string childTag)
        {
            if (root == null) return null;
            foreach (var r in root.GetComponentsInChildren<GaussianSplatRenderer>(true))
            {
                if (r.gameObject.CompareTag(childTag)) return r;
            }
            return null;
        }

        private void InitializeSafely()
        {
            // 1) Find and validate renderer
            _splatRenderer = FindWithChildTag(gameObject, "SplatRenderer");
            if (_splatRenderer == null)
                throw new InvalidOperationException("SplatRenderer with tag 'SplatRenderer' not found or missing GaussianSplatRenderer component.");

            if (!(_splatRenderer.asset is GaussianGaMeSSplatAsset asset))
                throw new InvalidOperationException("SplatRenderer.asset is not a GaussianGaMeSSplatAsset or is null.");

            gaussianGaMeSSplatAsset = asset;

            // 2) Validate asset payloads
            numberPtsPerTriangle = gaussianGaMeSSplatAsset.numberOfSplatsPerFace;

            if (numberPtsPerTriangle <= 0)
                throw new InvalidOperationException($"Invalid numberOfSplatsPerFace: {numberPtsPerTriangle}");

            // 3) Decode binary data -> Native Arrays (validate sizes)
            byte[] fileBytes = gaussianGaMeSSplatAsset.alphaData.bytes;
            byte[] fileScaleBytes = gaussianGaMeSSplatAsset.scaleData.bytes;

            int faceCountEstimate = gaussianGaMeSSplatAsset.splatCount / numberPtsPerTriangle;
            if (faceCountEstimate <= 0)
                throw new InvalidOperationException("Calculated face count is zero or negative after dividing splatCount by numberPtsPerTriangle.");

            decodedAlphasNative = DecodeAlphasToNativeFloat3(fileBytes, faceCountEstimate, numberPtsPerTriangle, Allocator.Persistent);
            RegisterNativeCleanup(() => { if (decodedAlphasNative.IsCreated) decodedAlphasNative.Dispose(); });

            decodedScalesNative = DecodeScalesToNative(fileScaleBytes, gaussianGaMeSSplatAsset.splatCount, Allocator.Persistent);
            RegisterNativeCleanup(() => { if (decodedScalesNative.IsCreated) decodedScalesNative.Dispose(); });

            // 4) Optional mesh replacement (Resources.Load) validation
            if (!blenderMesh)
            {

                var loaded = Resources.Load<GameObject>(gaussianGaMeSSplatAsset.objPath);
                if (loaded == null)
                    throw new InvalidOperationException($"Resources.Load failed for path '{gaussianGaMeSSplatAsset.objPath}'.");

                var child = loaded.transform.childCount > 0 ? loaded.transform.GetChild(0) : null;
                if (child == null)
                    throw new InvalidOperationException("Loaded object from Resources has no children to get MeshFilter from.");

                var meshFilter = child.GetComponent<MeshFilter>();
                if (meshFilter == null || meshFilter.sharedMesh == null)
                    throw new InvalidOperationException("Child GameObject does not contain a MeshFilter with a sharedMesh.");

                MeshFilter addedMeshFilter = gameObject.AddComponent<MeshFilter>();
                addedMeshFilter.mesh = meshFilter.sharedMesh;
            }

            // 5) Disable child mesh renderers safely
            foreach (var mr in GetComponentsInChildren<MeshRenderer>())
            {
                if (mr != null) mr.enabled = false;
            }

            uniformScale = transform.localScale.x;

            // 6) Deforming mesh validation
            var mf = GetComponent<MeshFilter>();
            if (mf == null || mf.mesh == null)
                throw new InvalidOperationException("Missing MeshFilter or mesh on this GameObject.");

            deformingMesh = mf.mesh;

            var verts = deformingMesh.vertices;
            var tris = deformingMesh.triangles;

            if (verts == null || verts.Length == 0)
                throw new InvalidOperationException("Deforming mesh has no vertices.");
            if (tris == null || tris.Length == 0)
                throw new InvalidOperationException("Deforming mesh has no triangles.");

            int vertexCount = verts.Length;
            int triangleCount = tris.Length;

            // Allocate per-vertex and triangle native arrays and register cleanup
            originalVertices = new NativeArray<float3>(vertexCount, Allocator.Persistent);
            RegisterNativeCleanup(() => { if (originalVertices.IsCreated) originalVertices.Dispose(); });

            displacedVertices = new NativeArray<float3>(vertexCount, Allocator.Persistent);
            RegisterNativeCleanup(() => { if (displacedVertices.IsCreated) displacedVertices.Dispose(); });

            vertexVelocities = new NativeArray<float3>(vertexCount, Allocator.Persistent);
            RegisterNativeCleanup(() => { if (vertexVelocities.IsCreated) vertexVelocities.Dispose(); });

            triangles = new NativeArray<int>(triangleCount, Allocator.Persistent);
            RegisterNativeCleanup(() => { if (triangles.IsCreated) triangles.Dispose(); });

            // copy data
            for (int i = 0; i < vertexCount; i++)
            {
                float3 v = verts[i];
                originalVertices[i] = v;
                displacedVertices[i] = v;
                vertexVelocities[i] = new float3(0, 0, 0);
            }

            for (int i = 0; i < triangleCount; i++)
                triangles[i] = tris[i];

            // 7) Create runtime input splats data creator (validate pointCloudPath)
            creator = new GaussianSplatRuntimeAssetCreator();
            if (string.IsNullOrEmpty(gaussianGaMeSSplatAsset.pointCloudPath))
                throw new InvalidOperationException("pointCloudPath on GaussianGaMeSSplatAsset is null or empty.");

            runTimeInputSplatsData = creator.CreateAsset(gaussianGaMeSSplatAsset.pointCloudPath);
            // Register only if creator returned a NativeArray or similar; we assume it's a NativeArray<InputSplatData>
            RegisterNativeCleanup(() => { if (runTimeInputSplatsData.IsCreated) runTimeInputSplatsData.Dispose(); });

            // 8) Selection vs full mode
            if (IsSelectionMode())
            {
                InitializeSelectionMode();
            }
            else
            {
                InitializeFullMode();
            }

            // Clear deferred cleanup actions because initialization succeeded.
            _deferredCleanup.Clear();
        }
        private void InitializeSelectionMode()
        {
            float minY = float.MaxValue;
            float maxY = float.MinValue;


            Transform meshTransform = transform;
            Bounds bounds = GetWorldBounds(boundingBoxObject);
            var vertexSet = new HashSet<int>();
            var originalTriangleIndicesList = new List<int>();
            var backgroundTriangleIndicesList = new List<int>();
            var backgroundVertexSet = new HashSet<int>();

            for (int i = 0; i < triangles.Length; i += 3)
            {
                int i0 = triangles[i];
                int i1 = triangles[i + 1];
                int i2 = triangles[i + 2];

                Vector3 v0 = meshTransform.TransformPoint(originalVertices[i0]);
                Vector3 v1 = meshTransform.TransformPoint(originalVertices[i1]);
                Vector3 v2 = meshTransform.TransformPoint(originalVertices[i2]);

                var v0y = originalVertices[i0].y;
                var v1y = originalVertices[i1].y;
                var v2y = originalVertices[i2].y;

                if (IsPointInsideOBB(boundingBoxObject.transform, v0) ||
                    IsPointInsideOBB(boundingBoxObject.transform, v1) ||
                    IsPointInsideOBB(boundingBoxObject.transform, v2))
                {
                    vertexSet.Add(i0); vertexSet.Add(i1); vertexSet.Add(i2);
                    originalTriangleIndicesList.Add(i / 3);

                    if (v0y < minY)
                    {
                        minPointWorld = v0;
                        minPointLocal = originalVertices[i0];
                        minY = v0y;
                    }
                    if (v1y < minY)
                    {
                        minPointWorld = v1;
                        minPointLocal = originalVertices[i1];
                        minY = v1y;
                    }
                    if (v2y < minY)
                    {
                        minPointWorld = v2;
                        minPointLocal = originalVertices[i2];
                        minY = v2y;
                    }
                    //max

                    if (v0y > maxY)
                    {
                        maxPointWorld = v0;
                        maxPointLocal = originalVertices[i0];
                        maxY = v0y;
                    }
                    if (v1y > maxY)
                    {
                        maxPointWorld = v1;
                        maxPointLocal = originalVertices[i1];
                        maxY = v1y;
                    }
                    if (v2y > maxY)
                    {
                        maxPointWorld = v2;
                        maxPointLocal = originalVertices[i2];
                        maxY = v2y;
                    }
                }
                else
                {
                    backgroundVertexSet.Add(i0); backgroundVertexSet.Add(i1); backgroundVertexSet.Add(i2);
                    backgroundTriangleIndicesList.Add(i / 3);
                }
            }

            //CreateOrUpdateRuntimeMarkers();

            // Allocate & register
            selectedVertexIndices = new NativeArray<int>(vertexSet.Count, Allocator.Persistent);
            selectedVertexWeights = new NativeArray<float>(vertexSet.Count, Allocator.Persistent);
            RegisterNativeCleanup(() => { if (selectedVertexIndices.IsCreated) selectedVertexIndices.Dispose(); });

            originalTriangleIndices = new NativeArray<int>(originalTriangleIndicesList.ToArray(), Allocator.Persistent);
            RegisterNativeCleanup(() => { if (originalTriangleIndices.IsCreated) originalTriangleIndices.Dispose(); });

            selectedBackgroundVertexIndices = new NativeArray<int>(backgroundVertexSet.Count, Allocator.Persistent);
            RegisterNativeCleanup(() => { if (selectedBackgroundVertexIndices.IsCreated) selectedBackgroundVertexIndices.Dispose(); });

            backgroundTriangleIndices = new NativeArray<int>(backgroundTriangleIndicesList.ToArray(), Allocator.Persistent);
            RegisterNativeCleanup(() => { if (backgroundTriangleIndices.IsCreated) backgroundTriangleIndices.Dispose(); });

            // fill sets
            int idx = 0;
            int idx2 = 0;
            float denom = maxPointLocal.y - minPointLocal.y;
            if (math.abs(denom) < 1e-6f) denom = 1f;
            foreach (int i in vertexSet)
            {

                selectedVertexIndices[idx++] = i;
                float y = originalVertices[i].y;
                selectedVertexWeights[idx2++] = 1f - math.clamp((y - minPointLocal.y) / denom, 0f, 1.0f);

            }
            idx = 0;
            foreach (int i in backgroundVertexSet) selectedBackgroundVertexIndices[idx++] = i;

            // Input allocations
            inputSplatsData = new NativeArray<InputSplatData>(originalTriangleIndices.Length * numberPtsPerTriangle, Allocator.Persistent);
            RegisterNativeCleanup(() => { if (inputSplatsData.IsCreated) inputSplatsData.Dispose(); });

            faceVertices = SplatMathUtils.GetMeshFaceSelectedVerticesNative(displacedVertices, triangles, originalTriangleIndices, Allocator.Persistent);
            RegisterNativeCleanup(() => { if (faceVertices.IsCreated) faceVertices.Dispose(); });

            xyzValues = CreateXYZDataSelected(decodedAlphasNative, faceVertices, originalTriangleIndices, numberPtsPerTriangle);
            RegisterNativeCleanup(() => { if (xyzValues.IsCreated) xyzValues.Dispose(); });

            (rotations, scalings) = CreateScaleRotationDataSelected(faceVertices, decodedScalesNative, originalTriangleIndices, numberPtsPerTriangle);
            RegisterNativeCleanup(() => { if (rotations.IsCreated) rotations.Dispose(); if (scalings.IsCreated) scalings.Dispose(); });

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

            // Background
            backgroundInputSplatsData = new NativeArray<InputSplatData>(backgroundTriangleIndices.Length * numberPtsPerTriangle, Allocator.Persistent);
            RegisterNativeCleanup(() => { if (backgroundInputSplatsData.IsCreated) backgroundInputSplatsData.Dispose(); });

            bgFaceVertices = SplatMathUtils.GetMeshFaceSelectedVerticesNative(displacedVertices, triangles, backgroundTriangleIndices, Allocator.Persistent);
            RegisterNativeCleanup(() => { if (bgFaceVertices.IsCreated) bgFaceVertices.Dispose(); });

            bgXyzValues = CreateXYZDataSelected(decodedAlphasNative, bgFaceVertices, backgroundTriangleIndices, numberPtsPerTriangle);
            RegisterNativeCleanup(() => { if (bgXyzValues.IsCreated) bgXyzValues.Dispose(); });

            (bgRotations, bgScalings) = CreateScaleRotationDataSelected(bgFaceVertices, decodedScalesNative, backgroundTriangleIndices, numberPtsPerTriangle);
            RegisterNativeCleanup(() => { if (bgRotations.IsCreated) bgRotations.Dispose(); if (bgScalings.IsCreated) bgScalings.Dispose(); });

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
        }

        private void CreateOrUpdateRuntimeMarkers()
        {
            // create marker helper
            if (topMarker == null)
            {
                topMarker = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                topMarker.name = gameObject.name + "_TopMarker";
                DestroyImmediate(topMarker.GetComponent<Collider>()); // no collider
                topMarker.hideFlags = HideFlags.DontSaveInBuild | HideFlags.NotEditable;
            }

            if (bottomMarker == null)
            {
                bottomMarker = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                bottomMarker.name = gameObject.name + "_BottomMarker";
                DestroyImmediate(bottomMarker.GetComponent<Collider>());
                bottomMarker.hideFlags = HideFlags.DontSaveInBuild | HideFlags.NotEditable;
            }

            topMarker.transform.position = maxPointWorld;
            bottomMarker.transform.position = minPointWorld;

            // scale markers to match gizmoSphereRadius (approximate)
            float scale = gizmoSphereRadius * 2f;
            topMarker.transform.localScale = new Vector3(scale, scale, scale);
            bottomMarker.transform.localScale = new Vector3(scale, scale, scale);

            // set marker colors (shared material to avoid leaking many materials)
            SetMarkerColor(topMarker, topColor);
            SetMarkerColor(bottomMarker, bottomColor);
        }

        private void SetMarkerColor(GameObject go, Color col)
        {
            var mr = go.GetComponent<MeshRenderer>();
            if (mr == null) return;
            // Use an instance material only once (small leak risk if frequently created; acceptable for debug helpers)
            if (mr.sharedMaterial == null || mr.sharedMaterial.name == "Default-Material")
                mr.sharedMaterial = new Material(Shader.Find("Standard"));
            mr.sharedMaterial.color = col;
        }

        private void InitializeFullMode()
        {

            Transform meshTransform = transform;
            var vertexSet = new HashSet<int>();
            float minY = float.MaxValue;
            float maxY = float.MinValue;

            for (int i = 0; i < triangles.Length; i += 3)
            {
                int i0 = triangles[i];
                int i1 = triangles[i + 1];
                int i2 = triangles[i + 2];

                Vector3 v0 = meshTransform.TransformPoint(originalVertices[i0]);
                Vector3 v1 = meshTransform.TransformPoint(originalVertices[i1]);
                Vector3 v2 = meshTransform.TransformPoint(originalVertices[i2]);

                var v0y = originalVertices[i0].y;
                var v1y = originalVertices[i1].y;
                var v2y = originalVertices[i2].y;

                vertexSet.Add(i0); vertexSet.Add(i1); vertexSet.Add(i2);


                if (v0y < minY)
                {
                    minPointWorld = v0;
                    minPointLocal = originalVertices[i0];
                    minY = v0y;
                }
                if (v1y < minY)
                {
                    minPointWorld = v1;
                    minPointLocal = originalVertices[i1];
                    minY = v1y;
                }
                if (v2y < minY)
                {
                    minPointWorld = v2;
                    minPointLocal = originalVertices[i2];
                    minY = v2y;
                }
                //max

                if (v0y > maxY)
                {
                    maxPointWorld = v0;
                    maxPointLocal = originalVertices[i0];
                    maxY = v0y;
                }
                if (v1y > maxY)
                {
                    maxPointWorld = v1;
                    maxPointLocal = originalVertices[i1];
                    maxY = v1y;
                }
                if (v2y > maxY)
                {
                    maxPointWorld = v2;
                    maxPointLocal = originalVertices[i2];
                    maxY = v2y;
                }


            }
            selectedVertexWeights = new NativeArray<float>(vertexSet.Count, Allocator.Persistent);
            // fill sets
            int idx = 0;
            int idx2 = 0;
            float denom = maxPointLocal.y - minPointLocal.y;
            if (math.abs(denom) < 1e-6f) denom = 1f;
            foreach (int i in vertexSet)
            {

                float y = originalVertices[i].y;
                selectedVertexWeights[idx2++] = math.clamp((y - minPointLocal.y) / denom, 0f, 1f);

            }


            faceVertices = SplatMathUtils.GetMeshFaceVerticesNative(gameObject, displacedVertices, triangles, Allocator.Persistent);
            RegisterNativeCleanup(() => { if (faceVertices.IsCreated) faceVertices.Dispose(); });

            xyzValues = CreateXYZData(decodedAlphasNative, faceVertices, gaussianGaMeSSplatAsset.splatCount / numberPtsPerTriangle, numberPtsPerTriangle);
            RegisterNativeCleanup(() => { if (xyzValues.IsCreated) xyzValues.Dispose(); });

            (rotations, scalings) = CreateScaleRotationData(faceVertices, decodedScalesNative, numberPtsPerTriangle);
            RegisterNativeCleanup(() => { if (rotations.IsCreated) rotations.Dispose(); if (scalings.IsCreated) scalings.Dispose(); });

            inputSplatsData = new NativeArray<InputSplatData>(_splatRenderer.asset.splatCount, Allocator.Persistent);
            RegisterNativeCleanup(() => { if (inputSplatsData.IsCreated) inputSplatsData.Dispose(); });

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
        private void RegisterNativeCleanup(Action cleanupAction)
        {
            if (cleanupAction != null) _deferredCleanup.Add(cleanupAction);
        }

        private void RunDeferredCleanup()
        {
            for (int i = _deferredCleanup.Count - 1; i >= 0; --i)
            {
                try { _deferredCleanup[i]?.Invoke(); }
                catch (Exception e) { Debug.LogWarning($"Cleanup action failed: {e.Message}"); }
            }
            _deferredCleanup.Clear();
        }

        bool IsPointInsideOBB(Transform boxTransform, Vector3 point)
        {

            MeshFilter meshFilter = boxTransform.GetComponent<MeshFilter>();
            if (meshFilter == null || meshFilter.sharedMesh == null)
                return false;

            Bounds localBounds = meshFilter.sharedMesh.bounds;

            Vector3 localPoint = boxTransform.InverseTransformPoint(point);

            Vector3 halfSize = localBounds.extents;
            return Mathf.Abs(localPoint.x) <= halfSize.x &&
                   Mathf.Abs(localPoint.y) <= halfSize.y &&
                   Mathf.Abs(localPoint.z) <= halfSize.z;
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


            var dis = 1f;
            bool mouseDown = isClicked;

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
                        selectedVertexWeights = selectedVertexWeights,
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
                        selectedVertexWeights = selectedVertexWeights,

                    };
                    JobHandle handle = springJob.Schedule(displacedVertices.Length, 64);
                    handle.Complete();
                }

            }
            else
            {
                if (IsSelectionMode())
                {
                    needsRebuild = ReturnToOriginalShapeSelected(selectedVertexIndices);
                }
                else
                {
                    needsRebuild = ReturnToOriginalShape();
                }

            }

            deformingMesh.SetVertices(displacedVertices);
            // deformingMesh.RecalculateNormals();

            if (!isCreateAssetJobActive && needsRebuild)
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
                needsRebuild = false; // reset until something changes next
            }



        }

        private void OnDrawGizmos()
        {

        }


        private void OnDestroy()
        {
            // 1) Ensure jobs that produce/consume data are finished.
            try
            {
                if (isCreateAssetJobActive)
                {
                    try
                    {
                        createAssetJobHandle.Complete();
                    }
                    catch (Exception ex)
                    {
                        Debug.LogWarning($"Failed to complete createAssetJobHandle in OnDestroy: {ex.Message}");
                    }
                    isCreateAssetJobActive = false;
                }
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"Error while completing jobs in OnDestroy: {ex.Message}");
            }

            // 2) Dispose in reverse dependency order. Each call is safe and idempotent.
            // Background / input arrays first
            DisposeIfCreated(ref backgroundInputSplatsData);
            DisposeIfCreated(ref inputSplatsData);
            DisposeIfCreated(ref runTimeInputSplatsData);

            // Background job outputs & temporaries
            DisposeIfCreated(ref bgXyzValues);
            DisposeIfCreated(ref bgRotations);
            DisposeIfCreated(ref bgScalings);
            DisposeIfCreated(ref bgFaceVertices);

            // Main job outputs & temporaries
            DisposeIfCreated(ref xyzValues);
            DisposeIfCreated(ref rotations);
            DisposeIfCreated(ref scalings);
            DisposeIfCreated(ref faceVertices);

            // Decoded raw data
            DisposeIfCreated(ref decodedAlphasNative);
            DisposeIfCreated(ref decodedScalesNative);

            // Selection arrays
            DisposeIfCreated(ref selectedBackgroundVertexIndices);
            DisposeIfCreated(ref backgroundTriangleIndices);
            DisposeIfCreated(ref selectedVertexIndices);
            DisposeIfCreated(ref selectedVertexWeights);
            DisposeIfCreated(ref originalTriangleIndices);

            // Mesh arrays last (they are more fundamental)
            DisposeIfCreated(ref triangles);
            DisposeIfCreated(ref vertexVelocities);
            DisposeIfCreated(ref displacedVertices);
            DisposeIfCreated(ref originalVertices);
        }

        // Exception-safe, idempotent dispose helper
        private void DisposeIfCreated<T>(ref NativeArray<T> array) where T : struct
        {
            try
            {
                if (array.IsCreated)
                {
                    array.Dispose();
                }
            }
            catch (Exception e)
            {
                Debug.LogWarning($"Error disposing NativeArray<{typeof(T).Name}>: {e.Message}");
            }
            finally
            {
                // Reset to default so future disposals / checks are safe and can't double-dispose
                array = default;
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

        bool ReturnToOriginalShape()
        {
            float maxDispSq = 0f;
            for (int i = 0; i < displacedVertices.Length; i++)
            {
                float3 diff = displacedVertices[i] - originalVertices[i];
                float dsq = math.lengthsq(diff);
                if (dsq > maxDispSq) maxDispSq = dsq;
                displacedVertices[i] = Vector3.Lerp(displacedVertices[i], originalVertices[i], Time.deltaTime * 5f);
            }

            deformingMesh.SetVertices(displacedVertices);
            return maxDispSq > returnFinishEpsilon * returnFinishEpsilon;
        }

        bool ReturnToOriginalShapeSelected(NativeArray<int> selectedVertexIndices)
        {
            float maxDispSq = 0f;
            for (int i = 0; i < selectedVertexIndices.Length; i++)
            {

                int index = selectedVertexIndices[i];
                float3 diff = displacedVertices[index] - originalVertices[index];
                float dsq = math.lengthsq(diff);
                if (dsq > maxDispSq) maxDispSq = dsq;
                displacedVertices[index] = Vector3.Lerp(displacedVertices[index], originalVertices[index], Time.deltaTime * 5f);
            }

            deformingMesh.SetVertices(displacedVertices);
            return maxDispSq > returnFinishEpsilon * returnFinishEpsilon;
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
            [ReadOnly] public NativeArray<float> selectedVertexWeights;
            const float kWeightEps = 1e-3f;   // clamp threshold
            const float kVelEpsSq = 1e-10f;  // tiny velocity^2

            public void Execute(int i)
            {
                // float3 velocity = vertexVelocities[i];
                // if (velocity.x != 0f && velocity.y != 0f && velocity.z != 0f)
                // {
                //     float3 displacement = (displacedVertices[i] - originalVertices[i]) * uniformScale;
                //     velocity -= displacement * springForce * deltaTime;
                //     velocity *= 1f - damping * deltaTime;
                //     vertexVelocities[i] = velocity;

                //     displacedVertices[i] += velocity * (deltaTime / uniformScale);
                // }

                float w = 1.0f;

                float3 v = vertexVelocities[i];
                if (v.x != 0f && v.y != 0f && v.z != 0f)
                {
                    // If weight is ~0, hard-freeze the vertex
                    if (w <= kWeightEps)
                    {
                        vertexVelocities[i] = float3.zero;
                        displacedVertices[i] = originalVertices[i];
                        return;
                    }

                    // displacement in local space
                    float3 disp = (displacedVertices[i] - originalVertices[i]) * uniformScale;

                    // apply spring only (scaled by weight)
                    v -= disp * (springForce * w) * deltaTime;

                    // DAMPING SHOULD NOT BE SCALED BY WEIGHT (or make it stronger near bottom)
                    v *= 1f - damping * deltaTime;

                    // integrate position **scaled by weight** so bottom moves less
                    float3 vWeighted = v * w;
                    displacedVertices[i] += vWeighted * (deltaTime / uniformScale);

                    // optionally store weighted velocity to kill carry-over near bottom
                    vertexVelocities[i] = (math.lengthsq(vWeighted) < kVelEpsSq) ? float3.zero : vWeighted;

                }

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

            public float topY;
            public float bottomY;

            [NativeDisableParallelForRestriction] public NativeArray<float3> displacedVertices;
            [NativeDisableParallelForRestriction] public NativeArray<float3> originalVertices;
            [NativeDisableParallelForRestriction] public NativeArray<float3> vertexVelocities;

            [ReadOnly] public NativeArray<float> selectedVertexWeights;
            [ReadOnly] public NativeArray<int> selectedVertexIndices;

            const float kWeightEps = 1e-3f;   // clamp threshold
            const float kVelEpsSq = 1e-10f;  // tiny velocity^2

            public void Execute(int index)
            {/*
                int i = selectedVertexIndices[index];
                float w = selectedVertexWeights[index];

                float3 velocity = vertexVelocities[i];

                float3 localPos = displacedVertices[i];


                if (velocity.x != 0f && velocity.y != 0f && velocity.z != 0f)
                {
                    float3 displacement = (displacedVertices[i] - originalVertices[i]) * uniformScale;
                    velocity -= displacement * springForce * w * deltaTime;
                    velocity *= 1f - damping * deltaTime;
                    vertexVelocities[i] = velocity * w;
                    displacedVertices[i] += velocity * w * (deltaTime / uniformScale);

                }
                 */
                int i = selectedVertexIndices[index];
                float w = selectedVertexWeights[index];

                float3 v = vertexVelocities[i];
                if (v.x != 0f && v.y != 0f && v.z != 0f)
                {
                    // If weight is ~0, hard-freeze the vertex
                    if (w <= kWeightEps)
                    {
                        vertexVelocities[i] = float3.zero;
                        displacedVertices[i] = originalVertices[i];
                        return;
                    }

                    // displacement in local space
                    float3 disp = (displacedVertices[i] - originalVertices[i]) * uniformScale;

                    // apply spring only (scaled by weight)
                    v -= disp * (springForce * w) * deltaTime;

                    // DAMPING SHOULD NOT BE SCALED BY WEIGHT (or make it stronger near bottom)
                    v *= 1f - damping * deltaTime;

                    // integrate position **scaled by weight** so bottom moves less
                    float3 vWeighted = v * w;
                    displacedVertices[i] += vWeighted * (deltaTime / uniformScale);

                    // optionally store weighted velocity to kill carry-over near bottom
                    vertexVelocities[i] = (math.lengthsq(vWeighted) < kVelEpsSq) ? float3.zero : vWeighted;

                }


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

                if (force == Vector3.zero)
                {
                    vertexVelocities[i] = Vector3.zero;
                }
                else
                {
                    Vector3 pointToVertex = displacedVertices[i] - (float3)pointLocal;
                    pointToVertex *= uniformScale;

                    float attenuation = 1f / (1f + pointToVertex.sqrMagnitude);
                    Vector3 appliedForce = force * attenuation * deltaTime;
                    vertexVelocities[i] += (float3)appliedForce;


                }

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
                if (force == Vector3.zero)
                {
                    vertexVelocities[i] = Vector3.zero;
                }
                else
                {
                    Vector3 pointToVertex = displacedVertices[i] - (float3)pointLocal;
                    pointToVertex *= uniformScale;

                    float attenuation = 1f / (1f + pointToVertex.sqrMagnitude);
                    Vector3 appliedForce = force * attenuation * deltaTime;
                    vertexVelocities[i] += (float3)appliedForce;

                }


            }
        }

        private const float SLEEPING_THRESHOLD_SQR = 0.0001f * 0.0001f;

        [BurstCompile]
        struct AddStretchForceJobSelected : IJobParallelFor
        {

            [NativeDisableParallelForRestriction] public NativeArray<float3> displacedVertices;
            [NativeDisableParallelForRestriction] public NativeArray<float3> originalVertices;
            [NativeDisableParallelForRestriction] public NativeArray<float3> vertexVelocities;

            [ReadOnly] public NativeArray<int> selectedVertexIndices;
            [ReadOnly] public Vector3 currPos;
            [ReadOnly] public Vector3 prevPos;

            [ReadOnly] public float deltaTime;
            [ReadOnly] public float damping;



            public void Execute(int index)
            {
                int i = selectedVertexIndices[index];
                var originalVertex = (float3)originalVertices[i];


                //   float stretchFactor = SplatAccessor.CalculateStretchFactor(originalVertex, anchorPoint, originalTop, stretchAxis);
                var distance = Vector3.Distance(currPos, (Vector3)originalVertex);
                float t = Mathf.Clamp01(1f - distance / 5.0f);
                var force = t;

                float influence = SplatAccessor.Falloff(t);

                // Move vertex toward anchor, but influenced by falloff
                Vector3 direction = (currPos - (Vector3)originalVertex);
                displacedVertices[i] = originalVertex + (float3)direction * (1f - influence);

            }
        }

        static float Falloff(float t)
        {
            // Example: smoothstep-like falloff
            return t * t * (3f - 2f * t); // Smoothstep
        }

        public static float CalculateStretchFactor(Vector3 vertex, Vector3 anchorPoint, Vector3 originalTop, Vector3 stretchAxis)
        {
            Vector3 axis = stretchAxis.normalized;

            float vertexDistance = Vector3.Dot(vertex - anchorPoint, axis);
            float totalDistance = Vector3.Dot(originalTop - anchorPoint, axis);

            if (totalDistance < 0.001f)
            {
                return 0;
            }


            return Mathf.Clamp01(vertexDistance / totalDistance);
        }


        public void AddDeformingForce(Vector3 point, Vector3 force)
        {


            Vector3 pointLocal = transform.InverseTransformPoint(point);
            Vector3 forceLocal = transform.InverseTransformDirection(force);

            if (IsSelectionMode())
            {
                var job = new AddDeformingForceJobSelected
                {
                    displacedVertices = displacedVertices,
                    vertexVelocities = vertexVelocities,
                    selectedVertexIndices = selectedVertexIndices,
                    pointLocal = pointLocal,
                    force = forceLocal,
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

            needsRebuild = true;
        }


        public void AddStretchingForce(Vector3 currPos, Vector3 prevPos)
        {

            if (IsSelectionMode())
            {
                var job = new AddStretchForceJobSelected
                {
                    displacedVertices = displacedVertices,
                    originalVertices = originalVertices,
                    vertexVelocities = vertexVelocities,
                    currPos = currPos,
                    prevPos = prevPos,


                    selectedVertexIndices = selectedVertexIndices,

                    deltaTime = Time.deltaTime,
                    damping = damping,


                };

                JobHandle handle = job.Schedule(selectedVertexIndices.Length, 64);
                handle.Complete();
            }



        }



        unsafe void CreateAsset()
        {
            if (creator != null && gaussianGaMeSSplatAsset)
            {
                var newAsset = creator.CreateAsset("new asset", inputSplatsData, gaussianGaMeSSplatAsset.alphaData, gaussianGaMeSSplatAsset.scaleData, gaussianGaMeSSplatAsset.pointCloudPath);
                _splatRenderer.InjectAsset(newAsset);

            }
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
            int bytesPerPoint = floatsPerPoint * GaussianSplatAsset.GetVectorSize(GaussianSplatAsset.VectorFormat.Norm11); // 3 floats * 4 bytes = 12 bytes
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
            int vectorSize = GaussianSplatAsset.GetVectorSize(GaussianSplatAsset.VectorFormat.Norm11);
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