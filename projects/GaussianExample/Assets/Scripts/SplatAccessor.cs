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
using UnityEngine.XR;
using GaussianSplatting.Runtime;
using GaussianSplatting.Runtime.GaMeS;

public class SplatAccessor : MonoBehaviour, IDeformable
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

    private bool isPressed = false;


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

    public static void MirrorAlongY(Mesh mesh)
    {
        // Define mirror matrix (flip X)
        Matrix4x4 mirrorMatrix = Matrix4x4.Scale(new Vector3(-1, 1, 1));

        // Transform vertices
        Vector3[] vertices = mesh.vertices;
        for (int i = 0; i < vertices.Length; i++)
        {
            vertices[i] = mirrorMatrix.MultiplyPoint3x4(vertices[i]);
        }
        mesh.vertices = vertices;

        // Flip triangle winding (to keep normals outward)
        int[] triangles = mesh.triangles;
        for (int i = 0; i < triangles.Length; i += 3)
        {
            int temp = triangles[i + 1];
            triangles[i + 1] = triangles[i + 2];
            triangles[i + 2] = temp;
        }
        mesh.triangles = triangles;

        // Recalculate normals & bounds
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();
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

        decodedAlphasNative = GaMeSUtils.DecodeAlphasToNativeFloat3(fileBytes, faceCountEstimate, numberPtsPerTriangle, Allocator.Persistent);
        RegisterNativeCleanup(() => { if (decodedAlphasNative.IsCreated) decodedAlphasNative.Dispose(); });

        decodedScalesNative = GaMeSUtils.DecodeScalesToNative(fileScaleBytes, gaussianGaMeSSplatAsset.splatCount, Allocator.Persistent);
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
            addedMeshFilter.mesh = GaMeSUtils.TransformMesh(meshFilter.sharedMesh);
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
                UpdateMinMaxMeshMargins(v0, v1, v2, originalVertices[i0], originalVertices[i1], originalVertices[i2], ref minPointWorld, ref maxPointWorld, ref minPointLocal, ref maxPointLocal, ref minY, ref maxY);
            }
            else
            {
                backgroundVertexSet.Add(i0); backgroundVertexSet.Add(i1); backgroundVertexSet.Add(i2);
                backgroundTriangleIndicesList.Add(i / 3);
            }

        }

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
            selectedVertexWeights[idx2++] = math.clamp((y - minPointLocal.y) / denom, 0f, 1.0f);

        }
        idx = 0;
        foreach (int i in backgroundVertexSet) selectedBackgroundVertexIndices[idx++] = i;

        // Input allocations
        inputSplatsData = new NativeArray<InputSplatData>(originalTriangleIndices.Length * numberPtsPerTriangle, Allocator.Persistent);
        RegisterNativeCleanup(() => { if (inputSplatsData.IsCreated) inputSplatsData.Dispose(); });

        faceVertices = SplatMathUtils.GetMeshFaceSelectedVerticesNative(displacedVertices, triangles, originalTriangleIndices, Allocator.Persistent);
        RegisterNativeCleanup(() => { if (faceVertices.IsCreated) faceVertices.Dispose(); });

        xyzValues = GaMeSUtils.CreateXYZDataSelected(decodedAlphasNative, faceVertices, originalTriangleIndices, numberPtsPerTriangle);
        RegisterNativeCleanup(() => { if (xyzValues.IsCreated) xyzValues.Dispose(); });

        (rotations, scalings) = GaMeSUtils.CreateScaleRotationDataSelected(faceVertices, decodedScalesNative, originalTriangleIndices, numberPtsPerTriangle);
        RegisterNativeCleanup(() => { if (rotations.IsCreated) rotations.Dispose(); if (scalings.IsCreated) scalings.Dispose(); });

        var job = new GaMeSUtils.CreateAssetDataJobSelected()
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

        bgXyzValues = GaMeSUtils.CreateXYZDataSelected(decodedAlphasNative, bgFaceVertices, backgroundTriangleIndices, numberPtsPerTriangle);
        RegisterNativeCleanup(() => { if (bgXyzValues.IsCreated) bgXyzValues.Dispose(); });

        (bgRotations, bgScalings) = GaMeSUtils.CreateScaleRotationDataSelected(bgFaceVertices, decodedScalesNative, backgroundTriangleIndices, numberPtsPerTriangle);
        RegisterNativeCleanup(() => { if (bgRotations.IsCreated) bgRotations.Dispose(); if (bgScalings.IsCreated) bgScalings.Dispose(); });

        var jobBg = new GaMeSUtils.CreateAssetDataJobSelected()
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

    //TODO: extract debug function
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

            UpdateMinMaxMeshMargins(v0, v1, v2, originalVertices[i0], originalVertices[i1], originalVertices[i2], ref minPointWorld, ref maxPointWorld, ref minPointLocal, ref maxPointLocal, ref minY, ref maxY);


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

        xyzValues = GaMeSUtils.CreateXYZData(decodedAlphasNative, faceVertices, gaussianGaMeSSplatAsset.splatCount / numberPtsPerTriangle, numberPtsPerTriangle);
        RegisterNativeCleanup(() => { if (xyzValues.IsCreated) xyzValues.Dispose(); });

        (rotations, scalings) = GaMeSUtils.CreateScaleRotationData(faceVertices, decodedScalesNative, numberPtsPerTriangle);
        RegisterNativeCleanup(() => { if (rotations.IsCreated) rotations.Dispose(); if (scalings.IsCreated) scalings.Dispose(); });

        inputSplatsData = new NativeArray<InputSplatData>(_splatRenderer.asset.splatCount, Allocator.Persistent);
        RegisterNativeCleanup(() => { if (inputSplatsData.IsCreated) inputSplatsData.Dispose(); });

        var job = new GaMeSUtils.CreateAssetDataJob()
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

        if (ForceModeManager.Instance.CurrentForceMode == ForceMode.Drag)
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


        if (ForceModeManager.Instance.CurrentForceMode == ForceMode.None)
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


        if (!isCreateAssetJobActive && needsRebuild)
        {
            //TODO: check how we can optimaze it
            deformingMesh.SetVertices(displacedVertices);
            if (IsSelectionMode())
            {

                faceVertices = SplatMathUtils.GetMeshFaceSelectedVerticesNative(displacedVertices, triangles, originalTriangleIndices, Allocator.Persistent);
                xyzValues = GaMeSUtils.CreateXYZDataSelected(decodedAlphasNative, faceVertices, originalTriangleIndices, numberPtsPerTriangle);
                (rotations, scalings) = GaMeSUtils.CreateScaleRotationDataSelected(faceVertices, decodedScalesNative, originalTriangleIndices, numberPtsPerTriangle);

                var job = new GaMeSUtils.CreateAssetDataJobSelected()
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
                xyzValues = GaMeSUtils.CreateXYZData(decodedAlphasNative, faceVertices, _splatRenderer.asset.splatCount / numberPtsPerTriangle, numberPtsPerTriangle);
                (rotations, scalings) = GaMeSUtils.CreateScaleRotationData(faceVertices, decodedScalesNative, numberPtsPerTriangle);

                var job = new GaMeSUtils.CreateAssetDataJob()
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
    private void UpdateMinMaxMeshMargins(
  Vector3 v0, Vector3 v1, Vector3 v2,
     Vector3 local0, Vector3 local1, Vector3 local2,
     ref Vector3 minPointWorld, ref Vector3 maxPointWorld,
     ref Vector3 minPointLocal, ref Vector3 maxPointLocal, ref float minY, ref float maxY)
    {

        if (v0.y < minY)
        {
            minPointWorld = v0;
            minPointLocal = local0;
            minY = v0.y;
        }
        if (v1.y < minY)
        {
            minPointWorld = v1;
            minPointLocal = local1;
            minY = v1.y;
        }
        if (v2.y < minY)
        {
            minPointWorld = v2;
            minPointLocal = local2;
            minY = v2.y;
        }

        if (v0.y > maxY)
        {
            maxPointWorld = v0;
            maxPointLocal = local0;
            maxY = v0.y;
        }
        if (v1.y > maxY)
        {
            maxPointWorld = v1;
            maxPointLocal = local1;
            maxY = v1.y;
        }
        if (v2.y > maxY)
        {
            maxPointWorld = v2;
            maxPointLocal = local2;
            maxY = v2.y;
        }
    }

    void ProcessDeformationInput()
    {

        if (isPressed)
        {
            var springJob = new VertexPressJobSelected
            {
                deltaTime = Time.deltaTime,
                uniformScale = uniformScale,
                damageMultiplier = 2.5f,
                displacedVertices = displacedVertices,
                originalVertices = originalVertices,
                vertexVelocities = vertexVelocities,
                selectedVertexIndices = selectedVertexIndices
            };
            JobHandle handle = springJob.Schedule(selectedVertexIndices.Length, 64);
            handle.Complete();
            isPressed = false;
            needsRebuild = true;
        }
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

        //  deformingMesh.SetVertices(displacedVertices);
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

        // deformingMesh.SetVertices(displacedVertices);
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
        {
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
    public struct VertexPressJobSelected : IJobParallelFor
    {
        public float deltaTime;
        public float uniformScale;
        public float damageMultiplier;

        [NativeDisableParallelForRestriction] public NativeArray<float3> displacedVertices;
        [NativeDisableParallelForRestriction] public NativeArray<float3> originalVertices;
        [NativeDisableParallelForRestriction] public NativeArray<float3> vertexVelocities;
        [ReadOnly] public NativeArray<int> selectedVertexIndices;

        public void Execute(int index)
        {
            int i = selectedVertexIndices[index];

            float3 deform = damageMultiplier * vertexVelocities[i];
            displacedVertices[i] -= deform * (deltaTime);

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

    [BurstCompile]
    struct AddPressForceJobSelected : IJobParallelFor
    {

        [NativeDisableParallelForRestriction] public NativeArray<float3> displacedVertices;
        [NativeDisableParallelForRestriction] public NativeArray<float3> originalVertices;

        [NativeDisableParallelForRestriction] public NativeArray<float3> vertexVelocities;

        [ReadOnly] public NativeArray<int> selectedVertexIndices;

        [ReadOnly] public Vector3 pointLocal;
        [ReadOnly] public float uniformScale;
        [ReadOnly] public float deltaTime;
        [ReadOnly] public float deformRadius;
        [ReadOnly] public float maxDeform;
        [ReadOnly] public float damageFalloff;

        public void Execute(int index)
        {
            int i = selectedVertexIndices[index];
            Vector3 distanceFromCollision = displacedVertices[i] - (float3)pointLocal;
            Vector3 distanceFromOriginal = originalVertices[i] - displacedVertices[i];
            distanceFromCollision *= uniformScale;
            distanceFromOriginal *= uniformScale;

            float distFromCollision = distanceFromCollision.magnitude;
            float distFromOrigin = distanceFromOriginal.magnitude;
            if (distFromCollision < deformRadius)
            {
                // Smooth falloff
                float falloff = 1 - (distFromCollision / deformRadius) * damageFalloff;

                float xDeform = pointLocal.x * falloff;
                float yDeform = pointLocal.y * falloff;
                float zDeform = pointLocal.z * falloff;

                xDeform = Mathf.Clamp(xDeform, 0, maxDeform);
                yDeform = Mathf.Clamp(yDeform, 0, maxDeform);
                zDeform = Mathf.Clamp(zDeform, 0, maxDeform);

                vertexVelocities[i] += new float3(xDeform, yDeform, zDeform) * deltaTime;

            }

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
        //forceLocal = new Vector3(forceLocal.z, forceLocal.y, forceLocal.x);
        //pointLocal = new Vector3(pointLocal.z, pointLocal.y, pointLocal.x);



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


    public void AddPressForce(Vector3 point, Vector3 pressNormal, float maxDeform, float radius, float damageFalloff)
    {
        Vector3 pointLocal = transform.InverseTransformPoint(point);

        if (IsSelectionMode())
        {
            var job = new AddPressForceJobSelected
            {
                displacedVertices = displacedVertices,
                originalVertices = originalVertices,
                vertexVelocities = vertexVelocities,
                maxDeform = maxDeform,
                damageFalloff = damageFalloff,
                deformRadius = radius,
                uniformScale = uniformScale,
                selectedVertexIndices = selectedVertexIndices,
                deltaTime = Time.deltaTime,
                pointLocal = pointLocal
            };

            JobHandle handle = job.Schedule(selectedVertexIndices.Length, 64);
            handle.Complete();
        }

        isPressed = true;

    }


    unsafe void CreateAsset()
    {
        if (creator != null && gaussianGaMeSSplatAsset)
        {

            var newAsset = creator.CreateAsset("new asset", inputSplatsData, gaussianGaMeSSplatAsset.alphaData, gaussianGaMeSSplatAsset.scaleData, gaussianGaMeSSplatAsset.pointCloudPath);
            _splatRenderer.InjectAsset(newAsset);

        }
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
