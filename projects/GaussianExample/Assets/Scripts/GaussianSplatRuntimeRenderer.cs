using System;
using System.Collections;
using System.Collections.Generic;
using GaussianSplatting.Runtime;
using GaussianSplatting.Runtime.GaMeS;
using GaussianSplatting.Runtime.Utils;
using GaussianSplatting.Shared;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

public class GaussianSplatRuntimeRenderer : MonoBehaviour
{

    public bool blenderMesh = false;

    private GaussianSplatRenderer _splatRenderer;
    private GaussianGaMeSSplatAsset gaussianGaMeSSplatAsset = null;
    private GaussianSplatRuntimeAssetCreator creator = null;
    private int numberPtsPerTriangle = 3;


    NativeArray<float3> decodedAlphasNative;
    NativeArray<float> decodedScalesNative;
    NativeArray<float3> originalVertices;
    NativeArray<float3> displacedVertices;
    NativeArray<int> triangles;

    private NativeArray<InputSplatData> runTimeInputSplatsData;
    private NativeArray<InputSplatData> inputSplatsData;

    private NativeArray<float3> xyzValues;
    private NativeArray<quaternion> rotations;

    private NativeArray<float3> scalings;

    private NativeArray<float3> faceVertices;

    float uniformScale = 1f;

    private readonly List<Action> _deferredCleanup = new List<Action>();
    Mesh deformingMesh;
    private JobHandle createAssetJobHandle;
    private bool needsRebuild = false;
    private bool isCreateAssetJobActive = false;

    MeshCollider meshCollider;

    void Awake()
    {
        meshCollider = GetComponent<MeshCollider>();
    }
    void Start()
    {
        try
        {
            InitializeSafely();

        }
        catch (Exception ex)
        {
            Debug.LogError($"GaussianSplatInitializer: Initialization failed: {ex.Message}\n{ex.StackTrace}");

        }
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
        _splatRenderer = GetComponent<GaussianSplatRenderer>();

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
        MirrorAlongY(deformingMesh);

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

        triangles = new NativeArray<int>(triangleCount, Allocator.Persistent);
        RegisterNativeCleanup(() => { if (triangles.IsCreated) triangles.Dispose(); });

        // copy data
        for (int i = 0; i < vertexCount; i++)
        {
            float3 v = verts[i];
            originalVertices[i] = v;
            displacedVertices[i] = v;
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

        InitializeFullMode();


        // Clear deferred cleanup actions because initialization succeeded.
        _deferredCleanup.Clear();
    }

    private void InitializeFullMode()
    {

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

    unsafe void CreateAsset()
    {
        if (creator != null && gaussianGaMeSSplatAsset)
        {

            var newAsset = creator.CreateAsset("new asset", inputSplatsData, gaussianGaMeSSplatAsset.alphaData, gaussianGaMeSSplatAsset.scaleData, gaussianGaMeSSplatAsset.pointCloudPath);
            _splatRenderer.InjectAsset(newAsset);

        }
    }

    private void RegisterNativeCleanup(Action cleanupAction)
    {
        if (cleanupAction != null) _deferredCleanup.Add(cleanupAction);
    }


    // Update is called once per frame
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


        if (!isCreateAssetJobActive && needsRebuild)
        {
            deformingMesh.SetVertices(displacedVertices);
            // Refresh collider
            if (meshCollider != null)
            {
                meshCollider.sharedMesh = null;          // clear reference
                meshCollider.sharedMesh = deformingMesh; // re-assign
            }

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

            isCreateAssetJobActive = true;
            needsRebuild = false; // reset until something changes next
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
        DisposeIfCreated(ref inputSplatsData);
        DisposeIfCreated(ref runTimeInputSplatsData);

        // Main job outputs & temporaries
        DisposeIfCreated(ref xyzValues);
        DisposeIfCreated(ref rotations);
        DisposeIfCreated(ref scalings);
        DisposeIfCreated(ref faceVertices);

        // Decoded raw data
        DisposeIfCreated(ref decodedAlphasNative);
        DisposeIfCreated(ref decodedScalesNative);


        // Mesh arrays last (they are more fundamental)
        DisposeIfCreated(ref triangles);
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
}
