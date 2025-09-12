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

public class SplatPressDeformate : MonoBehaviour, IDeformable
{
    public bool blenderMesh = false;
    public float springForce = 10f;
    public float damping = 2f;

    private GaussianSplatRenderer _splatRenderer;
    private GaussianGaMeSSplatAsset gaussianGaMeSSplatAsset = null;
    private GaussianSplatRuntimeAssetCreator creator = null;
    private int numberPtsPerTriangle = 3;
    public float maxDeformation = 10.0f;

    NativeArray<float3> decodedAlphasNative;
    NativeArray<float> decodedScalesNative;
    NativeArray<float3> originalVertices;
    NativeArray<float3> displacedVertices;
    NativeArray<float3> vertexVelocities;
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
    // Start is called before the first frame update

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
            vertices[i] = new Vector3(-vertices[i].x, vertices[i].y, -vertices[i].z);
        }
        mesh.vertices = vertices;

        // Flip triangle winding (to keep normals outward)
        int[] triangles = mesh.triangles;
        for (int i = 0; i < triangles.Length; i += 3)
        {

            var v1 = triangles[i + 1];
            var v2 = triangles[i + 2];
            triangles[i + 1] = v2;
            triangles[i + 2] = v1;
        }
        mesh.triangles = triangles;

        // Recalculate normals & bounds
        // mesh.RecalculateNormals();
        //mesh.RecalculateBounds();
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
            addedMeshFilter.mesh = GaMeSUtils.TransformMesh(Instantiate(meshFilter.sharedMesh));
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
        //MirrorAlongY(deformingMesh);

        // Add MeshCollider with deformingMesh



        meshCollider.sharedMesh = deformingMesh;
        meshCollider.convex = true;



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

        InitializeFullMode();


        // Clear deferred cleanup actions because initialization succeeded.
        _deferredCleanup.Clear();
    }

    private void InitializeFullMode()
    {

        Transform meshTransform = transform;

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

    public void AddDeformingForce(Vector3 point, Vector3 force)
    {
        return;
    }

    [BurstCompile]
    public struct VertexSpringJob : IJobParallelFor
    {
        public float deltaTime;
        public float springForce;
        public float damping;
        public float uniformScale;

        [NativeDisableParallelForRestriction] public NativeArray<float3> displacedVertices;
        [NativeDisableParallelForRestriction] public NativeArray<float3> originalVertices;
        [NativeDisableParallelForRestriction] public NativeArray<float3> vertexVelocities;

        public void Execute(int index)
        {
            int i = index;
            float3 v = vertexVelocities[i];

            // displacement in local space
            float3 disp = (displacedVertices[i] - originalVertices[i]) * uniformScale;

            // apply spring only (scaled by weight)
            v -= disp * springForce * deltaTime;

            // DAMPING SHOULD NOT BE SCALED BY WEIGHT (or make it stronger near bottom)
            v *= 1f - damping * deltaTime;

            // integrate position **scaled by weight** so bottom moves less

            displacedVertices[i] += v * (deltaTime / uniformScale);


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

    public void SetClickState(bool clicking)
    {
        // throw new NotImplementedException();
    }



    public void AddPressForce(Vector3 worldPoint, Vector3 pressNormal, float maxDeform, float radius, float damageFalloff)
    {
        Vector3 localPoint = transform.InverseTransformPoint(worldPoint);
        Vector3 localNormal = transform.InverseTransformDirection(pressNormal);
        Debug.Log("maxDeform");
        Debug.Log(maxDeform);
        Debug.Log("radius");
        Debug.Log(radius);
        Debug.Log("damageFallof");
        Debug.Log(damageFalloff);

        if (IsMaxDeformed(maxDeform)) return;

        var job = new PressForceJob
        {
            pressPoint = localPoint,
            radius = radius,
            intensity = 0.15f,
            falloff = damageFalloff,
            displacedVertices = displacedVertices,
            originalVertices = originalVertices,
            uniformScale = uniformScale,
            pressNormal = localNormal

        };
        JobHandle handle = job.Schedule(displacedVertices.Length, 64);
        handle.Complete();
        needsRebuild = true;
        /*
                var job = new AddPressForceJob
                {
                    pointLocal = localPoint,
                    uniformScale = uniformScale,
                    deltaTime = Time.deltaTime,
                    deformRadius = 0.95f,
                    maxDeform = 0.51f,
                    damageFalloff = 0.45f,
                    damageMultiplier = 0.25f,
                    displacedVertices = displacedVertices,
                    originalVertices = originalVertices
                };
                JobHandle handle = job.Schedule(displacedVertices.Length, 64);
                handle.Complete();
                needsRebuild = true;
        */
    }

    public bool IsMaxDeformed(float maxDeform)
    {
        // compute maximum deviation from original vertices
        float maxDeviation = 0f;
        for (int i = 0; i < displacedVertices.Length; i++)
        {
            float d = Vector3.Distance(displacedVertices[i], originalVertices[i]);
            if (d * uniformScale > maxDeviation) maxDeviation = d;
        }

        return maxDeviation >= maxDeform;
    }



    [BurstCompile]
    struct PressForceJob : IJobParallelFor
    {
        [ReadOnly] public Vector3 pressPoint;
        [ReadOnly] public float radius;
        [ReadOnly] public float intensity;
        [ReadOnly] public float falloff;
        [ReadOnly] public float uniformScale;
        public NativeArray<float3> displacedVertices;
        public NativeArray<float3> originalVertices;
        [ReadOnly] public Vector3 pressNormal;



        public void Execute(int index)
        {

            Vector3 vertex = displacedVertices[index];
            float distance = Vector3.Distance(vertex, pressPoint);


            if (distance * uniformScale < radius)
            {
                float falloffFactor = math.pow(1 - (distance / radius), falloff);
                float displacement = -intensity * falloffFactor;

                // NOTE: we use local "up" (Y axis). You can replace with normal direction if needed.
                vertex += pressNormal * displacement;
                //vertex += new Vector3(0f, 0f, displacement);
            }


            displacedVertices[index] = vertex;



        }
    }

    [BurstCompile]
    struct AddPressForceJob : IJobParallelFor
    {

        [NativeDisableParallelForRestriction] public NativeArray<float3> displacedVertices;
        [NativeDisableParallelForRestriction] public NativeArray<float3> originalVertices;

        [ReadOnly] public Vector3 pointLocal;
        [ReadOnly] public float uniformScale;
        [ReadOnly] public float deltaTime;
        [ReadOnly] public float deformRadius;
        [ReadOnly] public float maxDeform;
        [ReadOnly] public float damageFalloff;
        [ReadOnly] public float damageMultiplier;



        public void Execute(int index)
        {
            int i = index;
            Vector3 distanceFromCollision = displacedVertices[i] - (float3)pointLocal;
            Vector3 distanceFromOriginal = originalVertices[i] - displacedVertices[i];
            distanceFromCollision *= uniformScale;
            distanceFromOriginal *= uniformScale;

            float distFromCollision = distanceFromCollision.magnitude;
            float distFromOrigin = distanceFromOriginal.magnitude;
            if (distFromCollision < deformRadius && distFromOrigin < maxDeform)
            {
                // Smooth falloff
                float falloff = 1 - (distFromCollision / deformRadius) * damageFalloff;

                float xDeform = pointLocal.x * falloff;
                float yDeform = pointLocal.y * falloff;
                float zDeform = pointLocal.z * falloff;

                xDeform = Mathf.Clamp(xDeform, 0, maxDeform);
                yDeform = Mathf.Clamp(yDeform, 0, maxDeform);
                zDeform = Mathf.Clamp(zDeform, 0, maxDeform);

                //vertexVelocities[i] += new float3(xDeform, yDeform, zDeform) * deltaTime;
                float3 deform = new float3(xDeform, yDeform, zDeform) * damageMultiplier;
                displacedVertices[i] -= deform;

            }

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


}
