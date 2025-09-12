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

public class SplatDeformate : MonoBehaviour, IDeformable
{

    private GaussianSplatRenderer _splatRenderer;

    Mesh deformingMesh;
    public float springForce = 20f;
    public bool blenderMesh = false;
    float uniformScale = 1f;
    public float damping = 5f;


    private int numberPtsPerTriangle = 3;

    public GameObject boundingBoxObject;

    private bool isPressed = false;


    private NativeArray<int> selectedVertexIndices;

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

    private bool isCreateAssetJobActive = false;


    private GaussianSplatRuntimeAssetCreator creator = null;
    private GaussianGaMeSSplatAsset gaussianGaMeSSplatAsset = null;

    bool IsSelectionMode() => boundingBoxObject != null;

    // Keep track of cleanup actions to run if initialization fails halfway.
    private readonly List<Action> _deferredCleanup = new List<Action>();

    private bool needsRebuild = true;
    [SerializeField] private float returnFinishEpsilon = 1e-4f;

    void Awake()
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

        // Add MeshCollider with deformingMesh
        MeshCollider addedCollider = gameObject.GetComponent<MeshCollider>();
        if (addedCollider == null)
            addedCollider = gameObject.AddComponent<MeshCollider>();

        addedCollider.sharedMesh = deformingMesh;
        addedCollider.convex = true;

        addedCollider.enabled = false;
        addedCollider.enabled = true;


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

        var vertexSet = new HashSet<int>();

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
        else
        {
            needsRebuild = ReturnToOriginalShape();
        }



        if (!isCreateAssetJobActive && needsRebuild)
        {

            //TODO: check how we can optimaze it
            deformingMesh.SetVertices(displacedVertices);

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

        // Selection arrays
        DisposeIfCreated(ref selectedVertexIndices);
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




    public void AddDeformingForce(Vector3 point, Vector3 force)
    {


        Vector3 pointLocal = transform.InverseTransformPoint(point);
        Vector3 forceLocal = transform.InverseTransformDirection(force);



        var job = new AddDeformingForceJob
        {
            displacedVertices = displacedVertices,
            vertexVelocities = vertexVelocities,
            pointLocal = pointLocal,
            force = forceLocal,
            uniformScale = uniformScale,
            deltaTime = Time.deltaTime
        };

        JobHandle handle = job.Schedule(displacedVertices.Length, 64);
        handle.Complete();


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



}
