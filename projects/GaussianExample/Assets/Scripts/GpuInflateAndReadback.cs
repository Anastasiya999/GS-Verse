using System;
using System.Collections.Generic;
using GaussianSplatting.Runtime;
using GaussianSplatting.Runtime.GaMeS;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.Rendering;

//[RequireComponent(typeof(MeshFilter))]
public class GpuInflateAndReadback : MonoBehaviour
{
    // Update is called once per frame[Header("XR Input Action (Trigger)")]
    public InputActionReference triggerAction;
    [Header("Shader")]
    public ComputeShader inflateCompute;

    [Header("Anchor")]
    public Transform anchor;
    public float anchorRadius = 0.1f; // in local space units (same as mesh vertices)

    private GaussianSplatRenderer _splatRenderer;

    [Range(-1f, 2f)] public float inflateAmount = 0.1f;
    public float inflateSpeed = 0.3f;
    public float maxInflate = 1.5f;

    private GaussianSplatAssetUpdater gaussianSplatAssetUpdater;

    private Mesh workingMesh;
    private NativeArray<int> triangles;

    private ComputeBuffer basePosBuffer, normalBuffer, outPosBuffer;
    private ComputeBuffer faceMaskBuffer;
    private float[] faceMask;
    private int kernel, vertexCount;
    private uint threadGroupSizeX = 256;
    private bool isInflating = false;

    private ConstantForce gravityForce;

    private Vector3[] basePositionsCPU;
    private Vector3[] normalsCPU;
    public List<Transform> anchorPoints;  // fill in inspector
    private GaussianSplatRuntimeAsset _lastInjectedAsset;
    private readonly List<Action> _deferredCleanup = new List<Action>();
    public static void MirrorAlongY(Mesh mesh)
    {
        // Transform vertices
        Vector3[] vertices = mesh.vertices;
        for (int i = 0; i < vertices.Length; i++)
        {
            vertices[i] = new Vector3(-vertices[i].x, vertices[i].y, vertices[i].z);
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
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();
    }

    void UpdateFaceMask(Vector3 localAnchor)
    {
        if (faceMask == null || faceMask.Length != vertexCount)
            faceMask = new float[vertexCount];
        else
            Array.Clear(faceMask, 0, faceMask.Length);

        for (int t = 0; t < triangles.Length; t += 3)
        {
            int i0 = triangles[t];
            int i1 = triangles[t + 1];
            int i2 = triangles[t + 2];

            Vector3 c = (basePositionsCPU[i0] + basePositionsCPU[i1] + basePositionsCPU[i2]) / 3f;

            float mask = c.z >= localAnchor.z ? 1f : 0f;

            // assign mask to all vertices of that face
            faceMask[i0] = Mathf.Max(faceMask[i0], mask);
            faceMask[i1] = Mathf.Max(faceMask[i1], mask);
            faceMask[i2] = Mathf.Max(faceMask[i2], mask);
        }

        if (faceMaskBuffer == null || faceMaskBuffer.count != vertexCount)
        {
            faceMaskBuffer?.Dispose();
            faceMaskBuffer = new ComputeBuffer(vertexCount, sizeof(float));
            inflateCompute.SetBuffer(kernel, "_FaceMask", faceMaskBuffer);
        }

        faceMaskBuffer.SetData(faceMask);
    }




    void Awake()
    {

        gravityForce = GetComponentInParent<ConstantForce>();
        gravityForce.enabled = false;

        _splatRenderer = GetComponent<GaussianSplatRenderer>();

        if (_splatRenderer == null)
            throw new InvalidOperationException("SplatRenderer with tag 'SplatRenderer' not found or missing GaussianSplatRenderer component.");

        if (!(_splatRenderer.asset is GaussianGaMeSSplatAsset asset))
            throw new InvalidOperationException("SplatRenderer.asset is not a GaussianGaMeSSplatAsset or is null.");

        gaussianSplatAssetUpdater = new GaussianSplatAssetUpdater(asset);
        var loaded = Resources.Load<GameObject>(asset.objPath);
        if (loaded == null)
            throw new InvalidOperationException($"Resources.Load failed for path '{asset.objPath}'.");

        var child = loaded.transform.childCount > 0 ? loaded.transform.GetChild(0) : null;
        if (child == null)
            throw new InvalidOperationException("Loaded object from Resources has no children to get MeshFilter from.");

        var meshFilter = child.GetComponent<MeshFilter>();
        if (meshFilter == null || meshFilter.sharedMesh == null)
            throw new InvalidOperationException("Child GameObject does not contain a MeshFilter with a sharedMesh.");

        MeshFilter addedMeshFilter = gameObject.AddComponent<MeshFilter>();
        addedMeshFilter.mesh = Instantiate(meshFilter.sharedMesh);
        var mf = GetComponent<MeshFilter>();

        if (mf.sharedMesh == null)
        {
            Debug.LogError("MeshFilter has no mesh.");
            enabled = false;
            return;
        }

        workingMesh = Instantiate(mf.sharedMesh);
        GaMeSUtils.TransformMesh(workingMesh);

        if (!workingMesh.isReadable)
        {
            Debug.LogError("Mesh must be readable (Import Settings > Read/Write Enabled).");
            enabled = false;
            return;
        }

        basePositionsCPU = workingMesh.vertices;
        normalsCPU = workingMesh.normals;
        vertexCount = workingMesh.vertexCount;

        basePosBuffer = new ComputeBuffer(vertexCount, sizeof(float) * 3);
        normalBuffer = new ComputeBuffer(vertexCount, sizeof(float) * 3);
        outPosBuffer = new ComputeBuffer(vertexCount, sizeof(float) * 3);

        basePosBuffer.SetData(basePositionsCPU);
        normalBuffer.SetData(normalsCPU);

        kernel = inflateCompute.FindKernel("CSMain");

        inflateCompute.SetInt("_VertexCount", vertexCount);
        inflateCompute.SetBuffer(kernel, "_BasePositions", basePosBuffer);
        inflateCompute.SetBuffer(kernel, "_Normals", normalBuffer);
        inflateCompute.SetBuffer(kernel, "_OutPositions", outPosBuffer);

        // find minY / maxY from original mesh
        float minY = float.MaxValue, maxY = float.MinValue;
        foreach (var v in basePositionsCPU)
        {
            if (v.y < minY) minY = v.y;
            if (v.y > maxY) maxY = v.y;
        }

        Vector3 localAnchor = transform.InverseTransformPoint(anchor.position);

        inflateCompute.SetFloat("_MinY", anchor.position.y);
        inflateCompute.SetFloat("_MaxY", maxY);

        var tris = workingMesh.triangles;

        if (tris == null || tris.Length == 0)
            throw new InvalidOperationException("Deforming mesh has no triangles.");

        int triangleCount = tris.Length;

        triangles = new NativeArray<int>(triangleCount, Allocator.Persistent);
        RegisterNativeCleanup(() => { if (triangles.IsCreated) triangles.Dispose(); });


        for (int i = 0; i < triangleCount; i++)
            triangles[i] = tris[i];



        // build mask once
        UpdateFaceMask(localAnchor);
    }

    private void RegisterNativeCleanup(Action cleanupAction)
    {
        if (cleanupAction != null) _deferredCleanup.Add(cleanupAction);
    }

    void Update()
    {
        float triggerValue = triggerAction.action.ReadValue<float>();

        if (triggerValue > 0.1f) // if trigger pressed
        {
            isInflating = true;
        }
        else
        {
            isInflating = false;
        }

        if (isInflating && inflateAmount != maxInflate)
        {
            inflateAmount = Mathf.Min(maxInflate, inflateAmount + inflateSpeed * Time.deltaTime);

            inflateCompute.SetFloat("_InflateAmount", inflateAmount);
            int groups = Mathf.CeilToInt(vertexCount / (float)threadGroupSizeX);
            inflateCompute.Dispatch(kernel, groups, 1, 1);

            // Read back once, apply to both meshes
            AsyncGPUReadback.Request(outPosBuffer, req =>
        {
            if (req.hasError) return;

            if (this == null || !this.isActiveAndEnabled)
            {

                return;
            }

            NativeArray<float3> newPositions = req.GetData<float3>();

            //Generate splats based on new Positions
            //update asset
            UpdateGaMeSAsset(newPositions, triangles);

            // Check meshes are still valid
            if (workingMesh != null)
            {
                workingMesh.SetVertices(newPositions);
                workingMesh.RecalculateNormals();
                workingMesh.RecalculateBounds();
            }


        });

        }
        else
        {
            if (inflateAmount == maxInflate && !gravityForce.enabled)
            {
                Debug.Log(inflateAmount == maxInflate);

                gravityForce.enabled = true;
                MeshCollider addedCollider = gameObject.GetComponent<MeshCollider>();
                if (addedCollider == null)
                    addedCollider = gameObject.AddComponent<MeshCollider>();

                addedCollider.sharedMesh = workingMesh;
                addedCollider.convex = true;
            }
        }

    }

    void UpdateGaMeSAsset(NativeArray<float3> displacedVertices, NativeArray<int> triangles)
    {
        var newAsset = gaussianSplatAssetUpdater.CreateAsset(gameObject, displacedVertices, triangles);
        if (newAsset != null)
        {
            _lastInjectedAsset?.Dispose(); // free old asset's NativeArrays

            _splatRenderer.InjectAsset(newAsset);
            _lastInjectedAsset = newAsset;
        }

    }


    void OnDestroy()
    {
        gaussianSplatAssetUpdater?.Dispose();
        _lastInjectedAsset?.Dispose();
        // dispose GPU buffers (these are IDisposable for compute GPU resources)
        try
        {
            basePosBuffer?.Dispose();
        }
        catch (Exception e) { Debug.LogWarning("Error disposing basePosBuffer: " + e); }

        try
        {
            normalBuffer?.Dispose();
        }
        catch (Exception e) { Debug.LogWarning("Error disposing normalBuffer: " + e); }

        try
        {
            outPosBuffer?.Dispose();
        }
        catch (Exception e) { Debug.LogWarning("Error disposing outPosBuffer: " + e); }

        try
        {
            faceMaskBuffer?.Dispose();
        }
        catch (Exception e) { Debug.LogWarning("Error disposing faceMaskBuffer: " + e); }

        // Run any deferred cleanup (e.g. NativeArray disposal)
        foreach (var action in _deferredCleanup)
        {
            try
            {
                action?.Invoke();
            }
            catch (Exception e)
            {
                Debug.LogWarning("Deferred cleanup action threw: " + e);
            }
        }
        _deferredCleanup.Clear();
        gaussianSplatAssetUpdater?.Dispose();

        // If a MeshCollider or other component holds the mesh, clear the reference so Unity can GC the mesh
        var mc = GetComponent<MeshCollider>();
        if (mc != null)
        {
            mc.sharedMesh = null;
        }

        // Destroy any instantiated Mesh (workingMesh). Must use DestroyImmediate if in editor and during edit-time
        if (workingMesh != null)
        {
#if UNITY_EDITOR
            UnityEngine.Object.DestroyImmediate(workingMesh);
#else
        UnityEngine.Object.Destroy(workingMesh);
#endif
            workingMesh = null;
        }

        // Also destroy the Mesh on the MeshFilter we added — if it's the same as workingMesh it was already destroyed,
        // but if not, clear and destroy it to be safe:
        var mf = GetComponent<MeshFilter>();
        if (mf != null)
        {
            if (mf.sharedMesh != null)
            {
                // If it's a different mesh instance destroy it
                if (mf.sharedMesh != workingMesh)
                {
#if UNITY_EDITOR
                    UnityEngine.Object.DestroyImmediate(mf.sharedMesh);
#else
                UnityEngine.Object.Destroy(mf.sharedMesh);
#endif
                }
                mf.sharedMesh = null;
            }
        }

    }
}
