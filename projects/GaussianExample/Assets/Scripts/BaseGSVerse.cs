using System;
using System.Collections.Generic;
using UnityEngine;
using Unity.Collections;
using Unity.Mathematics;
using Unity.Jobs;
using GaussianSplatting.Runtime;
using GaussianSplatting.Runtime.GaMeS;
using GaussianSplatting.Shared;
using GaussianSplatting.Runtime.Utils;

public class GSBase : MonoBehaviour
{
    public event Action<Vector3[]> OnInitVertices;
    public event Action<float> OnUpdate;
    protected GaussianSplatRenderer _splatRenderer;
    protected GaussianGaMeSSplatAsset _asset;
    protected GaussianSplatRuntimeAssetCreator _creator;

    protected bool _needsRebuild = false;
    public bool needsRebuild
    {
        get => _needsRebuild;
        set => _needsRebuild = value;
    }

    protected float _uniformScale = 1f;
    public float uniformScale => _uniformScale;

    // Decoded data
    protected NativeArray<float3> _decodedAlphasNative;
    protected NativeArray<float> _decodedScalesNative;
    protected Mesh _mesh;
    public Mesh mesh => _mesh;

    // Runtime arrays
    public NativeArray<float3> displacedVertices => _displacedVertices;
    public NativeArray<float3> originalVertices => _originalVertices;
    public NativeArray<float3> GetDisplacedVertices() => _displacedVertices;
    public NativeArray<float3> GetOriginalVertices() => _originalVertices;
    protected NativeArray<float3> _displacedVertices;
    protected NativeArray<float3> _originalVertices;

    protected NativeArray<int> _triangles;
    protected NativeArray<InputSplatData> _inputSplatsData;
    protected NativeArray<InputSplatData> _runTimeInputSplatsData;
    protected NativeArray<float3> _xyzValues;
    protected NativeArray<quaternion> _rotations;
    protected NativeArray<float3> _scalings;
    protected NativeArray<float3> _faceVertices;

    protected JobHandle createAssetJobHandle;
    protected bool isCreateAssetJobActive = false;
    protected int numberPtsPerTriangle = 3;
    protected readonly List<Action> _deferredCleanup = new List<Action>();

    #region Unity lifecycle
    private void Awake()
    {
        InitializeSafely();
    }

    public void UpdateMesh()
    {
        _mesh.SetVertices(_displacedVertices);
    }

    private void Update()
    {
        if (isCreateAssetJobActive)
        {
            if (createAssetJobHandle.IsCompleted)
            {
                createAssetJobHandle.Complete();

                CreateAsset();

                _xyzValues.Dispose();
                _rotations.Dispose();
                _scalings.Dispose();
                _faceVertices.Dispose();

                isCreateAssetJobActive = false;
            }
            return;
        }

        OnUpdate?.Invoke(Time.deltaTime);


        if (!isCreateAssetJobActive && _needsRebuild)
        {

            mesh.SetVertices(_displacedVertices);
            ScheduleAssetRebuild();
            _needsRebuild = false;
        }
    }
    void InitializeRendererAndAsset()
    {

        _splatRenderer = gameObject.GetComponent<GaussianSplatRenderer>();
        if (_splatRenderer == null) throw new InvalidOperationException("GaussianSplatRenderer not found.");

        if (!(_splatRenderer.asset is GaussianGaMeSSplatAsset asset))
            throw new InvalidOperationException("SplatRenderer.asset is not a GaussianGaMeSSplatAsset or is null.");

        _asset = asset;

        numberPtsPerTriangle = _asset.numberOfSplatsPerFace;
        if (numberPtsPerTriangle <= 0) throw new InvalidOperationException("Invalid numberOfSplatsPerFace.");
    }

    protected virtual void ScheduleAssetRebuild()
    {
        _faceVertices = SplatMathUtils.GetMeshFaceVerticesNative(gameObject, _displacedVertices, _triangles, Allocator.Persistent);
        _xyzValues = GaMeSUtils.CreateXYZData(_decodedAlphasNative, _faceVertices, _asset.splatCount / numberPtsPerTriangle, numberPtsPerTriangle);
        (_rotations, _scalings) = GaMeSUtils.CreateScaleRotationData(_faceVertices, _decodedScalesNative, numberPtsPerTriangle);


        var job = new GaMeSUtils.CreateAssetDataJob()
        {
            m_InputPos = _xyzValues,
            m_InputRot = _rotations,
            m_InputScale = _scalings,
            m_Output = _inputSplatsData,
            m_PrevOutput = _runTimeInputSplatsData,

        };
        createAssetJobHandle = job.Schedule(_xyzValues.Length, 8192);

        isCreateAssetJobActive = true;
    }

    protected virtual void OnDestroyCleanup()
    {
        DisposeIfCreated(ref _inputSplatsData);
        DisposeIfCreated(ref _runTimeInputSplatsData);

        // Main job outputs & temporaries
        DisposeIfCreated(ref _xyzValues);
        DisposeIfCreated(ref _rotations);
        DisposeIfCreated(ref _scalings);
        DisposeIfCreated(ref _faceVertices);

        // Decoded raw data
        DisposeIfCreated(ref _decodedAlphasNative);
        DisposeIfCreated(ref _decodedScalesNative);

        // Mesh arrays last (they are more fundamental)
        DisposeIfCreated(ref _triangles);
        DisposeIfCreated(ref _displacedVertices);
        DisposeIfCreated(ref _originalVertices);
    }

    private void OnDestroy()
    {
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

        OnDestroyCleanup();

        RunDeferredCleanup();
    }
    #endregion

    #region Initialization
    public void InitializeSafely()
    {
        InitializeRendererAndAsset();

        DecodeAssetData();

        LoadAndSetupMesh();

        CreateRuntimeBuffers();
        // 7) Create runtime input splats data creator (validate pointCloudPath)
        _creator = new GaussianSplatRuntimeAssetCreator();
        if (string.IsNullOrEmpty(_asset.pointCloudPath))
            throw new InvalidOperationException("pointCloudPath on GaussianGaMeSSplatAsset is null or empty.");

        _runTimeInputSplatsData = _creator.CreateAsset(_asset.pointCloudPath);

        RegisterNativeCleanup(() => { if (_runTimeInputSplatsData.IsCreated) _runTimeInputSplatsData.Dispose(); });

        InitializeFullMode();

        OnInitVertices?.Invoke(_mesh.vertices);
    }

    void DecodeAssetData()
    {
        var alphaBytes = _asset.alphaData.bytes;
        var scaleBytes = _asset.scaleData.bytes;
        int faceCountEstimate = _asset.splatCount / numberPtsPerTriangle;
        if (faceCountEstimate <= 0) throw new InvalidOperationException("Calculated face count is zero.");

        _decodedAlphasNative = GaMeSUtils.DecodeAlphasToNativeFloat3(alphaBytes, faceCountEstimate, numberPtsPerTriangle, Allocator.Persistent);
        RegisterNativeCleanup(() => { if (_decodedAlphasNative.IsCreated) _decodedAlphasNative.Dispose(); });

        _decodedScalesNative = GaMeSUtils.DecodeScalesToNative(scaleBytes, _asset.splatCount, Allocator.Persistent);
        RegisterNativeCleanup(() => { if (_decodedScalesNative.IsCreated) _decodedScalesNative.Dispose(); });
    }

    protected virtual void LoadAndSetupMesh()
    {
        // 4) Load source mesh and attach transformed mesh to our MeshFilter
        var loaded = Resources.Load<GameObject>(_asset.objPath);
        if (loaded == null) throw new InvalidOperationException($"Resources.Load failed for path '{_asset.objPath}'.");

        var child = loaded.transform.childCount > 0 ? loaded.transform.GetChild(0) : null;
        if (child == null) throw new InvalidOperationException("Loaded object has no child to get MeshFilter from.");

        var meshFilter = child.GetComponent<MeshFilter>();
        if (meshFilter == null || meshFilter.sharedMesh == null) throw new InvalidOperationException("Source MeshFilter missing.");

        // Create a unique, runtime-only copy of the mesh
        Mesh meshCopy = Instantiate(meshFilter.sharedMesh);

        // Apply any transforms your utility needs
        meshCopy = GaMeSUtils.TransformMesh(meshCopy, _asset.useMeshLeftHandedCS);

        // Assign it to your new MeshFilter
        MeshFilter addedMeshFilter = gameObject.AddComponent<MeshFilter>();
        addedMeshFilter.mesh = meshCopy;

        _mesh = addedMeshFilter.mesh;

        foreach (var mr in GetComponentsInChildren<MeshRenderer>())
            if (mr != null) mr.enabled = false;

        _uniformScale = transform.localScale.y;
    }

    void CreateRuntimeBuffers()
    {
        var verts = _mesh.vertices;
        var tris = _mesh.triangles;

        if (verts == null || verts.Length == 0)
            throw new InvalidOperationException("Deforming mesh has no vertices.");
        if (tris == null || tris.Length == 0)
            throw new InvalidOperationException("Deforming mesh has no triangles.");

        int vertexCount = verts.Length;
        int triangleCount = tris.Length;

        _originalVertices = new NativeArray<float3>(vertexCount, Allocator.Persistent);
        RegisterNativeCleanup(() => { if (_originalVertices.IsCreated) _originalVertices.Dispose(); });

        _displacedVertices = new NativeArray<float3>(vertexCount, Allocator.Persistent);
        RegisterNativeCleanup(() => { if (_displacedVertices.IsCreated) _displacedVertices.Dispose(); });

        _triangles = new NativeArray<int>(triangleCount, Allocator.Persistent);
        RegisterNativeCleanup(() => { if (_triangles.IsCreated) _triangles.Dispose(); });

        for (int i = 0; i < vertexCount; i++)
        {
            float3 v = verts[i];
            _originalVertices[i] = v;
            _displacedVertices[i] = v;
        }

        for (int i = 0; i < triangleCount; i++)
            _triangles[i] = tris[i];
    }


    protected virtual void InitializeFullMode()
    {

        Transform meshTransform = transform;

        var vertexSet = new HashSet<int>();

        for (int i = 0; i < _triangles.Length; i += 3)
        {
            int i0 = _triangles[i];
            int i1 = _triangles[i + 1];
            int i2 = _triangles[i + 2];

            Vector3 v0 = meshTransform.TransformPoint(_originalVertices[i0]);
            Vector3 v1 = meshTransform.TransformPoint(_originalVertices[i1]);
            Vector3 v2 = meshTransform.TransformPoint(_originalVertices[i2]);

            var v0y = _originalVertices[i0].y;
            var v1y = _originalVertices[i1].y;
            var v2y = _originalVertices[i2].y;

            vertexSet.Add(i0); vertexSet.Add(i1); vertexSet.Add(i2);
        }

        _faceVertices = SplatMathUtils.GetMeshFaceVerticesNative(gameObject, _displacedVertices, _triangles, Allocator.Persistent);
        RegisterNativeCleanup(() => { if (_faceVertices.IsCreated) _faceVertices.Dispose(); });

        _xyzValues = GaMeSUtils.CreateXYZData(_decodedAlphasNative, _faceVertices, _asset.splatCount / numberPtsPerTriangle, numberPtsPerTriangle);
        RegisterNativeCleanup(() => { if (_xyzValues.IsCreated) _xyzValues.Dispose(); });

        (_rotations, _scalings) = GaMeSUtils.CreateScaleRotationData(_faceVertices, _decodedScalesNative, numberPtsPerTriangle);
        RegisterNativeCleanup(() => { if (_rotations.IsCreated) _rotations.Dispose(); if (_scalings.IsCreated) _scalings.Dispose(); });

        _inputSplatsData = new NativeArray<InputSplatData>(_splatRenderer.asset.splatCount, Allocator.Persistent);
        RegisterNativeCleanup(() => { if (_inputSplatsData.IsCreated) _inputSplatsData.Dispose(); });

        var job = new GaMeSUtils.CreateAssetDataJob()
        {
            m_InputPos = _xyzValues,
            m_InputRot = _rotations,
            m_InputScale = _scalings,
            m_PrevOutput = _runTimeInputSplatsData,
            m_Output = _inputSplatsData
        };

        createAssetJobHandle = job.Schedule(_xyzValues.Length, 8192);
        createAssetJobHandle.Complete();

        CreateAsset();
    }

    protected unsafe void CreateAsset()
    {
        if (_creator != null && _asset)
        {
            var newAsset = _creator.CreateAsset("new asset", _inputSplatsData, _asset.alphaData, _asset.scaleData, _asset.pointCloudPath);
            _splatRenderer.InjectAsset(newAsset);

        }
    }
    #endregion

    #region Helpers & cleanup

    protected void RegisterNativeCleanup(Action cleanupAction)
    {
        if (cleanupAction != null) _deferredCleanup.Add(cleanupAction);
    }

    protected void RunDeferredCleanup()
    {
        for (int i = _deferredCleanup.Count - 1; i >= 0; --i)
        {
            try { _deferredCleanup[i]?.Invoke(); }
            catch (Exception e) { Debug.LogWarning($"Cleanup action failed: {e.Message}"); }
        }
        _deferredCleanup.Clear();
    }

    protected void DisposeIfCreated<T>(ref NativeArray<T> array) where T : struct
    {
        try
        {
            if (array.IsCreated) array.Dispose();
        }
        catch (Exception e)
        {
            Debug.LogWarning($"Error disposing NativeArray<{typeof(T).Name}>: {e.Message}");
        }
        finally { array = default; }
    }
    #endregion
}

