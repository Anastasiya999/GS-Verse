using System;
using System.Collections;
using System.Collections.Generic;
using GaussianSplatting.Runtime;
using GaussianSplatting.Runtime.GaMeS;
using GaussianSplatting.Runtime.Utils;
using GaussianSplatting.Shared;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

public class GaussianSplatAssetUpdater : IDisposable
{
    private GaussianGaMeSSplatAsset _asset;
    private GaussianSplatRuntimeAssetCreator _creator;

    private int _numberPtsPerTriangle;

    // Native arrays used for computation and job data
    private NativeArray<float3> _decodedAlphasNative;
    private NativeArray<float> _decodedScalesNative;

    private NativeArray<float3> _faceVertices;
    private NativeArray<float3> _xyzValues;
    private NativeArray<quaternion> _rotations;
    private NativeArray<float3> _scalings;

    private NativeArray<InputSplatData> _inputSplatsData;
    private NativeArray<InputSplatData> _runTimeInputSplatsData;
    private JobHandle _createAssetJobHandle;
    private bool _jobActive;

    private List<Action> _deferredCleanup = new List<Action>();
    public void Dispose()
    {
        foreach (var cleanup in _deferredCleanup)
        {
            try
            {
                cleanup?.Invoke();
            }
            catch (Exception e)
            {
                Debug.LogWarning($"Deferred cleanup threw an exception: {e}");
            }
        }
        _deferredCleanup.Clear();
    }

    public GaussianSplatAssetUpdater(GaussianGaMeSSplatAsset asset)
    {
        _asset = asset ?? throw new ArgumentNullException(nameof(asset));
        _creator = new GaussianSplatRuntimeAssetCreator();

        _numberPtsPerTriangle = _asset.numberOfSplatsPerFace;
        if (_numberPtsPerTriangle <= 0)
            throw new InvalidOperationException("numberOfSplatsPerFace must be greater than 0");

        Initialize();
    }

    private void Initialize()
    {
        // Decode raw asset data to native arrays
        int faceCountEstimate = _asset.splatCount / _numberPtsPerTriangle;

        _decodedAlphasNative = GaMeSUtils.DecodeAlphasToNativeFloat3(_asset.alphaData.bytes, faceCountEstimate, _numberPtsPerTriangle, Allocator.Persistent);
        RegisterCleanup(() => { if (_decodedAlphasNative.IsCreated) _decodedAlphasNative.Dispose(); });

        _decodedScalesNative = GaMeSUtils.DecodeScalesToNative(_asset.scaleData.bytes, _asset.splatCount, Allocator.Persistent);
        RegisterCleanup(() => { if (_decodedScalesNative.IsCreated) _decodedScalesNative.Dispose(); });

        // Initialize previous splats data as empty (will be set later)
        _runTimeInputSplatsData = _creator.CreateAsset(_asset.pointCloudPath);
        // Register only if creator returned a NativeArray or similar; we assume it's a NativeArray<InputSplatData>
        RegisterCleanup(() => { if (_runTimeInputSplatsData.IsCreated) _runTimeInputSplatsData.Dispose(); });
    }

    public GaussianSplatRuntimeAsset CreateAsset(GameObject meshGameObject, NativeArray<float3> displacedVertices, NativeArray<int> triangles)
    {

        try
        {
            _faceVertices = SplatMathUtils.GetMeshFaceVerticesNative(meshGameObject, displacedVertices, triangles, Allocator.Persistent);
            RegisterCleanup(() => { if (_faceVertices.IsCreated) _faceVertices.Dispose(); });

            _xyzValues = GaMeSUtils.CreateXYZData(_decodedAlphasNative, _faceVertices, _asset.splatCount / _numberPtsPerTriangle, _numberPtsPerTriangle);
            RegisterCleanup(() => { if (_xyzValues.IsCreated) _xyzValues.Dispose(); });

            (_rotations, _scalings) = GaMeSUtils.CreateScaleRotationData(_faceVertices, _decodedScalesNative, _numberPtsPerTriangle);
            RegisterCleanup(() => { if (_rotations.IsCreated) _rotations.Dispose(); if (_scalings.IsCreated) _scalings.Dispose(); });

            _inputSplatsData = new NativeArray<InputSplatData>(_asset.splatCount, Allocator.Persistent);
            RegisterCleanup(() => { if (_inputSplatsData.IsCreated) _inputSplatsData.Dispose(); });

            var job = new GaMeSUtils.CreateAssetDataJob()
            {
                m_InputPos = _xyzValues,
                m_InputRot = _rotations,
                m_InputScale = _scalings,
                m_PrevOutput = _runTimeInputSplatsData,
                m_Output = _inputSplatsData
            };

            _createAssetJobHandle = job.Schedule(_xyzValues.Length, 8192);
            _createAssetJobHandle.Complete();

            var newAsset = _creator.CreateAsset("new asset", _inputSplatsData, _asset.alphaData, _asset.scaleData, _asset.pointCloudPath);
            // Dispose temporary arrays immediately
            if (_faceVertices.IsCreated) _faceVertices.Dispose();
            if (_xyzValues.IsCreated) _xyzValues.Dispose();
            if (_rotations.IsCreated) _rotations.Dispose();
            if (_scalings.IsCreated) _scalings.Dispose();
            if (_inputSplatsData.IsCreated) _inputSplatsData.Dispose();
            return newAsset;
        }
        catch
        {
            // Dispose temporary arrays if anything fails
            if (_faceVertices.IsCreated) _faceVertices.Dispose();
            if (_xyzValues.IsCreated) _xyzValues.Dispose();
            if (_rotations.IsCreated) _rotations.Dispose();
            if (_scalings.IsCreated) _scalings.Dispose();
            if (_inputSplatsData.IsCreated) _inputSplatsData.Dispose();
            throw;
        }

    }

    private void RegisterCleanup(Action cleanupAction)
    {
        if (cleanupAction != null) _deferredCleanup.Add(cleanupAction);
    }

}
