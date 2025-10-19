using System.Collections.Generic;
using UnityEngine;
using GaussianSplatting.Shared;
using Unity.Mathematics;
using Unity.Collections;
using System;
using Unity.Jobs;
using Unity.Burst;

public class SplatSegmentedDeformate : MonoBehaviour, IDeformable
{
    public float springForce = 20f;
    public float dragStrength = 100.0f;
    float uniformScale = 1f;
    public float damping = 5f;
    NativeArray<float3> vertexVelocities;

    private readonly List<Action> _deferredCleanup = new List<Action>();

    [SerializeField] private float returnFinishEpsilon = 1e-4f;

    [SerializeField] private GSVerseSegmented _gsVerse;


    void Awake()
    {

        try
        {
            if (_gsVerse == null)
                _gsVerse = GetComponent<GSVerseSegmented>();

            if (_gsVerse != null)
            {
                _gsVerse.OnInitVertices += Initialize;
                _gsVerse.OnUpdate += UpdateMesh;
            }
            else
            {
                Debug.LogError("GSVerse not found!");
            }

        }
        catch (Exception ex)
        {
            RunDeferredCleanup();
            throw ex;
        }

    }

    void UpdateMesh(float deltaTime)
    {

        if (ForceModeManager.Instance.CurrentForceMode == ForceMode.Drag)
        {
            if (_gsVerse == null) return;

            var displacedVertices = _gsVerse.GetDisplacedVertices();
            var originalVertices = _gsVerse.originalVertices;
            var selectedVertexWeights = _gsVerse.selectedVertexWeights;
            var selectedVertexIndices = _gsVerse.selectedVertexIndices;
            var maxPointLocal = _gsVerse.maxPointLocal;
            var minPointLocal = _gsVerse.minPointLocal;

            if (!displacedVertices.IsCreated || !originalVertices.IsCreated || !vertexVelocities.IsCreated)
                return;


            var springJob = new VertexSpringJobSelected
            {
                deltaTime = deltaTime,
                springForce = springForce,
                damping = damping,
                uniformScale = _gsVerse.uniformScale,
                displacedVertices = displacedVertices,
                originalVertices = originalVertices,
                vertexVelocities = vertexVelocities,
                selectedVertexWeights = selectedVertexWeights,
                selectedVertexIndices = selectedVertexIndices,
                topY = maxPointLocal.y,
                bottomY = minPointLocal.y
            };
            JobHandle handle = springJob.Schedule(selectedVertexIndices.Length, 64);
            handle.Complete();
        }


        if (ForceModeManager.Instance.CurrentForceMode == ForceMode.None)
        {
            _gsVerse.needsRebuild = ReturnToOriginalShapeSelected();
        }

    }


    public void Initialize(Vector3[] vertices)
    {
        int vertexCount = vertices.Length;

        if (vertexVelocities.IsCreated)
        {
            try { vertexVelocities.Dispose(); } catch (Exception e) { Debug.LogWarning($"Error disposing previous vertexVelocities: {e.Message}"); }
            vertexVelocities = default;
        }

        vertexVelocities = new NativeArray<float3>(vertexCount, Allocator.Persistent);
        RegisterNativeCleanup(() => { if (vertexVelocities.IsCreated) vertexVelocities.Dispose(); });

        for (int i = 0; i < vertexCount; i++)
        {
            vertexVelocities[i] = new float3(0, 0, 0);
        }
    }

    private void OnDestroy()
    {
        // Unsubscribe events to avoid dangling references / callbacks after destroy
        if (_gsVerse != null)
        {
            _gsVerse.OnInitVertices -= Initialize;
            _gsVerse.OnUpdate -= UpdateMesh;
        }

        // Use single cleanup path — run deferred cleanups (they will dispose vertexVelocities)
        RunDeferredCleanup();

        // Safety: ensure vertexVelocities is disposed even if it wasn't registered (defensive)
        if (vertexVelocities.IsCreated)
        {
            try { vertexVelocities.Dispose(); } catch (Exception e) { Debug.LogWarning($"Error disposing vertexVelocities in OnDestroy: {e.Message}"); }
            vertexVelocities = default;
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

    bool ReturnToOriginalShapeSelected()
    {
        var displaced = _gsVerse.GetDisplacedVertices();
        var original = _gsVerse.GetOriginalVertices();
        var selectedVertexIndices = _gsVerse.selectedVertexIndices;
        float maxDispSq = 0f;
        for (int i = 0; i < selectedVertexIndices.Length; i++)
        {
            int index = selectedVertexIndices[i];
            float3 diff = displaced[index] - original[index];
            float dsq = math.lengthsq(diff);
            if (dsq > maxDispSq) maxDispSq = dsq;
            displaced[index] = Vector3.Lerp(displaced[index], original[index], Time.deltaTime * 5f);
        }

        return maxDispSq > returnFinishEpsilon * returnFinishEpsilon;
    }


    [BurstCompile]
    public struct VertexSpringJobSelected : IJobParallelFor
    {
        public float deltaTime;
        public float springForce;
        public float damping;
        public float uniformScale;

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
                float attenuation = 1f / (1f + pointToVertex.sqrMagnitude);
                Vector3 appliedForce = force * attenuation * deltaTime;
                vertexVelocities[i] += (float3)appliedForce;
            }
        }
    }



    public void AddDeformingForce(Vector3 point, Vector3 force)
    {
        Vector3 pointLocal = transform.InverseTransformPoint(point);
        Vector3 forceLocal = transform.InverseTransformDirection(force);

        var displacedVertices = _gsVerse.displacedVertices;
        var originalVertices = _gsVerse.originalVertices;
        var selectedVertexIndices = _gsVerse.selectedVertexIndices;

        if (!displacedVertices.IsCreated || !originalVertices.IsCreated || !vertexVelocities.IsCreated)
        {
            return;
        }


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
        _gsVerse.needsRebuild = true;

    }

    public void AddPressForce(Vector3 point, Vector3 pressNormal)
    {


    }


}
