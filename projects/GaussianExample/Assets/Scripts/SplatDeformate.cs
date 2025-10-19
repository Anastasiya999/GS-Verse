using System.Collections.Generic;
using UnityEngine;
using GaussianSplatting.Shared;
using Unity.Mathematics;
using Unity.Collections;
using System;
using Unity.Jobs;
using Unity.Burst;


public class SplatDeformate : MonoBehaviour, IDeformable
{
    [SerializeField] private GSVerse _gsVerse;
    public float springForce = 20f;
    public float damping = 5f;

    NativeArray<float3> vertexVelocities;

    private readonly List<Action> _deferredCleanup = new List<Action>();

    [SerializeField] private float returnFinishEpsilon = 1e-4f;

    void Awake()
    {

        try
        {
            if (_gsVerse == null)
                _gsVerse = GetComponent<GSVerse>();

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
            throw;
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
    void UpdateMesh(float deltaTime)
    {

        if (ForceModeManager.Instance.CurrentForceMode == ForceMode.Drag)
        {
            if (_gsVerse == null) return;

            var displacedVertices = _gsVerse.GetDisplacedVertices();
            var originalVertices = _gsVerse.originalVertices;

            if (!displacedVertices.IsCreated || !originalVertices.IsCreated || !vertexVelocities.IsCreated)
                return;


            var springJob = new VertexSpringJob
            {
                deltaTime = Time.deltaTime,
                springForce = springForce,
                damping = damping,
                uniformScale = _gsVerse.uniformScale,
                displacedVertices = displacedVertices,
                originalVertices = originalVertices,
                vertexVelocities = vertexVelocities,


            };
            JobHandle handle = springJob.Schedule(displacedVertices.Length, 64);
            handle.Complete();
        }
        else
        {

            _gsVerse.needsRebuild = ReturnToOriginalShape();
            if (_gsVerse.needsRebuild) _gsVerse.UpdateMesh();
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

    bool ReturnToOriginalShape()
    {
        var displaced = _gsVerse.GetDisplacedVertices();
        var original = _gsVerse.GetOriginalVertices();

        float maxDispSq = 0f;
        for (int i = 0; i < _gsVerse.displacedVertices.Length; i++)
        {
            float3 diff = displaced[i] - original[i];
            float dsq = math.lengthsq(diff);
            if (dsq > maxDispSq) maxDispSq = dsq;
            displaced[i] = Vector3.Lerp(displaced[i], original[i], Time.deltaTime * 5f);
        }

        return maxDispSq > returnFinishEpsilon * returnFinishEpsilon;
    }


    [BurstCompile]
    public struct VertexSpringJob : IJobParallelFor
    {
        public float deltaTime;
        public float springForce;
        public float damping;
        public float uniformScale;

        public NativeArray<float3> displacedVertices;
        public NativeArray<float3> originalVertices;
        public NativeArray<float3> vertexVelocities;

        const float kWeightEps = 1e-3f;   // clamp threshold
        const float kVelEpsSq = 1e-10f;  // tiny velocity^2

        public void Execute(int i)
        {

            float3 v = vertexVelocities[i];
            if (v.x != 0f && v.y != 0f && v.z != 0f)
            {

                // displacement in local space
                float3 disp = (displacedVertices[i] - originalVertices[i]) * uniformScale;

                // apply spring only (scaled by weight)
                v -= disp * springForce * deltaTime;

                // DAMPING SHOULD NOT BE SCALED BY WEIGHT (or make it stronger near bottom)
                v *= 1f - damping * deltaTime;

                // integrate position **scaled by weight** so bottom moves less
                float3 vWeighted = v;
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
        if (!_gsVerse.displacedVertices.IsCreated) return;

        Vector3 pointLocal = transform.InverseTransformPoint(point);
        Vector3 forceLocal = transform.InverseTransformDirection(force);

        var job = new AddDeformingForceJob
        {
            displacedVertices = _gsVerse.displacedVertices,
            vertexVelocities = vertexVelocities,
            pointLocal = pointLocal,
            force = forceLocal,
            uniformScale = _gsVerse.uniformScale,
            deltaTime = Time.deltaTime
        };

        JobHandle handle = job.Schedule(_gsVerse.displacedVertices.Length, 64);
        handle.Complete();

        _gsVerse.needsRebuild = true;
    }


    public void AddPressForce(Vector3 point, Vector3 pressNormal)
    {

    }


}
