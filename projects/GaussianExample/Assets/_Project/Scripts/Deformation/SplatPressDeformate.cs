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
    public float springForce = 10f;
    public float damping = 2f;
    public float maxDeform = 0.5f;
    public float radius = 0.85f;
    public float damageFalloff = 1.0f;

    NativeArray<float3> vertexVelocities;
    private readonly List<Action> _deferredCleanup = new List<Action>();

    [SerializeField] private GSVerse _gsVerse;

    void Awake()
    {
        try
        {
            if (_gsVerse == null)
                _gsVerse = GetComponent<GSVerse>();

            if (_gsVerse != null)
            {
                _gsVerse.OnInitVertices += Initialize;
                _gsVerse.needsColliderUpdate = true;
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

    private void Initialize(Vector3[] vertices)
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

    private void UpdateMesh(float obj)
    {

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

    public void AddPressForce(Vector3 worldPoint, Vector3 pressNormal)
    {
        Vector3 localPoint = transform.InverseTransformPoint(worldPoint);
        Vector3 localNormal = transform.InverseTransformDirection(pressNormal);
        var displacedVertices = _gsVerse.GetDisplacedVertices();
        var originalVertices = _gsVerse.originalVertices;
        var uniformScale = _gsVerse.uniformScale;


        if (IsMaxDeformed()) return;

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
        _gsVerse.needsRebuild = true;

    }

    public bool IsMaxDeformed()
    {
        // compute maximum deviation from original vertices
        var displacedVertices = _gsVerse.displacedVertices;
        var originalVertices = _gsVerse.originalVertices;
        var uniformScale = _gsVerse.uniformScale;
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

        // Unsubscribe events to avoid dangling references / callbacks after destroy
        if (_gsVerse != null)
        {
            _gsVerse.OnInitVertices -= Initialize;
            _gsVerse.OnUpdate -= UpdateMesh;
        }

        // Use single cleanup path — run deferred cleanups (they will dispose vertexVelocities)
        RunDeferredCleanup();

        DisposeIfCreated(ref vertexVelocities);

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
