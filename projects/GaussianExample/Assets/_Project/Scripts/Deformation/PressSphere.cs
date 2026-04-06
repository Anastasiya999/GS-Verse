using System;
using System.Collections;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Jobs;
using Unity.Burst;
using Unity.Mathematics;
using UnityEngine;

public class PressSphere : MonoBehaviour
{
    NativeArray<float3> originalVertices;
    NativeArray<float3> displacedVertices;
    NativeArray<float3> vertexVelocities;
    NativeArray<int> triangles;
    float uniformScale = 1f;

    Mesh deformingMesh;
    MeshCollider meshCollider;

    void Awake()
    {
        meshCollider = GetComponent<MeshCollider>();
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


    // Start is called before the first frame update
    void Start()
    {
        deformingMesh = GetComponent<MeshFilter>().mesh;
        MirrorAlongY(deformingMesh);

        var verts = deformingMesh.vertices;
        originalVertices = new NativeArray<float3>(verts.Length, Allocator.Persistent);
        displacedVertices = new NativeArray<float3>(verts.Length, Allocator.Persistent);

        int vertexCount = verts.Length;


        // Allocate per-vertex and triangle native arrays and register cleanup
        originalVertices = new NativeArray<float3>(vertexCount, Allocator.Persistent);

        displacedVertices = new NativeArray<float3>(vertexCount, Allocator.Persistent);

        vertexVelocities = new NativeArray<float3>(vertexCount, Allocator.Persistent);


        // copy data
        for (int i = 0; i < vertexCount; i++)
        {
            float3 v = verts[i];
            originalVertices[i] = v;
            displacedVertices[i] = v;
            vertexVelocities[i] = new float3(0, 0, 0);
        }


    }

    public void AddPressForce(Vector3 worldPoint, Vector3 pressNormal, float maxDeform, float radius, float damageFalloff)
    {
        Vector3 localPoint = transform.InverseTransformPoint(worldPoint);
        Vector3 localNormal = transform.InverseTransformDirection(pressNormal);
        Debug.Log("deforming");
        // if (IsMaxDeformed()) return;

        /* var job = new PressForceJob
         {
             pressPoint = localPoint,
             radius = radius,
             intensity = maxDeform,
             falloff = damageFalloff,
             displacedVertices = displacedVertices,
             originalVertices = originalVertices,
             uniformScale = uniformScale,
             pressNormal = pressNormal

         };
         JobHandle handle = job.Schedule(displacedVertices.Length, 64);
         handle.Complete();
         needsRebuild = true;
 */
        Debug.Log("press");
        Debug.Log(radius);
        Debug.Log(maxDeform);
        Debug.Log("press");

        var job = new AddPressForceJob
        {
            pointLocal = localPoint,
            uniformScale = uniformScale,
            deltaTime = Time.deltaTime,
            deformRadius = 0.25f,
            maxDeform = 0.5f,
            damageFalloff = 0.8f,
            damageMultiplier = 0.5f,
            displacedVertices = displacedVertices,
            originalVertices = originalVertices
        };
        JobHandle handle = job.Schedule(displacedVertices.Length, 64);
        handle.Complete();

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

                //vertexVelocities[i] += new float3(xDeform, yDeform, zDeform) * deltaTime;
                float3 deform = new float3(xDeform, yDeform, zDeform) * damageMultiplier;
                displacedVertices[i] -= deform;

            }

        }
    }


    // Update is called once per frame
    void Update()
    {
        deformingMesh.SetVertices(displacedVertices);
        deformingMesh.RecalculateNormals();

        deformingMesh.RecalculateBounds();

        // Refresh collider
        if (meshCollider != null)
        {
            meshCollider.sharedMesh = null;          // clear reference
            meshCollider.sharedMesh = deformingMesh; // re-assign
        }
    }
}


