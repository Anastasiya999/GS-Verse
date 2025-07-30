using System.Collections;
using System.Collections.Generic;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Collections;
using Unity.Mathematics;
using GaussianSplatting.Runtime;
using UnityEngine;

public class MorphStretcher : MonoBehaviour
{

    [Tooltip("The object that will control the stretch. Moving this object stretches the mesh.")]
    public Transform target;

    [Tooltip("The local point on the mesh that will remain stationary. (0, -0.5, 0) is the bottom of a default Unity cylinder or cube.")]
    public Vector3 anchorPoint = new Vector3(0, -0.5f, 0);

    [Tooltip("The axis in local space along which the mesh will stretch.")]
    public Vector3 stretchAxis = Vector3.up;
    private SplatAccessor currentDeformer;
    private Vector3 originalTop;
    // Start is called before the first frame update
    void Start()
    {
        currentDeformer = GameObject.FindGameObjectWithTag("Deformable").GetComponent<SplatAccessor>();
        // FindOriginalTop();

    }


    // private void FindOriginalTop()
    // {
    //     // Normalize the axis to ensure calculations are correct
    //     Vector3 axis = stretchAxis.normalized;
    //     float maxDistance = -Mathf.Infinity;

    //     var originalVertices = currentDeformer.GetOriginalVertices();
    //     Debug.Log(originalVertices.Length);

    //     // Find the furthest point from the anchor along the axis
    //     foreach (Vector3 vertex in originalVertices)
    //     {
    //         float distance = Vector3.Dot(vertex - anchorPoint, axis);


    //         if (distance > maxDistance)
    //         {
    //             maxDistance = distance;
    //         }
    //     }

    //     // The original "top" is the anchor point plus the direction and max distance
    //     originalTop = anchorPoint + (axis * maxDistance);
    //     Debug.Log(originalTop);
    // }

    // Update is called once per frame
    void Update()
    {

    }
}
