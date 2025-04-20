using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MeshDeformerInput : MonoBehaviour
{
    // Start is called before the first frame update
    public float force = 1.1f;
    public float forceOffset = 0.01f;
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetMouseButton(0))
        {
            HandleInput();
        }
    }

    void HandleInput()
    {
        Ray inputRay = Camera.main.ScreenPointToRay(Input.mousePosition);
        RaycastHit hit;

        if (Physics.Raycast(inputRay, out hit))
        {
            SplatAccessor deformer = hit.collider.GetComponent<SplatAccessor>();

            if (deformer)
            {
                Vector3 point = hit.point;

                point += hit.normal * forceOffset;
                deformer.AddDeformingForce(point, force);
                deformer.AddDeformingForce(point, force);
            }
        }

    }
}
