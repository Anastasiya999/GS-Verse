using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MeshDeformerInput : MonoBehaviour
{
    // Start is called before the first frame update
    public float force = 1.1f;
    public float forceOffset = 0.01f;
    Vector3? lastHitPoint = null;
    private SplatAccessor lastDeformer = null;


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
        bool mouseDown = Input.GetMouseButton(0);

        if (Physics.Raycast(inputRay, out hit))
        {
            SplatAccessor deformer = hit.collider.GetComponent<SplatAccessor>();

            if (deformer)
            {
                // Vector3 point = hit.point;

                // point += hit.normal * forceOffset;




                Vector3 currentHitPoint = hit.point + hit.normal * forceOffset;

                if (lastHitPoint.HasValue)
                {
                    Vector3 dragDirection = (currentHitPoint - lastHitPoint.Value);
                    float dragStrength = 100f; // tweak to get a stronger or softer effect

                    Vector3 dragForce = dragDirection * dragStrength;
                    deformer.AddDeformingForce(currentHitPoint, dragForce);
                    lastDeformer.SetClickState(true);
                    //deformer.AddDeformingForce(currentHitPoint, force);
                }

                deformer.SetClickState(mouseDown);

                // Reset the previous one if it's different
                if (lastDeformer != null && lastDeformer != deformer)
                {
                    lastDeformer.SetClickState(false);
                }

                lastDeformer = deformer;

                lastHitPoint = currentHitPoint;
            }



        }
        else
        {
            if (lastDeformer != null)
            {
                lastDeformer.SetClickState(false);
                lastDeformer = null;
            }

            // Reset when not hitting the object
            lastHitPoint = null;
        }

    }
}
