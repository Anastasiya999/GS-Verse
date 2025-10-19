using System.Collections;
using GaussianSplatting.Runtime;
using System.Collections.Generic;
using UnityEngine;

public class MeshDeformerInput : MonoBehaviour
{
    // Start is called before the first frame update
    public float force = 1.1f;
    public float forceOffset = 0.01f;
    Vector3? lastHitPoint = null;
    private SplatAccessor lastDeformer = null;
    public float dragStrength = 100f;
    public Transform anchorPoint;
    private float? lockedInteractionDistance;

    private Vector3 lastMousePosition;


    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {

        if (Input.GetMouseButton(0))
        {
            HandleInput2();
        }
    }


    void HandleInput()
    {
        Ray inputRay = Camera.main.ScreenPointToRay(Input.mousePosition);
        RaycastHit hit;
        bool mouseDown = Input.GetMouseButton(0);


        if (Physics.Raycast(inputRay, out hit))
        {

            SplatAccessor deformer = hit.collider.GetComponentInParent<SplatAccessor>();

            if (deformer)
            {

                Vector3 currentHitPoint = hit.point + hit.normal * forceOffset;

                if (mouseDown)
                {
                    if (lastHitPoint.HasValue)
                    {
                        Vector3 dragDirection = currentHitPoint - lastHitPoint.Value;
                        float dragStrength = 100f;



                        Vector3 dragForce = dragDirection * dragStrength;
                        if (dragDirection.sqrMagnitude < 0.05f) { dragForce = Vector3.zero; Debug.Log("zero"); }
                        deformer.AddDeformingForce(currentHitPoint, dragForce);
                        deformer.SetClickState(true);
                        Debug.Log(dragDirection);
                    }

                    lastHitPoint = currentHitPoint;
                    lastDeformer = deformer;
                }
                else
                {
                    // Mouse released
                    deformer.SetClickState(false);
                    lastHitPoint = null;
                    lastDeformer = null;
                }
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

    void HandleInput2()
    {
        Ray inputRay = Camera.main.ScreenPointToRay(Input.mousePosition);
        RaycastHit hitInfo;
        bool mouseDown = Input.GetMouseButton(0);
        Vector3 currentMousePosition = Input.mousePosition;

        if (mouseDown)
        {
            if (lastDeformer == null)
            {
                var hitFound = Physics.Raycast(inputRay, out hitInfo);

                if (hitFound)
                {

                    SplatAccessor deformer = hitInfo.collider.GetComponentInParent<SplatAccessor>();


                    // Add the tag check like in ProcessRayInteraction
                    if (deformer != null)
                    {

                        lastDeformer = deformer;
                        lastDeformer.SetClickState(true);

                        // LOCK the interaction distance for this entire gesture
                        lockedInteractionDistance = hitInfo.distance;

                        Vector3 initialHitPoint = hitInfo.point + hitInfo.normal * forceOffset;
                        lastHitPoint = initialHitPoint;

                        // Optional: Apply an initial force on click if desired
                        // lastDeformer.AddDeformingForce(initialHitPoint, -hitInfo.normal * initialForce);
                    }
                }
            }
            else if (lastDeformer != null && lockedInteractionDistance.HasValue)
            {
                // Use the mouse ray direction, not camera forward
                Vector3 rayDirection = inputRay.direction;
                Vector3 rayOrigin = inputRay.origin;

                // Calculate the VIRTUAL hit point at the locked distance along the mouse ray
                Vector3 currentVirtualPoint = rayOrigin + rayDirection * lockedInteractionDistance.Value;

                // Remove this unused variable (it was calculated but never used)
                Vector3 currentLocalPoint = lastDeformer.transform.InverseTransformPoint(currentVirtualPoint);

                // Now, calculate drag and force based on this virtual point
                if (lastHitPoint.HasValue)
                {
                    // Calculate drag direction from the original hit point toward current mouse position
                    Vector3 dragDirection = currentLocalPoint - lastDeformer.transform.InverseTransformPoint(lastHitPoint.Value);
                    Vector3 dragForce = dragDirection * dragStrength;
                    Debug.Log(dragForce);
                    if (lastMousePosition != null && (lastMousePosition - currentMousePosition).sqrMagnitude < 0.05f) { dragForce = Vector3.zero; Debug.Log("zero"); }

                    // Apply the force at the original hit point, pulling toward the mouse
                    lastDeformer.AddDeformingForce(lastHitPoint.Value, dragForce);
                }

                // DON'T update lastHitPoint - keep it as the original hit point for stretching
            }
        }
        else
        {
            // Add null check to prevent errors
            if (lastDeformer != null)
            {
                lastDeformer.SetClickState(false);
                // Reset all state like in ProcessRayInteraction
                //ResetDeformationState(); // Or manually reset the variables:
                lastHitPoint = null;
                lastDeformer = null;
                lockedInteractionDistance = null;
            }
        }

        lastMousePosition = currentMousePosition;
    }
    void OnDrawGizmos()
    {
        if (anchorPoint != null)
        {
            Gizmos.color = Color.green;
            Gizmos.DrawSphere(anchorPoint.position, 0.05f);
            Gizmos.DrawLine(Camera.main.transform.position, anchorPoint.position);
        }
    }
    /*
        void HandleInput()
        {
            bool mouseDown = Input.GetMouseButton(0);

            if (mouseDown)
            {
                if (lastDeformer == null)
                {
                    // On initial mouse down, do a raycast to anchor the deformer
                    Ray inputRay = Camera.main.ScreenPointToRay(Input.mousePosition);
                    RaycastHit hit;

                    if (Physics.Raycast(inputRay, out hit))
                    {
                        lastDeformer = hit.collider.GetComponent<SplatAccessor>();
                        if (lastDeformer != null)
                        {
                            lastMouseWorldPoint = hit.point + hit.normal * forceOffset;
                            lastDeformer.SetClickState(true);

                        }
                    }
                }
                else
                {
                    // Continue applying deformation in mouse direction

                    Ray inputRay = Camera.main.ScreenPointToRay(Input.mousePosition);
                    Vector3 worldPoint = inputRay.origin + inputRay.direction * 3f; // Adjust distance as needed

                    if (lastMouseWorldPoint.HasValue)
                    {
                        Vector3 dragDirection = worldPoint - lastMouseWorldPoint.Value;
                        Vector3 dragForce = Vector3.up * dragStrength;

                        Debug.Log("" + lastMouseWorldPoint.Value);

                        lastDeformer.AddDeformingPullForce(worldPoint);
                    }

                    lastMouseWorldPoint = worldPoint;
                }
            }
            else
            {
                // Mouse released: reset everything
                if (lastDeformer != null)
                {
                    lastDeformer.SetClickState(false);
                    //  lastDeformer = null;
                }
                lastMouseWorldPoint = null;
            }
        }
        */
}
