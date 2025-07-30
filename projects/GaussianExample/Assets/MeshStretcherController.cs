using System.Collections;
using GaussianSplatting.Runtime;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;


public class MeshStretcherController : MonoBehaviour
{

    [Header("XR Setup")]
    [Tooltip("The XR Ray Interactor on this controller.")]
    public UnityEngine.XR.Interaction.Toolkit.Interactors.XRRayInteractor rayInteractor;
    [Tooltip("Input Action for the trigger or button press (e.g., Select).")]
    public InputActionReference stretchActionReference;

    [Header("Deformation Settings")]
    [Tooltip("Offset from the hit surface along the normal to apply the force.")]
    public float forceOffset = 0.01f; // Example value, adjust as needed
    [Tooltip("Multiplier for the drag force calculation.")]
    public float dragStrength = 100f;

    private bool isStretching = false;
    private bool actionHeld = false;
    private SplatAccessor currentDeformer;
    private Vector3? lastXRHitPoint;

    private float? lockedInteractionDistance;


    void Awake()
    {
        if (rayInteractor == null)
        {
            rayInteractor = GetComponent<UnityEngine.XR.Interaction.Toolkit.Interactors.XRRayInteractor>();
            if (rayInteractor == null)
            {
                Debug.LogError("XRDeformerController requires an XRRayInteractor component.", this);
                enabled = false;
                return;
            }
        }

    }

    void Start()
    {
        stretchActionReference.action.started += OnDeformActionStarted;
        stretchActionReference.action.canceled += OnDeformActionCanceled;
    }

    void OnDisable()
    {
        stretchActionReference.action.started -= OnDeformActionStarted;
        stretchActionReference.action.canceled -= OnDeformActionCanceled;


        if (actionHeld && currentDeformer != null)
        {
            currentDeformer.SetClickState(false);
        }
        ResetDeformationState();
    }

    private void OnDeformActionStarted(InputAction.CallbackContext context)
    {
        actionHeld = true;

        // Try to initiate interaction immediately if ray is already on target
        ProcessRayInteraction();
    }

    private void OnDeformActionCanceled(InputAction.CallbackContext context)
    {
        actionHeld = false;
        if (currentDeformer != null)
        {
            currentDeformer.SetClickState(false);
        }
        ResetDeformationState();
    }


    // Update is called once per frame
    void Update()
    {
        if (actionHeld)
        {
            ProcessRayInteraction();
        }
    }

    void ProcessRayInteraction()
    {
        // --- SECTION 1: Handle the case where the action is HELD ---
        if (actionHeld)
        {

            // If we are not currently locked onto a deformer, try to find one.
            if (currentDeformer == null)
            {
                var hitFound = rayInteractor.TryGetCurrent3DRaycastHit(out RaycastHit hitInfo);
                Debug.Log(hitFound);
                if (hitFound)
                {

                    SplatAccessor deformerOnHit = hitInfo.collider.GetComponentInParent<SplatAccessor>();
                    Debug.Log(hitInfo.collider.gameObject);
                    // We found a valid new deformer to start interacting with
                    if (deformerOnHit != null && hitInfo.collider.gameObject.CompareTag("Collider"))
                    {
                        currentDeformer = deformerOnHit;
                        currentDeformer.SetClickState(true);

                        // LOCK the interaction distance for this entire gesture
                        lockedInteractionDistance = hitInfo.distance;

                        Vector3 initialHitPoint = hitInfo.point + hitInfo.normal * forceOffset;
                        lastXRHitPoint = initialHitPoint;

                        // Optional: Apply an initial force on click if desired
                        // currentDeformer.AddDeformingForce(initialHitPoint, -hitInfo.normal * initialForce);
                    }
                }
            }
            // If we are already locked onto a deformer, continue the interaction
            else if (currentDeformer != null && lockedInteractionDistance.HasValue)
            {
                Transform rayTransform = rayInteractor.rayOriginTransform;

                // Calculate the VIRTUAL hit point at the locked distance
                Vector3 currentVirtualPoint = rayTransform.position + rayTransform.forward * lockedInteractionDistance.Value;

                // Convert the current virtual point to the deformer's local space
                Vector3 currentLocalPoint = currentDeformer.transform.InverseTransformPoint(currentVirtualPoint); ;

                // Now, calculate drag and force based on this virtual point
                if (lastXRHitPoint.HasValue)
                {
                    Vector3 localDragDirection = currentVirtualPoint - lastXRHitPoint.Value;
                    // Vector3 dragForce = localDragDirection * dragStrength;

                    Vector3 worldDragDirection = currentDeformer.transform.TransformDirection(localDragDirection);
                    Vector3 dragForce = worldDragDirection * dragStrength;

                    if (worldDragDirection.sqrMagnitude < 0.0001f) dragForce = Vector3.zero;

                    // Apply the force at the new virtual point
                    currentDeformer.AddDeformingForce(currentVirtualPoint, dragForce);

                }

                // Update the last point for the next frame's calculation
                lastXRHitPoint = currentVirtualPoint;
            }
        }
        // --- SECTION 2: Handle the case where the action is RELEASED or was not held ---
        else
        {
            // If we were deforming something, it's time to release it.
            if (currentDeformer != null)
            {
                currentDeformer.SetClickState(false);
                ResetDeformationState();
                Debug.Log(Vector3.zero == Vector3.zero);
            }
        }
    }
    /*
        void ProcessRayInteraction()
        {
            if (rayInteractor.TryGetCurrent3DRaycastHit(out RaycastHit hitInfo))
            {

                SplatAccessor deformerOnHit = hitInfo.collider.GetComponentInParent<SplatAccessor>();


                if (deformerOnHit != null && hitInfo.collider.gameObject.CompareTag("Collider"))
                {
                    // We are hitting a deformable object
                    Vector3 currentCalculatedHitPoint = hitInfo.point + hitInfo.normal * forceOffset;

                    if (actionHeld) // Equivalent to Input.GetMouseButton(0)
                    {

                        // If this is a new deformer or we switched to it
                        if (currentDeformer != deformerOnHit)
                        {
                            if (currentDeformer != null)
                            {
                                currentDeformer.SetClickState(false); // Release old one
                            }
                            currentDeformer = deformerOnHit;
                            currentDeformer.SetClickState(true);
                            lastXRHitPoint = currentCalculatedHitPoint; // Initialize for this new interaction
                        }
                        else if (lastXRHitPoint.HasValue) // Continue dragging on the same deformer
                        {
                            Vector3 dragDirection = currentCalculatedHitPoint - lastXRHitPoint.Value;
                            Vector3 dragForce = dragDirection * dragStrength;

                            currentDeformer.AddDeformingForce(currentCalculatedHitPoint, dragForce);
                            Debug.Log($"hit object2: {hitInfo.collider.gameObject.name}");
                            // SetClickState(true) should already be active from initial contact or switch
                        }
                        else // Became active on this deformer, but no previous point (should be rare with current logic)
                        {
                            currentDeformer.SetClickState(true);
                        }
                        lastXRHitPoint = currentCalculatedHitPoint;

                    }


                    currentDeformer = deformerOnHit;


                    return;
                }

            }



            // If ray is not hitting a valid SplatAccessor (or hitting nothing)
            // and an action is held, or if we previously had a deformer
            if (currentDeformer != null)
            {
                if (!actionHeld || // If button was released (handled by OnDeformActionCanceled, but good for clarity)
                    (rayInteractor.TryGetCurrent3DRaycastHit(out RaycastHit currentHit) && currentHit.collider.GetComponent<SplatAccessor>() != currentDeformer) || // Hitting something else
                    !rayInteractor.TryGetCurrent3DRaycastHit(out _)) // Hitting nothing
                {
                    currentDeformer.SetClickState(false);
                    ResetDeformationState();
                }
            }
        }
    */
    private void ResetDeformationState()
    {
        lastXRHitPoint = null;
        currentDeformer = null; // Important to clear the current deformer reference
    }


}
