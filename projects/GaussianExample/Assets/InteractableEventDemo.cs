using System.Collections;
using System.Collections.Generic;
using UnityEngine.XR.Interaction.Toolkit;
using UnityEngine.XR.Interaction.Toolkit.Interactors;
using UnityEngine;
using UnityEngine.XR.Interaction.Toolkit.Interactors.Visuals;
using UnityEngine.InputSystem;
using GaussianSplatting.Runtime;

public class InteractableEventDemo : MonoBehaviour
{
    public NearFarInteractor nearFarInteractor;
    public InputActionReference stretchActionReference;
    public float forceOffset = 0.01f; // Example value, adjust as needed
    private float? lockedInteractionDistance = float.PositiveInfinity;
    public SplatAccessor currentDeformer;
    private bool actionHeld = false;
    public float dragStrength = 100f;
    private Vector3? lastXRHitPoint;
    private Transform rayTransform;

    public void PlayHoverEnter(HoverEnterEventArgs args)
    {
        Vector3 endPoint = Vector3.zero;
        Vector3 endNormal = Vector3.zero;

        nearFarInteractor.TryGetCurveEndPoint(out endPoint, snapToSelectedAttachIfAvailable: true);
        nearFarInteractor.TryGetCurveEndNormal(out endNormal, snapToSelectedAttachIfAvailable: true);
        Collider interactableCollider = args.interactableObject.colliders[0];
        Vector3 closestPoint = interactableCollider.ClosestPoint(endPoint);

        float distance = Vector3.Distance(args.interactorObject.transform.position, closestPoint);



        if (actionHeld)
        {
            //get
            //lockedDistance
            //lastHitPoint

            if (lockedInteractionDistance == float.PositiveInfinity)
            {
                currentDeformer.SetClickState(true);
                lockedInteractionDistance = Vector3.Distance(args.interactorObject.transform.position, endPoint);
                // currentDeformer.SetClickState(true);
                Vector3 initialHitPoint = endPoint + endNormal * forceOffset;
                lastXRHitPoint = initialHitPoint;
                rayTransform = args.interactorObject.transform;
                Debug.Log("setting true");
            }

        }
        else
        {
            // If we were deforming something, it's time to release it.
            if (lockedInteractionDistance != float.PositiveInfinity)
            {
                currentDeformer.SetClickState(false);
                Debug.Log("setting false");

            }
        }


    }

    void Start()
    {
        stretchActionReference.action.started += OnDeformActionStarted;
        stretchActionReference.action.canceled += OnDeformActionCanceled;
    }


    private void OnDeformActionStarted(InputAction.CallbackContext context)
    {
        actionHeld = true;

    }

    private void OnDeformActionCanceled(InputAction.CallbackContext context)
    {
        actionHeld = false;

    }



    public void PlayHoverSelect(SelectEnterEventArgs args)
    {
        actionHeld = false;


    }
    public void PlayHoverSelect(SelectExitEventArgs args)
    {
        actionHeld = false;


    }
    public void Update()
    {
        if (actionHeld)
        {

            if (lastXRHitPoint.HasValue)
            {
                //rembeber transform


                // Calculate the VIRTUAL hit point at the locked distance
                Vector3 currentVirtualPoint = rayTransform.position + rayTransform.forward * lockedInteractionDistance.Value;

                // Convert the current virtual point to the deformer's local space
                Vector3 currentLocalPoint = currentDeformer.transform.InverseTransformPoint(currentVirtualPoint); ;

                // Now, calculate drag and force based on this virtual point
                if (lastXRHitPoint.HasValue)
                {

                    Vector3 localDragDirection = currentLocalPoint - lastXRHitPoint.Value;
                    // Vector3 dragForce = localDragDirection * dragStrength;

                    Vector3 worldDragDirection = currentDeformer.transform.TransformDirection(localDragDirection);
                    Vector3 dragForce = worldDragDirection * dragStrength;

                    if (worldDragDirection.sqrMagnitude < 0.0001f)
                    {
                        dragForce = Vector3.zero;
                        Debug.Log("zero");
                    }

                    // Apply the force at the new virtual point
                    currentDeformer.AddDeformingForce(currentVirtualPoint, dragForce);

                }

                // Update the last point for the next frame's calculation
                lastXRHitPoint = currentLocalPoint;
            }
        }
    }

}
