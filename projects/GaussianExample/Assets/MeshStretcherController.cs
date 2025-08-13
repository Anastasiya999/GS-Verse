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
    public float forceOffset = 0.01f;
    [Tooltip("Multiplier for the drag force calculation.")]
    public float dragStrength = 100f;
    private bool actionHeld = false;
    private SplatAccessor currentDeformer;
    private Vector3? lastXRHitPoint;
    bool isFirstFrameAfterClick = false;
    private float? lockedInteractionDistance;


    void Awake()
    {
        if (rayInteractor == null)
        {
            rayInteractor = GetComponent<UnityEngine.XR.Interaction.Toolkit.Interactors.XRRayInteractor>();
            if (rayInteractor == null)
            {

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
        isFirstFrameAfterClick = true;

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

    void Update()
    {
        if (actionHeld)
        {
            ProcessRayInteraction();
        }

    }

    void ProcessRayInteraction()
    {

        if (!actionHeld)
        {
            if (currentDeformer != null)
            {
                currentDeformer.SetClickState(false);
                ResetDeformationState();
            }
            return;
        }

        if (currentDeformer == null)
        {
            // First-time hit detection
            if (rayInteractor.TryGetCurrent3DRaycastHit(out RaycastHit hitInfo))
            {
                SplatAccessor deformerOnHit = hitInfo.collider.GetComponentInParent<SplatAccessor>();
                // if (deformerOnHit != null && hitInfo.collider.gameObject.CompareTag("Collider"))
                if (deformerOnHit != null)
                {
                    currentDeformer = deformerOnHit;
                    currentDeformer.SetClickState(true);

                    // Lock interaction distance
                    lockedInteractionDistance = hitInfo.distance;

                    // Store initial point
                    Vector3 initialHitPoint = hitInfo.point + hitInfo.normal * forceOffset;
                    lastXRHitPoint = currentDeformer.transform.InverseTransformPoint(initialHitPoint);
                }
            }
        }
        else if (lockedInteractionDistance.HasValue)
        {
            // Skip applying force if it's the first frame after click
            if (isFirstFrameAfterClick)
            {
                isFirstFrameAfterClick = false;
                return;
            }

            Transform rayTransform = rayInteractor.rayOriginTransform;
            Vector3 currentVirtualPoint = rayTransform.position + rayTransform.forward * lockedInteractionDistance.Value;
            Vector3 currentLocalPoint = currentDeformer.transform.InverseTransformPoint(currentVirtualPoint);

            if (lastXRHitPoint.HasValue)
            {
                Vector3 localDragDirection = currentLocalPoint - lastXRHitPoint.Value;
                Vector3 worldDragDirection = currentDeformer.transform.TransformDirection(localDragDirection);

                Vector3 dragForce = worldDragDirection.sqrMagnitude < 0.0001f
                    ? Vector3.zero
                    : worldDragDirection * dragStrength;

                currentDeformer.AddDeformingForce(currentVirtualPoint, currentDeformer.transform.InverseTransformDirection(dragForce));
            }

            lastXRHitPoint = currentLocalPoint;
        }
    }
    private void ResetDeformationState()
    {
        lastXRHitPoint = null;
        currentDeformer = null;
    }


}
