using System.Collections;
using GaussianSplatting.Runtime;
using GaussianSplatting.Shared;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.XR.Interaction.Toolkit;


public class MeshStretcherController : MonoBehaviour
{

    [Header("XR Setup")]
    [Tooltip("The XR Ray Interactor on this controller.")]
    public UnityEngine.XR.Interaction.Toolkit.Interactors.XRRayInteractor rayInteractor;
    [Tooltip("Input Action for the trigger or button press (e.g., Select).")]
    public InputActionReference stretchActionReference;
    public InputActionReference pressActionReference;

    public Transform leftController; // assign in Inspector


    [Header("Deformation Settings")]
    [Tooltip("Offset from the hit surface along the normal to apply the force.")]
    public float forceOffset = 0.01f;
    [Tooltip("Multiplier for the drag force calculation.")]
    public float dragStrength = 100f;
    private IDeformable currentDeformer;
    private Vector3? lastXRHitPoint;
    bool isFirstFrameAfterClick = false;
    private float? lockedInteractionDistance;


    public PressSphere pressSphere;

    public float rayLength = 5f;


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
        stretchActionReference.action.started += ctx => OnActionStarted(ForceMode.Drag);
        stretchActionReference.action.canceled += ctx => OnActionCanceled(ForceMode.Drag);

        pressActionReference.action.started += ctx => OnActionStarted(ForceMode.Press);
        pressActionReference.action.canceled += ctx => OnActionCanceled(ForceMode.Press);
    }

    void OnDisable()
    {
        stretchActionReference.action.started -= ctx => OnActionStarted(ForceMode.Drag);
        stretchActionReference.action.canceled -= ctx => OnActionCanceled(ForceMode.Drag);

        pressActionReference.action.started -= ctx => OnActionStarted(ForceMode.Press);
        pressActionReference.action.canceled -= ctx => OnActionCanceled(ForceMode.Press);


        ResetDeformationState();
    }

    void Update()
    {
        var currentMode = ForceModeManager.Instance.CurrentForceMode;

        switch (currentMode)
        {
            case ForceMode.Press:
                ProcessPressDeformation();
                break;
            case ForceMode.Drag:
                ProcessRayInteraction();
                break;
        }
    }


    private void OnActionStarted(ForceMode mode)
    {
        ForceModeManager.Instance.SetForceMode(mode);
        isFirstFrameAfterClick = true;
    }

    private void OnActionCanceled(ForceMode mode)
    {
        if (ForceModeManager.Instance.CurrentForceMode == mode)
        {
            ForceModeManager.Instance.SetForceMode(ForceMode.None);

            ResetDeformationState();
        }
    }
    void ProcessPressDeformation()
    {
        Ray ray = new Ray(leftController.position, leftController.forward);

        if (Physics.Raycast(ray, out RaycastHit hit, rayLength))
        {
            IDeformable deformerOnHit = hit.collider.GetComponent<IDeformable>();

            if (deformerOnHit != null)
            {
                // Optional: scale press force by trigger value
                float pressStrength = pressActionReference.action.ReadValue<float>();

                // Add press deformation (radius, intensity, falloff could be tweaked)
                deformerOnHit.AddPressForce(
                    hit.point,
                    hit.normal,
                    0.85f * pressStrength,   // maxdeform
                    0.85f * pressStrength,    // radius
                    1.0f                     // falloff
                );
            }
        }
    }
    void ProcessRayInteraction()
    {

        if (currentDeformer == null)
        {
            // First-time hit detection
            if (rayInteractor.TryGetCurrent3DRaycastHit(out RaycastHit hitInfo))
            {
                IDeformable deformerOnHit = hitInfo.collider.GetComponentInParent<IDeformable>();

                if (deformerOnHit != null)
                {

                    currentDeformer = deformerOnHit;

                    // Lock interaction distance
                    lockedInteractionDistance = hitInfo.distance;

                    // Store initial point IN WORLD SPACE (simpler & less prone to scale bugs)
                    Vector3 initialHitPointWorld = hitInfo.point + hitInfo.normal * forceOffset;
                    lastXRHitPoint = initialHitPointWorld;
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
            Vector3 currentVirtualPointWorld = rayTransform.position + rayTransform.forward * lockedInteractionDistance.Value;

            if (lastXRHitPoint.HasValue)
            {
                // Compute world-space drag
                Vector3 worldDrag = currentVirtualPointWorld - lastXRHitPoint.Value;

                float triggerValue = 1f;
                if (stretchActionReference != null && stretchActionReference.action != null)
                {
                    // If the action is analog (0..1), use it; if not, this returns 0/1 depending on binding
                    triggerValue = stretchActionReference.action.ReadValue<float>();
                }

                // Build world force
                Vector3 worldForce = worldDrag * dragStrength * triggerValue;


                currentDeformer.AddDeformingForce(currentVirtualPointWorld, worldForce);


            }

            // Update last world-space point
            lastXRHitPoint = currentVirtualPointWorld;
        }
    }

    private void ResetDeformationState()
    {
        lastXRHitPoint = null;
        currentDeformer = null;
    }


}
