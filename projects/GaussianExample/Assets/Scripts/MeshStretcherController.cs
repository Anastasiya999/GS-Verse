using GaussianSplatting.Runtime;
using GaussianSplatting.Shared;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.XR.Interaction.Toolkit.Interactors;


public class MeshStretcherController : MonoBehaviour
{
    [Header("XR Setup")]
    public XRRayInteractor rayInteractorL;
    public XRRayInteractor rayInteractorR;

    public InputActionReference stretchActionReferenceL;
    public InputActionReference pressActionReferenceL;

    public InputActionReference stretchActionReferenceR;
    public InputActionReference pressActionReferenceR;

    [Header("Deformation Settings")]
    public float forceOffset = 0.01f;
    public float dragStrength = 100f;
    public float rayLength = 5f;

    // Track separate state for each hand
    private HandState leftHand = new HandState();
    private HandState rightHand = new HandState();

    void Start()
    {
        SubscribeInput(stretchActionReferenceL, ForceMode.Drag, leftHand);
        SubscribeInput(pressActionReferenceL, ForceMode.Press, leftHand);

        SubscribeInput(stretchActionReferenceR, ForceMode.Drag, rightHand);
        SubscribeInput(pressActionReferenceR, ForceMode.Press, rightHand);

     
        leftHand.interactor = rayInteractorL;
        leftHand.stretchAction = stretchActionReferenceL;
        leftHand.pressAction = pressActionReferenceL;

        rightHand.interactor = rayInteractorR;
        rightHand.stretchAction = stretchActionReferenceR;
        rightHand.pressAction = pressActionReferenceR;
    }

    void OnDisable()
    {
        UnsubscribeInput(stretchActionReferenceL, ForceMode.Drag, leftHand);
        UnsubscribeInput(pressActionReferenceL, ForceMode.Press, leftHand);

        UnsubscribeInput(stretchActionReferenceR, ForceMode.Drag, rightHand);
        UnsubscribeInput(pressActionReferenceR, ForceMode.Press, rightHand);
    }

    void Update()
    {
        ProcessHand(leftHand);
        ProcessHand(rightHand);
    }

    private void ProcessHand(HandState hand)
    {
        switch (hand.currentMode)
        {
            case ForceMode.Press:
                ProcessPressDeformation(hand);
                break;
            case ForceMode.Drag:
                ProcessRayInteraction(hand);
                break;
        }
    }

    private void SubscribeInput(InputActionReference actionRef, ForceMode mode, HandState hand)
    {
        if (actionRef == null) return;

        actionRef.action.started += ctx => OnActionStarted(mode, hand);
        actionRef.action.canceled += ctx => OnActionCanceled(mode, hand);
    }

    private void UnsubscribeInput(InputActionReference actionRef, ForceMode mode, HandState hand)
    {
        if (actionRef == null) return;

        actionRef.action.started -= ctx => OnActionStarted(mode, hand);
        actionRef.action.canceled -= ctx => OnActionCanceled(mode, hand);
    }

    private void OnActionStarted(ForceMode mode, HandState hand)
    {
        hand.currentMode = mode;
        hand.isFirstFrameAfterClick = true;
        ForceModeManager.Instance.SetForceMode(mode);
    }

    private void OnActionCanceled(ForceMode mode, HandState hand)
    {
        if (hand.currentMode == mode)
        {
            hand.currentMode = ForceMode.None;
            hand.Reset();

            if (ForceModeManager.Instance.CurrentForceMode == mode)
            {
                ForceModeManager.Instance.SetForceMode(ForceMode.None);
            }
        }
    }

    private void ProcessPressDeformation(HandState hand)
    {
        if (hand.interactor != null && hand.interactor.TryGetCurrent3DRaycastHit(out RaycastHit hit))
        {
            IDeformable deformer = hit.collider.GetComponent<IDeformable>();

            if (deformer != null)
            {
                float pressStrength = hand.pressAction?.action?.ReadValue<float>() ?? 1f;
                deformer.AddPressForce(hit.point, hit.normal);
            }
        }
    }

    private void ProcessRayInteraction(HandState hand)
    {
        if (hand.interactor == null)
            return;

        if (hand.currentDeformer == null)
        {
            if (hand.interactor.TryGetCurrent3DRaycastHit(out RaycastHit hitInfo))
            {
                IDeformable deformerOnHit = hitInfo.collider.GetComponentInParent<IDeformable>();

                if (deformerOnHit != null)
                {
                    hand.currentDeformer = deformerOnHit;
                    hand.lockedDistance = hitInfo.distance;
                    hand.lastHitPoint = hitInfo.point + hitInfo.normal * forceOffset;
                }
            }
        }
        else if (hand.lockedDistance.HasValue)
        {
            if (hand.isFirstFrameAfterClick)
            {
                hand.isFirstFrameAfterClick = false;
                hand.lastHitPoint = hand.interactor.rayOriginTransform.position + hand.interactor.rayOriginTransform.forward * hand.lockedDistance.Value;
                return;
            }

            Vector3 currentPoint = hand.interactor.rayOriginTransform.position + hand.interactor.rayOriginTransform.forward * hand.lockedDistance.Value;

            if (hand.lastHitPoint.HasValue)
            {
                Vector3 drag = currentPoint - hand.lastHitPoint.Value;
                float triggerValue = hand.stretchAction?.action?.ReadValue<float>() ?? 1f;

               
                Vector3 worldForce = drag * dragStrength * triggerValue;

                hand.currentDeformer.AddDeformingForce(currentPoint, worldForce);
              
            }

            hand.lastHitPoint = currentPoint;
        }
    }

    // Helper class to track each hand’s state
    private class HandState
    {
        public XRRayInteractor interactor;
        public InputActionReference stretchAction;
        public InputActionReference pressAction;

        public ForceMode currentMode = ForceMode.None;
        public IDeformable currentDeformer;
        public Vector3? lastHitPoint;
        public float? lockedDistance;
        public bool isFirstFrameAfterClick = false;

        public void Reset()
        {
            currentDeformer = null;
            lastHitPoint = null;
            lockedDistance = null;
            isFirstFrameAfterClick = false;
        }
    }
}
