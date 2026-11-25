using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.XR.Interaction.Toolkit;
using UnityEngine.XR.Interaction.Toolkit.Interactors;

public class XRTriggerStretchByRay : MonoBehaviour
{
    [Header("Input")]
    public InputActionReference triggerActionLeft; // XR trigger reference
    public InputActionReference triggerActionRight; // XR trigger reference

    [Header("Stretch Settings")]
    public float baseStretch = 0.1f;      // minimum stretch speed when hovering
    public float triggerBoost = 0.4f;     // extra stretch multiplier with trigger
    public float maxScale = 3f;           // maximum allowed scale
    public float minScale = 0.5f;         // minimum allowed scale
    public float resetSpeed = 5f;         // speed at which it resets to original scale

//  [Header("References")]
    private Transform controllerTransform; // XR controller transform (ray origin)


    private bool isHovered = false;
    private Vector3 initialScale;

    void Awake()
    {
        initialScale = transform.localScale;
    }

    // Called when XR controller hovers over the object
    public void OnHoverEnter(HoverEnterEventArgs args)
    {
        isHovered = true;

        // Try to grab controller transform automatically if available
        if (controllerTransform== null && args.interactorObject is XRRayInteractor rayInteractor)
        {
            controllerTransform = rayInteractor.transform;
        }
    }

    // Called when hover stops
    public void OnHoverExit(HoverExitEventArgs args)
    {
     //   isHovered = false;
    }

    void Update()
    {
        float triggerValueLeft = triggerActionLeft.action.ReadValue<float>(); // 0..1
        float triggerValueRight = triggerActionRight.action.ReadValue<float>(); // 0..1
      
        float triggerValue = Mathf.Max(triggerValueLeft, triggerValueRight);
        bool isTriggered = triggerValue > 0;

        if (isHovered && isTriggered && controllerTransform != null)
        {
            // Stretch amount based on trigger
            float stretchAmount = baseStretch + triggerValue * triggerBoost;

            // Use controller's forward as direction
            Vector3 direction = controllerTransform.forward.normalized;

            // Project direction into object's local space
            Vector3 localDir = transform.InverseTransformDirection(direction);

            // Build new scale
            Vector3 newScale = transform.localScale;
            newScale += new Vector3(
                localDir.x * stretchAmount * Time.deltaTime,
                localDir.y * stretchAmount * Time.deltaTime,
                localDir.z * stretchAmount * Time.deltaTime
            );

            // Clamp scales individually
            newScale.x = Mathf.Clamp(newScale.x, minScale, maxScale);
            newScale.y = Mathf.Clamp(newScale.y, minScale, maxScale);
            newScale.z = Mathf.Clamp(newScale.z, minScale, maxScale);

            transform.localScale = newScale;
        }
        else
        {
            // Smooth reset to initial scale
            transform.localScale = Vector3.Lerp(
                transform.localScale,
                initialScale,
                resetSpeed * Time.deltaTime
            );
        }
    }
}
