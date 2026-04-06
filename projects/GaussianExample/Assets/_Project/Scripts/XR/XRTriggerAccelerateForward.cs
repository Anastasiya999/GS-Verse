using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.XR.Interaction.Toolkit;

public class XRTriggerAccelerateForward : MonoBehaviour
{
    [Header("Input")]
    public InputActionReference triggerActionLeft;
    public InputActionReference triggerActionRight;

    [Header("Movement Settings")]
    public float baseAcceleration = 2f;
    public float triggerBoost = 5f;
    public float maxSpeed = 10f;
    public float drag = 1f;

    public enum MovementDirection { Forward, Backward, Right, Left, Up, Down }
    [Header("Direction")]
    public MovementDirection moveDirection = MovementDirection.Forward;

    private float currentSpeed = 0f;
    private bool isHovered = false;
    private Rigidbody rb;

    void Awake()
    {
        rb = GetComponent<Rigidbody>();

        if (rb == null)
        {
            rb = gameObject.AddComponent<Rigidbody>();
        }

        rb.constraints = RigidbodyConstraints.FreezeRotation; // prevent toy car tipping
    }

    // Called when XR controller hovers over the car
    public void OnHoverEnter(HoverEnterEventArgs args)
    {
        isHovered = true;

    }

    // Called when hover stops
    public void OnHoverExit(HoverExitEventArgs args)
    {
        isHovered = false;

    }

    void FixedUpdate()
    {
        float triggerValueLeft = triggerActionLeft.action.ReadValue<float>(); // 0 to 1
        float triggerValueRight = triggerActionRight.action.ReadValue<float>(); // 0 to 1
        float triggerValue = Mathf.Max(triggerValueLeft, triggerValueRight);

        if (isHovered && triggerValue > 0)
        {
            // Apply acceleration
            float accel = baseAcceleration + triggerValue * triggerBoost;
            currentSpeed += accel * Time.fixedDeltaTime;
        }
        else
        {
            // Apply drag when not hovered
            currentSpeed -= drag * Time.fixedDeltaTime;
        }

        // Clamp speed
        currentSpeed = Mathf.Clamp(currentSpeed, 0f, maxSpeed);

        Vector3 direction = GetDirectionVector();

        // Move in chosen direction
        rb.MovePosition(rb.position + direction * currentSpeed * Time.fixedDeltaTime);

    }

    private Vector3 GetDirectionVector()
    {
        switch (moveDirection)
        {
            case MovementDirection.Forward: return transform.forward;
            case MovementDirection.Backward: return -transform.forward;
            case MovementDirection.Right: return transform.right;
            case MovementDirection.Left: return -transform.right;
            case MovementDirection.Up: return transform.up;
            case MovementDirection.Down: return -transform.up;
            default: return transform.forward;
        }
    }
}


