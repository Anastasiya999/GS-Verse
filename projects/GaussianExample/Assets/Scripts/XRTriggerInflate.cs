using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;

public class XRTriggerRotate : MonoBehaviour
{


    // Update is called once per frame[Header("XR Input Action (Trigger)")]
    public InputActionReference triggerAction;

    [Header("Rotation Settings")]
    public float speed = 5f;       // how strong the push is when pressing trigger

    public float limit = 1f;

    private bool isSwing = false;


    void Update()
    {
        float triggerValue = triggerAction.action.ReadValue<float>();

        if (triggerValue > 0.1f) // if trigger pressed
        {
            isSwing = true;
        }
        else
        {
            isSwing = false;
        }

        if (isSwing)
        {
            limit += 0.5f;
            limit = Mathf.Clamp(limit, 1f, 55f);
            float angle = limit * Mathf.Sin(Time.time);
            transform.localRotation = Quaternion.Euler(angle, 0, 0);
        }
        else
        {
            limit -= 0.05f;
            limit = Mathf.Clamp(limit, 0f, 55f);
            float angle = limit * Mathf.Sin(Time.time);
            transform.localRotation = Quaternion.Euler(angle, 0, 0);
        }

    }
}
