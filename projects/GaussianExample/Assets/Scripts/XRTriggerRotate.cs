using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.XR.Interaction.Toolkit;

public class XRTriggerRotate : MonoBehaviour
{
    // Update is called once per frame[Header("XR Input Action (Trigger)")]
    public InputActionReference triggerActionLeft;
    public InputActionReference triggerActionRight;


    [Header("Rotation Settings")]
    public float speed = 5f;       // how strong the push is when pressing trigger

    public float limit = 1f;

    private bool isSwing = false;
    private bool isLighten = false;


    public void OnRotateLight(HoverEnterEventArgs args)
    {
        float triggerValueLeft = triggerActionLeft.action.ReadValue<float>();
        float triggerValueRight = triggerActionRight.action.ReadValue<float>();
        float triggerValue = Mathf.Max(triggerValueLeft, triggerValueRight);

        if (triggerValue > 0.1f)
            isSwing = true;


    }

    public void OnFirstHover(HoverEnterEventArgs args)
    {

        isSwing = true;

    }

    public void OnHoverExit(HoverExitEventArgs args)
    {
        float triggerValueLeft = triggerActionLeft.action.ReadValue<float>();
        float triggerValueRight = triggerActionRight.action.ReadValue<float>();
        float triggerValue = Mathf.Max(triggerValueLeft, triggerValueRight);

        if (triggerValue < 0.1f)
            isSwing = false;

    }

    void Update()
    {
        float triggerValueLeft = triggerActionLeft.action.ReadValue<float>();
        float triggerValueRight = triggerActionRight.action.ReadValue<float>();
        float triggerValue = Mathf.Max(triggerValueLeft, triggerValueRight);



        if (isSwing)
        {
            if (!isLighten)
            {
                var light = gameObject.GetComponentInChildren<Light>();
                light.intensity = 1;
                isLighten = true;

            }
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
