using System.Collections;
using GaussianSplatting.Runtime;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.Interaction.Toolkit;


public class MorphSplat : MonoBehaviour
{
    // Start is called before the first frame update

    public SplatAccessor currentDeformer;
    public float dragStrength = 100f;
    private Vector3 LastPosition;
    private UnityEngine.XR.Interaction.Toolkit.Interactables.XRBaseInteractable interactable;
    void Start()
    {
        interactable = GetComponent<UnityEngine.XR.Interaction.Toolkit.Interactables.XRBaseInteractable>();
        if (interactable != null)
        {
            interactable.selectExited.AddListener(OnSelectExit);
            interactable.selectEntered.AddListener(OnSelectEnter);
        }
    }

    private void OnSelectExit(SelectExitEventArgs args)
    {
        currentDeformer.SetClickState(false);
        Debug.Log("object was deselected");
    }


    private void OnSelectEnter(SelectEnterEventArgs args)
    {
        currentDeformer.SetClickState(true);
        Debug.Log("object was selected");
    }
    // Update is called once per frame
    void Update()
    {
        Vector3 currentVirtualPoint = transform.position;



        // Convert the current virtual point to the deformer's local space
        Vector3 currentLocalPoint = currentDeformer.transform.InverseTransformPoint(currentVirtualPoint); ;

        // Now, calculate drag and force based on this virtual point
        if (LastPosition != currentVirtualPoint)
        {
            // currentDeformer.SetClickState(true);
            Vector3 localDragDirection = currentVirtualPoint - LastPosition;
            // Vector3 dragForce = localDragDirection * dragStrength;

            Vector3 worldDragDirection = currentDeformer.transform.TransformDirection(localDragDirection);
            Vector3 dragForce = worldDragDirection * dragStrength;

            // Apply the force at the new virtual point
            currentDeformer.AddDeformingForce(currentVirtualPoint, dragForce);

        }


        LastPosition = currentVirtualPoint;


    }
}
