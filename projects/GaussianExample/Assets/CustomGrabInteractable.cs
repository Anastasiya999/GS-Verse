using UnityEngine;
using UnityEngine.XR.Interaction.Toolkit;
using UnityEngine.XR.Interaction.Toolkit.Interactables;
using UnityEngine.XR.Interaction.Toolkit.Interactors;

public class CustomGrabInteractable : XRGrabInteractable
{
    private Vector3 initialLocalPos;
    private Quaternion initialLocalRot;
    private Vector3 grabOffsetPos;
    private Quaternion grabOffsetRot;

    protected override void OnSelectEntering(SelectEnterEventArgs args)
    {
        base.OnSelectEntering(args);

        // Save the initial local offset between attach point and object
        // if (args.interactorObject is XRBaseInteractor interactor)
        // {
        //     Transform attach = interactor.GetAttachTransform(this);

        //     // Grab offset in world space
        //     grabOffsetPos = transform.position - attach.position;
        //     grabOffsetRot = Quaternion.Inverse(attach.rotation) * transform.rotation;
        // }
    }

    protected override void OnSelectEntered(SelectEnterEventArgs args)
    {
        base.OnSelectEntered(args);

        // if (args.interactorObject is XRBaseInteractor interactor)
        // {
        //     Transform attach = interactor.GetAttachTransform(this);

        //     // Set world position and rotation with preserved offset
        //     attach.position = transform.position - grabOffsetPos;
        //     attach.rotation = transform.rotation * grabOffsetRot;
        //     Debug.Log("transform");
        // }
    }
}
