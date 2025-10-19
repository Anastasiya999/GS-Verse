using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.Interaction.Toolkit;
using UnityEngine.XR.Interaction.Toolkit.Interactables;

[RequireComponent(typeof(XRBaseInteractable))]
public class XRHoverAudio : MonoBehaviour
{
    [SerializeField] private AudioSource audioSource;
    // Start is called before the first frame update

    public void OnHoverEnter(HoverEnterEventArgs args)
    {
        if (audioSource != null && !audioSource.isPlaying)
        {
            audioSource.Play();
        }
    }

    public void OnHoverExit(HoverExitEventArgs args)
    {
        if (audioSource != null && audioSource.isPlaying)
        {
            audioSource.Stop();
        }
    }

}
