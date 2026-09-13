using UnityEngine;

public class CyclicAudioPlayer : MonoBehaviour
{
    public AudioSource audioSource;       // Assign in inspector
    public AudioClip[] audioClips;        // Drag & drop 3 (or more) clips here
    private int currentIndex = 0;         // Tracks which clip is next

    // Call this method when controller trigger event happens
    public void PlayNextClip()
    {
        if (audioClips.Length == 0) return;

        audioSource.clip = audioClips[currentIndex];
        audioSource.Play();

        // Advance index and wrap around
        currentIndex = (currentIndex + 1) % audioClips.Length;
    }
}
