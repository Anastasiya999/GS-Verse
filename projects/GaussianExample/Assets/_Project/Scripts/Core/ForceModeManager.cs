using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public enum ForceMode
{
    None,
    Press,
    Drag
}

public class ForceModeManager : MonoBehaviour
{
    public static ForceModeManager Instance { get; private set; }

    public ForceMode CurrentForceMode { get; private set; } = ForceMode.None;

    private void Awake()
    {
        if (Instance == null)
            Instance = this;
        else
            Destroy(gameObject);
    }

    public void SetForceMode(ForceMode mode)
    {
        CurrentForceMode = mode;
    }
}

