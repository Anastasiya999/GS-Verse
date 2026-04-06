using UnityEngine;
public class GSVerse : GSBase
{
    MeshCollider meshCollider;

    protected bool _needsColliderUpdate = false;
    public bool needsColliderUpdate
    {
        get => _needsColliderUpdate;
        set => _needsColliderUpdate = value;
    }
    protected override void LoadAndSetupMesh()
    {
        base.LoadAndSetupMesh();

        meshCollider = gameObject.GetComponent<MeshCollider>();
        if (meshCollider == null)
            meshCollider = gameObject.AddComponent<MeshCollider>();

        meshCollider.sharedMesh = mesh;
        meshCollider.convex = true;

        meshCollider.enabled = false;
        meshCollider.enabled = true;
    }

    protected override void ScheduleAssetRebuild()
    {
        base.ScheduleAssetRebuild();
        if (needsColliderUpdate && meshCollider != null)
        {
            meshCollider.sharedMesh = null;
            meshCollider.sharedMesh = _mesh;

        }
    }
}

