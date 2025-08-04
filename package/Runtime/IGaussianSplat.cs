using UnityEngine;
using Unity.Collections;

namespace GaussianSplatting.Runtime
{
    public interface IGaussianSplatAsset
    {
        string name { get; }
        int splatCount { get; }
        IGaussianSplatData posData { get; }

        IGaussianSplatData otherData { get; }
        IGaussianSplatData shData { get; }
        IGaussianSplatData colorData { get; }
        IGaussianSplatData chunkData { get; }
        string pointCloudPath { get; }
        TextAsset alphaData { get; }
        TextAsset scaleData { get; }
        Vector3 boundsMin { get; }
        Vector3 boundsMax { get; }
        GaussianSplatAsset.VectorFormat posFormat { get; }
        GaussianSplatAsset.VectorFormat scaleFormat { get; }
        GaussianSplatAsset.SHFormat shFormat { get; }
        GaussianSplatAsset.ColorFormat colorFormat { get; }
        Hash128 dataHash { get; }
        int formatVersion { get; }
        GaussianSplatAsset.CameraInfo[] cameras { get; }

        bool isGaMeS_asset { get; }
        string objPath { get; }


        void Dispose();
    }

    public interface IGaussianSplatData
    {
        long dataSize { get; }
        byte[] bytes { get; }
        NativeArray<T> GetData<T>() where T : struct;
    }
}