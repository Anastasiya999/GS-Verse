using UnityEngine;
using Unity.Collections;

namespace GaussianSplatting.Runtime

{
    public interface IGaMeSGaussianSplatAsset
    {
        string objPath { get; }
        string pointCloudPath { get; }
        TextAsset alphaData { get; }
        TextAsset scaleData { get; }

        void SetObjPath(string path);
        void SetNumberOfSplatsPerFace(int numberOfSplats);
    }


}