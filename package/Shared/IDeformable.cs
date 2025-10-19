using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace GaussianSplatting.Shared
{
    public interface IDeformable
    {
        // Short tap / press behavior
        void AddPressForce(Vector3 worldPoint, Vector3 pressNormal);

        // Continuous dragging/deforming behavior
        void AddDeformingForce(Vector3 worldPoint, Vector3 worldForce);
    }
}
