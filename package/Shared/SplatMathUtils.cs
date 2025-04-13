using System.Collections.Generic;
using UnityEngine;
using Unity.Mathematics;
using Unity.Collections;

namespace GaussianSplatting.Shared
{
    public static class SplatMathUtils
    {
        public static List<Vector3> CalculateXYZ(
            List<Vector3> vertices,
            int numPtsEachTriangle,
            List<List<List<float>>> alphas)
        {
            List<Vector3> xyz = new List<Vector3>();
            int numTriangles = vertices.Count / 3;
            var normalizedAlphas = NormalizeAlphas(alphas);

            for (int i = 0; i < numTriangles; i++)
            {
                Vector3 v0 = vertices[i * 3];
                Vector3 v1 = vertices[i * 3 + 1];
                Vector3 v2 = vertices[i * 3 + 2];

                var triangleAlphas = normalizedAlphas[i];

                for (int j = 0; j < numPtsEachTriangle; j++)
                {
                    var pointAlphas = triangleAlphas[j];
                    Vector3 point = (pointAlphas[0] * v0) + (pointAlphas[1] * v1) + (pointAlphas[2] * v2);
                    xyz.Add(point);
                }
            }

            return xyz;
        }

        public static (List<Quaternion> rotations, List<Vector3> scalings) GenerateRotationsAndScales(List<Vector3> vertices, List<float> scales, int numPtsEachTriangle)
        {
            int numTriangles = vertices.Count / 3;
            float eps_s0 = 1e-8f;
            List<Quaternion> rotations = new List<Quaternion>(numTriangles * numPtsEachTriangle);
            List<Vector3> scalings = new List<Vector3>(numTriangles * numPtsEachTriangle);

            for (int i = 0; i < numTriangles; i++)
            {
                // Extract the three vertices of the current triangle
                Vector3 v0 = vertices[i * 3];
                Vector3 v1 = vertices[i * 3 + 1];
                Vector3 v2 = vertices[i * 3 + 2];

                Vector3 normal = Vector3.Cross(v1 - v0, v2 - v0).normalized;

                Vector3 centroid = (v0 + v1 + v2) / 3.0f;

                Vector3 basis1 = (v1 - centroid).normalized;

                // Calculate the second basis vector (v2) using Gram-Schmidt
                Vector3 v2Init = v2 - centroid;
                Vector3 basis2 = (v2Init - Vector3.Dot(v2Init, normal) * normal - Vector3.Dot(v2Init, basis1) * basis1).normalized;

                // Scaling factors
                float s1 = (v1 - centroid).magnitude / 2.0f; // Scaling factor for v1
                float s2 = Vector3.Dot(v2Init, basis2) / 2.0f; // Scaling factor for v2
                float s0 = eps_s0;

                Matrix4x4 rotationMatrix = new Matrix4x4();

                rotationMatrix.SetColumn(0, new Vector4(normal.x, normal.y, normal.z, 0)); // x-axis
                rotationMatrix.SetColumn(1, new Vector4(basis1.x, basis1.y, basis1.z, 0)); // y-axis
                rotationMatrix.SetColumn(2, new Vector4(basis2.x, basis2.y, basis2.z, 0)); // z-axis
                rotationMatrix.SetColumn(3, new Vector4(0, 0, 0, 1)); // z-axis


                Quaternion rotation = rotationMatrix.rotation;
                rotation = new Quaternion(rotation.w, rotation.x, rotation.y, rotation.z);

                for (int j = 0; j < numPtsEachTriangle; j++)
                {
                    float scaleFactor = scales[i * 5 + j];
                    float x = math.log(ReLU(scaleFactor * s0) + eps_s0);
                    float y = math.log(ReLU(scaleFactor * s1) + eps_s0);
                    float z = math.log(ReLU(scaleFactor * s2) + eps_s0);
                    scalings.Add(new Vector3(x, y, z));

                    rotations.Add(rotation);

                }

            }

            return (rotations, scalings);
        }

        public static (NativeArray<Quaternion> rotations, NativeArray<Vector3> scalings)
    GenerateRotationsAndScalesNative(
        List<Vector3> vertices,
        List<float> scales,
        int numPtsEachTriangle,
        Allocator allocator)
        {
            int numTriangles = vertices.Count / 3;
            int totalPoints = numTriangles * numPtsEachTriangle;
            float eps_s0 = 1e-8f;

            NativeArray<Quaternion> rotations = new NativeArray<Quaternion>(totalPoints, allocator);
            NativeArray<Vector3> scalings = new NativeArray<Vector3>(totalPoints, allocator);

            for (int i = 0; i < numTriangles; i++)
            {
                Vector3 v0 = vertices[i * 3];
                Vector3 v1 = vertices[i * 3 + 1];
                Vector3 v2 = vertices[i * 3 + 2];

                Vector3 normal = Vector3.Cross(v1 - v0, v2 - v0).normalized;
                Vector3 centroid = (v0 + v1 + v2) / 3.0f;
                Vector3 basis1 = (v1 - centroid).normalized;

                Vector3 v2Init = v2 - centroid;
                Vector3 basis2 = (v2Init - Vector3.Dot(v2Init, normal) * normal - Vector3.Dot(v2Init, basis1) * basis1).normalized;

                float s1 = (v1 - centroid).magnitude / 2.0f;
                float s2 = Vector3.Dot(v2Init, basis2) / 2.0f;
                float s0 = eps_s0;

                Matrix4x4 rotationMatrix = new Matrix4x4();
                rotationMatrix.SetColumn(0, new Vector4(normal.x, normal.y, normal.z, 0));
                rotationMatrix.SetColumn(1, new Vector4(basis1.x, basis1.y, basis1.z, 0));
                rotationMatrix.SetColumn(2, new Vector4(basis2.x, basis2.y, basis2.z, 0));
                rotationMatrix.SetColumn(3, new Vector4(0, 0, 0, 1));

                Quaternion rotation = rotationMatrix.rotation;
                rotation = new Quaternion(rotation.w, rotation.x, rotation.y, rotation.z); // If needed

                for (int j = 0; j < numPtsEachTriangle; j++)
                {
                    float scaleFactor = scales[i * numPtsEachTriangle + j];
                    float x = math.log(ReLU(scaleFactor * s0) + eps_s0);
                    float y = math.log(ReLU(scaleFactor * s1) + eps_s0);
                    float z = math.log(ReLU(scaleFactor * s2) + eps_s0);

                    int index = i * numPtsEachTriangle + j;
                    scalings[index] = new Vector3(x, y, z);
                    rotations[index] = rotation;
                }
            }

            return (rotations, scalings);
        }

        public static NativeArray<Vector3> ToNativeArray(List<Vector3> list, Allocator allocator = Allocator.TempJob)
        {
            NativeArray<Vector3> nativeArray = new NativeArray<Vector3>(list.Count, allocator, NativeArrayOptions.UninitializedMemory);
            for (int i = 0; i < list.Count; i++)
            {
                nativeArray[i] = list[i];
            }
            return nativeArray;
        }
        private static List<List<List<float>>> NormalizeAlphas(List<List<List<float>>> alphas)
        {
            List<List<List<float>>> normalizedAlphas = new List<List<List<float>>>();

            foreach (var outerList in alphas)
            {
                List<List<float>> normalizedOuterList = new List<List<float>>();

                foreach (var innerList in outerList)
                {
                    List<float> reluApplied = new List<float>();
                    foreach (float x in innerList)
                    {
                        reluApplied.Add(ReLU(x) + 1e-8f);
                    }

                    float sum = 0f;
                    foreach (float x in reluApplied)
                    {
                        sum += x;
                    }

                    List<float> normalizedInnerList = new List<float>();
                    foreach (float x in reluApplied)
                    {
                        normalizedInnerList.Add(x / sum);
                    }

                    normalizedOuterList.Add(normalizedInnerList);
                }

                normalizedAlphas.Add(normalizedOuterList);
            }

            return normalizedAlphas;
        }

        public static float ReLU(float x)
        {
            return math.max(0, x);
        }

        public static List<Vector3> GetMeshFaceVertices(GameObject gameObject, bool isTransform = true)
        {
            MeshFilter meshFilter = gameObject.GetComponent<MeshFilter>();
            if (meshFilter == null)
            {
                Debug.LogError("No MeshFilter component found on the GameObject.");
                return null;
            }

            Mesh mesh = meshFilter.sharedMesh;
            if (mesh == null)
            {
                Debug.LogError("No mesh found in the MeshFilter component.");
                return null;
            }


            Vector3[] vertices = isTransform ? TransformVertices(mesh.vertices) : mesh.vertices;
            List<Vector3> faceVerticesList = new List<Vector3>();
            var triangles = mesh.triangles;

            int totalFaces = triangles.Length / 3;

            for (int i = 0; i < totalFaces; i++)
            {
                int baseIndex = i * 3;
                Vector3 v0 = vertices[triangles[baseIndex]];
                Vector3 v1 = vertices[triangles[baseIndex + 1]];
                Vector3 v2 = vertices[triangles[baseIndex + 2]];

                faceVerticesList.Add(v0);
                faceVerticesList.Add(v1);
                faceVerticesList.Add(v2);
            }

            return faceVerticesList;
        }

        public static Vector3[] TransformVertices(Vector3[] vertices)
        {
            Vector3[] transformedVertices = new Vector3[vertices.Length];

            for (int i = 0; i < vertices.Length; i++)
            {
                transformedVertices[i] = new Vector3(
                    vertices[i].x,
                    -vertices[i].z,
                    vertices[i].y
                );
            }

            return transformedVertices;
        }

    }
}
