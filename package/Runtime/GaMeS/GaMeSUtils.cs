using System.Collections.Generic;
using GaussianSplatting.Runtime.Utils;
using UnityEngine;
using Unity.Mathematics;
using Unity.Collections;
using Unity.Jobs;
using Unity.Burst;
using System;
using System.IO;

namespace GaussianSplatting.Runtime.GaMeS
{
    public static class GaMeSUtilsEditor
    {
        private static readonly float C0 = 0.28209479177387814f;

        public class ModelParams
        {
            public List<List<List<float>>> _alpha { get; set; }
            public List<List<float>> _scale { get; set; }
        }


        public static List<Vector3> CalculateXYZ(
     List<Vector3> vertices, int numPtsEachTriangle, List<List<List<float>>> alphas)
        {
            List<Vector3> xyz = new List<Vector3>();
            int numTriangles = vertices.Count / 3;
            var normalizedAlphas = alphas;

            for (int i = 0; i < numTriangles; i++)
            {

                Vector3 v0 = vertices[i * 3];
                Vector3 v1 = vertices[i * 3 + 1];
                Vector3 v2 = vertices[i * 3 + 2];

                var triangleAphas = normalizedAlphas[i];

                for (int j = 0; j < numPtsEachTriangle; j++)
                {
                    var pointAlphas = triangleAphas[j];

                    Vector3 point = (pointAlphas[0] * v0) + (pointAlphas[1] * v1) + (pointAlphas[2] * v2);
                    xyz.Add(point);
                }

            }

            return xyz;
        }

        public static List<Vector3> GenerateNormals(int numPts)
        {
            List<Vector3> normals = new List<Vector3>(numPts);
            for (int i = 0; i < numPts; i++)
            {
                normals.Add(Vector3.zero);
            }

            return normals;
        }


        public static List<Vector3> SH2RGB(List<Vector3> shs)
        {
            List<Vector3> rgbs = new List<Vector3>(shs.Count);


            for (int i = 0; i < shs.Count; i++)
            {
                Vector3 sh = shs[i];
                Vector3 rgb = new Vector3(
                    sh.x * C0 + 0.5f,  // R
                    sh.y * C0 + 0.5f,  // G
                    sh.z * C0 + 0.5f   // B
                );

                rgbs.Add(rgb);
            }

            return rgbs;
        }

        public static List<Vector3> GenerateRandomColors(int numPts)
        {
            List<Vector3> colors = new List<Vector3>(numPts);
            for (int i = 0; i < numPts; i++)
            {

                float r = UnityEngine.Random.value;
                float g = UnityEngine.Random.value;
                float b = UnityEngine.Random.value;

                // Scale down the values by dividing by 255
                Vector3 color = new Vector3(r / 255.0f, g / 255.0f, b / 255.0f);
                colors.Add(color);
            }

            return colors;
        }

        public static List<List<float[]>> CreateFeatures(List<Vector3> colors, int maxShDegree)
        {
            // Step 1: Convert RGB to SH
            List<Vector3> fusedColor = RGB2SH(colors);

            // Step 2: Initialize features array with zeros
            int featureLength = (maxShDegree + 1) * (maxShDegree + 1);
            List<List<float[]>> features = new List<List<float[]>>(fusedColor.Count);

            for (int i = 0; i < colors.Count; i++)
            {
                List<float[]> channels = new List<float[]>();

                float[] featureR = new float[featureLength];
                float[] featureG = new float[featureLength];
                float[] featureB = new float[featureLength];

                // Step 4: Assign the SH values (RGB to SH conversion) to the first 3 features
                featureR[0] = fusedColor[i].x;  // R
                featureG[0] = fusedColor[i].y;  // G
                featureB[0] = fusedColor[i].z;  // B

                // Step 5: Set the rest of the feature vector to zero
                for (int j = 1; j < featureLength; j++)
                {
                    featureR[j] = 0;
                    featureG[j] = 0;
                    featureB[j] = 0;
                }

                channels.Add(featureR);
                channels.Add(featureG);
                channels.Add(featureB);

                // Add to the overall features list
                features.Add(channels);

            }

            return features;
        }

        public static List<Vector3> GetMeshFaceVertices(Mesh mesh, Transform meshTransform)
        {
            Vector3[] vertices = mesh.vertices;
            List<Vector3> faceVerticesList = new List<Vector3>();
            var triangles = mesh.triangles;
            int totalFaces = triangles.Length / 3;

            for (int i = 0; i < totalFaces; i++)
            {
                int baseIndex = i * 3;
                Vector3 v0 = meshTransform.TransformPoint(vertices[triangles[baseIndex]]);
                Vector3 v1 = meshTransform.TransformPoint(vertices[triangles[baseIndex + 1]]);
                Vector3 v2 = meshTransform.TransformPoint(vertices[triangles[baseIndex + 2]]);

                faceVerticesList.Add(v0);
                faceVerticesList.Add(v1);
                faceVerticesList.Add(v2);
            }

            return faceVerticesList;
        }

        public static unsafe (List<List<List<float>>> alphas, List<List<float>> scales) LoadModelParams(string modelParamsPath)
        {
            string json = File.ReadAllText(modelParamsPath);

            // Parse the JSON into the ModelParams object
            var modelParams = JSONParser.FromJson<ModelParams>(json);

            if (modelParams == null)
            {
                //TODO: add assert
                return (null, null);
            }

            return (modelParams._alpha, modelParams._scale);
        }

        public static NativeArray<InputSplatData> ReplaceSplatData(NativeArray<InputSplatData> inputSplats, NativeArray<InputSplatData> inputSplatsWithColors, Allocator allocator = Allocator.Persistent)
        {
            int length = inputSplats.Length;
            NativeArray<InputSplatData> newSplats = new NativeArray<InputSplatData>(length, allocator);

            for (int i = 0; i < length; i++)
            {
                newSplats[i] = new InputSplatData
                {
                    pos = inputSplats[i].pos,
                    nor = inputSplatsWithColors[i].nor,
                    scale = inputSplats[i].scale,
                    rot = inputSplats[i].rot,

                    dc0 = inputSplatsWithColors[i].dc0,
                    sh1 = inputSplatsWithColors[i].sh1,
                    sh2 = inputSplatsWithColors[i].sh2,
                    sh3 = inputSplatsWithColors[i].sh3,
                    sh4 = inputSplatsWithColors[i].sh4,
                    sh5 = inputSplatsWithColors[i].sh5,
                    sh6 = inputSplatsWithColors[i].sh6,
                    sh7 = inputSplatsWithColors[i].sh7,
                    sh8 = inputSplatsWithColors[i].sh8,
                    sh9 = inputSplatsWithColors[i].sh9,
                    shA = inputSplatsWithColors[i].shA,
                    shB = inputSplatsWithColors[i].shB,
                    shC = inputSplatsWithColors[i].shC,
                    shD = inputSplatsWithColors[i].shD,
                    shE = inputSplatsWithColors[i].shE,
                    shF = inputSplatsWithColors[i].shF,
                    opacity = inputSplatsWithColors[i].opacity
                };
            }

            return newSplats;
        }


        public static List<List<List<float>>> NormalizeAlphas(List<List<List<float>>> alphas)
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
                        reluApplied.Add(GaMeSUtils.ReLU(x) + 1e-8f);
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



        public static unsafe (List<Quaternion> rotations, List<Vector3> scalings) GenerateRotationsAndScales(List<Vector3> vertices, List<List<float>> scales, int numPtsEachTriangle)
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

                //Converting to Quaternion
                Quaternion rotation = rotationMatrix.rotation;
                rotation = new Quaternion(rotation.w, rotation.x, rotation.y, rotation.z);


                for (int j = 0; j < numPtsEachTriangle; j++)
                {
                    List<float> scaleFactor = scales[i * numPtsEachTriangle + j];
                    float x = math.log(GaMeSUtils.ReLU(scaleFactor[0] * s0) + eps_s0);
                    float y = math.log(GaMeSUtils.ReLU(scaleFactor[0] * s1) + eps_s0);
                    float z = math.log(GaMeSUtils.ReLU(scaleFactor[0] * s2) + eps_s0);
                    scalings.Add(new Vector3(x, y, z));
                    //broadcast rotations
                    rotations.Add(rotation);

                }


            }

            return (rotations, scalings);
        }


        private static List<Vector3> RGB2SH(List<Vector3> rgbs)
        {
            List<Vector3> shs = new List<Vector3>(rgbs.Count);


            for (int i = 0; i < rgbs.Count; i++)
            {
                Vector3 rgb = rgbs[i];
                Vector3 sh = new Vector3(
                    (rgb.x - 0.5f) / C0,  // R -> SH conversion
                    (rgb.y - 0.5f) / C0,  // G -> SH conversion
                    (rgb.z - 0.5f) / C0   // B -> SH conversion
                );
                shs.Add(sh);
            }

            return shs;
        }


    }
    public static class GaMeSUtils
    {
        public static Mesh TransformMesh(Mesh sourceMesh,
                              bool rotate90X = true,
                              bool mirrorX = true,
                              bool flipWinding = true)
        {
            if (sourceMesh == null) return null;

            Mesh mesh = sourceMesh;

            // Apply rotation then mirror to vertices
            Vector3[] verts = mesh.vertices;
            Quaternion rot = rotate90X ? Quaternion.Euler(90f, 0f, 0f) : Quaternion.identity;
            for (int i = 0; i < verts.Length; i++)
            {
                Vector3 v = verts[i];
                v = rot * v;
                if (mirrorX) v.x = -v.x;
                verts[i] = v;
            }
            mesh.vertices = verts;

            // Fix triangle winding per submesh
            for (int s = 0; s < mesh.subMeshCount; s++)
            {
                int[] tris = mesh.GetTriangles(s);
                if (flipWinding)
                {
                    for (int i = 0; i + 2 < tris.Length; i += 3)
                    {
                        int tmp = tris[i + 1];
                        tris[i + 1] = tris[i + 2];
                        tris[i + 2] = tmp;
                    }
                }
                mesh.SetTriangles(tris, s);
            }

            // Normals: either recalc (safe) or transform existing normals

            Vector3[] normals = mesh.normals;
            if (normals != null && normals.Length == verts.Length)
            {
                for (int i = 0; i < normals.Length; i++)
                {
                    Vector3 n = normals[i];
                    n = rot * n;            // rotate normal
                    if (mirrorX) n.x = -n.x; // mirror normal (inverse-transpose for mirror is same flip)
                    normals[i] = n;
                }
                mesh.normals = normals;
            }
            else
            {
                mesh.RecalculateNormals();
            }


            mesh.RecalculateBounds();

            return mesh;
        }


        public static NativeArray<float> DecodeScalesToNative(byte[] fileBytes, int numberOfSplats, Allocator allocator)
        {
            int vectorSize = GaussianSplatAsset.GetVectorSize(GaussianSplatAsset.VectorFormat.Norm11);
            int requiredLength = numberOfSplats * vectorSize;


            if (fileBytes.Length < requiredLength)
            {
                Debug.LogError($"Insufficient data: expected {requiredLength} bytes, but got {fileBytes.Length} bytes.");
                return default;
            }

            NativeArray<float> decodedScales = new NativeArray<float>(numberOfSplats, allocator, NativeArrayOptions.UninitializedMemory);

            for (int i = 0; i < numberOfSplats; i++)
            {
                int offset = i * vectorSize;
                float scale = BitConverter.ToSingle(fileBytes, offset);
                decodedScales[i] = scale;
            }

            return decodedScales;
        }

        public static NativeArray<float3> DecodeAlphasToNativeFloat3(byte[] fileBytes, int numFaces, int numPointsPerTriangle, Allocator allocator)
        {
            int floatsPerPoint = 3; // each float3 = 3 floats
            int bytesPerPoint = floatsPerPoint * GaussianSplatAsset.GetVectorSize(GaussianSplatAsset.VectorFormat.Norm11); // 3 floats * 4 bytes = 12 bytes
            int totalPoints = numFaces * numPointsPerTriangle;
            int expectedBytes = totalPoints * bytesPerPoint;


            if (fileBytes.Length < expectedBytes)
            {
                Debug.LogError($"Insufficient data: expected {expectedBytes} bytes, but got {fileBytes.Length} bytes.");
                return default;
            }

            NativeArray<float3> alphas = new NativeArray<float3>(totalPoints, allocator);

            for (int i = 0; i < totalPoints; i++)
            {
                int offset = i * bytesPerPoint;

                float x = BitConverter.ToSingle(fileBytes, offset);
                float y = BitConverter.ToSingle(fileBytes, offset + 4);
                float z = BitConverter.ToSingle(fileBytes, offset + 8);

                alphas[i] = new float3(x, y, z);

            }


            return alphas;
        }

        [BurstCompile]
        public struct CreateAssetDataJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<quaternion> m_InputRot;
            [ReadOnly] public NativeArray<float3> m_InputScale;
            [ReadOnly] public NativeArray<float3> m_InputPos;
            [ReadOnly] public NativeArray<InputSplatData> m_PrevOutput;
            [NativeDisableParallelForRestriction] public NativeArray<InputSplatData> m_Output;

            public unsafe void Execute(int i)
            {

                m_Output[i] = new InputSplatData
                {
                    pos = m_InputPos[i],
                    scale = m_InputScale[i],
                    rot = m_InputRot[i],

                    nor = m_PrevOutput[i].nor,
                    dc0 = m_PrevOutput[i].dc0,
                    sh1 = m_PrevOutput[i].sh1,
                    sh2 = m_PrevOutput[i].sh2,
                    sh3 = m_PrevOutput[i].sh3,
                    sh4 = m_PrevOutput[i].sh4,
                    sh5 = m_PrevOutput[i].sh5,
                    sh6 = m_PrevOutput[i].sh6,
                    sh7 = m_PrevOutput[i].sh7,
                    sh8 = m_PrevOutput[i].sh8,
                    sh9 = m_PrevOutput[i].sh9,
                    shA = m_PrevOutput[i].shA,
                    shB = m_PrevOutput[i].shB,
                    shC = m_PrevOutput[i].shC,
                    shD = m_PrevOutput[i].shD,
                    shE = m_PrevOutput[i].shE,
                    shF = m_PrevOutput[i].shF,
                    opacity = m_PrevOutput[i].opacity
                };
            }
        }



        [BurstCompile]
        public struct CreateAssetDataJobSelected : IJobParallelFor
        {
            [ReadOnly] public NativeArray<quaternion> m_InputRot;
            [ReadOnly] public NativeArray<float3> m_InputScale;
            [ReadOnly] public NativeArray<float3> m_InputPos;
            [ReadOnly] public NativeArray<InputSplatData> m_PrevOutput;
            [NativeDisableParallelForRestriction] public NativeArray<InputSplatData> m_Output;

            public int m_numberPtsPerTriangle;

            [ReadOnly] public NativeArray<int> m_originalTriangleIndices;

            public unsafe void Execute(int i)
            {

                int triIndex = i / m_numberPtsPerTriangle;
                int pointIndex = i % m_numberPtsPerTriangle;

                int originalTriIndex = m_originalTriangleIndices[triIndex];
                int index = originalTriIndex * m_numberPtsPerTriangle + pointIndex;


                m_Output[i] = new InputSplatData
                {
                    pos = m_InputPos[i],
                    scale = m_InputScale[i],
                    rot = m_InputRot[i],

                    nor = m_PrevOutput[index].nor,
                    dc0 = m_PrevOutput[index].dc0,
                    sh1 = m_PrevOutput[index].sh1,
                    sh2 = m_PrevOutput[index].sh2,
                    sh3 = m_PrevOutput[index].sh3,
                    sh4 = m_PrevOutput[index].sh4,
                    sh5 = m_PrevOutput[index].sh5,
                    sh6 = m_PrevOutput[index].sh6,
                    sh7 = m_PrevOutput[index].sh7,
                    sh8 = m_PrevOutput[index].sh8,
                    sh9 = m_PrevOutput[index].sh9,
                    shA = m_PrevOutput[index].shA,
                    shB = m_PrevOutput[index].shB,
                    shC = m_PrevOutput[index].shC,
                    shD = m_PrevOutput[index].shD,
                    shE = m_PrevOutput[index].shE,
                    shF = m_PrevOutput[index].shF,
                    opacity = m_PrevOutput[index].opacity
                };

            }
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

        public static NativeArray<float3> GetMeshFaceVerticesNative(GameObject gameObject, NativeArray<float3> vertices, NativeArray<int> triangles, Allocator allocator)
        {
            MeshFilter meshFilter = gameObject.GetComponent<MeshFilter>();
            if (meshFilter == null)
            {
                Debug.LogError("No MeshFilter component found on the GameObject.");
                return default;
            }

            Mesh mesh = meshFilter.mesh;
            if (mesh == null)
            {
                Debug.LogError("No mesh found in the MeshFilter component.");
                return default;
            }


            int totalFaces = triangles.Length / 3;

            NativeArray<float3> faceVertices = new NativeArray<float3>(totalFaces * 3, allocator);

            for (int i = 0; i < totalFaces; i++)
            {
                int baseIndex = i * 3;

                faceVertices[baseIndex] = TransformVertex(vertices[triangles[baseIndex]]);
                faceVertices[baseIndex + 1] = TransformVertex(vertices[triangles[baseIndex + 2]]);
                faceVertices[baseIndex + 2] = TransformVertex(vertices[triangles[baseIndex + 1]]);
            }

            return faceVertices;
        }

        public static Vector3 TransformVertex(Vector3 v)
        {
            return new Vector3(
                    -v.x,
                    v.y,
                     v.z
                    );

        }

        public static NativeArray<float3> CreateXYZData(NativeArray<float3> alphas, NativeArray<float3> vertices, int numTriangles, int numPtsEachTriangle)
        {

            NativeArray<float3> data = new(numTriangles * numPtsEachTriangle, Allocator.Persistent);

            CreateXYZDataJob job = new CreateXYZDataJob
            {
                m_Alphas = alphas,
                m_Vertices = vertices,
                m_Output = data,
                m_numberPtsPerTriangle = numPtsEachTriangle
            };
            job.Schedule(numTriangles * numPtsEachTriangle, 8192).Complete();

            return data;
        }

        [BurstCompile]
        struct CreateXYZDataJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float3> m_Alphas;
            [ReadOnly] public NativeArray<float3> m_Vertices;
            public int m_numberPtsPerTriangle;
            [NativeDisableParallelForRestriction] public NativeArray<float3> m_Output;

            public unsafe void Execute(int index)
            {
                int triangleIndex = index / m_numberPtsPerTriangle;
                int pointIndex = index % m_numberPtsPerTriangle;

                int v0Idx = triangleIndex * 3;
                float3 v0 = m_Vertices[v0Idx];
                float3 v1 = m_Vertices[v0Idx + 1];
                float3 v2 = m_Vertices[v0Idx + 2];

                int alphaIndex = triangleIndex * m_numberPtsPerTriangle + pointIndex;
                float3 alpha = m_Alphas[alphaIndex];
                float3 point = alpha.x * v0 + alpha.y * v1 + alpha.z * v2;
                m_Output[index] = point;
            }
        }

        public static NativeArray<float3> CreateXYZDataSelected(NativeArray<float3> alphas, NativeArray<float3> selectedVertices, NativeArray<int> originalTriangleIndices, int numPtsEachTriangle)
        {

            int numTriangles = selectedVertices.Length / 3;
            int totalPoints = numTriangles * numPtsEachTriangle;
            NativeArray<float3> data = new(totalPoints, Allocator.Persistent);


            CreateXYZDataJobSelected job = new CreateXYZDataJobSelected
            {
                m_Alphas = alphas,
                m_Vertices = selectedVertices,
                m_Output = data,
                m_numberPtsPerTriangle = numPtsEachTriangle,
                m_originalTriangleIndices = originalTriangleIndices,

            };
            job.Schedule(totalPoints, 8192).Complete();

            return data;
        }


        struct CreateXYZDataJobSelected : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float3> m_Alphas;
            [ReadOnly] public NativeArray<float3> m_Vertices;
            public int m_numberPtsPerTriangle;
            [NativeDisableParallelForRestriction] public NativeArray<float3> m_Output;
            [ReadOnly] public NativeArray<int> m_originalTriangleIndices;

            public unsafe void Execute(int index)
            {
                //Face index
                int triIndex = index / m_numberPtsPerTriangle;
                //Splat index
                int pointIndex = index % m_numberPtsPerTriangle;

                //Vertex base index
                int v0Idx = triIndex * 3;

                float3 v0 = m_Vertices[v0Idx];
                float3 v1 = m_Vertices[v0Idx + 1];
                float3 v2 = m_Vertices[v0Idx + 2];


                int originalTriIndex = m_originalTriangleIndices[triIndex];

                int alphaIndex = originalTriIndex * m_numberPtsPerTriangle + pointIndex;

                float3 alpha = m_Alphas[alphaIndex];
                float3 point = alpha.x * v0 + alpha.y * v1 + alpha.z * v2;
                m_Output[index] = point;
            }
        }

        [BurstCompile]
        private struct FaceVerticesJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float3> vertices;
            [ReadOnly] public NativeArray<int> triangles;
            [ReadOnly] public NativeArray<int> originalTriangleIndices;

            [WriteOnly] public NativeArray<float3> faceVertices;

            public void Execute(int index)
            {
                int triIndex = originalTriangleIndices[index];
                int baseIdx = triIndex * 3;

                int i0 = triangles[baseIdx];
                int i1 = triangles[baseIdx + 1];
                int i2 = triangles[baseIdx + 2];

                faceVertices[index * 3 + 0] = vertices[i0];
                faceVertices[index * 3 + 1] = vertices[i1];
                faceVertices[index * 3 + 2] = vertices[i2];
            }


        }

        public static NativeArray<float3> GetMeshFaceSelectedVerticesNative(
        NativeArray<float3> vertices,
        NativeArray<int> triangles,
        NativeArray<int> originalTriangleIndices,
        Allocator allocator)
        {
            var faceVertices = new NativeArray<float3>(originalTriangleIndices.Length * 3, allocator);

            var job = new FaceVerticesJob
            {
                vertices = vertices,
                triangles = triangles,
                originalTriangleIndices = originalTriangleIndices,
                faceVertices = faceVertices
            };

            job.Schedule(originalTriangleIndices.Length, 8192).Complete();

            return faceVertices;
        }


        [BurstCompile]
        public struct CreateRotationsAndScalesJobSelected : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float3> m_Vertices;
            [ReadOnly] public NativeArray<float> m_Scales;
            public int m_numPtsEachTriangle;

            [WriteOnly] public NativeArray<quaternion> Rotations;
            [WriteOnly] public NativeArray<float3> Scalings;

            [ReadOnly] public NativeArray<int> m_originalTriangleIndices;

            public void Execute(int index)
            {
                int triangleIndex = index / m_numPtsEachTriangle;
                int pointIndex = index % m_numPtsEachTriangle;

                int v0Idx = triangleIndex * 3;
                float3 v0 = m_Vertices[v0Idx];
                float3 v1 = m_Vertices[v0Idx + 1];
                float3 v2 = m_Vertices[v0Idx + 2];

                float3 normal = math.normalize(math.cross(v1 - v0, v2 - v0));
                float3 centroid = (v0 + v1 + v2) / 3.0f;
                float3 basis1 = math.normalize(v1 - centroid);

                float3 v2Init = v2 - centroid;
                float3 basis2 = math.normalize(
                    v2Init - math.dot(v2Init, normal) * normal - math.dot(v2Init, basis1) * basis1
                );

                float s1 = math.length(v1 - centroid) / 2.0f;
                float s2 = math.dot(v2Init, basis2) / 2.0f;
                float s0 = 1e-8f;

                int originalTriIndex = m_originalTriangleIndices[triangleIndex];
                int scaleIndex = originalTriIndex * m_numPtsEachTriangle + pointIndex;

                float scaleFactor = m_Scales[scaleIndex];
                float x = math.log(math.max(0, scaleFactor * s0) + s0);
                float y = math.log(math.max(0, scaleFactor * s1) + s0);
                float z = math.log(math.max(0, scaleFactor * s2) + s0);

                float3 scaleVec = new float3(x, y, z);

                // Emulate Unity's Matrix4x4.SetColumn + .rotation behavior
                float4 col0 = new float4(normal, 0f);  // x-axis
                float4 col1 = new float4(basis1, 0f);  // y-axis
                float4 col2 = new float4(basis2, 0f);  // z-axis

                float3x3 rotMatrix = new float3x3(
                    new float3(col0.x, col0.y, col0.z),
                    new float3(col1.x, col1.y, col1.z),
                    new float3(col2.x, col2.y, col2.z)
                );

                quaternion q = quaternion.LookRotationSafe(rotMatrix.c2, rotMatrix.c1); // basis2 (z), basis1 (y)
                quaternion reordered = new quaternion(q.value.w, q.value.x, q.value.y, q.value.z); // match Unity style

                Rotations[index] = reordered;
                Scalings[index] = scaleVec;
            }
        }



        public static (NativeArray<quaternion> rotations, NativeArray<float3> scalings) CreateScaleRotationData(NativeArray<float3> vertices, NativeArray<float> scales, int numPtsEachTriangle)
        {

            int numTriangles = vertices.Length / 3;
            int totalPoints = numTriangles * numPtsEachTriangle;

            NativeArray<quaternion> rotations = new NativeArray<quaternion>(totalPoints, Allocator.Persistent);
            NativeArray<float3> scalings = new NativeArray<float3>(totalPoints, Allocator.Persistent);


            CreateRotationsAndScalesJob job = new CreateRotationsAndScalesJob
            {
                m_Vertices = vertices,
                m_Scales = scales,
                m_numPtsEachTriangle = numPtsEachTriangle,
                Rotations = rotations,
                Scalings = scalings
            };
            job.Schedule(totalPoints, 9192).Complete();

            return (rotations, scalings);
        }

        public static (NativeArray<quaternion> rotations, NativeArray<float3> scalings) CreateScaleRotationDataSelected(NativeArray<float3> selectedVertices, NativeArray<float> scales, NativeArray<int> originalTriangleIndices, int numPtsEachTriangle)
        {

            int numTriangles = selectedVertices.Length / 3;
            int totalPoints = numTriangles * numPtsEachTriangle;

            NativeArray<quaternion> rotations = new NativeArray<quaternion>(totalPoints, Allocator.Persistent);
            NativeArray<float3> scalings = new NativeArray<float3>(totalPoints, Allocator.Persistent);


            CreateRotationsAndScalesJobSelected job = new CreateRotationsAndScalesJobSelected
            {
                m_Vertices = selectedVertices,
                m_Scales = scales,
                m_numPtsEachTriangle = numPtsEachTriangle,
                Rotations = rotations,
                Scalings = scalings,
                m_originalTriangleIndices = originalTriangleIndices
            };
            job.Schedule(totalPoints, 9192).Complete();

            return (rotations, scalings);
        }

        [BurstCompile]
        public struct CreateRotationsAndScalesJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float3> m_Vertices;
            [ReadOnly] public NativeArray<float> m_Scales;
            public int m_numPtsEachTriangle;

            [WriteOnly] public NativeArray<quaternion> Rotations;
            [WriteOnly] public NativeArray<float3> Scalings;

            public void Execute(int index)
            {

                int triangleIndex = index / m_numPtsEachTriangle;
                int pointIndex = index % m_numPtsEachTriangle;

                int v0Idx = triangleIndex * 3;
                float3 v0 = m_Vertices[v0Idx];
                float3 v1 = m_Vertices[v0Idx + 1];
                float3 v2 = m_Vertices[v0Idx + 2];

                float3 normal = math.normalize(math.cross(v1 - v0, v2 - v0));
                float3 centroid = (v0 + v1 + v2) / 3.0f;
                float3 basis1 = math.normalize(v1 - centroid);

                float3 v2Init = v2 - centroid;
                float3 basis2 = math.normalize(
                    v2Init - math.dot(v2Init, normal) * normal - math.dot(v2Init, basis1) * basis1
                );

                float s1 = math.length(v1 - centroid) / 2.0f;
                float s2 = math.dot(v2Init, basis2) / 2.0f;
                float s0 = 1e-8f;

                float scaleFactor = m_Scales[index];
                float x = math.log(math.max(0, scaleFactor * s0) + s0);
                float y = math.log(math.max(0, scaleFactor * s1) + s0);
                float z = math.log(math.max(0, scaleFactor * s2) + s0);

                float3 scaleVec = new float3(x, y, z);

                // Emulate Unity's Matrix4x4.SetColumn + .rotation behavior
                float4 col0 = new float4(normal, 0f);  // x-axis
                float4 col1 = new float4(basis1, 0f);  // y-axis
                float4 col2 = new float4(basis2, 0f);  // z-axis

                float3x3 rotMatrix = new float3x3(
                    new float3(col0.x, col0.y, col0.z),
                    new float3(col1.x, col1.y, col1.z),
                    new float3(col2.x, col2.y, col2.z)
                );

                quaternion q = quaternion.LookRotationSafe(rotMatrix.c2, rotMatrix.c1); // basis2 (z), basis1 (y)
                quaternion reordered = new quaternion(q.value.w, q.value.x, q.value.y, q.value.z); // match Unity style

                Rotations[index] = reordered;
                Scalings[index] = scaleVec;
            }
        }


    }




}
