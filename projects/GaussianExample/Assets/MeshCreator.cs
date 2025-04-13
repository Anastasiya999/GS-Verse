using System.Collections;
using System.Collections.Generic;
using ThreeDeeBear.Models.Ply;
using System.IO;
using UnityEngine;

public class MeshCreator : MonoBehaviour
{
    private static readonly string plyFilePath = "Assets/colored_converted_meshFromServerOld.ply";
    //works
    //private static readonly string plyFilePath = "Assets/converted_hotdog.ply";
    private GameObject model;
    public Material material;
    private PlyResult ply;
    private Vector3[] vertices;
    private int[] triangles;

    void Start()
    {
        // Parse the .ply file


        if (!System.IO.File.Exists(plyFilePath))
        {
            Debug.LogError("PLY file not found at: " + plyFilePath);
            return;
        }

        string fileContent = System.IO.File.ReadAllText(plyFilePath);
        Debug.Log("PLY File Content:\n" + fileContent); // Check if content is correct

        ply = PlyHandler.GetVerticesAndTriangles(plyFilePath);

        vertices = ply.Vertices.ToArray();
        triangles = ply.Triangles.ToArray();

        if (vertices.Length == 0 || triangles.Length == 0)
        {
            Debug.LogError("PLY file contains no valid mesh data.");
            Debug.LogError(vertices.Length);
            //no triangles
            Debug.LogError(triangles.Length);
            return;
        }

        Debug.Log($"Loaded {vertices.Length} vertices and {triangles.Length / 3} triangles.");
        Debug.LogError(triangles);

        Mesh mesh = new Mesh();
        // mesh.Clear();
        mesh.vertices = vertices;
        mesh.triangles = triangles;

        // mesh.RecalculateBounds();
        mesh.RecalculateNormals();
        // mesh.RecalculateTangents();

        model = new GameObject(mesh != null ? Path.GetFileNameWithoutExtension(plyFilePath) : "PLY");


        MeshFilter meshFilter = model.AddComponent<MeshFilter>();
        meshFilter.mesh = mesh;

        // Optionally, add a MeshRenderer component to render the mesh
        MeshRenderer meshRenderer = model.AddComponent<MeshRenderer>();
        meshRenderer.material = new Material(Shader.Find("Standard"));

        /*
                if (plyResult != null)
                {
                    // Create a new mesh
                    Mesh mesh = new Mesh();

                    // Assign vertices
                    mesh.vertices = plyResult.Vertices.ToArray();

                    // Assign triangles
                    mesh.triangles = plyResult.Triangles.ToArray();


                    // Assign colors (if available)
                    if (plyResult.Colors != null && plyResult.Colors.Count > 0)
                    {
                        mesh.colors = plyResult.Colors.ToArray();
                    }

                    // Log vertex colors
                    if (plyResult.Colors != null && plyResult.Colors.Count > 0)
                    {
                        Debug.Log("Vertex Colors Found: " + plyResult.Colors.Count);
                        for (int i = 0; i < Mathf.Min(plyResult.Colors.Count, 10); i++) // Log first 10 colors
                        {
                            Debug.Log($"Vertex {i}: Color {plyResult.Colors[i]}");
                        }
                    }
                    else
                    {
                        Debug.LogWarning("No vertex colors found in PLY file.");
                    }

                    // Recalculate normals and bounds
                    mesh.RecalculateNormals();
                    mesh.RecalculateBounds();

                    // Assign the mesh to a MeshFilter component
                    MeshFilter meshFilter = gameObject.AddComponent<MeshFilter>();
                    meshFilter.mesh = mesh;

                    // Optionally, add a MeshRenderer component to render the mesh
                    MeshRenderer meshRenderer = gameObject.AddComponent<MeshRenderer>();
                    meshRenderer.material = new Material(Shader.Find("Standard"));
                }
                else
                {
                    Debug.LogError("Failed to parse the .ply file.");
                }
        */
    }
}

