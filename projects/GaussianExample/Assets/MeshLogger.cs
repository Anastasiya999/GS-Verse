using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

public class MeshLogger : MonoBehaviour
{
    void Start()
    {
        // Get MeshFilter component
        MeshFilter meshFilter = GetComponent<MeshFilter>();

        if (meshFilter == null || meshFilter.mesh == null)
        {
            Debug.LogError("MeshFilter or Mesh is missing!");
            return;
        }

        Mesh mesh = meshFilter.mesh;
        Vector3[] vertices = mesh.vertices;

        // Get triangles (faces)
        int[] triangles = mesh.triangles;

        Debug.Log($"Total Faces: {triangles.Length }");

         Debug.Log($"Total Vertices: { mesh.vertices.Length}");



        // Set the path to save the file (within the project folder)
        string filePath = Application.dataPath + "/FaceCenters.txt";

        // Create a StreamWriter to write to the file
        StreamWriter writer = new StreamWriter(filePath);

        // Write header to the file
        writer.WriteLine("Face Index, X, Y, Z");

        // Loop through all the triangles and calculate face centers
        for (int i = 0; i < triangles.Length; i += 3)
        {
            // Get the three vertices of the face
            Vector3 v1 = vertices[triangles[i]];
            Vector3 v2 = vertices[triangles[i + 1]];
            Vector3 v3 = vertices[triangles[i + 2]];

            // Compute the center of the face
            Vector3 faceCenter = (v1 + v2 + v3) / 3f;

            // Write the face center to the text file
            writer.WriteLine($"{i / 3}, {faceCenter.x}, {faceCenter.y}, {faceCenter.z}");
        }

        // Close the writer to finalize the file
        writer.Close();

        Debug.Log("Face centers written to: " + filePath);
        }
    
}
