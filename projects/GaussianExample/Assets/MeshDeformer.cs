using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MeshDeformer : MonoBehaviour
{
    // Start is called before the first frame update
    Mesh deformingMesh;
    public float springForce = 20f;
    float uniformScale = 1f;
    public float damping = 5f;

    Vector3[] originalVertices, displacedVertices;
    Vector3[] vertexVelocities;
    void Start()
    {
        //GetComponent<MeshRenderer>().enabled = false; // Hide the mesh
        uniformScale = transform.localScale.x;
        deformingMesh = GetComponent<MeshFilter>().mesh;
        originalVertices = deformingMesh.vertices;
        displacedVertices = new Vector3[originalVertices.Length];
        for (int i = 0; i < originalVertices.Length; i++)
        {
            displacedVertices[i] = originalVertices[i];
        }
        vertexVelocities = new Vector3[originalVertices.Length];
    }

    // Update is called once per frame
    void Update()
    {
        for (int i = 0; i < displacedVertices.Length; i++)
        {
            UpdateVertex(i);
        }
        deformingMesh.vertices = displacedVertices;
        //asset
        deformingMesh.RecalculateNormals();

    }

    void UpdateVertex(int i)
    {
        Vector3 velocity = vertexVelocities[i];
        Vector3 displacement = displacedVertices[i] - originalVertices[i];
        displacement *= uniformScale;
        velocity -= displacement * springForce * Time.deltaTime;
        velocity *= 1f - damping * Time.deltaTime;
        vertexVelocities[i] = velocity;
        displacedVertices[i] += velocity * (Time.deltaTime / uniformScale);

        Debug.DrawLine(transform.TransformPoint(originalVertices[i]),
                  transform.TransformPoint(displacedVertices[i]), Color.red);
    }

    public void AddDeformingForce(Vector3 point, float force)
    {
        for (int i = 0; i < displacedVertices.Length; i++)
        {
            AddForceToVertex(i, point, force);
        }
        Debug.DrawLine(Camera.main.transform.position, point);
    }

    void AddForceToVertex(int i, Vector3 point, float force)
    {
        point = transform.InverseTransformPoint(point);
        Vector3 pointToVertex = displacedVertices[i] - point;

        pointToVertex *= uniformScale;
        float attenuatedForce = force / (1f + pointToVertex.sqrMagnitude);
        float velocity = attenuatedForce * Time.deltaTime;
        vertexVelocities[i] += pointToVertex.normalized * velocity;
    }
}
