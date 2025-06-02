// SPDX-License-Identifier: MIT

using System;
using System.Collections.Generic;
using System.IO;
using GaussianSplatting.Editor.Utils;
using GaussianSplatting.Runtime;
using GaussianSplatting.Runtime.Utils;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEditor;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using ThreeDeeBear.Models.Ply;

namespace GaussianSplatting.Editor
{
    [BurstCompile]
    public class GaussianSplatAssetCreator : EditorWindow
    {
        const string kProgressTitle = "Creating Gaussian Splat Asset";
        const string kCamerasJson = "cameras.json";
        const string kPrefQuality = "nesnausk.GaussianSplatting.CreatorQuality";
        const string kPrefOutputFolder = "nesnausk.GaussianSplatting.CreatorOutputFolder";
        private static readonly float C0 = 0.28209479177387814f;

        private static readonly int precision = 4;

        private static readonly string filePath = "Assets/mesh_vertices.txt";

        private Dictionary<string, int> trimeshFaceMapping = new Dictionary<string, int>();
        private List<Vector3> m_Vertices;

        enum DataQuality
        {
            VeryHigh,
            High,
            Medium,
            Low,
            VeryLow,
            Custom,
        }

        readonly FilePickerControl m_FilePicker = new();

        [SerializeField] string m_InputFile;
        [SerializeField] string m_InputPointCloudFile;
        [SerializeField] string m_ObjFilePath;
        [SerializeField] string m_InputJsonFile;  // For the JSON file
        [SerializeField] string m_InputJsonTrimeshMappingFile;
        [SerializeField] bool m_ImportCameras = true;
        [SerializeField] GameObject m_SelectedSceneObject;

        [SerializeField] string m_OutputFolder = "Assets/GaussianAssets";
        [SerializeField] DataQuality m_Quality = DataQuality.Medium;
        [SerializeField] GaussianSplatAsset.VectorFormat m_FormatPos;
        [SerializeField] GaussianSplatAsset.VectorFormat m_FormatScale;
        [SerializeField] GaussianSplatAsset.ColorFormat m_FormatColor;
        [SerializeField] GaussianSplatAsset.SHFormat m_FormatSH;

        string m_ErrorMessage;
        string m_PrevPlyPath;
        int m_PrevVertexCount;
        long m_PrevFileSize;

        GameObject loadedObject;

        bool isUsingChunks =>
            m_FormatPos != GaussianSplatAsset.VectorFormat.Float32 ||
            m_FormatScale != GaussianSplatAsset.VectorFormat.Float32 ||
            m_FormatColor != GaussianSplatAsset.ColorFormat.Float32x4 ||
            m_FormatSH != GaussianSplatAsset.SHFormat.Float32;

        [MenuItem("Tools/Gaussian Splats/Create GaussianSplatAsset")]
        public static void Init()
        {
            var window = GetWindowWithRect<GaussianSplatAssetCreator>(new Rect(50, 50, 360, 440), false, "Gaussian Splat Creator", true);
            window.minSize = new Vector2(320, 320);
            window.maxSize = new Vector2(1500, 1500);
            window.Show();
        }

        void Awake()
        {
            m_Quality = (DataQuality)EditorPrefs.GetInt(kPrefQuality, (int)DataQuality.Medium);
            m_OutputFolder = EditorPrefs.GetString(kPrefOutputFolder, "Assets/GaussianAssets");
        }

        void OnEnable()
        {
            ApplyQualityLevel();
        }

        void OnGUI()
        {
            EditorGUILayout.Space();
            GUILayout.Label("Input data", EditorStyles.boldLabel);
            var rect = EditorGUILayout.GetControlRect(true);
            m_InputFile = m_FilePicker.PathFieldGUI(rect, new GUIContent("Input mesh PLY File"), m_InputFile, "ply", "PointCloudFile");

            var rectSecond = EditorGUILayout.GetControlRect(true);
            m_InputPointCloudFile = m_FilePicker.PathFieldGUI(rectSecond, new GUIContent("Input pointCloud PLY File"), m_InputPointCloudFile, "ply", "PointCloudFile");

            // JSON file input field
            string jsonFilePath = m_InputJsonFile;
            jsonFilePath = EditorGUILayout.TextField("Input JSON params File", jsonFilePath);
            if (GUILayout.Button("Browse for JSON File..."))
            {

                string path = EditorUtility.OpenFilePanel("Select JSON File", "", "json");
                if (!string.IsNullOrEmpty(path))
                {
                    m_InputJsonFile = path;
                }
            }

            //  EditorGUILayout.BeginHorizontal();
            GUILayout.Label("Input Data Source", EditorStyles.boldLabel);

            // Scene Object Selection
            m_SelectedSceneObject = (GameObject)EditorGUILayout.ObjectField("Scene Object", m_SelectedSceneObject, typeof(GameObject), true);



            GUILayout.Label("Preprocessing", EditorStyles.boldLabel);


            EditorGUILayout.Space();

            m_ImportCameras = EditorGUILayout.Toggle("Import Cameras", m_ImportCameras);


            if (m_InputPointCloudFile != m_PrevPlyPath && !string.IsNullOrWhiteSpace(m_InputPointCloudFile))
            {
                m_PrevVertexCount = 0;
                m_ErrorMessage = null;
                try
                {
                    m_PrevVertexCount = GaussianFileReader.ReadFileHeader(m_InputPointCloudFile);
                }
                catch (Exception ex)
                {
                    m_ErrorMessage = ex.Message;
                }

                m_PrevFileSize = File.Exists(m_InputPointCloudFile) ? new FileInfo(m_InputFile).Length : 0;
                m_PrevFilePath = m_InputPointCloudFile;


            }

            if (m_PrevVertexCount > 0)
                EditorGUILayout.LabelField("File Size", $"{EditorUtility.FormatBytes(m_PrevFileSize)} - {m_PrevVertexCount:N0} splats");
            else
                GUILayout.Space(EditorGUIUtility.singleLineHeight);

            EditorGUILayout.Space();

            GUILayout.Label("Output", EditorStyles.boldLabel);

            rect = EditorGUILayout.GetControlRect(true);
            string newOutputFolder = m_FilePicker.PathFieldGUI(rect, new GUIContent("Output Folder"), m_OutputFolder, null, "GaussianAssetOutputFolder");
            if (newOutputFolder != m_OutputFolder)
            {
                m_OutputFolder = newOutputFolder;
                EditorPrefs.SetString(kPrefOutputFolder, m_OutputFolder);
            }

            var newQuality = (DataQuality)EditorGUILayout.EnumPopup("Quality", m_Quality);
            if (newQuality != m_Quality)
            {
                m_Quality = newQuality;
                EditorPrefs.SetInt(kPrefQuality, (int)m_Quality);
                ApplyQualityLevel();
            }

            long sizePos = 0, sizeOther = 0, sizeCol = 0, sizeSHs = 0, totalSize = 0;
            if (m_PrevVertexCount > 0)
            {
                sizePos = GaussianSplatAsset.CalcPosDataSize(m_PrevVertexCount, m_FormatPos);
                sizeOther = GaussianSplatAsset.CalcOtherDataSize(m_PrevVertexCount, m_FormatScale);
                sizeCol = GaussianSplatAsset.CalcColorDataSize(m_PrevVertexCount, m_FormatColor);
                sizeSHs = GaussianSplatAsset.CalcSHDataSize(m_PrevVertexCount, m_FormatSH);
                long sizeChunk = isUsingChunks ? GaussianSplatAsset.CalcChunkDataSize(m_PrevVertexCount) : 0;
                totalSize = sizePos + sizeOther + sizeCol + sizeSHs + sizeChunk;
            }

            const float kSizeColWidth = 70;
            EditorGUI.BeginDisabledGroup(m_Quality != DataQuality.Custom);
            EditorGUI.indentLevel++;
            GUILayout.BeginHorizontal();
            m_FormatPos = (GaussianSplatAsset.VectorFormat)EditorGUILayout.EnumPopup("Position", m_FormatPos);
            GUILayout.Label(sizePos > 0 ? EditorUtility.FormatBytes(sizePos) : string.Empty, GUILayout.Width(kSizeColWidth));
            GUILayout.EndHorizontal();
            GUILayout.BeginHorizontal();
            m_FormatScale = (GaussianSplatAsset.VectorFormat)EditorGUILayout.EnumPopup("Scale", m_FormatScale);
            GUILayout.Label(sizeOther > 0 ? EditorUtility.FormatBytes(sizeOther) : string.Empty, GUILayout.Width(kSizeColWidth));
            GUILayout.EndHorizontal();
            GUILayout.BeginHorizontal();
            m_FormatColor = (GaussianSplatAsset.ColorFormat)EditorGUILayout.EnumPopup("Color", m_FormatColor);
            GUILayout.Label(sizeCol > 0 ? EditorUtility.FormatBytes(sizeCol) : string.Empty, GUILayout.Width(kSizeColWidth));
            GUILayout.EndHorizontal();
            GUILayout.BeginHorizontal();
            m_FormatSH = (GaussianSplatAsset.SHFormat)EditorGUILayout.EnumPopup("SH", m_FormatSH);
            GUIContent shGC = new GUIContent();
            shGC.text = sizeSHs > 0 ? EditorUtility.FormatBytes(sizeSHs) : string.Empty;
            if (m_FormatSH >= GaussianSplatAsset.SHFormat.Cluster64k)
            {
                shGC.tooltip = "Note that SH clustering is not fast! (3-10 minutes for 6M splats)";
                shGC.image = EditorGUIUtility.IconContent("console.warnicon.sml").image;
            }
            GUILayout.Label(shGC, GUILayout.Width(kSizeColWidth));
            GUILayout.EndHorizontal();
            EditorGUI.indentLevel--;
            EditorGUI.EndDisabledGroup();
            if (totalSize > 0)
                EditorGUILayout.LabelField("Asset Size", $"{EditorUtility.FormatBytes(totalSize)} - {(double)m_PrevFileSize / totalSize:F2}x smaller");
            else
                GUILayout.Space(EditorGUIUtility.singleLineHeight);


            EditorGUILayout.Space();
            GUILayout.BeginHorizontal();
            GUILayout.Space(30);
            if (GUILayout.Button("Create Asset"))
            {
                CreateAsset();
            }
            GUILayout.Space(30);
            GUILayout.EndHorizontal();

            if (!string.IsNullOrWhiteSpace(m_ErrorMessage))
            {
                EditorGUILayout.HelpBox(m_ErrorMessage, MessageType.Error);
            }
        }

        void ApplyQualityLevel()
        {
            switch (m_Quality)
            {
                case DataQuality.Custom:
                    break;
                case DataQuality.VeryLow: // 18.62x smaller, 32.27 PSNR
                    m_FormatPos = GaussianSplatAsset.VectorFormat.Norm11;
                    m_FormatScale = GaussianSplatAsset.VectorFormat.Norm6;
                    m_FormatColor = GaussianSplatAsset.ColorFormat.BC7;
                    m_FormatSH = GaussianSplatAsset.SHFormat.Cluster4k;
                    break;
                case DataQuality.Low: // 14.01x smaller, 35.17 PSNR
                    m_FormatPos = GaussianSplatAsset.VectorFormat.Norm11;
                    m_FormatScale = GaussianSplatAsset.VectorFormat.Norm6;
                    m_FormatColor = GaussianSplatAsset.ColorFormat.Norm8x4;
                    m_FormatSH = GaussianSplatAsset.SHFormat.Cluster16k;
                    break;
                case DataQuality.Medium: // 5.14x smaller, 47.46 PSNR
                    m_FormatPos = GaussianSplatAsset.VectorFormat.Norm11;
                    m_FormatScale = GaussianSplatAsset.VectorFormat.Norm11;
                    m_FormatColor = GaussianSplatAsset.ColorFormat.Norm8x4;
                    m_FormatSH = GaussianSplatAsset.SHFormat.Norm6;
                    break;
                case DataQuality.High: // 2.94x smaller, 57.77 PSNR
                    m_FormatPos = GaussianSplatAsset.VectorFormat.Norm16;
                    m_FormatScale = GaussianSplatAsset.VectorFormat.Norm16;
                    m_FormatColor = GaussianSplatAsset.ColorFormat.Float16x4;
                    m_FormatSH = GaussianSplatAsset.SHFormat.Norm11;
                    break;
                case DataQuality.VeryHigh: // 1.05x smaller
                    m_FormatPos = GaussianSplatAsset.VectorFormat.Float32;
                    m_FormatScale = GaussianSplatAsset.VectorFormat.Float32;
                    m_FormatColor = GaussianSplatAsset.ColorFormat.Float32x4;
                    m_FormatSH = GaussianSplatAsset.SHFormat.Float32;
                    break;
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        // input file splat data is expected to be in this format
        public struct InputSplatData
        {
            public Vector3 pos;
            public Vector3 nor;
            public Vector3 dc0;
            public Vector3 sh1, sh2, sh3, sh4, sh5, sh6, sh7, sh8, sh9, shA, shB, shC, shD, shE, shF;
            public float opacity;
            public Vector3 scale;
            public Quaternion rot;
        }

        static T CreateOrReplaceAsset<T>(T asset, string path) where T : UnityEngine.Object
        {
            T result = AssetDatabase.LoadAssetAtPath<T>(path);
            if (result == null)
            {
                AssetDatabase.CreateAsset(asset, path);
                result = asset;
            }
            else
            {
                if (typeof(Mesh).IsAssignableFrom(typeof(T))) { (result as Mesh)?.Clear(); }
                EditorUtility.CopySerialized(asset, result);
            }
            return result;
        }

        private void LoadVerticesFromSceneObject()
        {
            if (m_SelectedSceneObject == null)
            {
                Debug.LogError("No GameObject selected.");
                return;
            }

            MeshFilter meshFilter = m_SelectedSceneObject.GetComponent<MeshFilter>();

            Color[] vertexColors = meshFilter.mesh.colors;

            Debug.Log("No colors found in the MeshFilter component.");

            // Print the vertex colors (for debugging)
            for (int i = 0; i < vertexColors.Length; i++)
            {
                Debug.Log("Vertex " + i + " Color: " + vertexColors[i]);
            }

            if (meshFilter != null)
            {
                m_Vertices = new List<Vector3>(meshFilter.sharedMesh.vertices);
                Debug.Log($"Loaded {m_Vertices.Count} vertices, {meshFilter.sharedMesh.triangles.Length} triangles from scene object.");

                DebugMeshTriangles(meshFilter.sharedMesh.vertices, meshFilter.sharedMesh.triangles);
            }
            else
            {
                Debug.LogError("Selected GameObject does not have a MeshFilter component.");
            }
        }

        private static GameObject CreateGameObjectFromPly(string filePath, GameObject targetObject)
        {
            if (targetObject == null)
            {
                Debug.LogError("Target GameObject is null. Ensure it exists before calling this function.");
                return null;
            }


            if (!File.Exists(filePath))
            {
                Debug.LogError("PLY file not found at: " + filePath);
                return null;
            }

            PlyResult ply = PlyHandler.GetVerticesAndTriangles(filePath);

            if (ply.Vertices == null || ply.Vertices.Count == 0 || ply.Triangles == null || ply.Triangles.Count == 0)
            {
                Debug.LogError("PLY file contains no valid mesh data.");
                return null;
            }

            MeshFilter meshFilter = targetObject.GetComponent<MeshFilter>();
            if (meshFilter != null && meshFilter.sharedMesh != null)
            {
                if (meshFilter.sharedMesh != null)
                {

                    UnityEngine.Object.DestroyImmediate(meshFilter.sharedMesh);

                }

            }
            else
            {
                meshFilter = targetObject.AddComponent<MeshFilter>(); // Add if missing
            }

            // Create new mesh
            Mesh mesh = new Mesh
            {
                vertices = ply.Vertices.ToArray(),
                triangles = ply.Triangles.ToArray()


            };
            // Get the first triangle's vertices
            Vector3 v1 = mesh.vertices[mesh.triangles[0]];
            Vector3 v2 = mesh.vertices[mesh.triangles[1]];
            Vector3 v3 = mesh.vertices[mesh.triangles[2]];



            Debug.Log($"First face vertices from asset creator:\n{v1}\n{v2}\n{v3}");

            Vector3 v4 = mesh.vertices[mesh.triangles[3]];
            Vector3 v5 = mesh.vertices[mesh.triangles[4]];
            Vector3 v6 = mesh.vertices[mesh.triangles[5]];

            Debug.Log($"Second face vertices from asset creator:\n{v4}\n{v5}\n{v6}");
            mesh.RecalculateNormals(); // Update normals

            meshFilter.sharedMesh = mesh; // Assign new mesh

            MeshRenderer meshRenderer = targetObject.GetComponent<MeshRenderer>();
            if (meshRenderer == null)
            {
                meshRenderer = targetObject.AddComponent<MeshRenderer>(); // Add if missing
            }

            //
            meshRenderer.material = new Material(Shader.Find("Standard"));

            return targetObject;
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

        private void InitializeTrimeshFaceMapping(string jsonFilePath)
        {
            if (string.IsNullOrEmpty(jsonFilePath) || !File.Exists(jsonFilePath))
            {
                Debug.LogError("JSON file path is invalid or file does not exist.");
                return;
            }

            // Read the JSON file
            string jsonContent = File.ReadAllText(jsonFilePath);

            // Parse the JSON into the TrimeshMapping object
            var trimeshMapping = JSONParser.FromJson<Dictionary<string, int>>(jsonContent);

            if (trimeshMapping == null)
            {
                Debug.LogError("Failed to parse JSON file or faceMappings is null.");
                return;
            }

            // Populate the trimeshFaceMapping dictionary
            trimeshFaceMapping = trimeshMapping;
            Debug.Log("Trimesh face mapping initialized successfully.");
        }

        string GetFaceKey(Vector3 v0, Vector3 v1, Vector3 v2)
        {
            // Convert from Unity to trimesh coordinate system (negate the x-coordinate)
            Vector3 cv0 = new Vector3(v0.x, v0.y, v0.z);
            Vector3 cv1 = new Vector3(v1.x, v1.y, v1.z);
            Vector3 cv2 = new Vector3(v2.x, v2.y, v2.z);

            // Extract all coordinates into a single list
            float[] coordinates = new float[]
            {
            Mathf.Abs(cv0.x), Mathf.Abs(cv0.y), Mathf.Abs(cv0.z),
            Mathf.Abs(cv1.x), Mathf.Abs(cv1.y), Mathf.Abs(cv1.z),
            Mathf.Abs(cv2.x), Mathf.Abs(cv2.y), Mathf.Abs(cv2.z)
            };

            // Sort all coordinates in ascending order
            Array.Sort(coordinates);

            // Quantize the sorted coordinates
            string[] keyComponents = new string[coordinates.Length];
            for (int i = 0; i < coordinates.Length; i++)
            {
                keyComponents[i] = Quantize(coordinates[i]).ToString();
            }

            // Join the components into a single key
            string key = string.Join("_", keyComponents);
            return key;
        }

        string Quantize(float value)
        {
            float scale = Mathf.Pow(10, precision);
            float rounded = Mathf.Round(value * scale) / scale;

            // Build a format string that always displays at least one decimal digit.
            // For example, if precision is 5, the format becomes "0.0####".
            string format;
            if (precision > 0)
            {
                // Force one decimal digit, then allow up to (precision - 1) optional decimals.
                format = "0.0" + new string('#', precision - 1);
            }
            else
            {
                format = "0";
            }

            return rounded.ToString(format);
        }

        // Function to calculate the cosine of angles and generate face keys
        private void DebugMeshTriangles(Vector3[] vertices, int[] triangles)
        {
            int foundCount = 0;
            int totalFaces = triangles.Length / 3;

            for (int i = 0; i < totalFaces; i++)
            {
                // Extract the vertices of the current triangle
                int baseIndex = i * 3;
                Vector3 v0 = vertices[triangles[baseIndex]];
                Vector3 v1 = vertices[triangles[baseIndex + 1]];
                Vector3 v2 = vertices[triangles[baseIndex + 2]];

                // Generate the face key
                string faceKey = GetFaceKey(v0, v1, v2);

                // Check if the face key exists in the mapping
                if (trimeshFaceMapping.TryGetValue(faceKey, out int trimeshOrder))
                {
                    foundCount++;
                    Debug.Log($"  Unity Face {i} maps to trimesh order index: {trimeshOrder}");
                }
                else
                {
                    Debug.LogWarning($"  Unity Face {i} not found in mapping. Key: {faceKey}");
                }
            }

            // Log the summary of the mapping
            Debug.Log($"Debug Mapping Summary: Total Faces = {totalFaces}, Found = {foundCount}, Missing = {totalFaces - foundCount}");
        }
        unsafe (List<List<List<float>>> alphas, List<List<float>> scales) LoadModelParams(string modelParamsPath)
        {
            string json = File.ReadAllText(modelParamsPath);

            // Parse the JSON into the ModelParams object
            var modelParams = JSONParser.FromJson<ModelParams>(json);

            if (modelParams == null)
            {
                Debug.LogError("Failed to parse model_params.json");
                return (null, null);
            }

            // Debug: Print the first values of _alpha and _scale
            if (modelParams._alpha != null && modelParams._alpha.Count > 0)
            {
                Debug.Log($"First _alpha value: {modelParams._alpha.Count} {modelParams._alpha[0].Count}, type of alphas: {modelParams._alpha[0][0].Count}");
            }

            if (modelParams._scale != null && modelParams._scale.Count > 0)
            {
                Debug.Log($"First _scale value: {modelParams._scale[0][0]}, {modelParams._alpha.Count}, {modelParams._scale.Count}");
            }

            return (modelParams._alpha, modelParams._scale);
        }

        unsafe void CreateAsset()
        {
            m_ErrorMessage = null;
            if (string.IsNullOrWhiteSpace(m_InputFile))
            {
                m_ErrorMessage = $"Select input mesh PLY file";
                return;
            }

            if (string.IsNullOrWhiteSpace(m_InputPointCloudFile))
            {
                m_ErrorMessage = $"Select input pointCloud PLY file";
                return;
            }

            if (string.IsNullOrWhiteSpace(m_OutputFolder) || !m_OutputFolder.StartsWith("Assets/"))
            {
                m_ErrorMessage = $"Output folder must be within project, was '{m_OutputFolder}'";
                return;
            }
            Directory.CreateDirectory(m_OutputFolder);

            EditorUtility.DisplayProgressBar(kProgressTitle, "Reading data files", 0.0f);
            GaussianSplatAsset.CameraInfo[] cameras = LoadJsonCamerasFile(m_InputPointCloudFile, m_ImportCameras);

            //create object
            // var gameObject = CreateGameObjectFromPly(m_InputFile, m_SelectedSceneObject);
            var gameObject = m_SelectedSceneObject;
            var (alphas, scales) = LoadModelParams(m_InputJsonFile);

            var normalizedAlpha = NormalizeAlphas(alphas);

            Debug.Log($"Finished generating colors. Total colors created: {normalizedAlpha.Count} vs {normalizedAlpha[0].Count} vs {normalizedAlpha[0][0].Count} ");

            using NativeArray<InputSplatData> inputSplats = CreateSplatDataFromMemory(normalizedAlpha, scales, gameObject, 9);

            using NativeArray<InputSplatData> inputSplatsColored = LoadInputSplatFile(m_InputPointCloudFile);

            using NativeArray<InputSplatData> inputSplatsWithColors = ReplaceSplatData(inputSplats, inputSplatsColored);

            Debug.Log($"InputSplat count {inputSplats.Length}");

            Debug.Log($"InputSplatWithColors count {inputSplatsWithColors.Length}");

            Debug.Log($"First Splat Position from Create asset: {inputSplatsWithColors[0].scale} {GaussianUtils.LinearScale(inputSplatsWithColors[0].scale)}");



            if (inputSplatsWithColors.Length == 0)
            {
                EditorUtility.ClearProgressBar();
                return;
            }

            float3 boundsMin, boundsMax;
            var boundsJob = new CalcBoundsJob
            {
                m_BoundsMin = &boundsMin,
                m_BoundsMax = &boundsMax,
                m_SplatData = inputSplatsWithColors
            };
            boundsJob.Schedule().Complete();

            EditorUtility.DisplayProgressBar(kProgressTitle, "Morton reordering", 0.05f);
            ReorderMorton(inputSplatsWithColors, boundsMin, boundsMax);

            DebugInputSplats(inputSplatsWithColors);
            // cluster SHs
            NativeArray<int> splatSHIndices = default;
            NativeArray<GaussianSplatAsset.SHTableItemFloat16> clusteredSHs = default;
            if (m_FormatSH >= GaussianSplatAsset.SHFormat.Cluster64k)
            {
                EditorUtility.DisplayProgressBar(kProgressTitle, "Cluster SHs", 0.2f);
                ClusterSHs(inputSplatsWithColors, m_FormatSH, out clusteredSHs, out splatSHIndices);
            }

            string baseName = Path.GetFileNameWithoutExtension(FilePickerControl.PathToDisplayString(m_InputFile));

            EditorUtility.DisplayProgressBar(kProgressTitle, "Creating data objects", 0.7f);
            GaussianSplatAsset asset = ScriptableObject.CreateInstance<GaussianSplatAsset>();
            asset.Initialize(inputSplatsWithColors.Length, m_FormatPos, m_FormatScale, m_FormatColor, m_FormatSH, boundsMin, boundsMax, cameras, m_InputPointCloudFile);
            asset.name = baseName;

            var dataHash = new Hash128((uint)asset.splatCount, (uint)asset.formatVersion, 0, 0);

            string pathChunk = $"{m_OutputFolder}/{baseName}_chk.bytes";
            string pathAlpha = $"{m_OutputFolder}/{baseName}_alpha.bytes";
            string pathScale = $"{m_OutputFolder}/{baseName}_scale.bytes";
            string pathPos = $"{m_OutputFolder}/{baseName}_pos.bytes";
            string pathOther = $"{m_OutputFolder}/{baseName}_oth.bytes";
            string pathCol = $"{m_OutputFolder}/{baseName}_col.bytes";
            string pathSh = $"{m_OutputFolder}/{baseName}_shs.bytes";
            //TODO: without LinearizeData(inputSplatsWithColors);
            LinearizeData(inputSplatsWithColors);

            // if we are using full lossless (FP32) data, then do not use any chunking, and keep data as-is
            bool useChunks = isUsingChunks;
            if (useChunks)
                CreateChunkData(inputSplatsWithColors, pathChunk, ref dataHash);
            Debug.Log($"First Splat Position from Create asset after reorder: {inputSplatsWithColors[0].scale} {normalizedAlpha[0][0].Count}");

            CreatePositionsData(inputSplatsWithColors, pathPos, ref dataHash);
            CreateScaleData(scales, pathScale, ref dataHash);
            CreateAlphasData(normalizedAlpha, pathAlpha, ref dataHash, normalizedAlpha[0][0].Count);
            CreateOtherData(inputSplatsWithColors, pathOther, ref dataHash, splatSHIndices);
            CreateColorData(inputSplatsWithColors, pathCol, ref dataHash);
            CreateSHData(inputSplatsWithColors, pathSh, ref dataHash, clusteredSHs);
            asset.SetDataHash(dataHash);

            splatSHIndices.Dispose();
            clusteredSHs.Dispose();

            // files are created, import them so we can get to the imported objects, ugh
            EditorUtility.DisplayProgressBar(kProgressTitle, "Initial texture import", 0.85f);
            AssetDatabase.Refresh(ImportAssetOptions.ForceUncompressedImport);

            EditorUtility.DisplayProgressBar(kProgressTitle, "Setup data onto asset", 0.95f);
            asset.SetAssetFiles(
                useChunks ? AssetDatabase.LoadAssetAtPath<TextAsset>(pathChunk) : null,
                AssetDatabase.LoadAssetAtPath<TextAsset>(pathPos),
                AssetDatabase.LoadAssetAtPath<TextAsset>(pathOther),
                AssetDatabase.LoadAssetAtPath<TextAsset>(pathCol),
                AssetDatabase.LoadAssetAtPath<TextAsset>(pathSh), AssetDatabase.LoadAssetAtPath<TextAsset>(pathAlpha), AssetDatabase.LoadAssetAtPath<TextAsset>(pathScale));

            var assetPath = $"{m_OutputFolder}/{baseName}.asset";
            var savedAsset = CreateOrReplaceAsset(asset, assetPath);

            EditorUtility.DisplayProgressBar(kProgressTitle, "Saving assets", 0.99f);
            AssetDatabase.SaveAssets();
            EditorUtility.ClearProgressBar();

            Selection.activeObject = savedAsset;
        }

        NativeArray<InputSplatData> LoadInputSplatFile(string filePath)
        {
            NativeArray<InputSplatData> data = default;
            if (!File.Exists(filePath))
            {
                m_ErrorMessage = $"Did not find {filePath} file";
                return data;
            }
            try
            {
                GaussianFileReader.ReadFile(filePath, out data);
            }
        }

        private void DebugInputSplats(NativeArray<InputSplatData> inputSplats)

        {
            Debug.Log($"InputSplat count {inputSplats.Length}:");
            for (int i = 0; i < 100; i++)
            {
                InputSplatData splat = inputSplats[i];
                Debug.Log($"InputSplat {i}:");
                Debug.Log($"  Position: {splat.pos}, type {splat.pos[0].GetType()}");
            }
        }

        //GetPositions
        //Generate Sch
        float3[] GenerateRandomSHs(int numPts)
        {
            float3[] shs = new float3[numPts];
            for (int i = 0; i < numPts; i++)
            {
                shs[i] = new float3(
                    UnityEngine.Random.value / 255.0f,
                    UnityEngine.Random.value / 255.0f,
                    UnityEngine.Random.value / 255.0f
                );
            }
            return shs;
        }

        List<Vector3> GenerateNormals(int numPts)
        {
            List<Vector3> normals = new List<Vector3>(numPts);
            for (int i = 0; i < numPts; i++)
            {
                normals.Add(Vector3.zero);
            }

            Debug.Log($"Finished generating normals. Total normals created: {normals.Count}");
            return normals;
        }


        public static Vector3[] GetMeshVerticesCosinus(GameObject gameObject)
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

            // Get the original vertices and triangles
            Vector3[] vertices = mesh.vertices;
            int[] triangles = mesh.triangles;

            // Ensure there are at least 3 vertices and 1 triangle
            if (vertices.Length < 3 || triangles.Length < 3)
            {
                Debug.LogError("Mesh does not have enough vertices or triangles.");
                return null;
            }

            // Get the indices of the first triangle
            int v0Index = triangles[3];
            int v1Index = triangles[4];
            int v2Index = triangles[5];

            // Get the vertices of the first triangle
            Vector3 v0 = vertices[v0Index];
            Vector3 v1 = vertices[v1Index];
            Vector3 v2 = vertices[v2Index];

            // Compute the edge vectors
            Vector3 edge0 = v1 - v0;
            Vector3 edge1 = v2 - v0;
            Vector3 edge2 = v2 - v1;

            // Normalize the edge vectors
            Vector3 edge0Normalized = edge0.normalized;
            Vector3 edge1Normalized = edge1.normalized;
            Vector3 edge2Normalized = edge2.normalized;

            // Calculate the cosines of the angles using the dot product
            float cosAngle0 = Vector3.Dot(edge0Normalized, edge1Normalized);
            float cosAngle1 = Vector3.Dot(-edge0Normalized, edge2Normalized);
            float cosAngle2 = Vector3.Dot(-edge1Normalized, -edge2Normalized);

            // Print the cosines of the angles
            Debug.Log($"Cosine of angle at v0: {cosAngle0}");
            Debug.Log($"Cosine of angle at v1: {cosAngle1}");
            Debug.Log($"Cosine of angle at v2: {cosAngle2}");

            // Return the vertices of the first triangle (optional)
            return new Vector3[] { v0, v1, v2 };
        }

        public static List<Vector3> GetMeshFaceVertices(GameObject gameObject)
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


            // Vector3[] vertices = mesh.vertices;
            Vector3[] vertices = TransformVertices(mesh.vertices);

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
        List<Vector3> CalculateXYZ(
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
            return math.max(0, x);// x < 0 ? 0 : x;
        }



        unsafe (List<Quaternion> rotations, List<Vector3> scalings) GenerateRotationsAndScales(List<Vector3> vertices, List<List<float>> scales, int numPtsEachTriangle)
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
                    float x = math.log(ReLU(scaleFactor[0] * s0) + eps_s0);
                    float y = math.log(ReLU(scaleFactor[0] * s1) + eps_s0);
                    float z = math.log(ReLU(scaleFactor[0] * s2) + eps_s0);
                    scalings.Add(new Vector3(x, y, z));
                    //broadcast rotations
                    rotations.Add(rotation);

                }


            }

            return (rotations, scalings);
        }

        private Vector3 ProjectOnto(Vector3 v, Vector3 u)
        {
            return Vector3.Dot(v, u) * u;
        }

        unsafe (List<Quaternion> rotations, List<Vector3> scalings) GenerateRotationsAndScalesV2(List<Vector3> vertices, List<List<float>> scales, int numPtsEachTriangle)
        {
            int numTriangles = vertices.Count / 3;
            float eps_s0 = 1e-8f;
            List<Quaternion> rotations = new List<Quaternion>(numTriangles * numPtsEachTriangle);
            List<Vector3> scalings = new List<Vector3>(numTriangles * numPtsEachTriangle);

            for (int i = 0; i < numTriangles; i++)
            {
                // Extract the three vertices of the current triangle
                Vector3 va = vertices[i * 3];
                Vector3 vb = vertices[i * 3 + 1];
                Vector3 vc = vertices[i * 3 + 2];

                Vector3 normal = Vector3.Cross(vb - va, vc - va);
                Vector3 v0 = normal / (normal.magnitude + eps_s0);
                Vector3 mean = (va + vb + vc) / 3.0f;
                Vector3 v1 = vb - mean;
                var v1_norm = (v1.magnitude + eps_s0);
                v1 = v1 / v1_norm;
                var v2_init = vb - mean;
                Vector3 v2 = (v2_init - ProjectOnto(v2_init, v0) - ProjectOnto(v2_init, v1));
                v2 = v2 / (v2.magnitude + eps_s0);

                float s1 = v1_norm / 2.0f;
                float s2 = Vector3.Dot(v2_init, v2) / 2.0f;
                float s0 = eps_s0;



                for (int j = 0; j < numPtsEachTriangle; j++)
                {
                    List<float> scaleFactor = scales[i * 5 + j];
                    float x = math.log(ReLU(scaleFactor[0] * s0) + eps_s0);
                    float y = math.log(ReLU(scaleFactor[0] * s1) + eps_s0);
                    float z = math.log(ReLU(scaleFactor[0] * s2) + eps_s0);
                    scalings.Add(new Vector3(x, y, z));


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
            // Debug.Log($"Finished generating colors. Total colors created: {colors.Count}");
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



        unsafe NativeArray<InputSplatData> CreateSplatDataFromMemory(List<List<List<float>>> alphas, List<List<float>> scales, GameObject gameObject, int maxShDegree)
        {
            int numOfSplatsPerFace = alphas[0][0].Count;
            var faceVertices = GetMeshFaceVertices(gameObject);
            List<Vector3> positions = CalculateXYZ(faceVertices, numOfSplatsPerFace, alphas);
            List<Vector3> normals = GenerateNormals(positions.Count);
            List<Vector3> colors = SH2RGB(GenerateRandomColors(positions.Count));
            List<List<float[]>> features = CreateFeatures(colors, maxShDegree);
            var (rotations, scalings) = GenerateRotationsAndScales(faceVertices, scales, numOfSplatsPerFace);
            Debug.Log($"First Splat rotation from CreateSplatDataFromMemory: {alphas[0].Count}");


            NativeArray<InputSplatData> data = new NativeArray<InputSplatData>(positions.Count, Allocator.Persistent);

            for (int i = 0; i < rotations.Count; i++)
            {
                data[i] = new InputSplatData
                {
                    pos = positions[i],
                    nor = normals[i],
                    scale = scalings[i],
                    dc0 = new Vector3(features[i][0][0], features[i][1][0], features[i][2][0]),
                    opacity = colors[i].x,
                    rot = rotations[i],
                    // SH coefficients
                    sh1 = new Vector3(features[i][0][1], features[i][1][1], features[i][2][1]),
                    sh2 = new Vector3(features[i][0][2], features[i][1][2], features[i][2][2]),
                    sh3 = new Vector3(features[i][0][3], features[i][1][3], features[i][2][3]),
                    sh4 = new Vector3(features[i][0][4], features[i][1][4], features[i][2][4]),
                    sh5 = new Vector3(features[i][0][5], features[i][1][5], features[i][2][5]),
                    sh6 = new Vector3(features[i][0][6], features[i][1][6], features[i][2][6]),
                    sh7 = new Vector3(features[i][0][7], features[i][1][7], features[i][2][7]),
                    sh8 = new Vector3(features[i][0][8], features[i][1][8], features[i][2][8]),
                    sh9 = new Vector3(features[i][0][9], features[i][1][9], features[i][2][9]),
                    shA = new Vector3(features[i][0][10], features[i][1][10], features[i][2][10]),
                    shB = new Vector3(features[i][0][11], features[i][1][11], features[i][2][11]),
                    shC = new Vector3(features[i][0][12], features[i][1][12], features[i][2][12]),
                    shD = new Vector3(features[i][0][13], features[i][1][13], features[i][2][13]),
                    shE = new Vector3(features[i][0][14], features[i][1][14], features[i][2][14]),
                    shF = new Vector3(features[i][0][15], features[i][1][15], features[i][2][15]),


                };
            }

            return data;

        }

        unsafe NativeArray<InputSplatData> LoadPLYSplatFile(string plyPath)
        {
            NativeArray<InputSplatData> data = default;
            if (!File.Exists(plyPath))
            {
                m_ErrorMessage = $"Did not find {plyPath} file";
                return data;
            }

            int splatCount;
            int vertexStride;
            NativeArray<byte> verticesRawData;
            try
            {
                PLYFileReader.ReadFile(plyPath, out splatCount, out vertexStride, out _, out verticesRawData);
            }
            catch (Exception ex)
            {
                m_ErrorMessage = ex.Message;
                return data;
            }

            if (UnsafeUtility.SizeOf<InputSplatData>() != vertexStride)
            {
                m_ErrorMessage = $"PLY vertex size mismatch, expected {UnsafeUtility.SizeOf<InputSplatData>()} but file has {vertexStride}";
                return data;
            }

            // reorder SHs
            NativeArray<float> floatData = verticesRawData.Reinterpret<float>(1);
            ReorderSHs(splatCount, (float*)floatData.GetUnsafePtr());

            return verticesRawData.Reinterpret<InputSplatData>(1);
        }

        [BurstCompile]
        static unsafe void ReorderSHs(int splatCount, float* data)
        {
            int splatStride = UnsafeUtility.SizeOf<InputSplatData>() / 4;
            int shStartOffset = 9, shCount = 15;
            float* tmp = stackalloc float[shCount * 3];
            int idx = shStartOffset;
            for (int i = 0; i < splatCount; ++i)
            {
                for (int j = 0; j < shCount; ++j)
                {
                    tmp[j * 3 + 0] = data[idx + j];
                    tmp[j * 3 + 1] = data[idx + j + shCount];
                    tmp[j * 3 + 2] = data[idx + j + shCount * 2];
                }

                for (int j = 0; j < shCount * 3; ++j)
                {
                    data[idx + j] = tmp[j];
                }

                idx += splatStride;
            }
        }

        [BurstCompile]
        struct CalcBoundsJob : IJob
        {
            [NativeDisableUnsafePtrRestriction] public unsafe float3* m_BoundsMin;
            [NativeDisableUnsafePtrRestriction] public unsafe float3* m_BoundsMax;
            [ReadOnly] public NativeArray<InputSplatData> m_SplatData;

            public unsafe void Execute()
            {
                float3 boundsMin = float.PositiveInfinity;
                float3 boundsMax = float.NegativeInfinity;

                for (int i = 0; i < m_SplatData.Length; ++i)
                {
                    float3 pos = m_SplatData[i].pos;
                    boundsMin = math.min(boundsMin, pos);
                    boundsMax = math.max(boundsMax, pos);
                }
                *m_BoundsMin = boundsMin;
                *m_BoundsMax = boundsMax;
            }
        }

        [BurstCompile]
        struct ReorderMortonJob : IJobParallelFor
        {
            const float kScaler = (float)((1 << 21) - 1);
            public float3 m_BoundsMin;
            public float3 m_InvBoundsSize;
            [ReadOnly] public NativeArray<InputSplatData> m_SplatData;
            public NativeArray<(ulong, int)> m_Order;

            public void Execute(int index)
            {
                float3 pos = ((float3)m_SplatData[index].pos - m_BoundsMin) * m_InvBoundsSize * kScaler;
                uint3 ipos = (uint3)pos;
                ulong code = GaussianUtils.MortonEncode3(ipos);
                m_Order[index] = (code, index);
            }
        }

        struct OrderComparer : IComparer<(ulong, int)>
        {
            public int Compare((ulong, int) a, (ulong, int) b)
            {
                if (a.Item1 < b.Item1) return -1;
                if (a.Item1 > b.Item1) return +1;
                return a.Item2 - b.Item2;
            }
        }

        static void ReorderMorton(NativeArray<InputSplatData> splatData, float3 boundsMin, float3 boundsMax)
        {
            ReorderMortonJob order = new ReorderMortonJob
            {
                m_SplatData = splatData,
                m_BoundsMin = boundsMin,
                m_InvBoundsSize = 1.0f / (boundsMax - boundsMin),
                m_Order = new NativeArray<(ulong, int)>(splatData.Length, Allocator.TempJob)
            };
            order.Schedule(splatData.Length, 4096).Complete();
            order.m_Order.Sort(new OrderComparer());

            NativeArray<InputSplatData> copy = new(order.m_SplatData, Allocator.TempJob);
            for (int i = 0; i < copy.Length; ++i)
                order.m_SplatData[i] = copy[order.m_Order[i].Item2];
            copy.Dispose();

            order.m_Order.Dispose();
        }

        [BurstCompile]
        static unsafe void GatherSHs(int splatCount, InputSplatData* splatData, float* shData)
        {
            for (int i = 0; i < splatCount; ++i)
            {
                UnsafeUtility.MemCpy(shData, ((float*)splatData) + 9, 15 * 3 * sizeof(float));
                splatData++;
                shData += 15 * 3;
            }
        }

        [BurstCompile]
        struct ConvertSHClustersJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float3> m_Input;
            public NativeArray<GaussianSplatAsset.SHTableItemFloat16> m_Output;
            public void Execute(int index)
            {
                var addr = index * 15;
                GaussianSplatAsset.SHTableItemFloat16 res;
                res.sh1 = new half3(m_Input[addr + 0]);
                res.sh2 = new half3(m_Input[addr + 1]);
                res.sh3 = new half3(m_Input[addr + 2]);
                res.sh4 = new half3(m_Input[addr + 3]);
                res.sh5 = new half3(m_Input[addr + 4]);
                res.sh6 = new half3(m_Input[addr + 5]);
                res.sh7 = new half3(m_Input[addr + 6]);
                res.sh8 = new half3(m_Input[addr + 7]);
                res.sh9 = new half3(m_Input[addr + 8]);
                res.shA = new half3(m_Input[addr + 9]);
                res.shB = new half3(m_Input[addr + 10]);
                res.shC = new half3(m_Input[addr + 11]);
                res.shD = new half3(m_Input[addr + 12]);
                res.shE = new half3(m_Input[addr + 13]);
                res.shF = new half3(m_Input[addr + 14]);
                res.shPadding = default;
                m_Output[index] = res;
            }
        }
        static bool ClusterSHProgress(float val)
        {
            EditorUtility.DisplayProgressBar(kProgressTitle, $"Cluster SHs ({val:P0})", 0.2f + val * 0.5f);
            return true;
        }

        static unsafe void ClusterSHs(NativeArray<InputSplatData> splatData, GaussianSplatAsset.SHFormat format, out NativeArray<GaussianSplatAsset.SHTableItemFloat16> shs, out NativeArray<int> shIndices)
        {
            shs = default;
            shIndices = default;

            int shCount = GaussianSplatAsset.GetSHCount(format, splatData.Length);
            if (shCount >= splatData.Length) // no need to cluster, just use raw data
                return;

            const int kShDim = 15 * 3;
            const int kBatchSize = 2048;
            float passesOverData = format switch
            {
                GaussianSplatAsset.SHFormat.Cluster64k => 0.3f,
                GaussianSplatAsset.SHFormat.Cluster32k => 0.4f,
                GaussianSplatAsset.SHFormat.Cluster16k => 0.5f,
                GaussianSplatAsset.SHFormat.Cluster8k => 0.8f,
                GaussianSplatAsset.SHFormat.Cluster4k => 1.2f,
                _ => throw new ArgumentOutOfRangeException(nameof(format), format, null)
            };

            float t0 = Time.realtimeSinceStartup;
            NativeArray<float> shData = new(splatData.Length * kShDim, Allocator.Persistent);
            GatherSHs(splatData.Length, (InputSplatData*)splatData.GetUnsafeReadOnlyPtr(), (float*)shData.GetUnsafePtr());

            NativeArray<float> shMeans = new(shCount * kShDim, Allocator.Persistent);
            shIndices = new(splatData.Length, Allocator.Persistent);

            KMeansClustering.Calculate(kShDim, shData, kBatchSize, passesOverData, ClusterSHProgress, shMeans, shIndices);
            shData.Dispose();

            shs = new NativeArray<GaussianSplatAsset.SHTableItemFloat16>(shCount, Allocator.Persistent);

            ConvertSHClustersJob job = new ConvertSHClustersJob
            {
                m_Input = shMeans.Reinterpret<float3>(4),
                m_Output = shs
            };
            job.Schedule(shCount, 256).Complete();
            shMeans.Dispose();
            float t1 = Time.realtimeSinceStartup;
            Debug.Log($"GS: clustered {splatData.Length / 1000000.0:F2}M SHs into {shCount / 1024}K ({passesOverData:F1}pass/{kBatchSize}batch) in {t1 - t0:F0}s");
        }

        [BurstCompile]
        struct LinearizeDataJob : IJobParallelFor
        {
            public NativeArray<InputSplatData> splatData;
            public void Execute(int index)
            {
                var splat = splatData[index];

                // rot
                var q = splat.rot;
                var qq = GaussianUtils.NormalizeSwizzleRotation(new float4(q.x, q.y, q.z, q.w));
                qq = GaussianUtils.PackSmallest3Rotation(qq);
                splat.rot = new Quaternion(qq.x, qq.y, qq.z, qq.w);

                // scale
                splat.scale = GaussianUtils.LinearScale(splat.scale);

                // color
                splat.dc0 = GaussianUtils.SH0ToColor(splat.dc0);
                splat.opacity = GaussianUtils.Sigmoid(splat.opacity);

                splatData[index] = splat;
            }
        }

        static void LinearizeData(NativeArray<InputSplatData> splatData)
        {
            LinearizeDataJob job = new LinearizeDataJob();
            job.splatData = splatData;
            job.Schedule(splatData.Length, 4096).Complete();
        }

        [BurstCompile]
        struct CalcChunkDataJob : IJobParallelFor
        {
            [NativeDisableParallelForRestriction] public NativeArray<InputSplatData> splatData;
            public NativeArray<GaussianSplatAsset.ChunkInfo> chunks;

            public void Execute(int chunkIdx)
            {
                float3 chunkMinpos = float.PositiveInfinity;
                float3 chunkMinscl = float.PositiveInfinity;
                float4 chunkMincol = float.PositiveInfinity;
                float3 chunkMinshs = float.PositiveInfinity;
                float3 chunkMaxpos = float.NegativeInfinity;
                float3 chunkMaxscl = float.NegativeInfinity;
                float4 chunkMaxcol = float.NegativeInfinity;
                float3 chunkMaxshs = float.NegativeInfinity;

                int splatBegin = math.min(chunkIdx * GaussianSplatAsset.kChunkSize, splatData.Length);
                int splatEnd = math.min((chunkIdx + 1) * GaussianSplatAsset.kChunkSize, splatData.Length);

                // calculate data bounds inside the chunk
                for (int i = splatBegin; i < splatEnd; ++i)
                {
                    InputSplatData s = splatData[i];

                    // transform scale to be more uniformly distributed
                    s.scale = math.pow(s.scale, 1.0f / 8.0f);
                    // transform opacity to be more unformly distributed
                    s.opacity = GaussianUtils.SquareCentered01(s.opacity);
                    splatData[i] = s;

                    chunkMinpos = math.min(chunkMinpos, s.pos);
                    chunkMinscl = math.min(chunkMinscl, s.scale);
                    chunkMincol = math.min(chunkMincol, new float4(s.dc0, s.opacity));
                    chunkMinshs = math.min(chunkMinshs, s.sh1);
                    chunkMinshs = math.min(chunkMinshs, s.sh2);
                    chunkMinshs = math.min(chunkMinshs, s.sh3);
                    chunkMinshs = math.min(chunkMinshs, s.sh4);
                    chunkMinshs = math.min(chunkMinshs, s.sh5);
                    chunkMinshs = math.min(chunkMinshs, s.sh6);
                    chunkMinshs = math.min(chunkMinshs, s.sh7);
                    chunkMinshs = math.min(chunkMinshs, s.sh8);
                    chunkMinshs = math.min(chunkMinshs, s.sh9);
                    chunkMinshs = math.min(chunkMinshs, s.shA);
                    chunkMinshs = math.min(chunkMinshs, s.shB);
                    chunkMinshs = math.min(chunkMinshs, s.shC);
                    chunkMinshs = math.min(chunkMinshs, s.shD);
                    chunkMinshs = math.min(chunkMinshs, s.shE);
                    chunkMinshs = math.min(chunkMinshs, s.shF);

                    chunkMaxpos = math.max(chunkMaxpos, s.pos);
                    chunkMaxscl = math.max(chunkMaxscl, s.scale);
                    chunkMaxcol = math.max(chunkMaxcol, new float4(s.dc0, s.opacity));
                    chunkMaxshs = math.max(chunkMaxshs, s.sh1);
                    chunkMaxshs = math.max(chunkMaxshs, s.sh2);
                    chunkMaxshs = math.max(chunkMaxshs, s.sh3);
                    chunkMaxshs = math.max(chunkMaxshs, s.sh4);
                    chunkMaxshs = math.max(chunkMaxshs, s.sh5);
                    chunkMaxshs = math.max(chunkMaxshs, s.sh6);
                    chunkMaxshs = math.max(chunkMaxshs, s.sh7);
                    chunkMaxshs = math.max(chunkMaxshs, s.sh8);
                    chunkMaxshs = math.max(chunkMaxshs, s.sh9);
                    chunkMaxshs = math.max(chunkMaxshs, s.shA);
                    chunkMaxshs = math.max(chunkMaxshs, s.shB);
                    chunkMaxshs = math.max(chunkMaxshs, s.shC);
                    chunkMaxshs = math.max(chunkMaxshs, s.shD);
                    chunkMaxshs = math.max(chunkMaxshs, s.shE);
                    chunkMaxshs = math.max(chunkMaxshs, s.shF);
                }

                // make sure bounds are not zero
                chunkMaxpos = math.max(chunkMaxpos, chunkMinpos + 1.0e-5f);
                chunkMaxscl = math.max(chunkMaxscl, chunkMinscl + 1.0e-5f);
                chunkMaxcol = math.max(chunkMaxcol, chunkMincol + 1.0e-5f);
                chunkMaxshs = math.max(chunkMaxshs, chunkMinshs + 1.0e-5f);

                // store chunk info
                GaussianSplatAsset.ChunkInfo info = default;
                info.posX = new float2(chunkMinpos.x, chunkMaxpos.x);
                info.posY = new float2(chunkMinpos.y, chunkMaxpos.y);
                info.posZ = new float2(chunkMinpos.z, chunkMaxpos.z);
                info.sclX = math.f32tof16(chunkMinscl.x) | (math.f32tof16(chunkMaxscl.x) << 16);
                info.sclY = math.f32tof16(chunkMinscl.y) | (math.f32tof16(chunkMaxscl.y) << 16);
                info.sclZ = math.f32tof16(chunkMinscl.z) | (math.f32tof16(chunkMaxscl.z) << 16);
                info.colR = math.f32tof16(chunkMincol.x) | (math.f32tof16(chunkMaxcol.x) << 16);
                info.colG = math.f32tof16(chunkMincol.y) | (math.f32tof16(chunkMaxcol.y) << 16);
                info.colB = math.f32tof16(chunkMincol.z) | (math.f32tof16(chunkMaxcol.z) << 16);
                info.colA = math.f32tof16(chunkMincol.w) | (math.f32tof16(chunkMaxcol.w) << 16);
                info.shR = math.f32tof16(chunkMinshs.x) | (math.f32tof16(chunkMaxshs.x) << 16);
                info.shG = math.f32tof16(chunkMinshs.y) | (math.f32tof16(chunkMaxshs.y) << 16);
                info.shB = math.f32tof16(chunkMinshs.z) | (math.f32tof16(chunkMaxshs.z) << 16);
                chunks[chunkIdx] = info;

                // adjust data to be 0..1 within chunk bounds
                for (int i = splatBegin; i < splatEnd; ++i)
                {
                    InputSplatData s = splatData[i];
                    s.pos = ((float3)s.pos - chunkMinpos) / (chunkMaxpos - chunkMinpos);
                    s.scale = ((float3)s.scale - chunkMinscl) / (chunkMaxscl - chunkMinscl);
                    s.dc0 = ((float3)s.dc0 - chunkMincol.xyz) / (chunkMaxcol.xyz - chunkMincol.xyz);
                    s.opacity = (s.opacity - chunkMincol.w) / (chunkMaxcol.w - chunkMincol.w);
                    s.sh1 = ((float3)s.sh1 - chunkMinshs) / (chunkMaxshs - chunkMinshs);
                    s.sh2 = ((float3)s.sh2 - chunkMinshs) / (chunkMaxshs - chunkMinshs);
                    s.sh3 = ((float3)s.sh3 - chunkMinshs) / (chunkMaxshs - chunkMinshs);
                    s.sh4 = ((float3)s.sh4 - chunkMinshs) / (chunkMaxshs - chunkMinshs);
                    s.sh5 = ((float3)s.sh5 - chunkMinshs) / (chunkMaxshs - chunkMinshs);
                    s.sh6 = ((float3)s.sh6 - chunkMinshs) / (chunkMaxshs - chunkMinshs);
                    s.sh7 = ((float3)s.sh7 - chunkMinshs) / (chunkMaxshs - chunkMinshs);
                    s.sh8 = ((float3)s.sh8 - chunkMinshs) / (chunkMaxshs - chunkMinshs);
                    s.sh9 = ((float3)s.sh9 - chunkMinshs) / (chunkMaxshs - chunkMinshs);
                    s.shA = ((float3)s.shA - chunkMinshs) / (chunkMaxshs - chunkMinshs);
                    s.shB = ((float3)s.shB - chunkMinshs) / (chunkMaxshs - chunkMinshs);
                    s.shC = ((float3)s.shC - chunkMinshs) / (chunkMaxshs - chunkMinshs);
                    s.shD = ((float3)s.shD - chunkMinshs) / (chunkMaxshs - chunkMinshs);
                    s.shE = ((float3)s.shE - chunkMinshs) / (chunkMaxshs - chunkMinshs);
                    s.shF = ((float3)s.shF - chunkMinshs) / (chunkMaxshs - chunkMinshs);
                    splatData[i] = s;
                }
            }
        }

        static void CreateChunkData(NativeArray<InputSplatData> splatData, string filePath, ref Hash128 dataHash)
        {
            int chunkCount = (splatData.Length + GaussianSplatAsset.kChunkSize - 1) / GaussianSplatAsset.kChunkSize;
            CalcChunkDataJob job = new CalcChunkDataJob
            {
                splatData = splatData,
                chunks = new(chunkCount, Allocator.TempJob),
            };

            job.Schedule(chunkCount, 8).Complete();

            dataHash.Append(ref job.chunks);

            using var fs = new FileStream(filePath, FileMode.Create, FileAccess.Write);
            fs.Write(job.chunks.Reinterpret<byte>(UnsafeUtility.SizeOf<GaussianSplatAsset.ChunkInfo>()));

            job.chunks.Dispose();
        }

        [BurstCompile]
        struct ConvertColorJob : IJobParallelFor
        {
            public int width, height;
            [ReadOnly] public NativeArray<float4> inputData;
            [NativeDisableParallelForRestriction] public NativeArray<byte> outputData;
            public GaussianSplatAsset.ColorFormat format;
            public int formatBytesPerPixel;

            public unsafe void Execute(int y)
            {
                int srcIdx = y * width;
                byte* dstPtr = (byte*)outputData.GetUnsafePtr() + y * width * formatBytesPerPixel;
                for (int x = 0; x < width; ++x)
                {
                    float4 pix = inputData[srcIdx];

                    switch (format)
                    {
                        case GaussianSplatAsset.ColorFormat.Float32x4:
                            {
                                *(float4*)dstPtr = pix;
                            }
                            break;
                        case GaussianSplatAsset.ColorFormat.Float16x4:
                            {
                                half4 enc = new half4(pix);
                                *(half4*)dstPtr = enc;
                            }
                            break;
                        case GaussianSplatAsset.ColorFormat.Norm8x4:
                            {
                                pix = math.saturate(pix);
                                uint enc = (uint)(pix.x * 255.5f) | ((uint)(pix.y * 255.5f) << 8) | ((uint)(pix.z * 255.5f) << 16) | ((uint)(pix.w * 255.5f) << 24);
                                *(uint*)dstPtr = enc;
                            }
                            break;
                    }

                    srcIdx++;
                    dstPtr += formatBytesPerPixel;
                }
            }
        }

        static ulong EncodeFloat3ToNorm16(float3 v) // 48 bits: 16.16.16
        {
            return (ulong)(v.x * 65535.5f) | ((ulong)(v.y * 65535.5f) << 16) | ((ulong)(v.z * 65535.5f) << 32);
        }
        static uint EncodeFloat3ToNorm11(float3 v) // 32 bits: 11.10.11
        {
            return (uint)(v.x * 2047.5f) | ((uint)(v.y * 1023.5f) << 11) | ((uint)(v.z * 2047.5f) << 21);
        }
        static ushort EncodeFloat3ToNorm655(float3 v) // 16 bits: 6.5.5
        {
            return (ushort)((uint)(v.x * 63.5f) | ((uint)(v.y * 31.5f) << 6) | ((uint)(v.z * 31.5f) << 11));
        }
        static ushort EncodeFloat3ToNorm565(float3 v) // 16 bits: 5.6.5
        {
            return (ushort)((uint)(v.x * 31.5f) | ((uint)(v.y * 63.5f) << 5) | ((uint)(v.z * 31.5f) << 11));
        }

        static uint EncodeQuatToNorm10(float4 v) // 32 bits: 10.10.10.2
        {
            return (uint)(v.x * 1023.5f) | ((uint)(v.y * 1023.5f) << 10) | ((uint)(v.z * 1023.5f) << 20) | ((uint)(v.w * 3.5f) << 30);
        }

        static unsafe void EmitEncodedVector(float3 v, byte* outputPtr, GaussianSplatAsset.VectorFormat format)
        {
            switch (format)
            {
                case GaussianSplatAsset.VectorFormat.Float32:
                    {
                        *(float*)outputPtr = v.x;
                        *(float*)(outputPtr + 4) = v.y;
                        *(float*)(outputPtr + 8) = v.z;
                    }
                    break;
                case GaussianSplatAsset.VectorFormat.Norm16:
                    {
                        ulong enc = EncodeFloat3ToNorm16(math.saturate(v));
                        *(uint*)outputPtr = (uint)enc;
                        *(ushort*)(outputPtr + 4) = (ushort)(enc >> 32);
                    }
                    break;
                case GaussianSplatAsset.VectorFormat.Norm11:
                    {
                        uint enc = EncodeFloat3ToNorm11(math.saturate(v));
                        *(uint*)outputPtr = enc;
                    }
                    break;
                case GaussianSplatAsset.VectorFormat.Norm6:
                    {
                        ushort enc = EncodeFloat3ToNorm655(math.saturate(v));
                        *(ushort*)outputPtr = enc;
                    }
                    break;
            }
        }

        static unsafe void EmitEncodedFloat(float v, byte* outputPtr)
        {
            *(float*)outputPtr = v;
        }

        [BurstCompile]
        struct CreatePositionsDataJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<InputSplatData> m_Input;
            public GaussianSplatAsset.VectorFormat m_Format;
            public int m_FormatSize;
            [NativeDisableParallelForRestriction] public NativeArray<byte> m_Output;

            public unsafe void Execute(int index)
            {
                byte* outputPtr = (byte*)m_Output.GetUnsafePtr() + index * m_FormatSize;
                EmitEncodedVector(m_Input[index].pos, outputPtr, m_Format);
            }
        }

        [BurstCompile]
        struct CreateOtherDataJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<InputSplatData> m_Input;
            [NativeDisableContainerSafetyRestriction][ReadOnly] public NativeArray<int> m_SplatSHIndices;
            public GaussianSplatAsset.VectorFormat m_ScaleFormat;
            public int m_FormatSize;
            [NativeDisableParallelForRestriction] public NativeArray<byte> m_Output;

            public unsafe void Execute(int index)
            {
                byte* outputPtr = (byte*)m_Output.GetUnsafePtr() + index * m_FormatSize;

                // rotation: 4 bytes
                {
                    Quaternion rotQ = m_Input[index].rot;
                    float4 rot = new float4(rotQ.x, rotQ.y, rotQ.z, rotQ.w);
                    uint enc = EncodeQuatToNorm10(rot);
                    *(uint*)outputPtr = enc;
                    outputPtr += 4;
                }

                // scale: 6, 4 or 2 bytes
                EmitEncodedVector(m_Input[index].scale, outputPtr, m_ScaleFormat);
                outputPtr += GaussianSplatAsset.GetVectorSize(m_ScaleFormat);

                // SH index
                if (m_SplatSHIndices.IsCreated)
                    *(ushort*)outputPtr = (ushort)m_SplatSHIndices[index];
            }
        }

        static int NextMultipleOf(int size, int multipleOf)
        {
            return (size + multipleOf - 1) / multipleOf * multipleOf;
        }

        void CreatePositionsData(NativeArray<InputSplatData> inputSplats, string filePath, ref Hash128 dataHash)
        {
            int dataLen = inputSplats.Length * GaussianSplatAsset.GetVectorSize(m_FormatPos);
            dataLen = NextMultipleOf(dataLen, 8); // serialized as ulong
            NativeArray<byte> data = new(dataLen, Allocator.TempJob);

            CreatePositionsDataJob job = new CreatePositionsDataJob
            {
                m_Input = inputSplats,
                m_Format = m_FormatPos,
                m_FormatSize = GaussianSplatAsset.GetVectorSize(m_FormatPos),
                m_Output = data
            };
            job.Schedule(inputSplats.Length, 8192).Complete();

            Debug.Log($"  New Encoded  from creat asset   : {data[0]}, Decoded: {m_FormatPos}");
            dataHash.Append(data);

            using var fs = new FileStream(filePath, FileMode.Create, FileAccess.Write);
            fs.Write(data);

            data.Dispose();
        }

        void CreateOtherData(NativeArray<InputSplatData> inputSplats, string filePath, ref Hash128 dataHash, NativeArray<int> splatSHIndices)
        {
            int formatSize = GaussianSplatAsset.GetOtherSizeNoSHIndex(m_FormatScale);
            if (splatSHIndices.IsCreated)
                formatSize += 2;
            int dataLen = inputSplats.Length * formatSize;

            dataLen = NextMultipleOf(dataLen, 8); // serialized as ulong
            NativeArray<byte> data = new(dataLen, Allocator.TempJob);

            CreateOtherDataJob job = new CreateOtherDataJob
            {
                m_Input = inputSplats,
                m_SplatSHIndices = splatSHIndices,
                m_ScaleFormat = m_FormatScale,
                m_FormatSize = formatSize,
                m_Output = data
            };
            job.Schedule(inputSplats.Length, 8192).Complete();

            dataHash.Append(data);

            using var fs = new FileStream(filePath, FileMode.Create, FileAccess.Write);
            fs.Write(data);

            data.Dispose();
        }

        static int SplatIndexToTextureIndex(uint idx)
        {
            uint2 xy = GaussianUtils.DecodeMorton2D_16x16(idx);
            uint width = GaussianSplatAsset.kTextureWidth / 16;
            idx >>= 8;
            uint x = (idx % width) * 16 + xy.x;
            uint y = (idx / width) * 16 + xy.y;
            return (int)(y * GaussianSplatAsset.kTextureWidth + x);
        }

        [BurstCompile]
        struct CreateColorDataJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<InputSplatData> m_Input;
            [NativeDisableParallelForRestriction] public NativeArray<float4> m_Output;

            public void Execute(int index)
            {
                var splat = m_Input[index];
                int i = SplatIndexToTextureIndex((uint)index);
                m_Output[i] = new float4(splat.dc0.x, splat.dc0.y, splat.dc0.z, splat.opacity);
            }
        }

        void CreateColorData(NativeArray<InputSplatData> inputSplats, string filePath, ref Hash128 dataHash)
        {
            var (width, height) = GaussianSplatAsset.CalcTextureSize(inputSplats.Length);
            NativeArray<float4> data = new(width * height, Allocator.TempJob);

            CreateColorDataJob job = new CreateColorDataJob();
            job.m_Input = inputSplats;
            job.m_Output = data;
            job.Schedule(inputSplats.Length, 8192).Complete();

            dataHash.Append(data);
            dataHash.Append((int)m_FormatColor);

            GraphicsFormat gfxFormat = GaussianSplatAsset.ColorFormatToGraphics(m_FormatColor);
            int dstSize = (int)GraphicsFormatUtility.ComputeMipmapSize(width, height, gfxFormat);

            if (GraphicsFormatUtility.IsCompressedFormat(gfxFormat))
            {
                Texture2D tex = new Texture2D(width, height, GraphicsFormat.R32G32B32A32_SFloat, TextureCreationFlags.DontInitializePixels | TextureCreationFlags.DontUploadUponCreate);
                tex.SetPixelData(data, 0);
                EditorUtility.CompressTexture(tex, GraphicsFormatUtility.GetTextureFormat(gfxFormat), 100);
                NativeArray<byte> cmpData = tex.GetPixelData<byte>(0);
                using var fs = new FileStream(filePath, FileMode.Create, FileAccess.Write);
                fs.Write(cmpData);

                DestroyImmediate(tex);
            }
            else
            {
                ConvertColorJob jobConvert = new ConvertColorJob
                {
                    width = width,
                    height = height,
                    inputData = data,
                    format = m_FormatColor,
                    outputData = new NativeArray<byte>(dstSize, Allocator.TempJob),
                    formatBytesPerPixel = dstSize / width / height
                };
                jobConvert.Schedule(height, 1).Complete();
                using var fs = new FileStream(filePath, FileMode.Create, FileAccess.Write);
                fs.Write(jobConvert.outputData);
                jobConvert.outputData.Dispose();
            }

            data.Dispose();
        }

        [BurstCompile]
        struct CreateSHDataJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<InputSplatData> m_Input;
            public GaussianSplatAsset.SHFormat m_Format;
            public NativeArray<byte> m_Output;
            public unsafe void Execute(int index)
            {
                var splat = m_Input[index];

                switch (m_Format)
                {
                    case GaussianSplatAsset.SHFormat.Float32:
                        {
                            GaussianSplatAsset.SHTableItemFloat32 res;
                            res.sh1 = splat.sh1;
                            res.sh2 = splat.sh2;
                            res.sh3 = splat.sh3;
                            res.sh4 = splat.sh4;
                            res.sh5 = splat.sh5;
                            res.sh6 = splat.sh6;
                            res.sh7 = splat.sh7;
                            res.sh8 = splat.sh8;
                            res.sh9 = splat.sh9;
                            res.shA = splat.shA;
                            res.shB = splat.shB;
                            res.shC = splat.shC;
                            res.shD = splat.shD;
                            res.shE = splat.shE;
                            res.shF = splat.shF;
                            res.shPadding = default;
                            ((GaussianSplatAsset.SHTableItemFloat32*)m_Output.GetUnsafePtr())[index] = res;
                        }
                        break;
                    case GaussianSplatAsset.SHFormat.Float16:
                        {
                            GaussianSplatAsset.SHTableItemFloat16 res;
                            res.sh1 = new half3(splat.sh1);
                            res.sh2 = new half3(splat.sh2);
                            res.sh3 = new half3(splat.sh3);
                            res.sh4 = new half3(splat.sh4);
                            res.sh5 = new half3(splat.sh5);
                            res.sh6 = new half3(splat.sh6);
                            res.sh7 = new half3(splat.sh7);
                            res.sh8 = new half3(splat.sh8);
                            res.sh9 = new half3(splat.sh9);
                            res.shA = new half3(splat.shA);
                            res.shB = new half3(splat.shB);
                            res.shC = new half3(splat.shC);
                            res.shD = new half3(splat.shD);
                            res.shE = new half3(splat.shE);
                            res.shF = new half3(splat.shF);
                            res.shPadding = default;
                            ((GaussianSplatAsset.SHTableItemFloat16*)m_Output.GetUnsafePtr())[index] = res;
                        }
                        break;
                    case GaussianSplatAsset.SHFormat.Norm11:
                        {
                            GaussianSplatAsset.SHTableItemNorm11 res;
                            res.sh1 = EncodeFloat3ToNorm11(splat.sh1);
                            res.sh2 = EncodeFloat3ToNorm11(splat.sh2);
                            res.sh3 = EncodeFloat3ToNorm11(splat.sh3);
                            res.sh4 = EncodeFloat3ToNorm11(splat.sh4);
                            res.sh5 = EncodeFloat3ToNorm11(splat.sh5);
                            res.sh6 = EncodeFloat3ToNorm11(splat.sh6);
                            res.sh7 = EncodeFloat3ToNorm11(splat.sh7);
                            res.sh8 = EncodeFloat3ToNorm11(splat.sh8);
                            res.sh9 = EncodeFloat3ToNorm11(splat.sh9);
                            res.shA = EncodeFloat3ToNorm11(splat.shA);
                            res.shB = EncodeFloat3ToNorm11(splat.shB);
                            res.shC = EncodeFloat3ToNorm11(splat.shC);
                            res.shD = EncodeFloat3ToNorm11(splat.shD);
                            res.shE = EncodeFloat3ToNorm11(splat.shE);
                            res.shF = EncodeFloat3ToNorm11(splat.shF);
                            ((GaussianSplatAsset.SHTableItemNorm11*)m_Output.GetUnsafePtr())[index] = res;
                        }
                        break;
                    case GaussianSplatAsset.SHFormat.Norm6:
                        {
                            GaussianSplatAsset.SHTableItemNorm6 res;
                            res.sh1 = EncodeFloat3ToNorm565(splat.sh1);
                            res.sh2 = EncodeFloat3ToNorm565(splat.sh2);
                            res.sh3 = EncodeFloat3ToNorm565(splat.sh3);
                            res.sh4 = EncodeFloat3ToNorm565(splat.sh4);
                            res.sh5 = EncodeFloat3ToNorm565(splat.sh5);
                            res.sh6 = EncodeFloat3ToNorm565(splat.sh6);
                            res.sh7 = EncodeFloat3ToNorm565(splat.sh7);
                            res.sh8 = EncodeFloat3ToNorm565(splat.sh8);
                            res.sh9 = EncodeFloat3ToNorm565(splat.sh9);
                            res.shA = EncodeFloat3ToNorm565(splat.shA);
                            res.shB = EncodeFloat3ToNorm565(splat.shB);
                            res.shC = EncodeFloat3ToNorm565(splat.shC);
                            res.shD = EncodeFloat3ToNorm565(splat.shD);
                            res.shE = EncodeFloat3ToNorm565(splat.shE);
                            res.shF = EncodeFloat3ToNorm565(splat.shF);
                            res.shPadding = default;
                            ((GaussianSplatAsset.SHTableItemNorm6*)m_Output.GetUnsafePtr())[index] = res;
                        }
                        break;
                    default:
                        break;
                }
            }
        }

        static void EmitSimpleDataFile<T>(NativeArray<T> data, string filePath, ref Hash128 dataHash) where T : unmanaged
        {
            dataHash.Append(data);
            using var fs = new FileStream(filePath, FileMode.Create, FileAccess.Write);
            fs.Write(data.Reinterpret<byte>(UnsafeUtility.SizeOf<T>()));
        }

        void CreateSHData(NativeArray<InputSplatData> inputSplats, string filePath, ref Hash128 dataHash, NativeArray<GaussianSplatAsset.SHTableItemFloat16> clusteredSHs)
        {
            if (clusteredSHs.IsCreated)
            {
                EmitSimpleDataFile(clusteredSHs, filePath, ref dataHash);
            }
            else
            {
                int dataLen = (int)GaussianSplatAsset.CalcSHDataSize(inputSplats.Length, m_FormatSH);
                NativeArray<byte> data = new(dataLen, Allocator.TempJob);
                CreateSHDataJob job = new CreateSHDataJob
                {
                    m_Input = inputSplats,
                    m_Format = m_FormatSH,
                    m_Output = data
                };
                job.Schedule(inputSplats.Length, 8192).Complete();
                EmitSimpleDataFile(data, filePath, ref dataHash);
                data.Dispose();
            }
        }

        static GaussianSplatAsset.CameraInfo[] LoadJsonCamerasFile(string curPath, bool doImport)
        {
            if (!doImport)
                return null;

            string camerasPath;
            while (true)
            {
                var dir = Path.GetDirectoryName(curPath);
                if (!Directory.Exists(dir))
                    return null;
                camerasPath = $"{dir}/{kCamerasJson}";
                if (File.Exists(camerasPath))
                    break;
                curPath = dir;
            }

            if (!File.Exists(camerasPath))
                return null;

            string json = File.ReadAllText(camerasPath);
            var jsonCameras = JSONParser.FromJson<List<JsonCamera>>(json);
            if (jsonCameras == null || jsonCameras.Count == 0)
                return null;

            var result = new GaussianSplatAsset.CameraInfo[jsonCameras.Count];
            for (var camIndex = 0; camIndex < jsonCameras.Count; camIndex++)
            {
                var jsonCam = jsonCameras[camIndex];
                var pos = new Vector3(jsonCam.position[0], jsonCam.position[1], jsonCam.position[2]);
                // the matrix is a "view matrix", not "camera matrix" lol
                var axisx = new Vector3(jsonCam.rotation[0][0], jsonCam.rotation[1][0], jsonCam.rotation[2][0]);
                var axisy = new Vector3(jsonCam.rotation[0][1], jsonCam.rotation[1][1], jsonCam.rotation[2][1]);
                var axisz = new Vector3(jsonCam.rotation[0][2], jsonCam.rotation[1][2], jsonCam.rotation[2][2]);

                axisy *= -1;
                axisz *= -1;

                var cam = new GaussianSplatAsset.CameraInfo
                {
                    pos = pos,
                    axisX = axisx,
                    axisY = axisy,
                    axisZ = axisz,
                    fov = 25 //@TODO
                };
                result[camIndex] = cam;
            }

            return result;
        }

        [Serializable]
        public class JsonCamera
        {
            public int id;
            public string img_name;
            public int width;
            public int height;
            public float[] position;
            public float[][] rotation;
            public float fx;
            public float fy;
        }
        [Serializable]
        public class ModelParams
        {
            public List<List<List<float>>> _alpha { get; set; }
            public List<List<float>> _scale { get; set; }
        }

    }
}
