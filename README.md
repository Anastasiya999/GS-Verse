# GS-Verse  

This repository hosts the **official implementation** of the paper:  
> **GS-Verse: Mesh-based Gaussian Splatting for Physics-aware Interaction in Virtual Reality**  
> _Authors_: Anastasiya Pechko, Piotr Borycki, Joanna Waczynska, Daniel Barczyk, Agata Szymańska, Sławomir Tadeja, Przemysław Spurek.  
> _2025_  


This project was developed based on an existing [Gaussian Splatting playground in Unity](https://github.com/aras-p/UnityGaussianSplatting).  
For detailed implementation notes and additional context, see the related [README](/projects/readme.md).


## Table of Contents

1. [Abstract](#abstract)  
2. [Prerequisites](#prerequisites)  
3. [Setup](#setup)
4. [Usage](#usage)  
5. [Asset creation (soon)](#asset-creation)
6. [License](#license)  

## Abstract 

As the demand for immersive 3D content grows, the need for intuitive and efficient interaction methods becomes paramount. Current techniques for physically manipulating 3D content within Virtual Reality (VR) often face significant limitations, including reliance on engineering-intensive processes and simplified geometric representations, such as tetrahedral cages, which can compromise visual fidelity and physical accuracy. In this paper, we introduce GS-Verse (Gaussian Splatting for Virtual Environment Rendering and Scene Editing), a novel method designed to overcome these challenges by directly integrating an object's mesh with a Gaussian Splatting (GS) representation. Our approach enables more precise surface approximation, leading to highly realistic deformations and interactions. By leveraging existing 3D mesh assets, GS-Verse facilitates seamless content reuse and simplifies the development workflow.
Moreover, our system is designed to be physics-engine-agnostic, granting developers robust deployment flexibility. This versatile architecture delivers a highly realistic, adaptable, and intuitive approach to interactive 3D manipulation. We rigorously validate our method against the current state-of-the-art technique that couples VR with GS in a comparative user study involving 18 participants. Specifically, we demonstrate that our approach is statistically significantly better for physics-aware stretching manipulation and is also more consistent in other physics-based manipulations like twisting and shaking. Further evaluation across various interactions and scenes confirms that our method consistently delivers high and reliable performance, showing its potential as a plausible alternative to existing methods.

### Prerequisites

For detailed hardware and software requirements, please refer to the [original project README](/projects/readme.md).

This project was developed and tested using the following setup:

- **Unity version:** 2022.3.47  
- **Platforms:** macOS (Metal) / Windows 11 Home  
- **Headset:** Meta Quest Pro  

For the **user study**, the GS systems were deployed on a desktop PC equipped with:  
- **CPU:** Intel® Core™ i7-14700K (3.40 GHz)  
- **Memory:** 32 GB RAM  
- **GPU:** NVIDIA GeForce RTX 4070 SUPER  
- **Operating System:** Windows 11 Home  
- **CUDA Toolkit:** Version 12.4  
  *(Build V12.4.99, cuda_12.4.r12.4/compiler.33961263_0)*  

### Setup

```bash
# Clone or download this repository
git clone https://github.com/Anastasiya999/GS-Verse.git
cd GS-Verse
```
Open `projects/GaussianExample` as a Unity project ( Unity 2022.3 version is used, but other versions might also work). Since the gaussian splat models are quite large, we have not included any in this Github repo. In order to test sample scenes, please download [link to zip on drive]. Place RoomScenes/* content under the Assets/RoomScenes and put Resources folder under Assets/ folder.
Open the project located at projects/GaussianExample in Unity 2022.3 (other Unity 2022 versions may also work).

Open the project located at `projects/GaussianExample` in **Unity 2022.3** (other Unity 2022 versions may also work).

Since the Gaussian Splat models are quite large, they are **not included** in this GitHub repository.
To test the sample scenes, please download the assets from [this link](https://drive.google.com/file/d/1W1AyAVohk6ITPe2Faz6F3SPZgpz2_edP/view?usp=sharing). After downloading, place the contents as follows:

- Extract the contents of `RoomScenes/*` into `Assets/RoomScenes/`
- Place the `Resources` folder under `Assets/`

Your final directory structure should look like this:

```css
Assets/
 ├── RoomScenes/
 │    └── [scene content files]
 └── Resources/
      └── [resource files]
```
⚠️ **Important:**  
Unity performs mesh optimization and vertex modifications when importing `.obj` assets.
Our project relies on the **exact face order**, so please ensure that every asset from the `Resources` folder has the correct import settings in the **Unity Inspector**.  

If not, adjust the settings to match the configuration shown in the following screenshot and click **Apply**:  
<img src="docs/Images/shotMeshInspector.png" alt="Screenshot" width="300"/>

To test the sample scene, open **`RoomScenes/darkRoom/363.unity`**.

- If you are using a **real device**, uncheck **XR Device Simulator** in the scene.  
- If you are using the **simulator**, use the following controller mappings:

  - **Stretch the fox** → Press `B` and move the mouse  
  - **Move the chair** → Press `G` and use the mouse to translate it  
  - **Swing the lamp** → Hover over the lamp with the controller ray and click the mouse to trigger the *Select* event

## Acknowledgments
The code was developed based on an existing [Gaussian Splatting playground in Unity](https://github.com/aras-p/UnityGaussianSplatting), [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) (3D) and [GaMeS](https://github.com/waczjoan/gaussian-mesh-splatting).

The project “Effective rendering of 3D objects using Gaussian Splatting in an Augmented Reality environment” (FENG.02.02-IP.05-0114/23) is carried out within the First Team programme of the Foundation for Polish Science co-financed by the European Union under the European Funds for Smart Economy 2021-2027 (FENG).
<div align="center">
<img src="docs/Images/fnp.png" />
</div>

### Licence

Please refer to the [original project Licence](https://github.com/aras-p/UnityGaussianSplatting?tab=readme-ov-file#license-and-external-code-used)