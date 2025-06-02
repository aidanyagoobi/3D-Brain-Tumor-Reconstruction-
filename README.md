# 3D Brain Tumor Reconstruction from MRI Data

Brain tumors are a deadly form of cancer that affect thousands of individuals annually. Highly precise technologies are essential to support doctors in diagnosing, modeling, and treating these tumors. This project reconstructs a **3D model** from a patient’s **2D MRI tumor profile** for use in:

-  Neurosurgical Planning  
-  Modeling & Simulation  
-  Medical Training  

---

## Motivation

Converting 2D MRI imaging data into 3D printable models presents significant challenges in medical visualization and personalized treatment. Current methods often lack precision in capturing tumor boundaries and surrounding tissue structures. This project introduces a streamlined pipeline that transforms 2D MRI slices into high-fidelity 3D models suitable for 3D printing.

---

## Input Data: MRI Profile

The MRI data is packaged in a `.mat` (MATLAB) file and includes:

- **Patient ID (PID)**
- **Tumor Mask**: A `512 x 512` binary matrix where `1` represents tumor tissue and `0` represents non-tumor tissue
- **Grayscale MRI Image**
- **Label**: Tumor classification tag

### Example MRI Data
![MRI Format](https://github.com/user-attachments/assets/d9949d65-4a83-4689-85af-4b731f18eaff)

---

##  Multi-Slice Profile

MRI scans are acquired at 6mm slice intervals, making it difficult to understand what lies between slices. This necessitates interpolation to build a continuous 3D structure.

### Multi-Angle View
![Slice View](https://github.com/user-attachments/assets/8e0366ca-23ce-46de-8a4c-0cf455ef827d)

---

##  Tumor Masking Example

Each mask defines tumorous regions per slice.

![Tumor Mask](https://github.com/user-attachments/assets/83a9fc63-7f6c-4a96-86a1-f6fdb9a27bea)

---

##  Interpolation: Filling the Gaps

To reconstruct 3D data from sparsely spaced MRI slices, a custom **weighted piecewise interpolation** algorithm was developed.

### Step 1: Signed Distance Transform

Each binary tumor mask is converted into a **signed distance matrix**, where:

- Negative values = inside the tumor
- Positive values = outside the tumor  
- Border proximity is encoded by magnitude

### Step 2: Slice Averaging

Using pairs of adjacent signed distance matrices (e.g., `e_i`, `h_i`), intermediate slices are calculated using **weighted averaging**, giving more weight to nearby slices.

![Interpolation Step](https://github.com/user-attachments/assets/13a46386-2a14-4788-bc5e-df5d26cc5d9e)

### Step 3: Recursive Interpolation

The process is applied recursively (typically 6 iterations) to generate high-resolution interpolated slices between actual MRI scans.

![Recursive Interpolation](https://github.com/user-attachments/assets/331ad926-3cb4-4728-a1ee-b4f4c5014943)

### Step 4: Reconstruction

The interpolated signed distances are converted back into binary masks. These are aligned along the Z-axis and passed to the **Marching Cubes** algorithm for 3D rendering.

---

##  Marching Cubes: 3D Geometry Generation

Marching Cubes converts the 3D voxel representation into a continuous 3D mesh suitable for printing.

![Marching Cubes](https://github.com/user-attachments/assets/bfb1a281-5b41-4429-ad49-411e0db7791e)

---

##  Final Patient Tumor Rendering

![Final Renderings](https://github.com/user-attachments/assets/56009365-faca-41f3-915e-3aa4edc5b4a4)

---

##  High-Resolution 3D Printed Tumor

This is an actual tumor, printed to scale, based on real patient data. Its size alone motivates the importance of this pipeline.

![3D Print](https://github.com/user-attachments/assets/73ea112f-2cc0-4fc4-a2ff-1b172836efdd)

---

##  Implementation Details

- Written entirely in Python + MATLAB
- Modular class-based code structure
- Compatible with any patient PID from the dataset
- Detailed comments and documentation included
- Designed for extensibility to other surgical domains (e.g., pelvic reconstruction)

### Additional Example: Pelvic Reconstruction
![Pelvis](https://github.com/user-attachments/assets/16bfb1c5-7d7a-4206-b925-054a4fa9d8f3)

> Note: Less interpolation precision is required for pelvic reconstruction due to wider anatomical margins compared to brain tissue.

---

##  Future Work

- Automating segmentation (deep learning-based tumor masking)
- Integrating MRI-to-Print pipelines into hospital imaging software
- Enhancing interpolation with neural shape completion techniques

---

##  Contact

For inquiries, improvements, or collaborations, feel free to reach out via email (ayagoobi@bu.edu) or open an issue on this repository.
