# saxomode

`saxomode` is an open-source toolkit for 3D ventricular shape analysis from cardiac magnetic resonance short axis view medical images.  
It provides an end-to-end pipeline for generating, aligning, and analyzing ventricular point clouds, including principal component analysis (PCA) and mesh generation.  
- Source code: https://github.com/ComputationalPhysiology/MAD-SSA.git
- Documentation: https://computationalphysiology.github.io/MAD-SSA/

## Install
Install with pip
```
python3 -m pip install saxomode
```
## Citing
If you use `saxomode` in your research project we would appreciate if you could use the following citation
```
@article{,
    doi = {},
    url = {},
    year = {2024},
    publisher = {},
    volume = {},
    number = {},
    pages = {},
    author = {Giulia Monopoli et al.},
    title = {},
    journal = {}
}
```

## License
MIT
<!-- 
## 📜 Table of Contents

- [Project Overview](#-project-overview)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#-usage)
  - [Directory Structure](#directory-structure)
  - [Running the Full Pipeline](#running-the-full-pipeline)
  - [Individual Scripts](#individual-scripts)


## 🔭 Project Overview

The main goal of this project is to automate the process of analyzing cardiac shapes from a cohort of subjects. The pipeline consists of the following major steps:

1.  **Point Cloud Generation**: Creates 3D point clouds of the endocardium and epicardium from `.h5` segmentation files.
2.  **Alignment**: Aligns the generated point clouds for all subjects to a common coordinate system. This step is crucial for meaningful comparison.
3.  **Principal Component Analysis (PCA)**: Performs PCA on the aligned point clouds to identify the main patterns of shape variation across the subjects.
4.  **3D Mesh Creation**: (Optional) Generates a 3D mesh from a point cloud and computes error metrics by comparing it to the ground truth segmentation.

The entire process is contained in `main.py`, which runs the necessary scripts in sequence.

## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ComputationalPhysiology/MAD-SSA.git
    ```

2.  **Install the required libraries:**
    ```bash
    pip install numpy pandas scikit-learn matplotlib plotly meshio open3d h5py structlog opencv-python
    ```

## ⚙️ Usage

This section explains how to use the scripts to process your data. -->
<!-- 
### Directory Structure

The scripts expect a specific directory structure, which is configured in `config.py`. You should have the following directories:

* `controls/ES_files_controls`: This is  where your subject data (in `.h5` format) is  stored.
  
Next, the script will generate:
* `controls/00_data/`: This is the **input directory** where your subject data point cloud result will be stored. Each subject should have its own subfolder.
* `Aligned_Models/`: This is the **output directory** for aligned point clouds.
* `controls/PCA_Results/`: This is the **output directory** for PCA results, including modes of variation and visualizations.
* `controls/settings/`: This contains the **fitting parameters** for the point cloud generation step.

### Running the Full Pipeline

The easiest way to run the entire pipeline is to execute the `main.py` script. This script will run the point cloud generation, alignment, and PCA in the correct order.

```bash
python main.py
```
### Individual Scripts

You can execute scripts individually:

| Script                | Description                                                | Usage                                       |
| :-------------------- | :--------------------------------------------------------- | :------------------------------------------ |
| **`main_pc.py`** | Generates 3D point clouds from `.h5` segmentations.        | `python main_pc.py [--name <subject>]`      |
| **`alignment.py`** | Aligns all generated point clouds to a common origin.      | `python alignment.py`                       |
| **`pca.py`** | Runs PCA on aligned point clouds to find shape variations. | `python pca.py`                             |
| **`create_3d_mesh.py`** | Creates a 3D mesh and computes error metrics.            | `python create_3d_mesh.py --name <subject>` |



For more advanced control, especially over the point cloud generation process, you can modify the patient-specific settings files.

#### Adjusting LAX Curve Smoothing

You can control the smoothing level of the LAX curves for both the epicardium and endocardium. This is useful for fine-tuning the point cloud to better fit the source data, while correcting for breath-hold misalignment.

* **How to adjust**: Modify the `lax_smooth_level_epi` and `lax_smooth_level_endo` values in the patient's corresponding JSON settings file.
* **Location**: These files are located in the directory specified by `SETTINGS_DIRECTORY` in `config.py` (e.g., `controls/settings/`).

**Example `settings/<patient_name>.json`:**

```json
{
    "mesh": {
        "fine": {
            "lax_smooth_level_epi": 80,
            "lax_smooth_level_endo": 50,
            "...": "..."
        }
    }
}
 -->
