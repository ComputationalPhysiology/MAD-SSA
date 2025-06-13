# 3D Cardiac Statistical Shape Analysis Pipeline

This project provides a complete pipeline for processing 3D cardiac medical imaging data, from initial segmentation masks to statistical shape analysis using Principal Component Analysis (PCA). The pipeline generates 3D point clouds from segmentations, aligns them, and then performs PCA to identify primary modes of shape variation. Additionally, it includes scripts to create 3D meshes and evaluate their accuracy against the original data.

## 📜 Table of Contents

- [Project Overview](#-project-overview)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#-usage)
  - [Directory Structure](#directory-structure)
  - [Running the Full Pipeline](#running-the-full-pipeline)
  - [Individual Scripts](#individual-scripts)
    - [1. Point Cloud Generation](#1-point-cloud-generation)
    - [2. Point Cloud Alignment](#2-point-cloud-alignment)
    - [3. PCA and Shape Analysis](#3-pca-and-shape-analysis)
    - [4. 3D Mesh Generation and Error Analysis](#4-3d-mesh-generation-and-error-analysis)
- [File Descriptions](#-file-descriptions)

## 🔭 Project Overview

The main goal of this project is to automate the process of analyzing cardiac shapes from a cohort of subjects. The pipeline consists of the following major steps:

1.  **Point Cloud Generation**: Creates 3D point clouds of the endocardium and epicardium from `.h5` segmentation files.
2.  **Alignment**: Aligns the generated point clouds for all subjects to a common coordinate system. This step is crucial for meaningful comparison.
3.  **Principal Component Analysis (PCA)**: Performs PCA on the aligned point clouds to identify the main patterns of shape variation across the subjects.
4.  **3D Mesh Creation**: (Optional) Generates a 3D mesh from a point cloud and computes error metrics by comparing it to the ground truth segmentation.

The entire process is orchestrated by `main.py`, which runs the necessary scripts in sequence.

## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

Make sure you have Python 3 installed. You will also need to install the required Python libraries. A `requirements.txt` file would be ideal for managing dependencies. Based on the imports, you'll need libraries such as:

* numpy
* pandas
* scikit-learn
* matplotlib
* plotly
* meshio
* open3d
* h5py
* structlog
* opencv-python

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

2.  **Install the required libraries:**
    ```bash
    pip install numpy pandas scikit-learn matplotlib plotly meshio open3d h5py structlog opencv-python
    ```

## ⚙️ Usage

This section explains how to use the scripts to process your data.

### Directory Structure

The scripts expect a specific directory structure, which is configured in `config.py`. You should have the following directories:

* `controls/00_data/`: This is the **input directory** where your subject data (in `.h5` format) is stored. Each subject should have its own subfolder.
* `Aligned_Models/`: This is the **output directory** for aligned point clouds.
* `controls/PCA_Results/`: This is the **output directory** for PCA results, including modes of variation and visualizations.

### Running the Full Pipeline

The easiest way to run the entire pipeline is to execute the `main.py` script. This script will run the point cloud generation, alignment, and PCA in the correct order.

```bash
python main.py
