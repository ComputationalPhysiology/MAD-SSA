# 🧪 Demo: Generate a Point Cloud from a Single Patient Segmentation

This demo shows how to use the `ventricshape-pc` CLI tool to generate **epicardial** and **endocardial** point clouds from `.h5` segmentation files.
We use a publicly available dataset available in https://www.ub.edu/mnms-2/.

---

## Folder Structure

```
project-root/
├── seg_files/
│   ├── 001.h5
│   └── 002.h5
```

Each `.h5` file corresponds to one patient.

---

##  Required `.h5` File Format

Each `.h5` file in `seg_files/` must contain:

- `LV_mask`: shape `[slices, height, width]` — binary left ventricle mask  
- `RV_mask`: shape `[slices, height, width]` — binary right ventricle mask  
- `resolution`: `[px, py, pz]` — voxel spacing in millimeters

---

## ⚙️ Settings File

A settings file is automatically generated in `settings/`. It contains point cloud fitting parameters:

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
```

You may manually fine-tune these parameters:

| Parameter              | Description                                 |
|------------------------|---------------------------------------------|
| `lax_smooth_level_epi` | Smoothness of epicardial LAX curves         |
| `lax_smooth_level_endo`| Smoothness of endocardial LAX curves        |

> Increasing these values produces smoother curves (useful for correcting motion or misalignment).

---


## 🚀 Run Point Cloud Generation

To process a segmentation file for patient `001`:

```bash
ventricshape-pc --n 001
```

This will:
- Load `seg_files/001.h5`
- Generate a settings file `settings/001.json` (if not already present)
- Output:
  - `results/001/points_cloud_epi.csv`
  - `results/001/points_cloud_endo.csv`

For each patient, a results folder is created (e.g., `results/001/`).  
This folder contains several subfolders and files that help you check the quality of each processing step:

- **01_Masks/**: Visualizations of the original segmentation masks and extracted contours for each slice.
- **02_ShaxBSpline/**: Plots of the short-axis B-spline fits to the contours.
- **03_LaxBSpline/**: Plots of the long-axis B-spline fits.
- **04_Contours/**: **Overlay plots of the generated contours with respect to the original segmentation stack**—useful for visually checking the accuracy of the contour extraction and fitting.
- **05_Point Cloud/**: 3D visualizations of the generated point clouds.

You can open the images in these folders to visually inspect each step of the pipeline and ensure the results are as expected.

---
##  Check  Fitting Accuracy

To verify  alignment with the original segmentation, generate a mesh:

```bash
ventricshape-createmesh --n 001
```

This generates:
- A `.vtk` mesh in: `results/001/06_Mesh/`
- A `.msh` file
- Diagnostic reports (fit quality, deviation from contours)

You can open the mesh in [ParaView](https://www.paraview.org/) for inspection.


---



