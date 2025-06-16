# 🧪 Demo: Generate a Point Cloud from a Single Patient Segmentation

This demo shows how to use the `ventricshape-pc` command-line tool to create an epicardial and endocardial point cloud from a `.h5` segmentation file.

## 📁 Folder Setup

```
project-root/
├── seg_files/
│   └── patient001.h5
├── settings/
│   └── patient001.json
```

## 📌 Required Segmentation File Format

Each `.h5` file in `seg_files/` must contain:

- `'LV_mask'`: shape `[slices, height, width]`
- `'RV_mask'`: shape `[slices, height, width]`
- `'resolution'`: `[px, py, pz]` — voxel spacing in mm

## ⚙️ Example Settings File

```json
{
  "mesh": {
    "high": {
      "smoothing": {
        "sigma_space": 3,
        "sigma_color": 3,
        "iterations": 5
      },
      "sampling_rate": 2
    }
  }
}
```

## 🚀 Run the Demo

```bash
ventricshape-pc   --sample_name patient001   --settings_dir settings   --patient_folder seg_files   --mesh_quality high   --mask_flag True
```

## ✅ Output

```
seg_files/
├── patient001.h5
├── points_cloud_epi.csv
├── points_cloud_endo.csv
```