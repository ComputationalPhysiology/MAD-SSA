# Demo: Run the Full Ventricular Shape Pipeline

This demo will show how to run the entire shape analysis pipeline using the single command-line tool:

- `saxomode-pipeline`

## Usage

Simply run the following command in your terminal:

```bash
saxomode-pipeline
```

This will sequentially execute all steps:

- Generate point clouds from segmentation files
- Align the point clouds
- Perform principal component analysis (PCA)