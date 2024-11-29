import yaml
import os

import numpy as np
import copy
import pandas as pd

from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

def visualize_mode(mode_vector, average_coords, output_path, scale=2.0):
    """
    Visualize a PCA mode as a point cloud.
    Args:
        mode_vector: The PCA mode vector reshaped to match the point cloud shape.
        average_coords: The average coordinates of the point cloud.
        output_path: Path to save the visualization image.
        scale: Multiplier to exaggerate the PCA mode for better visualization.
    """
    plus_coords = average_coords + scale * mode_vector
    minus_coords = average_coords - scale * mode_vector

    fig = plt.figure(figsize=(12, 6))

    # "+" mode
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(plus_coords[:, 0], plus_coords[:, 1], plus_coords[:, 2], c='blue', s=1)
    ax.set_title("Mode +")

    # "-" mode
    ax = fig.add_subplot(122, projection='3d')
    ax.scatter(minus_coords[:, 0], minus_coords[:, 1], minus_coords[:, 2], c='red', s=1)
    ax.set_title("Mode -")

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_tick_params(labelsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    plt.close()


def cumvar_plot(cum_variance, outdir):
    plt.plot(np.arange(1, len(cum_variance) +1), cum_variance, "-o", linewidth = 2)
    plt.xlabel("Number of modes")
    plt.ylim(0,1)
    plt.ylabel("Cumulative variance (%)")
    plt.grid()
    plt.savefig(os.path.join(outdir, "cumulative_variance.png"))


def pca_decomp(cases, input_dir, output_dir):
    
    outdir_modes = os.path.join(output_dir, "modes")
    

    point_coords = []

    for case in cases:
        print(f"Processing case {case}")
        file_path = os.path.join(input_dir, f"{case}_aligned_points.txt")
        try:
            point_cloud = np.loadtxt(file_path, delimiter=",")
            
            point_coords.append(point_cloud)
        except Exception as e:
            print(f"Skipping case {case} due to error: {e}")
            continue
   
    if not point_coords:
        print("No valid point clouds found. Exiting...")
        return

    point_coords = np.array(point_coords)
  
    cshape = point_coords.shape
    X = point_coords.reshape((cshape[0], cshape[1]*cshape[2]), order="C")
    
    # Compute average point cloud
    avg_coords = np.mean(point_coords, axis=0)

    # PCA
    pca = PCA(svd_solver="full")
    pca.fit(X)
    
    # Scores
    scores = pd.DataFrame(pca.components_ @ X.T,
                          columns=[f"M{i}" for i in range(1, X.shape[0] + 1)])
    scores.index = cases
    scores.to_csv(os.path.join(output_dir, "pca_scores.csv"))

    # Variance Ratios
    cum_variance = np.cumsum(pca.explained_variance_ratio_)
    cumvar_plot(cum_variance, output_dir)

    var_ratios_df = pd.DataFrame(pca.explained_variance_ratio_,
                                 index=[f"M{i}" for i in range(1, X.shape[0] + 1)],
                                 columns=["variance_ratio"])
    var_ratios_df.index.rename("mode", inplace=True)
    var_ratios_df.to_csv(os.path.join(output_dir, "pca_variance_ratios.csv"))

    
    # Visualize PCA Modes
    for i, mode in enumerate(pca.components_[:10], start=1):  # Viszzualize the first 10 modes
        mode_coords = mode.reshape(cshape[1], cshape[2], order="C")
        output_path = os.path.join(outdir_modes, f"mode_{i}.png")
        if not os.path.exists(outdir_modes):
            os.makedirs(outdir_modes)
        visualize_mode(mode_coords, avg_coords, output_path)



if __name__ == "__main__":
    input_directory = "../00_data"  
    cases =  os.listdir(input_directory)

    cases = [c for c in cases if c != ".DS_Store" ]
    outdir = os.path.join("./PCA","pca_results")
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    pca_decomp(cases,"/home/shared/MAD-SSA/Aligned_Models",outdir )
