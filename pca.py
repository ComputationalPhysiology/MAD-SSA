
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np
import plotly.graph_objects as go

def compute_heat_map(deformation):
    magnitudes = np.linalg.norm(deformation, axis=1)
    return magnitudes

def plot_heat_map(magnitudes, avg_coords, output_path):
    x, y, z = avg_coords[:, 0], avg_coords[:, 1], avg_coords[:, 2]
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=4,
            color=magnitudes, 
            colorscale='Hot',  
            colorbar=dict(title='Deformation Magnitude')
        ),
        name='Shape Variation Heat Map'
    ))

    # Update layout for better visualization
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        title="3D Heat Map of Shape Variation",
        legend=dict(
            x=0.05, y=0.95,
            bgcolor='rgba(255,255,255,0.7)',
            bordercolor='rgba(0,0,0,0.1)',
            borderwidth=1
        )
    )

    # Save plot as an HTML file
    fig.write_html(output_path)
    print(f"Visualization saved to {output_path}")


def compute_deformation_field(avg_coords, mode_vector, std_dev, scale=2.0):
    plus_coords = avg_coords + scale * std_dev * mode_vector
    minus_coords = avg_coords - scale * std_dev * mode_vector

    deformation = plus_coords - minus_coords
    return deformation



def plot_deformation_field(deformation, avg_coords, output_path):
    x, y, z = avg_coords[:, 0], avg_coords[:, 1], avg_coords[:, 2]
    u, v, w = deformation[:, 0], deformation[:, 1], deformation[:, 2]

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=3, color='blue', opacity=0.5),
        name='Average Coordinates'
    ))

    for xi, yi, zi, ui, vi, wi in zip(x, y, z, u, v, w):
        fig.add_trace(go.Scatter3d(
            x=[xi, xi + ui], y=[yi, yi + vi], z=[zi, zi + wi],
            mode='lines',
            line=dict(color='red', width=2),
            name='Deformation Vector'
        ))


    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        title="3D Deformation Field Visualization",
        legend=dict(
            x=0.05, y=0.95,
            bgcolor='rgba(255,255,255,0.7)',
            bordercolor='rgba(0,0,0,0.1)',
            borderwidth=1
        )
    )

    # Save plot as an HTML file
    fig.write_html(output_path)



def visualize_mode_with_std(mode_vector, avg_coords, std_dev, output_path, scale=2.0):

    # Scale the mode by its standard deviation
    plus_coords = avg_coords + scale * std_dev * mode_vector
    minus_coords = avg_coords - scale * std_dev * mode_vector
    
    # Calculate the overall size of the point cloud in the "+" and "-" directions
    var_plus = np.sum(plus_coords.std(axis=0))
    var_minus = np.sum(minus_coords.std(axis=0))

    # Flip sign if the "-" direction increases the overall size more than the "+" direction
    if var_minus > var_plus:
        plus_coords, minus_coords = minus_coords, plus_coords

    # Create subplot
    fig = make_subplots(rows=1, cols=3, specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]])


    trace1 = go.Scatter3d(x=plus_coords[:, 0], y=plus_coords[:, 1], z=plus_coords[:, 2], mode='markers', 
                          marker=dict(size=2, color=plus_coords[:, 2], colorscale='Bluered', opacity=0.8))
    fig.add_trace(trace1, row=1, col=1)

    # Average mode
    trace2 = go.Scatter3d(x=avg_coords[:, 0], y=avg_coords[:, 1], z=avg_coords[:, 2], mode='markers',
                          marker=dict(size=2, color=avg_coords[:, 2], colorscale='Bluered', opacity=0.8))
    fig.add_trace(trace2, row=1, col=2)

    # "-" mode
    trace3 = go.Scatter3d(x=minus_coords[:, 0], y=minus_coords[:, 1], z=minus_coords[:, 2], mode='markers',
                          marker=dict(size=2, color=minus_coords[:, 2], colorscale='Bluered', opacity=0.8))
    fig.add_trace(trace3, row=1, col=3)

    # Set the same scale for all the subplots and add titles
    fig.update_layout(
        
        annotations=[
            dict(text="Mode + (Scaled by STD)", x=0.18, y=0.95, showarrow=False, font=dict(size=14)),
            dict(text="Average Mode", x=0.5, y=0.95, showarrow=False, font=dict(size=14)),
            dict(text="Mode - (Scaled by STD)", x=0.82, y=0.95, showarrow=False, font=dict(size=14))
        ]
    )

    fig.write_html(output_path)

def visualize_modes_with_vector(avg_coords, mode_vector, std_dev, output_path, scale=2.0):

   
    # Compute the coordinates for "+" and "-" modes
    plus_coords = avg_coords + scale * std_dev * mode_vector
    minus_coords = avg_coords - scale * std_dev * mode_vector


    fig = go.Figure()

    # "+" Mode trace
    fig.add_trace(go.Scatter3d(
        x=plus_coords[:, 0], y=plus_coords[:, 1], z=plus_coords[:, 2],
        mode='markers',
        marker=dict(size=2, color='blue', opacity=0.5),
        name='Mode + (Scaled by STD)'
    ))

    # Average Mode trace
    fig.add_trace(go.Scatter3d(
        x=avg_coords[:, 0], y=avg_coords[:, 1], z=avg_coords[:, 2],
        mode='markers',
        marker=dict(size=2, color='green', opacity=0.5),
        name='Average Mode'
    ))

    # "-" Mode trace
    fig.add_trace(go.Scatter3d(
        x=minus_coords[:, 0], y=minus_coords[:, 1], z=minus_coords[:, 2],
        mode='markers',
        marker=dict(size=2, color='red', opacity=0.5),
        name='Mode - (Scaled by STD)'
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        title="3D Mode Visualization with Enhanced Variation Vector",
        legend=dict(
            x=0.05, y=0.95,
            bgcolor='rgba(255,255,255,0.7)',
            bordercolor='rgba(0,0,0,0.1)',
            borderwidth=1
        )
    )

    # Save plot as an HTML file
    fig.write_html(output_path)
    # print(f"Visualization saved to {output_path}")



def cumvar_plot(cum_variance, outdir):
    """
    Plot cumulative variance explained by PCA modes.
    """
    plt.plot(np.arange(1, len(cum_variance) + 1), cum_variance, "-o", linewidth=2)
    plt.xlabel("Number of Modes")
    plt.ylim(0, 1)
    plt.ylabel("Cumulative Variance (%)")
    plt.grid()
    plt.savefig(os.path.join(outdir, "cumulative_variance.png"))

def get_patient_height(patient_number):
    file_path = "/home/shared/ClinicalData_OUS_MADPatients_EIVIND_29_4_2021.xlsx"
    df = pd.read_excel(file_path)
   
    if pd.isna(df[df['Pat_no'] == patient_number][["Height"]].values.flatten()) :  
        return np.median(df["Height"].dropna().values)
    else:
        patient_height = df[df['Pat_no'] == patient_number][["Height"]].values.flatten()[0]
    
        return patient_height


def pca_decomp(cases, input_dir, output_dir, scale=2.0):
    """
    Perform PCA decomposition on aligned point clouds and visualize modes.

    Args:
        cases: List of case identifiers.
        input_dir: Directory containing aligned point cloud files.
        output_dir: Directory to save PCA results and visualizations.
        scale: Multiplier for standard deviation in mode visualization.
    """
    outdir_modes = os.path.join(output_dir, "modes")
    os.makedirs(outdir_modes, exist_ok=True)

    point_coords = []

    # Load point clouds
    for case in cases:
        file_path = os.path.join(input_dir, f"MAD_{case}_aligned_points.txt")
        try:
            point_cloud = np.loadtxt(file_path, delimiter=",")
            height = get_patient_height(int(case))
            # point_cloud[:, 2] /= height 
            point_coords.append(point_cloud/height)

        except Exception as e:
            print(f"Skipping case {case} due to error: {e}")
            continue

    if not point_coords:
        print("No valid point clouds found. Exiting...")
        return

    point_coords = np.array(point_coords)
    cshape = point_coords.shape
    X = point_coords.reshape((cshape[0], cshape[1] * cshape[2]), order="C")
    avg_coords = np.mean(point_coords, axis=0)

    # PCA decomposition
    pca = PCA(svd_solver="full")
    pca.fit(X)

   
    scores = pca.transform(X)
  
    std_devs = scores.std(axis=0)

    # Save scores
    scores_df = pd.DataFrame(scores, columns=[f"M{i}" for i in range(1, X.shape[0] + 1)])
    scores_df.index = cases
    scores_df.index.rename("Pat_no", inplace=True)
    scores_df.to_csv(os.path.join(output_dir, "pca_scores.csv"))

    # Save variance ratios
    cum_variance = np.cumsum(pca.explained_variance_ratio_)
    cumvar_plot(cum_variance, output_dir)
    var_ratios_df = pd.DataFrame(
        pca.explained_variance_ratio_,
        index=[f"M{i}" for i in range(1, X.shape[0] + 1)],
        columns=["Variance Ratio"],
    )
    var_ratios_df.index.rename("Mode", inplace=True)
    var_ratios_df.to_csv(os.path.join(output_dir, "pca_variance_ratios.csv"))
    outdir_point_clouds = os.path.join(output_dir, "point_clouds")
    os.makedirs(outdir_point_clouds, exist_ok=True)

    avg_coords_path = os.path.join(outdir_point_clouds, "avg_coords.txt")
    np.savetxt(avg_coords_path, avg_coords, delimiter=",")
    print(f"Average coordinates saved to {avg_coords_path}")
    # Visualize PCA Modes
    for i, (mode, std_dev) in enumerate(zip(pca.components_[:10], std_devs), start=1):  # First 10 modes
        mode_coords = mode.reshape(cshape[1], cshape[2], order="C")
        output_path = os.path.join(outdir_modes, f"mode_{i}.html")
        output_path2 = os.path.join(outdir_modes, f"mode_{i}_comparison.html")
        output_path3 = os.path.join(outdir_modes, f"mode_{i}_deformation.html")
        output_path4 = os.path.join(outdir_modes, f"mode_{i}_heat_map.html")


        plus_coords = avg_coords + scale * std_dev * mode_coords
        minus_coords = avg_coords - scale * std_dev * mode_coords

        # Save point clouds
        plus_path = os.path.join(outdir_point_clouds, f"mode_{i}_plus_coords.txt")
        minus_path = os.path.join(outdir_point_clouds, f"mode_{i}_minus_coords.txt")
        np.savetxt(plus_path, plus_coords, delimiter=",")
        np.savetxt(minus_path, minus_coords, delimiter=",")
        print(f"Mode {i}: plus_coords saved to {plus_path}, minus_coords saved to {minus_path}")

        visualize_mode_with_std(mode_coords, avg_coords, std_dev, output_path, scale)
        visualize_modes_with_vector(avg_coords, mode_coords, std_dev, output_path2, scale)
        deformation = compute_deformation_field(avg_coords, mode_coords, std_dev)
        plot_deformation_field(deformation, avg_coords, output_path3)
        magnitudes = compute_heat_map(deformation)

        plot_heat_map(magnitudes, avg_coords, output_path4)
if __name__ == "__main__":
    input_directory = "./Aligned_Models"
    cases = [c.split("_")[1] for c in os.listdir(input_directory) if c.endswith("_aligned_points.txt")]
    
    output_directory = "./PCA_Results_final_height"
    os.makedirs(output_directory, exist_ok=True)
    pca_decomp(cases, input_directory, output_directory, scale=2.0)

