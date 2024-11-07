# %%
import h5py
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import cv2 as cv
from tqdm import tqdm

import ventric_mesh.mesh_utils as mu
import ventric_mesh.utils as utils
import matplotlib.pyplot as plt
from structlog import get_logger

logger = get_logger()


# %%
def read_data_h5(file_dir):
    with h5py.File(file_dir, "r") as h5_file:
        LVmask = h5_file["sax_coords"][:]
        P_a_endo = h5_file["P_a_endo"][:]
        P_a_epi = h5_file["P_a_epi"][:]
    return LVmask, P_a_endo, P_a_epi 


def close_apex(LVmask):
    K, I, J = LVmask.shape
    mask_closed_apex = np.zeros((K + 1, I, J))
    mask_closed_apex[:-1, :, :] = LVmask
    kernel = np.ones((3, 3), np.uint8)
    mask_last_slice = np.uint8(LVmask[-1, :, :] * 255)
    mask_last_slice_closed = cv.dilate(mask_last_slice, kernel, iterations=20)
    mask_last_slice_closed = cv.erode(mask_last_slice_closed, kernel, iterations=25)
    plt.imshow(mask_last_slice_closed)
    plt.savefig('test.png')
    mask_closed_apex[-1, :, :] = mask_last_slice_closed
    return mask_closed_apex


def located_h5(data_address):
    h5_files = list(data_address.glob("*.h5"))
    if len(h5_files) != 1:
        logger.error("Data folder must contain exactly 1 .mat file.")
        return
    h5_file = h5_files[0]
    logger.info(f"{h5_file.name} is loading.")
    return h5_file

import numpy as np
import plotly.graph_objects as go

def restructure_coords_into_slices(coords, z_tolerance=0.1):
    """
    Restructures the input coordinates into slices based on z-values.

    Parameters:
    coords (np.ndarray): An n by 3 array where each row represents a 3D coordinate (x, y, z).
    z_tolerance (float): The maximum difference in z-values to consider points as part of the same slice.

    Returns:
    list: A list of K slices, where each slice is an M by 3 numpy array.
    """
    if not isinstance(coords, np.ndarray) or coords.shape[1] != 3:
        raise ValueError("Input must be a numpy array with shape (n, 3)")
    
    # Sort the coordinates based on the z-values
    coords = coords[coords[:, 2].argsort()[::-1]]

    # Group coordinates into slices
    slices = []
    current_slice = [coords[0]]

    for i in range(1, len(coords)):
        if abs(coords[i, 2] - current_slice[-1][2]) <= z_tolerance:
            current_slice.append(coords[i])
        else:
            slices.append(np.array(current_slice))
            current_slice = [coords[i]]
    
    # Append the last slice
    if current_slice:
        slices.append(np.array(current_slice))
    
    return slices

def average_z_for_slices(slices):
    """
    Adjusts the z-values of all points in each slice to the average z-value of that slice and logs the overall mean and standard deviation of distance changes.

    Parameters:
    slices (list): A list of K slices, where each slice is an M by 3 numpy array.

    Returns:
    list: A list of K slices with adjusted z-values.
    """
    adjusted_slices = []
    distance_changes = []

    for slice in slices:
        avg_z = np.mean(slice[:, 2])
        distances = np.abs(slice[:, 2] - avg_z)
        distance_changes.extend(distances)

        adjusted_slice = slice.copy()
        adjusted_slice[:, 2] = avg_z
        adjusted_slices.append(adjusted_slice)
    
    if distance_changes:
        mean_change = np.mean(distance_changes)
        std_change = np.std(distance_changes)
        logger.warning(f"Mean change in z-values: {mean_change:.4f} ± {std_change:.4f}")
    
    return adjusted_slices

def plot_3d_points(coords, additional_coords=None):
    """
    Plots 3D points given an n by 3 numpy array of coordinates, with an option to plot additional points on the same figure.

    Parameters:
    coords (np.ndarray): An n by 3 array where each row represents a 3D coordinate (x, y, z).
    additional_coords (np.ndarray, optional): An optional n by 3 array of additional coordinates to plot.
    """
    if not isinstance(coords, np.ndarray) or coords.shape[1] != 3:
        raise ValueError("Input must be a numpy array with shape (n, 3)")

    # Extract x, y, z components for original coords
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    # Create a 3D scatter plot for original coords
    fig = go.Figure(data=[go.Scatter3d(
        x=x, 
        y=y, 
        z=z, 
        mode='markers',
        marker=dict(
            size=5,
            color='blue',
            opacity=0.8
        ),
        name='Original'
    )])

    # Add additional coordinates if provided
    if additional_coords is not None:
        if not isinstance(additional_coords, np.ndarray) or additional_coords.shape[1] != 3:
            raise ValueError("Additional coordinates must be a numpy array with shape (n, 3)")
        
        x_add = additional_coords[:, 0]
        y_add = additional_coords[:, 1]
        z_add = additional_coords[:, 2]
        
        fig.add_trace(go.Scatter3d(
            x=x_add,
            y=y_add,
            z=z_add,
            mode='markers',
            marker=dict(
                size=5,
                color='red',
                opacity=0.8
            ),
            name='Adjusted'
        ))

    # Update layout for better visualization
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title="3D Points Visualization",
        margin=dict(l=0, r=0, b=0, t=30)
    )

    # Show the plot
    fig.show()



# %%
def create_mesh(mesh_settings, sample_directory, output_folder, plot_flag = True):
    output_folder = Path((sample_directory / "00_results"))
    output_folder.mkdir(exist_ok=True, parents=True)

    h5_file_address = located_h5(sample_directory)
    coords, P_a_endo, P_a_epi  = read_data_h5(h5_file_address.as_posix())
    LV_coords_raw = restructure_coords_into_slices(coords, z_tolerance=0.1)
    LV_coords_raw = average_z_for_slices(LV_coords_raw)
    plot_3d_points(coords, additional_coords=np.vstack(LV_coords_raw))
    breakpoint()
    # slice_thickness = 1
    logger.info(f"Reading mask with slice thickness of {slice_thickness}mm and resolution of {resolution}mm")
    LVmask = close_apex(LVmask_raw)
    logger.info("Mask is loaded and apex is closed")

    mask_epi, mask_endo = mu.get_endo_epi(LVmask)
    
    if plot_flag:
        outdir = output_folder / "01_Masks"
        outdir.mkdir(exist_ok=True)
        K = len(mask_epi)
        K_endo = len(mask_endo)
        for k in range(K):
            mask_epi_k = mask_epi[k]
            mask_endo_k = mask_endo[k]
            LVmask_k = LVmask[k]
            new_image = utils.image_overlay(LVmask_k, mask_epi_k, mask_endo_k)
            fnmae = outdir.as_posix() + "/" + str(k) + ".png"
            plt.imshow(new_image)
            dpi = np.round((300/resolution)/100)*100
            plt.savefig(fnmae, dpi=dpi)
            plt.close()
    
    coords_epi = mu.get_coords_from_mask(mask_epi, resolution, slice_thickness)
    coords_endo = mu.get_coords_from_mask(mask_endo, resolution, slice_thickness)

    tck_epi = mu.get_shax_from_coords(
        coords_epi, mesh_settings["smooth_level_epi"]
    )
    tck_endo = mu.get_shax_from_coords(
        coords_endo, mesh_settings["smooth_level_endo"]
    )
    K = len(tck_epi)

    if plot_flag:
        outdir = output_folder / "02_ShaxBSpline"
        outdir.mkdir(exist_ok=True)
        K_endo = len(tck_endo)
        for k in range(K):
            utils.plot_shax_with_coords(coords_epi, tck_epi, k, new_plot=True)
            if k < K_endo:
                utils.plot_shax_with_coords(coords_endo, tck_endo, k, color="b")
            fnmae = outdir.as_posix() + "/" + str(k) + ".png"
            plt.savefig(fnmae)
            plt.close()
    sample_points_epi = mu.get_sample_points_from_shax(
        tck_epi, mesh_settings["num_lax_points"]
    )
    sample_points_endo = mu.get_sample_points_from_shax(
        tck_endo, mesh_settings["num_lax_points"]
    )

    apex_threshold = mu.get_apex_threshold(sample_points_epi, sample_points_endo)
    LAX_points_epi, apex_epi = mu.create_lax_points(
        sample_points_epi, apex_threshold, slice_thickness
    )
    LAX_points_endo, apex_endo = mu.create_lax_points(
        sample_points_endo, apex_threshold, slice_thickness
    )
    tck_lax_epi = mu.get_lax_from_laxpoints(
        LAX_points_epi, mesh_settings["lax_smooth_level_epi"], mesh_settings["lax_spline_order_epi"]
    )
    tck_lax_endo = mu.get_lax_from_laxpoints(
        LAX_points_endo, mesh_settings["lax_smooth_level_endo"], mesh_settings["lax_spline_order_endo"]
    )
    if plot_flag:
        outdir = output_folder / "03_LaxBSpline"
        outdir.mkdir(exist_ok=True)
        fig = go.Figure()
        utils.plotly_3d_LAX(
            fig,
            range(int(mesh_settings["num_lax_points"] / 2)),
            tck_lax_epi,
            tck_endo=tck_lax_endo,
        )
        fnmae = outdir.as_posix() + "/lax_splines.html"
        fig.write_html(fnmae)

    tck_shax_epi = mu.get_shax_from_lax(
        tck_lax_epi,
        apex_epi,
        mesh_settings["num_z_sections_epi"],
        mesh_settings["z_sections_flag_epi"],
    )
    tck_shax_endo = mu.get_shax_from_lax(
        tck_lax_endo,
        apex_endo,
        mesh_settings["num_z_sections_endo"],
        mesh_settings["z_sections_flag_endo"],
    )
    if plot_flag:
        outdir = output_folder / "04_Contours"
        outdir.mkdir(exist_ok=True)
        fig = go.Figure()
        fig = mu.plotly_3d_contours(
            fig, tck_shax_epi, tck_lax_epi, tck_shax_endo, tck_lax_endo
        )
        fnmae = outdir.as_posix() + "/Contour.html"
        fig.write_html(fnmae)
    points_cloud_epi, k_apex_epi  = mu.create_point_cloud(
        tck_shax_epi,
        apex_epi,
        mesh_settings["seed_num_base_epi"],
        seed_num_threshold=mesh_settings["seed_num_threshold_epi"],
        update_seed_num_flag = False
    )
    points_cloud_endo, k_apex_endo = mu.create_point_cloud(
        tck_shax_endo,
        apex_endo,
        mesh_settings["seed_num_base_endo"],
        seed_num_threshold=mesh_settings["seed_num_threshold_endo"],
        update_seed_num_flag = False
    )
    
    points_cloud_epi_unique = points_cloud_epi[:k_apex_epi]
    points_cloud_endo_unique = points_cloud_endo[:k_apex_endo]
    
    updated_apex_epi = apex_epi
    updated_apex_epi[2] = points_cloud_epi_unique[-1][0,2]
    updated_tck_shax_epi = mu.get_shax_from_lax(
        tck_lax_epi,
        updated_apex_epi,
        mesh_settings["num_z_sections_epi"],
        mesh_settings["z_sections_flag_epi"],
    )
    
    updated_apex_endo = apex_endo
    updated_apex_endo[2] = points_cloud_endo_unique[-1][0,2]
    updated_tck_shax_endo = mu.get_shax_from_lax(
        tck_lax_endo,
        updated_apex_endo,
        mesh_settings["num_z_sections_endo"],
        mesh_settings["z_sections_flag_endo"],
    )
    
    updated_points_cloud_epi = []
    K = len(updated_tck_shax_epi)
    for k in tqdm(range(K), desc="Creating final unique point cloud ", ncols=100):
        tck_k = updated_tck_shax_epi[k]
        points = mu.equally_spaced_points_on_spline(tck_k, mesh_settings["seed_num_base_epi"])
        updated_points_cloud_epi.append(points)

    updated_points_cloud_endo = []
    K = len(updated_tck_shax_endo)
    for k in tqdm(range(K), desc="Creating final unique point cloud", ncols=100):
        tck_k = updated_tck_shax_endo[k]
        points = mu.equally_spaced_points_on_spline(tck_k, mesh_settings["seed_num_base_epi"])
        updated_points_cloud_endo.append(points)

    if plot_flag:
        outdir = output_folder / "05_Point Cloud"
        outdir.mkdir(exist_ok=True)
        fig = go.Figure()
        for points in updated_points_cloud_epi:
            mu.plot_3d_points_on_figure(points, fig=fig)
        fnmae = outdir.as_posix() + "/Points_cloud_epi.html"
        fig.write_html(fnmae)
        fig = go.Figure()
        for points in updated_points_cloud_endo:
            mu.plot_3d_points_on_figure(points, fig=fig)
        fnmae = outdir.as_posix() + "/Points_cloud_endo.html"
        fig.write_html(fnmae)
    return points_cloud_epi_unique, points_cloud_endo_unique
