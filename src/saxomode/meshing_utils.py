# %%
import h5py
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import cv2 as cv
from tqdm import tqdm
import open3d as o3d
import meshio
import pandas as pd
import ventric_mesh.mesh_utils as mu
import ventric_mesh.utils as utils
import matplotlib.pyplot as plt
from structlog import get_logger

logger = get_logger()
# res_21 = 

# %%
def read_data_h5_mask(file_dir):
    with h5py.File(file_dir, "r") as h5_file:
        LVmask = h5_file["LVmask"][:]
        resolution = h5_file["resolution"][:]
        print(f"Resolution is {resolution}")
        num_nonzero_first_slice = np.count_nonzero(LVmask[0])
    num_nonzero_last_slice = np.count_nonzero(LVmask[-1])

    if num_nonzero_first_slice < num_nonzero_last_slice:
        print("Stack is ordered from apex to base. Flipping the stack...")
        LVmask = np.flip(LVmask, axis=0)  # Flip along the first axis
   

    return LVmask, resolution

def read_data_h5_RVmask(file_dir):
    with h5py.File(file_dir, "r") as h5_file:
        RVmask = h5_file["RV_mask"][:]
        RVcom = h5_file["RV_com"][:]
        resolution = h5_file["resolution"][:]
    return RVmask, RVcom, resolution

def read_data_h5(file_dir):
    with h5py.File(file_dir, "r") as h5_file:
        LVmask = h5_file["sax_coords"][:]
        P_a_endo = h5_file["P_a_endo"][:]
        P_a_epi = h5_file["P_a_epi"][:]
        resolution = h5_file["resolution"][:]
        # slice_thicknesses = h5_file["slice_thicknesses"][:]
    return LVmask, P_a_endo, P_a_epi, resolution

def invert_coords(*coords):
    inverted_coords = []
    for coord in coords:
        inverted_coords.append(coord[:, ::-1])
    return inverted_coords

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

def calculate_closest_distances(point_cloud):
    """
    Calculate the distance to the closest point for each point in a given point cloud.

    Parameters:
    point_cloud (numpy.ndarray): A numpy array of shape (N, 3) representing the point cloud,
                                 where N is the number of points.

    Returns:
    numpy.ndarray: A 1D array of length N containing the distances to the closest point for each point.
    """
    num_points = point_cloud.shape[0]
    closest_distances = np.full(num_points, np.inf)

    for i in range(num_points):
        # Calculate distances from point i to all other points
        distances = np.linalg.norm(point_cloud - point_cloud[i], axis=1)
        distances[i] = np.inf  # Exclude self-distance
        closest_distances[i] = np.min(distances)

    return closest_distances

def count_neighbors_within_resolution(point_cloud, resolution):
    """
    Calculate the number of neighboring points within a given resolution for each point in a point cloud.

    Parameters:
    point_cloud (numpy.ndarray): A numpy array of shape (N, 3) representing the point cloud,
                                 where N is the number of points.
    resolution (float): The distance threshold to count neighboring points.

    Returns:
    numpy.ndarray: A 1D array of length N containing the count of points within the resolution for each point.
    """
    num_points = point_cloud.shape[0]
    neighbor_counts = np.zeros(num_points, dtype=int)

    for i in range(num_points):
        # Calculate distances from point i to all other points
        distances = np.linalg.norm(point_cloud - point_cloud[i], axis=1)
        # Count the number of points within the given resolution (excluding the point itself)
        neighbor_counts[i] = np.sum((distances < resolution) & (distances > 0))

    return neighbor_counts


def extract_boundary_points(point_cloud, resolution, threshold=4):
    """
    Extract boundary points from the point cloud by removing points that have at least `threshold` neighbors
    within the specified resolution.

    Parameters:
    point_cloud (numpy.ndarray): A numpy array of shape (N, 3) representing the point cloud.
    resolution (float): The distance threshold to count neighboring points.
    threshold (int): The number of neighbors a point must have to be considered non-boundary (default is 4).

    Returns:
    numpy.ndarray: A filtered array of points representing the boundary of the point cloud.
    """
    neighbor_counts = count_neighbors_within_resolution(point_cloud, resolution)
    boundary_points = point_cloud[neighbor_counts < threshold]

    return boundary_points



def dfs(graph, node, visited):
    """
    Depth-first search to explore connected components in a graph.

    Parameters:
    graph (dict): A dictionary representing the adjacency list of the graph.
    node: The starting node for the DFS.
    visited (set): A set to track visited nodes.
    """
    visited.add(node)
    for neighbour in graph[node]:
        if neighbour not in visited:
            dfs(graph, neighbour, visited)

def find_connected_subsets(boundary_points, resolution):
    """
    Find two connected subsets of boundary points using a graph-based approach.

    Parameters:
    boundary_points (numpy.ndarray): A numpy array representing the boundary points.
    resolution (float): The distance threshold to determine connectivity between points.

    Returns:
    tuple: Two sets of connected boundary points.
    """
    num_points = boundary_points.shape[0]
    graph = {i: set() for i in range(num_points)}

    # Build the graph based on points within the resolution distance
    for i in range(num_points):
        for j in range(i + 1, num_points):
            if np.linalg.norm(boundary_points[i] - boundary_points[j]) < resolution:
                graph[i].add(j)
                graph[j].add(i)

    # Find connected subsets using DFS
    visited = set()
    subsets = []

    for i in range(num_points):
        if i not in visited:
            subset = set()
            dfs(graph, i, subset)
            visited.update(subset)
            subsets.append(subset)

    # Return the first two connected subsets if available
    if len(subsets) >= 2:
        return boundary_points[list(subsets[0])], boundary_points[list(subsets[1])]
    elif len(subsets) == 1:
        return boundary_points[list(subsets[0])], np.array([])
    else:
        return np.array([]), np.array([])


def get_epi_endo_from_coords(LV_coords, resolution):
    coords_epi = []
    coords_endo = []
    for points in LV_coords:
        bc_points = extract_boundary_points(points, resolution)
        coords_epi_k, coords_endo_k = find_connected_subsets(bc_points, resolution*np.sqrt(2))
        if len(coords_epi_k)<len(coords_endo_k):
            temp = coords_endo_k
            coords_endo_k = coords_epi_k
            coords_epi_k = temp
        coords_epi.append(coords_epi_k)
        coords_endo.append(coords_endo_k)
    return coords_epi, coords_endo

def sort_epi_endo_coords(coords_epi, coords_endo, resolution):
    coords_epi_sorted = []
    coords_endo_sorted = []
    K = len(coords_epi)
    for k in range(K):
        coords_epi_k = coords_epi[k]
        coords_epi_k_sorted = mu.sorting_coords(coords_epi_k, resolution*np.sqrt(2))
        coords_epi_sorted.append(coords_epi_k_sorted)
        if len(coords_endo)>=K:
            coords_endo_k = coords_endo[k]
            coords_endo_k_sorted = mu.sorting_coords(coords_endo_k, resolution*np.sqrt(2))
            coords_endo_sorted.append(coords_endo_k_sorted)
    
    return coords_epi_sorted, coords_endo_sorted

# %%
def generate_pc(mesh_settings, output_folder, mask_flag, plot_flag = True):

    h5_file_address = located_h5(output_folder)
    if mask_flag:
        LVmask_raw, resolution_data = read_data_h5_mask(h5_file_address.as_posix())
        # LVmask_raw_inverted = np.transpose(LVmask_raw_reorderd, (2, 0, 1))
        # if mesh_settings["mask_is_inverted"]:
        #     LVmask_raw = LVmask_raw_inverted[::-1,:,:]
        # Filter out the slices where all I x J elements are zero, i.e., empty image
        LVmask_raw = LVmask_raw[~np.all(LVmask_raw == 0, axis=(1, 2))]        
        resolution = resolution_data[0] * 1.01
        slice_thickness = resolution_data[2]
        logger.info(f"Reading mask with slice thickness of {slice_thickness}mm and resolution of {resolution}mm")
        
        LVmask = close_apex(LVmask_raw)
        # LVmask = LVmask_raw_inverted
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
    else:
        coords, P_a_endo, P_a_epi, resolution_data  = read_data_h5(h5_file_address.as_posix())
        LV_coords_raw = restructure_coords_into_slices(coords, z_tolerance=0.1)
        LV_coords_raw = average_z_for_slices(LV_coords_raw)

        # Just saving raw data for inspection of the raw coordinates, e.g., if they form a close loop or not
        if plot_flag:
            outdir = output_folder / "01_Masks"
            outdir.mkdir(exist_ok=True)
            K = len(LV_coords_raw)
            for k in range(K):
                LVmask_k = LV_coords_raw[k]
                fnmae = outdir.as_posix() + "/" + str(k) + ".png"
                plt.scatter(LVmask_k[:,0], LVmask_k[:,1], s=1)
                plt.xlim((-50,+50))
                plt.ylim((-50,+50))
                plt.grid(True)
                plt.savefig(fnmae, dpi=300)
                plt.close()

        resolution = resolution_data[0] * 1.01
        slice_thickness = resolution_data[2]
        
        coords_epi_unsorted, coords_endo_unsorted = get_epi_endo_from_coords(LV_coords_raw, resolution)
        coords_epi, coords_endo= sort_epi_endo_coords(coords_epi_unsorted, coords_endo_unsorted, resolution)
        
        if plot_flag:
            outdir = output_folder / "01_Masks"
            outdir.mkdir(exist_ok=True)
            K = len(coords_epi)
            K_endo = len(coords_endo)
            for k in range(K):
                mask_epi_k = coords_epi[k]
                mask_endo_k = coords_endo[k]
                LVmask_k = LV_coords_raw[k]
                fnmae = outdir.as_posix() + "/" + str(k) + ".png"
                plt.scatter(LVmask_k[:,0], LVmask_k[:,1], s=1)
                plt.scatter(mask_epi_k[:,0], mask_epi_k[:,1], s=1, color = 'red')
                plt.scatter(mask_endo_k[:,0], mask_endo_k[:,1], s=1,color = 'blue')
                plt.plot(mask_epi_k[:,0], mask_epi_k[:,1], color = 'red')
                plt.plot(mask_endo_k[:,0], mask_endo_k[:,1], color = 'blue')
                plt.xlim((-50,+50))
                plt.ylim((-50,+50))
                plt.grid(True)
                plt.savefig(fnmae, dpi=300)
                plt.close()
            fnmae = outdir.as_posix() + "/" + str(k+1) + ".png"
            plt.scatter(P_a_epi[0], P_a_epi[1], s=1, color = 'red')
            plt.scatter(P_a_endo[0], P_a_endo[1], s=1, color = 'blue')
            plt.xlim((-50,+50))
            plt.ylim((-50,+50))
            plt.grid(True)
            plt.savefig(fnmae, dpi=300)
            plt.close()
    
        logger.info(f"Epi and Endo coords are extracted from point clouds")
        
        slice_thicknesses = [coords_epi[k][0,2]-coords_epi[k+1][0,2] for k in range(len(coords_epi)-1)]
        slice_thickness_ave = np.mean(slice_thicknesses)
        logger.warning(f"Slice thickness is {slice_thickness} while the averaged based on coords is {slice_thickness_ave}")
        
    
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
        
        fig = utils.plotly_3d_base_splines(tck_epi, color = "red")
        fig = utils.plotly_3d_base_splines(tck_endo, fig=fig, color="blue")
        fnmae = outdir.as_posix() + "/shax_bsplines.html"
        fig.write_html(fnmae)
    sample_points_epi = mu.get_sample_points_from_shax(
        tck_epi, mesh_settings["num_lax_points"]
    )
    sample_points_endo = mu.get_sample_points_from_shax(
        tck_endo, mesh_settings["num_lax_points"]
    )

    apex_threshold = mu.get_apex_threshold(sample_points_epi, sample_points_endo)
    logger.info("Using slice thickness average for lax points")
    if mask_flag:
        P_a_epi = None
        P_a_endo = None

    LAX_points_epi, apex_epi = mu.create_lax_points(
        sample_points_epi, apex_threshold, slice_thickness, apex_coord=P_a_epi
    )
    LAX_points_endo, apex_endo = mu.create_lax_points(
        sample_points_endo, apex_threshold, slice_thickness, apex_coord=P_a_endo
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

    z_base = coords_epi[0][0,2]
    tck_shax_epi = mu.get_shax_from_lax(
        tck_lax_epi,
        apex_epi,
        mesh_settings["num_z_sections_epi"],
        z_sections_flag = mesh_settings["z_sections_flag_epi"],
        z_base=z_base
    )
    tck_shax_endo = mu.get_shax_from_lax(
        tck_lax_endo,
        apex_endo,
        mesh_settings["num_z_sections_endo"],
        z_sections_flag = mesh_settings["z_sections_flag_endo"],
        z_base=z_base
    )
    if plot_flag:
        outdir = output_folder / "04_Contours"
        outdir.mkdir(exist_ok=True)
        fig = go.Figure()
        fig = mu.plotly_3d_contours(
            fig, tck_shax_epi, tck_lax_epi, tck_shax_endo, tck_lax_endo
        )
        # Adding original coords
        x = np.concatenate(coords_epi)[:,0]
        y = np.concatenate(coords_epi)[:,1]
        z = np.concatenate(coords_epi)[:,2]
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, marker=dict(size = 2, color="red")))
        x = np.concatenate(coords_endo)[:,0]
        y = np.concatenate(coords_endo)[:,1]
        z = np.concatenate(coords_endo)[:,2]
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, marker=dict(size = 2, color="blue")))
        fnmae = outdir.as_posix() + "/Contour.html"
        fig.write_html(fnmae)
    
    points_cloud_epi = mu.get_sample_points_from_shax(
        tck_shax_epi, mesh_settings["seed_num_base_epi"]
    )
    points_cloud_endo = mu.get_sample_points_from_shax(
        tck_shax_endo, mesh_settings["seed_num_base_endo"]
    )
    
    points_cloud_epi.append(apex_epi)
    points_cloud_endo.append(apex_endo)
    
    if plot_flag:
        outdir = output_folder / "05_Point Cloud"
        outdir.mkdir(exist_ok=True)
        fig = go.Figure()
        for points in points_cloud_epi:
            mu.plot_3d_points_on_figure(points, fig=fig)
        fnmae = outdir.as_posix() + "/Points_cloud_epi.html"
        fig.write_html(fnmae)
        fig = go.Figure()
        for points in points_cloud_endo:
            mu.plot_3d_points_on_figure(points, fig=fig)
        fnmae = outdir.as_posix() + "/Points_cloud_endo.html"
        fig.write_html(fnmae)
    
    return points_cloud_epi, points_cloud_endo


def generate_3d_mesh(
    points_cloud_epi, 
    points_cloud_endo, 
    outdir,
    k_apex_endo=18,
    k_apex_epi=18,
    SurfaceMeshSizeEpi=5,
    SurfaceMeshSizeEndo=5,
    MeshSizeMin=5,
    MeshSizeMax=10,
    num_mid_layers_base=3):
    # Calculate normals
    normals_list_endo = mu.calculate_normals(points_cloud_endo, k_apex_endo)
    normals_list_epi = mu.calculate_normals(points_cloud_epi, k_apex_epi)
    outdir = outdir / "06_Mesh"
    outdir.mkdir(exist_ok=True)
    
    mesh_epi_filename, mesh_endo_filename, mesh_base_filename = mu.VentricMesh_poisson(
        points_cloud_epi,
        points_cloud_endo,
        num_mid_layers_base,
        SurfaceMeshSizeEpi=SurfaceMeshSizeEpi,
        SurfaceMeshSizeEndo=SurfaceMeshSizeEndo,
        normals_list_epi = normals_list_epi,
        normals_list_endo = normals_list_endo,
        save_flag=True,
        filename_suffix="",
        result_folder=outdir.as_posix() + "/",
    )
    output_mesh_filename = outdir / 'Mesh_3D.msh'
    
    try:
        mu.generate_3d_mesh_from_seperate_stl(mesh_epi_filename, mesh_endo_filename, mesh_base_filename, output_mesh_filename.as_posix(),  MeshSizeMin=MeshSizeMin, MeshSizeMax=MeshSizeMax)
        
        # Read the .msh file and write to the vtk format
        mesh = meshio.read(output_mesh_filename)
        output_mesh_filename_vtk = outdir / 'Mesh_3D.vtk'
        meshio.write(output_mesh_filename_vtk, mesh)
    except Exception as e:
        error_str = str(e)
        if "No elements in volume 1" in error_str: 
            logger.error("3D volumetric mesh not generated, if needed try to check base and apex")
        else:
            # If it’s some other exception, re-raise so we don’t mask a different issue
            raise        
    
    return mesh_epi_filename, mesh_endo_filename, mesh_base_filename

def create_mesh_slice_by_slice(point_cloud, scale=1.5):
    vertices = []
    faces = []
    num_shax = len(point_cloud) - 2
    for k in range(num_shax):
        slice1 = np.array(point_cloud[k])
        slice2 = np.array(point_cloud[k + 1])
        slice_faces = mu.create_slice_mesh(slice1, slice2, scale)
        faces_offset = sum(map(len, vertices))
        faces.append(slice_faces + faces_offset)
        vertices.append(point_cloud[k])
    base_faces = mu.create_slice_mesh(slice2, point_cloud[-1], 1)
    base_face_offset = sum(map(len, vertices))
    faces.append(base_faces + base_face_offset)
    faces = np.vstack(faces)
    return np.vstack(point_cloud), faces

def generate_mesh_delauny(
    points_cloud_epi, 
    points_cloud_endo, 
    outdir,
    ):
    outdir = outdir / "06_Mesh_delauny"
    outdir.mkdir(exist_ok=True)
    vertices_epi, faces_epi = create_mesh_slice_by_slice(points_cloud_epi, scale=1.5)
    vertices_endo, faces_endo = create_mesh_slice_by_slice(points_cloud_endo, scale=1.5)
    
    mesh_epi = mu.create_mesh(vertices_epi, faces_epi)
    mesh_epi_filename = outdir / 'Mesh_epi.stl'
    mesh_epi.save(mesh_epi_filename)
    mesh_endo = mu.create_mesh(vertices_endo, faces_endo)
    mesh_endo_filename = outdir / 'Mesh_endo.stl'
    mesh_endo.save(mesh_endo_filename)
    
    return mesh_epi_filename, mesh_endo_filename

def calculate_mesh_error(mesh_epi_filename, mesh_endo_filename, coords_epi, coords_endo, outdir, resolution):
    fig = utils.plot_coords_and_mesh(coords_epi, coords_endo, mesh_epi_filename, mesh_endo_filename)
    fname = outdir.as_posix() + "/Mesh_vs_Coords.html"
    fig.write_html(fname)
    
    # Preparing error reports
    # making error report 
    errors_epi = utils.calculate_error_between_coords_and_mesh(coords_epi, mesh_epi_filename)
    errors_endo = utils.calculate_error_between_coords_and_mesh(coords_endo, mesh_endo_filename)

    return errors_epi, errors_endo

def get_xylim_for_error_hist(errors_epi, errors_endo):
    all_errors = np.concatenate([errors_epi, errors_endo])
    xlim = (np.min(all_errors), np.max(all_errors))
    hist_epi, _ = np.histogram(errors_epi, bins=30)
    hist_endo, _ = np.histogram(errors_endo, bins=30)
    max_y = max(np.max(hist_epi), np.max(hist_endo))
    ylim = (0, max_y + max_y * 0.1)  # Add 10% padding for aesthetics
    return xlim, ylim

def export_error_stats(errors_epi, errors_endo, outdir, resolution):
    xlim, ylim = get_xylim_for_error_hist(errors_epi, errors_endo)
     
    fname_epi = outdir / "Epi_mesh_errors.png"
    fname_endo = outdir / "Endo_mesh_errors.png"

    utils.plot_error_histogram(
        errors=errors_epi,
        fname=fname_epi,
        color='red',
        xlim=xlim,
        ylim=ylim,
        title_prefix='Epi', 
        resolution=resolution
    )

    utils.plot_error_histogram(
        errors=errors_endo,
        fname=fname_endo,
        color='blue',
        xlim=xlim,
        ylim=ylim,
        title_prefix='Endo', 
        resolution=resolution
    )
    fname_epi = fname_epi.as_posix()[:-4] + ".txt"
    utils.save_error_distribution_report(errors_epi,fname_epi, n_bins=10, surface_name="Epicardium", resolution=resolution)
    fname_endo = fname_endo.as_posix()[:-4] + ".txt"
    utils.save_error_distribution_report(errors_endo, fname_endo, n_bins=10,  surface_name="Endocardium", resolution=resolution)

def save_errors_to_dataframe(csv_file, sample_name, errors_epi, errors_endo, resolution):
    """
    Save or append errors to a CSV file using pandas for easier manipulation later.
    
    Args:
        csv_file (str or Path): Path to the CSV file.
        sample_name (str): Name of the current sample.
        errors_epi (array-like): Array of errors for the epicardial surface.
        errors_endo (array-like): Array of errors for the endocardial surface.
        resolution (float): Resolution to normalize the errors.
    """
    # Prepare data as a DataFrame
    data = {
        "Case Name": [sample_name],
        "Mean Epi Error": [np.mean(errors_epi) / resolution],
        "Mean Endo Error": [np.mean(errors_endo) / resolution],
        "Std Epi Error": [np.std(errors_epi) / resolution],
        "Std Endo Error": [np.std(errors_endo) / resolution],
    }
    df = pd.DataFrame(data)

    # Append to the file or create a new one
    try:
        if not csv_file.exists():
            df.to_csv(csv_file, index=False)
        else:
            df.to_csv(csv_file, mode='a', header=False, index=False)
        print(f"Data successfully saved to {csv_file}")
    except Exception as e:
        print(f"Error saving data to {csv_file}: {e}")
    