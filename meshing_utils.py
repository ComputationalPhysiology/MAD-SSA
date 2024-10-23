# %%
import h5py
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import cv2 as cv

import ventric_mesh.mesh_utils as mu
import ventric_mesh.utils as utils
import matplotlib.pyplot as plt
from structlog import get_logger

logger = get_logger()


# %%
def read_data_h5(file_dir):
    with h5py.File(file_dir, "r") as h5_file:
        LVmask = h5_file["LVmask"][:]
        slice_thickness = h5_file["slice_thickness"][0]
        resolution = h5_file["resolution"][0]
    return LVmask, slice_thickness, resolution


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


# %%
def create_mesh(mesh_settings, sample_directory, output_folder):
    output_folder = Path((sample_directory / "00_results"))
    output_folder.mkdir(exist_ok=True, parents=True)

    h5_file_address = located_h5(sample_directory)
    LVmask_raw, slice_thickness, resolution = read_data_h5(h5_file_address.as_posix())
    LVmask = close_apex(LVmask_raw)
    logger.info("Mask is loaded and apex is closed")

    mask_epi, mask_endo = mu.get_endo_epi(LVmask)

    coords_epi = mu.get_coords_from_mask(mask_epi, resolution, slice_thickness)
    coords_endo = mu.get_coords_from_mask(mask_endo, resolution, slice_thickness)

    tck_epi = mu.get_shax_from_coords(
        coords_epi, mesh_settings["smooth_level_epi"]
    )
    tck_endo = mu.get_shax_from_coords(
        coords_endo, mesh_settings["smooth_level_endo"]
    )
    K = len(tck_epi)
    plot_flag = True

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
    points_cloud_epi, k_apex_epi = mu.create_point_cloud(
        tck_shax_epi,
        apex_epi,
        mesh_settings["seed_num_base_epi"],
        seed_num_threshold=mesh_settings["seed_num_threshold_epi"],
    )
    points_cloud_endo, k_apex_endo = mu.create_point_cloud(
        tck_shax_endo,
        apex_endo,
        mesh_settings["seed_num_base_endo"],
        seed_num_threshold=mesh_settings["seed_num_threshold_endo"],
    )
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
    mesh_dir = output_folder / "06_Mesh"
    mesh_dir.mkdir(exist_ok=True, parents=True)
    LVmesh, mesh_epi, mesh_endo, mesh_base = mu.VentricMesh_poisson(
        points_cloud_epi,
        points_cloud_endo,
        mesh_settings["num_mid_layers_base"],
        SurfaceMeshSizeEpi=mesh_settings["SurfaceMeshSizeEpi"],
        SurfaceMeshSizeEndo=mesh_settings["SurfaceMeshSizeEndo"],
        MeshSizeMax=mesh_settings["MeshSizeMax"],
        MeshSizeMin=mesh_settings["MeshSizeMin"],
        save_flag=True,
        filename_suffix="",
        result_folder=mesh_dir.as_posix() + "/",
    )
    # utils.check_mesh_quality(LVmesh, file_path=outdir.as_posix() + "/Mesh_report.txt")
    if plot_flag:
        fig = utils.plot_coords_and_mesh(coords_epi, coords_endo, mesh_epi, mesh_endo)
        fname = mesh_dir.as_posix() + "/Mesh_vs_Coords.html"
        fig.write_html(fname)
        
    
    errors_epi = utils.calculate_error_between_coords_and_mesh(coords_epi, mesh_epi)
    errors_endo = utils.calculate_error_between_coords_and_mesh(coords_endo, mesh_endo)
    
    all_errors = np.concatenate([errors_epi, errors_endo])
    xlim = (np.min(all_errors), np.max(all_errors))
    hist_epi, _ = np.histogram(errors_epi, bins=30)
    hist_endo, _ = np.histogram(errors_endo, bins=30)
    max_y = max(np.max(hist_epi), np.max(hist_endo))
    ylim = (0, max_y + max_y * 0.1)  # Add 10% padding for aesthetics

    fname_epi = mesh_dir / "Epi_mesh_errors.png"
    fname_endo = mesh_dir / "Endo_mesh_errors.png"

    utils.plot_error_histogram(
        errors=errors_epi,
        fname=fname_epi,
        color='red',
        xlim=xlim,
        ylim=ylim,
        title_prefix='Epi'
    )

    utils.plot_error_histogram(
        errors=errors_endo,
        fname=fname_endo,
        color='blue',
        xlim=xlim,
        ylim=ylim,
        title_prefix='Endo'
    )
    fname_epi = fname_epi.as_posix()[:-4] + ".txt"
    utils.save_error_distribution_report(errors_epi,fname_epi, n_bins=10, surface_name="Epicardium")
    fname_endo = fname_endo.as_posix()[:-4] + ".txt"
    utils.save_error_distribution_report(errors_endo, fname_endo, n_bins=10,  surface_name="Endocardium")
    return LVmesh
