from pathlib import Path
import argparse
import json
import numpy as np

from structlog import get_logger

import meshing_utils
import ventric_mesh.mesh_utils as mu

logger = get_logger()

def load_settings(setting_dir, sample_name):
    settings_fname = setting_dir / f"{sample_name}.json"
    with open(settings_fname, "r") as file:
        settings = json.load(file)
    return settings

def load_original_data(sample_directory):
    h5_file_address = meshing_utils.located_h5(sample_directory)
    LVmask_raw, resolution_data = meshing_utils.read_data_h5_mask(h5_file_address.as_posix())
    LVmask_raw = LVmask_raw[~np.all(LVmask_raw == 0, axis=(1, 2))]        
    resolution = resolution_data[0]
    slice_thickness = resolution_data[2]
    logger.info(f"Reading mask with slice thickness of {slice_thickness}mm and resolution of {resolution}mm")
        
    LVmask = meshing_utils.close_apex(LVmask_raw)
    logger.info("Mask is loaded and apex is closed")

    mask_epi, mask_endo = mu.get_endo_epi(LVmask)
    coords_epi = mu.get_coords_from_mask(mask_epi, resolution, slice_thickness)
    coords_endo = mu.get_coords_from_mask(mask_endo, resolution, slice_thickness)
    
    return coords_epi, coords_endo, resolution

def main(args=None) -> int:
    """
    Parse the command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Data directory parameters
    parser.add_argument(
        "-n",
        "--name",
        default="MAD_1",
        type=str,
        help="The sample file name to be processed",
    )

    parser.add_argument(
        "-d",
        "--data_directory",
        default="/home/shared/00_data",
        type=Path,
        help="The directory where all the patients data stored.",
    )
    
    parser.add_argument(
        "--settings_dir",
        default="/home/shared/MAD-SSA/settings",
        type=Path,
        help="The settings directory where json files are stored.",
    )
    
    parser.add_argument(
        "-m",
        "--mesh_quality",
        default='fine',
        type=str,
        help="The mesh quality. Settings will be loaded accordingly from json file",
    )
    
    # flag for using mask or coords
    parser.add_argument(
        "-mask",
        action="store_true",
        help="The flag for whether using mask or not",
    )

    parser.add_argument(
        "-o",
        "--output_folder",
        default="00_results",
        type=str,
        help="The result folder name that would be created in the directory of the sample.",
    )
    args = parser.parse_args(args)

    sample_name = args.name
    data_directory = args.data_directory
    output_folder = args.output_folder
    settings_dir = args.settings_dir
    mesh_quality = args.mesh_quality
    mask_flag = args.mask
    settings = load_settings(settings_dir, sample_name)
    mesh_settings = settings["mesh"][mesh_quality]

    sample_directory = data_directory / sample_name
    points_cloud_epi, points_cloud_endo = meshing_utils.generate_pc(mesh_settings, sample_directory, output_folder, mask_flag, plot_flag=False)
    mesh_epi_fname, mesh_endo_fname, mesh_base_fname = meshing_utils.generate_3d_mesh(points_cloud_epi, points_cloud_endo, sample_directory, output_folder)

        
    all_errors = np.concatenate([errors_epi, errors_endo])
    xlim = (np.min(all_errors), np.max(all_errors))
    hist_epi, _ = np.histogram(errors_epi, bins=30)
    hist_endo, _ = np.histogram(errors_endo, bins=30)
    max_y = max(np.max(hist_epi), np.max(hist_endo))
    ylim = (0, max_y + max_y * 0.1)  # Add 10% padding for aesthetics

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
    

    
if __name__ == "__main__":
    main()
