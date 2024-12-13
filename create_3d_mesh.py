from pathlib import Path
import argparse
import json
import numpy as np

from structlog import get_logger

import meshing_utils
import ventric_mesh.mesh_utils as mu

logger = get_logger()

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

def convert_pc_to_stack(pc, num_z_sections=20):
    num_points_per_z = int((pc.shape[0] - 1) / num_z_sections)
    pc_list = []
    # Create the list of z sections
    for i in range(num_z_sections):  
        pc_list.append(pc[i * num_points_per_z:(i + 1) * num_points_per_z])

    # Add the last element with the remaining points 
    pc_list.append(pc[-1])
    
    return pc_list

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
        "--SurfaceMeshSizeEpi",
        default=3,
        type=float,
        help="The size of generated surface mesh on epi surface",
    )
    
    parser.add_argument(
        "--SurfaceMeshSizeEndo",
        default=3,
        type=float,
        help="The size of generated surface mesh on endo surface",
    )
    
    parser.add_argument(
        "--VolumeMeshSizeMin",
        default=5,
        type=float,
        help="The minimum size of generated volumetric (3D) mesh",
    )
    
    
    parser.add_argument(
        "--VolumeMeshSizeMax",
        default=10,
        type=float,
        help="The minimum size of generated volumetric (3D) mesh",
    )
    
    # flag for processing modes
    parser.add_argument(
        "-m",
        action="store_true",
        help="The flag for whether using mode data or not",
    )
    
    parser.add_argument(
        "--mode_numbers",
        nargs="+",
        type=int,
        default=1,
        help="The flag for whether using mode data or not",
    )
    
    parser.add_argument(
        "--mode_folder",
        default="modes",
        type=str,
        help="The folder containing mode data coordinates.",
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
    mode_flag = args.m
    mode_folder = args.mode_folder
    mode_numbers = args.mode_numbers
    SurfaceMeshSizeEndo = args.SurfaceMeshSizeEndo
    SurfaceMeshSizeEpi = args.SurfaceMeshSizeEpi
    VolumeMeshSizeMin = args.VolumeMeshSizeMin
    VolumeMeshSizeMax = args.VolumeMeshSizeMax

    if mode_flag:
        sample_directory = data_directory / mode_folder
        modes = sorted(sample_directory.glob("*.txt"))
        # the name of mode_numbers is misleading here we deal each file as a mode which is not correct!
        selected_modes = [modes[i - 1] for i in mode_numbers]
        for mode in selected_modes:
            logger.info(f"Mode {mode.stem} is being analysed ...")
            logger.info(f"--------------------------------------")
            # Load the saved point cloud data
            mode_pc = np.loadtxt(mode.as_posix(), delimiter=',')
            points_cloud_epi = mode_pc[:800]
            points_cloud_endo = mode_pc[800:]
            # Convering the np.array to list of np.array with number of z setions (slices)
            points_cloud_epi = convert_pc_to_stack(points_cloud_epi, num_z_sections=20)
            points_cloud_endo = convert_pc_to_stack(points_cloud_endo, num_z_sections=20)
            # Creating 3D and surface meshes of epi, endo and base
            outdir = mode.parent / f"00_results_{mode.stem}"
            outdir.mkdir(exist_ok=True)
            mesh_epi_fname, mesh_endo_fname, _ = meshing_utils.generate_3d_mesh(points_cloud_epi, points_cloud_endo, outdir, SurfaceMeshSizeEndo=SurfaceMeshSizeEndo, SurfaceMeshSizeEpi=SurfaceMeshSizeEpi, MeshSizeMin=VolumeMeshSizeMin, MeshSizeMax=VolumeMeshSizeMax)
            # calculating the error between raw data and surfaces meshes of epi and endo
            resolution = 0
            errors_epi, errors_endo = meshing_utils.calculate_mesh_error(mesh_epi_fname, mesh_endo_fname, points_cloud_epi[:-1], points_cloud_endo[:-1], outdir, resolution)
            meshing_utils.export_error_stats(errors_epi, errors_endo, outdir, resolution)
    else:
        sample_directory = data_directory / sample_name
        # Load raw data from mask and resolution
        coords_epi, coords_endo, resolution = load_original_data(sample_directory)
        # Load the saved point cloud data
        outdir = sample_directory / output_folder
        points_cloud_epi = np.loadtxt(outdir / 'points_cloud_epi.csv', delimiter=',')
        points_cloud_endo = np.loadtxt(outdir / 'points_cloud_endo.csv', delimiter=',')
        # Convering the np.array to list of np.array with number of z setions (slices)
        points_cloud_epi = convert_pc_to_stack(points_cloud_epi, num_z_sections=20)
        points_cloud_endo = convert_pc_to_stack(points_cloud_endo, num_z_sections=20)
        # Creating 3D and surface meshes of epi, endo and base
        mesh_epi_fname, mesh_endo_fname, _ = meshing_utils.generate_3d_mesh(points_cloud_epi, points_cloud_endo, outdir, SurfaceMeshSizeEndo=SurfaceMeshSizeEndo, SurfaceMeshSizeEpi=SurfaceMeshSizeEpi, MeshSizeMin=VolumeMeshSizeMin, MeshSizeMax=VolumeMeshSizeMax)
        # calculating the error between raw data and surfaces meshes of epi and endo
        errors_epi, errors_endo = meshing_utils.calculate_mesh_error(mesh_epi_fname, mesh_endo_fname, coords_epi, coords_endo, outdir, resolution)
        meshing_utils.export_error_stats(errors_epi, errors_endo, outdir, resolution)
    
if __name__ == "__main__":
    main()
