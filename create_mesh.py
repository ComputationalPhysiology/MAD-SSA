from pathlib import Path
import argparse
from structlog import get_logger

import meshing_utils

logger = get_logger()

# %%
data_address = Path("/home/shared/00_data/MAD_4/")


def get_default_mesh_settings():
    return {
        "seed_num_base_epi": 100,
        "seed_num_base_endo": 100,
        "num_z_sections_epi": 50,
        "num_z_sections_endo": 50,
        "num_mid_layers_base": 7,
        "smooth_level_epi": 0.1,
        "smooth_level_endo": 0.1,
        "num_lax_points": 64,
        "lax_smooth_level_epi": 20,
        "lax_smooth_level_endo": 20,
        "z_sections_flag_epi": 0,
        "z_sections_flag_endo": 0,
        "seed_num_threshold_epi": 25,
        "seed_num_threshold_endo": 20,
        "scale_for_delauny": 1.5,
        "t_mesh": -1,
        "MeshSizeMin": 0.5,
        "MeshSizeMax": 1,
        "SurfaceMeshSizeEndo": 2,
        "SurfaceMeshSizeEpi": 2,
    }


def main(args=None) -> int:
    """
    Parse the command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Data directory parameters
    parser.add_argument(
        "-n",
        "--name",
        default="MAD_4",
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

    mesh_settings = get_default_mesh_settings()

    sample_directory = data_directory / sample_name
    LVmesh = meshing_utils.create_mesh(mesh_settings, sample_directory, output_folder)


if __name__ == "__main__":
    main()
