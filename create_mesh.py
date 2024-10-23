from pathlib import Path
import argparse
import json
from structlog import get_logger

import meshing_utils

logger = get_logger()

def load_settings(setting_dir, sample_name):
    settings_fname = setting_dir / f"{sample_name}.json"
    with open(settings_fname, "r") as file:
        settings = json.load(file)
    return settings

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
    
    settings = load_settings(settings_dir, sample_name)
    mesh_settings = settings["mesh"][mesh_quality]

    sample_directory = data_directory / sample_name
    output_mesh_filename = meshing_utils.create_mesh(mesh_settings, sample_directory, output_folder)


if __name__ == "__main__":
    main()
