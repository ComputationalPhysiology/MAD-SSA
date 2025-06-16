from pathlib import Path
import json
import numpy as np
import os
from structlog import get_logger
import ventric_pca

logger = get_logger()

def load_settings(setting_dir, sample_name):
    settings_fname = setting_dir / f"{sample_name}.json"
    with open(settings_fname, "r") as file:
        settings = json.load(file)
    return settings

def process_patient(sample_name, data_directory, settings_dir, output_folder, mesh_quality, mask_flag):
    settings = load_settings(settings_dir, sample_name)
    mesh_settings = settings["mesh"][mesh_quality]

    sample_directory = data_directory / sample_name
  
    points_cloud_epi, points_cloud_endo = ventric_pca.meshing_utils.generate_pc(
        mesh_settings, sample_directory, output_folder, mask_flag
    )

    points_cloud_epi = np.vstack(points_cloud_epi)
    points_cloud_endo = np.vstack(points_cloud_endo)
    
    output_folder = Path((sample_directory / output_folder))
    os.makedirs(output_folder, exist_ok=True)
    np.savetxt(output_folder.as_posix()+'/points_cloud_epi.csv', points_cloud_epi, delimiter=',', fmt='%.8f')
    np.savetxt(output_folder.as_posix()+'/points_cloud_endo.csv', points_cloud_endo, delimiter=',', fmt='%.8f')
    logger.info(f"Point cloud generated for patient: {sample_name}")

