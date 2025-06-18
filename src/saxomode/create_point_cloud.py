from pathlib import Path
import json
import numpy as np
import os
from structlog import get_logger
import saxomode

logger = get_logger()

def load_settings(setting_dir, sample_name):
    settings_fname = Path(setting_dir) / f"{sample_name}.json"
    with open(settings_fname, "r") as file:
        settings = json.load(file)
    return settings

def process_patient(sample_name, settings_dir, patient_folder, mesh_quality, mask_flag):
    settings = load_settings(settings_dir, sample_name)
    mesh_settings = settings["mesh"][mesh_quality]


    points_cloud_epi, points_cloud_endo = saxomode.meshing_utils.generate_pc(
        mesh_settings, patient_folder, mask_flag
    )

    points_cloud_epi = np.vstack(points_cloud_epi)
    points_cloud_endo = np.vstack(points_cloud_endo)
    np.savetxt(patient_folder.as_posix()+'/points_cloud_epi.csv', points_cloud_epi, delimiter=',', fmt='%.8f')
    np.savetxt(patient_folder.as_posix()+'/points_cloud_endo.csv', points_cloud_endo, delimiter=',', fmt='%.8f')
    logger.info(f"Point cloud generated for patient: {sample_name}")

