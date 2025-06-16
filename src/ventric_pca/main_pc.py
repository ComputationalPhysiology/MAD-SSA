import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import ventric_pca.config as config 
from ventric_pca.utils import process_files_from_directory
from ventric_pca.create_point_cloud import process_patient
import shutil  

def main(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-n",
        "--name",
        default=None,
        type=str,
        help="The sample file name to be processed. If not provided, all patients in the directory will be processed.",
    )

    parser.add_argument(
        "-d",
        "--data_directory",
        default=os.getcwd()+'/data/',
        type=Path,
        help="The directory where all the patients' data is stored.",
    )
    
    parser.add_argument(
        "--settings_dir",
        default=os.getcwd()+'/settings/',
        type=Path,
        help="The settings directory where JSON files are stored.",
    )
    
    parser.add_argument(
        "-m",
        "--mesh_quality",
        default='fine',
        type=str,
        help="The mesh quality. Settings will be loaded accordingly from the JSON file.",
    )
    
    parser.add_argument(
        "-mask",
        action="store_true",
        help="The flag for whether using mask or not.",
    )

    parser.add_argument(
        "-o",
        "--output_folder",
        default="00_results",
        type=str,
        help="The result folder name that would be created in the directory of the sample.",
    )
    args = parser.parse_args(args)

    # Step 1: Process files to create settings
    process_files_from_directory(directory_path=args.data_directory, settings_folder=args.settings_dir)
    # Step 2: Generate point clouds
    data_directory = args.data_directory
    settings_dir = args.settings_dir
    output_folder = args.output_folder
    mesh_quality = args.mesh_quality
    data_root = Path("/home/shared/controls/00_data")

    if args.name:
        # Create folder for a single patient
        patient_name = args.name
        patient_folder = Path(data_root)/patient_name
        
        results_folder = patient_folder/output_folder
        
        patient_folder.mkdir(parents=True, exist_ok=True)
        results_folder.mkdir(parents=True, exist_ok=True)

        # Copy the corresponding .h5 file
        h5_file = Path(data_directory) / f"{patient_name}_original_segmentation.h5"
        if h5_file.exists():
            shutil.copy(h5_file, patient_folder)
        else:
            print(f"Warning: .h5 file for patient {patient_name} not found.")
        process_patient(args.name, data_root, settings_dir, results_folder, mesh_quality, mask_flag=True)
        
    else:
        # Create folders for all patients
        for filename in os.listdir(data_directory):
            if filename.endswith(".h5") and filename[:3].isdigit():
                patient_name = filename[:3]
                patient_folder = Path(data_root)/patient_name
        
                results_folder = patient_folder/output_folder
                
                patient_folder.mkdir(parents=True, exist_ok=True)
                results_folder.mkdir(parents=True, exist_ok=True)
                h5_file = Path(data_directory) / f"{patient_name}_original_segmentation.h5"
            if h5_file.exists():
                shutil.copy(h5_file, patient_folder)
            else:
                print(f"Warning: .h5 file for patient {patient_name} not found.")
            process_patient(patient_name, data_root, settings_dir, results_folder, mesh_quality, mask_flag=True)

if __name__ == "__main__":
    main()
