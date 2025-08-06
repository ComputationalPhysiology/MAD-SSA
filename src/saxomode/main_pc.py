import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from saxomode.utils import process_files_from_directory, create_patient_settings
from saxomode.create_point_cloud import process_patient
import shutil  

def run_analysis(
        name=None, 
        data_directory=os.getcwd()+'/seg_files/',
        settings_dir=os.getcwd()+'/settings/', 
        outdir=os.getcwd()+'/results/', 
        mesh_quality='fine', 
        mask_flag=True,
        sf_epi=False,
        sf_endo=False):
    
    if name:
        
        patient_name = name
        create_patient_settings(patient_name, settings_dir,lax_smooth_level_endo=sf_endo, lax_smooth_level_epi=sf_epi)
        patient_folder = Path(outdir)/patient_name
        patient_folder.mkdir(parents=True, exist_ok=True)
        h5_file = Path(data_directory) / f"{patient_name}_original_segmentation.h5"
        
        if h5_file.exists():
            shutil.copy(h5_file, patient_folder)
        else:
            print(f"Warning: .h5 file for patient {patient_name} not found.")
        process_patient(name, settings_dir, patient_folder, mesh_quality, mask_flag=True)
        
    else:
        process_files_from_directory(directory_path=data_directory, settings_folder=settings_dir)
        
        for filename in os.listdir(data_directory):
            if filename.endswith(".h5") and filename[:3].isdigit():
                patient_name = filename[:3]
                
                patient_folder = Path(outdir)/patient_name
                patient_folder.mkdir(parents=True, exist_ok=True)
                
                h5_file = Path(data_directory) / f"{patient_name}_original_segmentation.h5"
                
                if h5_file.exists():
                    shutil.copy(h5_file, patient_folder)
             
                else:
                    print(f"Warning: .h5 file for patient {patient_name} not found.")
                process_patient(patient_name, settings_dir, patient_folder, mesh_quality, mask_flag=True)


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
        default=os.getcwd()+'/seg_files/',
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
        "--outdir",
        default=os.getcwd()+'/results/',
        type=str,
        help="The result folder name that would be created in the directory of the sample.",
    )
    parser.add_argument(
        "-sfepi",
        action="store_true",
        help="assign new smoothing factor for lax epi",
    )
    parser.add_argument(
        "-sfendo",
        action="store_true",
        help="assign new smoothing factor for lax endo",
    )
    args = parser.parse_args(args)

    name = args.name
    data_directory = args.data_directory
    settings_dir = args.settings_dir
    outdir = args.outdir
    mesh_quality = args.mesh_quality
    mask_flag = args.mask
    sf_epi = args.sfepi
    sf_endo = args.sfedno

    run_analysis(name, data_directory, settings_dir, outdir, mesh_quality, mask_flag, sf_epi, sf_endo)

    
if __name__ == "__main__":
    main()
