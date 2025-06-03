import os
import numpy as np

directory = "/home/shared/MAD-SSA/FINAL_ALIGNMENT_base"
new_directory = "/home/shared/MAD-SSA/FINAL_ALIGNMENT_base/merged_points"
os.makedirs(new_directory, exist_ok=True)
files = [f for f in os.listdir(directory) if f.endswith('.txt')]

patient_files = {}

# Group files by patient ID
for file in files:
    patient_id = file.split('_')[1]
    if patient_id not in patient_files:
        patient_files[patient_id] = {}
    if 'epi' in file:
        patient_files[patient_id]['epi'] = file
    elif 'endo' in file:
        patient_files[patient_id]['endo'] = file

# Merge epi and endo points for each patient and save to a new file
for patient_id, file_dict in patient_files.items():
    if 'epi' in file_dict and 'endo' in file_dict:
        epi_file = os.path.join(directory, file_dict['epi'])
        endo_file = os.path.join(directory, file_dict['endo'])
        
        epi_points = np.loadtxt(epi_file, delimiter=',')
        endo_points = np.loadtxt(endo_file, delimiter=',')
        
        merged_points = np.vstack((epi_points, endo_points))
        
        output_file = os.path.join(new_directory, f"MAD_{patient_id}_merged_points.txt")
        np.savetxt(output_file, merged_points, delimiter=',')
        print(f"Merged points for patient {patient_id} saved to {output_file}")