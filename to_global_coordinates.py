
import matplotlib.pyplot as plt
import numpy as np
from pydicom import dcmread
import cv2
from pathlib import Path
import natsort
import json
import h5py
import nibabel as nib
import plotly.graph_objects as go
import os
from scipy.ndimage import zoom
import pandas as pd
from dataclasses import dataclass, field, InitVar
# ---------------- Helper Functions ----------------

def get_pixel_coordinates(dcm_file):
    """Calculate physical coordinates for each pixel in a DICOM image."""
    img = dcm_file.pixel_array
    img_dim = img.shape
    ipp = dcm_file.ImagePositionPatient
    iop = dcm_file.ImageOrientationPatient
    ps = dcm_file.PixelSpacing

    X_iop, Y_iop = np.asarray(iop[:3], dtype=float), np.asarray(iop[3:], dtype=float)
    
    # Transformation matrix for pixel to physical coordinates
    A = np.array([
        [X_iop[0] * ps[0], Y_iop[0] * ps[1], 0, ipp[0]],
        [X_iop[1] * ps[0], Y_iop[1] * ps[1], 0, ipp[1]],
        [X_iop[2] * ps[0], Y_iop[2] * ps[1], 0, ipp[2]]
    ])

    # Apply transformation
    P_out = np.zeros((img_dim[0], img_dim[1], 4))
    for i in range(img_dim[1]):
        for j in range(img_dim[0]):
            B = [[i], [j], [0], [1]]
            P_out[j, i, :] = np.vstack((np.matmul(A, B), img[j, i])).T
    return P_out

def get_3d_coordinates(dcm_file, i, j):
    """Return 3D coordinates for a given pixel location."""
    ipp, iop = dcm_file.ImagePositionPatient, dcm_file.ImageOrientationPatient
    ps = dcm_file.PixelSpacing
    X_iop, Y_iop = np.asarray(iop[:3], dtype=float), np.asarray(iop[3:], dtype=float)

    A = np.array([
        [X_iop[0] * ps[0], Y_iop[0] * ps[1], 0, ipp[0]],
        [X_iop[1] * ps[0], Y_iop[1] * ps[1], 0, ipp[1]],
        [X_iop[2] * ps[0], Y_iop[2] * ps[1], 0, ipp[2]]
    ])
    return np.matmul(A, np.array([[i], [j], [0], [1]])).flatten()[:3]

def load_lax(dcm_folder, target_frame_num):
    """Load a single frame from the LAX DICOM folder."""
    files = natsort.natsorted([f.name for f in dcm_folder.iterdir()])
    return [files[target_frame_num]]

def load_annotations(filepath):
    """Load annotation data from JSON file."""
    with open(filepath, "r") as json_file:
        data = json.load(json_file)
    return [PatientData(**data[i]) for i, _ in enumerate(data)]

def get_patient_data(patient_name, patients_list):
    """Retrieve specific patient data by name."""
    return next((patient for patient in patients_list if patient.patient_name == patient_name), None)

def get_value_for_case(case_name):
        """
        Reads the Excel file and returns the value for the given case_name.
        Skips the first row which contains titles.
        """
        case_name = int(case_name)
        input_file="/Users/giuliamonopoli/Desktop/PhD /shaping_mad/slice_gap.xlsx"
        try:
            df = pd.read_excel(input_file,names=["Case", "Value"])
            row = df.loc[df['Case'] == case_name]
            if not row.empty:
                value = row['Value'].values[0]  
                return int(value) if pd.notnull(value) else None
            else:
                return None  
        except Exception as e:
            print(f"Error reading the file: {e}")
            return None
# ---------------- Data Classes ----------------

from dataclasses import dataclass
from typing import List, Dict

@dataclass
class Annotation:
    mv_insert_septal: List[List[int]]
    mv_insert_lateral: List[List[int]]
    lv_base_septal: List[List[int]]
    lv_base_lateral: List[List[int]]
    leaflet_septal: List[List[int]]
    leaflet_lateral: List[List[int]]
@dataclass
class PatientData:
    patient_name: str
    key_frames: List[str]
    annotations: Dict[str, Annotation]
    bounding_box: List[int]
    flags: InitVar[List[int]] = None

def get_dicom_affine(dcm_file):
    ipp = np.array(dcm_file.ImagePositionPatient, dtype=float)
    iop = np.array(dcm_file.ImageOrientationPatient, dtype=float)
    ps = np.array(dcm_file.PixelSpacing, dtype=float)
    
    X_iop = iop[:3]
    Y_iop = iop[3:]
    
    A = np.array([
        [X_iop[0] * ps[1], Y_iop[0] * ps[0], 0, ipp[0]],
        [X_iop[1] * ps[1], Y_iop[1] * ps[0], 0, ipp[1]],
        [X_iop[2] * ps[1], Y_iop[2] * ps[0], 0, ipp[2]],
        [0, 0, 0, 1]
    ])  
    return A

def transform_nii_to_global(segmentation, dicom_affine):
    height, width = segmentation.shape
    coords = np.mgrid[0:height, 0:width].reshape(2, -1)
    ones = np.ones((1, coords.shape[1]))
    coords = np.vstack((coords, np.zeros((1, coords.shape[1])), ones))  
    global_coords = np.dot(dicom_affine, coords)
    segmentation_values = segmentation.flatten()

    return global_coords.reshape(4, height, width), segmentation_values.reshape(height, width)

def save_global_sax(patient, dcm_sax, segmentation_file):
    seg_nii = nib.load(segmentation_file)
    segmentation = seg_nii.get_fdata()

    X_seg_stack, Y_seg_stack, Z_seg_stack, seg_values_stack = [], [], [], []
    x_mask_coords, y_mask_coords, z_mask_coords = [], [], [] 
    slice_gap = get_value_for_case(patient)
        
    
    for i_sax in range(len(dcm_sax)):
        dcm_file = dcm_sax[i_sax]
        dicom_affine = get_dicom_affine(dcm_file)
        new_height, new_width = dcm_file.pixel_array.shape
     
        seg = np.transpose(segmentation[:, :, i_sax], (1, 0))
        
        if seg.shape != (new_height, new_width):
            zoom_factors = (new_height / seg.shape[0], new_width / seg.shape[1])
            reshaped_mask = zoom(seg, zoom_factors, order=0)
        else:
            reshaped_mask = seg
        
        # Transform segmentation to global coordinates
        segmentation_global, seg_values = transform_nii_to_global(reshaped_mask, dicom_affine)
        X_seg, Y_seg, Z_seg = segmentation_global[0], segmentation_global[1], segmentation_global[2]
        
        X_seg_stack.append(X_seg)
        Y_seg_stack.append(Y_seg)
        Z_seg_stack.append(Z_seg)
        seg_values_stack.append(seg_values)

        # Get indices where the mask is greater than 0
        mask_indices = reshaped_mask > 0
        
        # Save the coordinates where the mask is > 0
        x_mask_coords.append(X_seg[mask_indices])
        y_mask_coords.append(Y_seg[mask_indices])
        z_mask_coords.append(Z_seg[mask_indices])

    X_seg_stack = np.stack(X_seg_stack, axis=-1)
    Y_seg_stack = np.stack(Y_seg_stack, axis=-1)
    Z_seg_stack = np.stack(Z_seg_stack, axis=-1)
    seg_values_stack = np.stack(seg_values_stack, axis=-1)

    output_path = os.path.join(output_folder, f"{patient}_global_segmentation_3D.h5")

    resolutions = float(dcm_file.PixelSpacing[0]), float(dcm_file.PixelSpacing[1]), slice_gap
       
    try:
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('original_segmentation', data=segmentation)
            f.create_dataset('resolution', data=resolutions)
            f.create_dataset('x_mask_coords', data=np.concatenate(x_mask_coords))
            f.create_dataset('y_mask_coords', data=np.concatenate(y_mask_coords))
            f.create_dataset('z_mask_coords', data=np.concatenate(z_mask_coords))
        
        print(f"Saved all data to: {output_path}")
    except Exception as e:
        print(f"Error saving file: {e}")

# ---------------- Plotting ----------------

def plot_lax_with_sax(dcm_sax, dcm_lax, mv_insert_lateral, mv_insert_septal):
    """Plot LAX with SAX images and mitral valve insertion points."""
    fig = go.Figure()
    
    # LAX image
    dcm_file = dcm_lax[0]
    P = get_pixel_coordinates(dcm_file)
    X = P[:,:,0]
    Y = P[:,:,1]
    Z = P[:,:,2]    
    img = P[:,:,3]
    img = cv2.convertScaleAbs(img, alpha=255/img.max())

    # Define the two mitral valve insertion points
    i, j =  int(np.array(mv_insert_lateral[0][1])//2) , int(np.array(mv_insert_lateral[0][2])//2) 
    i2, j2 = int(np.array(mv_insert_septal[0][1])//2) , int(np.array(mv_insert_septal[0][2])//2)
    physical_coords = get_3d_coordinates(dcm_file, i, j)
    physical_coords2 = get_3d_coordinates(dcm_file, i2, j2)

    P1 = np.array(physical_coords)  # Mitral valve insertion point 1 (lateral)
    P2 = np.array(physical_coords2)  # Mitral valve insertion point 2 (septal)

    # Vector between P1 and P2 
    v = P2 - P1

    image_orientation = dcm_file.ImageOrientationPatient  

    R = np.array(image_orientation[:3])  
    C = np.array(image_orientation[3:])  
    normal_vector = np.cross(R, C)
    normal_vector /= np.linalg.norm(normal_vector)
    w = np.cross(normal_vector, v)
    w /= np.linalg.norm(w)  

    center = (P1 + P2) / 2
    semi_major_axis_length = np.linalg.norm(v)/2
    semi_minor_axis_length = np.linalg.norm(v)/2
    num_points = 100
    theta = np.linspace(0, 2 * np.pi, num_points)
    r = np.linspace(0, 1, num_points)
    theta, r = np.meshgrid(theta, r)
    x_ellipse = r * semi_major_axis_length * np.cos(theta)
    y_ellipse = r * semi_minor_axis_length * np.sin(theta)

    t = np.linspace(0, 2 * np.pi, 100)
    ellipse_x = center[0] + semi_major_axis_length * np.cos(t) * (v / np.linalg.norm(v))[0] + semi_minor_axis_length * np.sin(t) * normal_vector[0]
    ellipse_y = center[1] + semi_major_axis_length * np.cos(t) * (v / np.linalg.norm(v))[1] + semi_minor_axis_length * np.sin(t) * normal_vector[1]
    ellipse_z = center[2] + semi_major_axis_length * np.cos(t) * (v / np.linalg.norm(v))[2] + semi_minor_axis_length * np.sin(t) * normal_vector[2]

    # Transform points to 3D space
    ellipse_points_x = center[0] + x_ellipse * (v / np.linalg.norm(v))[0] + y_ellipse * normal_vector[0]
    ellipse_points_y = center[1] + x_ellipse * (v / np.linalg.norm(v))[1] + y_ellipse * normal_vector[1]
    ellipse_points_z = center[2] + x_ellipse * (v / np.linalg.norm(v))[2] + y_ellipse * normal_vector[2]

    # Create a surface plot for the LAX image
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, surfacecolor=img, colorscale="Greys_r", showscale=False, showlegend=True, name="LAX")])
    fig.update_traces(showscale=False)

    for i_sax in choose_sax:
        dcm_file = dcm_sax[i_sax]
        P = get_pixel_coordinates(dcm_file)
        X = P[:,:,0]
        Y = P[:,:,1]
        Z = P[:,:,2]
        img = P[:,:,3]
        img = cv2.convertScaleAbs(img, alpha=255/img.max())
        fig.add_surface(x=X, y=Y, z=Z,
                        name=str(f"SAX{i_sax}"),
                        surfacecolor=img,
                        colorscale="Greys_r",
                        showlegend=True,
                        opacity=1.0,
                        showscale=False)

    # Plot the normal vector with an arrow
    fig.add_trace(go.Cone(
        x=[center[0]], y=[center[1]], z=[center[2]],
        u=[normal_vector[0]], v=[normal_vector[1]], w=[normal_vector[2]],
        sizemode="absolute",
        sizeref=5,
        anchor="tail",
        colorscale=[[0, 'red'], [1, 'red']],
        showscale=False,
        name="Normal Vector"
    ))


    fig.add_trace(go.Scatter3d(x=[P1[0], P2[0]], y=[P1[1], P2[1]], z=[P1[2], P2[2]],
                            mode='markers', marker=dict(size=5, color='red'),
                            name="Mitral Valve Insertion Points"))

    # Plot mitral annulus approximation
    fig.add_trace(go.Scatter3d(x=ellipse_x, y=ellipse_y, z=ellipse_z,
                            mode='lines', line=dict(color='blue', width=4),
                            name="Mitral Annulus Ellipse"))

    # Plot the points inside the disk
    fig.add_trace(go.Scatter3d(x=ellipse_points_x.flatten(), y=ellipse_points_y.flatten(), z=ellipse_points_z.flatten(),
                            mode='markers', marker=dict(size=2, color='blue', opacity=0.5),
                            name="Points Inside Ellipse"))

    fig.update_layout(title="LAX Plane with Normal Vector, Mitral Valve Insertion Points, and Mitral Annulus",
                    scene=dict(aspectmode='data'), width=800, height=600)
    slider_sax = {
        
        'active': 100,
        'currentvalue': {'prefix': 'Opacity: '},
        'steps': [{
            'value': step/100,
            'label': f'{step}%',
            'visible': True,
            'execute': True,
            'method': 'restyle',
            'args': [{'opacity': step/100}, [i for i in range(1, n_max_sax+1)]]     # apply to sax only
        } for step in range(101)]
    }

    fig.layout.scene.camera.projection.type = "orthographic"

    fig.update_layout(sliders=[slider_sax])

    fig.show()

# ---------------- Main Execution ----------------


subject_id = 27
target_frame_num = 13 # ES frame 
dcm_folder_sax = Path(f"/Users/giuliamonopoli/Desktop/PhD /Data/ES_files/{subject_id}/DICOM_files")
dcm_folder_lax = Path(f"/Users/giuliamonopoli/Desktop/PhD /Data/MAD_OUS_sorted/{subject_id}/cine/4ch/")
annotation_path = "/Users/giuliamonopoli/Desktop/PhD /deepvalve/data/new_annotations"
segmentation_file = f"/Users/giuliamonopoli/Desktop/PhD /Data/ES_files/{subject_id}/NIfTI_files/{subject_id}_myo.nii"
output_folder = f"/Users/giuliamonopoli/Desktop/PhD /Data/ES_files/{subject_id}/Global_Segmentations"

# Load SAX and LAX DICOM files
sax_files = sorted([f for f in os.listdir(dcm_folder_sax) if f.endswith(".dcm")], key=lambda x: int(x.split('sliceloc_')[1].split('.')[0]))
lax_files = load_lax(dcm_folder_lax, target_frame_num)
dcm_sax = [dcmread(dcm_folder_sax / f) for f in sax_files]
dcm_lax = [dcmread(dcm_folder_lax / f) for f in lax_files]

# Load annotations
patient_data = load_annotations(annotation_path)
specific_patient_data = get_patient_data(f"MAD_{subject_id}_0", patient_data)
sd = specific_patient_data.annotations

# Extract Mitral Valve insertion points
mv_insert_septal = np.array(sd[str(target_frame_num)]["mv_insert_septal"])
mv_insert_lateral = np.array(sd[str(target_frame_num)]["mv_insert_lateral"])

n_max_sax = len(dcm_sax)
choose_sax = np.arange(0, n_max_sax)

os.makedirs(output_folder, exist_ok=True)
save_global_sax(subject_id, dcm_sax, segmentation_file)
plot_lax_with_sax(dcm_sax, dcm_lax, mv_insert_lateral, mv_insert_septal)
