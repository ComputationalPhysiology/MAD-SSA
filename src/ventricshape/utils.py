import os
import json
import numpy as np
import pandas as pd
import ventric_mesh.mesh_utils as mu
import plotly.graph_objects as go
from ventricshape.get_com_rv import get_rv_point
from structlog import get_logger
logger = get_logger()

###########################################################
# Function to create patient settings JSON file
def create_patient_settings(patient_name, settings_folder=None, lax_smooth_level_endo=None, lax_smooth_level_epi=None):
    default_settings = {
        "mesh": {
            "fine": {
                "mask_is_inverted": True,
                "seed_num_base_epi": 40,
                "seed_num_base_endo": 40,
                "num_z_sections_epi": 20,
                "num_z_sections_endo": 20,
                "smooth_level_epi": 0.1,
                "smooth_level_endo": 0.1,
                "num_lax_points": 32,
                "lax_smooth_level_epi": 80,
                "lax_smooth_level_endo": 50,
                "lax_spline_order_epi": 3,
                "lax_spline_order_endo": 2,
                "z_sections_flag_epi": 0,
                "z_sections_flag_endo": 0
            }
        }
    }

    if lax_smooth_level_endo is not None:
        default_settings["mesh"]["fine"]["lax_smooth_level_endo"] = lax_smooth_level_endo
    if lax_smooth_level_epi is not None:
        default_settings["mesh"]["fine"]["lax_smooth_level_epi"] = lax_smooth_level_epi
    if settings_folder is None:
        settings_folder = "/home/shared/controls/settings"
    
    os.makedirs(settings_folder, exist_ok=True)

    patient_file_path = os.path.join(settings_folder, f"{patient_name}.json")
    with open(patient_file_path, "w") as json_file:
        json.dump(default_settings, json_file, indent=4)

    print(f"Settings file created: {patient_file_path}")


def process_files_from_directory(directory_path = "/home/shared/controls/ES_files_controls", settings_folder=None, lax_smooth_level_endo=None, lax_smooth_level_epi=None):
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        return
    
    for filename in os.listdir(directory_path):
        if filename[:3].isdigit():  
            patient_name = filename[:3]
            create_patient_settings(patient_name, settings_folder)
            
########################################



# Function to compute the orthogonal projection of RV onto the z-axis
def find_rv_ortho(P_rv, P_origin, z_axis):
    if P_rv.shape[0] != 1 or P_origin.shape[0] != 1 or z_axis.shape[0] != 1:
        raise ValueError("Input points must be row vectors.")
    if P_rv.shape[1] != 3 or P_origin.shape[1] != 3 or z_axis.shape[1] != 3:
        raise ValueError("Input points must have 3 columns.")
    rv_to_origin = P_rv - P_origin
    
    num = -np.dot(rv_to_origin, z_axis.T)
    
    den = np.dot(z_axis, z_axis.T)
    if den == 0:
        raise ValueError("z_axis direction vector is zero; cannot compute orthogonal projection.")
    s = num / den
    
    return P_rv + s * z_axis

# Function to align points by rotating the y-axis towards RV
def align_y_axis_to_rv_new(P_origin, P_rv_new, points):
    v = P_rv_new - P_origin
    if v.shape[0] != 1:
        raise ValueError("Input vector must be a row vector.")
    v_xy = v[0, :2]
    v_xy_norm = np.linalg.norm(v_xy)
    if v_xy_norm == 0:
        raise ValueError("Vector is zero in the x-y plane; cannot determine angle.")
    v_xy /= v_xy_norm

    y_axis = np.array([0, -1])
    cos_theta = np.dot(v_xy, y_axis)
    sin_theta = v_xy[0] * y_axis[1] - v_xy[1] * y_axis[0]
   
    theta = np.arctan2(sin_theta, cos_theta)

    R_z = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])
    
    return points @ R_z.T, P_rv_new @ R_z.T


def process_subject(subject_id, input_dir, output_dir):
    print(subject_id)

    points_epi_file= os.path.join(input_dir,f"{subject_id}/points_cloud_epi.csv")
    points_endo_file = os.path.join(input_dir,f"{subject_id}/points_cloud_endo.csv")
  
  
    points_cloud_epi = np.array(pd.read_csv(points_epi_file,header=None))
    points_cloud_endo = np.array(pd.read_csv(points_endo_file,header=None))
    if points_cloud_epi.shape[0] == 0 or points_cloud_endo.shape[0] == 0:
        print(f"Problem subject {subject_id}: Empty point clouds.")
    elif points_cloud_epi.shape[0] != 801 and  points_cloud_endo.shape[0]!=801:
        print(f"Problem subject {subject_id}: Different number of points in epi and endo.")
   
    com = np.mean(points_cloud_epi, axis=0)
    P_origin = np.array([com])
    P_rv = np.array([get_rv_point(input_dir,subject_id)])

    
    z_axis = np.array([[0, 0, 1]])

    P_rv_new = find_rv_ortho(P_rv, P_origin, z_axis)
    
    centered_points_epi = points_cloud_epi - P_origin
    centered_points_endo = points_cloud_endo - P_origin
    centered_P_rv_new = P_rv_new - P_origin

    aligned_points_epi, rv_new_rotated = align_y_axis_to_rv_new(np.zeros_like(P_origin), centered_P_rv_new, centered_points_epi)
    aligned_points_endo, _ = align_y_axis_to_rv_new(np.zeros_like(P_origin), centered_P_rv_new, centered_points_endo)
    merged_point_cloud = np.concatenate(( aligned_points_epi,aligned_points_endo), axis=0)
    points_list = convert_pc_to_stack(merged_point_cloud, num_z_sections=20)
    output_file = os.path.join(output_dir, f"{subject_id}_merged_points.txt")
    apex = points_list.pop(-1)
    points_list = remove_duplicates(points_list)
    tck_endo = mu.get_shax_from_coords(points_list, 0.0)
    ordered_points = mu.get_sample_points_from_shax(tck_endo, 40)
    # plt.scatter(points_list[0][:, 0], points_list[0][:, 1], color="y")
    # plt.scatter(points_list[0][:5, 0], points_list[0][:5, 1], color="r")

    # # print(f"Number of points in ordered_points: {ordered_points.shape}")

    # plt.scatter(ordered_points[0][:5, 0], ordered_points[0][:5, 1], color="b")
    # plt.savefig("test.png")
    # exit()
    ordered_points.append(apex)
    ordered_points = np.vstack(ordered_points)
    np.savetxt(output_file, merged_point_cloud, fmt='%.6f', delimiter=',')

   
    # np.savetxt(outname.as_posix() + ".txt", ordered_points, delimiter=',',  fmt='%.8f')
    # np.savetxt(output_file_epi, aligned_points_epi, fmt='%.6f', delimiter=',')
    # np.savetxt(output_file_endo, aligned_points_endo, fmt='%.6f', delimiter=',')
    # print(f"Aligned points saved to {output_file_epi}")
    # print(f"Aligned points saved to {output_file_endo}")
    print(f"Aligned points saved to {output_file}")

    return aligned_points_epi,aligned_points_endo, rv_new_rotated,centered_points_epi,centered_points_endo,centered_P_rv_new


def visualize(aligned_points,aligned_points_endo, rv_new_rotated, P_origin,output_dir,centered_points_epi,centered_points_endo,centered_P_rv_new):
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=aligned_points[:, 0], y=aligned_points[:, 1], z=aligned_points[:, 2],
        mode='markers', marker=dict(color='blue', size=5), name='Aligned epi Points'
    ))
    fig.add_trace(go.Scatter3d(
        x=aligned_points_endo[:, 0], y=aligned_points_endo[:, 1], z=aligned_points_endo[:, 2],
        mode='markers', marker=dict(color='blue', size=3), name='Aligned endo Points'
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[rv_new_rotated[0, 0]], y=[rv_new_rotated[0, 1]], z=[rv_new_rotated[0, 2]],
        mode='markers', marker=dict(color='red', size=8), name='RV Aligned'
    ))
    fig.add_trace(go.Scatter3d(
        x=[P_origin[0, 0]], y=[P_origin[0, 1]], z=[P_origin[0, 2]],
        mode='markers', marker=dict(color='green', size=8), name='Origin'
    ))
    fig.add_trace(go.Scatter3d(
        x=[P_origin[0, 0], rv_new_rotated[0, 0]],
        y=[P_origin[0, 1], rv_new_rotated[0, 1]],
        z=[P_origin[0, 2], rv_new_rotated[0, 2]],
        mode='lines', line=dict(color='black', dash='dash'), name='Origin to RV'
    ))
    fig.add_trace(go.Scatter3d(
        x=centered_points_epi[:, 0], y=centered_points_epi[:, 1], z=centered_points_epi[:, 2],
        mode='markers', marker=dict(color='red', size=5), name=' EPi Points'
    ))
    fig.add_trace(go.Scatter3d(x=centered_points_endo[:, 0], y=centered_points_endo[:, 1], z=centered_points_endo[:, 2],mode='markers', marker=dict(color='red', size=3), name='Endo Points'))
    fig.add_trace(go.Scatter3d(
        x=[centered_P_rv_new[0, 0]], y=[centered_P_rv_new[0, 1]], z=[centered_P_rv_new[0, 2]],
        mode='markers', marker=dict(color='red', size=8), name='RV old'
    ))
    fig.add_trace(go.Scatter3d(
        x=[P_origin[0, 0], centered_P_rv_new[0, 0]],
        y=[P_origin[0, 1], centered_P_rv_new[0, 1]],
        z=[P_origin[0, 2], centered_P_rv_new[0, 2]],
        mode='lines', line=dict(color='green'), name='Origin to RV'
    ))
    fig.write_html(os.path.join(output_dir,"aligned_points_COMP.html"))


def convert_pc_to_stack(pc, num_z_sections=20):
    num_points_per_z = int((pc.shape[0]) / num_z_sections)
    pc_list = []
    # Create the list of z sections
    for i in range(num_z_sections):
        pc_list.append(pc[i * num_points_per_z : (i + 1) * num_points_per_z])

    # Add the last element with the remaining points
    pc_list.append(pc[-1])

    return pc_list

def remove_duplicates(endo_points_list):
    endo_points_list_no_duplicates = []
    for points in endo_points_list:
        unique_points = points[np.diff(points, axis=0, prepend=np.nan).any(axis=1)]
        endo_points_list_no_duplicates.append(unique_points)
        if unique_points.shape[0]<points.shape[0]:
            logger.warning("Several unique points are removed...")
    return endo_points_list_no_duplicates


def process_all_subjects(subject_ids, input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for subject_id in subject_ids:
        print(f"Processing subject {subject_id}...")
        out_plot = os.path.join(output_dir,"plots",subject_id)
        os.makedirs(out_plot, exist_ok=True)
        aligned_points_epi, aligned_points_endo, rv_new_rotated,centered_points_epi,centered_points_endo,centered_P_rv_new = process_subject(subject_id, input_dir, output_dir)
        # visualize(aligned_points_epi, aligned_points_endo,rv_new_rotated, np.zeros((1, 3)),out_plot,centered_points_epi,centered_points_endo,centered_P_rv_new)



