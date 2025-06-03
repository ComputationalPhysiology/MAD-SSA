from pathlib import Path
import numpy as np
import h5py
import os
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np

#%% Filling the gaps using dilation/erosion


def read_data_h5(file_dir):
    with h5py.File(file_dir, "r") as h5_file:
        LVmask = h5_file["LVmask"][:]
        slice_thickness = h5_file["slice_thickness"][0]
        resolution = h5_file["resolution"][0]
    return LVmask, slice_thickness, resolution

def close_apex(LVmask):
    K, I, J = LVmask.shape
    mask_closed_apex = np.zeros((K+1,I,J))
    mask_closed_apex[:-1,:,:] = LVmask
    kernel = np.ones((3, 3), np.uint8)
    mask_last_slice = np.uint8(LVmask[-1,:,:] * 255)
    mask_last_slice_closed = cv.dilate(mask_last_slice, kernel, iterations=6)
    mask_last_slice_closed = cv.erode(mask_last_slice_closed, kernel, iterations=8)
    mask_closed_apex[-1,:,:] = mask_last_slice_closed
    return mask_closed_apex





# %%
def plot_voxels(voxel_array, resolution, slice_thickness, alpha = 1):
    fig = plt.figure(figsize=plt.figaspect(1)*2)
    # fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111, projection='3d')

    voxel_array = np.transpose(voxel_array, (1, 2, 0))

    nx, ny, nz = voxel_array.shape

    x = np.arange(0, nx * resolution, resolution)
    y = np.arange(0, ny * resolution, resolution)
    # z = np.arange(0, nz * slice_thickness, slice_thickness)
    z = np.linspace(0, nz * slice_thickness, nz, endpoint=False)
    # expanded_voxel_array = np.repeat(voxel_array, slice_thickness, axis=2)
    # Set the Z-axis tick positions and labels
    z_ticks = np.arange(0, nz, 1)  # Original tick positions
    z_tick_labels = [f"{val * slice_thickness:.1f}" for val in z_ticks]  # Scale and format labels
    ax.set_zticks(z_ticks)
    ax.set_zticklabels(z_tick_labels)
    color = [1, 1 , 1, alpha]
    ax.voxels(voxel_array, facecolors=color, edgecolor='k')

    ax.set_xlabel('X Axis (mm)')
    ax.set_ylabel('Y Axis (mm)')
    ax.set_zlabel('Z Axis')
    return ax


LVmask_raw, slice_thickness, resolution = read_data_h5('data.h5')
#LVmask = close_apex(LVmask_raw)
LVmask = LVmask_raw
results_folder = Path('results')     

array_3d = LVmask[:,:,:]
ax=plot_voxels(array_3d, resolution, slice_thickness)
ax.view_init(elev=-150, azim=-45)
ax.set_box_aspect(aspect=(1, 1, 1))
ax.set_xlim([100, 160])
ax.set_ylim([100, 160])
ax.set_axis_off()
#plt.show()
fname = results_folder /'3D_view.png'
plt.savefig(fname.as_posix())


ax=plot_voxels(array_3d, resolution, slice_thickness)
ax.view_init(elev=-180, azim=-0, )
ax.set_xlim([100, 160])
ax.set_ylim([100, 160])
ax.set_box_aspect(aspect=(1, 1, 1))
ax.set_axis_off()
# plt.show()
fname = results_folder / 'Lateral_view.png'
plt.savefig(fname.as_posix())

ax=plot_voxels(array_3d, resolution, slice_thickness)
ax.view_init(elev=-90, azim=0)
ax.set_xlim([80, 150])
ax.set_ylim([100, 160])
ax.set_box_aspect(aspect=(1, 1, 1))
ax.set_axis_off()
# plt.show()
fname = results_folder / 'Top_view.png'
plt.savefig(fname.as_posix())


# %%
# import plotly.graph_objects as go
# import numpy as np

# def plot_voxels(voxel_array, resolution, slice_thickness):
#     # Transpose the array if necessary to align the axes correctly
#     voxel_array = np.transpose(voxel_array, (1, 2, 0))

#     # Get the dimensions of the array
#     nx, ny, nz = voxel_array.shape

#     # Create a list to hold the plotly cubes
#     cubes = []

#     # Iterate through the voxel array and create a cube for each True value
#     for x in range(nx):
#         for y in range(ny):
#             for z in range(nz):
#                 if voxel_array[x, y, z]:
#                     # Define the coordinates of the cube corners
#                     x0, x1 = x * resolution, (x + 1) * resolution
#                     y0, y1 = y * resolution, (y + 1) * resolution
#                     z0, z1 = z * slice_thickness, (z + 1) * slice_thickness
#                     cube = go.Mesh3d(
#                         # Vertices of the cube
#                         x=[x0, x1, x1, x0, x0, x1, x1, x0],
#                         y=[y0, y0, y1, y1, y0, y0, y1, y1],
#                         z=[z0, z0, z0, z0, z1, z1, z1, z1],
#                         # i, j and k define the vertices that compose each face of the cube
#                         i=[0, 0, 0, 4, 4, 7],
#                         j=[1, 2, 3, 5, 6, 3],
#                         k=[2, 3, 7, 6, 7, 6],
#                         opacity=0.5,
#                         color='black'
#                     )
#                     cubes.append(cube)

#     # Create the 3D plot
#     fig = go.Figure(data=cubes)
#     fig.update_layout(
#         scene=dict(
#             xaxis_title='X Axis (mm)',
#             yaxis_title='Y Axis (mm)',
#             zaxis_title='Z Axis (mm)'
#         ),
#         title='3D Boolean Array Visualization with Scaled Voxels'
#     )
#     return fig

# # Example usage
# t = 0
# array_3d = mask[:, :, :, t]  #
# fig=plot_voxels(array_3d, resolution, slice_thickness)
# fig.show()
# fig.write_html(data_folder+'/Raw Image.html')
