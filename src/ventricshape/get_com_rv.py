import h5py,os
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_dilation
import ventric_mesh.utils as utils
import matplotlib.pyplot as plt
import ventric_mesh.mesh_utils as mu
def is_connected(matrix):
    flag = False
    visited = set()
    visited_reversed = set()
    rows, cols = len(matrix), len(matrix[0])

    def dfs(r, c, flg):
        if flg:
            if (
                0 <= r < rows
                and 0 <= c < cols
                and matrix[r][c]
                and (r, c) not in visited
            ):
                visited.add((r, c))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    dfs(r + dr, c + dc, 1)
        else:
            if (
                0 <= r < rows
                and 0 <= c < cols
                and matrix[r][c]
                and (r, c) not in visited_reversed
            ):
                visited_reversed.add((r, c))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    dfs(r + dr, c + dc, 0)

    # Find first True value and start DFS from there
    for r in range(rows):
        if flag:
            break
        for c in range(cols):
            if matrix[r][c]:
                dfs(r, c, 1)
                flag = True
                break

    # Check for any unvisited True values
    indices = np.argwhere(matrix)
    index_set = set(map(tuple, indices))
    if visited == index_set:
        return True, visited, visited_reversed
    else:
        flag = False
        for c in range(cols - 1, -1, -1):
            if flag:
                break
            for r in range(rows):
                if matrix[r][c] and (r, c) not in visited:
                    dfs(r, c, 0)
                    flag = True
                    break

        return False, visited, visited_reversed

def get_endo_epi(mask):
    
    K, I, J = mask.shape
    kernel = np.ones((3, 3), np.uint8)
    mask_epi = np.zeros((K, I, J))
    
    for k in range(K):
        mask_k = mask[k, :, :]
        img = np.uint8(mask_k * 255)
        img_dilated = binary_dilation(img, structure=kernel).astype(img.dtype)
        img_edges = img_dilated - img
        img_edges[img_edges == 2] = 0
        flag, visited, visited_reversed = is_connected(img_edges)
        if flag:
            img_epi = img_edges
        else:
            img_epi = np.zeros((I, J), dtype=np.uint8)
            if len(visited) > len(visited_reversed):
                for x, y in visited:
                    img_epi[x, y] = 1
                
            else:
                for x, y in visited_reversed:
                    img_epi[x, y] = 1
               
        mask_epi[k, :, :] = img_epi
       
    return mask_epi
def image_overlay(img, epi):
    new_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] != 0:
                new_image[i, j] = [255, 255, 255]
            
            elif epi[i, j] != 0:
                new_image[i, j] = [255, 0, 0]
    return new_image
def read_rvdata_h5(file_dir):
    with h5py.File(file_dir, "r") as h5_file:
        RVmask = h5_file["RV_mask"][:]
        resolution_data = h5_file["resolution"][:]
    return RVmask, resolution_data

def get_rv_point(seg_dir, case,result_dir=os.getcwd()+'/results/'):
    rvmask,resolution_data = read_rvdata_h5(os.path.join(seg_dir,case,f"{case}_original_segmentation.h5"))

    rv = get_endo_epi(rvmask)
    K = len(rv)
    resolution = resolution_data[0] * 1.01
    slice_thickness = resolution_data[2]      
    coords_rv = mu.get_coords_from_mask(rv, resolution, slice_thickness)      

    outdir = Path(result_dir,case) / "RV_points"
    os.makedirs(outdir, exist_ok=True)
    for k in range(K):
        mask_epi_k = rv[k]
        LVmask_k = rvmask[k]
        new_image =image_overlay(LVmask_k, mask_epi_k)
        fnmae =  os.path.join(outdir, f" str{k}.png")
        plt.imshow(new_image)
        dpi = np.round((300/resolution)/100)*100
        plt.savefig(fnmae, dpi=dpi)
        plt.close()
        
    base = coords_rv[0]
    mean = np.mean(base, axis=0)
    coords_rv = np.concatenate(coords_rv)
    
    com = np.mean(coords_rv, axis=0)
    return com

