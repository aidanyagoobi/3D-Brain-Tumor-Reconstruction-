import sys
import numpy as np
import scipy
import os
import glob
import matplotlib.pyplot as plt
from skimage import measure
from scipy.ndimage import gaussian_filter
import pyvista as pv
import re

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class TumorSlice:
    def __init__(self, tumor_border, tumor_mask, patient_id, tumor_mask_processed=None, signed_distance=None):
        #self.tumor_border = tumor_border
        self.tumor_mask = tumor_mask #Tumor mask is a 512x512 txt file entry consisting of 0s and 1s where 1s represent tumours tissue
        self.tumor_mask_processed = tumor_mask_processed
        self.signed_distance = signed_distance
        self.patient_id = patient_id

    """This method takes in the Tumor Border txt file and returns
    it to a 2D-np array representation of 0s and 1s. Where 1s indicate 
    tumours tissue and 0s indicate non-tumours tissue."""
    def process_tumor_mask(self):
        processed_data = []
        file_name = self.tumor_mask
        with open(file_name, 'r') as file:
            for line in file:
                words = line.strip().split()
                processed_line = [1 if word.lower() == 'true' else 0 for word in words]
                processed_data.append(processed_line)

        data_array = np.array(processed_data)
        self.tumor_mask_processed = data_array

    """This method takes the processed tumor mask numpy array from the previous method 
    and generates, for each pixel, it's signed distance. Signed distance is represented as the 
    Euclidian distance to the nearest border value or 0 in this case"""
    def compute_signed_distance(self):
        dist_out = scipy.ndimage.distance_transform_edt(self.tumor_mask_processed == 0)
        dist_in = scipy.ndimage.distance_transform_edt(self.tumor_mask_processed == 1)
        signed_distance = dist_out.copy()
        signed_distance[self.tumor_mask_processed == 1] = -dist_in[self.tumor_mask_processed == 1]
        self.signed_distance = signed_distance

    """This method takes in an array of four slices F_i-1, F_i, F_i+1, and F_i+2. and generates a newly interpolated 
    slice between F_i and F_i+1. It does so by following the Piecewise Weighted Linear Interpolation Algorithm
    it provides a heavier weighting to the slices closer to f_i and a lesser weight to those farther"""
    @staticmethod
    def create_subdivision(slices):
        #Takes in an array of TumorSlice objects
        assert len(slices) == 4, "Cannot create subdivision with less than 4 slices"
        #slices in this method holds an array of signed_distances for each slice
        w = 1/16 #This is the subdivision tension, 1/16 is what was proposed in the paper as a good standard
        e_i = (2*w) * slices[0].signed_distance + (1 + (2*w)) * slices[1].signed_distance
        h_i = (2*w) * slices[3].signed_distance + (1 + (2*w)) * slices[2].signed_distance
        new_signed_distance = 0.5 * (e_i + h_i)
        #returns a new nmpy array 512x512
        # Smooth the SD field in‐plane before you ever make it binary:
        sigma = 2.0  # tune this to your image resolution
        new_signed_distance = gaussian_filter(new_signed_distance, sigma=sigma)
        return new_signed_distance

    """This method takes all of the slice objects, places the known slices in even positions, so 
    if the input if F_0, F_1, F_2, F_3: it instantiates F_new, F_0, F_new, F_1, F_new, F_2, F_new, F_3
    all the f_new's will be filled with newly interpolated slices following the create_subdivision method"""
    @staticmethod
    def run_interpolation(slices):
        #This takes in an array of TumorSlices
        assert len(slices) >= 4, "Cannot interpolate with less than 4 slices"
        num_slices = len(slices)
        interpolated_slices = [None] * ((num_slices * 2) - 1)
        index_to_add = 0
        #Here we are creating the even and odd indexing where the odd indexes hold the F_news to be created and the even indexes hold the original slices
        for i in range(len(interpolated_slices)):
            if i % 2 == 0:
                interpolated_slices[i] = slices[index_to_add]
                index_to_add += 1
            else:
                continue
        #print(interpolated_slices[0].signed_distance)
        #Now that we have populated the interpolated_slices array, it's time to fill it in
        len_interpolated_slices = len(interpolated_slices)
        for i in range(len_interpolated_slices):
            if i % 2 == 0:
                continue
            elif i == 1:
                # This is a special case where we are missing information at the very beginning, missing F_{-1}, it's mentioned in the paper as well
                new_signed_distance = TumorSlice.create_subdivision([slices[0], slices[0], slices[1], slices[2]])
                new_tumor = TumorSlice(None, None, None,  None, new_signed_distance)
                #new_tumor.compute_signed_distance()
                interpolated_slices[i] = new_tumor
            elif i == len_interpolated_slices - 2:
                # This is another special case where we are missing information at the very end, missing F_{n}, extrapolate at end
                idx = (i - 1) // 2
                new_signed_distance = TumorSlice.create_subdivision([slices[idx - 1], slices[idx], slices[idx + 1], slices[idx + 1]])
                new_tumor = TumorSlice(None, None, None,  None, new_signed_distance)
                #new_tumor.compute_signed_distance()
                interpolated_slices[i] = new_tumor
            else:
                # Normal case
                idx = (i - 1) // 2
                new_signed_distance = TumorSlice.create_subdivision([slices[idx - 1], slices[idx], slices[idx + 1], slices[idx + 2]])
                new_tumor = TumorSlice(None, None, None,  None, new_signed_distance)
                #new_tumor.compute_signed_distance()
                interpolated_slices[i] = new_tumor
        #print(interpolated_slices[1].tumor_mask_processed)
        return interpolated_slices



    """This method opens all the txt files and initializes constructors for them. It then returns all the objects
    as a list of tumor_slices"""
    @staticmethod
    def generate_slices_from_txt_files(folder_path):
        txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
        txt_files = sorted(
            txt_files,
            key=lambda fn: int(re.search(r'\d+', os.path.basename(fn)).group())
        )
        tumor_slices = []
        for txt_file in txt_files:
            print(f"Processing {txt_file}")
            tumor = TumorSlice(None, txt_file, None, None, None)
            tumor_slices.append(tumor)
            #Initialize the tumor with a processed np array
            tumor.process_tumor_mask()
            #Initialize the tumor with a signed distance
            tumor.compute_signed_distance()
            tumor_slices.append(tumor)
        return tumor_slices


    """This method adds a constant z axis to the numpy array tumor_mask_processed for visualization purposes, 
    it takes in a specified z height"""
    def add_z(self, z_height):
        if self.tumor_mask_processed.ndim == 2:
            h, w = self.tumor_mask_processed.shape
            z_layer = np.full((h, w), z_height)
            array_3d = np.stack((self.tumor_mask_processed, z_layer), axis=-1)
            self.tumor_mask_processed = array_3d
        elif self.tumor_mask_processed.ndim == 3:
            # Already has a z-axis, just update the z values
            self.tumor_mask_processed[..., 1] = z_height
        else:
            raise ValueError(f"Unexpected tumor_mask_processed shape: {self.tumor_mask_processed.shape}")

def stack_tumor_slices(slices, total_depth_mm):
    """
    Build a (Z, Y, X) binary volume _and_ compute the uniform Z-spacing so
    that the last slice lands exactly at total_depth_mm.
    """
    masks = [
        (tumor.signed_distance >= 0).astype(np.uint8)
        for tumor in slices
    ]
    volume = np.stack(masks, axis=0)           # shape == (N_slices, H, W)
    # N_slices − 1 gaps must span total_depth_mm
    dz = total_depth_mm / (volume.shape[0] - 1)
    return volume, dz


def visualize_slices(volume, num_slices_to_show=8):
    # volume shape: (num_slices, height, width)
    total_slices = volume.shape[0]
    step = total_slices // num_slices_to_show
    fig, axes = plt.subplots(1, num_slices_to_show, figsize=(20, 5))

    for i, ax in enumerate(axes):
        idx = i * step
        ax.imshow(volume[idx], cmap='gray')
        ax.axis('off')
        ax.set_title(f"Slice {idx}")
    plt.show()


def visualize_and_save_volume_pyvista(volume,
                                      z_spacing,
                                      threshold=0.5,
                                      stl_filename="tumor_model.stl"):
    """
    - Extracts the 0.5 isosurface from `volume` with correct Z spacing.
    - Writes it out as a binary STL.
    - Opens an interactive PyVista window to view it.
    """
    # 1) marching cubes with (dz, dy, dx)
    verts, faces, normals, _ = measure.marching_cubes(
        volume,
        level=threshold,
        spacing=(z_spacing, 1.0, 1.0)
    )
    # 2) PyVista expects faces as [3, i0, i1, i2, 3, …]
    faces_flat = np.hstack([
        np.full((faces.shape[0], 1), 3, dtype=np.int32),
        faces.astype(np.int32)
    ]).ravel()

    # 3) Build and save the mesh
    mesh = pv.PolyData(verts, faces_flat)
    mesh.save(stl_filename)
    print(f"▶ Saved binary STL to {stl_filename}")

    # 4) Visualize
    p = pv.Plotter()
    p.add_mesh(mesh, color="tomato", opacity=0.6, show_edges=True)
    p.add_axes()      # Draw the coordinate axes
    p.show_grid()     # Draw a grid on the floor
    p.show(title="3D Tumor Volume (PyVista)")


def fuse_sdfs(sdfs):
    return np.mean(np.stack(sdfs, axis=0), axis=0)

def marching_cubes_from_sdf(sdf, spacing, filename, hole_size=1000.0):
    # sdf : 3D float array
    # spacing : (dz, dy, dx)
    verts, faces, normals, values = measure.marching_cubes(
        sdf,
        level=0.0,      # <-- zero‐level of your signed distance
        spacing=spacing
    )

    # PyVista wants faces flattened as [3, i0, i1, i2, 3, …]
    faces_flat = np.hstack([
        np.full((faces.shape[0],1), 3, dtype=np.int32),
        faces.astype(np.int32)
    ]).ravel()

    mesh = pv.PolyData(verts, faces_flat)
    mesh = mesh.fill_holes(hole_size=hole_size)   # plug any tiny gaps
    mesh.save(filename)
    print(f"▶ wrote {filename}")

def load_and_interp(folder, n_levels):
    # 1) load + sort
    fnames = sorted(glob.glob(os.path.join(folder,'*.txt')),
                    key=lambda f:int(re.search(r'\d+',os.path.basename(f)).group()))
    slices = []
    for f in fnames:
        T = TumorSlice(None, f, None, None, None)
        T.process_tumor_mask()
        T.compute_signed_distance()
        slices.append(T)
    # 2) subdivide
    for _ in range(n_levels):
        slices = TumorSlice.run_interpolation(slices)
    return slices

def build_sdf_volume(slices, total_len_mm, axis):
    """
    slices: list of TumorSlice after interpolation
    total_len_mm: full extent along that axis
    axis: 0 for Z-stack, 1 for Y-stack, 2 for X-stack
    """
    # each TumorSlice.signed_distance is a 2D array (H,W)
    sd_layers = [t.signed_distance for t in slices]
    vol = np.stack(sd_layers, axis=0)  # shape (N,H,W)
    # compute spacing along that stack:
    dz = total_len_mm / (vol.shape[0]-1)
    # Now reorient so that all volumes end up in (Z,Y,X) coordinates:
    if axis == 0:
        # already (Z,Y,X)
        return vol, (dz,1.0,1.0)
    elif axis == 1:
        # coronal: (Y,Z,X) → want (Z,Y,X)
        vol = vol.transpose(1,0,2)
        return vol, (1.0,dz,1.0)
    else:
        # sagittal: (X,Y,Z) → want (Z,Y,X)
        vol = vol.transpose(2,1,0)
        return vol, (1.0,1.0,dz)





def export_filled_tumor_stl(volume, z_spacing, stl_filename="tumor_filled.stl", hole_size=1e3):
    """
    volume      : (N, H, W) uint8 array where 1 = tumor interior, 0 = background
    z_spacing   : float spacing between slices in the Z direction
    hole_size   : maximum size of holes to fill (in mesh units; tune as needed)
    """
    # 1) Run marching cubes on the *interior* mask
    verts, faces, normals, _ = measure.marching_cubes(
        volume,
        level=0.5,
        spacing=(z_spacing, 1.0, 1.0)
    )
    # 2) Build PyVista mesh (faces need the leading “3” per triangle)
    faces_flat = np.hstack([
        np.full((faces.shape[0], 1), 3, dtype=np.int32),
        faces.astype(np.int32)
    ]).ravel()
    mesh = pv.PolyData(verts, faces_flat)

    # 3) Fill any holes up to `hole_size`
    mesh = mesh.fill_holes(hole_size=hole_size)

    # 4) Save out a watertight STL
    mesh.save(stl_filename)
    print(f"▶ Saved watertight STL to {stl_filename}")


def main():
    SUBDIV = 6
    N_z = 7
    N_y = 7
    N_x = 7

    depth_z = 6 * (N_z - 1)
    depth_y = 6 * (N_y - 1)
    depth_x = 6 * (N_x - 1)

    # 1) build three SDF volumes
    sx = load_and_interp("tumor_mask_data/tumor_1_x", SUBDIV)
    sy = load_and_interp("tumor_mask_data/tumor_1_y", SUBDIV)
    sz = load_and_interp("tumor_mask_data/tumor_1_z", SUBDIV)

    vol_z, sp_z = build_sdf_volume(sz, depth_z, axis=0)  # (Z,Y,X)
    vol_y, sp_y = build_sdf_volume(sy, depth_y, axis=1)  # (Z,Y,X)
    vol_x, sp_x = build_sdf_volume(sx, depth_x, axis=2)  # (Z,Y,X)

    # 2) extract each orientation’s zero‐level surface directly from the SDF
    marching_cubes_from_sdf(vol_z, sp_z, "tumor_z_orig.stl")
    marching_cubes_from_sdf(vol_y, sp_y, "tumor_y_orig.stl")
    marching_cubes_from_sdf(vol_x, sp_x, "tumor_x_orig.stl")

    # 3) Resample vol_y & vol_x to match vol_z.shape
    target_shape = vol_z.shape  # e.g. (7, 512, 512)
    def resample_to_target(vol):
        factors = [t / v for t, v in zip(target_shape, vol.shape)]
        return scipy.ndimage.zoom(vol, factors, order=1)

    vol_y_rs = resample_to_target(vol_y)
    vol_x_rs = resample_to_target(vol_x)

    # 4) fuse the three SDFs via pointwise max
    sdf_fused = fuse_sdfs([vol_z, vol_y_rs, vol_x_rs])

    # 5) extract the fused zero‐level surface
    marching_cubes_from_sdf(sdf_fused, sp_z, "tumor_fused.stl")

if __name__ == "__main__":
    main()
