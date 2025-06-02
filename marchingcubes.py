import numpy as np
import imageio.v3 as imageio
import os
from scipy.ndimage import zoom
from skimage import measure
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# === CONFIG ===
folder_path = "/Users/ayagoobi/Desktop/brains3"  # <- update if needed
output_path = "tumor_model_interpolated.stl"
threshold = 130
slice_thickness_mm = 6.0  # real spacing between original slices
interpolation_factor = 5 # create 5x more slices

# === Load and Stack MRI Slices ===
slice_files = sorted(os.listdir(folder_path))
slices = [imageio.imread(os.path.join(folder_path, f)) for f in slice_files]
volume = np.stack(slices, axis=0)
print("Original volume shape:", volume.shape)

# === Interpolate to Fake More Depth ===
volume_interp = zoom(volume, (interpolation_factor, 1, 1), order=1)

z = 6 # slice index (depth)
y = 324  # row index (height)
x = 166  # column index (width)

intensity = volume_interp[z, y, x]
print(f"Intensity at (z={z}, y={y}, x={x}): {intensity}")


print("Interpolated volume shape:", volume_interp.shape)

# === Create Binary Mask ===
binary_volume = (volume_interp < threshold).astype(np.uint8)
print("Nonzero voxels after thresholding:", np.count_nonzero(binary_volume))

# === Marching Cubes Surface Extraction ===
verts, faces, normals, values = measure.marching_cubes(binary_volume, level=0)

# === Scale Vertices to Match Real-World Dimensions ===
verts[:, 2] *= (slice_thickness_mm / interpolation_factor)  # adjust new slice spacing

# === Export to STL ===
mesh = trimesh.Trimesh(vertices=verts, faces=faces)
mesh.export(output_path)
print(f"STL exported to: {output_path}")

