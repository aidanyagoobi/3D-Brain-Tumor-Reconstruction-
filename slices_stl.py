import os
import numpy as np
from scipy.io import loadmat
from wpiecewise import TumorSlice

# Import the STL mesh class from numpy-stl.
try:
    from stl import mesh
except ImportError:
    raise ImportError("Please install numpy-stl (pip install numpy-stl)")

def load_tumor_mask(mat_file):
    """
    Load the tumor mask from a MATLAB .mat file.
    First, try using scipy.io.loadmat; if that fails (for MATLAB v7.3 files),
    fall back to using h5py.
    """
    try:
        data = loadmat(mat_file, squeeze_me=True, struct_as_record=False)
        if 'cjdata' not in data:
            print(f"{mat_file}: 'cjdata' not found")
            return None
        cjdata = data['cjdata']
        if not hasattr(cjdata, 'tumorMask'):
            print(f"{mat_file}: 'tumorMask' not found")
            return None
        return cjdata.tumorMask
    except Exception as e:
        try:
            import h5py
            with h5py.File(mat_file, 'r') as f:
                if 'cjdata' not in f:
                    print(f"{mat_file}: 'cjdata' not found in HDF5")
                    return None
                grp = f['cjdata']
                if 'tumorMask' not in grp:
                    print(f"{mat_file}: 'tumorMask' not found in HDF5")
                    return None
                mask = grp['tumorMask'][()]
                return mask
        except Exception as he:
            print(f"Error loading {mat_file} with h5py: {he}")
            return None

def crop_to_tumor(mask):
    """
    Crop the mask to its bounding box defined by nonzero tumor pixels.
    Returns:
      cropped_mask: the extracted region,
      row_min, row_max, col_min, col_max: the pixel indices of the bounding box.
    If no nonzero region is found, returns (None, None, None, None, None).
    """
    tumor_coords = np.argwhere(mask)
    if tumor_coords.size == 0:
        return None, None, None, None, None
    row_min, col_min = tumor_coords.min(axis=0)
    row_max, col_max = tumor_coords.max(axis=0)
    cropped_mask = mask[row_min:row_max+1, col_min:col_max+1]
    return cropped_mask, row_min, row_max, col_min, col_max

def generate_triangles_for_slice(cropped_mask, row_min, col_min, pixel_size, z_offset):
    """
    Given a cropped tumor mask and its starting indices (row_min and col_min) in the full image,
    generate triangles for the slice. For every pixel (cell) in the cropped_mask that is nonzero,
    this function creates a full square using two triangles that cover the entire pixel area.
    
    Each pixel covers the area from:
      (col_min + j)*pixel_size to (col_min + j + 1)*pixel_size (in x), and
      (row_min + i)*pixel_size to (row_min + i + 1)*pixel_size (in y).
    
    Returns:
      An array of triangles with shape (n_triangles, 3, 3) where each triangle is defined 
      by three vertices (x, y, z).
    """
    nrows, ncols = cropped_mask.shape

    # Compute boundaries for every pixel (ncols+1 and nrows+1 grid points).
    x_coords = np.arange(col_min, col_min + ncols + 1) * pixel_size
    y_coords = np.arange(row_min, row_min + nrows + 1) * pixel_size

    triangles = []
    # Loop over every pixel in the cropped mask.
    for i in range(nrows):
        for j in range(ncols):
            # Check all four corners if any is nonzero (or you can check just the current pixel)
            if cropped_mask[i, j] != 0:
                # Define the corners of the pixel's square.
                bottom_left  = np.array([x_coords[j],   y_coords[i],   z_offset])
                bottom_right = np.array([x_coords[j+1], y_coords[i],   z_offset])
                top_left     = np.array([x_coords[j],   y_coords[i+1], z_offset])
                top_right    = np.array([x_coords[j+1], y_coords[i+1], z_offset])
                
                # Split the pixel along the diagonal from bottom_left to top_right.
                tri1 = np.array([bottom_left, bottom_right, top_right])
                tri2 = np.array([bottom_left, top_right, top_left])
                triangles.append(tri1)
                triangles.append(tri2)
    
    if triangles:
         return np.array(triangles)
    else:
         return np.empty((0, 3, 3))

def rotate_triangles_set1(triangles):
    """
    Rotate triangles 90° about the y axis.
    Using the rotation:
      x' = z, y' = y, z' = -x.
    """
    rotated = np.empty_like(triangles)
    rotated[:,:,0] = -triangles[:,:,1]   # new x = -y
    rotated[:,:,1] = -triangles[:,:,2]   # new y = -z
    rotated[:,:,2] = triangles[:,:,0]    # new z = x
    rotated = -rotated
    return rotated

def rotate_triangles_set2(triangles):
    """
    Return triangles unchanged (no rotation).
    """
    rotated = np.empty_like(triangles)
    rotated[:,:,0] = triangles[:,:,1]    # x' = y
    rotated[:,:,1] = -triangles[:,:,0]   # y' = -x
    rotated[:,:,2] = triangles[:,:,2]    # z' = z
    rotated[:,:,2] = -rotated[:,:,2]
    return rotated

def rotate_triangles_set3(triangles):
    """
    Rotate triangles -90° about the x axis.
    For each vertex (x, y, z), apply:
      x' = x,
      y' = z,
      z' = -y.
    """
    rotated = np.empty_like(triangles)
    rotated[:,:,0] = -triangles[:,:,2]    # new x = -z
    rotated[:,:,1] = triangles[:,:,0]     # new y = x
    rotated[:,:,2] = -triangles[:,:,1]    # new z = -y
    rotated[:,:,1] = -rotated[:,:,1]
    rotated[:,:,2] = rotated[:,:,2]
    rotated[:,:,0] = rotated[:,:,0] #slice axis
    return rotated



def move_to_origin(triangles):
    """
    Compute the bounding box center for the given triangles and subtract this center from every vertex.
    triangles: numpy array of shape (n, 3, 3)
    Returns the translated triangles.
    """
    if triangles.size == 0:
        return triangles
    # Flatten vertices to get all points
    pts = triangles.reshape(-1, 3)
    # Compute bounding box center
    center = (pts.min(axis=0) + pts.max(axis=0)) / 2.0
    # Translate all vertices so that the center moves to (0,0,0)
    return triangles - center

def main():
    # Process first set: files 49.mat to 55.mat.
    group1_files = [os.path.join("data", f"{i}.mat") for i in range(49, 56)]
    # Process second set: files 275.mat to 280.mat.
    group2_files = [os.path.join("data", f"{i}.mat") for i in range(275, 281)]
    # Process third set: files 510.mat to 513.mat.
    group3_files = [os.path.join("data", f"{i}.mat") for i in range(510, 514)]
    
    # Parameters.
    pixel_size = 0.49  # mm per pixel in-plane resolution.
    z_spacing = 7.0    # mm between slice centers.
    
    """Edit this group of triangles."""
    def build_group_triangles(mat_files,
                              rotate_fn,
                              sub_div=1,
                              base_spacing=z_spacing):
        # 1) load the seven original masks as TumorSlice objects
        ts = []
        for mf in mat_files:
            m = load_tumor_mask(mf)
            if m is None:
                print("Skipping", mf, "(no mask)")
                continue
            t = TumorSlice(None, None, None,
                           tumor_mask_processed=np.squeeze(m).astype(bool))
            t.compute_signed_distance()
            ts.append(t)

        # 2) run piece-wise weighted interpolation `sub_div` times
        for _ in range(sub_div):
            ts = TumorSlice.run_interpolation(ts)

        # 3) convert every slice (orig + interp) to triangles
        tris_list = []
        slice_spacing = base_spacing / (2 ** sub_div)
        for idx, t in enumerate(ts):
            bin_mask = (t.signed_distance < 0).astype(np.uint8)
            cropped, r0, r1, c0, c1 = crop_to_tumor(bin_mask)
            if cropped is None:
                continue
            z_off = idx * slice_spacing
            tris = generate_triangles_for_slice(
                cropped, r0, c0, pixel_size, z_off
            )
            if tris.size:
                tris_list.append(tris)

        if tris_list:
            tri_arr = np.concatenate(tris_list, axis=0)
            tri_arr = rotate_fn(tri_arr)
            return tri_arr
        return np.empty((0, 3, 3))

    # ===== build all three groups with the same interpolation =====
    SUBDIV = 3          # 1 → double the slice count; increase for more density

    group1_triangles = build_group_triangles(
        group1_files, rotate_triangles_set1, SUBDIV, z_spacing
    )
    group2_triangles = build_group_triangles(
        group2_files, rotate_triangles_set2, SUBDIV, z_spacing
    )
    group3_triangles = build_group_triangles(
        group3_files, rotate_triangles_set3, SUBDIV, z_spacing
    )
         
    group1_triangles = move_to_origin(group1_triangles)
    group2_triangles = move_to_origin(group2_triangles)
    group3_triangles = move_to_origin(group3_triangles)
    
    # Combine triangles from both groups.
    all_triangles = np.concatenate([group1_triangles, group2_triangles, group3_triangles], axis=0)
    print(f"Total triangles generated: {all_triangles.shape[0]}.")

    # Create the output directory.
    out_dir = os.path.join("outputTemp")
    os.makedirs(out_dir, exist_ok=True)

    # Save group 1 as a temporary STL.
    temp_file1 = os.path.join(out_dir, "temp_group1_DowntoTop.stl")
    stl_mesh1 = mesh.Mesh(np.zeros(group1_triangles.shape[0], dtype=mesh.Mesh.dtype))
    stl_mesh1.vectors = group1_triangles
    stl_mesh1.save(temp_file1)
    print(f"Temporary STL for group 1 saved to {temp_file1}")

    # Save group 2 as a temporary STL.
    temp_file2 = os.path.join(out_dir, "temp_group2_FronttoBack.stl")
    stl_mesh2 = mesh.Mesh(np.zeros(group2_triangles.shape[0], dtype=mesh.Mesh.dtype))
    stl_mesh2.vectors = group2_triangles
    stl_mesh2.save(temp_file2)
    print(f"Temporary STL for group 2 saved to {temp_file2}")

    # Save group 3 as a temporary STL.
    temp_file3 = os.path.join(out_dir, "temp_group3_LefttoRight.stl")
    stl_mesh3 = mesh.Mesh(np.zeros(group3_triangles.shape[0], dtype=mesh.Mesh.dtype))
    stl_mesh3.vectors = group3_triangles
    stl_mesh3.save(temp_file3)
    print(f"Temporary STL for group 3 saved to {temp_file3}")

    # Overlay the three STL files by loading them and combining their triangles.
    mesh1 = mesh.Mesh.from_file(temp_file1)
    mesh2 = mesh.Mesh.from_file(temp_file2)
    mesh3 = mesh.Mesh.from_file(temp_file3)
    all_vectors = np.concatenate([mesh1.vectors, mesh2.vectors, mesh3.vectors], axis=0)
    combined_mesh = mesh.Mesh(np.zeros(all_vectors.shape[0], dtype=mesh.Mesh.dtype))
    combined_mesh.vectors = all_vectors

    out_file = os.path.join(out_dir, "slices_combined_104281.stl")
    combined_mesh.save(out_file)
    print(f"Combined STL file saved to {out_file}")

if __name__ == "__main__":
    main()
