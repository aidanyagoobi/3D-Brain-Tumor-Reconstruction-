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
    try:
        data = loadmat(mat_file, squeeze_me=True, struct_as_record=False)
        cjdata = data.get('cjdata', None)
        if cjdata is None or not hasattr(cjdata, 'tumorMask'):
            print(f"{mat_file}: no 'tumorMask' found")
            return None
        return np.squeeze(cjdata.tumorMask)
    except Exception:
        import h5py
        with h5py.File(mat_file, 'r') as f:
            grp = f.get('cjdata', None)
            if grp is None or 'tumorMask' not in grp:
                print(f"{mat_file}: no 'tumorMask' in HDF5")
                return None
            return grp['tumorMask'][()]

def crop_to_tumor(mask):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None, None, None, None, None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0:r1+1, c0:c1+1], r0, r1, c0, c1

def generate_triangles_for_slice(cropped_mask, row_min, col_min, pixel_size, z_offset):
    nrows, ncols = cropped_mask.shape
    x = (col_min + np.arange(ncols+1)) * pixel_size
    y = (row_min + np.arange(nrows+1)) * pixel_size
    tris = []
    for i in range(nrows):
        for j in range(ncols):
            if not cropped_mask[i,j]:
                continue
            # only boundary pixels
            boundary = False
            for di,dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                ni, nj = i+di, j+dj
                if ni<0 or ni>=nrows or nj<0 or nj>=ncols or cropped_mask[ni,nj]==0:
                    boundary = True; break
            if not boundary:
                continue
            bl = np.array([ x[j],   y[i],   z_offset])
            br = np.array([ x[j+1], y[i],   z_offset])
            tl = np.array([ x[j],   y[i+1], z_offset])
            tr = np.array([ x[j+1], y[i+1], z_offset])
            tris.append(np.vstack((bl, br, tr)))
            tris.append(np.vstack((bl, tr, tl)))
    return np.array(tris) if tris else np.empty((0,3,3))

# --- FIXED ROTATIONS (copied from full‐mask script) ---
def rotate_triangles_set1(triangles):
    rotated = np.empty_like(triangles)
    rotated[:,:,0] = -triangles[:,:,1]   # new x = -y
    rotated[:,:,1] = -triangles[:,:,2]   # new y = -z
    rotated[:,:,2] =  triangles[:,:,0]   # new z = x
    rotated = -rotated                   # invert all to match original
    return rotated

def rotate_triangles_set2(triangles):
    rotated = np.empty_like(triangles)
    rotated[:,:,0] =  triangles[:,:,1]   # x' = y
    rotated[:,:,1] = -triangles[:,:,0]   # y' = -x
    rotated[:,:,2] =  triangles[:,:,2]   # z' = z
    rotated[:,:,2] = -rotated[:,:,2]     # invert z
    return rotated

def rotate_triangles_set3(triangles):
    rotated = np.empty_like(triangles)
    rotated[:,:,0] = -triangles[:,:,2]   # new x = -z
    rotated[:,:,1] =  triangles[:,:,0]   # new y = x
    rotated[:,:,2] =  triangles[:,:,1]   # new z = y
    rotated[:,:,2] = -rotated[:,:,2]     # invert z
    rotated[:,:,1] = -rotated[:,:,1]
    return rotated
# -----------------------------------------------

def move_to_origin(triangles):
    if triangles.size == 0:
        return triangles
    pts = triangles.reshape(-1,3)
    center = (pts.min(axis=0) + pts.max(axis=0)) / 2.0
    return triangles - center

def main():
    # ---- slice-ID ranges for the three orientations ----
    ranges = [range(49, 56), range(275, 281), range(510, 514)]
    pixel_size = 0.49
    z_spacing  = 7.0
    SUBDIV     = 8          # 1 doubles slice count; 2 quadruples, etc.

    out_dir  = os.path.join("output", "104281")
    temp_dir = os.path.join(out_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    # ---------- helper to build one group ----------
    def build_interp_outer(mat_ids, rotate_fn):
        files = [os.path.join("data", f"{i}.mat") for i in mat_ids]
        ts = []
        for f in files:
            m = load_tumor_mask(f)
            if m is None:
                continue
            t = TumorSlice(None, None, None,
                           tumor_mask_processed=np.squeeze(m).astype(bool))
            t.compute_signed_distance()
            ts.append(t)

        for _ in range(SUBDIV):
            ts = TumorSlice.run_interpolation(ts)

        tris_list = []
        slice_spacing = z_spacing / (2 ** SUBDIV)
        for idx, t in enumerate(ts):
            bin_mask = (t.signed_distance < 0).astype(np.uint8)
            crop, r0, r1, c0, c1 = crop_to_tumor(bin_mask)
            if crop is None:
                continue
            z_off = idx * slice_spacing
            tris  = generate_triangles_for_slice(crop, r0, c0,
                                                 pixel_size, z_off)
            if tris.size:
                tris_list.append(tris)

        if not tris_list:
            return np.empty((0, 3, 3))
        arr = np.concatenate(tris_list, axis=0)
        arr = rotate_fn(arr)
        return move_to_origin(arr)
    # -------------------------------------------------

    group1 = build_interp_outer(ranges[0], rotate_triangles_set1)
    group2 = build_interp_outer(ranges[1], rotate_triangles_set2)
    group3 = build_interp_outer(ranges[2], rotate_triangles_set3)

    all_groups = [g for g in (group1, group2, group3) if g.size]
    if not all_groups:
        print("No triangles generated.")
        return

    combined = np.concatenate(all_groups, axis=0)
    final_mesh = mesh.Mesh(np.zeros(combined.shape[0], dtype=mesh.Mesh.dtype))
    final_mesh.vectors = combined
    final_path = os.path.join(out_dir, "slices_combined_outer_104281.stl")
    final_mesh.save(final_path)
    print(f"Saved culled-slice outer STL to {final_path}")

if __name__ == '__main__':
    main()
