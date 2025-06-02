import os
import numpy as np
import open3d as o3d
from skimage import measure
from scipy import ndimage

def fill_slices(vol, axis):
    moved = np.moveaxis(vol, axis, 0)
    for i in range(moved.shape[0]):
        moved[i] = ndimage.binary_fill_holes(moved[i])
    return np.moveaxis(moved, 0, axis)

def main():
    IN_STL     = "output/104281/slices_combined_outer_104281.stl"
    OUT_STL    = "output/104281/solid_tumor_mc.stl"
    VOXEL_SIZE = 0.3
    SLICE_SP   = 7.0

    radius_vox  = int(np.ceil(1.2 * SLICE_SP / VOXEL_SIZE))
    base_struct = ndimage.generate_binary_structure(3, 1)

    os.makedirs(os.path.dirname(OUT_STL), exist_ok=True)

    print("1) Loading STL")
    mesh = o3d.io.read_triangle_mesh(IN_STL)
    if not mesh.has_triangles():
        raise RuntimeError("Failed to load mesh")

    print("2) Voxelizing")
    vg = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=VOXEL_SIZE)
    origin = np.array(vg.origin)
    vs     = vg.voxel_size
    idx    = np.array([v.grid_index for v in vg.get_voxels()], int)

    print("3) Building occupancy")
    max_i, max_j, max_k = idx.max(axis=0)
    vol = np.zeros((max_i+1, max_j+1, max_k+1), bool)
    vol[idx[:,0], idx[:,1], idx[:,2]] = True

    print("   dilating raw occupancy (2 iters)…")
    vol = ndimage.binary_dilation(vol, structure=base_struct, iterations=2)

    pad = radius_vox
    vol = np.pad(vol, pad, mode='constant', constant_values=False)
    origin -= pad * vs

    print("   filling 2D holes")
    vol = vol \
        | fill_slices(vol, 0) \
        | fill_slices(vol, 1) \
        | fill_slices(vol, 2)

    # ==== REPLACED CLOSING ==== 
    print(f"   dilating for closing ({radius_vox} iters)…")
    vol = ndimage.binary_dilation(vol, structure=base_struct, iterations=radius_vox)
    print(f"   eroding for closing ({radius_vox} iters)…")
    vol = ndimage.binary_erosion(vol, structure=base_struct, iterations=radius_vox)
    # =========================

    print("   filling internal cavities")
    vol = ndimage.binary_fill_holes(vol)

    print("4) Running marching_cubes")
    verts, faces, normals, _ = measure.marching_cubes(vol, level=0.5, spacing=(vs,vs,vs))
    verts += origin

    mesh_mc = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(verts),
        triangles=o3d.utility.Vector3iVector(faces)
    )
    mesh_mc.vertex_normals = o3d.utility.Vector3dVector(normals)

    print("5) Cleaning + saving")
    mesh_mc.remove_duplicated_vertices()
    mesh_mc.remove_degenerate_triangles()
    mesh_mc.remove_non_manifold_edges()
    mesh_mc.compute_vertex_normals()
    o3d.io.write_triangle_mesh(
        OUT_STL, mesh_mc,
        write_ascii=False, compressed=False,
        write_vertex_normals=True,
        write_vertex_colors=False,
        write_triangle_uvs=False
    )
    print("Done.")

if __name__ == "__main__":
    main()
