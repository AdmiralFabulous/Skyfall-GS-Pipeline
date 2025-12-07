"""
Export Gaussian Splatting model to actual mesh with faces
Uses marching cubes on a density volume created from Gaussians
"""

import sys

sys.path.insert(0, "/home/lx/Skyfall-GS")

import torch
import numpy as np
from plyfile import PlyData, PlyElement
import os
from scene.gaussian_model import GaussianModel


def gaussians_to_mesh(gaussians, output_path, resolution=256, density_threshold=0.5):
    """
    Convert Gaussians to mesh using volumetric density and marching cubes
    """
    from skimage import measure

    print(f"Creating density volume at resolution {resolution}...")

    # Get Gaussian parameters
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    scales = gaussians.get_scaling.detach().cpu().numpy()
    opacity = gaussians.get_opacity.detach().cpu().numpy().squeeze()
    features_dc = gaussians._features_dc.detach().cpu().numpy()

    # Filter by opacity
    mask = opacity > 0.1
    xyz = xyz[mask]
    scales = np.exp(scales[mask])  # Convert from log scale
    opacity = opacity[mask]
    features_dc = features_dc[mask]

    # Get colors
    C0 = 0.28209479177387814
    colors = 0.5 + features_dc[:, 0, :] * C0
    colors = np.clip(colors, 0, 1)

    print(f"Processing {len(xyz)} Gaussians...")

    # Compute bounds with padding
    padding = 0.1
    bounds_min = xyz.min(axis=0) - padding
    bounds_max = xyz.max(axis=0) + padding

    # Create volume grid
    x = np.linspace(bounds_min[0], bounds_max[0], resolution)
    y = np.linspace(bounds_min[1], bounds_max[1], resolution)
    z = np.linspace(bounds_min[2], bounds_max[2], resolution)

    voxel_size = (bounds_max - bounds_min) / resolution
    print(f"Voxel size: {voxel_size}")

    # Initialize density and color volumes
    density = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    color_volume = np.zeros((resolution, resolution, resolution, 3), dtype=np.float32)
    weight_volume = np.zeros((resolution, resolution, resolution), dtype=np.float32)

    # Splat Gaussians into volume (simplified - treating as isotropic)
    print("Splatting Gaussians into volume...")

    # Use average scale as radius for each Gaussian
    radii = scales.mean(axis=1) * 2  # 2 sigma

    for i in range(len(xyz)):
        if i % 10000 == 0:
            print(f"  Processing Gaussian {i}/{len(xyz)}...")

        pos = xyz[i]
        r = min(radii[i], 0.5)  # Clamp radius
        col = colors[i]
        op = opacity[i]

        # Find voxels within radius
        ix = int((pos[0] - bounds_min[0]) / voxel_size[0])
        iy = int((pos[1] - bounds_min[1]) / voxel_size[1])
        iz = int((pos[2] - bounds_min[2]) / voxel_size[2])

        # Radius in voxels
        rv = int(np.ceil(r / voxel_size.mean())) + 1

        # Splat into nearby voxels
        for dx in range(-rv, rv + 1):
            for dy in range(-rv, rv + 1):
                for dz in range(-rv, rv + 1):
                    vx, vy, vz = ix + dx, iy + dy, iz + dz
                    if (
                        0 <= vx < resolution
                        and 0 <= vy < resolution
                        and 0 <= vz < resolution
                    ):
                        # Distance from voxel center to Gaussian center
                        voxel_pos = (
                            bounds_min
                            + np.array([vx, vy, vz]) * voxel_size
                            + voxel_size / 2
                        )
                        dist = np.linalg.norm(voxel_pos - pos)

                        # Gaussian falloff
                        if dist < r * 3:
                            weight = op * np.exp(-0.5 * (dist / (r + 0.01)) ** 2)
                            density[vx, vy, vz] += weight
                            color_volume[vx, vy, vz] += col * weight
                            weight_volume[vx, vy, vz] += weight

    # Normalize colors
    valid = weight_volume > 0
    color_volume[valid] /= weight_volume[valid, np.newaxis]

    # Normalize density
    if density.max() > 0:
        density = density / density.max()

    print(f"Density range: {density.min():.4f} to {density.max():.4f}")
    print(f"Running marching cubes with threshold {density_threshold}...")

    # Marching cubes
    try:
        verts, faces, normals, values = measure.marching_cubes(
            density, level=density_threshold, spacing=voxel_size
        )

        # Offset vertices to world coordinates
        verts = verts + bounds_min

        print(f"Generated mesh: {len(verts)} vertices, {len(faces)} faces")

        # Sample colors at vertex positions
        vert_colors = np.zeros((len(verts), 3), dtype=np.uint8)
        for i, v in enumerate(verts):
            # Find nearest voxel
            ix = int(np.clip((v[0] - bounds_min[0]) / voxel_size[0], 0, resolution - 1))
            iy = int(np.clip((v[1] - bounds_min[1]) / voxel_size[1], 0, resolution - 1))
            iz = int(np.clip((v[2] - bounds_min[2]) / voxel_size[2], 0, resolution - 1))
            vert_colors[i] = (color_volume[ix, iy, iz] * 255).astype(np.uint8)

        # Save as PLY with faces
        save_mesh_ply(verts, faces, vert_colors, normals, output_path)

        # Also save as OBJ
        obj_path = output_path.replace(".ply", ".obj")
        save_mesh_obj(verts, faces, vert_colors, obj_path)

        return True

    except Exception as e:
        print(f"Marching cubes failed: {e}")
        print("Try lowering the density_threshold parameter")
        return False


def save_mesh_ply(verts, faces, colors, normals, output_path):
    """Save mesh as PLY with vertex colors"""

    # Vertex data
    vertex_dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    vertices = np.zeros(len(verts), dtype=vertex_dtype)
    vertices["x"] = verts[:, 0]
    vertices["y"] = verts[:, 1]
    vertices["z"] = verts[:, 2]
    vertices["nx"] = normals[:, 0]
    vertices["ny"] = normals[:, 1]
    vertices["nz"] = normals[:, 2]
    vertices["red"] = colors[:, 0]
    vertices["green"] = colors[:, 1]
    vertices["blue"] = colors[:, 2]

    # Face data
    face_dtype = [("vertex_indices", "i4", (3,))]
    faces_arr = np.zeros(len(faces), dtype=face_dtype)
    faces_arr["vertex_indices"] = faces

    vertex_el = PlyElement.describe(vertices, "vertex")
    face_el = PlyElement.describe(faces_arr, "face")

    PlyData([vertex_el, face_el], text=False).write(output_path)
    print(f"Saved mesh PLY to: {output_path}")


def save_mesh_obj(verts, faces, colors, output_path):
    """Save mesh as OBJ with vertex colors"""

    with open(output_path, "w") as f:
        f.write("# Skyfall-GS Mesh Export\n")
        f.write(f"# {len(verts)} vertices, {len(faces)} faces\n\n")

        # Write vertices with colors
        for v, c in zip(verts, colors):
            f.write(
                f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0] / 255:.4f} {c[1] / 255:.4f} {c[2] / 255:.4f}\n"
            )

        f.write("\n")

        # Write faces (OBJ uses 1-based indexing)
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

    print(f"Saved mesh OBJ to: {output_path}")


def main():
    ply_path = "/home/lx/Skyfall-GS/outputs/JAX/JAX_068_v2/point_cloud/iteration_25000/point_cloud.ply"
    export_dir = "/mnt/a/CLAUDE CODE/ELIVIEW CLAUDE CODE/skyfall_exports"

    print("=" * 60)
    print("Skyfall-GS Mesh Export (with faces)")
    print("=" * 60)

    # Load model
    print(f"\nLoading model from: {ply_path}")
    gaussians = GaussianModel(
        sh_degree=1,
        appearance_enabled=False,
        appearance_n_fourier_freqs=4,
        appearance_embedding_dim=32,
    )
    gaussians.load_ply(ply_path)
    print(f"Loaded {gaussians.get_xyz.shape[0]:,} Gaussians")

    # Export mesh
    os.makedirs(export_dir, exist_ok=True)
    mesh_path = os.path.join(export_dir, "JAX_068_mesh.ply")

    # Try with different resolutions/thresholds
    success = gaussians_to_mesh(
        gaussians,
        mesh_path,
        resolution=128,  # Lower = faster, higher = more detail
        density_threshold=0.1,  # Lower = more surface, higher = tighter surface
    )

    if success:
        print("\n" + "=" * 60)
        print("Mesh Export Complete!")
        print("=" * 60)
        print(f"\nFiles saved to: {export_dir}")
        print("  - JAX_068_mesh.ply (with faces)")
        print("  - JAX_068_mesh.obj")
    else:
        print("\nMesh generation failed. Try adjusting parameters.")


if __name__ == "__main__":
    main()
