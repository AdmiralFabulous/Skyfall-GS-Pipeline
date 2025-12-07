"""
Export Gaussian Splatting model to textured 3D mesh
Converts point cloud to mesh with vertex colors

Run this in WSL:
  cd ~/Skyfall-GS
  conda activate skyfall
  python /mnt/a/CLAUDE\ CODE/ELIVIEW\ CLAUDE\ CODE/export_mesh.py
"""

import sys

sys.path.insert(0, "/home/lx/Skyfall-GS")

import torch
import numpy as np
from plyfile import PlyData, PlyElement
import os
from scene.gaussian_model import GaussianModel


def export_colored_pointcloud(gaussians, output_path):
    """Export Gaussians as colored point cloud PLY"""
    xyz = gaussians.get_xyz.detach().cpu().numpy()

    # Get colors from spherical harmonics (DC component = base color)
    features_dc = gaussians._features_dc.detach().cpu().numpy()

    # Convert SH DC to RGB
    # The DC component is features_dc[:, 0, :] with shape (N, 3)
    colors = features_dc[:, 0, :]  # Shape: (N, 3)

    # SH to color conversion: C = 0.5 + SH_DC * C0 where C0 = 0.28209479177387814
    C0 = 0.28209479177387814
    colors = 0.5 + colors * C0
    colors = np.clip(colors * 255, 0, 255).astype(np.uint8)

    # Get opacity for filtering
    opacity = gaussians.get_opacity.detach().cpu().numpy().squeeze()

    # Filter by opacity (keep visible points)
    mask = opacity > 0.1
    xyz_filtered = xyz[mask]
    colors_filtered = colors[mask]

    print(f"Exporting {len(xyz_filtered)} points (filtered from {len(xyz)} by opacity)")

    # Create PLY with colors
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    vertices = np.zeros(len(xyz_filtered), dtype=dtype)
    vertices["x"] = xyz_filtered[:, 0]
    vertices["y"] = xyz_filtered[:, 1]
    vertices["z"] = xyz_filtered[:, 2]
    vertices["red"] = colors_filtered[:, 0]
    vertices["green"] = colors_filtered[:, 1]
    vertices["blue"] = colors_filtered[:, 2]

    el = PlyElement.describe(vertices, "vertex")
    PlyData([el], text=False).write(output_path)
    print(f"Saved colored point cloud to: {output_path}")
    return xyz_filtered, colors_filtered


def export_as_obj_with_points(xyz, colors, output_base):
    """Export as OBJ with vertex colors"""
    obj_path = output_base.replace(".ply", ".obj")

    # Normalize colors to 0-1 range
    colors_norm = colors.astype(float) / 255.0

    with open(obj_path, "w") as f:
        f.write(f"# Skyfall-GS Exported Point Cloud\n")
        f.write(f"# {len(xyz)} vertices\n\n")

        # Write vertices with colors (OBJ supports vertex colors as extension)
        for i, (pos, col) in enumerate(zip(xyz, colors_norm)):
            f.write(
                f"v {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} {col[0]:.4f} {col[1]:.4f} {col[2]:.4f}\n"
            )

    print(f"Saved OBJ to: {obj_path}")
    return obj_path


def create_splat_file(gaussians, output_base):
    """Export as .splat format for web viewers like https://antimatter15.com/splat/"""
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    scales = gaussians.get_scaling.detach().cpu().numpy()
    rotations = gaussians.get_rotation.detach().cpu().numpy()
    opacity = gaussians.get_opacity.detach().cpu().numpy().squeeze()
    features_dc = gaussians._features_dc.detach().cpu().numpy()

    # Filter by opacity
    mask = opacity > 0.1

    # Convert SH to colors
    C0 = 0.28209479177387814
    colors = 0.5 + features_dc[:, 0, :] * C0
    colors = np.clip(colors * 255, 0, 255).astype(np.uint8)

    # Apply mask
    xyz = xyz[mask]
    scales = scales[mask]
    rotations = rotations[mask]
    opacity = opacity[mask]
    colors = colors[mask]

    print(f"Exporting {len(xyz)} splats for web viewer")

    splat_path = output_base.replace(".ply", ".splat")

    # .splat binary format for antimatter15 viewer
    # Each splat: position(3xf32) + scale(3xf32) + color(4xu8 RGBA) + rotation(4xf32)
    with open(splat_path, "wb") as f:
        for i in range(len(xyz)):
            # Position (3 floats)
            f.write(xyz[i].astype(np.float32).tobytes())
            # Scale (3 floats, stored as log in model, need exp)
            s = np.exp(scales[i])
            f.write(s.astype(np.float32).tobytes())
            # RGBA (4 bytes)
            f.write(colors[i].tobytes())
            f.write(np.array([int(opacity[i] * 255)], dtype=np.uint8).tobytes())
            # Rotation quaternion (4 floats)
            f.write(rotations[i].astype(np.float32).tobytes())

    print(f"Saved .splat to: {splat_path}")
    return splat_path


def main():
    # Paths
    ply_path = "/home/lx/Skyfall-GS/outputs/JAX/JAX_068_v2/point_cloud/iteration_25000/point_cloud.ply"
    export_dir = "/mnt/a/CLAUDE CODE/ELIVIEW CLAUDE CODE/skyfall_exports"

    print("=" * 60)
    print("Skyfall-GS Model Export")
    print("=" * 60)
    print(f"\nLoading model from: {ply_path}")

    # Load model
    gaussians = GaussianModel(
        sh_degree=1,
        appearance_enabled=False,
        appearance_n_fourier_freqs=4,
        appearance_embedding_dim=32,
    )
    gaussians.load_ply(ply_path)

    n_gaussians = gaussians.get_xyz.shape[0]
    print(f"Loaded {n_gaussians:,} Gaussians")

    # Scene statistics
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    center = xyz.mean(axis=0)
    bounds_min = xyz.min(axis=0)
    bounds_max = xyz.max(axis=0)
    size = bounds_max - bounds_min

    print(f"\nScene Statistics:")
    print(f"  Center: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")
    print(
        f"  Bounds: [{bounds_min[0]:.2f} to {bounds_max[0]:.2f}] x [{bounds_min[1]:.2f} to {bounds_max[1]:.2f}] x [{bounds_min[2]:.2f} to {bounds_max[2]:.2f}]"
    )
    print(f"  Size: {size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f} units")

    # Create export directory
    os.makedirs(export_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("Exporting formats...")
    print("=" * 60)

    base_output = os.path.join(export_dir, "JAX_068_textured.ply")

    # 1. Colored point cloud PLY
    xyz_filtered, colors = export_colored_pointcloud(gaussians, base_output)

    # 2. OBJ with vertex colors
    export_as_obj_with_points(xyz_filtered, colors, base_output)

    # 3. .splat for web viewers
    create_splat_file(gaussians, base_output)

    print("\n" + "=" * 60)
    print("Export Complete!")
    print("=" * 60)
    print(f"\nFiles saved to: {export_dir}")
    print("\nView in:")
    print("  - MeshLab, CloudCompare, Blender: .ply or .obj")
    print("  - Web viewer: https://antimatter15.com/splat/ (.splat)")
    print("  - SuperSplat: https://playcanvas.com/supersplat (.ply)")


if __name__ == "__main__":
    main()
