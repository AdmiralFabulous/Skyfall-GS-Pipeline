# Skyfall-GS Pipeline

Satellite-to-Street-Level 3D Reconstruction using Gaussian Splatting.

## Overview

Combines two approaches:
1. **Skyfall-GS (Option A)**: Train on satellite imagery, use FLUX diffusion for street-level detail
2. **Vanilla 3DGS (Option C)**: Standard Gaussian Splatting for street panoramas

## Hardware
- GPU: RTX 5090 (32GB), Threadripper PRO 5975WX, 64GB RAM
- OS: Windows 11 + WSL2 Ubuntu 24.04

## Milestones

| Version | Date | Description |
|---------|------|-------------|
| v0.1.0 | Dec 2025 | Initial repository setup |

## References
- [Skyfall-GS Paper](https://arxiv.org/abs/2410.09566)
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)

