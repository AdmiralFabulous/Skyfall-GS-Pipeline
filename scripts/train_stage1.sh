#!/bin/bash
# Stage 1 Training - Satellite imagery
source ~/miniconda3/etc/profile.d/conda.sh
conda activate skyfall-gs
cd ~/Skyfall-GS

export TORCH_CUDA_ARCH_LIST="9.0"

DATASET=${1:-"./data/datasets_JAX/JAX_068/"}
OUTPUT=${2:-"./outputs/JAX/JAX_068_stage1"}

python train.py -s "$DATASET" -m "$OUTPUT" --eval \
    --kernel_size 0.1 --resolution 1 --sh_degree 1 \
    --appearance_enabled --lambda_depth 0 --iterations 25000 \
    --densify_until_iter 5000 --densify_grad_threshold 0.0005

