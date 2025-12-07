#!/bin/bash
# Stage 2 IDU - FLUX diffusion for street-level detail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate skyfall-gs
cd ~/Skyfall-GS

export TORCH_CUDA_ARCH_LIST="9.0"

DATASET=${1:-"./data/datasets_JAX/JAX_068/"}
CHECKPOINT=${2:-"./outputs/JAX/JAX_068_stage1/chkpnt25000.pth"}
OUTPUT=${3:-"./outputs/JAX/JAX_068_stage2_idu"}

python train.py -s "$DATASET" -m "$OUTPUT" \
    --start_checkpoint "$CHECKPOINT" --iterative_datasets_update --eval \
    --kernel_size 0.1 --resolution 1 --sh_degree 1 --appearance_enabled \
    --lambda_depth 0 --lambda_opacity 0 --idu_opacity_reset_interval 5000 \
    --idu_refine --idu_num_samples_per_view 2 --densify_grad_threshold 0.0002 \
    --idu_num_cams 6 --idu_use_flow_edit --idu_render_size 1024 \
    --idu_flow_edit_n_min 4 --idu_flow_edit_n_max 10 \
    --idu_grid_size 3 --idu_grid_width 512 --idu_grid_height 512 \
    --idu_episode_iterations 10000 --lambda_pseudo_depth 0.5

