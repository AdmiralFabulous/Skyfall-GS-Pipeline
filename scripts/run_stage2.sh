#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate skyfall-gs
cd ~/Skyfall-GS

export TORCH_CUDA_ARCH_LIST="9.0"
export USE_NINJA=0
export MAX_JOBS=4

echo "=== Starting Stage 2 IDU Training ==="

python train.py \
    -s ./data/datasets_JAX/JAX_068/ \
    -m ./outputs/JAX/JAX_068_stage2_idu \
    --start_checkpoint ./outputs/JAX/JAX_068_v2/chkpnt25000.pth \
    --iterative_datasets_update \
    --eval \
    --kernel_size 0.1 \
    --resolution 1 \
    --sh_degree 1 \
    --appearance_enabled \
    --lambda_depth 0 \
    --lambda_opacity 0 \
    --idu_opacity_reset_interval 5000 \
    --idu_refine \
    --idu_num_samples_per_view 2 \
    --densify_grad_threshold 0.0002 \
    --idu_num_cams 6 \
    --idu_use_flow_edit \
    --idu_render_size 1024 \
    --idu_flow_edit_n_min 4 \
    --idu_flow_edit_n_max 10 \
    --idu_grid_size 3 \
    --idu_grid_width 512 \
    --idu_grid_height 512 \
    --idu_episode_iterations 10000 \
    --idu_iter_full_train 0 \
    --idu_opacity_cooling_iterations 500 \
    --lambda_pseudo_depth 0.5 \
    --idu_densify_until_iter 9000 \
    --idu_train_ratio 0.75
