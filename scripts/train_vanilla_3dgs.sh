#!/bin/bash
# Vanilla 3DGS Training Script for street panoramas
source ~/miniconda3/etc/profile.d/conda.sh
conda activate skyfall-gs
cd ~/gaussian-splatting

DATASET=${1:-"./data/street_scene/"}
OUTPUT=${2:-"./output/street_scene"}

echo "=== Vanilla 3DGS Training ==="
echo "Dataset: $DATASET"
echo "Output: $OUTPUT"

python train.py -s "$DATASET" -m "$OUTPUT" --eval --iterations 30000

echo "=== Training Complete ==="
