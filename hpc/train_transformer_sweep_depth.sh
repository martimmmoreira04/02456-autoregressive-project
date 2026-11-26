#!/bin/sh
#BSUB -q gpuv100
#BSUB -J transformer_depth
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 3:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -o hpc/logs/transformer_depth_%J.out
#BSUB -e hpc/logs/transformer_depth_%J.err

# Depth sweep L8/L10/L12, everything else fixed at the v2 optimum.
# Submit from the repo root:   bsub < hpc/train_transformer_sweep_depth.sh

source hpc/env.sh

nvidia-smi --query-gpu=name,memory.total --format=csv

PYTHONPATH=src uv run python scripts/transformer/transformer_train.py \
  --config-name=transformer_train_sweep_depth -m hardware=cuda
