#!/bin/sh
#BSUB -q gpuv100
#BSUB -J transformer_best
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 2:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -o hpc/logs/transformer_best_%J.out
#BSUB -e hpc/logs/transformer_best_%J.err

# Submit from the repo root:   bsub < hpc/train_transformer_best.sh
source hpc/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv

PYTHONPATH=src uv run python scripts/transformer/transformer_train.py \
  --config-name=transformer_best hardware=cuda