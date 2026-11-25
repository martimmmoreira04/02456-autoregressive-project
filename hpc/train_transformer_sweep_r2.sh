#!/bin/sh
#BSUB -q gpuv100
#BSUB -J transformer_sweep_r2
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 12:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -o hpc/logs/transformer_sweep_r2_%J.out
#BSUB -e hpc/logs/transformer_sweep_r2_%J.err

# Round 2 of the transformer sweep (AdamW + Adam, dropout, bigger models, 30 epochs)
# Submit from the repo root:   bsub < hpc/train_transformer_sweep_r2.sh

source hpc/env.sh   # uv on the PATH, caches off the home quota

nvidia-smi --query-gpu=name,memory.total --format=csv   # logs which GPU the job got

PYTHONPATH=src uv run python scripts/transformer/transformer_train.py \
  --config-name=transformer_train_sweep_r2 -m hardware=cuda
