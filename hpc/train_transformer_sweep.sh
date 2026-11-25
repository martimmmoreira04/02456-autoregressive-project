#!/bin/sh
#BSUB -q gpuv100
#BSUB -J transformer_sweep
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 8:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -o hpc/logs/transformer_sweep_%J.out
#BSUB -e hpc/logs/transformer_sweep_%J.err

# Submit from the repo root:   bsub < hpc/train_transformer_sweep.sh

source hpc/env.sh   # uv on the PATH, caches off the home quota

nvidia-smi --query-gpu=name,memory.total --format=csv   # logs which GPU the job got

PYTHONPATH=src uv run python scripts/transformer/transformer_train.py \
  --config-name=transformer_train_sweep -m hardware=cuda
