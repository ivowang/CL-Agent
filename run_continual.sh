#!/bin/bash
#
# Continual Learning Training Script - Baseline Method
#
# This script runs continual learning with the baseline (naive) method,
# which uses a shared LoRA module across all tasks.
#
# Checkpoint structure:
#   checkpoints/baseline/{timestamp}/global_step_{N}/
#
# Usage:
#   # Train from scratch
#   bash run_continual.sh
#
#   # Resume from checkpoint
#   RESUME_CHECKPOINT=/path/to/checkpoint bash run_continual.sh
#
#   # Use specific GPUs
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_continual.sh

set -e

# Initialize micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate ragen

# Configuration
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export WANDB_MODE="${WANDB_MODE:-online}"

# Optional: Resume from checkpoint
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

echo "=============================================="
echo "RAGEN Continual Learning - Baseline Method"
echo "=============================================="
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "WANDB_MODE: ${WANDB_MODE}"
if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "Resume checkpoint: ${RESUME_CHECKPOINT}"
fi
echo "=============================================="

# Build command
CMD="python train_continual.py --config-name continual_learning"

# Add system config - use escaped quotes for Hydra string values
CMD="$CMD \"system.CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}'\""

# Add resume checkpoint if specified
if [ -n "$RESUME_CHECKPOINT" ]; then
    CMD="$CMD \"continual_learning.resume_checkpoint='${RESUME_CHECKPOINT}'\""
fi

echo "Running: $CMD"
eval $CMD
