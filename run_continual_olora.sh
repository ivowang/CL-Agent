#!/bin/bash
#
# Continual Learning Training Script - O-LoRA Method
#
# This script runs continual learning with O-LoRA (Orthogonal Low-Rank Adaptation),
# which learns each task in an orthogonal subspace to minimize catastrophic forgetting.
#
# Checkpoint structure:
#   checkpoints/olora/{timestamp}/global_step_{N}/
#
# Usage:
#   # Train from scratch (default order: Bandit -> Sokoban -> Frozen Lake)
#   bash run_continual_olora.sh
#
#   # Train with custom task order
#   TASK_ORDER=102 bash run_continual_olora.sh  # Sokoban -> Bandit -> Frozen Lake
#   TASK_ORDER=210 bash run_continual_olora.sh  # Frozen Lake -> Sokoban -> Bandit
#
#   # Resume from checkpoint
#   RESUME_CHECKPOINT=/path/to/checkpoint bash run_continual_olora.sh
#
#   # Use specific GPUs
#   CUDA_VISIBLE_DEVICES=4,5,6,7 bash run_continual_olora.sh
#
#   # Adjust O-LoRA parameters
#   LAMBDA_ORTHO=0.3 LAMBDA_L2=0.01 bash run_continual_olora.sh

set -e

# Initialize micromamba
# eval "$(micromamba shell hook --shell bash)"
# micromamba activate ragen

# Configuration
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export WANDB_MODE="${WANDB_MODE:-online}"
# Offline mode for HuggingFace to avoid network issues when loading local models
# export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# O-LoRA parameters
LAMBDA_ORTHO="${LAMBDA_ORTHO:-0.5}"
LAMBDA_L2="${LAMBDA_L2:-0.0}"
LORA_RANK="${LORA_RANK:-64}"

# Optional: Resume from checkpoint
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

# Optional: Task order (e.g., "012", "102", "210")
# 0=Bandit, 1=Sokoban, 2=Frozen Lake
TASK_ORDER="${TASK_ORDER:-}"

echo "=============================================="
echo "RAGEN Continual Learning - O-LoRA Method"
echo "=============================================="
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "WANDB_MODE: ${WANDB_MODE}"
echo "O-LoRA Parameters:"
echo "  - lambda_ortho: ${LAMBDA_ORTHO}"
echo "  - lambda_l2: ${LAMBDA_L2}"
echo "  - lora_rank: ${LORA_RANK}"
if [ -n "$TASK_ORDER" ]; then
    echo "Task order: ${TASK_ORDER}"
else
    echo "Task order: default (012 = Bandit -> Sokoban -> Frozen Lake)"
fi
if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "Resume checkpoint: ${RESUME_CHECKPOINT}"
fi
echo "=============================================="

# Build command
CMD="python train_continual.py --config-name continual_learning_olora"

# Add system config - use escaped quotes for Hydra string values
CMD="$CMD \"system.CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}'\""

# Add O-LoRA parameters
CMD="$CMD continual_learning.method.lambda_ortho=${LAMBDA_ORTHO}"
CMD="$CMD continual_learning.method.lambda_l2=${LAMBDA_L2}"
CMD="$CMD continual_learning.method.lora_rank=${LORA_RANK}"

# Add task order if specified
if [ -n "$TASK_ORDER" ]; then
    CMD="$CMD \"continual_learning.task_order='${TASK_ORDER}'\""
fi

# Add resume checkpoint if specified
if [ -n "$RESUME_CHECKPOINT" ]; then
    CMD="$CMD \"continual_learning.resume_checkpoint='${RESUME_CHECKPOINT}'\""
fi

echo "Running: $CMD"
eval $CMD
