#!/bin/bash
# Training script for 9-task HiDE-Prompt continual learning
#
# This script trains an agent sequentially on 9 tasks (3 environments × 3 difficulty levels)
# using HiDE-Prompt (Hierarchical Decomposition of Prompt-Based Continual Learning).
#
# Usage:
#   bash run_continual_9tasks_hideprompt.sh
#
# With custom parameters:
#   REG_WEIGHT=0.1 PROMPT_LENGTH=5 TASK_ORDER=012345678 bash run_continual_9tasks_hideprompt.sh
#
# With custom GPU:
#   CUDA_VISIBLE_DEVICES=3 TASK_ORDER=036147258 bash run_continual_9tasks_hideprompt.sh
#
# Environment variables:
#   CUDA_VISIBLE_DEVICES: GPU to use (will be passed to config)
#   TASK_ORDER: Task order (9-digit string, e.g., "012345678")
#   REG_WEIGHT: Weight for contrastive regularization loss (default: 0.1)
#   PROMPT_LENGTH: Length of each task-specific prompt (default: 5)
#   PROMPT_MOMENTUM: Momentum for prompt averaging (default: 0.01)
#   TEMPERATURE: Temperature for contrastive loss (default: 0.8)
#   STORAGE_METHOD: Feature storage method (default: multi-centroid)
#   N_CENTROIDS: Number of centroids for multi-centroid method (default: 10)
#   STEPS_PER_TASK: Training steps per task (default: 100)
#   RESUME_CHECKPOINT: Path to checkpoint to resume from (optional)

set -e

# Default parameters
REG_WEIGHT=${REG_WEIGHT:-0.1}
PROMPT_LENGTH=${PROMPT_LENGTH:-5}
PROMPT_MOMENTUM=${PROMPT_MOMENTUM:-0.01}
TEMPERATURE=${TEMPERATURE:-0.8}
STORAGE_METHOD=${STORAGE_METHOD:-multi-centroid}
N_CENTROIDS=${N_CENTROIDS:-10}
STEPS_PER_TASK=${STEPS_PER_TASK:-100}
TASK_ORDER=${TASK_ORDER:-}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}

# Build command
CMD="python train_continual.py --config-name continual_learning_9tasks_hideprompt"

# Override GPU settings if CUDA_VISIBLE_DEVICES is set
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    # Count number of GPUs
    IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
    NUM_GPUS=${#GPUS[@]}
    CMD="$CMD system.CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    CMD="$CMD trainer.n_gpus_per_node=$NUM_GPUS"
fi

# Add method parameters
CMD="$CMD continual_learning.method.reg_weight=$REG_WEIGHT"
CMD="$CMD continual_learning.method.prompt_length=$PROMPT_LENGTH"
CMD="$CMD continual_learning.method.prompt_momentum=$PROMPT_MOMENTUM"
CMD="$CMD continual_learning.method.temperature=$TEMPERATURE"
CMD="$CMD continual_learning.method.storage_method=$STORAGE_METHOD"
CMD="$CMD continual_learning.method.n_centroids=$N_CENTROIDS"

# Add training parameters
CMD="$CMD continual_learning.steps_per_task=$STEPS_PER_TASK"

# Add task order if specified
if [ -n "$TASK_ORDER" ]; then
    CMD="$CMD continual_learning.task_order=$TASK_ORDER"
fi

# Add resume checkpoint if specified
if [ -n "$RESUME_CHECKPOINT" ]; then
    CMD="$CMD continual_learning.resume_checkpoint=$RESUME_CHECKPOINT"
fi

# Print configuration
echo "=========================================="
echo "HiDE-Prompt 9-Task Continual Learning"
echo "=========================================="
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES (${NUM_GPUS} GPU(s))"
fi
echo "Contrastive Regularization Weight: $REG_WEIGHT"
echo "Prompt Length: $PROMPT_LENGTH"
echo "Prompt Momentum: $PROMPT_MOMENTUM"
echo "Temperature: $TEMPERATURE"
echo "Storage Method: $STORAGE_METHOD"
echo "Number of Centroids: $N_CENTROIDS"
echo "Steps per Task: $STEPS_PER_TASK"
if [ -n "$TASK_ORDER" ]; then
    echo "Task Order: $TASK_ORDER"
fi
if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "Resume from: $RESUME_CHECKPOINT"
fi
echo "=========================================="
echo ""

# Run training
echo "Running command:"
echo "$CMD"
echo ""

eval $CMD
