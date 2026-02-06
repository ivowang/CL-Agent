#!/bin/bash
#
# 9-Task Continual Learning Training Script - Baseline Method
#
# This script runs continual learning with the baseline (naive) method
# on 9 tasks (3 environments × 3 difficulty levels).
#
# Task Index Mapping:
#   0=BanditLow, 1=BanditMedium, 2=BanditHard
#   3=SokobanLow, 4=SokobanMedium, 5=SokobanHard
#   6=FrozenLakeLow, 7=FrozenLakeMedium, 8=FrozenLakeHard
#
# Predefined Task Orderings:
#   Order 1: "012345678" - Env-first, LMH: BL→BM→BH→SL→SM→SH→FL→FM→FH
#   Order 2: "210543876" - Env-first, HML: BH→BM→BL→SH→SM→SL→FH→FM→FL
#   Order 3: "036147258" - Difficulty-first, LMH: BL→SL→FL→BM→SM→FM→BH→SH→FH
#   Order 4: "258147036" - Difficulty-first, HML: BH→SH→FH→BM→SM→FM→BL→SL→FL
#
# Checkpoint structure:
#   checkpoints/baseline_9tasks/{timestamp}/global_step_{N}/
#
# Usage:
#   # Train from scratch (default order: 012345678)
#   bash runs/run_baseline.sh
#
#   # Train with predefined orderings
#   TASK_ORDER=012345678 bash runs/run_baseline.sh  # Order 1: Env-first, LMH
#   TASK_ORDER=210543876 bash runs/run_baseline.sh  # Order 2: Env-first, HML
#   TASK_ORDER=036147258 bash runs/run_baseline.sh  # Order 3: Diff-first, LMH
#   TASK_ORDER=258147036 bash runs/run_baseline.sh  # Order 4: Diff-first, HML
#
#   # Resume from checkpoint
#   RESUME_CHECKPOINT=/path/to/checkpoint bash runs/run_baseline.sh
#
#   # Use specific GPUs
#   CUDA_VISIBLE_DEVICES=0 TASK_ORDER=012345678 bash runs/run_baseline.sh

set -e

# Configuration
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export WANDB_MODE="${WANDB_MODE:-online}"

# Optional: Resume from checkpoint
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

# Optional: Task order (9-digit string)
TASK_ORDER="${TASK_ORDER:-}"

# Validate task order length if specified
if [ -n "$TASK_ORDER" ]; then
    if [ ${#TASK_ORDER} -ne 9 ]; then
        echo "ERROR: TASK_ORDER must be a 9-digit string (got ${#TASK_ORDER} characters: '$TASK_ORDER')"
        echo "Example valid orderings:"
        echo "  012345678 - Env-first, Low→Medium→Hard"
        echo "  210543876 - Env-first, Hard→Medium→Low"
        echo "  036147258 - Difficulty-first, Low→Medium→Hard"
        echo "  258147036 - Difficulty-first, Hard→Medium→Low"
        exit 1
    fi
fi

echo "=============================================================="
echo "RAGEN 9-Task Continual Learning - Baseline Method"
echo "=============================================================="
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "WANDB_MODE: ${WANDB_MODE}"
if [ -n "$TASK_ORDER" ]; then
    echo "Task order: ${TASK_ORDER}"
else
    echo "Task order: default (012345678 = BL→BM→BH→SL→SM→SH→FL→FM→FH)"
fi
if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "Resume checkpoint: ${RESUME_CHECKPOINT}"
fi
echo "=============================================================="

# Build command
CMD="python train_scripts/train_continual.py --config-name continual_learning_9tasks"

# Add system config - use escaped quotes for Hydra string values
CMD="$CMD \"system.CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}'\""

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
