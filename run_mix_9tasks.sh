#!/bin/bash
#
# Mix Training Script (9-task pool)
#
# This script runs mixed multi-task training on a configurable subset
# of the 9 environments (3 environments × 3 difficulty levels).
#
# Environments:
#   BanditLow, BanditMedium, BanditHard
#   SokobanLow, SokobanMedium, SokobanHard
#   FrozenLakeLow, FrozenLakeMedium, FrozenLakeHard
#
# Key features:
# - Selected environments are mixed in each training batch
# - Single shared LoRA module for all environments
# - No sequential task ordering
#
# Checkpoint structure:
#   checkpoints/mix_9tasks/{timestamp}/global_step_{N}/
#
# Usage:
#   # Train with default settings (BanditLow/BanditMedium/BanditHard)
#   bash run_mix_9tasks.sh
#
#   # Custom training steps and validation frequency
#   TOTAL_STEPS=900 TEST_FREQ=20 bash run_mix_9tasks.sh
#
#   # Select a custom task subset
#   MIX_TASKS=BanditLow,SokobanMedium,FrozenLakeHard bash run_mix_9tasks.sh
#
#   # Use specific GPUs
#   CUDA_VISIBLE_DEVICES=4 bash run_mix_9tasks.sh

set -e

# Configuration
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export WANDB_MODE="${WANDB_MODE:-online}"

# Optional parameters
TOTAL_STEPS="${TOTAL_STEPS:-}"
TEST_FREQ="${TEST_FREQ:-}"
SAVE_FREQ="${SAVE_FREQ:-}"
MIX_TASKS="${MIX_TASKS:-}"

echo "=============================================================="
echo "RAGEN Mix Training (Task Subset)"
echo "=============================================================="
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "WANDB_MODE: ${WANDB_MODE}"
if [ -n "$TOTAL_STEPS" ]; then
    echo "Total steps: ${TOTAL_STEPS}"
else
    echo "Total steps: 1800 (default)"
fi
if [ -n "$TEST_FREQ" ]; then
    echo "Test frequency: ${TEST_FREQ}"
fi
if [ -n "$SAVE_FREQ" ]; then
    echo "Save frequency: ${SAVE_FREQ}"
fi
if [ -n "$MIX_TASKS" ]; then
    echo "Task subset: ${MIX_TASKS}"
else
    echo "Task subset: BanditLow,BanditMedium,BanditHard (default)"
fi
echo "=============================================================="

# Build command
CMD="python train_mix.py --config-name mix_training_9tasks"

# Add system config
CMD="$CMD \"system.CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}'\""

# Add training parameters if specified
if [ -n "$TOTAL_STEPS" ]; then
    CMD="$CMD trainer.total_training_steps=${TOTAL_STEPS}"
fi

if [ -n "$TEST_FREQ" ]; then
    CMD="$CMD trainer.test_freq=${TEST_FREQ}"
fi

if [ -n "$SAVE_FREQ" ]; then
    CMD="$CMD trainer.save_freq=${SAVE_FREQ}"
fi

if [ -n "$MIX_TASKS" ]; then
    IFS=',' read -r -a TASK_ARRAY <<< "$MIX_TASKS"
    TASK_FILTERED=()
    for task in "${TASK_ARRAY[@]}"; do
        task="${task//[[:space:]]/}"
        if [ -n "$task" ]; then
            TASK_FILTERED+=("$task")
        fi
    done
    if [ "${#TASK_FILTERED[@]}" -eq 0 ]; then
        echo "Error: MIX_TASKS is set but empty"
        exit 1
    fi
    TASKS_JOINED=$(printf "%s," "${TASK_FILTERED[@]}")
    TASKS_JOINED="${TASKS_JOINED%,}"
    CMD="$CMD \"mix_training.task_subset=[$TASKS_JOINED]\""
fi

echo "Running: $CMD"
eval $CMD
