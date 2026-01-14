#!/bin/bash
#
# 9-Task Mix Training Script
#
# This script runs mixed multi-task training on all 9 environments
# simultaneously (3 environments × 3 difficulty levels).
#
# Environments:
#   BanditLow, BanditMedium, BanditHard
#   SokobanLow, SokobanMedium, SokobanHard
#   FrozenLakeLow, FrozenLakeMedium, FrozenLakeHard
#
# Key features:
# - All 9 environments are mixed in each training batch
# - Single shared LoRA module for all environments
# - No sequential task ordering
#
# Checkpoint structure:
#   checkpoints/mix_9tasks/{timestamp}/global_step_{N}/
#
# Usage:
#   # Train with default settings
#   bash run_mix_9tasks.sh
#
#   # Custom training steps and validation frequency
#   TOTAL_STEPS=900 TEST_FREQ=20 bash run_mix_9tasks.sh
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

echo "=============================================================="
echo "RAGEN 9-Task Mix Training"
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

echo "Running: $CMD"
eval $CMD
