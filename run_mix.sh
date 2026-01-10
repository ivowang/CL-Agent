#!/bin/bash
#
# Mix Training Script for RAGEN
#
# This script runs multi-task interleaved training where the agent
# trains on multiple environments in a round-robin fashion. After each
# training step, the environment is switched to the next one in the cycle.
#
# Key differences from continual learning:
# - All environments are trained simultaneously (interleaved)
# - Single shared LoRA module for all environments
# - Environment cycle: env1 → env2 → env3 → env1 → ...
#
# Checkpoint structure:
#   checkpoints/mix/{timestamp}/global_step_{N}/
#
# Usage:
#   # Train with default settings (uses GPU 0,1,2,3)
#   bash run_mix.sh
#
#   # Train with specific GPUs
#   CUDA_VISIBLE_DEVICES=4,5,6,7 bash run_mix.sh
#
#   # Customize training steps
#   TOTAL_STEPS=500 bash run_mix.sh
#
#   # Combined options
#   CUDA_VISIBLE_DEVICES=0,1 TOTAL_STEPS=200 TEST_FREQ=5 bash run_mix.sh

set -e

# Configuration
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export WANDB_MODE="${WANDB_MODE:-online}"

# Optional: Enable offline mode for HuggingFace
# export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# Training parameters
TOTAL_STEPS="${TOTAL_STEPS:-300}"
TEST_FREQ="${TEST_FREQ:-10}"
SAVE_FREQ="${SAVE_FREQ:-100}"

echo "=============================================="
echo "RAGEN Mix Training"
echo "=============================================="
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "WANDB_MODE: ${WANDB_MODE}"
echo "Training Parameters:"
echo "  - total_steps: ${TOTAL_STEPS}"
echo "  - test_freq: ${TEST_FREQ}"
echo "  - save_freq: ${SAVE_FREQ}"
echo ""
echo "Environments (interleaved):"
echo "  - Bandit"
echo "  - Sokoban (CoordSokoban)"
echo "  - Frozen Lake (CoordFrozenLake)"
echo "=============================================="

# Build command
CMD="python train_mix.py --config-name mix_training"

# Add system config
CMD="$CMD \"system.CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}'\""

# Add training parameters
CMD="$CMD trainer.total_training_steps=${TOTAL_STEPS}"
CMD="$CMD trainer.test_freq=${TEST_FREQ}"
CMD="$CMD trainer.save_freq=${SAVE_FREQ}"

echo "Running: $CMD"
echo ""

eval $CMD
