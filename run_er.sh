#!/bin/bash
#
# Experience Replay Training Script
#
# This script runs in-context learning through experience replay.
# It collects successful trajectories and uses them as examples in prompts.
#
# Output:
#   results/experience_replay/replay_buffer_{env_tag}.pkl
#
# Usage:
#   # Train on Bandit (default)
#   bash run_er.sh
#
#   # Train on specific environment
#   ENV_TAG=Bandit bash run_er.sh
#   ENV_TAG=CoordSokoban bash run_er.sh
#   ENV_TAG=CoordFrozenLake bash run_er.sh
#
#   # With custom buffer settings
#   BUFFER_SIZE=30 VAL_FREQ=10 bash run_er.sh
#
#   # Use specific GPUs
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_er.sh

set -e

# Configuration
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-online}"

# Environment to train on
# Options: Bandit, CoordSokoban, CoordFrozenLake
ENV_TAG="${ENV_TAG:-Bandit}"

# Experience Replay settings
BUFFER_SIZE="${BUFFER_SIZE:-20}"      # N: Max experiences to collect
VAL_FREQ="${VAL_FREQ:-1}"             # M: Validate every M experiences
MAX_EXAMPLES="${MAX_EXAMPLES:-10}"      # Examples to include in prompt
EXAMPLE_SELECTION="${EXAMPLE_SELECTION:-random}"  # random, recent, best

echo "=============================================="
echo "RAGEN Experience Replay Training"
echo "=============================================="
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "WANDB_MODE: ${WANDB_MODE}"
echo "Environment: ${ENV_TAG}"
echo "ER Parameters:"
echo "  - buffer_size (N): ${BUFFER_SIZE}"
echo "  - val_frequency (M): ${VAL_FREQ}"
echo "  - max_examples: ${MAX_EXAMPLES}"
echo "  - example_selection: ${EXAMPLE_SELECTION}"
echo "=============================================="

# Build command
CMD="python train_er.py --config-name experience_replay"

# Add system config
CMD="$CMD \"system.CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}'\""

# Add ER parameters
CMD="$CMD \"experience_replay.env_tag='${ENV_TAG}'\""
CMD="$CMD experience_replay.buffer_size=${BUFFER_SIZE}"
CMD="$CMD experience_replay.val_frequency=${VAL_FREQ}"
CMD="$CMD experience_replay.max_examples_in_prompt=${MAX_EXAMPLES}"
CMD="$CMD \"experience_replay.example_selection='${EXAMPLE_SELECTION}'\""

echo "Running: $CMD"
eval $CMD
