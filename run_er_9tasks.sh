#!/bin/bash
#
# 9-Task Experience Replay Training Script
#
# This script runs in-context learning through experience replay
# on any of the 9 environments (3 environment types × 3 difficulty levels).
#
# Supported Environments:
#   BanditLow, BanditMedium, BanditHard
#   SokobanLow, SokobanMedium, SokobanHard
#   FrozenLakeLow, FrozenLakeMedium, FrozenLakeHard
#
# Output:
#   results/experience_replay/replay_buffer_{env_tag}.pkl
#
# Usage:
#   # Train on specific 9-task environment
#   ENV_TAG=BanditLow bash run_er_9tasks.sh
#   ENV_TAG=BanditMedium bash run_er_9tasks.sh
#   ENV_TAG=BanditHard bash run_er_9tasks.sh
#   ENV_TAG=SokobanLow bash run_er_9tasks.sh
#   ENV_TAG=SokobanMedium bash run_er_9tasks.sh
#   ENV_TAG=SokobanHard bash run_er_9tasks.sh
#   ENV_TAG=FrozenLakeLow bash run_er_9tasks.sh
#   ENV_TAG=FrozenLakeMedium bash run_er_9tasks.sh
#   ENV_TAG=FrozenLakeHard bash run_er_9tasks.sh
#
#   # With custom buffer settings
#   ENV_TAG=SokobanMedium BUFFER_SIZE=30 VAL_FREQ=10 bash run_er_9tasks.sh
#
#   # Use specific GPUs
#   CUDA_VISIBLE_DEVICES=0 ENV_TAG=BanditLow bash run_er_9tasks.sh

set -e

# Configuration
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-online}"

# Environment to train on (required)
ENV_TAG="${ENV_TAG:-}"

# Validate environment tag
VALID_ENVS="BanditLow BanditMedium BanditHard SokobanLow SokobanMedium SokobanHard FrozenLakeLow FrozenLakeMedium FrozenLakeHard"

if [ -z "$ENV_TAG" ]; then
    echo "ERROR: ENV_TAG is required"
    echo "Valid options: $VALID_ENVS"
    echo "Usage: ENV_TAG=BanditLow bash run_er_9tasks.sh"
    exit 1
fi

# Check if ENV_TAG is valid
VALID=0
for env in $VALID_ENVS; do
    if [ "$ENV_TAG" = "$env" ]; then
        VALID=1
        break
    fi
done

if [ $VALID -eq 0 ]; then
    echo "ERROR: Invalid ENV_TAG: $ENV_TAG"
    echo "Valid options: $VALID_ENVS"
    exit 1
fi

# Experience Replay settings
BUFFER_SIZE="${BUFFER_SIZE:-20}"      # N: Max experiences to collect
VAL_FREQ="${VAL_FREQ:-1}"             # M: Validate every M experiences
MAX_EXAMPLES="${MAX_EXAMPLES:-10}"    # Examples to include in prompt
EXAMPLE_SELECTION="${EXAMPLE_SELECTION:-random}"  # random, recent, best

echo "=============================================================="
echo "RAGEN 9-Task Experience Replay Training"
echo "=============================================================="
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "WANDB_MODE: ${WANDB_MODE}"
echo "Environment: ${ENV_TAG}"
echo "ER Parameters:"
echo "  - buffer_size (N): ${BUFFER_SIZE}"
echo "  - val_frequency (M): ${VAL_FREQ}"
echo "  - max_examples: ${MAX_EXAMPLES}"
echo "  - example_selection: ${EXAMPLE_SELECTION}"
echo "=============================================================="

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
