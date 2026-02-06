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
#   ENV_TAG=BanditLow bash runs/run_er.sh
#   ENV_TAG=BanditMedium bash runs/run_er.sh
#   ENV_TAG=BanditHard bash runs/run_er.sh
#   ENV_TAG=SokobanLow bash runs/run_er.sh
#   ENV_TAG=SokobanMedium bash runs/run_er.sh
#   ENV_TAG=SokobanHard bash runs/run_er.sh
#   ENV_TAG=FrozenLakeLow bash runs/run_er.sh
#   ENV_TAG=FrozenLakeMedium bash runs/run_er.sh
#   ENV_TAG=FrozenLakeHard bash runs/run_er.sh
#
#   # With custom buffer settings
#   ENV_TAG=SokobanMedium BUFFER_SIZE=30 bash runs/run_er.sh
#
#   # Use specific GPUs
#   CUDA_VISIBLE_DEVICES=0 ENV_TAG=BanditLow bash runs/run_er.sh

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
    echo "Usage: ENV_TAG=BanditLow bash runs/run_er.sh"
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
BUFFER_SIZE="${BUFFER_SIZE:-20}"      # N: Max successful rollouts to collect

echo "=============================================================="
echo "RAGEN 9-Task Experience Replay Training"
echo "=============================================================="
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "WANDB_MODE: ${WANDB_MODE}"
echo "Environment: ${ENV_TAG}"
echo "ER Parameters:"
echo "  - buffer_size (N): ${BUFFER_SIZE}"
echo "=============================================================="

# Build command
CMD="python train_scripts/train_er.py --config-name experience_replay"

# Add system config
CMD="$CMD \"system.CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}'\""

# Add ER parameters
CMD="$CMD \"experience_replay.env_tag='${ENV_TAG}'\""
CMD="$CMD experience_replay.buffer_size=${BUFFER_SIZE}"

echo "Running: $CMD"
eval $CMD
