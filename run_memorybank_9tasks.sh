#!/bin/bash
#
# 9-Task MemoryBank Training Script
#
# This script runs MemoryBank (retrieval-based in-context memory)
# on any of the 9 environments (3 environment types × 3 difficulty levels),
# using the same benchmark settings as Mix training.
#
# Supported Environments:
#   BanditLow, BanditMedium, BanditHard
#   SokobanLow, SokobanMedium, SokobanHard
#   FrozenLakeLow, FrozenLakeMedium, FrozenLakeHard
#
# Usage:
#   ENV_TAG=BanditLow bash run_memorybank_9tasks.sh
#   ENV_TAG=SokobanMedium BUFFER_SIZE=30 TOP_K=6 bash run_memorybank_9tasks.sh
#   CUDA_VISIBLE_DEVICES=0 ENV_TAG=FrozenLakeHard bash run_memorybank_9tasks.sh

set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-online}"

ENV_TAG="${ENV_TAG:-}"
BUFFER_SIZE="${BUFFER_SIZE:-20}"
TOP_K="${TOP_K:-4}"
DECAY_TAU="${DECAY_TAU:-5.0}"

VALID_ENVS="BanditLow BanditMedium BanditHard SokobanLow SokobanMedium SokobanHard FrozenLakeLow FrozenLakeMedium FrozenLakeHard"

if [ -z "$ENV_TAG" ]; then
    echo "ERROR: ENV_TAG is required"
    echo "Valid options: $VALID_ENVS"
    exit 1
fi

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

echo "=============================================================="
echo "RAGEN 9-Task MemoryBank Training"
echo "=============================================================="
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "WANDB_MODE: ${WANDB_MODE}"
echo "Environment: ${ENV_TAG}"
echo "MemoryBank Parameters:"
echo "  - buffer_size (N): ${BUFFER_SIZE}"
echo "  - top_k: ${TOP_K}"
echo "  - decay_tau: ${DECAY_TAU}"
echo "=============================================================="

CMD="python train_memorybank.py --config-name memorybank"
CMD="$CMD \"system.CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}'\""
CMD="$CMD \"memorybank.env_tag='${ENV_TAG}'\""
CMD="$CMD memorybank.buffer_size=${BUFFER_SIZE}"
CMD="$CMD memorybank.top_k=${TOP_K}"
CMD="$CMD memorybank.decay_tau=${DECAY_TAU}"

echo "Running: $CMD"
eval $CMD
