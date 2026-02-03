#!/bin/bash
#
# MemoryBank Launcher
#
# Launches MemoryBank training on specified environments using specified GPUs.
# Each GPU runs MemoryBank on one environment in its corresponding tmux session.
#
# Usage:
#   bash launch_memorybank.sh GPU1 ENV1 [GPU2 ENV2 ...]
#
# Example:
#   BUFFER_SIZE=30 TOP_K=6 bash launch_memorybank.sh 0 BanditLow 1 SokobanLow
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BUFFER_SIZE="${BUFFER_SIZE:-}"
TOP_K="${TOP_K:-}"
DECAY_TAU="${DECAY_TAU:-}"

VALID_ENVS="BanditLow BanditMedium BanditHard SokobanLow SokobanMedium SokobanHard FrozenLakeLow FrozenLakeMedium FrozenLakeHard"

if [ $# -lt 2 ] || [ $((# % 2)) -ne 0 ]; then
    echo "Usage: bash launch_memorybank.sh GPU1 ENV1 [GPU2 ENV2 ...]"
    echo "Valid environments: $VALID_ENVS"
    exit 1
fi

echo "=============================================================="
echo "Launching MemoryBank Training"
echo "=============================================================="
if [ -n "$BUFFER_SIZE" ]; then echo "Buffer size: $BUFFER_SIZE"; fi
if [ -n "$TOP_K" ]; then echo "Top-K: $TOP_K"; fi
if [ -n "$DECAY_TAU" ]; then echo "Decay tau: $DECAY_TAU"; fi
echo ""

EXTRA_ARGS=""
if [ -n "$BUFFER_SIZE" ]; then EXTRA_ARGS="$EXTRA_ARGS BUFFER_SIZE=$BUFFER_SIZE"; fi
if [ -n "$TOP_K" ]; then EXTRA_ARGS="$EXTRA_ARGS TOP_K=$TOP_K"; fi
if [ -n "$DECAY_TAU" ]; then EXTRA_ARGS="$EXTRA_ARGS DECAY_TAU=$DECAY_TAU"; fi

LAUNCHED=0
while [ $# -ge 2 ]; do
    GPU="$1"
    ENV_TAG="$2"
    shift 2

    SESSION_NAME="clagent-mb-$GPU"

    VALID=0
    for env in $VALID_ENVS; do
        if [ "$ENV_TAG" = "$env" ]; then
            VALID=1
            break
        fi
    done

    if [ $VALID -eq 0 ]; then
        echo "ERROR: Invalid environment: $ENV_TAG"
        echo "Valid environments: $VALID_ENVS"
        exit 1
    fi

    if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "Creating tmux session $SESSION_NAME..."
        tmux new-session -d -s "$SESSION_NAME" -c "$SCRIPT_DIR"
        tmux send-keys -t "$SESSION_NAME" 'eval "$(micromamba shell hook --shell bash)" && micromamba activate ragen' Enter
        sleep 1
    fi

    echo "Launching MemoryBank on $ENV_TAG using GPU $GPU (session: $SESSION_NAME)"
    CMD="cd $SCRIPT_DIR && CUDA_VISIBLE_DEVICES=$GPU ENV_TAG=$ENV_TAG$EXTRA_ARGS bash run_memorybank_9tasks.sh"
    tmux send-keys -t "$SESSION_NAME" "$CMD" Enter
    LAUNCHED=$((LAUNCHED + 1))
done

echo ""
echo "=============================================================="
echo "$LAUNCHED MemoryBank training(s) launched!"
echo "=============================================================="
