#!/bin/bash
#
# Mix Training Launcher
#
# Launches mix training (multi-task learning on all 9 environments) on a specified GPU.
# Mix training trains on all environments simultaneously, so there's no task order.
#
# Usage:
#   bash launch_mix.sh GPU
#
# Example:
#   bash launch_mix.sh 0    # Use GPU 0
#   bash launch_mix.sh 4    # Use GPU 4
#
# With custom parameters:
#   TOTAL_STEPS=900 TEST_FREQ=20 bash launch_mix.sh 0
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Optional parameters passed through environment
TOTAL_STEPS="${TOTAL_STEPS:-}"
TEST_FREQ="${TEST_FREQ:-}"
SAVE_FREQ="${SAVE_FREQ:-}"

# Check arguments
if [ $# -ne 1 ]; then
    echo "Usage: bash launch_mix.sh GPU"
    echo "Example: bash launch_mix.sh 0"
    exit 1
fi

GPU="$1"
SESSION_NAME="clagent-runs-$GPU"

echo "=============================================================="
echo "Launching Mix Training (9 Environments)"
echo "=============================================================="
echo "GPU: $GPU"
echo "Session: $SESSION_NAME"
if [ -n "$TOTAL_STEPS" ]; then echo "Total steps: $TOTAL_STEPS"; fi
if [ -n "$TEST_FREQ" ]; then echo "Test frequency: $TEST_FREQ"; fi
if [ -n "$SAVE_FREQ" ]; then echo "Save frequency: $SAVE_FREQ"; fi
echo ""

# Ensure tmux session exists
if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Creating tmux session $SESSION_NAME..."
    tmux new-session -d -s "$SESSION_NAME" -c "$SCRIPT_DIR"
    tmux send-keys -t "$SESSION_NAME" 'eval "$(micromamba shell hook --shell bash)" && micromamba activate ragen' Enter
    sleep 1
fi

# Build extra args
EXTRA_ARGS=""
if [ -n "$TOTAL_STEPS" ]; then EXTRA_ARGS="$EXTRA_ARGS TOTAL_STEPS=$TOTAL_STEPS"; fi
if [ -n "$TEST_FREQ" ]; then EXTRA_ARGS="$EXTRA_ARGS TEST_FREQ=$TEST_FREQ"; fi
if [ -n "$SAVE_FREQ" ]; then EXTRA_ARGS="$EXTRA_ARGS SAVE_FREQ=$SAVE_FREQ"; fi

# Send command to tmux session
CMD="cd $SCRIPT_DIR && CUDA_VISIBLE_DEVICES=$GPU$EXTRA_ARGS bash run_mix_9tasks.sh"
echo "Launching: $CMD"
tmux send-keys -t "$SESSION_NAME" "$CMD" Enter

echo ""
echo "=============================================================="
echo "Mix training launched!"
echo "=============================================================="
echo ""
echo "Monitor with: tmux attach -t $SESSION_NAME"
