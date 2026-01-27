#!/bin/bash
#
# Experience Replay Launcher
#
# Launches experience replay training on specified environments using specified GPUs.
# Each GPU runs ER on one environment in its corresponding tmux session.
#
# Environments:
#   BanditLow, BanditMedium, BanditHard
#   SokobanLow, SokobanMedium, SokobanHard
#   FrozenLakeLow, FrozenLakeMedium, FrozenLakeHard
#
# Usage:
#   bash launch_er.sh GPU1 ENV1 [GPU2 ENV2 ...]
#
# Examples:
#   # Run ER on all 9 environments using GPUs 0-8
#   bash launch_er.sh 0 BanditLow 1 BanditMedium 2 BanditHard 3 SokobanLow 4 SokobanMedium 5 SokobanHard 6 FrozenLakeLow 7 FrozenLakeMedium
#
#   # Run ER on 3 environments
#   bash launch_er.sh 0 BanditLow 1 SokobanLow 2 FrozenLakeLow
#
# With custom parameters:
#   BUFFER_SIZE=30 bash launch_er.sh 0 BanditLow
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Optional parameters passed through environment
BUFFER_SIZE="${BUFFER_SIZE:-}"

# Valid environments
VALID_ENVS="BanditLow BanditMedium BanditHard SokobanLow SokobanMedium SokobanHard FrozenLakeLow FrozenLakeMedium FrozenLakeHard"

# Check arguments (must be even number, at least 2)
if [ $# -lt 2 ] || [ $((# % 2)) -ne 0 ]; then
    echo "Usage: bash launch_er.sh GPU1 ENV1 [GPU2 ENV2 ...]"
    echo ""
    echo "Valid environments: $VALID_ENVS"
    echo ""
    echo "Examples:"
    echo "  bash launch_er.sh 0 BanditLow"
    echo "  bash launch_er.sh 0 BanditLow 1 SokobanLow 2 FrozenLakeLow"
    exit 1
fi

echo "=============================================================="
echo "Launching Experience Replay Training"
echo "=============================================================="
if [ -n "$BUFFER_SIZE" ]; then echo "Buffer size: $BUFFER_SIZE"; fi
echo ""

# Build extra args
EXTRA_ARGS=""
if [ -n "$BUFFER_SIZE" ]; then EXTRA_ARGS="$EXTRA_ARGS BUFFER_SIZE=$BUFFER_SIZE"; fi

# Process GPU-ENV pairs
LAUNCHED=0
while [ $# -ge 2 ]; do
    GPU="$1"
    ENV_TAG="$2"
    shift 2

    SESSION_NAME="clagent-runs-$GPU"

    # Validate environment
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

    # Ensure tmux session exists
    if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "Creating tmux session $SESSION_NAME..."
        tmux new-session -d -s "$SESSION_NAME" -c "$SCRIPT_DIR"
        tmux send-keys -t "$SESSION_NAME" 'eval "$(micromamba shell hook --shell bash)" && micromamba activate ragen' Enter
        sleep 1
    fi

    echo "Launching ER on $ENV_TAG using GPU $GPU (session: $SESSION_NAME)"

    # Send command to tmux session
    CMD="cd $SCRIPT_DIR && CUDA_VISIBLE_DEVICES=$GPU ENV_TAG=$ENV_TAG$EXTRA_ARGS bash run_er_9tasks.sh"
    tmux send-keys -t "$SESSION_NAME" "$CMD" Enter
    LAUNCHED=$((LAUNCHED + 1))
done

echo ""
echo "=============================================================="
echo "$LAUNCHED ER training(s) launched!"
echo "=============================================================="
