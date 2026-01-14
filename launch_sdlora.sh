#!/bin/bash
#
# SD-LoRA Continual Learning Launcher
#
# Launches SD-LoRA CL training on all 4 task orders using specified GPUs.
# Each GPU runs one task order in its corresponding tmux session.
#
# Task Orders:
#   Order 1: 012345678 - Env-first, LMH
#   Order 2: 210543876 - Env-first, HML
#   Order 3: 036147258 - Difficulty-first, LMH
#   Order 4: 258147036 - Difficulty-first, HML
#
# Usage:
#   bash launch_sdlora.sh GPU1 GPU2 GPU3 GPU4
#
# Example:
#   bash launch_sdlora.sh 0 1 2 3    # Use GPUs 0,1,2,3
#   bash launch_sdlora.sh 4 5 6 7    # Use GPUs 4,5,6,7
#
# With custom hyperparameters:
#   LAMBDA_ORTHO=0.3 LAMBDA_L2=0.01 bash launch_sdlora.sh 0 1 2 3
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Task orders
TASK_ORDERS=("012345678" "210543876" "036147258" "258147036")
ORDER_NAMES=("env_lmh" "env_hml" "diff_lmh" "diff_hml")

# Optional hyperparameters passed through environment
LAMBDA_ORTHO="${LAMBDA_ORTHO:-}"
LAMBDA_L2="${LAMBDA_L2:-}"
LORA_RANK="${LORA_RANK:-}"

# Check arguments
if [ $# -ne 4 ]; then
    echo "Usage: bash launch_sdlora.sh GPU1 GPU2 GPU3 GPU4"
    echo "Example: bash launch_sdlora.sh 0 1 2 3"
    exit 1
fi

GPUS=("$1" "$2" "$3" "$4")

echo "=============================================================="
echo "Launching SD-LoRA CL Training on 4 Task Orders"
echo "=============================================================="
echo "GPUs: ${GPUS[*]}"
if [ -n "$LAMBDA_ORTHO" ]; then echo "Lambda ortho: $LAMBDA_ORTHO"; fi
if [ -n "$LAMBDA_L2" ]; then echo "Lambda L2: $LAMBDA_L2"; fi
if [ -n "$LORA_RANK" ]; then echo "LoRA rank: $LORA_RANK"; fi
echo ""

# Ensure tmux sessions exist
for gpu in "${GPUS[@]}"; do
    SESSION_NAME="clagent-runs-$gpu"
    if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "Creating tmux session $SESSION_NAME..."
        tmux new-session -d -s "$SESSION_NAME" -c "$SCRIPT_DIR"
        tmux send-keys -t "$SESSION_NAME" 'eval "$(micromamba shell hook --shell bash)" && micromamba activate ragen' Enter
        sleep 1
    fi
done

# Build extra args for hyperparameters
EXTRA_ARGS=""
if [ -n "$LAMBDA_ORTHO" ]; then EXTRA_ARGS="$EXTRA_ARGS LAMBDA_ORTHO=$LAMBDA_ORTHO"; fi
if [ -n "$LAMBDA_L2" ]; then EXTRA_ARGS="$EXTRA_ARGS LAMBDA_L2=$LAMBDA_L2"; fi
if [ -n "$LORA_RANK" ]; then EXTRA_ARGS="$EXTRA_ARGS LORA_RANK=$LORA_RANK"; fi

# Launch training for each task order
for i in {0..3}; do
    GPU="${GPUS[$i]}"
    TASK_ORDER="${TASK_ORDERS[$i]}"
    ORDER_NAME="${ORDER_NAMES[$i]}"
    SESSION_NAME="clagent-runs-$GPU"

    echo "Launching Order $((i+1)) ($ORDER_NAME): $TASK_ORDER on GPU $GPU (session: $SESSION_NAME)"

    # Send command to tmux session
    CMD="cd $SCRIPT_DIR && CUDA_VISIBLE_DEVICES=$GPU TASK_ORDER=$TASK_ORDER$EXTRA_ARGS bash run_continual_9tasks_sdlora.sh"
    tmux send-keys -t "$SESSION_NAME" "$CMD" Enter
done

echo ""
echo "=============================================================="
echo "All 4 SD-LoRA trainings launched!"
echo "=============================================================="
echo ""
echo "Monitor sessions with:"
for i in {0..3}; do
    GPU="${GPUS[$i]}"
    ORDER_NAME="${ORDER_NAMES[$i]}"
    echo "  tmux attach -t clagent-runs-$GPU  # Order $((i+1)) ($ORDER_NAME)"
done
