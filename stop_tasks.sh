#!/bin/bash
#
# Stop CL-Agent Training Tasks
#
# Stops training tasks running in specified tmux sessions.
# Works with all training methods (Baseline, O-LoRA, SD-LoRA, ER, etc.)
# Sends Ctrl-C to interrupt the running processes.
#
# Usage:
#   bash stop_tasks.sh GPU1 [GPU2 GPU3 ...]
#   bash stop_tasks.sh all              # Stop all 4 standard sessions (GPUs 0-3)
#   bash stop_tasks.sh 0 1              # Stop GPUs 0 and 1
#   bash stop_tasks.sh --kill-session GPU1 [GPU2 ...]  # Also kill tmux sessions
#
# Examples:
#   bash stop_tasks.sh 0 1 2 3          # Stop training on GPUs 0,1,2,3
#   bash stop_tasks.sh all              # Stop all 4 standard sessions
#   bash stop_tasks.sh --kill-session 0 1  # Stop and kill sessions for GPUs 0,1
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default: don't kill the tmux session, just interrupt the process
KILL_SESSION=false

# Parse options
GPUS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --kill-session|-k)
            KILL_SESSION=true
            shift
            ;;
        all)
            GPUS=(0 1 2 3)
            shift
            ;;
        *)
            GPUS+=("$1")
            shift
            ;;
    esac
done

# Check if any GPUs specified
if [ ${#GPUS[@]} -eq 0 ]; then
    echo "Usage: bash stop_tasks.sh [--kill-session] GPU1 [GPU2 GPU3 ...]"
    echo "       bash stop_tasks.sh all"
    echo ""
    echo "Options:"
    echo "  --kill-session, -k    Also kill the tmux session after stopping the task"
    echo "  all                   Stop all 4 standard sessions (GPUs 0-3)"
    echo ""
    echo "Examples:"
    echo "  bash stop_tasks.sh 0 1 2 3"
    echo "  bash stop_tasks.sh all"
    echo "  bash stop_tasks.sh --kill-session 0 1"
    exit 1
fi

echo "=============================================================="
echo "Stopping CL-Agent Training Tasks"
echo "=============================================================="
echo "Target GPUs: ${GPUS[*]}"
echo "Kill sessions: $KILL_SESSION"
echo ""

STOPPED_COUNT=0
FAILED_COUNT=0

for gpu in "${GPUS[@]}"; do
    SESSION_NAME="clagent-runs-$gpu"
    
    # Check if session exists
    if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "⚠ Session $SESSION_NAME does not exist (GPU $gpu)"
        ((FAILED_COUNT++))
        continue
    fi
    
    echo "Stopping task in session: $SESSION_NAME (GPU $gpu)"
    
    # Send Ctrl-C to interrupt the running process
    tmux send-keys -t "$SESSION_NAME" C-c
    
    # Wait a moment for graceful shutdown
    sleep 0.5
    
    # If requested, kill the entire tmux session
    if [ "$KILL_SESSION" = true ]; then
        echo "  Killing tmux session: $SESSION_NAME"
        tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
    fi
    
    ((STOPPED_COUNT++))
done

echo ""
echo "=============================================================="
echo "Summary"
echo "=============================================================="
echo "✓ Stopped: $STOPPED_COUNT tasks"
if [ $FAILED_COUNT -gt 0 ]; then
    echo "✗ Failed: $FAILED_COUNT tasks"
fi

if [ "$KILL_SESSION" = false ] && [ $STOPPED_COUNT -gt 0 ]; then
    echo ""
    echo "Note: tmux sessions are still running. Attach with:"
    for gpu in "${GPUS[@]}"; do
        SESSION_NAME="clagent-runs-$gpu"
        if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
            echo "  tmux attach -t $SESSION_NAME"
        fi
    done
    echo ""
    echo "To also kill sessions, use: bash stop_tasks.sh --kill-session ${GPUS[*]}"
fi

echo ""
