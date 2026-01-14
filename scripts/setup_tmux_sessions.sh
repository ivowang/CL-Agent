#!/bin/bash
#
# Setup tmux sessions for CL-Agent training
#
# Creates 8 tmux sessions (clagent-runs-0 to clagent-runs-7) if they don't exist,
# each with the ragen micromamba environment activated.
#
# Usage:
#   bash scripts/setup_tmux_sessions.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Setting up tmux sessions for CL-Agent training..."
echo "Project directory: $PROJECT_DIR"

# Create sessions 0-7 if they don't exist
for i in {0..7}; do
    SESSION_NAME="clagent-runs-$i"

    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "Session $SESSION_NAME already exists"
    else
        echo "Creating session $SESSION_NAME..."
        tmux new-session -d -s "$SESSION_NAME" -c "$PROJECT_DIR"

        # Activate ragen environment in the session
        tmux send-keys -t "$SESSION_NAME" 'eval "$(micromamba shell hook --shell bash)" && micromamba activate ragen' Enter

        # Wait a bit for environment to activate
        sleep 1

        echo "Session $SESSION_NAME created and ragen environment activated"
    fi
done

echo ""
echo "All tmux sessions are ready!"
echo "Sessions: clagent-runs-0 to clagent-runs-7"
echo ""
echo "To attach to a session: tmux attach -t clagent-runs-X"
echo "To list all sessions: tmux list-sessions"
