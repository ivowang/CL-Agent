#!/bin/bash
# Training script for 9-task L2P continual learning
#
# Usage:
#   bash runs/run_l2p.sh
#
# With custom parameters:
#   POOL_SIZE=10 PROMPT_LENGTH=10 TOP_K=4 EMBEDDING_KEY=mean STEPS_PER_TASK=100 bash runs/run_l2p.sh
#
# With custom GPU:
#   CUDA_VISIBLE_DEVICES=3 TASK_ORDER=036147258 bash runs/run_l2p.sh

set -e

# Ensure we use the local verl implementation (not a different installed copy).
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH="${SCRIPT_DIR}/verl:${PYTHONPATH}"

# Default parameters
POOL_SIZE=${POOL_SIZE:-36}
PROMPT_LENGTH=${PROMPT_LENGTH:-10}
TOP_K=${TOP_K:-4}
EMBEDDING_KEY=${EMBEDDING_KEY:-mean}
PROMPT_INIT=${PROMPT_INIT:-uniform}
PROMPT_KEY=${PROMPT_KEY:-true}
PROMPT_KEY_INIT=${PROMPT_KEY_INIT:-uniform}
USE_PROMPT_MASK=${USE_PROMPT_MASK:-true}
PULL_COEFF=${PULL_COEFF:-1.0}
STEPS_PER_TASK=${STEPS_PER_TASK:-100}
TASK_ORDER=${TASK_ORDER:-}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}
ROLLOUT_MICRO_BATCH_SIZE=${ROLLOUT_MICRO_BATCH_SIZE:-32}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}

# Build command
CMD="python train_scripts/train_continual.py --config-name continual_learning_9tasks_l2p"

# Override GPU settings if CUDA_VISIBLE_DEVICES is set
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
    NUM_GPUS=${#GPUS[@]}
    CMD="$CMD system.CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    CMD="$CMD trainer.n_gpus_per_node=$NUM_GPUS"
fi

# Add method parameters
CMD="$CMD continual_learning.method.pool_size=$POOL_SIZE"
CMD="$CMD continual_learning.method.prompt_length=$PROMPT_LENGTH"
CMD="$CMD continual_learning.method.top_k=$TOP_K"
CMD="$CMD continual_learning.method.embedding_key=$EMBEDDING_KEY"
CMD="$CMD continual_learning.method.prompt_init=$PROMPT_INIT"
CMD="$CMD continual_learning.method.prompt_key=$PROMPT_KEY"
CMD="$CMD continual_learning.method.prompt_key_init=$PROMPT_KEY_INIT"
CMD="$CMD continual_learning.method.use_prompt_mask=$USE_PROMPT_MASK"
CMD="$CMD continual_learning.method.pull_constraint_coeff=$PULL_COEFF"
CMD="$CMD actor_rollout_ref.rollout.micro_batch_size=$ROLLOUT_MICRO_BATCH_SIZE"
CMD="$CMD micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU"
CMD="$CMD actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE"

# Add training parameters
CMD="$CMD continual_learning.steps_per_task=$STEPS_PER_TASK"

# Add task order if specified
if [ -n "$TASK_ORDER" ]; then
    CMD="$CMD continual_learning.task_order=$TASK_ORDER"
fi

# Add resume checkpoint if specified
if [ -n "$RESUME_CHECKPOINT" ]; then
    CMD="$CMD continual_learning.resume_checkpoint=$RESUME_CHECKPOINT"
fi

# Print configuration
echo "=========================================="
echo "L2P 9-Task Continual Learning"
echo "=========================================="
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES (${NUM_GPUS} GPU(s))"
fi
echo "Pool Size: $POOL_SIZE"
echo "Prompt Length: $PROMPT_LENGTH"
echo "Top-K: $TOP_K"
echo "Embedding Key: $EMBEDDING_KEY"
echo "Prompt Init: $PROMPT_INIT"
echo "Prompt Key: $PROMPT_KEY"
echo "Prompt Key Init: $PROMPT_KEY_INIT"
echo "Use Prompt Mask: $USE_PROMPT_MASK"
echo "Pull Constraint Coeff: $PULL_COEFF"
echo "Rollout Micro Batch Size: $ROLLOUT_MICRO_BATCH_SIZE"
echo "Micro Batch Size Per GPU: $MICRO_BATCH_SIZE_PER_GPU"
echo "PPO Mini Batch Size: $PPO_MINI_BATCH_SIZE"
echo "Steps per Task: $STEPS_PER_TASK"
if [ -n "$TASK_ORDER" ]; then
    echo "Task Order: $TASK_ORDER"
fi
if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "Resume from: $RESUME_CHECKPOINT"
fi
echo "=========================================="
echo ""

# Run training
echo "Running command:"
echo "$CMD"
echo ""

eval $CMD
