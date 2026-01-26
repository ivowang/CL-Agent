#!/bin/bash
# Training script for 9-task MoE (Mixture of Experts)
#
# This script trains an agent using MoE method where each task has its own
# independent LoRA module. Training uses mixed sampling but maintains complete
# task independence.
#
# Usage:
#   bash run_moe_9tasks.sh
#
# With custom parameters:
#   TOTAL_STEPS=900 TEST_FREQ=20 bash run_moe_9tasks.sh
#
# With custom GPU:
#   CUDA_VISIBLE_DEVICES=0 bash run_moe_9tasks.sh
#
# Environment variables:
#   CUDA_VISIBLE_DEVICES: GPU to use (will be passed to config)
#   TOTAL_STEPS: Total training steps (default: 900)
#   TEST_FREQ: Validation frequency (default: 20)
#   SAVE_FREQ: Checkpoint save frequency (default: 100)

set -e

# Default parameters
TOTAL_STEPS=${TOTAL_STEPS:-900}
TEST_FREQ=${TEST_FREQ:-20}
SAVE_FREQ=${SAVE_FREQ:-100}

# Build command
CMD="python train_moe.py --config-name moe_training_9tasks"

# Override GPU settings if CUDA_VISIBLE_DEVICES is set
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    # Count number of GPUs
    IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
    NUM_GPUS=${#GPUS[@]}
    CMD="$CMD system.CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    CMD="$CMD trainer.n_gpus_per_node=$NUM_GPUS"
fi

# Add training parameters
CMD="$CMD trainer.total_training_steps=$TOTAL_STEPS"
CMD="$CMD trainer.test_freq=$TEST_FREQ"
CMD="$CMD trainer.save_freq=$SAVE_FREQ"

# Print configuration
echo "=========================================="
echo "MoE 9-Task Training"
echo "=========================================="
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES (${NUM_GPUS} GPU(s))"
fi
echo "Total Steps: $TOTAL_STEPS"
echo "Test Frequency: $TEST_FREQ"
echo "Save Frequency: $SAVE_FREQ"
echo "=========================================="
echo ""
echo "Training Details:"
echo "- 9 independent LoRA modules (one per task)"
echo "- 9 independent actor-critic pairs"
echo "- Serial sampling from all tasks"
echo "- Serial updates for all tasks"
echo "=========================================="
echo ""

# Run training
echo "Running command:"
echo "$CMD"
echo ""

eval $CMD
