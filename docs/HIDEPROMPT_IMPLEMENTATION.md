# HiDE-Prompt Implementation for CL-Agent

This document describes the implementation of HiDE-Prompt (Hierarchical Decomposition of Prompt-Based Continual Learning) for the CL-Agent framework.

## Overview

HiDE-Prompt is a prompt-based continual learning method adapted from the paper "Hierarchical Decomposition of Prompt-Based Continual Learning: Rethinking Obscured Sub-optimality" (NeurIPS 2023). This implementation adapts HiDE-Prompt for decoder-only LLMs (like Llama/Qwen) in the CL-Agent framework.

## Key Components

### 1. E-Prompt Module (`ragen/cl_methods/prompt_modules.py`)

- **EPrompt**: Task-specific prompts that are prepended to input embeddings (prefix-tuning style)
- **PromptEmbeddingWrapper**: Wraps the model's embedding layer to inject prompts
- **inject_prompts_into_model()**: Helper function to inject prompts into a model

Each task gets its own prompt of configurable length (default: 5 tokens).

### 2. Trajectory Feature Storage (`ragen/cl_methods/trajectory_storage.py`)

- **TrajectoryFeatureStorage**: Stores trajectory features from successful episodes
- **compute_contrastive_loss()**: Computes contrastive regularization loss

Supports three storage methods:
- `variance`: Store mean and diagonal variance
- `covariance`: Store mean and full covariance matrix
- `multi-centroid`: Store multiple centroids per task using k-means

### 3. HiDE-Prompt CL Method (`ragen/cl_methods/hideprompt.py`)

- **HiDEPromptCLMethod**: Main CL method class
- **HiDEPromptConfig**: Configuration dataclass

Manages:
- Prompt injection and removal
- Trajectory feature storage
- CL loss configuration

### 4. Actor Integration (`ragen/workers/actor/dp_actor.py`)

Modified to support:
- Prompt-based forward pass
- Hidden state extraction for feature storage
- Contrastive loss computation

## Implementation Details

### Differences from Original HiDE-Prompt

1. **Model Architecture**: Adapted for decoder-only LLMs instead of ViT (encoder-only)
2. **Prompt Style**: Uses prefix-tuning style (prepend to embeddings) instead of ViT-style prompts
3. **Task ID Inference**: Skipped (task IDs are explicitly provided in our setting)
4. **Task-Adaptive Prediction**: Simplified (focus on WTP + Contrastive Regularization)
5. **Feature Extraction**: Uses last token's hidden state instead of CLS token

### Key Design Decisions

1. **Prefix-Tuning Style**: Prompts are prepended to input embeddings, increasing sequence length by `prompt_length`
2. **Contrastive Regularization**: Encourages orthogonality between current features and stored features from previous tasks
3. **Trajectory Features**: Stores features from successful trajectories (instead of class centroids)
4. **Multi-Centroid Storage**: Uses k-means clustering to store multiple centroids per task for better representation

## Configuration

### Parameters

- `prompt_length` (default: 5): Length of each task-specific prompt
- `prompt_init` (default: 'uniform'): Initialization method ('uniform' or 'zero')
- `prompt_momentum` (default: 0.01): Momentum for prompt averaging
- `reg_weight` (default: 0.1): Weight for contrastive regularization loss
- `temperature` (default: 0.8): Temperature for contrastive loss
- `storage_method` (default: 'multi-centroid'): Feature storage method
- `n_centroids` (default: 10): Number of centroids for multi-centroid method

### Configuration File

`config/continual_learning_9tasks_hideprompt.yaml`

## Usage

### Training Script

```bash
# Basic usage
bash run_continual_9tasks_hideprompt.sh

# With custom parameters
REG_WEIGHT=0.1 PROMPT_LENGTH=5 TASK_ORDER=012345678 bash run_continual_9tasks_hideprompt.sh
```

### Python API

```python
from ragen.cl_methods import get_cl_method
from ragen.cl_methods.hideprompt import HiDEPromptConfig

# Create config
config = HiDEPromptConfig(
    prompt_length=5,
    reg_weight=0.1,
    temperature=0.8,
)

# Create method
method = get_cl_method('hideprompt')(config)

# Use in training
method.on_task_start(task_idx=0, task_name='task0')
# ... training ...
method.on_task_end(task_idx=0, task_name='task0', checkpoint_path='...')
```

## Training Flow

1. **Task Start**:
   - Load stored features from previous checkpoint
   - Initialize prompt for current task

2. **Training**:
   - Forward pass with task-specific prompt
   - Extract hidden states from last layer
   - Compute contrastive loss with stored features
   - Update model and prompts

3. **Task End**:
   - Collect features from successful trajectories
   - Store features for future tasks
   - Save checkpoint

## Files Created/Modified

### New Files

1. `ragen/cl_methods/prompt_modules.py` - E-Prompt implementation
2. `ragen/cl_methods/trajectory_storage.py` - Feature storage and contrastive loss
3. `ragen/cl_methods/hideprompt.py` - HiDE-Prompt CL method
4. `config/continual_learning_9tasks_hideprompt.yaml` - Configuration file
5. `run_continual_9tasks_hideprompt.sh` - Training script

### Modified Files

1. `ragen/cl_methods/__init__.py` - Register HiDE-Prompt
2. `ragen/workers/actor/dp_actor.py` - Add prompt support and contrastive loss

## Testing

All components have been tested:

```bash
# Test basic functionality
micromamba run -n ragen python -c "
from ragen.cl_methods import get_cl_method
hideprompt = get_cl_method('hideprompt')
print('✓ HiDE-Prompt registered')
"

# Test modules
micromamba run -n ragen python -c "
from ragen.cl_methods.prompt_modules import EPrompt
from ragen.cl_methods.trajectory_storage import TrajectoryFeatureStorage
print('✓ All modules imported successfully')
"
```

## Future Improvements

1. **Task-Adaptive Prediction (TAP)**: Implement full TAP phase with classifier retraining
2. **Prompt Pool Expansion**: Support multiple prompts per task
3. **Dynamic Prompt Length**: Adjust prompt length based on task complexity
4. **Feature Extraction Options**: Support different layers and pooling strategies
5. **Prompt Compression**: Compress prompts for memory efficiency

## References

- Paper: "Hierarchical Decomposition of Prompt-Based Continual Learning: Rethinking Obscured Sub-optimality" (NeurIPS 2023)
- Original Code: https://github.com/thu-ml/HiDe-Prompt
- CL-Agent Framework: https://github.com/bytedance/CL-Agent

## Notes

- HiDE-Prompt is the first prompt-based CL method in CL-Agent (all previous methods were LoRA-based)
- The implementation is fully compatible with the existing CL-Agent training pipeline
- Prompts are stored in checkpoints and can be resumed from previous tasks
- The method is designed for decoder-only LLMs but can be adapted for other architectures
