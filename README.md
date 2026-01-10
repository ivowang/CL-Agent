# CL-Agent: On the Role of Parameter Efficient Continual Learning in Self-Evolving Agents

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

CL-Agent is a research framework for training LLM-based agents using reinforcement learning with a focus on **continual learning** across multiple interactive environments. Built on top of [RAGEN](https://github.com/mll-lab-nu/RAGEN), it provides a modular architecture for implementing and comparing various continual learning strategies.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Supported Environments](#supported-environments)
- [Continual Learning Methods](#continual-learning-methods)
- [Training Scripts](#training-scripts)
- [Configuration](#configuration)
- [Experiment Tracking](#experiment-tracking)

---

## Overview

CL-Agent enables researchers to investigate how LLM-based agents can learn sequentially across diverse interactive tasks while mitigating catastrophic forgetting. The framework supports:

- **Multi-turn agent-environment interactions** with RL-based policy optimization
- **Low-Rank Adaptation (LoRA)** for parameter-efficient fine-tuning
- **Multiple continual learning paradigms**: sequential, interleaved, and in-context learning
- **Distributed training** via Ray and FSDP for scalability

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CL-Agent Framework                          │
├─────────────────────────────────────────────────────────────────┤
│  Continual Learning Methods                                      │
│  ├── Baseline (Naive Sequential)                                 │
│  ├── O-LoRA (Orthogonal Low-Rank Adaptation)                    │
│  ├── Experience Replay (In-Context Learning)                    │
│  └── Mix (Interleaved Multi-Task)                               │
├─────────────────────────────────────────────────────────────────┤
│  Agent Trainer (Ray-based Distributed Training)                 │
│  └── PPO with LoRA fine-tuning                                  │
├─────────────────────────────────────────────────────────────────┤
│  Interactive Environments                                        │
│  ├── Bandit         │  Sokoban        │  Frozen Lake            │
│  ├── Countdown      │  WebShop        │  Lean (Theorem Proving) │
│  └── MetaMathQA     │  Sudoku         │  ...                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Single Environment Training (Standard RL)

```bash
# Train on Sokoban with default settings
python train.py --config-name _2_sokoban

# Train on Bandit with specific GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --config-name _1_bandit
```

### Continual Learning - Baseline Method

```bash
# Sequential training: Bandit → Sokoban → Frozen Lake
bash run_continual.sh

# Custom task order (Frozen Lake → Sokoban → Bandit)
TASK_ORDER=210 bash run_continual.sh
```

### Continual Learning - O-LoRA Method

```bash
# O-LoRA with orthogonal subspace learning
bash run_continual_olora.sh

# With custom hyperparameters
LAMBDA_ORTHO=0.3 LAMBDA_L2=0.01 bash run_continual_olora.sh
```

### Experience Replay (In-Context Learning)

```bash
# Train on Bandit with experience buffer
ENV_TAG=Bandit BUFFER_SIZE=20 bash run_er.sh
```

### Mix Training (Interleaved Multi-Task)

```bash
# Interleaved training across all environments
bash run_mix.sh

# With custom settings
TOTAL_STEPS=300 TEST_FREQ=10 bash run_mix.sh
```

---

## Supported Environments

| Environment | Type | Description | Task |
|-------------|------|-------------|------|
| **Bandit** | Decision | Two-armed bandit | Choose optimal arm based on context |
| **CoordSokoban** | Planning | Sokoban puzzle with coordinates | Push boxes to target positions |
| **CoordFrozenLake** | Navigation | Frozen lake with slippery ice | Navigate to goal avoiding holes |

### Environment Tags

```yaml
# Use these tags in configuration
Bandit              # Two-armed bandit
CoordSokoban        # Sokoban with coordinate observation
CoordFrozenLake     # Frozen Lake with coordinate observation
```

---

## Continual Learning Methods

### 1. Baseline (Naive Sequential)

Standard sequential training where a single LoRA module is trained across all tasks in order. This serves as the baseline for catastrophic forgetting analysis.

```bash
bash run_continual.sh
```

**Key Parameters:**
- `steps_per_task`: Training steps per environment (default: 100)
- `task_order`: Order of environments (e.g., "012", "210")

### 2. O-LoRA (Orthogonal Low-Rank Adaptation)

Based on [Wang et al., 2023](https://arxiv.org/abs/2306.07869), O-LoRA learns each task in an orthogonal subspace to minimize interference between tasks.

```bash
bash run_continual_olora.sh
```

**Key Parameters:**
- `lambda_ortho`: Orthogonal loss weight (default: 0.5)
- `lambda_l2`: L2 regularization weight (default: 0.0)
- `lora_rank`: LoRA rank per task (default: 64)

**Loss Function:**
```
L_total = L_PPO + λ_ortho * L_ortho + λ_l2 * L_l2
```

Where `L_ortho` enforces orthogonality between current and frozen LoRA subspaces.

### 3. Experience Replay (In-Context Learning)

A parameter-free approach that stores successful trajectories and uses them as in-context examples for future interactions.

```bash
ENV_TAG=Bandit bash run_er.sh
```

**Key Parameters:**
- `buffer_size`: Maximum experiences to collect (default: 20)
- `val_frequency`: Validate every N new experiences (default: 5)
- `max_examples_in_prompt`: Examples per prompt (default: 3)
- `example_selection`: Selection strategy - `random`, `recent`, or `best`

**Mechanism:**
1. Successful trajectories are stored in a replay buffer
2. When the agent faces a new instance, relevant examples are retrieved
3. Examples are concatenated into the prompt for in-context learning
4. No parameter updates during this phase

### 4. Mix (Interleaved Multi-Task)

Trains on all environments simultaneously by cycling through them in a round-robin fashion.

```bash
bash run_mix.sh
```

**Key Parameters:**
- `total_training_steps`: Total steps across all environments (default: 300)
- `test_freq`: Validation frequency (default: 10)

**Training Pattern:**
```
Step 1: Bandit → Step 2: Sokoban → Step 3: Frozen Lake → Step 4: Bandit → ...
```

---

## Training Scripts

### Script Overview

| Script | Method | Description |
|--------|--------|-------------|
| `run_continual.sh` | Baseline | Sequential training with shared LoRA |
| `run_continual_olora.sh` | O-LoRA | Sequential with orthogonal subspaces |
| `run_er.sh` | Experience Replay | In-context learning with buffer |
| `run_mix.sh` | Mix | Interleaved multi-task training |
| `run.sh` | Standard RL | Single environment training |

### Common Environment Variables

```bash
# GPU selection
CUDA_VISIBLE_DEVICES=0,1,2,3

# WandB mode (online/offline/disabled)
WANDB_MODE=online

# Resume from checkpoint
RESUME_CHECKPOINT=/path/to/checkpoint

# Task order for CL methods
TASK_ORDER=012  # 0=Bandit, 1=Sokoban, 2=FrozenLake
```

### Checkpoints

Checkpoints are saved in:
```
checkpoints/{method_name}/{timestamp}/global_step_{N}/
├── actor/
│   └── model.pt
├── critic/
│   └── model.pt
└── metadata.json
```

### Resuming Training

```bash
# Resume baseline training
RESUME_CHECKPOINT=checkpoints/baseline/20260110_120000/global_step_100 \
  bash run_continual.sh

# Resume O-LoRA training
RESUME_CHECKPOINT=checkpoints/olora/20260110_130000/global_step_200 \
  bash run_continual_olora.sh
```

---

## Configuration

### Configuration Files

| Config | Purpose |
|--------|---------|
| `config/base.yaml` | Base settings shared across methods |
| `config/continual_learning.yaml` | Baseline CL configuration |
| `config/continual_learning_olora.yaml` | O-LoRA configuration |
| `config/experience_replay.yaml` | Experience Replay configuration |
| `config/mix_training.yaml` | Mix training configuration |
| `config/envs.yaml` | Environment definitions |

### Key Configuration Options

```yaml
# Base model settings
model_path: Qwen/Qwen2.5-0.5B-Instruct
lora:
  rank: 64
  alpha: 64
  target_modules: all-linear

# Training settings
trainer:
  total_training_steps: 200
  test_freq: 10
  save_freq: 100
  val_before_train: True
  logger: ['console', 'wandb']

# Agent settings
agent_proxy:
  max_turn: 5
  max_actions_per_turn: 2
  enable_think: True

# Environment settings
es_manager:
  train:
    env_groups: 8
    group_size: 16
  val:
    env_groups: 32
    group_size: 16
```

### Hydra Overrides

Override any configuration via command line:

```bash
python train_continual.py --config-name continual_learning \
  trainer.total_training_steps=500 \
  continual_learning.steps_per_task=150 \
  "system.CUDA_VISIBLE_DEVICES='0,1'"
```

---

## Experiment Tracking

### WandB Integration

All experiments are automatically logged to Weights & Biases.

**Logged Metrics:**
- Training: `actor/loss`, `critic/loss`, `kl_divergence`
- Validation: `val-env/{EnvName}/success`, `val-env/{EnvName}/reward`
- Timing: `timing_s/step`, `timing_s/rollout`, `timing_s/update`
- Throughput: `perf/tokens_per_second`, `perf/samples_per_second`

**Run Naming Convention:**
```
{timestamp}_{method}_{current_task}
Example: 20260110_143052_olora_bandit
```

### Viewing Results

```bash
# Set WandB project
export WANDB_PROJECT=CL-Agent_continual_learning

# Run training
bash run_continual_olora.sh
```

Access results at: `https://wandb.ai/{your-username}/CL-Agent_continual_learning`

---

## Extending the Framework

### Adding New CL Methods

1. Create a new method file in `CL-Agent/cl_methods/`:

```python
# CL-Agent/cl_methods/my_method.py
from CL-Agent.cl_methods.base import BaseCLMethod
from CL-Agent.cl_methods.registry import register_cl_method

@register_cl_method("my_method")
class MyCLMethod(BaseCLMethod):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Initialize method-specific state
    
    def on_task_start(self, task_idx: int, task_name: str, checkpoint_path: Optional[str] = None):
        # Called before each task begins
        pass
    
    def on_task_end(self, task_idx: int, task_name: str, checkpoint_path: str):
        # Called after each task completes
        pass
    
    def get_cl_loss_config(self) -> Dict[str, Any]:
        # Return configuration for CL-specific loss computation
        return {"method_name": "my_method", ...}
```

2. Register in `CL-Agent/cl_methods/__init__.py`:

```python
from .my_method import MyCLMethod
```

3. Create corresponding configuration and training script.

### Adding New Environments

1. Create environment directory in `CL-Agent/env/`:

```
CL-Agent/env/my_env/
├── __init__.py
├── config.py
└── env.py
```

2. Implement the environment interface:

```python
# CL-Agent/env/my_env/env.py
from CL-Agent.env.base import Env

class MyEnv(Env):
    def __init__(self, config: MyEnvConfig):
        super().__init__(config)
    
    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict]:
        # Return initial observation
        pass
    
    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        # Return (observation, reward, done, info)
        pass
```

3. Register in `config/envs.yaml`.

---

## Acknowledgments

- [RAGEN](https://github.com/mll-lab-nu/RAGEN) - The codebase we built our project on
- [veRL](https://github.com/volcengine/verl) - The underlying RL training framework
- [vLLM](https://github.com/vllm-project/vllm) - High-throughput LLM inference
- [PEFT](https://github.com/huggingface/peft) - Parameter-efficient fine-tuning
- [Hydra](https://hydra.cc/) - Configuration management
- [WandB](https://wandb.ai/) - Experiment tracking
