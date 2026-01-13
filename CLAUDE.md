# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CL-Agent is a research framework for training LLM-based agents using reinforcement learning with a focus on **continual learning** across multiple interactive environments. Built on top of RAGEN (which uses veRL for RL training and vLLM for inference), it enables investigating how LLM-based agents can learn sequentially across diverse tasks while mitigating catastrophic forgetting.

The framework uses:
- **Multi-turn agent-environment interactions** with PPO-based policy optimization
- **Low-Rank Adaptation (LoRA)** for parameter-efficient fine-tuning
- **Ray + FSDP** for distributed training across multiple GPUs
- **Hydra** for hierarchical configuration management
- **WandB** for experiment tracking

## Development Setup

### Environment Setup

```bash
# Create and activate conda environment
conda create -n ragen python=3.9 -y
conda activate ragen

# Install the package in editable mode
pip install -e .

# Install PyTorch with CUDA 12.4
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Install flash-attention (optional but recommended)
conda install -c "nvidia/label/cuda-12.4.0" cuda-toolkit -y
export CUDA_HOME=$CONDA_PREFIX
pip3 install flash-attn --no-build-isolation

# Install remaining requirements
pip install -r requirements.txt

# Initialize and install veRL submodule
git submodule init
git submodule update
cd verl
pip install -e .
cd ..
```

## Common Commands

### Single Environment Training

```bash
# Train on Bandit environment
python train_continual.py --config-name _1_bandit

# Train on Sokoban with specific GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_continual.py --config-name _2_sokoban

# Train on Frozen Lake
python train_continual.py --config-name _3_frozen_lake
```

### Continual Learning Methods

```bash
# Baseline method (shared LoRA across all tasks)
bash run_continual.sh

# With custom task order (0=Bandit, 1=Sokoban, 2=FrozenLake)
TASK_ORDER=210 bash run_continual.sh  # FrozenLake -> Sokoban -> Bandit

# O-LoRA method (orthogonal subspace learning)
bash run_continual_olora.sh

# With custom hyperparameters
LAMBDA_ORTHO=0.3 LAMBDA_L2=0.01 bash run_continual_olora.sh

# Experience Replay (in-context learning)
ENV_TAG=Bandit BUFFER_SIZE=20 bash run_er.sh

# Mix training (interleaved multi-task)
bash run_mix.sh
```

### Resume Training

```bash
# Resume baseline training from checkpoint
RESUME_CHECKPOINT=checkpoints/baseline/20260110_120000/global_step_100 bash run_continual.sh

# Resume O-LoRA training
RESUME_CHECKPOINT=checkpoints/olora/20260110_130000/global_step_200 bash run_continual_olora.sh
```

### Configuration Overrides

```bash
# Override config values via command line
python train_continual.py --config-name continual_learning \
  trainer.total_training_steps=500 \
  continual_learning.steps_per_task=150 \
  "system.CUDA_VISIBLE_DEVICES='0,1,2,3'"
```

## Architecture

### High-Level Structure

The codebase follows a modular architecture separating concerns between:
- **Environments** (`ragen/env/`): Task definitions and reward computation
- **CL Methods** (`ragen/cl_methods/`): Continual learning algorithms
- **Trainers** (`ragen/trainer/`): Training loops and coordination
- **Workers** (`ragen/workers/`): Ray-based distributed workers (Actor, Critic, RefPolicy)
- **Agent Proxy** (`ragen/llm_agent/`): Multi-turn interaction management
- **Configuration** (`config/`): Hydra YAML configs

### Distributed Training Architecture

The framework uses a **controller-worker** pattern with Ray:

1. **Controller (Trainer)**:
   - Orchestrates the training loop
   - Manages Ray worker groups (Actor, Critic, RefPolicy, optionally RewardModel)
   - Cannot directly access model parameters (due to FSDP distribution)
   - Communicates with workers via Ray RPC

2. **Workers** (Ray actors on GPUs):
   - **ActorRolloutRefWorker**: Runs policy rollouts (vLLM) and policy updates (FSDP)
   - **CriticWorker**: Value function training
   - **RefPolicy**: Reference policy for KL divergence
   - Each worker has FSDP-sharded model replicas

3. **Key Implication**: CL methods must work with checkpoint paths and state dicts, not direct model references. Loss computations (like O-LoRA orthogonal loss) happen inside workers, not in the trainer.

### Continual Learning Method Flow

All CL methods follow this lifecycle:

```
┌─────────────────────────────────────────────────────┐
│ Trainer (train_continual.py)                       │
│                                                     │
│  For each task in sequence:                        │
│    1. cl_method.on_task_start()                    │
│    2. cl_method.get_cl_loss_config()               │
│       └─> Sent to workers via set_cl_config()      │
│    3. trainer.fit()                                │
│       └─> Workers compute CL loss if configured    │
│    4. Save checkpoint                              │
│    5. cl_method.on_task_end()                      │
└─────────────────────────────────────────────────────┘
```

**CL Methods** (`ragen/cl_methods/`):
- `BaseCLMethod`: Abstract base with lifecycle hooks
- `NaiveCLMethod`: Baseline - single shared LoRA across all tasks
- `OLoRACLMethod`: Learns each task in orthogonal subspace to previous tasks
- `MixCLMethod`: Interleaved multi-task training

**CL Loss Functions** (`ragen/cl_methods/loss_functions.py`):
- `compute_olora_loss()`: Orthogonality constraint between current and frozen LoRA params
- Loss is computed in workers during backward pass
- Formula: `L_total = L_PPO + λ_ortho * L_ortho + λ_l2 * L_l2`

### Multi-Turn Agent Interaction

The `agent_proxy` module manages multi-turn interactions:

1. **Context Window Modes**:
   - `full`: All previous turns in context (default)
   - `limited_multi_turn`: Last k turns in context
   - `single_turn`: No history, each turn independent

2. **Action Parsing**:
   - Actions separated by `action_sep` (default: "||")
   - Max `max_actions_per_turn` actions per turn
   - Think tokens (before actions) optional via `enable_think`

3. **Reward Assignment**:
   - Token-level rewards computed from environment feedback
   - `use_turn_scores=False`: Sum of token scores becomes reward for last turn
   - `use_turn_scores=True`: Individual turn scores used in GAE

### Environment Manager

`EnvStateManager` (`ragen/llm_agent/es_manager.py`) handles:
- Parallel environment execution (env_groups × group_size instances)
- State synchronization across Ray workers
- Format validation (penalizes malformed actions via `format_penalty`)
- Environment registration from `ragen/env/` submodules

Environments must implement:
- `reset(seed) -> (observation, info)`
- `step(action) -> (observation, reward, done, info)`

### Checkpoint Structure

```
checkpoints/
├── baseline/                    # Baseline CL method
│   └── {timestamp}/
│       ├── global_step_100/
│       │   ├── actor/
│       │   │   └── model.pt
│       │   └── critic/
│       │       └── model.pt
│       └── global_step_200/
│           └── ...
└── olora/                       # O-LoRA CL method
    └── {timestamp}/
        └── ...
```

- Checkpoints saved at `steps_per_task` intervals (end of each task)
- `latest_checkpointed_iteration.txt` tracks global step
- FSDP state dicts stored per worker role

### Configuration Hierarchy

Hydra configs use inheritance:
```
config/
├── base.yaml                    # Base settings (model, LoRA, PPO params)
├── envs.yaml                    # Environment definitions
├── continual_learning.yaml      # Baseline CL (extends base)
├── continual_learning_olora.yaml # O-LoRA CL (extends base)
├── experience_replay.yaml       # ER method
├── mix_training.yaml            # Mix method
├── _1_bandit.yaml              # Single-env: Bandit
├── _2_sokoban.yaml             # Single-env: Sokoban
└── _3_frozen_lake.yaml         # Single-env: Frozen Lake
```

**Key Config Sections**:
- `model_path`: HuggingFace model (e.g., "Qwen/Qwen2.5-0.5B-Instruct")
- `lora`: LoRA rank, alpha, target modules
- `actor_rollout_ref`: Actor/rollout/ref policy settings
- `critic`: Value function settings
- `algorithm`: PPO algorithm params (gamma, lambda, advantage estimator)
- `trainer`: Training loop params (steps, save_freq, test_freq)
- `agent_proxy`: Multi-turn interaction settings
- `es_manager`: Environment manager settings (env_groups, group_size)

## Key Implementation Details

### Validation on Multiple Tasks

`ContinualLearningAgentTrainer` extends `RayAgentTrainer` to validate on all seen tasks:
- `init_cl_validation()`: Creates ES managers for tasks 0 to current_task_idx
- During validation, evaluates on each previous task separately
- Metrics logged as `val-env/{TaskName}/success`, `val-env/{TaskName}/reward`

### Global Step Management

- Global step = cumulative steps across all tasks
- Current task = `global_step // steps_per_task`
- When resuming, `get_global_steps_from_checkpoint()` reads from:
  1. `latest_checkpointed_iteration.txt`
  2. Checkpoint directory name (`{timestamp}_{total_steps}`)
  3. Subdirectory name (`global_step_{N}`)

### Task Order Specification

Use `task_order` string to reorder tasks:
- "012" (default): Bandit → Sokoban → Frozen Lake
- "102": Sokoban → Bandit → Frozen Lake
- "210": Frozen Lake → Sokoban → Bandit

Set via environment variable or config:
```bash
TASK_ORDER=210 bash run_continual.sh
# or
python train_continual.py --config-name continual_learning continual_learning.task_order=210
```

### O-LoRA Orthogonal Loss

Computed in `ragen/cl_methods/loss_functions.py`:
1. Load frozen LoRA params from previous task checkpoints
2. For each LoRA layer, compute orthogonality:
   ```
   L_ortho = Σ ||A_new^T @ A_frozen||²
   ```
3. Add to total loss: `loss += lambda_ortho * L_ortho + lambda_l2 * L_l2`

Workers call `get_cl_loss_fn()` during training to apply this loss.

### Experience Replay (In-Context Learning)

- Parameter-free: No weight updates during replay phase
- Stores successful trajectories in buffer (max `buffer_size`)
- Retrieves examples via selection strategy: `random`, `recent`, or `best`
- Concatenates `max_examples_in_prompt` examples into prompt
- Validates every `val_frequency` new experiences

## Testing

```bash
# Run tests
pytest tests/

# Test specific environment
python -c "from ragen.env.bandit.env import BanditEnv; env = BanditEnv(...); print(env.reset())"
```

## WandB Integration

All training runs log to WandB automatically:
- Project: `ragen_continual_learning` (or configured `trainer.project_name`)
- Run name: `{timestamp}_{method}_{task_name}`
- Logged metrics:
  - Training: `actor/loss`, `critic/loss`, `kl_divergence`
  - Validation: `val-env/{EnvName}/success`, `val-env/{EnvName}/reward`
  - Timing: `timing_s/step`, `timing_s/rollout`, `timing_s/update`
  - Throughput: `perf/tokens_per_second`

Control WandB mode:
```bash
export WANDB_MODE=online   # online, offline, or disabled
```

## Adding New Components

### Adding a New Environment

1. Create environment directory:
   ```
   ragen/env/my_env/
   ├── __init__.py
   ├── config.py      # MyEnvConfig dataclass
   └── env.py         # MyEnv(Env) class
   ```

2. Implement `reset()` and `step()` methods

3. Register in `ragen/env/__init__.py`:
   ```python
   REGISTERED_ENVS['my_env'] = MyEnv
   REGISTERED_ENV_CONFIGS['my_env'] = MyEnvConfig
   ```

4. Add to `config/envs.yaml`

### Adding a New CL Method

1. Create method file `ragen/cl_methods/my_method.py`:
   ```python
   from .base import BaseCLMethod, CLMethodConfig
   from .registry import register_cl_method

   @register_cl_method("my_method")
   class MyMethod(BaseCLMethod):
       def on_task_start(self, task_idx, task_name, checkpoint_path):
           # Called before task starts
           pass

       def on_task_end(self, task_idx, task_name, checkpoint_path):
           # Called after task completes
           pass

       def get_cl_loss_config(self):
           # Return dict for worker-side loss computation
           return {"method_name": "my_method", ...}
   ```

2. Import in `ragen/cl_methods/__init__.py`

3. Create config `config/my_method.yaml` extending `base.yaml`

4. Create shell script `run_my_method.sh`

## Important Notes

- **Never access models directly in trainer**: Use checkpoint paths and Ray RPC
- **CL loss computed in workers**: Not in trainer (due to FSDP sharding)
- **Checkpoints save at task boundaries**: `save_freq = steps_per_task`
- **Ray cleanup**: `ray.shutdown()` called between tasks to free resources
- **Config resolution**: All `${...}` interpolations must be resolved before worker creation
- **Environment variable escaping**: Use `"system.CUDA_VISIBLE_DEVICES='0,1,2,3'"` in Hydra overrides
- **Response length**: Set `actor_rollout_ref.rollout.response_length` based on expected trajectory length per turn
