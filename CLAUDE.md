# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

CL-Agent is a research framework for training LLM-based agents using reinforcement learning with a focus on **continual learning** across multiple interactive environments. Built on top of RAGEN/VERL, it provides modular architecture for implementing and comparing various continual learning strategies.

## Python Environment

This project uses micromamba to manage python environments. There has been a micromamba environment named "ragen" provided for this project in the server, use the following commmand to activate it:
```bash
micromamba activate ragen
```

## Common Commands

### 9-Task Benchmark (3 environments × 3 difficulties)
```bash
# Task orderings (9-digit string):
#   012345678 - Env-first, LMH: BL→BM→BH→SL→SM→SH→FL→FM→FH
#   210543876 - Env-first, HML: BH→BM→BL→SH→SM→SL→FH→FM→FL
#   036147258 - Difficulty-first, LMH: BL→SL→FL→BM→SM→FM→BH→SH→FH
#   258147036 - Difficulty-first, HML: BH→SH→FH→BM→SM→FM→BL→SL→FL

TASK_ORDER=012345678 bash run_continual_9tasks.sh
TASK_ORDER=036147258 bash run_continual_9tasks_olora.sh
TASK_ORDER=258147036 bash run_continual_9tasks_sdlora.sh
```

### Environment Variables
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3   # GPU selection
WANDB_MODE=online              # WandB mode (online/offline/disabled)
TASK_ORDER=012                 # Task order: 0=Bandit, 1=Sokoban, 2=FrozenLake
RESUME_CHECKPOINT=/path/to/checkpoint
```

## Architecture

### Three-Layer Design
```
Training Scripts (train_continual.py, train_er.py, train_mix.py)
           ↓
Trainer Layer (RayAgentTrainer → ContinualLearningAgentTrainer → CLTrainerWithMethod)
           ↓
Worker Layer (DataParallelPPOActor, CriticWorker, EnvStateManager, AgentProxy)
```

### Key Modules

**Continual Learning Methods** (`ragen/cl_methods/`)
- `base.py`: Abstract `BaseCLMethod` interface with hooks: `on_task_start()`, `on_task_end()`, `get_cl_loss_config()`
- `naive.py`: Baseline - shared LoRA across all tasks
- `olora.py`: O-LoRA - learns each task in orthogonal subspaces
- `sdlora.py`: SD-LoRA - separate LoRA per task with learnable scaling factors
- `experience_replay.py`: Stores successful trajectories for in-context learning
- `mix.py`: All environments mixed in each batch
- `loss_functions.py`: CL-specific loss computation (orthogonality loss, etc.)

**Trainers** (`ragen/trainer/`)
- `agent_trainer.py`: Base Ray/FSDP distributed trainer
- `cl_agent_trainer.py`: Multi-task validation support
- `cl_trainer_with_method.py`: Pluggable CL method integration
- `mix_trainer.py`: Interleaved multi-task training

**Workers** (`ragen/workers/`)
- `actor/dp_actor.py`: PPO actor with CL loss injection (stores `_frozen_lora_params` for O-LoRA)
- `critic/dp_critic.py`: Value function training
- `fsdp_workers.py`: FSDP sharding and LoRA adapter management

**Environments** (`ragen/env/`)
- Base classes: `BaseEnv`, `BaseDiscreteActionEnv`, `BaseLanguageBasedEnv`
- Implementations: `bandit/`, `sokoban/`, `frozen_lake/`
- Environment tags: `Bandit`, `CoordSokoban`, `CoordFrozenLake`

**LLM Agent** (`ragen/llm_agent/`)
- `agent_proxy.py`: Multi-turn agent-environment interaction loop
- `es_manager.py`: Manages parallel environment instances
- `ctx_manager.py`: Builds prompts with observations and history

### Configuration (Hydra-based)
- `config/base.yaml`: Base settings
- `config/continual_learning_9tasks.yaml`: Sequential CL config
- `config/continual_learning_9tasks_olora.yaml`: O-LoRA config
- `config/continual_learning_9tasks_sdlora.yaml`: SD-LoRA config
- `config/experience_replay.yaml`: ER config
- `config/mix_training_9tasks.yaml`: Mix training config
- `config/envs.yaml`: Environment definitions

### Data Flow in Training

1. **Rollout**: Workers generate LLM outputs, interact with environments, collect trajectories
2. **Reward**: Compute rewards based on environment feedback
3. **Update**: PPO loss + optional CL loss (e.g., orthogonality for O-LoRA), gradient step
4. **Validation**: Evaluate on current task + all previous tasks

### CL Loss Injection Pattern
Workers receive CL config from trainer and compute additional losses:
```python
# In DataParallelPPOActor
if self._cl_config.get('has_cl_loss'):
    cl_loss = compute_olora_loss(current_lora, frozen_lora_params, lambda_ortho)
    total_loss = ppo_loss + cl_loss
```

## Extension Points

### Adding a New CL Method
1. Create `ragen/cl_methods/your_method.py` inheriting from `BaseCLMethod`
2. Implement: `on_task_start()`, `on_task_end()`, `get_cl_loss_config()`
3. Register with `@register_cl_method("your_method")`
4. Add loss function in `loss_functions.py` if needed

### Adding a New Environment
1. Create `ragen/env/your_env/env.py` inheriting from appropriate base class
2. Implement: `reset()`, `step()`, `get_all_actions()`
3. Register in `ragen/env/__init__.py`
4. Configure in `config/envs.yaml`

## Checkpoints

Saved to `checkpoints/{method}/{timestamp}/global_step_{N}/`:
- `actor/model.pt`, `critic/model.pt`
- `cl_method_state.pt` (for O-LoRA, SD-LoRA)
- `scaling_factors.pt` (for SD-LoRA)
- `metadata.json`
