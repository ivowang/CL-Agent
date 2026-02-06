# CL-Agent: Continual Learning for LLM Agents

**Methods Implemented**
- Naive shared LoRA (baseline)
- O-LoRA
- SD-LoRA
- HiDE-Prompt
- L2P
- Experience Replay (in-context)
- MemoryBank (retrieval-augmented in-context)
- Mix (multi-task interleaved training)
- MoE (per-task experts)

**Environments**
- Bandit
- Sokoban
- FrozenLake

**Code Structure**
- `cl_methods/`: Continual learning methods and method registry.
- `env/`: Environment implementations and configs.
- `llm_agent/`: Agent proxy, context manager, and environment state manager.
- `trainer/`: Trainers and rollout logic.
- `workers/`: Actor/critic workers and FSDP integrations.
- `runs/`: Entry-point scripts for experiments.
- `train_scripts/`: Training entry points.
- `config/`: Hydra configs for all experiments.
- `verl/`: External training backend (submodule).

**Usage**
- Install dependencies: `pip install -r requirements.txt`
- Install the project: `pip install -e .`
- Init submodule: `git submodule update --init`
- Install VERL: `cd verl && pip install -e .`

**Run Experiments (Examples)**
- Baseline continual learning: `bash runs/run_baseline.sh`
- O-LoRA: `bash runs/run_olora.sh`
- SD-LoRA: `bash runs/run_sdlora.sh`
- HiDE-Prompt: `bash runs/run_hideprompt.sh`
- L2P: `bash runs/run_l2p.sh`
- Experience Replay: `ENV_TAG=BanditLow bash runs/run_er.sh`
- MemoryBank: `ENV_TAG=BanditLow bash runs/run_memorybank.sh`
- Mix training: `bash runs/run_mix.sh`
- MoE: `bash runs/run_moe.sh`

All experiments are configuration-driven via `config/*.yaml`. Key runtime overrides include `CUDA_VISIBLE_DEVICES`, `WANDB_MODE`, `TASK_ORDER`, `ENV_TAG`, and `RESUME_CHECKPOINT`.
