# Repository Guidelines

## Project Structure & Module Organization
- `ragen/`: Core package. `cl_methods/` (naive, O-LoRA, SD-LoRA, replay, mix), `llm_agent/` (agent proxy, context, env manager), `trainer/` (Ray/FSDP trainers), `workers/` (actor/critic/sharding), `env/` (bandit, sokoban, frozen_lake implementations), `patches/` utilities.
- `config/`: Hydra configs for CL, ER, mix, and env definitions (`continual_learning*.yaml`, `experience_replay.yaml`, `envs.yaml`).
- `scripts/`: Environment setup and monitoring helpers; `launch_*.sh` and `run_*` scripts orchestrate multi-task tmux runs and ER jobs.
- `tests/`: Pytest suites for env rendering, context handling, rollout filtering, and seed iteration.
- Data/outputs: `checkpoints/`, `outputs/`, `wandb/`, and `data/` hold run artifacts; avoid committing large files.

## Setup & Environment
- Activate the micromamba env provisioned for the project: `micromamba activate ragen`.
- Install dependencies: `pip install -e .`, `pip install -r requirements.txt`, `git submodule update --init`, then `cd verl && pip install -e .`.
- Torch reference: `pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124` (ensure matching CUDA).
- For flash-attn, install CUDA toolkit if needed and export `CUDA_HOME=$CONDA_PREFIX` before `pip install flash-attn --no-build-isolation`.

## Build, Test, and Development Commands
- Continual learning (Hydra defaults to `config/`): `python train_continual.py --config-name continual_learning` (use `--config-name continual_learning_olora` or `continual_learning_sdlora` variants).
- 9-task runs: `TASK_ORDER=012345678 bash run_continual_9tasks.sh` (baseline), `bash run_continual_9tasks_olora.sh`, `bash run_continual_9tasks_sdlora.sh`; `launch_baseline.sh GPU1 GPU2 GPU3 GPU4` spawns tmux sessions per order.
- Experience replay: `ENV_TAG=BanditLow bash run_er_9tasks.sh` (Bandit/Sokoban/FrozenLake × {Low,Medium,Hard}); for single env config use `python train_er.py --config-name experience_replay experience_replay.env_tag=CoordSokoban`.
- Testing: run `pytest -q` from repo root; target or add tests under `tests/` with `test_*.py` files and `test_*` functions.

## Coding Style & Naming Conventions
- Python style follows PEP 8, 4-space indents, snake_case for modules/functions, CapWords for classes; keep functions short and typed where practical.
- Keep logic config-driven; prefer Hydra overrides over hardcoded constants. Document new configs in `config/*.yaml` with concise comments.
- When extending agents or CL methods, reuse base interfaces (`BaseCLMethod`, trainers, workers) instead of ad-hoc hooks.

## Testing Guidelines
- Mirror module paths when adding coverage (e.g., `ragen/llm_agent/*` → `tests/llm_agent/test_*.py`).
- Use deterministic seeds in tests that touch randomness; mock external calls (LLM/tokenizer) as in `tests/llm_agent/test_context_window.py`.
- Before PRs, run relevant subsets: `pytest tests/llm_agent -q` or full suite if changes are broad.

## Commit & Pull Request Guidelines
- Commit messages: short, imperative summaries (e.g., `fix olora and sdlora`, `add stop_tasks.sh`); keep first line under ~72 chars.
- PRs should include: what changed, configs/env tags used, key metrics or sample logs, and a test plan with exact commands. Link issues and note any new dependencies or checkpoints produced.

## Configuration & Ops Tips
- Common env vars: `CUDA_VISIBLE_DEVICES`, `WANDB_MODE=online|offline|disabled`, `TASK_ORDER`, `ENV_TAG`, `RESUME_CHECKPOINT`.
- Checkpoints live in `checkpoints/{method}/...`; store bulky artifacts outside git and verify permissions before uploading. Use `scripts/monitor_training.py` for live runs and `stop_tasks.sh` to clean tmux sessions responsibly.
