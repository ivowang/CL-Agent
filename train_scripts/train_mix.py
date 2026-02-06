"""
Mix Training Script for RAGEN

This script trains an agent on multiple environments in an interleaved manner,
switching between environments after each training step. Unlike continual
learning methods that train tasks sequentially, this method trains all tasks
simultaneously with a single shared LoRA module.

Usage:
    python train_scripts/train_mix.py --config-name mix_training_9tasks
    
    # With custom parameters
    python train_scripts/train_mix.py --config-name mix_training_9tasks trainer.total_training_steps=300
"""

import os
import sys
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import hydra
import ray
import torch
from omegaconf import OmegaConf, DictConfig, open_dict

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.utils import Role
from verl.utils.fs import copy_to_local
from verl.utils import hf_tokenizer, hf_processor
from verl import DataProto

from utils import register_resolvers
register_resolvers()

from trainer.mix_trainer import MixAgentTrainer
from trainer.agent_trainer import ResourcePoolManager
from workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
from cl_methods.mix import MixCLMethod, MixConfig


# Dummy reward manager (same as in train_continual.py)
class DummyRewardManager():
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score

    def __call__(self, data: DataProto, return_dict=False):
        if 'rm_scores' in data.batch.keys():
            if return_dict:
                return {
                    "reward_tensor": data.batch['rm_scores'],
                    "reward_extra_info": {},
                }
            return data.batch['rm_scores']
        reward_tensor, reward_extra_info = self.compute_score(
            data_source=data.non_tensor_batch['data_source'],
            solution_str=data.non_tensor_batch['responses_str'],
        )
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        return reward_tensor


def get_custom_reward_fn(config):
    """Get the custom reward function based on config."""
    import importlib.util
    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")
    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    if spec is None:
        raise RuntimeError(f"Failed to create module spec from '{file_path}'")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}")
    function_name = reward_fn_config.get("name")
    if not function_name:
        raise ValueError("Function name not specified in custom_reward_function config")
    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")
    print(f"using customized reward function '{function_name}' from '{file_path}'")
    return getattr(module, function_name)


# Add config directories for Hydra
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.argv.extend([
    "--config-dir", os.path.join(script_dir, "config"),
    "--config-dir", os.path.join(script_dir, "verl/verl/trainer/config")
])


def add_dependency_and_validate_config(config):
    """Add dependencies and validate config."""
    OmegaConf.set_struct(config, True)
    with open_dict(config):
        # Set CUDA devices
        cuda_devices = str(config.system.CUDA_VISIBLE_DEVICES)
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
        
        # Dynamically set n_gpus_per_node based on CUDA_VISIBLE_DEVICES
        actual_gpus = len(cuda_devices.split(','))
        config.trainer.n_gpus_per_node = actual_gpus
        print(f"[Mix] Using {actual_gpus} GPUs: {cuda_devices}")
        
        # Set batch sizes based on environment config
        if hasattr(config, 'es_manager'):
            es_train = config.es_manager.train
            train_batch_size = es_train.env_groups * es_train.group_size
            
            config.data.train_batch_size = train_batch_size
            config.actor_rollout_ref.actor.ppo_mini_batch_size = train_batch_size // 4
            config.critic.ppo_mini_batch_size = train_batch_size // 4
    
    return config


def build_env_configs(config) -> List[Dict[str, Any]]:
    """Build environment configurations from mix_training config."""
    mix_config = config.mix_training
    env_configs = []
    
    for env in mix_config.environments:
        env_config = {
            'name': env.name,
            'train_tags': list(env.train_tags),
            'train_n_groups': list(env.train_n_groups),
            'val_tags': list(env.val_tags),
            'val_n_groups': list(env.val_n_groups),
        }
        env_configs.append(env_config)
    
    return env_configs


def _normalize_task_subset(task_subset: Optional[Any]) -> List[str]:
    if task_subset is None:
        return []
    if isinstance(task_subset, str):
        parts = [part.strip() for part in task_subset.split(",") if part.strip()]
        return parts
    return list(task_subset)


def select_env_configs(config) -> List[Dict[str, Any]]:
    """Select environment configs based on mix_training.task_subset."""
    all_env_configs = build_env_configs(config)
    task_to_env = {}
    available_tasks = []

    for env_config in all_env_configs:
        for tag in env_config["train_tags"]:
            if tag in task_to_env:
                raise ValueError(f"Duplicate training tag '{tag}' in mix_training.environments.")
            task_to_env[tag] = env_config
            available_tasks.append(tag)

    raw_task_subset = config.mix_training.get("task_subset", None)
    task_subset = _normalize_task_subset(raw_task_subset)

    if raw_task_subset is None:
        task_subset = available_tasks
    elif not task_subset:
        raise ValueError("mix_training.task_subset must include at least one task.")

    duplicates = []
    seen = set()
    for tag in task_subset:
        if tag in seen:
            duplicates.append(tag)
        seen.add(tag)
    if duplicates:
        raise ValueError(f"mix_training.task_subset contains duplicates: {sorted(set(duplicates))}")

    unknown = [tag for tag in task_subset if tag not in task_to_env]
    if unknown:
        raise ValueError(
            "mix_training.task_subset contains unknown tasks: "
            f"{unknown}. Available tasks: {available_tasks}"
        )

    selected_env_configs = [task_to_env[tag] for tag in task_subset]
    if not selected_env_configs:
        raise ValueError("No environments selected. Check mix_training.environments and task_subset.")
    for env_config in selected_env_configs:
        print(f"[Mix] Environment config: {env_config['name']}")
        print(f"      Train: {env_config['train_tags']} x {env_config['train_n_groups']}")
        print(f"      Val:   {env_config['val_tags']} x {env_config['val_n_groups']}")
    return selected_env_configs


def apply_mix_env_overrides(config, env_configs: List[Dict[str, Any]]) -> None:
    """Update train ES manager settings based on selected environments."""
    all_train_tags = []
    all_train_n_groups = []

    for env_config in env_configs:
        all_train_tags.extend(env_config["train_tags"])
        all_train_n_groups.extend(env_config["train_n_groups"])

    with open_dict(config):
        config.es_manager.train.env_configs.tags = all_train_tags
        config.es_manager.train.env_configs.n_groups = all_train_n_groups
        config.es_manager.train.env_groups = sum(all_train_n_groups)


def create_mix_method(config, env_configs: List[Dict]) -> MixCLMethod:
    """Create and initialize the MixCLMethod instance."""
    # Build MixConfig from environment configs
    mix_config = MixConfig(
        name="mix",
        env_tags=[e['name'] for e in env_configs],
        train_n_groups=[e['train_n_groups'][0] for e in env_configs],
        val_n_groups=[e['val_n_groups'][0] for e in env_configs],
    )
    
    mix_method = MixCLMethod(mix_config)
    return mix_method


@hydra.main(version_base=None, config_path="../config", config_name="mix_training_9tasks")
def main(config: DictConfig):
    """
    Main entry point for mix training.
    Trains on multiple environments in an interleaved manner.
    """
    print("=" * 60)
    print("RAGEN Mix Training")
    print("=" * 60)
    
    mix_config = config.mix_training
    total_steps = config.trainer.total_training_steps
    
    # Get timestamp for experiment naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Update experiment name
    with open_dict(config):
        config.trainer.experiment_name = f"mix_{timestamp}"
    
    # Build environment configs and apply task subset selection
    env_configs = select_env_configs(config)
    apply_mix_env_overrides(config, env_configs)
    num_envs = len(env_configs)
    selected_tasks = [tag for env_config in env_configs for tag in env_config["train_tags"]]
    
    print(f"[Mix] Total training steps: {total_steps}")
    print(f"[Mix] Number of environments: {num_envs}")
    print(f"[Mix] Selected tasks: {selected_tasks}")
    print(f"[Mix] Each environment will get ~{total_steps // num_envs} training steps")
    
    # Create MixCLMethod
    mix_method = create_mix_method(config, env_configs)
    
    # Add dependencies and validate config
    config = add_dependency_and_validate_config(config)
    
    # Resolve all interpolations
    OmegaConf.resolve(config)
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()
    
    # Setup tokenizer and processor
    local_path = copy_to_local(config.actor_rollout_ref.model.path)
    tokenizer = hf_tokenizer(local_path)
    processor = hf_processor(local_path, use_fast=True)
    
    # Setup role worker mapping
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
    }
    if config.actor_rollout_ref.actor.use_ref:
        role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
    else:
        config.actor_rollout_ref.actor.use_kl_loss = False
    
    # Setup resource pools
    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
    }
    if config.actor_rollout_ref.actor.use_ref:
        mapping[Role.RefPolicy] = global_pool_id
    
    if config.reward_model.enable:
        from workers.fsdp_workers import RewardModelWorker
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id
    
    # Setup reward functions
    reward_manager_cls = DummyRewardManager
    compute_score = get_custom_reward_fn(config)
    reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, compute_score=compute_score)
    val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, compute_score=compute_score)
    
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    
    # Mark task as started
    mix_method.on_task_start(0, "mix_training", None)
    
    # Create trainer
    print("[Mix] Creating MixAgentTrainer...")
    trainer = MixAgentTrainer(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=RayWorkerGroup,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
        mix_method=mix_method,
        env_configs=env_configs,
    )
    
    # Initialize workers
    print("[Mix] Initializing workers...")
    trainer.init_workers()
    trainer.init_agent_proxy()
    
    # Run training
    print("[Mix] Starting training loop...")
    print("=" * 60)
    trainer.fit()
    
    # Update mix_method's total_steps (training loop ran inside trainer.fit())
    mix_method.total_steps = total_steps
    
    # Mark task as ended
    checkpoint_dir = getattr(config.trainer, 'default_local_dir', 'checkpoints/mix')
    checkpoint_path = os.path.join(checkpoint_dir, f"global_step_{total_steps}")
    mix_method.on_task_end(0, "mix_training", checkpoint_path)
    
    print("=" * 60)
    print("[Mix] Training completed!")
    print(f"[Mix] Total steps: {mix_method.total_steps}")
    print(f"[Mix] Environments trained simultaneously: {mix_method.env_tags}")
    print("=" * 60)
    
    # Cleanup
    ray.shutdown()


if __name__ == "__main__":
    main()
