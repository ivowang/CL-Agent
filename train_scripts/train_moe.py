"""
MoE (Mixture of Experts) Training Script for RAGEN

This script trains an agent using MoE method where each task has its own
independent LoRA module (expert). Training uses mixed sampling (all tasks
in each step) but maintains complete task independence.

Key features:
- 9 independent LoRA modules (one per task)
- 9 independent actor-critic pairs
- Serial sampling: sample from each task sequentially
- Serial updates: update each task's LoRA sequentially

Usage:
    python train_scripts/train_moe.py --config-name moe_training_9tasks

    # With custom parameters
    python train_scripts/train_moe.py --config-name moe_training_9tasks trainer.total_training_steps=900
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

from trainer.moe_trainer import MoEAgentTrainer
from trainer.agent_trainer import ResourcePoolManager
from workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
from cl_methods.moe import MoECLMethod, MoEConfig


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
        print(f"[MoE] Using {actual_gpus} GPUs: {cuda_devices}")

        # Set batch sizes based on environment config
        if hasattr(config, 'es_manager'):
            es_train = config.es_manager.train
            train_batch_size = es_train.env_groups * es_train.group_size

            config.data.train_batch_size = train_batch_size
            config.actor_rollout_ref.actor.ppo_mini_batch_size = train_batch_size // 4
            config.critic.ppo_mini_batch_size = train_batch_size // 4

    return config


def build_task_configs(config) -> List[Dict[str, Any]]:
    """Build task configurations from moe_training config."""
    moe_config = config.moe_training
    task_configs = []

    for task in moe_config.tasks:
        task_config = {
            'name': task.name,
            'train_tags': list(task.train_tags),
            'train_n_groups': list(task.train_n_groups),
            'val_tags': list(task.val_tags),
            'val_n_groups': list(task.val_n_groups),
        }
        task_configs.append(task_config)
        print(f"[MoE] Task config: {task.name}")
        print(f"      Train: {task.train_tags} x {task.train_n_groups}")
        print(f"      Val:   {task.val_tags} x {task.val_n_groups}")

    return task_configs


def create_moe_method(config, task_configs: List[Dict]) -> MoECLMethod:
    """Create and initialize the MoECLMethod instance."""
    # Build MoEConfig from task configs
    moe_config = MoEConfig(
        name="moe",
        num_tasks=len(task_configs),
        task_names=[t['name'] for t in task_configs],
        lora_rank=config.lora.rank,
        lora_alpha=config.lora.alpha,
    )

    moe_method = MoECLMethod(moe_config)
    return moe_method


@hydra.main(version_base=None, config_path="../config", config_name="moe_training_9tasks")
def main(config: DictConfig):
    """
    Main entry point for MoE training.
    Trains with independent LoRA per task using mixed sampling.
    """
    print("=" * 60)
    print("RAGEN MoE Training")
    print("=" * 60)

    moe_config = config.moe_training
    total_steps = config.trainer.total_training_steps

    # Get timestamp for experiment naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Update experiment name
    with open_dict(config):
        config.trainer.experiment_name = f"moe_{timestamp}"

    # Build task configs
    task_configs = build_task_configs(config)
    num_tasks = len(task_configs)

    print(f"[MoE] Total training steps: {total_steps}")
    print(f"[MoE] Number of tasks (experts): {num_tasks}")
    print(f"[MoE] Each task has independent LoRA module")

    # Create MoECLMethod
    moe_method = create_moe_method(config, task_configs)

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
    moe_method.on_task_start(0, "moe_training", None)

    # Create MoE trainer
    print("[MoE] Creating MoE trainer...")
    trainer = MoEAgentTrainer(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
        moe_method=moe_method,
        task_configs=task_configs,
    )

    # Start training (will initialize workers internally)
    print("[MoE] Starting training...")
    trainer.fit()

    # Mark task as ended
    checkpoint_dir = os.path.join(
        config.trainer.default_local_dir,
        f"global_step_{total_steps}"
    )
    moe_method.on_task_end(0, "moe_training", checkpoint_dir)

    print("[MoE] Training completed!")

    # Cleanup
    ray.shutdown()


if __name__ == "__main__":
    main()
