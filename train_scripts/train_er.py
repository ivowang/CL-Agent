"""
Experience Replay Training Script for RAGEN

This script trains an agent using Experience Replay (ER) method,
which collects successful trajectories and uses them as in-context
examples without updating model parameters.

Usage:
    # Train on Bandit environment
    python train_scripts/train_er.py --config-name experience_replay
    
    # Train on specific environment
    python train_scripts/train_er.py --config-name experience_replay \
        experience_replay.env_tag=CoordSokoban

Or use the shell script:
    bash runs/run_er.sh
    ENV_TAG=CoordSokoban bash runs/run_er.sh
"""

import ray
import hydra
import os
import sys
import time
from datetime import datetime
from pprint import pprint
from copy import deepcopy

from omegaconf import OmegaConf, open_dict

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils import register_resolvers
register_resolvers()


def add_dependency_and_validate_config(config):
    """Validate and add dependencies to config."""
    assert "qwen" in config.model_path.lower() or (not config.enable_response_mask)
    assert len(str(config.system.CUDA_VISIBLE_DEVICES).split(',')) == config.trainer.n_gpus_per_node
    config.data.train_batch_size = config.es_manager.train.env_groups * config.es_manager.train.group_size
    return config


def _find_mix_env_config(config, env_tag: str):
    if not hasattr(config, "mix_training"):
        raise ValueError("mix_training config not found; ensure experience_replay.yaml defaults to mix_training_9tasks")
    for env_cfg in config.mix_training.environments:
        if env_tag in list(env_cfg.train_tags):
            return env_cfg
    raise ValueError(f"Unknown env_tag '{env_tag}'. Not found in mix_training.environments.")


def setup_env_config(config, env_tag: str):
    """Setup environment configuration based on mix_training_9tasks benchmark settings."""
    env_cfg = _find_mix_env_config(config, env_tag)
    with open_dict(config):
        # Set n_gpus_per_node based on CUDA_VISIBLE_DEVICES
        n_gpus = len(str(config.system.CUDA_VISIBLE_DEVICES).split(','))
        config.trainer.n_gpus_per_node = n_gpus

        # Train environment (match Mix Training groups)
        config.es_manager.train.env_configs.tags = list(env_cfg.train_tags)
        config.es_manager.train.env_configs.n_groups = list(env_cfg.train_n_groups)
        config.es_manager.train.env_groups = sum(env_cfg.train_n_groups)

        # Validation environment (match Mix Training tags/groups)
        config.es_manager.val.env_configs.tags = list(env_cfg.val_tags)
        config.es_manager.val.env_configs.n_groups = list(env_cfg.val_n_groups)
        config.es_manager.val.env_groups = sum(env_cfg.val_n_groups)

        # Update experiment name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.trainer.experiment_name = f"er_{env_tag}_{timestamp}"

    return config


@hydra.main(version_base=None, config_path="../config", config_name="experience_replay")
def main(config):
    """
    Main entry point for Experience Replay training.
    """
    from verl.utils.fs import copy_to_local
    from verl.utils import hf_tokenizer, hf_processor
    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
    from workers.fsdp_workers import ActorRolloutRefWorker
    from verl.single_controller.ray import RayWorkerGroup
    from verl.single_controller.ray.base import create_colocated_worker_cls
    from trainer.er_trainer import ExperienceReplayTrainer, ERConfig
    
    er_config_dict = config.experience_replay
    env_tag = er_config_dict.env_tag
    
    # Setup environment configuration
    config = setup_env_config(config, env_tag)
    config = add_dependency_and_validate_config(config)
    
    # Resolve all interpolations
    OmegaConf.resolve(config)
    
    # Print configuration
    print("=" * 60)
    print("RAGEN Experience Replay Training")
    print("=" * 60)
    print(f"Environment: {env_tag}")
    print(f"Buffer size (N): {er_config_dict.buffer_size}")
    print("=" * 60)
    
    # Setup environment
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.system.CUDA_VISIBLE_DEVICES)
    print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(runtime_env={
            'env_vars': {
                'TOKENIZERS_PARALLELISM': 'true',
                'NCCL_DEBUG': 'WARN',
                'VLLM_LOGGING_LEVEL': 'WARN',
                "RAY_DEBUG": "legacy"
            }
        })
    
    # Setup tokenizer and processor
    local_path = copy_to_local(config.actor_rollout_ref.model.path)
    tokenizer = hf_tokenizer(local_path)
    processor = hf_processor(local_path, use_fast=True)
    
    # Setup role worker mapping
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
    }
    
    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    
    mapping = {
        Role.ActorRollout: global_pool_id,
    }
    
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    
    # IMPORTANT: Must create resource pools first
    resource_pool_manager.create_resource_pool()
    
    # Create worker group using the same pattern as RayPPOTrainer
    print("[ER] Creating actor rollout worker group...")
    
    # Get resource pool using Role enum
    resource_pool = resource_pool_manager.get_resource_pool(Role.ActorRollout)
    
    # Create class dict for colocated worker
    from verl.single_controller.ray import RayClassWithInitArgs
    actor_rollout_cls = RayClassWithInitArgs(
        cls=role_worker_mapping[Role.ActorRollout],
        config=config.actor_rollout_ref,
        role="actor_rollout",
    )
    class_dict = {'actor_rollout': actor_rollout_cls}
    worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
    
    # Create worker group
    wg_dict = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=worker_dict_cls,
        default_megatron_kwargs={},
    )
    
    # Spawn and get actor rollout worker group
    all_wg = wg_dict.spawn(prefix_set=class_dict.keys())
    actor_rollout_wg = all_wg["actor_rollout"]
    
    # Initialize workers
    print("[ER] Initializing workers...")
    actor_rollout_wg.init_model()
    
    # Create ER config
    er_config = ERConfig(
        buffer_size=er_config_dict.buffer_size,
        env_tag=env_tag,
        output_dir=er_config_dict.output_dir,
    )
    
    # Create trainer
    trainer = ExperienceReplayTrainer(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        actor_rollout_wg=actor_rollout_wg,
        er_config=er_config,
    )
    
    # Run training
    final_stats = trainer.fit()
    
    # Cleanup
    ray.shutdown()
    
    print(f"\n{'=' * 60}")
    print("Training Complete!")
    print(f"{'=' * 60}")
    pprint(final_stats)
    
    return final_stats


if __name__ == '__main__':
    # Add config directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.argv.extend([
        "--config-dir", os.path.join(script_dir, "config"),
        "--config-dir", os.path.join(script_dir, "verl/verl/trainer/config"),
    ])
    main()
