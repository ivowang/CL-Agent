"""
Experience Replay Training Script for RAGEN

This script trains an agent using Experience Replay (ER) method,
which collects successful trajectories and uses them as in-context
examples without updating model parameters.

Usage:
    # Train on Bandit environment
    python train_er.py --config-name experience_replay
    
    # Train on specific environment
    python train_er.py --config-name experience_replay \
        experience_replay.env_tag=CoordSokoban

Or use the shell script:
    bash run_er.sh
    ENV_TAG=CoordSokoban bash run_er.sh
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

from ragen.utils import register_resolvers
register_resolvers()


def add_dependency_and_validate_config(config):
    """Validate and add dependencies to config."""
    assert config.micro_batch_size_per_gpu * config.trainer.n_gpus_per_node <= config.actor_rollout_ref.actor.ppo_mini_batch_size
    assert config.actor_rollout_ref.actor.ppo_mini_batch_size % (config.micro_batch_size_per_gpu * config.trainer.n_gpus_per_node) == 0
    assert "qwen" in config.model_path.lower() or (not config.enable_response_mask)
    assert len(str(config.system.CUDA_VISIBLE_DEVICES).split(',')) == config.trainer.n_gpus_per_node
    
    config.data.train_batch_size = config.es_manager.train.env_groups * config.es_manager.train.group_size
    return config


def setup_env_config(config, env_tag: str):
    """
    Setup environment configuration based on env_tag.
    
    Args:
        config: The configuration object
        env_tag: Environment tag (e.g., "Bandit", "CoordSokoban", "CoordFrozenLake")
    """
    with open_dict(config):
        # Set n_gpus_per_node based on CUDA_VISIBLE_DEVICES
        n_gpus = len(str(config.system.CUDA_VISIBLE_DEVICES).split(','))
        config.trainer.n_gpus_per_node = n_gpus
        
        # Adjust batch sizes based on GPU count
        # Ensure ppo_mini_batch_size is divisible by (micro_batch_size * n_gpus)
        micro_batch = config.micro_batch_size_per_gpu
        if config.actor_rollout_ref.actor.ppo_mini_batch_size < micro_batch * n_gpus:
            config.actor_rollout_ref.actor.ppo_mini_batch_size = micro_batch * n_gpus
        
        # Set train environment
        config.es_manager.train.env_configs.tags = [env_tag]
        config.es_manager.train.env_configs.n_groups = [config.es_manager.train.env_groups]
        
        # Set validation environment
        config.es_manager.val.env_configs.tags = [env_tag]
        config.es_manager.val.env_configs.n_groups = [config.es_manager.val.env_groups]
        
        # Update experiment name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.trainer.experiment_name = f"er_{env_tag}_{timestamp}"
    
    return config


@hydra.main(version_base=None, config_path="config", config_name="experience_replay")
def main(config):
    """
    Main entry point for Experience Replay training.
    """
    from verl.utils.fs import copy_to_local
    from verl.utils import hf_tokenizer, hf_processor
    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
    from ragen.workers.fsdp_workers import ActorRolloutRefWorker
    from verl.single_controller.ray import RayWorkerGroup
    from verl.single_controller.ray.base import create_colocated_worker_cls
    from ragen.trainer.er_trainer import ExperienceReplayTrainer, ERConfig
    
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
    print(f"Validation frequency (M): {er_config_dict.val_frequency}")
    print(f"Max examples in prompt: {er_config_dict.max_examples_in_prompt}")
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
        val_frequency=er_config_dict.val_frequency,
        env_tag=env_tag,
        max_examples_in_prompt=er_config_dict.max_examples_in_prompt,
        example_selection=er_config_dict.get('example_selection', 'random'),
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
