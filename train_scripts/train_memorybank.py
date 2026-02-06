"""
MemoryBank Training Script for RAGEN

This script runs in-context learning with retrieval-based memory,
collecting successful trajectories into a MemoryBank and validating
after each addition. No parameter updates.
"""

import os
import sys
import ray
import hydra
from datetime import datetime
from omegaconf import OmegaConf, open_dict

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils import register_resolvers
register_resolvers()


def add_dependency_and_validate_config(config):
    assert "qwen" in config.model_path.lower() or (not config.enable_response_mask)
    assert len(str(config.system.CUDA_VISIBLE_DEVICES).split(',')) == config.trainer.n_gpus_per_node
    config.data.train_batch_size = config.es_manager.train.env_groups * config.es_manager.train.group_size
    return config


def _find_mix_env_config(config, env_tag: str):
    if not hasattr(config, "mix_training"):
        raise ValueError("mix_training config not found; ensure memorybank.yaml defaults to mix_training_9tasks")
    for env_cfg in config.mix_training.environments:
        if env_tag in list(env_cfg.train_tags):
            return env_cfg
    raise ValueError(f"Unknown env_tag '{env_tag}'. Not found in mix_training.environments.")


def setup_env_config(config, env_tag: str):
    env_cfg = _find_mix_env_config(config, env_tag)
    with open_dict(config):
        n_gpus = len(str(config.system.CUDA_VISIBLE_DEVICES).split(','))
        config.trainer.n_gpus_per_node = n_gpus

        config.es_manager.train.env_configs.tags = list(env_cfg.train_tags)
        config.es_manager.train.env_configs.n_groups = list(env_cfg.train_n_groups)
        config.es_manager.train.env_groups = sum(env_cfg.train_n_groups)

        config.es_manager.val.env_configs.tags = list(env_cfg.val_tags)
        config.es_manager.val.env_configs.n_groups = list(env_cfg.val_n_groups)
        config.es_manager.val.env_groups = sum(env_cfg.val_n_groups)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.trainer.experiment_name = f"memorybank_{env_tag}_{timestamp}"
    return config


@hydra.main(version_base=None, config_path="../config", config_name="memorybank")
def main(config):
    from verl.utils.fs import copy_to_local
    from verl.utils import hf_tokenizer, hf_processor
    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
    from workers.fsdp_workers import ActorRolloutRefWorker
    from verl.single_controller.ray import RayWorkerGroup
    from verl.single_controller.ray.base import create_colocated_worker_cls
    from trainer.memorybank_trainer import MemoryBankTrainer
    from cl_methods.memorybank import MemoryBankConfig

    mb_cfg = config.memorybank
    env_tag = mb_cfg.env_tag

    config = setup_env_config(config, env_tag)
    config = add_dependency_and_validate_config(config)
    OmegaConf.resolve(config)

    print("=" * 60)
    print("RAGEN MemoryBank Training")
    print("=" * 60)
    print(f"Environment: {env_tag}")
    print(f"Buffer size (N): {mb_cfg.buffer_size}")
    print(f"Top-K memories: {mb_cfg.top_k}")
    print("=" * 60)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.system.CUDA_VISIBLE_DEVICES)
    print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get('CUDA_VISIBLE_DEVICES', '')

    if not ray.is_initialized():
        ray.init(runtime_env={
            'env_vars': {
                'TOKENIZERS_PARALLELISM': 'true',
                'NCCL_DEBUG': 'WARN',
                'VLLM_LOGGING_LEVEL': 'WARN',
                "RAY_DEBUG": "legacy"
            }
        })

    local_path = copy_to_local(config.actor_rollout_ref.model.path)
    tokenizer = hf_tokenizer(local_path)
    processor = hf_processor(local_path, use_fast=True)

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
    resource_pool_manager.create_resource_pool()

    resource_pool = resource_pool_manager.get_resource_pool(Role.ActorRollout)
    from verl.single_controller.ray import RayClassWithInitArgs
    actor_rollout_cls = RayClassWithInitArgs(
        cls=role_worker_mapping[Role.ActorRollout],
        config=config.actor_rollout_ref,
        role="actor_rollout",
    )
    class_dict = {'actor_rollout': actor_rollout_cls}
    worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)

    wg_dict = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=worker_dict_cls,
        default_megatron_kwargs={},
    )

    all_wg = wg_dict.spawn(prefix_set=class_dict.keys())
    actor_rollout_wg = all_wg["actor_rollout"]

    print("[MemoryBank] Initializing workers...")
    actor_rollout_wg.init_model()

    mb_config = MemoryBankConfig(
        buffer_size=mb_cfg.buffer_size,
        env_tag=env_tag,
        top_k=mb_cfg.top_k,
        decay_tau=mb_cfg.decay_tau,
        output_dir=mb_cfg.output_dir,
    )

    trainer = MemoryBankTrainer(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        actor_rollout_wg=actor_rollout_wg,
        mb_config=mb_config,
    )

    final_stats = trainer.fit()
    ray.shutdown()

    print(f"\n{'=' * 60}")
    print("Training Complete!")
    print(f"{'=' * 60}")
    print(final_stats)
    return final_stats


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    import sys
    sys.argv.extend([
        "--config-dir", os.path.join(script_dir, "config"),
        "--config-dir", os.path.join(script_dir, "verl/verl/trainer/config"),
    ])
    main()
