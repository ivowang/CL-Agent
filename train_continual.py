"""
Continual Learning Trainer for RAGEN

This script enables sequential training on multiple environments using different
continual learning methods (naive/baseline, O-LoRA, etc.).

Checkpoint Structure:
    checkpoints/baseline/{timestamp}_{total_steps}/   # For naive/baseline method
    checkpoints/olora/{timestamp}_{total_steps}/      # For O-LoRA method
    
Usage:
    # Baseline (naive): shared LoRA - train from scratch
    python train_continual.py --config-name continual_learning
    
    # O-LoRA: orthogonal subspace learning - train from scratch
    python train_continual.py --config-name continual_learning_olora
    
    # Resume from checkpoint
    python train_continual.py --config-name continual_learning \
        continual_learning.resume_checkpoint=/path/to/checkpoint

Or use the shell scripts:
    bash run_continual.sh        # Baseline
    bash run_continual_olora.sh  # O-LoRA
"""

import ray
import hydra
import os
import time
from datetime import datetime
from verl import DataProto
import torch
import numpy as np
from ragen.utils import register_resolvers
register_resolvers()
import sys
import socket
from pprint import pprint
from omegaconf import OmegaConf, open_dict
from copy import deepcopy

# Import CL methods
from ragen.cl_methods import get_cl_method
from ragen.cl_methods.base import CLMethodConfig
from ragen.cl_methods.olora import OLoRAConfig, OLoRACLMethod
from ragen.cl_methods.naive import NaiveCLConfig, NaiveCLMethod


class DummyRewardManager():
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score

    def __call__(self, data: DataProto, return_dict=False):
        if 'rm_scores' in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch['rm_scores']}
            else:
                return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        all_scores = []
        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
            score = data_item.non_tensor_batch['reward']
            score = float(score)
            reward_tensor[i, valid_response_length - 1] = score
            all_scores.append(score)
            data_source = data_item.non_tensor_batch.get('data_source', 'default')
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        if return_dict:
            return {"reward_tensor": reward_tensor}
        else:
            return reward_tensor


def get_custom_reward_fn(config):
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


def add_dependency_and_validate_config(config):
    """Validate and add dependencies to config."""
    assert config.micro_batch_size_per_gpu * config.trainer.n_gpus_per_node <= config.actor_rollout_ref.actor.ppo_mini_batch_size
    assert config.actor_rollout_ref.actor.ppo_mini_batch_size % (config.micro_batch_size_per_gpu * config.trainer.n_gpus_per_node) == 0
    assert "qwen" in config.model_path.lower() or (not config.enable_response_mask)
    assert len(str(config.system.CUDA_VISIBLE_DEVICES).split(',')) == config.trainer.n_gpus_per_node
    
    context_window_mode = getattr(config.agent_proxy, "context_window_mode", "full")
    if context_window_mode in ("single_turn", "limited_multi_turn"):
        assert config.es_manager.train.env_groups * config.es_manager.train.group_size * config.actor_rollout_ref.rollout.rollout_filter_ratio * config.agent_proxy.max_turn >= config.ppo_mini_batch_size
    else:
        assert config.es_manager.train.env_groups * config.es_manager.train.group_size * config.actor_rollout_ref.rollout.rollout_filter_ratio >= config.ppo_mini_batch_size
    
    assert config.algorithm.bi_level_gae == False or config.algorithm.adv_estimator == "gae"
    assert config.algorithm.bi_level_gae == False or (not config.agent_proxy.use_turn_scores)
    
    config.data.train_batch_size = config.es_manager.train.env_groups * config.es_manager.train.group_size
    return config


def get_cl_method_config(cl_config):
    """Extract CL method configuration from the continual_learning config."""
    method_config = getattr(cl_config, 'method', None)
    if method_config is None:
        # Default to naive/baseline method
        return {'name': 'baseline'}
    result = OmegaConf.to_container(method_config, resolve=True)
    # Normalize method name: 'naive' -> 'baseline' for consistency
    if result.get('name') == 'naive':
        result['name'] = 'baseline'
    return result


def get_checkpoint_dir(base_dir, method_name):
    """Get the checkpoint directory for a specific CL method."""
    # Normalize method name
    if method_name == 'naive':
        method_name = 'baseline'
    return os.path.join(base_dir, method_name)


def get_global_steps_from_checkpoint(checkpoint_path):
    """Extract global steps from checkpoint path or checkpoint metadata."""
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        return 0
    
    # Try to read from latest_checkpointed_iteration.txt
    iteration_file = os.path.join(checkpoint_path, 'latest_checkpointed_iteration.txt')
    if os.path.exists(iteration_file):
        with open(iteration_file, 'r') as f:
            try:
                return int(f.read().strip())
            except ValueError:
                pass
    
    # Try to extract from directory name (format: {timestamp}_{total_steps})
    dirname = os.path.basename(checkpoint_path.rstrip('/'))
    parts = dirname.split('_')
    if len(parts) >= 2:
        try:
            return int(parts[-1])
        except ValueError:
            pass
    
    # Try to find global_step_* subdirectory
    for item in os.listdir(checkpoint_path):
        if item.startswith('global_step_'):
            try:
                return int(item.split('_')[-1])
            except ValueError:
                pass
    
    return 0


def init_cl_method(cl_method_config, task_idx, config, cl_method_state=None):
    """Initialize CL method with proper configuration."""
    method_name = cl_method_config.get('name', 'baseline')
    
    if method_name == 'olora':
        method_cfg = OLoRAConfig(
            name='olora',
            current_task_idx=task_idx,
            lambda_ortho=cl_method_config.get('lambda_ortho', 0.5),
            lambda_l2=cl_method_config.get('lambda_l2', 0.0),
            lora_rank_per_task=cl_method_config.get('lora_rank', 64),
            checkpoint_base_dir=config.trainer.default_local_dir,
        )
        method = OLoRACLMethod(method_cfg)
    else:
        method_cfg = NaiveCLConfig(
            name='baseline',
            current_task_idx=task_idx,
            checkpoint_base_dir=config.trainer.default_local_dir,
        )
        method = NaiveCLMethod(method_cfg)
    
    # Restore state from previous tasks
    if cl_method_state:
        method.load_state_dict(cl_method_state)
    
    return method


def run_continual_training(config, start_global_step, checkpoint_dir, timestamp, 
                           all_env_configs, cl_method_config, cl_method_state=None,
                           resume_checkpoint=None):
    """
    Run continual training from a given starting step.
    
    This function handles the entire training loop across all tasks,
    automatically switching tasks when steps_per_task is reached.
    """
    from verl.utils.fs import copy_to_local
    from verl.utils import hf_tokenizer, hf_processor
    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
    from ragen.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
    from verl.single_controller.ray import RayWorkerGroup
    from ragen.trainer.cl_agent_trainer import ContinualLearningAgentTrainer
    
    cl_config = config.continual_learning
    tasks = list(cl_config.tasks)
    steps_per_task = cl_config.steps_per_task
    total_tasks = len(tasks)
    total_steps = steps_per_task * total_tasks
    
    method_name = cl_method_config.get('name', 'baseline')
    
    # Calculate which task we should start from
    start_task_idx = start_global_step // steps_per_task
    steps_in_current_task = start_global_step % steps_per_task
    
    if start_task_idx >= total_tasks:
        print(f"[CL] All tasks already completed (global_step={start_global_step})")
        return
    
    print(f"\n[CL] Starting from global_step={start_global_step}")
    print(f"[CL] Current task index: {start_task_idx}")
    print(f"[CL] Steps completed in current task: {steps_in_current_task}")
    
    # Setup environment
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.system.CUDA_VISIBLE_DEVICES)
    print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    
    current_global_step = start_global_step
    current_checkpoint = resume_checkpoint
    
    for task_idx in range(start_task_idx, total_tasks):
        task = tasks[task_idx]
        task_name = task.name
        
        # Calculate steps to train for this task
        if task_idx == start_task_idx:
            steps_remaining = steps_per_task - steps_in_current_task
        else:
            steps_remaining = steps_per_task
        
        print(f"\n{'=' * 60}")
        print(f"[CL-{method_name}] Task {task_idx}: {task_name}")
        print(f"[CL-{method_name}] Global steps: {current_global_step} -> {current_global_step + steps_remaining}")
        print(f"[CL-{method_name}] Steps to train: {steps_remaining}")
        print(f"{'=' * 60}\n")
        
        # Initialize Ray for this task
        if not ray.is_initialized():
            ray.init(runtime_env={
                'env_vars': {
                    'TOKENIZERS_PARALLELISM': 'true',
                    'NCCL_DEBUG': 'WARN',
                    'VLLM_LOGGING_LEVEL': 'WARN',
                    "RAY_DEBUG": "legacy"
                }
            })
        
        # Calculate target global step after this task
        target_global_step = current_global_step + steps_remaining
        
        # Create task-specific config
        task_config = deepcopy(config)
        with open_dict(task_config):
            # IMPORTANT: total_training_steps must be the TARGET global step,
            # not the number of steps to train. This is because when loading
            # from checkpoint, global_steps is restored from the checkpoint,
            # and the trainer checks: is_last_step = global_steps >= total_training_steps
            task_config.trainer.total_training_steps = target_global_step
            task_config.trainer.experiment_name = f"{cl_config.experiment_name}_{method_name}"
            
            # Update environment configs
            task_config.es_manager.train.env_configs.tags = list(task.train_tags)
            task_config.es_manager.train.env_configs.n_groups = list(task.train_n_groups)
            task_config.es_manager.train.env_groups = sum(task.train_n_groups)
            
            task_config.es_manager.val.env_configs.tags = list(task.val_tags)
            task_config.es_manager.val.env_configs.n_groups = list(task.val_n_groups)
            task_config.es_manager.val.env_groups = sum(task.val_n_groups)
            
            # Set checkpoint directory (single directory for all tasks)
            task_config.trainer.default_local_dir = checkpoint_dir
            
            # Configure checkpoint save frequency
            # Use steps_per_task so checkpoints are saved at the end of each task
            # (global_steps % steps_per_task == 0 when a task completes)
            task_config.trainer.save_freq = steps_per_task
            
            # Set resume mode
            if current_checkpoint is None:
                task_config.trainer.resume_mode = "disable"
            else:
                task_config.trainer.resume_mode = "resume_path"
                task_config.trainer.resume_from_path = current_checkpoint
        
        task_config = add_dependency_and_validate_config(task_config)
        
        # Resolve all interpolations in the config (e.g., ${model_path})
        OmegaConf.resolve(task_config)
        
        # Setup tokenizer and processor
        local_path = copy_to_local(task_config.actor_rollout_ref.model.path)
        tokenizer = hf_tokenizer(local_path)
        processor = hf_processor(local_path, use_fast=True)
        
        # Setup role worker mapping
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
        }
        if task_config.actor_rollout_ref.actor.use_ref:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
        else:
            task_config.actor_rollout_ref.actor.use_kl_loss = False
        
        global_pool_id = 'global_pool'
        resource_pool_spec = {
            global_pool_id: [task_config.trainer.n_gpus_per_node] * task_config.trainer.nnodes,
        }
        
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }
        if task_config.actor_rollout_ref.actor.use_ref:
            mapping[Role.RefPolicy] = global_pool_id
        
        if task_config.reward_model.enable:
            from ragen.workers.fsdp_workers import RewardModelWorker
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id
        
        # Setup reward functions
        reward_manager_cls = DummyRewardManager
        compute_score = get_custom_reward_fn(task_config)
        reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, compute_score=compute_score)
        val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, compute_score=compute_score)
        
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
        
        # Initialize CL method
        cl_method = init_cl_method(cl_method_config, task_idx, task_config, cl_method_state)
        cl_method.on_task_start(task_idx, task_name, current_checkpoint)
        
        # Get CL loss config for workers
        cl_loss_config = cl_method.get_cl_loss_config()
        
        # Create trainer
        trainer = ContinualLearningAgentTrainer(
            config=task_config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=RayWorkerGroup,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            cl_val_env_configs=all_env_configs[:task_idx + 1],
            current_task_idx=task_idx,
            cl_method_config=cl_method_config,
            cl_method_state=cl_loss_config,
        )
        trainer.init_workers()
        trainer.init_agent_proxy()
        trainer.init_cl_validation()
        trainer.set_cl_config_on_workers()
        
        # Run training
        trainer.fit()
        
        # Update global step
        current_global_step += steps_remaining
        
        # Update checkpoint path - use the new naming convention
        current_checkpoint = os.path.join(
            checkpoint_dir,
            f"global_step_{current_global_step}"
        )
        
        # Notify CL method that task is ending
        cl_method.on_task_end(task_idx, task_name, current_checkpoint)
        cl_method_state = cl_method.get_state_dict()
        
        print(f"\n[CL-{method_name}] Task {task_idx} ({task_name}) completed!")
        print(f"[CL-{method_name}] Global step: {current_global_step}")
        print(f"[CL-{method_name}] Checkpoint: {current_checkpoint}")
        
        # Shutdown Ray between tasks to clean up resources
        ray.shutdown()
    
    print(f"\n{'=' * 60}")
    print(f"[CL-{method_name}] All tasks completed!")
    print(f"[CL-{method_name}] Final checkpoint: {current_checkpoint}")
    print(f"{'=' * 60}\n")


@hydra.main(version_base=None, config_path="config", config_name="continual_learning")
def main(config):
    """
    Main entry point for continual learning.
    Sequentially trains on multiple environments using the selected CL method.
    """
    cl_config = config.continual_learning
    tasks = list(cl_config.tasks)
    steps_per_task = cl_config.steps_per_task
    total_tasks = len(tasks)
    total_steps = steps_per_task * total_tasks
    
    # Get CL method configuration
    cl_method_config = get_cl_method_config(cl_config)
    method_name = cl_method_config.get('name', 'baseline')
    
    # Get resume checkpoint (optional)
    resume_checkpoint = getattr(cl_config, 'resume_checkpoint', None)
    
    # Determine checkpoint base directory
    checkpoint_base_dir = getattr(cl_config, 'checkpoint_base_dir', 'checkpoints')
    
    # Build all_env_configs for multi-task validation
    all_env_configs = []
    for task in tasks:
        all_env_configs.append({
            'name': task.name,
            'tags': list(task.val_tags),
            'n_groups': list(task.val_n_groups),
        })
    
    # Determine starting point
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        # Resume from checkpoint
        start_global_step = get_global_steps_from_checkpoint(resume_checkpoint)
        # The checkpoint directory is the parent of the global_step_* directory
        checkpoint_dir = os.path.dirname(resume_checkpoint.rstrip('/'))
        # Extract timestamp from the directory name
        timestamp = os.path.basename(checkpoint_dir)
        print(f"[Resume] Loading checkpoint from: {resume_checkpoint}")
        print(f"[Resume] Global step: {start_global_step}")
    else:
        # Start from scratch
        start_global_step = 0
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = os.path.join(
            checkpoint_base_dir, 
            method_name, 
            f"{timestamp}"
        )
        resume_checkpoint = None
    
    # Print training info
    print("=" * 60)
    print("RAGEN Continual Learning")
    print("=" * 60)
    print(f"CL Method: {method_name}")
    if method_name == 'olora':
        print(f"  - lambda_ortho: {cl_method_config.get('lambda_ortho', 0.5)}")
        print(f"  - lambda_l2: {cl_method_config.get('lambda_l2', 0.0)}")
    print(f"Tasks: {[t.name for t in tasks]}")
    print(f"Steps per task: {steps_per_task}")
    print(f"Total steps: {total_steps}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    if resume_checkpoint:
        start_task = start_global_step // steps_per_task
        print(f"Resuming from step {start_global_step} (task {start_task})")
    print("=" * 60)
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Run training
    run_continual_training(
        config=config,
        start_global_step=start_global_step,
        checkpoint_dir=checkpoint_dir,
        timestamp=timestamp,
        all_env_configs=all_env_configs,
        cl_method_config=cl_method_config,
        cl_method_state=None,
        resume_checkpoint=resume_checkpoint,
    )


if __name__ == '__main__':
    # Add config directories - same as train.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.argv.extend([
        "--config-dir", os.path.join(script_dir, "config"),
        "--config-dir", os.path.join(script_dir, "verl/verl/trainer/config"),
    ])
    main()
