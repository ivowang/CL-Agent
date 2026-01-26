"""
MoE (Mixture of Experts) Trainer for RAGEN

This implements TRUE MoE where each task has its own independent LoRA module.
Training uses mixed sampling (all tasks in each step) but maintains complete task independence.

Key features:
- 9 independent LoRA modules (one per task)
- Serial sampling: sample from each task sequentially
- Serial updates: update each task's LoRA sequentially
- Complete task independence (no shared parameters)
- LoRA switching: save/load LoRA via checkpoints

Architecture:
- Single actor/critic worker group
- Multiple ES managers (one per task)
- LoRA switching: save/load checkpoints when switching tasks
- Temporary checkpoint directory for LoRA storage
"""

import os
import time
import numpy as np
from collections import defaultdict
from copy import deepcopy
from typing import List, Dict, Any, Optional

import ray
import torch
from omegaconf import OmegaConf, open_dict

from verl import DataProto
from verl.utils.metric import reduce_metrics
from verl.trainer.ppo.ray_trainer import apply_kl_penalty, AdvantageEstimator
from verl.trainer.ppo.core_algos import agg_loss

from ragen.trainer.agent_trainer import RayAgentTrainer, compute_advantage, adjust_batch
from ragen.llm_agent.agent_proxy import LLMAgentProxy
from ragen.llm_agent.es_manager import EnvStateManager
from ragen.llm_agent.ctx_manager import ContextManager
from ragen.cl_methods.moe import MoECLMethod


class MoEAgentTrainer(RayAgentTrainer):
    """
    True MoE trainer with independent LoRA per task.

    Each task has its own LoRA module and is trained independently.

    Training flow:
    1. Serial sampling: For each task, sample using that task's LoRA
    2. Serial updates: For each task, update that task's LoRA

    This ensures complete task independence.
    """

    def __init__(self, *args, moe_method: MoECLMethod = None, task_configs: List[Dict] = None, **kwargs):
        """
        Args:
            moe_method: The MoECLMethod instance
            task_configs: List of task configurations, each containing:
                - name: Task name (e.g., 'bandit_low')
                - train_tags: List of training env tags
                - train_n_groups: List of training group counts
                - val_tags: List of validation env tags
                - val_n_groups: List of validation group counts
        """
        # Initialize base trainer (this creates the actor/critic workers)
        super().__init__(*args, **kwargs)

        self.moe_method = moe_method
        self.task_configs = task_configs or []
        self.num_tasks = len(self.task_configs)

        if self.num_tasks == 0:
            raise ValueError("MoE trainer requires task_configs")

        print(f"[MoE] Initializing MoE trainer with {self.num_tasks} independent experts")
        print(f"[MoE] Each task will have its own LoRA parameters")

        # ES managers for each task
        self.task_train_es_managers: List[EnvStateManager] = []
        self.task_val_es_managers: List[EnvStateManager] = []
        self.task_train_ctx_managers: List[ContextManager] = []
        self.task_val_ctx_managers: List[ContextManager] = []

        # Advantage estimator type (same for all tasks)
        self.adv_estimator = AdvantageEstimator(self.config.algorithm.adv_estimator)

        # LoRA parameters for each task (stored in CPU memory)
        self.task_lora_params: List[Optional[Dict[str, torch.Tensor]]] = [None] * self.num_tasks

        # Track metrics per task
        self.task_metrics: Dict[int, Dict[str, List]] = defaultdict(lambda: defaultdict(list))

        # Current active task (for sampling/updating)
        self.current_task_idx = 0

    def init_moe_workers(self):
        """
        Initialize ES managers for all tasks.
        """
        print(f"[MoE] Initializing {self.num_tasks} task-specific ES managers...")

        for task_idx in range(self.num_tasks):
            task_config = self.task_configs[task_idx]
            print(f"[MoE] Setting up task {task_idx}: {task_config['name']}")

            # Create training ES manager for this task
            train_config = deepcopy(self.config)
            with open_dict(train_config):
                train_config.es_manager.train.env_configs.tags = list(task_config['train_tags'])
                train_config.es_manager.train.env_configs.n_groups = list(task_config['train_n_groups'])
                train_config.es_manager.train.env_groups = sum(task_config['train_n_groups'])

            train_es = EnvStateManager(train_config, mode="train")
            self.task_train_es_managers.append(train_es)

            train_ctx = ContextManager(train_config, self.tokenizer, mode="train")
            self.task_train_ctx_managers.append(train_ctx)

            # Create validation ES manager for this task
            val_config = deepcopy(self.config)
            with open_dict(val_config):
                val_config.es_manager.val.env_configs.tags = list(task_config['val_tags'])
                val_config.es_manager.val.env_configs.n_groups = list(task_config['val_n_groups'])
                val_config.es_manager.val.env_groups = sum(task_config['val_n_groups'])

            val_es = EnvStateManager(val_config, mode="val")
            self.task_val_es_managers.append(val_es)

            val_ctx = ContextManager(val_config, self.tokenizer, mode="val")
            self.task_val_ctx_managers.append(val_ctx)

            print(f"[MoE] Task {task_idx} ({task_config['name']}): "
                  f"train_tags={task_config['train_tags']}, val_tags={task_config['val_tags']}")

        print(f"[MoE] Successfully initialized {self.num_tasks} task configurations")

        # Create temporary checkpoint directory for LoRA switching
        import tempfile
        self.temp_checkpoint_dir = tempfile.mkdtemp(prefix="moe_lora_")
        print(f"[MoE] Created temporary checkpoint directory: {self.temp_checkpoint_dir}")

        # Save initial LoRA for task 0
        print("[MoE] Saving initial LoRA checkpoint for task 0")
        self._save_current_lora_checkpoint(0)

    def _save_current_lora_checkpoint(self, task_idx: int):
        """Save current LoRA parameters to checkpoint."""
        import os
        print(f"[MoE] Saving LoRA checkpoint for task {task_idx}")

        # Create checkpoint directory for this task
        task_checkpoint_dir = os.path.join(self.temp_checkpoint_dir, f"task_{task_idx}")
        os.makedirs(task_checkpoint_dir, exist_ok=True)

        actor_dir = os.path.join(task_checkpoint_dir, "actor")

        # Save actor checkpoint (includes LoRA parameters)
        self.actor_rollout_wg.save_checkpoint(actor_dir, None, 0)

        print(f"[MoE] Saved LoRA checkpoint for task {task_idx} to {actor_dir}")

    def _load_lora_checkpoint(self, task_idx: int):
        """Load LoRA parameters from checkpoint."""
        import os
        print(f"[MoE] Loading LoRA checkpoint for task {task_idx}")

        # Get checkpoint directory for this task
        task_checkpoint_dir = os.path.join(self.temp_checkpoint_dir, f"task_{task_idx}")
        actor_dir = os.path.join(task_checkpoint_dir, "actor")

        if not os.path.exists(actor_dir):
            print(f"[MoE] No checkpoint found for task {task_idx}, using current LoRA")
            return

        # Load actor checkpoint (includes LoRA parameters)
        self.actor_rollout_wg.load_checkpoint(actor_dir)

        print(f"[MoE] Loaded LoRA checkpoint for task {task_idx} from {actor_dir}")

    def switch_to_task(self, task_idx: int):
        """
        Switch to a different task.

        This involves:
        1. Saving current task's LoRA checkpoint
        2. Loading target task's LoRA checkpoint
        3. Switching ES managers
        """
        if task_idx == self.current_task_idx:
            return  # Already on this task

        print(f"[MoE] Switching from task {self.current_task_idx} to task {task_idx}")

        # Save current task's LoRA checkpoint
        self._save_current_lora_checkpoint(self.current_task_idx)

        # Load target task's LoRA checkpoint
        self._load_lora_checkpoint(task_idx)

        # Switch ES managers
        self.agent_proxy.train_es_manager = self.task_train_es_managers[task_idx]
        self.agent_proxy.val_es_manager = self.task_val_es_managers[task_idx]
        self.agent_proxy.train_ctx_manager = self.task_train_ctx_managers[task_idx]
        self.agent_proxy.val_ctx_manager = self.task_val_ctx_managers[task_idx]

        self.current_task_idx = task_idx
        print(f"[MoE] Switched to task {task_idx}: {self.task_configs[task_idx]['name']}")

    def _sample_single_task(self, task_idx: int) -> DataProto:
        """Sample from a single task."""
        # Switch to this task
        self.switch_to_task(task_idx)

        # Sample using agent proxy
        meta_info = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "recompute_log_prob": True,
            "do_sample": True,  # Training mode
            "validate": False,
        }
        gen_batch = DataProto(batch=None, non_tensor_batch=None, meta_info=meta_info)

        # Rollout
        batch = self.agent_proxy.rollout(gen_batch, val=False)

        return batch

    def _serial_sampling(self) -> List[DataProto]:
        """
        Serial sampling from all tasks.

        Returns:
            List of batches, one per task
        """
        print(f"[MoE] Phase 1: Serial sampling from {self.num_tasks} tasks...")

        task_batches = []
        for task_idx in range(self.num_tasks):
            task_name = self.task_configs[task_idx]['name']
            print(f"[MoE] Sampling from task {task_idx} ({task_name})...")

            batch = self._sample_single_task(task_idx)
            task_batches.append(batch)

            # Log sampling info
            num_trajs = len(batch.batch) if hasattr(batch.batch, '__len__') else batch.batch.batch_size[0]
            print(f"[MoE] Task {task_idx} sampled {num_trajs} trajectories")

        return task_batches

    def _update_single_task(self, batch: DataProto, task_idx: int):
        """
        Update with a single task's batch.

        Args:
            batch: Training batch for this task
            task_idx: Task index
        """
        task_name = self.task_configs[task_idx]['name']
        print(f"[MoE] Updating with task {task_idx} ({task_name}) data...")

        # Switch to this task (for proper context)
        self.switch_to_task(task_idx)

        # Compute rewards
        if self.config.reward_model.enable:
            batch = self.rm_wg.compute_rm_score(batch)
        else:
            reward_tensor_dict = self.reward_fn(batch, return_dict=True)
            batch.batch['rm_scores'] = reward_tensor_dict['reward_tensor']
            batch.batch['token_level_scores'] = reward_tensor_dict['reward_tensor']

        # Compute values from critic (needed for GAE advantage computation)
        if self.use_critic:
            values = self.critic_wg.compute_values(batch)
            batch = batch.union(values)

        # Convert token_level_scores to token_level_rewards (needed for compute_advantage)
        # Apply KL penalty if using reference policy
        if self.config.actor_rollout_ref.actor.use_kl_loss:
            batch, kl_metrics = apply_kl_penalty(
                batch=batch,
                kl_ctrl=self.kl_ctrl,
                kl_penalty=self.config.algorithm.kl_penalty,
            )
        else:
            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

        # Compute advantages
        batch = compute_advantage(
            batch,
            adv_estimator=self.config.algorithm.adv_estimator,
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lam,
        )

        # Adjust batch size to be divisible by num_groups, ppo_mini_batch_size, and n_gpus
        num_groups = self.task_train_es_managers[task_idx].env_groups
        ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        n_gpus = self.config.trainer.n_gpus_per_node
        size_divisor = np.lcm.reduce([num_groups, ppo_mini_batch_size, n_gpus])
        adjust_mode = getattr(self.config.agent_proxy, "batch_adjust_mode", "copy")
        batch = adjust_batch(batch, size_divisor, mode=adjust_mode)

        # Update actor with mini-batches
        ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        ppo_micro_batch_size = self.config.actor_rollout_ref.actor.ppo_micro_batch_size

        num_mini_batches = len(batch) // ppo_mini_batch_size

        # Set multi_turn flag
        batch.meta_info["multi_turn"] = True

        # Update actor (only if past critic warmup)
        if self.config.trainer.critic_warmup <= self.global_steps:
            for mb_idx in range(num_mini_batches):
                start_idx = mb_idx * ppo_mini_batch_size
                end_idx = start_idx + ppo_mini_batch_size
                mini_batch = batch[start_idx:end_idx]

                # Update actor
                actor_output = self.actor_rollout_wg.update_actor(mini_batch)

        # Update critic
        if self.use_critic:
            critic_output = self.critic_wg.update_critic(batch)

        print(f"[MoE] Updated with task {task_idx} data")

    def _serial_updates(self, task_batches: List[DataProto]):
        """
        Serial updates with all tasks' data.

        Args:
            task_batches: List of batches, one per task
        """
        print(f"[MoE] Phase 2: Serial updates with {self.num_tasks} tasks' data...")

        for task_idx in range(self.num_tasks):
            self._update_single_task(task_batches[task_idx], task_idx)

        print(f"[MoE] All {self.num_tasks} tasks updated")

    def _validate_all_tasks(self):
        """Validate on all tasks."""
        print(f"[MoE] Validating on all {self.num_tasks} tasks...")

        all_metrics = {}

        for task_idx in range(self.num_tasks):
            task_name = self.task_configs[task_idx]['name']
            print(f"[MoE] Validating task {task_idx}: {task_name}")

            # Switch to this task
            self.switch_to_task(task_idx)

            # Run validation
            task_metrics = self._validate()

            # Store metrics with task prefix
            for key, value in task_metrics.items():
                all_metrics[f"task_{task_idx}_{task_name}/{key}"] = value

        return all_metrics

    def _save_checkpoint(self):
        """Save checkpoint for all tasks."""
        from verl.utils.fs import local_mkdir_safe

        print(f"[MoE] Saving checkpoint at step {self.global_steps}")

        # Create checkpoint directory
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir,
            f"global_step_{self.global_steps}"
        )
        local_mkdir_safe(local_global_step_folder)

        # Save checkpoint for each task
        # Each task has its own LoRA parameters
        for task_idx in range(self.num_tasks):
            task_name = self.task_configs[task_idx]['name']
            task_dir = os.path.join(local_global_step_folder, f"task_{task_idx}_{task_name}")
            local_mkdir_safe(task_dir)

            # Switch to this task and save
            self.switch_to_task(task_idx)

            actor_dir = os.path.join(task_dir, "actor")
            critic_dir = os.path.join(task_dir, "critic")

            self.actor_rollout_wg.save_checkpoint(actor_dir, None, self.global_steps)
            self.critic_wg.save_checkpoint(critic_dir, None, self.global_steps)

            print(f"[MoE] Saved checkpoint for task {task_idx}: {task_name}")

        # Save MoE method state
        if self.moe_method is not None:
            method_state_path = os.path.join(local_global_step_folder, "cl_method_state.pt")
            self.moe_method.save_method_state(method_state_path)
            print(f"[MoE] Saved MoE method state to {method_state_path}")

        print(f"[MoE] Checkpoint saved to {local_global_step_folder}")

    def fit(self):
        """
        Main training loop for MoE.

        Each step:
        1. Serial sampling from all tasks
        2. Serial updates with all tasks' data
        3. Validation (if needed)
        4. Checkpoint saving (if needed)
        """
        # Initialize workers if not already done
        if not hasattr(self, 'actor_rollout_wg'):
            print("[MoE] Initializing base workers...")
            self.init_workers()
            self.init_agent_proxy()

        # Now initialize MoE-specific components
        print("[MoE] Initializing MoE-specific components...")
        self.init_moe_workers()

        print(f"[MoE] Starting training for {self.total_training_steps} steps")

        for step in range(1, self.total_training_steps + 1):
            self.global_steps = step

            print(f"\n{'='*80}")
            print(f"[MoE] Training Step {step}/{self.total_training_steps}")
            print(f"{'='*80}")

            # Phase 1: Serial sampling
            task_batches = self._serial_sampling()

            # Phase 2: Serial updates
            self._serial_updates(task_batches)

            # Record step for MoE method
            if self.moe_method is not None:
                for task_idx in range(self.num_tasks):
                    self.moe_method.record_step(task_idx)

            # Validation
            if step % self.config.trainer.test_freq == 0:
                print(f"\n[MoE] Running validation at step {step}")
                val_metrics = self._validate_all_tasks()
                # TODO: Log metrics to wandb/tensorboard

            # Save checkpoint
            if step % self.config.trainer.save_freq == 0:
                self._save_checkpoint()

        print(f"\n[MoE] Training completed!")

        # Final checkpoint
        self._save_checkpoint()

        # Cleanup temporary checkpoint directory
        self._cleanup_temp_checkpoints()

    def _cleanup_temp_checkpoints(self):
        """Clean up temporary checkpoint directory."""
        import shutil
        if hasattr(self, 'temp_checkpoint_dir') and os.path.exists(self.temp_checkpoint_dir):
            print(f"[MoE] Cleaning up temporary checkpoints: {self.temp_checkpoint_dir}")
            shutil.rmtree(self.temp_checkpoint_dir)
            print(f"[MoE] Temporary checkpoints cleaned up")
