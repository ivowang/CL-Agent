"""
Continual Learning Trainer with Pluggable CL Methods

This trainer extends the CL trainer to support different CL algorithms
(e.g., Naive, O-LoRA) through a pluggable method framework.
"""

import os
import time
import numpy as np
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
from tqdm import tqdm

import ray
import torch
from omegaconf import OmegaConf, open_dict

from verl import DataProto
from verl.utils.metric import reduce_metrics
from verl.utils.debug import marked_timer
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.ray_trainer import apply_kl_penalty, AdvantageEstimator
from verl.trainer.ppo.core_algos import agg_loss

from ragen.trainer.agent_trainer import RayAgentTrainer, compute_advantage, adjust_batch
from ragen.trainer.cl_agent_trainer import ContinualLearningAgentTrainer
from ragen.llm_agent.es_manager import EnvStateManager
from ragen.cl_methods import get_cl_method, BaseCLMethod
from ragen.cl_methods.base import CLMethodConfig
from ragen.cl_methods.olora import OLoRAConfig


class CLTrainerWithMethod(ContinualLearningAgentTrainer):
    """
    Extended CL Trainer that integrates pluggable CL methods.
    
    This trainer:
    - Supports different CL methods (naive, olora, etc.)
    - Applies CL-specific losses during training
    - Handles task transitions according to the CL method
    """
    
    def __init__(self, *args, cl_method_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize CL method
        self.cl_method_config = cl_method_config or {}
        self.cl_method = self._init_cl_method()
        
    def _init_cl_method(self) -> BaseCLMethod:
        """Initialize the CL method based on config."""
        method_name = self.cl_method_config.get('name', 'naive')
        
        # Build method-specific config
        if method_name == 'olora':
            config = OLoRAConfig(
                name='olora',
                current_task_idx=self.current_task_idx,
                lambda_ortho=self.cl_method_config.get('lambda_ortho', 0.5),
                lambda_l2=self.cl_method_config.get('lambda_l2', 0.0),
                lora_rank_per_task=self.cl_method_config.get('lora_rank', 64),
                checkpoint_base_dir=self.config.trainer.default_local_dir,
            )
        else:
            config = CLMethodConfig(
                name=method_name,
                current_task_idx=self.current_task_idx,
                checkpoint_base_dir=self.config.trainer.default_local_dir,
            )
        
        # Get and instantiate the method
        method_cls = get_cl_method(method_name)
        method = method_cls(config)
        
        print(f"[CLTrainerWithMethod] Initialized CL method: {method_name}")
        return method
    
    def init_cl_method_for_task(self, task_idx: int, task_name: str):
        """Initialize CL method for a new task."""
        self.cl_method.current_task_idx = task_idx
        
        # Load previous task state if available
        if task_idx > 0:
            prev_task_idx = task_idx - 1
            prev_task_name = self.cl_val_env_configs[prev_task_idx]['name'] if self.cl_val_env_configs else f"task{prev_task_idx}"
            state = self.cl_method.load_task_state(prev_task_idx, prev_task_name)
            if state:
                print(f"[CLTrainerWithMethod] Loaded state from task {prev_task_idx}")
        
        # Call on_task_start
        # Note: We pass None for model here since the actual model is in the worker
        self.cl_method.on_task_start(None, task_idx, task_name)
    
    def finalize_task(self, task_idx: int, task_name: str):
        """Finalize CL method after a task."""
        self.cl_method.on_task_end(None, task_idx, task_name)
        self.cl_method.save_task_state(None, task_idx, task_name)
    
    def _compute_cl_loss_metrics(self, base_metrics: dict) -> dict:
        """
        Compute CL-specific loss and add to metrics.
        Note: The actual CL loss computation happens in the worker.
        This method prepares the CL config to be passed to workers.
        """
        cl_info = self.cl_method.get_method_info()
        
        # Add CL method info to metrics
        metrics = {
            f"cl/{k}": v for k, v in cl_info.items() 
            if isinstance(v, (int, float))
        }
        
        return metrics
    
    def fit(self):
        """
        Training loop with CL method integration.
        """
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()
        
        start_step = self.global_steps

        # Initialize CL method for current task
        task_name = self.cl_val_env_configs[self.current_task_idx]['name'] if self.cl_val_env_configs else f"task{self.current_task_idx}"
        self.init_cl_method_for_task(self.current_task_idx, task_name)

        # Perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate_all_tasks()
            val_metrics.update(self._validate())
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, 
                          desc=f"CL Training Task {self.current_task_idx} ({self.cl_method.name})")

        self.global_steps += 1
        last_val_metrics = None

        self.start_time = time.time()
        
        for step in range(self.total_training_steps):
            timing_raw = {}
            batch = DataProto()
            is_last_step = self.global_steps >= self.total_training_steps

            with marked_timer("step", timing_raw):
                # Generate batch
                with marked_timer("gen", timing_raw):
                    batch = self.agent_proxy.rollout(batch, val=False)
                    batch, metrics = self.rollout_filter.filter(batch)

                    num_groups = self.config.es_manager.train.env_groups
                    ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
                    n_gpus = self.config.trainer.n_gpus_per_node
                    size_divisor = np.lcm.reduce([num_groups, ppo_mini_batch_size, n_gpus])
                    adjust_mode = getattr(self.config.agent_proxy, "batch_adjust_mode", "copy")
                    batch = adjust_batch(batch, size_divisor, mode=adjust_mode)

                    batch_size = batch.batch["input_ids"].shape[0]
                    num_mini_batches = batch_size // ppo_mini_batch_size
                    metrics.update({
                        "train/batch_size": batch_size,
                        "train/num_mini_batches": num_mini_batches,
                    })
                    metrics.update({"train/" + key: value for key, value in batch.meta_info["metrics"].items()})

                if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                    print("[NotImplemented] REMAX implementation is not tested yet in RAGEN. Exiting.")
                    exit()

                batch.non_tensor_batch["uid"] = batch.non_tensor_batch["group_ids"]
                batch.batch["response_mask"] = batch.batch["loss_mask"]
                
                if self.config.trainer.balance_batch:
                    self._balance_batch(batch, metrics=metrics)

                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                
                # Add CL method info to batch meta for worker
                batch.meta_info["cl_method"] = {
                    'name': self.cl_method.name,
                    'current_task_idx': self.current_task_idx,
                    'config': self.cl_method_config,
                }

                if self.use_rm:
                    with marked_timer("reward", timing_raw):
                        reward_tensor = self.rm_wg.compute_rm_score(batch)
                        batch = batch.union(reward_tensor)

                if self.config.reward_model.launch_reward_fn_async:
                    future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                else:
                    reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                with marked_timer("old_log_prob", timing_raw, color="blue"):
                    old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                    entropys = old_log_prob.batch["entropys"]
                    response_masks = batch.batch["response_mask"]
                    loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                    entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                    old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                    metrics.update(old_log_prob_metrics)
                    old_log_prob.batch.pop("entropys")
                    batch = batch.union(old_log_prob)

                if self.use_reference_policy:
                    with marked_timer("ref", timing_raw, color="olive"):
                        if not self.ref_in_actor:
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                        else:
                            ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                        batch = batch.union(ref_log_prob)
                        from verl.utils.torch_functional import masked_mean
                        avg_ref_log_prob = masked_mean(ref_log_prob.batch["ref_log_prob"], batch.batch["response_mask"])
                        metrics.update({"rollout/ref_log_prob": avg_ref_log_prob})

                if self.use_critic:
                    with marked_timer("values", timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                with marked_timer("adv", timing_raw):
                    if self.config.reward_model.launch_reward_fn_async:
                        reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                    batch.batch["token_level_scores"] = reward_tensor

                    if reward_extra_infos_dict:
                        batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                    if self.config.algorithm.use_kl_in_reward:
                        batch, kl_metrics = apply_kl_penalty(
                            batch, kl_ctrl=self.kl_ctrl_in_reward, 
                            kl_penalty=self.config.algorithm.kl_penalty, multi_turn=True
                        )
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)

                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                        num_repeat=self.config.actor_rollout_ref.rollout.n,
                        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        multi_turn=True,
                        high_level_gamma=self.config.algorithm.high_level_gamma,
                        bi_level_gae=self.config.algorithm.bi_level_gae,
                    )

                if self.use_critic:
                    with marked_timer("update_critic", timing_raw, color="pink"):
                        critic_output = self.critic_wg.update_critic(batch)
                    critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                    metrics.update(critic_output_metrics)

                if self.config.trainer.critic_warmup <= self.global_steps:
                    with marked_timer("update_actor", timing_raw):
                        batch.meta_info["multi_turn"] = True
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)

                # Multi-task validation
                if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                   (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                    with marked_timer("testing", timing_raw):
                        val_metrics = self._validate_all_tasks()
                        val_metrics.update(self._validate())
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and \
                   (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                    with marked_timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            # Collect metrics
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
            metrics.update({"timing_s/total": time.time() - self.start_time})
            
            # Add CL specific metrics
            metrics.update({
                "cl/current_task_idx": self.current_task_idx,
                "cl/task_local_step": self.global_steps - start_step,
                "cl/method": self.cl_method.name,
            })
            metrics.update(self._compute_cl_loss_metrics(metrics))

            logger.log(data=metrics, step=self.global_steps)

            if is_last_step:
                pprint(f"[CL Task {self.current_task_idx}] Final validation metrics: {last_val_metrics}")
                # Finalize CL method for this task
                self.finalize_task(self.current_task_idx, task_name)
                progress_bar.close()
                return

            progress_bar.update(1)
            self.global_steps += 1

