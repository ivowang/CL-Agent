"""
Mix Trainer for RAGEN

This trainer implements multi-task mixed training where the agent
trains on multiple environments simultaneously in each batch. Each
parallel sampling step includes samples from all environments mixed together.

Key features:
- All environments are mixed in each training batch
- Single shared LoRA module for all environments
- Validation on all environments at each validation step
- Proper per-environment metrics logging
"""

import os
import time
import numpy as np
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
from tqdm import tqdm
from typing import List, Dict, Any, Optional

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
from ragen.llm_agent.agent_proxy import LLMAgentProxy
from ragen.llm_agent.es_manager import EnvStateManager
from ragen.cl_methods.mix import MixCLMethod


class MixAgentTrainer(RayAgentTrainer):
    """
    Multi-task mixed trainer.
    
    This trainer samples from all environments simultaneously in each batch,
    mixing different environment samples together. All environments share
    the same LoRA parameters.
    """
    
    def __init__(self, *args, mix_method: MixCLMethod = None, env_configs: List[Dict] = None, **kwargs):
        """
        Args:
            mix_method: The MixCLMethod instance (used for tracking, not cycling)
            env_configs: List of environment configurations, each containing:
                - name: Environment name (e.g., 'bandit', 'sokoban', 'frozen_lake')
                - train_tags: List of training env tags
                - train_n_groups: List of training group counts
                - val_tags: List of validation env tags
                - val_n_groups: List of validation group counts
        """
        super().__init__(*args, **kwargs)
        self.mix_method = mix_method
        self.env_configs = env_configs or []
        
        # Validation ES managers (one per environment for separate validation)
        self.val_es_managers: Dict[str, EnvStateManager] = {}
        
        # Track metrics per environment
        self.env_metrics: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))
        
    def init_mix_environments(self):
        """Initialize ES managers: one mixed ES manager for training, separate ones for validation."""
        if not self.env_configs:
            print("[Mix] Warning: No environment configs provided")
            return
            
        print(f"[Mix] Initializing mixed training with {len(self.env_configs)} environments")
        
        # === Create a SINGLE mixed training ES manager with ALL environments ===
        all_train_tags = []
        all_train_n_groups = []
        
        for env_config in self.env_configs:
            all_train_tags.extend(env_config['train_tags'])
            all_train_n_groups.extend(env_config['train_n_groups'])
        
        # Create mixed training config
        mixed_train_config = deepcopy(self.config)
        with open_dict(mixed_train_config):
            mixed_train_config.es_manager.train.env_configs.tags = all_train_tags
            mixed_train_config.es_manager.train.env_configs.n_groups = all_train_n_groups
            mixed_train_config.es_manager.train.env_groups = sum(all_train_n_groups)
        
        # Create the single mixed training ES manager
        self.mixed_train_es_manager = EnvStateManager(mixed_train_config, mode="train")
        print(f"[Mix] Created MIXED training ES manager with tags={all_train_tags}, n_groups={all_train_n_groups}")
        print(f"[Mix] Total training env instances: {sum(all_train_n_groups)} groups × {mixed_train_config.es_manager.train.group_size} = {sum(all_train_n_groups) * mixed_train_config.es_manager.train.group_size}")
        
        # Update agent proxy to use the mixed ES manager
        self.agent_proxy.train_es_manager = self.mixed_train_es_manager
        
        # Update ctx_manager's env_nums for proper metric normalization
        new_env_nums = {}
        for n_group, env_tag in zip(all_train_n_groups, all_train_tags):
            if env_tag in new_env_nums:
                new_env_nums[env_tag] += n_group * self.mixed_train_es_manager.group_size
            else:
                new_env_nums[env_tag] = n_group * self.mixed_train_es_manager.group_size
        self.agent_proxy.train_ctx_manager.env_nums = new_env_nums
        print(f"[Mix] Updated env_nums for training: {new_env_nums}")
        
        # === Create SEPARATE validation ES managers for each environment ===
        for env_config in self.env_configs:
            env_name = env_config['name']
            
            val_config = deepcopy(self.config)
            with open_dict(val_config):
                val_config.es_manager.val.env_configs.tags = list(env_config['val_tags'])
                val_config.es_manager.val.env_configs.n_groups = list(env_config['val_n_groups'])
                val_config.es_manager.val.env_groups = sum(env_config['val_n_groups'])
            
            val_es = EnvStateManager(val_config, mode="val")
            self.val_es_managers[env_name] = val_es
            print(f"[Mix] Created validation ES manager for {env_name}: tags={env_config['val_tags']}")
    
    def _validate_all_envs(self) -> Dict[str, Any]:
        """Validate on all environments separately and return combined metrics."""
        from ragen.llm_agent.ctx_manager import ContextManager
        
        all_metrics = {}
        
        for env_name, val_es_manager in self.val_es_managers.items():
            print(f"[Mix] Validating on environment: {env_name}")
            
            # Store original ES manager and ctx_manager
            original_val_es = self.agent_proxy.val_es_manager
            original_val_ctx = self.agent_proxy.val_ctx_manager
            
            # Temporarily switch to this environment's validation ES manager
            self.agent_proxy.val_es_manager = val_es_manager
            
            # CRITICAL FIX: Create a new ContextManager for this environment
            # The old ctx_manager's prefix_lookup maps env_ids to the WRONG environment instructions
            # We need a fresh ctx_manager that knows about this specific environment
            self.agent_proxy.val_ctx_manager = ContextManager(
                val_es_manager.sys_config,  # Use the ES manager's full config
                self.tokenizer,
                mode="val"
            )
            
            try:
                # Run validation
                env_metrics = self._validate_single_env(env_name)
                
                # Add environment prefix to metrics
                for key, value in env_metrics.items():
                    prefixed_key = f"mix_{env_name}/{key}"
                    all_metrics[prefixed_key] = value
                    
            finally:
                # Restore original ES manager and ctx_manager
                self.agent_proxy.val_es_manager = original_val_es
                self.agent_proxy.val_ctx_manager = original_val_ctx
        
        return all_metrics
    
    def _validate_single_env(self, env_name: str) -> Dict[str, Any]:
        """Validate on a single environment."""
        data_source_lst = []
        reward_extra_infos_dict = defaultdict(list)
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        env_metric_dict = {}
        
        for step in range(self.config.trainer.validation_steps):
            meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            test_gen_batch = DataProto(batch=None, non_tensor_batch=None, meta_info=meta_info)
            
            test_batch = self.agent_proxy.rollout(test_gen_batch, val=True)
            
            for key, value in test_batch.meta_info["metrics"].items():
                metric_key = f"val-env/{key}"
                if metric_key not in env_metric_dict:
                    env_metric_dict[metric_key] = []
                env_metric_dict[metric_key].append(value)
            
            # Process batch outputs
            batch_size = test_batch.batch["input_ids"].shape[0]
            output_ids = test_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            
            input_texts = ["" for _ in range(batch_size)]
            sample_inputs.extend(input_texts)
            sample_outputs.extend(output_texts)
            
            # Compute rewards
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)
            reward_extra_infos_dict["reward"].extend(scores)
            
            data_sources_batch = test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0])
            data_source_lst.append(data_sources_batch)
        
        # Process metrics
        data_sources = np.concatenate(data_source_lst, axis=0)
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = reduce_metrics(env_metric_dict)
        
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (var_name == core_var) and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]) and (f"@{n_max}" in metric_name):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val
        
        return metric_dict
    
    def fit(self):
        """
        Main training loop with mixed-environment sampling.
        
        Each training step samples from ALL environments mixed together,
        then uses the combined batch to update the shared LoRA parameters.
        """
        self.global_steps = self.config.trainer.get("start_step", 0)
        self.start_time = time.time()
        
        # Initialize multi-environment setup (single mixed ES manager)
        self.init_mix_environments()
        
        progress_bar = tqdm(
            range(self.global_steps, self.total_training_steps),
            desc="Training (Mixed)",
            initial=self.global_steps,
            total=self.total_training_steps,
        )
        
        # Initialize logger once for the entire training
        from verl.utils.tracking import Tracking
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
        )
        
        # Pre-training validation if configured
        if self.config.trainer.val_before_train and self.val_reward_fn is not None:
            print("[Mix] Running pre-training validation on all environments...")
            val_metrics = self._validate_all_envs()
            logger.log(data=val_metrics, step=self.global_steps)
        
        last_val_metrics = {}
        
        # Get environment names for logging
        env_names = [env_config['name'] for env_config in self.env_configs]
        print(f"[Mix] Training with mixed environments: {env_names}")
        
        for step in progress_bar:
            is_last_step = step == self.total_training_steps - 1
            timing_raw = {}
            metrics = {}
            
            # Wrap entire step in timing context (required by compute_throughout_metrics)
            with marked_timer("step", timing_raw):
                # Log that this is a mixed batch
                metrics["mix/batch_type"] = "mixed"
                metrics["mix/num_environments"] = len(self.env_configs)
                
                # === Training Step (mixed batch from all environments) ===
                with marked_timer("gen", timing_raw, color="green"):
                    # Rollout on ALL environments mixed together
                    meta_info = {
                        "eos_token_id": self.tokenizer.eos_token_id,
                        "pad_token_id": self.tokenizer.pad_token_id,
                        "recompute_log_prob": False,
                        "do_sample": True,
                        "temperature": self.config.actor_rollout_ref.rollout.temperature,
                    }
                    gen_batch = DataProto(batch=None, non_tensor_batch=None, meta_info=meta_info)
                    batch = self.agent_proxy.rollout(gen_batch, val=False)
                    
                    # Log training metrics (now includes all environments)
                    if "metrics" in batch.meta_info:
                        for key, value in batch.meta_info["metrics"].items():
                            # Log with "train_mixed" prefix for combined metrics
                            metrics[f"train_mixed/{key}"] = value
                
                # Prepare batch for training
                batch.non_tensor_batch["uid"] = batch.non_tensor_batch["group_ids"]
                batch.batch["response_mask"] = batch.batch["loss_mask"]
                
                if self.config.trainer.balance_batch:
                    self._balance_batch(batch, metrics=metrics)
                
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                
                # Compute rewards
                if self.use_rm:
                    with marked_timer("reward", timing_raw):
                        reward_tensor = self.rm_wg.compute_rm_score(batch)
                        batch = batch.union(reward_tensor)
                
                if self.config.reward_model.launch_reward_fn_async:
                    future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                else:
                    reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                
                # Compute old log probs
                with marked_timer("old_log_prob", timing_raw, color="blue"):
                    old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                    entropys = old_log_prob.batch["entropys"]
                    response_masks = batch.batch["response_mask"]
                    loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                    entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                    metrics["actor/entropy"] = entropy_agg.detach().item()
                    old_log_prob.batch.pop("entropys")
                    batch = batch.union(old_log_prob)
                
                # Reference policy log probs
                if self.use_reference_policy:
                    with marked_timer("ref", timing_raw, color="olive"):
                        if not self.ref_in_actor:
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                        else:
                            ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                        batch = batch.union(ref_log_prob)
                        from verl.utils.torch_functional import masked_mean
                        avg_ref_log_prob = masked_mean(ref_log_prob.batch["ref_log_prob"], batch.batch["response_mask"])
                        metrics["rollout/ref_log_prob"] = avg_ref_log_prob
                
                # Compute values
                if self.use_critic:
                    with marked_timer("values", timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)
                
                # Compute advantages
                with marked_timer("adv", timing_raw):
                    if self.config.reward_model.launch_reward_fn_async:
                        reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                    batch.batch["token_level_scores"] = reward_tensor
                    
                    if reward_extra_infos_dict:
                        batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                    
                    if self.config.algorithm.use_kl_in_reward:
                        batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty, multi_turn=True)
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
                
                # Update critic
                if self.use_critic:
                    with marked_timer("update_critic", timing_raw, color="pink"):
                        critic_output = self.critic_wg.update_critic(batch)
                    critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                    metrics.update(critic_output_metrics)
                
                # Update actor
                if self.config.trainer.critic_warmup <= self.global_steps:
                    with marked_timer("update_actor", timing_raw):
                        batch.meta_info["multi_turn"] = True
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)
                
                # === Validation ===
                if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                    with marked_timer("testing", timing_raw):
                        val_metrics = self._validate_all_envs()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)
                
                # === Checkpoint ===
                if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                    with marked_timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()
                
                # No environment switching needed - we sample from all environments each step
            
            # Collect and log metrics (outside of step timer)
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
            metrics["timing_s/total"] = time.time() - self.start_time
            
            # Log metrics using the pre-initialized logger
            logger.log(data=metrics, step=self.global_steps)
            
            if is_last_step:
                pprint(f"Final validation metrics: {last_val_metrics}")
                progress_bar.close()
                return
            
            progress_bar.update(1)
            self.global_steps += 1
