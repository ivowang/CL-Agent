"""
Continual Learning Agent Trainer for RAGEN

This module extends the RayAgentTrainer to support validation on multiple 
environments (current + previously learned tasks) for continual learning scenarios.
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
from ragen.llm_agent.es_manager import EnvStateManager


class ContinualLearningAgentTrainer(RayAgentTrainer):
    """
    Extended Agent Trainer for Continual Learning.
    Supports validation on multiple environments (current + previously learned).
    Also handles CL-specific configurations for methods like O-LoRA.
    """
    
    def __init__(self, *args, cl_val_env_configs=None, current_task_idx=0, 
                 cl_method_config=None, cl_method_state=None, **kwargs):
        """
        Args:
            cl_val_env_configs: List of dicts with environment configs for multi-task validation.
                Each dict should have: {'name': str, 'tags': list, 'n_groups': list}
            current_task_idx: Index of the current task being trained (0-indexed)
            cl_method_config: CL method configuration dict with:
                - name: Method name ('naive', 'olora', etc.)
                - lambda_ortho: Weight for orthogonal loss (O-LoRA)
                - lambda_l2: Weight for L2 regularization (O-LoRA)
            cl_method_state: State from previous tasks (e.g., frozen LoRA params)
        """
        super().__init__(*args, **kwargs)
        self.cl_val_env_configs = cl_val_env_configs or []
        self.current_task_idx = current_task_idx
        self.cl_val_es_managers = []
        self.cl_method_config = cl_method_config or {'name': 'naive'}
        self.cl_method_state = cl_method_state or {}
        
    def init_cl_validation(self):
        """Initialize ES managers for validation on all previous and current tasks."""
        if not self.cl_val_env_configs:
            print("[CL] No multi-task validation configs provided, skipping CL validation init")
            return
            
        print(f"[CL] Initializing multi-task validation for tasks 0 to {self.current_task_idx}")
        self.cl_val_es_managers = []
        
        for task_idx, env_config in enumerate(self.cl_val_env_configs):
            if task_idx > self.current_task_idx:
                break  # Only validate on current and previous tasks
                
            # Create a modified config for validation on this task
            val_config = deepcopy(self.config)
            with open_dict(val_config):
                val_config.es_manager.val.env_configs.tags = env_config['tags']
                val_config.es_manager.val.env_configs.n_groups = env_config['n_groups']
                # Adjust env_groups to match n_groups sum
                val_config.es_manager.val.env_groups = sum(env_config['n_groups'])
            
            val_es_manager = EnvStateManager(val_config, mode="val")
            self.cl_val_es_managers.append({
                'es_manager': val_es_manager,
                'task_name': env_config['name'],
                'task_idx': task_idx
            })
            print(f"[CL] Created validation ES manager for task {task_idx}: {env_config['name']}")
    
    def set_cl_config_on_workers(self):
        """
        Set CL configuration on all actor workers.
        This is called after workers are initialized and before training starts.
        """
        method_name = self.cl_method_config.get('name', 'naive')
        print(f"[CL] Setting CL config on workers: method={method_name}, task_idx={self.current_task_idx}")
        
        # Build CL config for workers
        cl_config = {
            'method': method_name,
            'current_task_idx': self.current_task_idx,
            'lambda_ortho': self.cl_method_config.get('lambda_ortho', 0.5),
            'lambda_l2': self.cl_method_config.get('lambda_l2', 0.0),
            # Load frozen params from state if available
            'frozen_lora_params': self.cl_method_state.get('frozen_lora_params', None),
        }
        
        # Set config on all actor rollout workers
        if hasattr(self, 'actor_rollout_wg') and self.actor_rollout_wg is not None:
            try:
                # Call set_cl_config on all workers
                # Note: RayWorkerGroup's execute_all_sync doesn't require underscore prefix
                workers = self.actor_rollout_wg._workers if hasattr(self.actor_rollout_wg, '_workers') else []
                num_workers = len(workers) if workers else 0
                
                # Execute on all workers using ray
                import ray
                results = []
                for worker in workers:
                    try:
                        # Each worker is a ray actor
                        result = ray.get(worker.set_cl_config.remote(cl_config))
                        results.append(result)
                    except AttributeError:
                        # Worker doesn't have set_cl_config method, skip
                        pass
                
                print(f"[CL] Successfully set CL config on {len(results)}/{num_workers} workers")
            except Exception as e:
                print(f"[CL] Warning: Could not set CL config on workers: {e}")
                print("[CL] This may be expected if workers don't support CL methods")
    
    def _validate_all_tasks(self):
        """
        Validate on all tasks (current + previous).
        Returns combined metrics with task-specific prefixes.
        """
        if not self.cl_val_es_managers:
            return {}
            
        all_metrics = {}
        
        for task_info in self.cl_val_es_managers:
            task_name = task_info['task_name']
            task_idx = task_info['task_idx']
            es_manager = task_info['es_manager']
            
            print(f"[CL] Validating on task {task_idx}: {task_name}")
            
            task_metrics = self._validate_single_task(es_manager, task_name)
            
            # Add task-specific prefix to metrics
            for key, value in task_metrics.items():
                prefixed_key = f"cl_task{task_idx}_{task_name}/{key}"
                all_metrics[prefixed_key] = value
        
        return all_metrics
    
    def _validate_single_task(self, es_manager, task_name):
        """Validate on a single task using the given ES manager."""
        data_source_lst = []
        reward_extra_infos_dict = defaultdict(list)
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        env_metric_dict = {}
        
        # Store original ES managers and ctx_manager env_nums
        original_val_es = self.agent_proxy.val_es_manager
        original_env_nums = self.agent_proxy.val_ctx_manager.env_nums.copy()
        
        # Temporarily replace with task-specific ES manager
        self.agent_proxy.val_es_manager = es_manager
        
        # Also update the ctx_manager's env_nums to match the new ES manager
        # This is crucial for correct metric normalization in _compute_metrics
        # Note: es_manager.config is already the es_manager.val sub-config (set in EnvStateManager.__init__)
        new_env_nums = {}
        es_cfg = es_manager.config
        for n_group, env_tag in zip(es_cfg.env_configs.n_groups, es_cfg.env_configs.tags):
            new_env_nums[env_tag] = n_group * es_cfg.group_size
        self.agent_proxy.val_ctx_manager.env_nums = new_env_nums
        
        try:
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
                
        finally:
            # Restore original ES manager and env_nums
            self.agent_proxy.val_es_manager = original_val_es
            self.agent_proxy.val_ctx_manager.env_nums = original_env_nums
        
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
        The training loop with continual learning support.
        Extends the original fit() with multi-task validation.
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
        
        # Track the starting step for this task
        start_step = self.global_steps

        # Perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            # Multi-task validation
            val_metrics = self._validate_all_tasks()
            # Standard validation for current task
            val_metrics.update(self._validate())
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, 
                          desc=f"CL Training Task {self.current_task_idx}")

        self.global_steps += 1
        last_val_metrics = None

        def _process_batch_for_logging(batch):
            inputs_raw = batch.batch["input_ids"]
            inputs = [self.tokenizer.decode(input_ids, skip_special_tokens=True) for input_ids in inputs_raw]
            outputs = [""] * len(inputs)
            scores = batch.batch["rm_scores"].sum(-1).cpu().tolist()
            return inputs, outputs, scores

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
                        # Validate on all tasks (CL validation)
                        val_metrics = self._validate_all_tasks()
                        # Also run standard validation for current task
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
            })

            logger.log(data=metrics, step=self.global_steps)

            if is_last_step:
                pprint(f"[CL Task {self.current_task_idx}] Final validation metrics: {last_val_metrics}")
                progress_bar.close()
                return

            progress_bar.update(1)
            self.global_steps += 1

