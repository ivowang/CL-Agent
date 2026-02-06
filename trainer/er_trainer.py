"""
Experience Replay Trainer for RAGEN (In-Context Memory Only).

This trainer collects successful rollouts and injects them into prompts
as in-context examples. No model parameters are updated.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Dict, List, Any, Tuple

import numpy as np
from omegaconf import OmegaConf
from verl import DataProto
from verl.utils.tracking import Tracking
from verl.single_controller.ray.base import RayWorkerGroup

from llm_agent.agent_proxy import LLMAgentProxy, VllmWrapperWg
from llm_agent.es_manager import EnvStateManager
from llm_agent.ctx_manager import ContextManager
from cl_methods.experience_replay import ExperienceReplayMethod, ERConfig, Experience


class ERContextManager(ContextManager):
    """ContextManager that prepends replay examples to the system prompt."""

    def __init__(self, config, tokenizer, processor=None, mode: str = "train"):
        super().__init__(config, tokenizer, processor, mode)
        self._examples_text = ""

    def set_examples_text(self, examples_text: str):
        self._examples_text = examples_text

    def _build_system_content(self, env_id: int) -> str:
        base_content = self.prefix_lookup.get(env_id, "")
        if self._examples_text:
            return self._examples_text + base_content
        return base_content


class ERAgentProxy(LLMAgentProxy):
    """AgentProxy using ERContextManager."""

    def __init__(self, config, actor_rollout_wg, tokenizer):
        self.config = config
        self.train_ctx_manager = ERContextManager(config, tokenizer, mode="train")
        self.train_es_manager = EnvStateManager(config, mode="train")
        self.val_ctx_manager = ERContextManager(config, tokenizer, mode="val")
        self.val_es_manager = EnvStateManager(config, mode="val")
        self.actor_wg = actor_rollout_wg
        self.tokenizer = tokenizer
        self._last_padded_inputs = None

    def set_examples_text(self, examples_text: str, for_val: bool = False):
        if for_val:
            self.val_ctx_manager.set_examples_text(examples_text)
        else:
            self.train_ctx_manager.set_examples_text(examples_text)

    def generate_sequences(self, lm_inputs: DataProto):
        if isinstance(self.actor_wg, RayWorkerGroup):
            from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
            padded_lm_inputs, pad_size = pad_dataproto_to_divisor(
                lm_inputs, self.actor_wg.world_size
            )
            self._last_padded_inputs = padded_lm_inputs
            padded_lm_outputs = self.actor_wg.generate_sequences(padded_lm_inputs)
            if lm_inputs.meta_info.get("skip_generation", False):
                return lm_inputs
            lm_outputs = unpad_dataproto(padded_lm_outputs, pad_size=pad_size)
            lm_outputs.meta_info = lm_inputs.meta_info
            lm_outputs.non_tensor_batch = lm_inputs.non_tensor_batch
        elif isinstance(self.actor_wg, VllmWrapperWg):
            lm_outputs = self.actor_wg.generate_sequences(lm_inputs)
        else:
            raise ValueError(f"Unsupported actor worker type: {type(self.actor_wg)}")
        return lm_outputs


class ExperienceReplayTrainer:
    """
    ER trainer that:
    1) collects successful rollouts,
    2) appends them to a replay buffer,
    3) validates after each new experience,
    4) terminates when the buffer is full.
    """

    def __init__(self, config, tokenizer, processor, actor_rollout_wg, er_config: ERConfig):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.actor_rollout_wg = actor_rollout_wg
        self.er_config = er_config

        self.er_method = ExperienceReplayMethod(er_config)
        self.agent_proxy = ERAgentProxy(config, actor_rollout_wg, tokenizer)

        self.rollout_count = 0
        self.start_time = None

    def _update_examples_in_prompts(self):
        examples_text = self.er_method.get_examples_text()
        self.agent_proxy.set_examples_text(examples_text, for_val=False)
        self.agent_proxy.set_examples_text(examples_text, for_val=True)

    def _build_meta_info(self, val: bool) -> Dict[str, Any]:
        meta_info = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "recompute_log_prob": False,
        }
        if val:
            meta_info["do_sample"] = self.config.actor_rollout_ref.rollout.val_kwargs.do_sample
            meta_info["validate"] = True
        else:
            meta_info["do_sample"] = True
        return meta_info

    def _run_rollout(self, val: bool) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        batch = DataProto(batch=None, non_tensor_batch=None, meta_info=self._build_meta_info(val=val))
        rollouts = self.agent_proxy.rollout(batch, val=val)
        rollout_states = (
            self.agent_proxy.val_es_manager.get_rollout_states()
            if val
            else self.agent_proxy.train_es_manager.get_rollout_states()
        )
        metrics = rollouts.meta_info.get("metrics", {})
        return rollout_states, metrics

    def _aggregate_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        if not metrics_list:
            return {}
        agg = defaultdict(list)
        for metrics in metrics_list:
            for k, v in metrics.items():
                agg[k].append(v)
        return {k: float(np.mean(v)) for k, v in agg.items()}

    def _run_validation(self) -> Dict[str, float]:
        metrics_list = []
        for _ in range(self.config.trainer.validation_steps):
            _, metrics = self._run_rollout(val=True)
            metrics_list.append(metrics)
        mean_metrics = self._aggregate_metrics(metrics_list)
        val_metrics = {f"val/{k}": v for k, v in mean_metrics.items()}

        # Convenience: log a single success_rate if unambiguous
        success_keys = [k for k in mean_metrics.keys() if k.endswith("/success")]
        if len(success_keys) == 1:
            val_metrics["val/success_rate"] = mean_metrics[success_keys[0]]

        val_metrics["val/buffer_size"] = self.er_method.buffer.size
        val_metrics["val/buffer_fill_ratio"] = self.er_method.buffer.size / self.er_method.buffer.max_size
        return val_metrics

    def fit(self) -> Dict[str, Any]:
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.start_time = time.time()
        print(f"\n{'=' * 60}")
        print("Experience Replay Training (In-Context Memory)")
        print(f"{'=' * 60}")
        print(f"Environment: {self.er_config.env_tag}")
        print(f"Buffer size (N): {self.er_config.buffer_size}")
        print(f"{'=' * 60}\n")

        # Initial validation (buffer size 0)
        self._update_examples_in_prompts()
        val_metrics = self._run_validation()
        val_metrics["rollout_count"] = 0
        logger.log(data=val_metrics, step=0)
        print("[ER] Initial validation (buffer size 0):")
        print(val_metrics)

        # Main loop: keep sampling until buffer is full
        while not self.er_method.is_full():
            self.rollout_count += 1
            self._update_examples_in_prompts()

            rollout_states, train_metrics = self._run_rollout(val=False)
            successes = self.er_method.extract_successful_experiences(rollout_states)

            # Log rollout-level training metrics (optional)
            if train_metrics:
                log_metrics = {f"train/{k}": v for k, v in train_metrics.items()}
                log_metrics["train/rollout_count"] = self.rollout_count
                logger.log(data=log_metrics, step=self.er_method.buffer.size)

            if not successes:
                continue

            for exp in successes:
                if self.er_method.is_full():
                    break
                self.er_method.add_experience(exp)
                self._update_examples_in_prompts()
                val_metrics = self._run_validation()
                val_metrics["rollout_count"] = self.rollout_count
                logger.log(data=val_metrics, step=self.er_method.buffer.size)
                print(f"[ER] Validation at buffer size {self.er_method.buffer.size}:")
                print(val_metrics)

        # Save buffer
        if self.er_config.output_dir:
            os.makedirs(self.er_config.output_dir, exist_ok=True)
            save_path = os.path.join(
                self.er_config.output_dir,
                f"replay_buffer_{self.er_config.env_tag}.pkl"
            )
            self.er_method.save(save_path)
            print(f"[ER] Buffer saved to {save_path}")

        total_time = time.time() - self.start_time
        final_stats = {
            "total_rollouts": self.rollout_count,
            "total_time": total_time,
            "final_buffer_size": self.er_method.buffer.size,
        }
        print(f"\n[ER] Training complete. Rollouts: {self.rollout_count}, time: {total_time:.1f}s")
        return final_stats
