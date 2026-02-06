"""
MemoryBank Trainer for RAGEN (In-Context Memory with Retrieval).

Collects successful rollouts into a memory bank, retrieves relevant memories
for each new state, and validates after each memory addition. No parameter updates.
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
from cl_methods.memorybank import MemoryBankMethod, MemoryBankConfig


class MemoryBankContextManager(ContextManager):
    """ContextManager that retrieves memories per env state."""

    def __init__(self, config, tokenizer, processor=None, mode: str = "train"):
        super().__init__(config, tokenizer, processor, mode)
        self._memorybank: MemoryBankMethod | None = None
        self._current_step: int = 0
        self._examples_text_by_env: Dict[int, str] = {}

    def set_memorybank(self, memorybank: MemoryBankMethod):
        self._memorybank = memorybank

    def set_current_step(self, step: int):
        self._current_step = step

    def get_lm_inputs(self, env_outputs: List[Dict], prepare_for_update: bool) -> DataProto:
        # Build memory examples per env based on current state
        self._examples_text_by_env = {}
        if self._memorybank is not None:
            for env_output in env_outputs:
                env_id = env_output["env_id"]
                history = env_output.get("history", [])
                if history:
                    query = history[-1].get("state", "")
                else:
                    query = ""
                examples_text = self._memorybank.get_examples_text(query, self._current_step)
                if examples_text:
                    self._examples_text_by_env[env_id] = examples_text
        return super().get_lm_inputs(env_outputs, prepare_for_update)

    def _build_system_content(self, env_id: int) -> str:
        base_content = self.prefix_lookup.get(env_id, "")
        examples_text = self._examples_text_by_env.get(env_id, "")
        if examples_text:
            return examples_text + base_content
        return base_content


class MemoryBankAgentProxy(LLMAgentProxy):
    """AgentProxy using MemoryBankContextManager."""

    def __init__(self, config, actor_rollout_wg, tokenizer):
        self.config = config
        self.train_ctx_manager = MemoryBankContextManager(config, tokenizer, mode="train")
        self.train_es_manager = EnvStateManager(config, mode="train")
        self.val_ctx_manager = MemoryBankContextManager(config, tokenizer, mode="val")
        self.val_es_manager = EnvStateManager(config, mode="val")
        self.actor_wg = actor_rollout_wg
        self.tokenizer = tokenizer
        self._last_padded_inputs = None

    def set_memorybank(self, memorybank: MemoryBankMethod, for_val: bool = False):
        if for_val:
            self.val_ctx_manager.set_memorybank(memorybank)
        else:
            self.train_ctx_manager.set_memorybank(memorybank)

    def set_current_step(self, step: int, for_val: bool = False):
        if for_val:
            self.val_ctx_manager.set_current_step(step)
        else:
            self.train_ctx_manager.set_current_step(step)

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


class MemoryBankTrainer:
    """MemoryBank trainer with retrieval-based in-context memory."""

    def __init__(self, config, tokenizer, processor, actor_rollout_wg, mb_config: MemoryBankConfig):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.actor_rollout_wg = actor_rollout_wg
        self.mb_config = mb_config

        self.memorybank = MemoryBankMethod(mb_config)
        self.agent_proxy = MemoryBankAgentProxy(config, actor_rollout_wg, tokenizer)
        self.agent_proxy.set_memorybank(self.memorybank, for_val=False)
        self.agent_proxy.set_memorybank(self.memorybank, for_val=True)

        self.rollout_count = 0
        self.start_time = None

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
            meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
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

        success_keys = [k for k in mean_metrics.keys() if k.endswith("/success")]
        if len(success_keys) == 1:
            val_metrics["val/success_rate"] = mean_metrics[success_keys[0]]

        val_metrics["val/buffer_size"] = self.memorybank.bank.size
        val_metrics["val/buffer_fill_ratio"] = self.memorybank.bank.size / self.memorybank.bank.config.buffer_size
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
        print("MemoryBank Training (In-Context Retrieval)")
        print(f"{'=' * 60}")
        print(f"Environment: {self.mb_config.env_tag}")
        print(f"Buffer size (N): {self.mb_config.buffer_size}")
        print(f"Top-K memories: {self.mb_config.top_k}")
        print(f"{'=' * 60}\n")

        # Initial validation (buffer size 0)
        self.agent_proxy.set_current_step(self.rollout_count, for_val=False)
        self.agent_proxy.set_current_step(self.rollout_count, for_val=True)
        val_metrics = self._run_validation()
        val_metrics["rollout_count"] = 0
        logger.log(data=val_metrics, step=0)
        print("[MemoryBank] Initial validation (buffer size 0):")
        print(val_metrics)

        while not self.memorybank.is_full():
            self.rollout_count += 1
            self.agent_proxy.set_current_step(self.rollout_count, for_val=False)

            rollout_states, train_metrics = self._run_rollout(val=False)
            successes = self.memorybank.extract_successful_memories(rollout_states, step=self.rollout_count)

            if train_metrics:
                log_metrics = {f"train/{k}": v for k, v in train_metrics.items()}
                log_metrics["train/rollout_count"] = self.rollout_count
                logger.log(data=log_metrics, step=self.memorybank.bank.size)

            if not successes:
                continue

            for entry in successes:
                if self.memorybank.is_full():
                    break
                self.memorybank.add_memory(entry)
                self.agent_proxy.set_current_step(self.rollout_count, for_val=True)
                val_metrics = self._run_validation()
                val_metrics["rollout_count"] = self.rollout_count
                logger.log(data=val_metrics, step=self.memorybank.bank.size)
                print(f"[MemoryBank] Validation at buffer size {self.memorybank.bank.size}:")
                print(val_metrics)

        if self.mb_config.output_dir:
            os.makedirs(self.mb_config.output_dir, exist_ok=True)
            save_path = os.path.join(
                self.mb_config.output_dir,
                f"memorybank_{self.mb_config.env_tag}.pkl"
            )
            self.memorybank.save(save_path)
            print(f"[MemoryBank] Memory bank saved to {save_path}")

        total_time = time.time() - self.start_time
        final_stats = {
            "total_rollouts": self.rollout_count,
            "total_time": total_time,
            "final_buffer_size": self.memorybank.bank.size,
        }
        print(f"\n[MemoryBank] Training complete. Rollouts: {self.rollout_count}, time: {total_time:.1f}s")
        return final_stats
