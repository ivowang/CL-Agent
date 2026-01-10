"""
Experience Replay Trainer for RAGEN

This trainer implements in-context learning through experience replay.
It collects successful trajectories and uses them as examples in prompts,
without updating model parameters.
"""

import os
import time
import numpy as np
from datetime import datetime
from pprint import pprint
from tqdm import tqdm
from typing import Dict, List, Optional, Any, Tuple
from copy import deepcopy

import ray
from omegaconf import OmegaConf, open_dict

from verl import DataProto
from verl.utils.tracking import Tracking
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray.base import RayWorkerGroup

from ragen.llm_agent.agent_proxy import LLMAgentProxy, VllmWrapperWg
from ragen.llm_agent.es_manager import EnvStateManager
from ragen.llm_agent.ctx_manager import ContextManager
from ragen.cl_methods.experience_replay import (
    ExperienceReplayMethod, 
    ERConfig, 
    Experience,
    ReplayBuffer
)


class ERContextManager(ContextManager):
    """
    Extended ContextManager that injects experience examples into prompts.
    """
    
    def __init__(self, config, tokenizer, processor=None, mode: str = "train"):
        super().__init__(config, tokenizer, processor, mode)
        self._examples_text = ""
        
    def set_examples_text(self, examples_text: str):
        """Set the examples text to prepend to system prompts."""
        self._examples_text = examples_text
    
    def _build_system_content(self, env_id: int) -> str:
        """Override to inject examples into system prompt."""
        base_content = self.prefix_lookup.get(env_id, "")
        if self._examples_text:
            return self._examples_text + base_content
        return base_content


class ERAgentProxy(LLMAgentProxy):
    """
    Extended AgentProxy that uses ERContextManager.
    """
    
    def __init__(self, config, actor_rollout_wg, tokenizer):
        # Don't call parent __init__, we'll set up managers ourselves
        self.config = config
        self.train_ctx_manager = ERContextManager(config, tokenizer, mode="train")
        self.train_es_manager = EnvStateManager(config, mode="train")
        self.val_ctx_manager = ERContextManager(config, tokenizer, mode="val")
        self.val_es_manager = EnvStateManager(config, mode="val")
        self.actor_wg = actor_rollout_wg
        self.tokenizer = tokenizer
        self._last_padded_inputs = None
    
    def set_examples_text(self, examples_text: str, for_val: bool = False):
        """Set examples text for context managers."""
        if for_val:
            self.val_ctx_manager.set_examples_text(examples_text)
        else:
            self.train_ctx_manager.set_examples_text(examples_text)
    
    def generate_sequences(self, lm_inputs: DataProto):
        """Generate sequences using the actor worker group."""
        if isinstance(self.actor_wg, RayWorkerGroup):
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
    Trainer for Experience Replay method.
    
    This trainer:
    1. Runs rollouts to collect experiences
    2. Stores successful trajectories in a replay buffer
    3. Uses buffer contents as in-context examples
    4. Validates at specified intervals
    5. Ends when buffer is full
    """
    
    def __init__(
        self,
        config,
        tokenizer,
        processor,
        actor_rollout_wg,
        er_config: ERConfig,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.actor_rollout_wg = actor_rollout_wg
        self.er_config = er_config
        
        # Initialize ER method
        self.er_method = ExperienceReplayMethod(er_config)
        
        # Initialize agent proxy with extended context manager
        self.agent_proxy = ERAgentProxy(config, actor_rollout_wg, tokenizer)
        
        # Track statistics
        self.rollout_count = 0
        self.start_time = None
    
    def _update_examples_in_prompts(self):
        """Update the examples text in context managers."""
        examples_text = self.er_method.get_examples_text()
        self.agent_proxy.set_examples_text(examples_text, for_val=False)
        self.agent_proxy.set_examples_text(examples_text, for_val=True)
    
    def _run_rollout(self) -> Tuple[List[Dict], Dict[str, float]]:
        """
        Run a single rollout batch and return states and metrics.
        
        Returns:
            Tuple of (rollout_states, metrics)
        """
        batch = DataProto()
        batch.meta_info = {}
        
        # Run rollout
        rollouts = self.agent_proxy.rollout(batch, val=False)
        
        # Get rollout states from ES manager
        rollout_states = self.agent_proxy.train_es_manager.get_rollout_states()
        
        # Extract metrics
        metrics = rollouts.meta_info.get("metrics", {})
        
        return rollout_states, metrics
    
    def _run_validation(self) -> Dict[str, float]:
        """
        Run validation and return metrics.
        
        Returns:
            Dict of validation metrics
        """
        print(f"[ER] Running validation with {self.er_method.buffer.size} examples in buffer...")
        
        batch = DataProto()
        batch.meta_info = {}
        
        # Run validation rollout
        rollouts = self.agent_proxy.rollout(batch, val=True)
        
        # Get rollout states
        rollout_states = self.agent_proxy.val_es_manager.get_rollout_states()
        
        # Compute metrics
        metrics = rollouts.meta_info.get("metrics", {})
        
        # Prefix with "val/" and add buffer stats
        val_metrics = {f"val/{k}": v for k, v in metrics.items()}
        val_metrics.update({
            "val/buffer_size": self.er_method.buffer.size,
            "val/buffer_fill_ratio": self.er_method.buffer.size / self.er_method.buffer.max_size,
        })
        
        return val_metrics
    
    def _compute_success_rate(self, rollout_states: List[Dict]) -> float:
        """Compute success rate from rollout states."""
        if not rollout_states:
            return 0.0
        
        successes = 0
        for state in rollout_states:
            metrics = state.get('metrics', {})
            for key, value in metrics.items():
                if key.endswith('/success') and value == 1.0:
                    successes += 1
                    break
        
        return successes / len(rollout_states)
    
    def fit(self) -> Dict[str, Any]:
        """
        Main training loop for Experience Replay.
        
        Key behavior: Only ONE experience is added per validation cycle.
        If a rollout produces multiple successes, we add them one at a time,
        running validation after each addition.
        
        Returns:
            Final statistics and metrics
        """
        # Initialize logger
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        
        self.start_time = time.time()
        
        print(f"\n{'=' * 60}")
        print(f"Experience Replay Training")
        print(f"{'=' * 60}")
        print(f"Environment: {self.er_config.env_tag}")
        print(f"Buffer size (N): {self.er_config.buffer_size}")
        print(f"Validation frequency (M): {self.er_config.val_frequency}")
        print(f"Max examples in prompt: {self.er_config.max_examples_in_prompt}")
        print(f"{'=' * 60}\n")
        
        # Initial validation (with empty buffer)
        print("[ER] Initial validation (no examples yet)...")
        val_metrics = self._run_validation()
        val_metrics['rollout_count'] = 0
        val_metrics['buffer_size'] = 0
        logger.log(data=val_metrics, step=0)
        pprint(val_metrics)
        
        progress_bar = tqdm(total=self.er_config.buffer_size, desc="Collecting experiences")
        
        # Pending experiences queue (from rollouts that produced multiple successes)
        pending_experiences: List[Experience] = []
        
        while not self.er_method.is_training_complete():
            # If we have pending experiences, use one; otherwise run a new rollout
            if pending_experiences:
                # Take one experience from pending queue
                exp_to_add = pending_experiences.pop(0)
                new_experiences = [exp_to_add]
                success_rate = None  # Not applicable for pending
                train_metrics = {}
            else:
                # Run a new rollout
                self.rollout_count += 1
                
                # Update examples in prompts before rollout
                self._update_examples_in_prompts()
                
                # Run rollout
                rollout_states, train_metrics = self._run_rollout()
                
                # Extract successful experiences
                new_experiences = self.er_method.extract_successful_experiences(rollout_states)
                
                # Compute statistics
                success_rate = self._compute_success_rate(rollout_states)
                
                # If multiple successes, take only the first one and queue the rest
                if len(new_experiences) > 1:
                    pending_experiences.extend(new_experiences[1:])
                    new_experiences = new_experiences[:1]
                    print(f"[ER] Got {len(new_experiences) + len(pending_experiences)} successes, "
                          f"adding 1 now, {len(pending_experiences)} pending")
            
            # Add at most ONE experience to buffer
            if new_experiences:
                num_added, buffer_full = self.er_method.add_experiences(new_experiences[:1])
            else:
                num_added = 0
                buffer_full = False
            
            # Log training metrics
            log_metrics = {
                "train/new_experiences_this_rollout": len(new_experiences),
                "train/experiences_added": num_added,
                "train/pending_experiences": len(pending_experiences),
                "train/buffer_size": self.er_method.buffer.size,
                "train/buffer_fill_ratio": self.er_method.buffer.size / self.er_method.buffer.max_size,
                "train/rollout_count": self.rollout_count,
            }
            if success_rate is not None:
                log_metrics["train/success_rate"] = success_rate
            log_metrics.update({f"train/{k}": v for k, v in train_metrics.items()})
            
            # Use buffer_size as the step for logging (to align with validation)
            current_step = self.er_method.buffer.size
            logger.log(data=log_metrics, step=current_step)
            
            # Update progress bar
            progress_bar.n = self.er_method.buffer.size
            progress_bar.refresh()
            
            # Print progress when we add an experience
            if num_added > 0:
                elapsed = time.time() - self.start_time
                print(f"[ER] Buffer {self.er_method.buffer.size}/{self.er_config.buffer_size} | "
                      f"Rollouts: {self.rollout_count} | "
                      f"Pending: {len(pending_experiences)} | "
                      f"Time: {elapsed:.1f}s")
            
            # Check if we should validate (after adding each experience)
            if num_added > 0 and self.er_method.should_validate():
                val_metrics = self._run_validation()
                val_metrics['rollout_count'] = self.rollout_count
                val_metrics['buffer_size'] = self.er_method.buffer.size
                logger.log(data=val_metrics, step=current_step)
                self.er_method.mark_validated()
                print(f"[ER] Validation at buffer size {self.er_method.buffer.size}:")
                pprint(val_metrics)
            
            # Check if buffer is full
            if buffer_full:
                print(f"\n[ER] Buffer full! Running final validation...")
                break
        
        progress_bar.close()
        
        # Final validation
        print(f"\n{'=' * 60}")
        print(f"[ER] Final Validation")
        print(f"{'=' * 60}")
        
        final_val_metrics = self._run_validation()
        final_val_metrics['rollout_count'] = self.rollout_count
        final_val_metrics['final'] = True
        logger.log(data=final_val_metrics, step=self.rollout_count)
        
        # Compute final statistics
        total_time = time.time() - self.start_time
        final_stats = {
            'total_rollouts': self.rollout_count,
            'total_time': total_time,
            'final_buffer_size': self.er_method.buffer.size,
            'experiences_collected': self.er_method.buffer.total_collected,
            **self.er_method.get_stats(),
        }
        
        print(f"\n[ER] Training Complete!")
        print(f"{'=' * 60}")
        print(f"Total rollouts: {self.rollout_count}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Buffer size: {self.er_method.buffer.size}/{self.er_config.buffer_size}")
        print(f"{'=' * 60}")
        
        pprint({"Final validation metrics": final_val_metrics})
        
        # Save buffer
        if self.er_config.output_dir:
            save_path = os.path.join(
                self.er_config.output_dir,
                f"replay_buffer_{self.er_config.env_tag}.pkl"
            )
            self.er_method.save(save_path)
            print(f"[ER] Buffer saved to {save_path}")
        
        return final_stats


def create_er_trainer(
    config,
    tokenizer,
    processor,
    actor_rollout_wg,
    buffer_size: int,
    val_frequency: int,
    env_tag: str,
    max_examples: int = 3,
    output_dir: str = "results/experience_replay",
) -> ExperienceReplayTrainer:
    """
    Factory function to create an ER trainer.
    
    Args:
        config: Main config
        tokenizer: Tokenizer
        processor: Processor (optional)
        actor_rollout_wg: Actor rollout worker group
        buffer_size: N - maximum experiences to collect
        val_frequency: M - validate every M experiences
        env_tag: Environment type
        max_examples: Maximum examples to include in prompt
        output_dir: Output directory for saving
        
    Returns:
        ExperienceReplayTrainer instance
    """
    er_config = ERConfig(
        buffer_size=buffer_size,
        val_frequency=val_frequency,
        env_tag=env_tag,
        max_examples_in_prompt=max_examples,
        output_dir=output_dir,
    )
    
    return ExperienceReplayTrainer(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        actor_rollout_wg=actor_rollout_wg,
        er_config=er_config,
    )
