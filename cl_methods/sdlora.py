"""
SD-LoRA: Scalable Decomposed Low-Rank Adaptation for Continual Learning

Based on the paper: "SD-LoRA: Scalable Decomposed Low-Rank Adaptation for Continual Learning"
Reference implementation: /home/wangziyi/ref/SD-Lora-CL/

Key idea: Learn each task with separate LoRA adapters and use learnable scaling factors
to combine them during inference. This allows the model to retain knowledge from
previous tasks while learning new ones.

Architecture:
    Task 0: LoRA_0 (A_0, B_0) with scaling_factor_0 - trained, then frozen
    Task 1: LoRA_1 (A_1, B_1) with scaling_factor_1 - trained, then frozen
    Task 2: LoRA_2 (A_2, B_2) with scaling_factor_2 - currently training
    ...

Forward pass formula (with normalization):
    output = W @ x + Σ_i scaling_factor[i] * (B_i @ A_i @ x) / (||B_i|| * ||A_i||)

Training:
    - Current task's LoRA (A_i, B_i) is trainable
    - Current task's scaling_factor_i is trainable
    - All previous tasks' scaling_factors are also trainable (fine-tuned)
    - Previous tasks' LoRA params are FROZEN

NOTE: In distributed training (Ray/FSDP), the trainer cannot directly access the model.
The scaling factor combination must be done in the worker. This class manages the
CL state and configuration, which is passed to workers.
"""

import os
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from .base import BaseCLMethod, CLMethodConfig
from .registry import register_cl_method
from .multi_lora import MultiLoRAManager, extract_lora_params_from_model


@dataclass
class SDLoRAConfig(CLMethodConfig):
    """Configuration for SD-LoRA CL method."""
    name: str = "sdlora"
    # SD-LoRA specific parameters
    scaling_factor_init: float = 0.8  # Initial value for scaling factors
    normalize_lora: bool = True  # Whether to normalize LoRA outputs by weight norms
    # LoRA configuration per task
    lora_rank_per_task: int = 64  # Rank for each task's LoRA
    # Whether to reinitialize LoRA for each new task
    reinit_lora_per_task: bool = True
    # Maximum number of tasks to support (for pre-allocating scaling factors)
    max_tasks: int = 20


@register_cl_method("sdlora")
class SDLoRACLMethod(BaseCLMethod):
    """
    SD-LoRA (Scalable Decomposed Low-Rank Adaptation) Continual Learning Method.

    This method learns each task with its OWN LoRA parameters by:
    1. For task 0: Train LoRA_0 with scaling_factor_0, then freeze LoRA_0
    2. For task i > 0:
       - Reinitialize new LoRA_i parameters
       - Train LoRA_i with scaling_factor_i
       - Also fine-tune all previous scaling_factors (but not their LoRAs)
       - Freeze LoRA_i after training

    Forward pass combines all task LoRAs with learned scaling:
        output = W @ x + Σ_i scaling_factor[i] * (B_i @ A_i @ x) / (||B_i|| * ||A_i||)

    The normalization by weight norms ensures stable combination across tasks.
    """

    def __init__(self, config: SDLoRAConfig):
        super().__init__(config)
        self.config: SDLoRAConfig = config

        # Multi-LoRA manager for handling multiple task LoRAs
        self.multi_lora_manager = MultiLoRAManager(
            method='sdlora',
            lora_rank=config.lora_rank_per_task,
            lora_alpha=config.lora_rank_per_task,
            scaling_factor_init=config.scaling_factor_init,
            normalize_lora=config.normalize_lora,
            max_tasks=config.max_tasks,
        )

        # Store paths to LoRA checkpoints from previous tasks
        self.task_checkpoints: Dict[int, str] = {}

        # Track accumulated rank from previous tasks
        self.accumulated_rank = 0

        # Scaling factors for each task (stored as dict: task_idx -> scaling_factor_value)
        # These will be passed to the worker and trained there
        self.scaling_factors: Dict[int, float] = {}

        self.log_info(f"Initialized SD-LoRA with scaling_factor_init={config.scaling_factor_init}, "
                      f"normalize_lora={config.normalize_lora}")

    def on_task_start(self, task_idx: int, task_name: str,
                      prev_checkpoint_path: Optional[str] = None) -> None:
        """
        Called at the beginning of each task.

        For SD-LoRA:
        1. If task_idx > 0, load and freeze previous task's LoRA
        2. Initialize scaling factor for current task
        3. Signal that current LoRA should be reinitialized
        """
        super().on_task_start(task_idx, task_name, prev_checkpoint_path)

        self.multi_lora_manager.set_current_task(task_idx)

        if task_idx > 0 and prev_checkpoint_path:
            # Store the checkpoint path from the previous task
            self.task_checkpoints[task_idx - 1] = prev_checkpoint_path
            self.accumulated_rank += self.config.lora_rank_per_task

            # Load scaling factors from previous checkpoint if available
            self._load_scaling_factors_from_checkpoint(prev_checkpoint_path)

            # Load and freeze previous task's LoRA parameters
            success = self.multi_lora_manager.load_frozen_loras_from_checkpoint(
                prev_checkpoint_path,
                task_idx - 1,
                device=torch.device('cpu')
            )

            if success:
                self.log_info(f"Loaded and froze LoRA from task {task_idx - 1}")
                # Update scaling factor in manager
                if task_idx - 1 in self.scaling_factors:
                    self.multi_lora_manager.scaling_factors[task_idx - 1] = self.scaling_factors[task_idx - 1]
            else:
                self.log_info(f"Warning: Could not load LoRA from task {task_idx - 1}")

            self.log_info(f"Accumulated LoRA rank: {self.accumulated_rank}")

            if self.config.reinit_lora_per_task:
                self.log_info(f"LoRA will be reinitialized for task {task_idx}")

        # Initialize scaling factor for current task
        self.scaling_factors[task_idx] = self.config.scaling_factor_init
        self.multi_lora_manager.scaling_factors[task_idx] = self.config.scaling_factor_init
        self.log_info(f"Initialized scaling factor for task {task_idx}: {self.scaling_factors[task_idx]}")

    def on_task_end(self, task_idx: int, task_name: str,
                    checkpoint_path: str) -> None:
        """
        Called at the end of each task.
        Store the checkpoint path and save scaling factors.
        """
        super().on_task_end(task_idx, task_name, checkpoint_path)

        # Store this task's checkpoint
        self.task_checkpoints[task_idx] = checkpoint_path
        self.log_info(f"Stored checkpoint for task {task_idx}: {checkpoint_path}")

        # Save scaling factors
        self._save_scaling_factors(checkpoint_path)

        # Save multi-LoRA manager state
        manager_path = os.path.join(checkpoint_path, 'multi_lora_manager.pt')
        self.multi_lora_manager.save(manager_path)

    def _load_scaling_factors_from_checkpoint(self, checkpoint_path: str) -> None:
        """Load scaling factors from a checkpoint."""
        scaling_path = os.path.join(checkpoint_path, 'scaling_factors.pt')
        if os.path.exists(scaling_path):
            try:
                saved_factors = torch.load(scaling_path, map_location='cpu')
                if isinstance(saved_factors, dict):
                    self.scaling_factors.update(saved_factors)
                    self.multi_lora_manager.scaling_factors.update(saved_factors)
                    self.log_info(f"Loaded scaling factors from {scaling_path}: {self.scaling_factors}")
            except Exception as e:
                self.log_info(f"Warning: Could not load scaling factors: {e}")

    def _save_scaling_factors(self, checkpoint_path: str) -> None:
        """Save scaling factors to checkpoint."""
        os.makedirs(checkpoint_path, exist_ok=True)
        scaling_path = os.path.join(checkpoint_path, 'scaling_factors.pt')
        torch.save(self.scaling_factors, scaling_path)
        self.log_info(f"Saved scaling factors to {scaling_path}")

    def get_cl_loss_config(self) -> Dict[str, Any]:
        """
        Return configuration for SD-LoRA forward pass modification.
        This config will be passed to the worker.

        For SD-LoRA, the worker needs:
        - All frozen LoRA params from previous tasks
        - Scaling factors for each task
        - Configuration for normalization
        """
        # Collect all frozen LoRA params
        all_frozen_lora_params = {}
        for task_idx, task_lora in self.multi_lora_manager.frozen_task_loras.items():
            all_frozen_lora_params[task_idx] = {
                'A': task_lora.lora_A,
                'B': task_lora.lora_B,
                'scaling_factor': task_lora.scaling_factor,
            }

        # For backward compatibility
        frozen_lora_params = None
        if self.current_task_idx > 0 and (self.current_task_idx - 1) in all_frozen_lora_params:
            frozen_lora_params = {}
            prev_task = all_frozen_lora_params[self.current_task_idx - 1]
            for module_name in prev_task['A'].keys():
                frozen_lora_params[module_name] = {
                    'A': prev_task['A'][module_name],
                    'B': prev_task['B'].get(module_name),
                }

        return {
            'method_name': 'sdlora',
            'current_task_idx': self.current_task_idx,
            'has_cl_loss': False,  # SD-LoRA doesn't have explicit CL loss
            'has_forward_modification': self.current_task_idx > 0,  # Modify forward after first task
            'scaling_factor_init': self.config.scaling_factor_init,
            'normalize_lora': self.config.normalize_lora,
            'accumulated_rank': self.accumulated_rank,
            'task_checkpoints': self.task_checkpoints.copy(),
            'reinit_lora_per_task': self.config.reinit_lora_per_task,
            'scaling_factors': self.scaling_factors.copy(),
            # All frozen LoRA params from all previous tasks
            'all_frozen_lora_params': all_frozen_lora_params,
            # For backward compatibility
            'frozen_lora_params': frozen_lora_params,
            # Multi-LoRA manager state for forward pass
            'multi_lora_manager_state': self.multi_lora_manager.get_state_dict(),
        }

    def update_scaling_factors(self, new_factors: Dict[int, float]) -> None:
        """
        Update scaling factors from worker (after training step).
        This is called by the trainer to sync scaling factors from the worker.
        """
        self.scaling_factors.update(new_factors)
        self.multi_lora_manager.scaling_factors.update(new_factors)
        self.log_info(f"Updated scaling factors: {self.scaling_factors}")

    def get_state_dict(self) -> Dict[str, Any]:
        """Get the current state as a dictionary (for passing between tasks)."""
        state = super().get_state_dict()
        state.update({
            'accumulated_rank': self.accumulated_rank,
            'task_checkpoints': self.task_checkpoints.copy(),
            'scaling_factors': self.scaling_factors.copy(),
            'scaling_factor_init': self.config.scaling_factor_init,
            'normalize_lora': self.config.normalize_lora,
            'multi_lora_manager_state': self.multi_lora_manager.get_state_dict(),
        })
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load state from a dictionary."""
        super().load_state_dict(state)
        if state:
            self.accumulated_rank = state.get('accumulated_rank', 0)
            self.task_checkpoints = state.get('task_checkpoints', {})
            self.scaling_factors = state.get('scaling_factors', {})

            # Load multi-LoRA manager state
            manager_state = state.get('multi_lora_manager_state')
            if manager_state:
                self.multi_lora_manager.load_state_dict(manager_state)

    def save_method_state(self, checkpoint_path: str,
                          additional_state: Optional[Dict] = None) -> str:
        """Save SD-LoRA specific state."""
        state = additional_state or {}
        state['accumulated_rank'] = self.accumulated_rank
        state['task_checkpoints'] = self.task_checkpoints
        state['scaling_factors'] = self.scaling_factors
        state['scaling_factor_init'] = self.config.scaling_factor_init
        state['normalize_lora'] = self.config.normalize_lora
        state['multi_lora_manager_state'] = self.multi_lora_manager.get_state_dict()

        # Also save scaling factors separately for easy access
        self._save_scaling_factors(checkpoint_path)

        # Save multi-LoRA manager
        manager_path = os.path.join(checkpoint_path, 'multi_lora_manager.pt')
        self.multi_lora_manager.save(manager_path)

        return super().save_method_state(checkpoint_path, state)

    def load_method_state(self, checkpoint_path: str) -> Optional[Dict]:
        """Load SD-LoRA specific state."""
        state = super().load_method_state(checkpoint_path)

        if state:
            self.accumulated_rank = state.get('accumulated_rank', 0)
            self.task_checkpoints = state.get('task_checkpoints', {})
            self.scaling_factors = state.get('scaling_factors', {})

            # Load multi-LoRA manager state
            manager_state = state.get('multi_lora_manager_state')
            if manager_state:
                self.multi_lora_manager.load_state_dict(manager_state)

            self.log_info(f"Loaded SD-LoRA state: accumulated_rank={self.accumulated_rank}, "
                          f"scaling_factors={self.scaling_factors}")

        # Also try to load scaling factors from separate file
        self._load_scaling_factors_from_checkpoint(checkpoint_path)

        # Load multi-LoRA manager from separate file
        manager_path = os.path.join(checkpoint_path, 'multi_lora_manager.pt')
        if os.path.exists(manager_path):
            self.multi_lora_manager.load(manager_path)

        return state

    def get_method_info(self) -> Dict[str, Any]:
        """Return information about the method."""
        return {
            'name': 'sdlora',
            'description': 'Scalable Decomposed Low-Rank Adaptation for Continual Learning',
            'has_cl_loss': False,
            'has_forward_modification': self.current_task_idx > 0,
            'scaling_factor_init': self.config.scaling_factor_init,
            'normalize_lora': self.config.normalize_lora,
            'accumulated_rank': self.accumulated_rank,
            'current_task': self.current_task_idx,
            'num_frozen_tasks': len(self.multi_lora_manager.frozen_task_loras),
            'scaling_factors': self.scaling_factors,
            'task_history': self.task_history
        }
