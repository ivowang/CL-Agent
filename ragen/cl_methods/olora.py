"""
O-LoRA: Orthogonal Low-Rank Adaptation for Continual Learning

Based on the paper: "Orthogonal Subspace Learning for Language Model Continual Learning"
(Wang et al., 2023) - https://arxiv.org/abs/2310.14152

Key idea: Learn new tasks in orthogonal subspaces to minimize interference with
previously learned tasks.

NOTE: In distributed training (Ray/FSDP), the trainer cannot directly access the model.
The orthogonal loss computation must be done in the worker. This class manages the
CL state and configuration, which is passed to workers.
"""

import os
import torch
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from .base import BaseCLMethod, CLMethodConfig
from .registry import register_cl_method


@dataclass 
class OLoRAConfig(CLMethodConfig):
    """Configuration for O-LoRA CL method."""
    name: str = "olora"
    # O-LoRA specific parameters
    lambda_ortho: float = 0.5  # Weight for orthogonal loss
    lambda_l2: float = 0.0  # Weight for L2 regularization on new LoRA params
    # LoRA configuration per task
    lora_rank_per_task: int = 64  # Rank for each task's LoRA
    # Whether to reinitialize LoRA for each new task
    reinit_lora_per_task: bool = True


@register_cl_method("olora")
class OLoRACLMethod(BaseCLMethod):
    """
    O-LoRA (Orthogonal Low-Rank Adaptation) Continual Learning Method.
    
    This method learns each task in an orthogonal subspace by:
    1. Storing LoRA parameters from previous tasks
    2. For new tasks, reinitializing trainable LoRA parameters
    3. Applying orthogonal constraints during training (in worker)
    
    The orthogonal loss is: L_ortho = sum(|A_old @ A_new.T|)
    
    Since we cannot directly access the model in the trainer, we:
    - Store frozen LoRA params from checkpoints
    - Pass config to workers for loss computation
    """
    
    def __init__(self, config: OLoRAConfig):
        super().__init__(config)
        self.config: OLoRAConfig = config
        
        # Store paths to LoRA checkpoints from previous tasks
        self.task_checkpoints: Dict[int, str] = {}
        
        # Track accumulated rank from previous tasks
        self.accumulated_rank = 0
        
        self.log_info(f"Initialized O-LoRA with λ_ortho={config.lambda_ortho}, λ_l2={config.lambda_l2}")
    
    def on_task_start(self, task_idx: int, task_name: str,
                      prev_checkpoint_path: Optional[str] = None) -> None:
        """
        Called at the beginning of each task.
        """
        super().on_task_start(task_idx, task_name, prev_checkpoint_path)
        
        if task_idx > 0 and prev_checkpoint_path:
            # Store the checkpoint path from the previous task
            self.task_checkpoints[task_idx - 1] = prev_checkpoint_path
            self.accumulated_rank += self.config.lora_rank_per_task
            self.log_info(f"Stored checkpoint from task {task_idx - 1}: {prev_checkpoint_path}")
            self.log_info(f"Accumulated LoRA rank: {self.accumulated_rank}")
            
            if self.config.reinit_lora_per_task:
                self.log_info(f"LoRA will be reinitialized for task {task_idx}")
    
    def on_task_end(self, task_idx: int, task_name: str,
                    checkpoint_path: str) -> None:
        """
        Called at the end of each task.
        Store the checkpoint path for future reference.
        """
        super().on_task_end(task_idx, task_name, checkpoint_path)
        
        # Store this task's checkpoint
        self.task_checkpoints[task_idx] = checkpoint_path
        self.log_info(f"Stored checkpoint for task {task_idx}: {checkpoint_path}")
    
    def get_cl_loss_config(self) -> Dict[str, Any]:
        """
        Return configuration for O-LoRA loss computation.
        This config will be passed to the worker for loss computation.
        
        For O-LoRA, the worker needs:
        - Frozen LoRA params from previous tasks
        - Lambda values for loss computation
        """
        # Load frozen params from previous task checkpoint if available
        frozen_lora_params = None
        if self.current_task_idx > 0 and self.task_checkpoints:
            # Get the most recent checkpoint (from the previous task)
            prev_task_idx = self.current_task_idx - 1
            if prev_task_idx in self.task_checkpoints:
                from .loss_functions import load_frozen_lora_params_from_checkpoint
                checkpoint_path = self.task_checkpoints[prev_task_idx]
                frozen_lora_params = load_frozen_lora_params_from_checkpoint(
                    checkpoint_path, 
                    device=torch.device('cpu')
                )
                self.log_info(f"Loaded frozen LoRA params from task {prev_task_idx}")
        
        return {
            'method_name': 'olora',
            'current_task_idx': self.current_task_idx,
            'has_cl_loss': self.current_task_idx > 0,  # Only after first task
            'lambda_ortho': self.config.lambda_ortho,
            'lambda_l2': self.config.lambda_l2,
            'accumulated_rank': self.accumulated_rank,
            'task_checkpoints': self.task_checkpoints.copy(),
            'reinit_lora_per_task': self.config.reinit_lora_per_task,
            'frozen_lora_params': frozen_lora_params,
        }
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get the current state as a dictionary (for passing between tasks)."""
        state = super().get_state_dict()
        state.update({
            'accumulated_rank': self.accumulated_rank,
            'task_checkpoints': self.task_checkpoints.copy(),
            'lambda_ortho': self.config.lambda_ortho,
            'lambda_l2': self.config.lambda_l2,
        })
        return state
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load state from a dictionary."""
        super().load_state_dict(state)
        if state:
            self.accumulated_rank = state.get('accumulated_rank', 0)
            self.task_checkpoints = state.get('task_checkpoints', {})
    
    def save_method_state(self, checkpoint_path: str,
                          additional_state: Optional[Dict] = None) -> str:
        """Save O-LoRA specific state."""
        state = additional_state or {}
        state['accumulated_rank'] = self.accumulated_rank
        state['task_checkpoints'] = self.task_checkpoints
        state['lambda_ortho'] = self.config.lambda_ortho
        state['lambda_l2'] = self.config.lambda_l2
        
        return super().save_method_state(checkpoint_path, state)
    
    def load_method_state(self, checkpoint_path: str) -> Optional[Dict]:
        """Load O-LoRA specific state."""
        state = super().load_method_state(checkpoint_path)
        
        if state:
            self.accumulated_rank = state.get('accumulated_rank', 0)
            self.task_checkpoints = state.get('task_checkpoints', {})
            self.log_info(f"Loaded O-LoRA state: accumulated_rank={self.accumulated_rank}")
        
        return state
    
    def get_method_info(self) -> Dict[str, Any]:
        """Return information about the method."""
        return {
            'name': 'olora',
            'description': 'Orthogonal Low-Rank Adaptation for Continual Learning',
            'has_cl_loss': self.current_task_idx > 0,
            'lambda_ortho': self.config.lambda_ortho,
            'lambda_l2': self.config.lambda_l2,
            'accumulated_rank': self.accumulated_rank,
            'current_task': self.current_task_idx,
            'num_previous_tasks': len(self.task_checkpoints),
            'task_history': self.task_history
        }
