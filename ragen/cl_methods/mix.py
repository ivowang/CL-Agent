"""
Mix Training Method - Multi-Task Mixed Training

This method trains on ALL environments SIMULTANEOUSLY in each batch.
Each parallel sampling step includes samples from all environments mixed
together, and the combined batch is used to update the shared LoRA parameters.

Key features:
- All environments mixed in each training batch
- Single shared LoRA module for all tasks
- No catastrophic forgetting concerns (all tasks trained together)
- No environment cycling or sequential task ordering
"""

import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from .base import BaseCLMethod, CLMethodConfig
from .registry import register_cl_method


@dataclass
class MixConfig(CLMethodConfig):
    """Configuration for Mix training method."""
    name: str = "mix"
    # List of environment tags to include in mixed batches
    env_tags: List[str] = field(default_factory=lambda: ["Bandit", "CoordSokoban", "CoordFrozenLake"])
    # Number of groups per environment (for training)
    train_n_groups: List[int] = field(default_factory=lambda: [4, 4, 4])
    # Number of groups per environment (for validation)
    val_n_groups: List[int] = field(default_factory=lambda: [32, 32, 32])


@register_cl_method("mix")
class MixCLMethod(BaseCLMethod):
    """
    Mix Training Method.
    
    This method trains on all environments simultaneously in each batch.
    Each training step samples from all environments mixed together.
    All environments share the same LoRA parameters.
    
    Unlike continual learning methods where tasks are trained sequentially,
    this method trains all tasks in a truly parallel manner.
    """
    
    def __init__(self, config: MixConfig):
        super().__init__(config)
        self.config: MixConfig = config
        
        # Track environments (no cycling needed, all are used each step)
        self.env_tags = list(config.env_tags)
        self.num_envs = len(self.env_tags)
        
        # Statistics
        self.total_steps = 0
        
        self.log_info(f"Initialized Mix method with {self.num_envs} environments (all mixed each step): {self.env_tags}")
    
    def get_all_env_tags(self) -> List[str]:
        """Get all environment tags that are mixed in each batch."""
        return self.env_tags.copy()
    
    def get_num_envs(self) -> int:
        """Get the number of environments being mixed."""
        return self.num_envs
    
    def record_step(self) -> None:
        """Record that a training step has been completed."""
        self.total_steps += 1
    
    def get_mixed_env_config(self) -> Dict[str, Any]:
        """
        Get the combined environment configuration for mixed training.
        Returns configuration for all environments mixed together.
        """
        return {
            'tags': self.env_tags,
            'train_n_groups': list(self.config.train_n_groups),
            'val_n_groups': list(self.config.val_n_groups),
        }
    
    def on_task_start(self, task_idx: int, task_name: str,
                      prev_checkpoint_path: Optional[str] = None) -> None:
        """
        For mix method, this is called once at the beginning of training.
        task_idx should always be 0.
        """
        super().on_task_start(task_idx, task_name, prev_checkpoint_path)
        self.log_info(f"Starting mixed training with all environments: {self.env_tags}")
    
    def on_task_end(self, task_idx: int, task_name: str,
                    checkpoint_path: str) -> None:
        """
        For mix method, this is called once at the end of training.
        """
        super().on_task_end(task_idx, task_name, checkpoint_path)
        self.log_info(f"Mixed training completed. Total steps: {self.total_steps}")
    
    def get_cl_loss_config(self) -> Dict[str, Any]:
        """
        Return CL loss configuration.
        For mix method, there is no additional CL loss.
        """
        return {
            'method_name': 'mix',
            'mixed_envs': self.env_tags,
            'num_envs': self.num_envs,
            'has_cl_loss': False,
            'cl_loss_weight': 0.0,
            'frozen_lora_params': None,
        }
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get the current state as a dictionary."""
        state = super().get_state_dict()
        state.update({
            'total_steps': self.total_steps,
            'env_tags': self.env_tags,
        })
        return state
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load state from a dictionary."""
        super().load_state_dict(state)
        if state:
            self.total_steps = state.get('total_steps', 0)
            self.env_tags = state.get('env_tags', self.env_tags)
            self.num_envs = len(self.env_tags)
    
    def get_method_info(self) -> Dict[str, Any]:
        """Return information about the method."""
        return {
            'name': 'mix',
            'description': 'Multi-task mixed training with shared LoRA (all envs per batch)',
            'has_cl_loss': False,
            'num_envs': self.num_envs,
            'env_tags': self.env_tags,
            'total_steps': self.total_steps,
        }
