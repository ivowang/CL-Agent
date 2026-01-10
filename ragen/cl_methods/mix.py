"""
Mix Training Method - Multi-Task Interleaved Training

This method trains on multiple environments in an interleaved manner,
switching between environments after each training step. Unlike continual
learning methods that train tasks sequentially, mix method trains all 
tasks simultaneously with a single shared LoRA module.

Key features:
- Interleaved training: env1 → env2 → env3 → env1 → ...
- Single shared LoRA module for all tasks
- No catastrophic forgetting concerns (all tasks trained together)
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
    # List of environment tags to cycle through
    env_tags: List[str] = field(default_factory=lambda: ["Bandit", "CoordSokoban", "CoordFrozenLake"])
    # Number of groups per environment (for training)
    train_n_groups: List[int] = field(default_factory=lambda: [8, 8, 8])
    # Number of groups per environment (for validation)
    val_n_groups: List[int] = field(default_factory=lambda: [32, 32, 32])


@register_cl_method("mix")
class MixCLMethod(BaseCLMethod):
    """
    Mix Training Method.
    
    This method interleaves training across multiple environments,
    switching to a different environment after each training step.
    All environments share the same LoRA parameters.
    
    Unlike continual learning methods where tasks are trained sequentially,
    this method trains all tasks in parallel through interleaving.
    """
    
    def __init__(self, config: MixConfig):
        super().__init__(config)
        self.config: MixConfig = config
        
        # Track which environment we're currently on
        self.current_env_idx = 0
        self.env_tags = list(config.env_tags)
        self.num_envs = len(self.env_tags)
        
        # Statistics
        self.steps_per_env: Dict[str, int] = {tag: 0 for tag in self.env_tags}
        
        self.log_info(f"Initialized Mix method with {self.num_envs} environments: {self.env_tags}")
    
    def get_current_env_tag(self) -> str:
        """Get the tag of the current environment."""
        return self.env_tags[self.current_env_idx]
    
    def get_current_env_idx(self) -> int:
        """Get the index of the current environment."""
        return self.current_env_idx
    
    def advance_to_next_env(self) -> str:
        """
        Move to the next environment in the cycle.
        Returns the tag of the new current environment.
        """
        # Update statistics for current env
        current_tag = self.get_current_env_tag()
        self.steps_per_env[current_tag] += 1
        
        # Advance to next environment
        self.current_env_idx = (self.current_env_idx + 1) % self.num_envs
        new_tag = self.get_current_env_tag()
        
        self.log_info(f"Switched from {current_tag} to {new_tag}")
        return new_tag
    
    def get_env_config_for_current(self) -> Dict[str, Any]:
        """
        Get the environment configuration for the current environment.
        Used to configure ES Manager for each step.
        """
        idx = self.current_env_idx
        return {
            'tag': self.env_tags[idx],
            'train_n_groups': [self.config.train_n_groups[idx]],
            'val_n_groups': [self.config.val_n_groups[idx]],
        }
    
    def on_task_start(self, task_idx: int, task_name: str,
                      prev_checkpoint_path: Optional[str] = None) -> None:
        """
        For mix method, this is called once at the beginning of training.
        task_idx should always be 0.
        """
        super().on_task_start(task_idx, task_name, prev_checkpoint_path)
        self.log_info(f"Starting mix training with environments: {self.env_tags}")
    
    def on_task_end(self, task_idx: int, task_name: str,
                    checkpoint_path: str) -> None:
        """
        For mix method, this is called once at the end of training.
        """
        super().on_task_end(task_idx, task_name, checkpoint_path)
        self.log_info(f"Mix training completed. Steps per environment: {self.steps_per_env}")
    
    def get_cl_loss_config(self) -> Dict[str, Any]:
        """
        Return CL loss configuration.
        For mix method, there is no additional CL loss.
        """
        return {
            'method_name': 'mix',
            'current_env_idx': self.current_env_idx,
            'current_env_tag': self.get_current_env_tag(),
            'has_cl_loss': False,
            'cl_loss_weight': 0.0,
            'frozen_lora_params': None,
        }
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get the current state as a dictionary."""
        state = super().get_state_dict()
        state.update({
            'current_env_idx': self.current_env_idx,
            'steps_per_env': self.steps_per_env.copy(),
            'env_tags': self.env_tags,
        })
        return state
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load state from a dictionary."""
        super().load_state_dict(state)
        if state:
            self.current_env_idx = state.get('current_env_idx', 0)
            self.steps_per_env = state.get('steps_per_env', {tag: 0 for tag in self.env_tags})
            self.env_tags = state.get('env_tags', self.env_tags)
            self.num_envs = len(self.env_tags)
    
    def get_method_info(self) -> Dict[str, Any]:
        """Return information about the method."""
        return {
            'name': 'mix',
            'description': 'Multi-task interleaved training with shared LoRA',
            'has_cl_loss': False,
            'current_env_idx': self.current_env_idx,
            'current_env_tag': self.get_current_env_tag(),
            'num_envs': self.num_envs,
            'env_tags': self.env_tags,
            'steps_per_env': self.steps_per_env,
        }
