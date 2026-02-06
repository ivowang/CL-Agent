"""
Naive/Baseline Continual Learning Method

This is the baseline method where all tasks share the same LoRA parameters.
No special handling for catastrophic forgetting.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .base import BaseCLMethod, CLMethodConfig
from .registry import register_cl_method


@dataclass
class NaiveCLConfig(CLMethodConfig):
    """Configuration for Naive/Baseline CL method."""
    name: str = "baseline"  # Can also be 'naive'


# Register as both 'baseline' and 'naive' for compatibility
@register_cl_method("baseline")
@register_cl_method("naive")
class NaiveCLMethod(BaseCLMethod):
    """
    Naive/Baseline Continual Learning Method.
    
    Simply continues training the same LoRA parameters across all tasks.
    This serves as a baseline to measure the effectiveness of other CL methods
    in mitigating catastrophic forgetting.
    """
    
    def __init__(self, config: NaiveCLConfig):
        super().__init__(config)
        self.log_info("Initialized Baseline CL method (shared LoRA)")
    
    def on_task_start(self, task_idx: int, task_name: str,
                      prev_checkpoint_path: Optional[str] = None) -> None:
        """Called at the beginning of each task."""
        super().on_task_start(task_idx, task_name, prev_checkpoint_path)
        self.log_info(f"Baseline method: continuing with existing LoRA parameters")
    
    def on_task_end(self, task_idx: int, task_name: str,
                    checkpoint_path: str) -> None:
        """Called at the end of each task."""
        super().on_task_end(task_idx, task_name, checkpoint_path)
    
    def get_cl_loss_config(self) -> Dict[str, Any]:
        """
        Return CL loss configuration.
        For baseline method, there is no additional loss.
        """
        return {
            'method_name': 'baseline',
            'current_task_idx': self.current_task_idx,
            'has_cl_loss': False,
            'cl_loss_weight': 0.0,
            'frozen_lora_params': None,
        }
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get the current state as a dictionary."""
        state = super().get_state_dict()
        return state
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load state from a dictionary."""
        super().load_state_dict(state)
    
    def get_method_info(self) -> Dict[str, Any]:
        """Return information about the method."""
        return {
            'name': 'baseline',
            'description': 'Baseline method with shared LoRA parameters',
            'has_cl_loss': False,
            'current_task': self.current_task_idx,
            'task_history': self.task_history
        }
