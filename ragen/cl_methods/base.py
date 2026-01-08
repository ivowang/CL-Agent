"""
Base class for Continual Learning Methods

Note: In distributed training (Ray/FSDP), the trainer cannot directly access
the model. CL methods should work with checkpoint paths and state dicts,
not direct model references.
"""

import os
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


@dataclass
class CLMethodConfig:
    """Base configuration for CL methods."""
    name: str = "base"
    # Task information
    current_task_idx: int = 0
    total_tasks: int = 3
    task_names: List[str] = field(default_factory=list)
    # LoRA configuration
    lora_rank: int = 64
    lora_alpha: int = 64
    # Checkpoint paths
    checkpoint_base_dir: str = "checkpoints/continual_learning"


class BaseCLMethod(ABC):
    """
    Base class for Continual Learning methods.
    
    IMPORTANT: In distributed training, the trainer cannot directly access 
    the model. Methods should work with:
    - checkpoint_path: Path to saved model checkpoint
    - State dicts loaded from checkpoints
    
    Each CL method should implement:
    - on_task_start: Called when starting a new task (no model access)
    - on_task_end: Called when finishing a task (no model access)
    - get_cl_loss_config: Return config for CL-specific loss (computed in worker)
    """
    
    def __init__(self, config: CLMethodConfig):
        self.config = config
        self.current_task_idx = config.current_task_idx
        self.task_history: List[Dict[str, Any]] = []
        
    @property
    def name(self) -> str:
        """Return the name of the CL method."""
        return self.config.name
    
    def on_task_start(self, task_idx: int, task_name: str, 
                      prev_checkpoint_path: Optional[str] = None) -> None:
        """
        Called at the beginning of each task.
        
        Args:
            task_idx: Index of the current task (0-indexed)
            task_name: Name of the current task
            prev_checkpoint_path: Path to checkpoint from previous task (if any)
        """
        self.current_task_idx = task_idx
        self.log_info(f"Starting task {task_idx}: {task_name}")
        if prev_checkpoint_path:
            self.log_info(f"Previous checkpoint: {prev_checkpoint_path}")
        
        self.task_history.append({
            'task_idx': task_idx,
            'task_name': task_name,
            'status': 'started',
            'prev_checkpoint': prev_checkpoint_path
        })
    
    def on_task_end(self, task_idx: int, task_name: str,
                    checkpoint_path: str) -> None:
        """
        Called at the end of each task.
        
        Args:
            task_idx: Index of the current task
            task_name: Name of the current task
            checkpoint_path: Path where checkpoint was saved
        """
        self.log_info(f"Completed task {task_idx}: {task_name}")
        self.log_info(f"Checkpoint saved at: {checkpoint_path}")
        
        # Update task history
        for task in self.task_history:
            if task['task_idx'] == task_idx:
                task['status'] = 'completed'
                task['checkpoint_path'] = checkpoint_path
    
    def get_cl_loss_config(self) -> Dict[str, Any]:
        """
        Return configuration for CL-specific loss computation.
        This config will be passed to the worker for loss computation.
        
        Returns:
            Dict with loss configuration (e.g., lambda values, frozen params)
        """
        return {
            'method_name': self.name,
            'current_task_idx': self.current_task_idx,
            'has_cl_loss': False,
        }
    
    def get_checkpoint_path(self, task_idx: int, task_name: str) -> str:
        """Get the checkpoint path for a specific task."""
        return os.path.join(
            self.config.checkpoint_base_dir,
            f"task{task_idx}_{task_name}"
        )
    
    def save_method_state(self, checkpoint_path: str,
                          additional_state: Optional[Dict] = None) -> str:
        """
        Save CL method-specific state to a file.
        
        Args:
            checkpoint_path: Base checkpoint path
            additional_state: Additional state to save
            
        Returns:
            Path where the state was saved
        """
        os.makedirs(checkpoint_path, exist_ok=True)
        
        state = {
            'method_name': self.name,
            'current_task_idx': self.current_task_idx,
            'task_history': self.task_history,
            'config': {
                'name': self.config.name,
                'lora_rank': self.config.lora_rank,
            }
        }
        if additional_state:
            state.update(additional_state)
        
        state_path = os.path.join(checkpoint_path, 'cl_method_state.pt')
        torch.save(state, state_path)
        self.log_info(f"Saved CL method state to {state_path}")
        
        return state_path
    
    def load_method_state(self, checkpoint_path: str) -> Optional[Dict]:
        """
        Load CL method-specific state from a file.
        
        Args:
            checkpoint_path: Base checkpoint path
            
        Returns:
            The loaded state dict, or None if not found
        """
        state_path = os.path.join(checkpoint_path, 'cl_method_state.pt')
        
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location='cpu')
            self.task_history = state.get('task_history', [])
            self.log_info(f"Loaded CL method state from {state_path}")
            return state
        return None
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get the current state as a dictionary (for passing between tasks)."""
        return {
            'method_name': self.name,
            'current_task_idx': self.current_task_idx,
            'task_history': self.task_history,
        }
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load state from a dictionary."""
        if state:
            self.task_history = state.get('task_history', [])
            # Don't override current_task_idx, as it's set by the trainer
    
    def log_info(self, message: str):
        """Log information with CL method prefix."""
        print(f"[CL-{self.name}] {message}")
