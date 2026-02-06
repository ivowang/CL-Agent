"""
MoE (Mixture of Experts) Method - Independent LoRA per Task

This method trains independent LoRA modules for each task, where each task
has its own expert (LoRA adapter). During training, all tasks are sampled
in a mixed manner (like Mix training), but each task uses its own LoRA
for both sampling and parameter updates.

Key features:
- 9 independent LoRA modules (one per task)
- 9 independent actor workers (one per task)
- 9 independent critic workers (one per task)
- Serial sampling: sample from each task sequentially with its own LoRA
- Serial updates: update each task's LoRA sequentially
- Each task is essentially trained independently, but in a mixed sampling manner

This is fundamentally different from:
- Mix: All tasks share one LoRA
- Continual Learning: Tasks trained sequentially, not mixed
"""

import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from .base import BaseCLMethod, CLMethodConfig
from .registry import register_cl_method


@dataclass
class MoEConfig(CLMethodConfig):
    """Configuration for MoE method."""
    name: str = "moe"

    # Number of tasks (experts)
    num_tasks: int = 9

    # LoRA configuration per task
    lora_rank: int = 64
    lora_alpha: int = 64

    # Task configurations
    task_names: List[str] = field(default_factory=lambda: [
        "bandit_low", "bandit_medium", "bandit_hard",
        "sokoban_low", "sokoban_medium", "sokoban_hard",
        "frozen_lake_low", "frozen_lake_medium", "frozen_lake_hard"
    ])


@register_cl_method("moe")
class MoECLMethod(BaseCLMethod):
    """
    MoE (Mixture of Experts) Method.

    This method maintains independent LoRA modules for each task (expert).
    Each task has its own actor and critic workers with dedicated LoRA parameters.

    Training flow:
    1. Serial sampling: For each task, sample trajectories using that task's LoRA
    2. Serial updates: For each task, update that task's LoRA with its trajectories

    This ensures complete task independence while using mixed sampling.
    """

    def __init__(self, config: MoEConfig):
        super().__init__(config)
        self.config: MoEConfig = config

        # Validate task names
        if len(config.task_names) != config.num_tasks:
            raise ValueError(f"Number of task names ({len(config.task_names)}) "
                           f"must match num_tasks ({config.num_tasks})")

        self.task_names = list(config.task_names)
        self.num_tasks = config.num_tasks

        # Track training statistics per task
        self.task_steps: Dict[int, int] = {i: 0 for i in range(self.num_tasks)}
        self.total_steps = 0

        self.log_info(f"Initialized MoE method with {self.num_tasks} independent experts (tasks)")
        self.log_info(f"Task names: {self.task_names}")

    def get_num_tasks(self) -> int:
        """Get the number of tasks (experts)."""
        return self.num_tasks

    def get_task_names(self) -> List[str]:
        """Get all task names."""
        return self.task_names.copy()

    def get_task_name(self, task_idx: int) -> str:
        """Get the name of a specific task."""
        if 0 <= task_idx < self.num_tasks:
            return self.task_names[task_idx]
        raise ValueError(f"Invalid task_idx: {task_idx}, must be in [0, {self.num_tasks})")

    def record_step(self, task_idx: int) -> None:
        """Record that a training step has been completed for a task."""
        if 0 <= task_idx < self.num_tasks:
            self.task_steps[task_idx] += 1
        self.total_steps += 1

    def on_task_start(self, task_idx: int, task_name: str,
                      prev_checkpoint_path: Optional[str] = None) -> None:
        """
        Called at the beginning of training.
        For MoE, this is called once with task_idx=0.
        """
        super().on_task_start(task_idx, task_name, prev_checkpoint_path)
        self.log_info(f"Starting MoE training with {self.num_tasks} independent experts")

    def on_task_end(self, task_idx: int, task_name: str,
                    checkpoint_path: str) -> None:
        """
        Called at the end of training.
        For MoE, this is called once with task_idx=0.
        """
        super().on_task_end(task_idx, task_name, checkpoint_path)
        self.log_info(f"MoE training completed. Total steps: {self.total_steps}")
        self.log_info(f"Steps per task: {self.task_steps}")

    def get_cl_loss_config(self) -> Dict[str, Any]:
        """
        Return CL loss configuration.
        For MoE method, there is no additional CL loss.
        """
        return {
            'method_name': 'moe',
            'num_tasks': self.num_tasks,
            'task_names': self.task_names,
            'has_cl_loss': False,
            'cl_loss_weight': 0.0,
            'frozen_lora_params': None,
        }

    def get_state_dict(self) -> Dict[str, Any]:
        """Get the current state as a dictionary."""
        state = super().get_state_dict()
        state.update({
            'num_tasks': self.num_tasks,
            'task_names': self.task_names,
            'task_steps': self.task_steps.copy(),
            'total_steps': self.total_steps,
        })
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load state from a dictionary."""
        super().load_state_dict(state)
        if state:
            self.num_tasks = state.get('num_tasks', self.num_tasks)
            self.task_names = state.get('task_names', self.task_names)
            self.task_steps = state.get('task_steps', self.task_steps)
            self.total_steps = state.get('total_steps', 0)

    def save_method_state(self, checkpoint_path: str,
                          additional_state: Optional[Dict] = None) -> str:
        """Save MoE specific state."""
        state = additional_state or {}
        state['num_tasks'] = self.num_tasks
        state['task_names'] = self.task_names
        state['task_steps'] = self.task_steps
        state['total_steps'] = self.total_steps

        return super().save_method_state(checkpoint_path, state)

    def load_method_state(self, checkpoint_path: str) -> Optional[Dict]:
        """Load MoE specific state."""
        state = super().load_method_state(checkpoint_path)

        if state:
            self.num_tasks = state.get('num_tasks', self.num_tasks)
            self.task_names = state.get('task_names', self.task_names)
            self.task_steps = state.get('task_steps', self.task_steps)
            self.total_steps = state.get('total_steps', 0)

            self.log_info(f"Loaded MoE state: {self.num_tasks} tasks, {self.total_steps} total steps")

        return state

    def get_method_info(self) -> Dict[str, Any]:
        """Return information about the method."""
        return {
            'name': 'moe',
            'description': 'Mixture of Experts with independent LoRA per task',
            'has_cl_loss': False,
            'num_tasks': self.num_tasks,
            'task_names': self.task_names,
            'task_steps': self.task_steps,
            'total_steps': self.total_steps,
        }
