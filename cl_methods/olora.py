"""
O-LoRA: Orthogonal Low-Rank Adaptation for Continual Learning

Based on the paper: "Orthogonal Subspace Learning for Language Model Continual Learning"
(Wang et al., 2023) - https://arxiv.org/abs/2310.14152

Key idea: Learn new tasks in orthogonal subspaces to minimize interference with
previously learned tasks. Each task has its OWN LoRA parameters, and orthogonal
constraints are applied between the current task's LoRA and all previous tasks' LoRAs.

Architecture:
    Task 0: LoRA_0 (A_0, B_0) - trained, then frozen
    Task 1: LoRA_1 (A_1, B_1) - trained with orthogonal constraint to A_0
    Task 2: LoRA_2 (A_2, B_2) - trained with orthogonal constraint to A_0, A_1
    ...

Forward pass:
    output = W @ x + Σ_i (B_i @ A_i @ x) * scaling

Orthogonal loss:
    L_ortho = λ_ortho * Σ_i |A_frozen[i] @ A_current.T|

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
from .multi_lora import MultiLoRAManager, extract_lora_params_from_model


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

    This method learns each task with its OWN LoRA parameters by:
    1. For task 0: Train LoRA_0, then freeze it
    2. For task i > 0:
       - Reinitialize new LoRA_i parameters
       - Train with orthogonal constraint: L_ortho = Σ_j<i |A_j @ A_i.T|
       - Freeze LoRA_i after training

    The orthogonal loss ensures that new task's LoRA operates in a subspace
    orthogonal to all previous tasks, minimizing catastrophic forgetting.

    Forward pass combines all task LoRAs:
        output = W @ x + Σ_i (B_i @ A_i @ x) * scaling
    """

    def __init__(self, config: OLoRAConfig):
        super().__init__(config)
        self.config: OLoRAConfig = config

        # Multi-LoRA manager for handling multiple task LoRAs
        self.multi_lora_manager = MultiLoRAManager(
            method='olora',
            lora_rank=config.lora_rank_per_task,
            lora_alpha=config.lora_rank_per_task,  # Usually alpha = rank
        )

        # Store paths to LoRA checkpoints from previous tasks
        self.task_checkpoints: Dict[int, str] = {}

        # Track accumulated rank from previous tasks
        self.accumulated_rank = 0

        self.log_info(f"Initialized O-LoRA with λ_ortho={config.lambda_ortho}, λ_l2={config.lambda_l2}")

    def on_task_start(self, task_idx: int, task_name: str,
                      prev_checkpoint_path: Optional[str] = None) -> None:
        """
        Called at the beginning of each task.

        For O-LoRA:
        1. If task_idx > 0, load and freeze previous task's LoRA
        2. Signal that current LoRA should be reinitialized
        """
        super().on_task_start(task_idx, task_name, prev_checkpoint_path)

        self.multi_lora_manager.set_current_task(task_idx)

        if task_idx > 0 and prev_checkpoint_path:
            # Store the checkpoint path from the previous task
            self.task_checkpoints[task_idx - 1] = prev_checkpoint_path
            self.accumulated_rank += self.config.lora_rank_per_task

            # Load and freeze previous task's LoRA parameters
            success = self.multi_lora_manager.load_frozen_loras_from_checkpoint(
                prev_checkpoint_path,
                task_idx - 1,
                device=torch.device('cpu')
            )

            if success:
                self.log_info(f"Loaded and froze LoRA from task {task_idx - 1}")
            else:
                self.log_info(f"Warning: Could not load LoRA from task {task_idx - 1}")

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

        # Save multi-LoRA manager state
        manager_path = os.path.join(checkpoint_path, 'multi_lora_manager.pt')
        self.multi_lora_manager.save(manager_path)

    def get_cl_loss_config(self) -> Dict[str, Any]:
        """
        Return configuration for O-LoRA loss computation.
        This config will be passed to the worker for loss computation.

        For O-LoRA, the worker needs:
        - All frozen LoRA params from previous tasks
        - Lambda values for loss computation
        - Flag to reinitialize LoRA for new task
        """
        # Collect all frozen LoRA params
        all_frozen_lora_params = {}
        for task_idx, task_lora in self.multi_lora_manager.frozen_task_loras.items():
            all_frozen_lora_params[task_idx] = {
                'A': task_lora.lora_A,
                'B': task_lora.lora_B,
            }

        # For backward compatibility, also provide the most recent frozen params
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
            'method_name': 'olora',
            'current_task_idx': self.current_task_idx,
            'has_cl_loss': self.current_task_idx > 0,  # Only after first task
            'lambda_ortho': self.config.lambda_ortho,
            'lambda_l2': self.config.lambda_l2,
            'accumulated_rank': self.accumulated_rank,
            'task_checkpoints': self.task_checkpoints.copy(),
            'reinit_lora_per_task': self.config.reinit_lora_per_task,
            # All frozen LoRA params from all previous tasks
            'all_frozen_lora_params': all_frozen_lora_params,
            # For backward compatibility
            'frozen_lora_params': frozen_lora_params,
            # Multi-LoRA manager state for forward pass
            'multi_lora_manager_state': self.multi_lora_manager.get_state_dict(),
        }

    def get_state_dict(self) -> Dict[str, Any]:
        """Get the current state as a dictionary (for passing between tasks)."""
        state = super().get_state_dict()
        state.update({
            'accumulated_rank': self.accumulated_rank,
            'task_checkpoints': self.task_checkpoints.copy(),
            'lambda_ortho': self.config.lambda_ortho,
            'lambda_l2': self.config.lambda_l2,
            'multi_lora_manager_state': self.multi_lora_manager.get_state_dict(),
        })
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load state from a dictionary."""
        super().load_state_dict(state)
        if state:
            self.accumulated_rank = state.get('accumulated_rank', 0)
            self.task_checkpoints = state.get('task_checkpoints', {})

            # Load multi-LoRA manager state
            manager_state = state.get('multi_lora_manager_state')
            if manager_state:
                self.multi_lora_manager.load_state_dict(manager_state)

    def save_method_state(self, checkpoint_path: str,
                          additional_state: Optional[Dict] = None) -> str:
        """Save O-LoRA specific state."""
        state = additional_state or {}
        state['accumulated_rank'] = self.accumulated_rank
        state['task_checkpoints'] = self.task_checkpoints
        state['lambda_ortho'] = self.config.lambda_ortho
        state['lambda_l2'] = self.config.lambda_l2
        state['multi_lora_manager_state'] = self.multi_lora_manager.get_state_dict()

        # Also save multi-LoRA manager separately
        manager_path = os.path.join(checkpoint_path, 'multi_lora_manager.pt')
        self.multi_lora_manager.save(manager_path)

        return super().save_method_state(checkpoint_path, state)

    def load_method_state(self, checkpoint_path: str) -> Optional[Dict]:
        """Load O-LoRA specific state."""
        state = super().load_method_state(checkpoint_path)

        if state:
            self.accumulated_rank = state.get('accumulated_rank', 0)
            self.task_checkpoints = state.get('task_checkpoints', {})

            # Load multi-LoRA manager state
            manager_state = state.get('multi_lora_manager_state')
            if manager_state:
                self.multi_lora_manager.load_state_dict(manager_state)

            self.log_info(f"Loaded O-LoRA state: accumulated_rank={self.accumulated_rank}, "
                          f"frozen_tasks={len(self.multi_lora_manager.frozen_task_loras)}")

        # Also try to load from separate file
        manager_path = os.path.join(checkpoint_path, 'multi_lora_manager.pt')
        if os.path.exists(manager_path):
            self.multi_lora_manager.load(manager_path)

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
            'num_frozen_tasks': len(self.multi_lora_manager.frozen_task_loras),
            'task_history': self.task_history
        }
