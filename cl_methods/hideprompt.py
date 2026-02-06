"""
HiDE-Prompt: Hierarchical Decomposition of Prompt-Based Continual Learning

Based on the paper: "Hierarchical Decomposition of Prompt-Based Continual Learning:
Rethinking Obscured Sub-optimality" (NeurIPS 2023)

This implementation adapts HiDE-Prompt for decoder-only LLMs in the CL-Agent framework.

Key components:
1. Within-Task Prediction (WTP): Train task-specific prompts
2. Contrastive Regularization: Encourage orthogonality between task features
3. Trajectory Feature Storage: Store successful trajectory features for each task

Note: We skip Task-Identity Inference (TII) since task IDs are explicitly provided.
"""

import os
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from .base import BaseCLMethod, CLMethodConfig
from .registry import register_cl_method
from .prompt_modules import EPrompt, inject_prompts_into_model, remove_prompts_from_model
from .trajectory_storage import TrajectoryFeatureStorage, compute_contrastive_loss


@dataclass
class HiDEPromptConfig(CLMethodConfig):
    """Configuration for HiDE-Prompt CL method."""
    name: str = "hideprompt"

    # Prompt parameters
    prompt_length: int = 5  # Length of each task-specific prompt
    prompt_init: str = 'uniform'  # Initialization method ('uniform' or 'zero')
    prompt_momentum: float = 0.01  # Momentum for prompt averaging (0.0 = no momentum)

    # Contrastive regularization parameters
    reg_weight: float = 0.1  # Weight for contrastive regularization loss
    temperature: float = 0.8  # Temperature for contrastive loss

    # Feature storage parameters
    storage_method: str = 'multi-centroid'  # 'variance', 'covariance', or 'multi-centroid'
    n_centroids: int = 10  # Number of centroids for multi-centroid method

    # Feature extraction parameters
    extract_layer: str = 'last'  # Which layer to extract features from ('last' or layer index)
    feature_dim: Optional[int] = None  # Feature dimension (auto-detected if None)


@register_cl_method("hideprompt")
class HiDEPromptCLMethod(BaseCLMethod):
    """
    HiDE-Prompt Continual Learning Method.

    This method uses task-specific prompts (prefix-tuning style) combined with
    contrastive regularization to prevent catastrophic forgetting.

    Training flow:
    1. For each task, train a task-specific prompt
    2. During training, apply contrastive regularization to encourage orthogonality
       between current features and stored features from previous tasks
    3. After each task, store trajectory features from successful episodes
    4. At inference, use the appropriate task-specific prompt
    """

    def __init__(self, config: HiDEPromptConfig):
        super().__init__(config)
        self.config: HiDEPromptConfig = config

        # E-Prompt module (will be created when model is available)
        self.e_prompt: Optional[EPrompt] = None
        self.prompt_injected = False

        # Trajectory feature storage
        self.feature_storage = TrajectoryFeatureStorage(
            storage_method=config.storage_method,
            n_centroids=config.n_centroids,
        )

        # Track which tasks have been trained
        self.trained_tasks: List[int] = []

        self.log_info(f"Initialized HiDE-Prompt with prompt_length={config.prompt_length}, "
                      f"reg_weight={config.reg_weight}, storage_method={config.storage_method}")

    def on_task_start(self, task_idx: int, task_name: str,
                      prev_checkpoint_path: Optional[str] = None) -> None:
        """
        Called at the beginning of each task.

        For HiDE-Prompt:
        1. Load stored features from previous checkpoint if available
        2. Initialize prompt for current task (already done in e_prompt)
        """
        super().on_task_start(task_idx, task_name, prev_checkpoint_path)

        # Load feature storage from previous checkpoint
        if prev_checkpoint_path and task_idx > 0:
            storage_path = os.path.join(prev_checkpoint_path, 'trajectory_features.pt')
            if os.path.exists(storage_path):
                try:
                    self.feature_storage.load(storage_path)
                    self.log_info(f"Loaded trajectory features from {storage_path}")
                    self.log_info(f"Stored features for tasks: {self.feature_storage.get_all_task_ids()}")
                except Exception as e:
                    self.log_info(f"Warning: Could not load trajectory features: {e}")

        self.log_info(f"HiDE-Prompt: Starting task {task_idx} with prompt momentum={self.config.prompt_momentum}")

    def on_task_end(self, task_idx: int, task_name: str,
                    checkpoint_path: str) -> None:
        """
        Called at the end of each task.

        For HiDE-Prompt:
        1. Mark task as trained
        2. Save trajectory features
        """
        super().on_task_end(task_idx, task_name, checkpoint_path)

        # Mark task as trained
        if task_idx not in self.trained_tasks:
            self.trained_tasks.append(task_idx)

        # Save trajectory features
        os.makedirs(checkpoint_path, exist_ok=True)
        storage_path = os.path.join(checkpoint_path, 'trajectory_features.pt')
        self.feature_storage.save(storage_path)
        self.log_info(f"Saved trajectory features to {storage_path}")

    def add_trajectory_features(
        self,
        task_idx: int,
        features: torch.Tensor,
    ) -> None:
        """
        Add trajectory features for a task.

        This should be called after collecting features from successful trajectories.

        Args:
            task_idx: Task ID
            features: Feature tensor of shape (num_samples, feature_dim)
        """
        self.feature_storage.add_task_features(task_idx, features)
        self.log_info(f"Added {features.shape[0]} trajectory features for task {task_idx}")

    def get_cl_loss_config(self) -> Dict[str, Any]:
        """
        Return configuration for HiDE-Prompt loss computation.

        This config will be passed to the worker for computing contrastive loss.
        """
        # Get stored means for contrastive loss
        stored_means = None
        if self.current_task_idx > 0:
            stored_means = self.feature_storage.get_all_means()

        return {
            'method_name': 'hideprompt',
            'current_task_idx': self.current_task_idx,
            'has_cl_loss': True,
            'cl_loss_weight': self.config.reg_weight,

            # Prompt parameters
            'prompt_length': self.config.prompt_length,
            'prompt_momentum': self.config.prompt_momentum,

            # Contrastive loss parameters
            'reg_weight': self.config.reg_weight,
            'temperature': self.config.temperature,
            'stored_means': stored_means,

            # Feature extraction parameters
            'extract_layer': self.config.extract_layer,
            'feature_dim': self.config.feature_dim,

            # Feature storage info
            'storage_method': self.config.storage_method,
            'n_centroids': self.config.n_centroids,
        }

    def inject_prompts(self, model: nn.Module, num_tasks: int) -> None:
        """
        Inject prompts into the model.

        This should be called once when the model is first created.

        Args:
            model: The model to inject prompts into
            num_tasks: Total number of tasks
        """
        if self.prompt_injected:
            self.log_info("Prompts already injected, skipping")
            return

        try:
            self.e_prompt = inject_prompts_into_model(
                model=model,
                num_tasks=num_tasks,
                prompt_length=self.config.prompt_length,
                embed_dim=self.config.feature_dim,
                prompt_init=self.config.prompt_init,
            )
            self.prompt_injected = True
            self.log_info(f"Successfully injected prompts into model (num_tasks={num_tasks})")
        except Exception as e:
            self.log_info(f"Error injecting prompts: {e}")
            raise

    def remove_prompts(self, model: nn.Module) -> None:
        """Remove prompts from the model."""
        if not self.prompt_injected:
            return

        try:
            remove_prompts_from_model(model)
            self.prompt_injected = False
            self.log_info("Removed prompts from model")
        except Exception as e:
            self.log_info(f"Error removing prompts: {e}")

    def get_state_dict(self) -> Dict[str, Any]:
        """Get the current state as a dictionary."""
        state = super().get_state_dict()
        state.update({
            'trained_tasks': self.trained_tasks.copy(),
            'prompt_injected': self.prompt_injected,
            'feature_storage': {
                'storage_method': self.feature_storage.storage_method,
                'n_centroids': self.feature_storage.n_centroids,
                'task_features': self.feature_storage.task_features,
            },
        })

        # Save E-Prompt state if available
        if self.e_prompt is not None:
            state['e_prompt_state'] = self.e_prompt.state_dict()

        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load state from a dictionary."""
        super().load_state_dict(state)

        if state:
            self.trained_tasks = state.get('trained_tasks', [])
            self.prompt_injected = state.get('prompt_injected', False)

            # Load feature storage
            if 'feature_storage' in state:
                fs_state = state['feature_storage']
                self.feature_storage.storage_method = fs_state['storage_method']
                self.feature_storage.n_centroids = fs_state['n_centroids']
                self.feature_storage.task_features = fs_state['task_features']

            # Load E-Prompt state if available
            if 'e_prompt_state' in state and self.e_prompt is not None:
                self.e_prompt.load_state_dict(state['e_prompt_state'])

    def save_method_state(self, checkpoint_path: str,
                          additional_state: Optional[Dict] = None) -> str:
        """Save HiDE-Prompt specific state."""
        state = additional_state or {}
        state['trained_tasks'] = self.trained_tasks
        state['prompt_injected'] = self.prompt_injected

        # Save E-Prompt state
        if self.e_prompt is not None:
            e_prompt_path = os.path.join(checkpoint_path, 'e_prompt.pt')
            torch.save(self.e_prompt.state_dict(), e_prompt_path)
            self.log_info(f"Saved E-Prompt state to {e_prompt_path}")

        # Save feature storage (already done in on_task_end)

        return super().save_method_state(checkpoint_path, state)

    def load_method_state(self, checkpoint_path: str) -> Optional[Dict]:
        """Load HiDE-Prompt specific state."""
        state = super().load_method_state(checkpoint_path)

        if state:
            self.trained_tasks = state.get('trained_tasks', [])
            self.prompt_injected = state.get('prompt_injected', False)

        # Load E-Prompt state
        e_prompt_path = os.path.join(checkpoint_path, 'e_prompt.pt')
        if os.path.exists(e_prompt_path) and self.e_prompt is not None:
            try:
                self.e_prompt.load_state_dict(torch.load(e_prompt_path, map_location='cpu'))
                self.log_info(f"Loaded E-Prompt state from {e_prompt_path}")
            except Exception as e:
                self.log_info(f"Warning: Could not load E-Prompt state: {e}")

        # Load feature storage (already done in on_task_start)

        return state

    def get_method_info(self) -> Dict[str, Any]:
        """Return information about the method."""
        return {
            'name': 'hideprompt',
            'description': 'HiDE-Prompt: Hierarchical Decomposition of Prompt-Based Continual Learning',
            'has_cl_loss': True,
            'prompt_length': self.config.prompt_length,
            'prompt_momentum': self.config.prompt_momentum,
            'reg_weight': self.config.reg_weight,
            'storage_method': self.config.storage_method,
            'current_task': self.current_task_idx,
            'trained_tasks': self.trained_tasks,
            'num_stored_tasks': len(self.feature_storage.get_all_task_ids()),
            'task_history': self.task_history,
        }
