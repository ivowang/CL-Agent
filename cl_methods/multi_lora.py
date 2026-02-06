"""
Multi-LoRA Manager for Continual Learning

This module provides utilities for managing multiple LoRA adapters across tasks
in continual learning scenarios. It supports:
- O-LoRA: Orthogonal constraints between task-specific LoRA modules
- SD-LoRA: Learnable scaling factors for combining multiple LoRA outputs

The key insight is that for true continual learning with LoRA:
1. Each task should have its OWN LoRA parameters (not shared)
2. Previous task LoRA parameters should be FROZEN
3. The forward pass should COMBINE outputs from all task LoRAs

Architecture:
    Task 0: LoRA_0 (A_0, B_0) - trained, then frozen
    Task 1: LoRA_1 (A_1, B_1) - trained, then frozen
    Task 2: LoRA_2 (A_2, B_2) - currently training
    ...

Forward pass (SD-LoRA style):
    output = W @ x + Σ_i scaling_factor[i] * (B_i @ A_i @ x)

Forward pass (O-LoRA style):
    output = W @ x + Σ_i (B_i @ A_i @ x)  # with orthogonal constraint on A matrices
"""

import os
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class TaskLoRAParams:
    """Container for a single task's LoRA parameters."""
    task_idx: int
    lora_A: Dict[str, torch.Tensor]  # module_name -> A matrix
    lora_B: Dict[str, torch.Tensor]  # module_name -> B matrix
    scaling_factor: float = 1.0  # For SD-LoRA
    is_frozen: bool = False

    def to_device(self, device: torch.device) -> 'TaskLoRAParams':
        """Move all tensors to specified device."""
        return TaskLoRAParams(
            task_idx=self.task_idx,
            lora_A={k: v.to(device) for k, v in self.lora_A.items()},
            lora_B={k: v.to(device) for k, v in self.lora_B.items()},
            scaling_factor=self.scaling_factor,
            is_frozen=self.is_frozen
        )


class MultiLoRAManager:
    """
    Manages multiple LoRA adapters for continual learning.

    This class handles:
    1. Storing frozen LoRA parameters from previous tasks
    2. Computing combined LoRA outputs in forward pass
    3. Computing orthogonal loss (O-LoRA) or scaling factor loss (SD-LoRA)
    4. Saving/loading multi-task LoRA checkpoints
    """

    def __init__(
        self,
        method: str = 'olora',  # 'olora' or 'sdlora'
        lora_rank: int = 64,
        lora_alpha: int = 64,
        scaling_factor_init: float = 0.8,
        normalize_lora: bool = True,
        max_tasks: int = 20,
    ):
        self.method = method
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / lora_rank
        self.scaling_factor_init = scaling_factor_init
        self.normalize_lora = normalize_lora
        self.max_tasks = max_tasks

        # Storage for frozen task LoRA parameters
        self.frozen_task_loras: Dict[int, TaskLoRAParams] = {}

        # Learnable scaling factors for SD-LoRA (stored as nn.ParameterList in model)
        # Here we just track the values
        self.scaling_factors: Dict[int, float] = {}

        # Current task index
        self.current_task_idx: int = 0

        # Module names that have LoRA applied
        self.lora_module_names: List[str] = []

    def set_current_task(self, task_idx: int) -> None:
        """Set the current task index."""
        self.current_task_idx = task_idx
        # Initialize scaling factor for new task
        if task_idx not in self.scaling_factors:
            self.scaling_factors[task_idx] = self.scaling_factor_init

    def freeze_task_lora(
        self,
        task_idx: int,
        lora_params: Dict[str, Dict[str, torch.Tensor]],
        scaling_factor: Optional[float] = None
    ) -> None:
        """
        Freeze and store LoRA parameters from a completed task.

        Args:
            task_idx: Index of the task to freeze
            lora_params: Dict mapping module_name -> {'A': tensor, 'B': tensor}
            scaling_factor: Optional scaling factor (for SD-LoRA)
        """
        lora_A = {}
        lora_B = {}

        for module_name, params in lora_params.items():
            if params.get('A') is not None:
                # Clone and detach to ensure no gradients
                lora_A[module_name] = params['A'].clone().detach().cpu()
            if params.get('B') is not None:
                lora_B[module_name] = params['B'].clone().detach().cpu()

            # Track module names
            if module_name not in self.lora_module_names:
                self.lora_module_names.append(module_name)

        sf = scaling_factor if scaling_factor is not None else self.scaling_factors.get(task_idx, 1.0)

        self.frozen_task_loras[task_idx] = TaskLoRAParams(
            task_idx=task_idx,
            lora_A=lora_A,
            lora_B=lora_B,
            scaling_factor=sf,
            is_frozen=True
        )

        print(f"[MultiLoRA] Frozen task {task_idx} LoRA params: {len(lora_A)} modules")

    def load_frozen_loras_from_checkpoint(
        self,
        checkpoint_path: str,
        task_idx: int,
        device: torch.device = torch.device('cpu')
    ) -> bool:
        """
        Load frozen LoRA parameters from a checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint directory
            task_idx: Task index for these parameters
            device: Device to load to (default CPU for memory efficiency)

        Returns:
            True if successfully loaded, False otherwise
        """
        from .loss_functions import load_frozen_lora_params_from_checkpoint

        frozen_params = load_frozen_lora_params_from_checkpoint(checkpoint_path, device)

        if frozen_params:
            # Load scaling factor if available
            scaling_path = os.path.join(checkpoint_path, 'scaling_factors.pt')
            sf = self.scaling_factor_init
            if os.path.exists(scaling_path):
                try:
                    sf_dict = torch.load(scaling_path, map_location='cpu')
                    if isinstance(sf_dict, dict) and task_idx in sf_dict:
                        sf = sf_dict[task_idx]
                except Exception as e:
                    print(f"[MultiLoRA] Warning: Could not load scaling factor: {e}")

            self.freeze_task_lora(task_idx, frozen_params, sf)
            return True
        return False

    def compute_frozen_lora_output(
        self,
        x: torch.Tensor,
        module_name: str,
        device: torch.device
    ) -> torch.Tensor:
        """
        Compute the combined output from all frozen task LoRAs for a specific module.

        For O-LoRA: output = Σ_i (B_i @ A_i @ x) * scaling
        For SD-LoRA: output = Σ_i scaling_factor[i] * (B_i @ A_i @ x) / (||B_i|| * ||A_i||)

        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            module_name: Name of the module (to look up correct LoRA params)
            device: Device to compute on

        Returns:
            Combined LoRA output tensor
        """
        if not self.frozen_task_loras:
            return torch.zeros_like(x)

        combined_output = torch.zeros_like(x)

        for task_idx, task_lora in self.frozen_task_loras.items():
            if module_name not in task_lora.lora_A or module_name not in task_lora.lora_B:
                continue

            # Get A and B matrices, move to device
            A = task_lora.lora_A[module_name].to(device)
            B = task_lora.lora_B[module_name].to(device)

            # Compute LoRA output: B @ A @ x
            # A: [r, in_features], B: [out_features, r]
            # x: [batch, seq_len, in_features]
            with torch.no_grad():
                # x @ A.T -> [batch, seq_len, r]
                # then @ B.T -> [batch, seq_len, out_features]
                lora_out = F.linear(F.linear(x, A), B)

                if self.method == 'sdlora' and self.normalize_lora:
                    # Normalize by weight norms
                    norm_A = torch.norm(A)
                    norm_B = torch.norm(B)
                    if norm_A > 0 and norm_B > 0:
                        lora_out = lora_out / (norm_A * norm_B)
                    # Apply scaling factor
                    sf = task_lora.scaling_factor
                    lora_out = lora_out * sf
                else:
                    # O-LoRA: just apply standard scaling
                    lora_out = lora_out * self.scaling

                combined_output = combined_output + lora_out

        return combined_output

    def compute_orthogonal_loss(
        self,
        current_lora_A: Dict[str, torch.Tensor],
        device: torch.device
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute orthogonal loss between current task's LoRA A matrices
        and all frozen tasks' LoRA A matrices.

        O-LoRA loss: L_ortho = Σ_i Σ_j |A_frozen[i] @ A_current.T|

        Args:
            current_lora_A: Dict mapping module_name -> current A matrix
            device: Device to compute on

        Returns:
            Tuple of (loss tensor, metrics dict)
        """
        if not self.frozen_task_loras:
            return torch.tensor(0.0, device=device), {'ortho_loss': 0.0}

        ortho_loss = torch.tensor(0.0, device=device)
        num_pairs = 0

        for module_name, current_A in current_lora_A.items():
            if current_A is None:
                continue

            for task_idx, task_lora in self.frozen_task_loras.items():
                if module_name not in task_lora.lora_A:
                    continue

                frozen_A = task_lora.lora_A[module_name].to(device)

                # Orthogonal loss: |frozen_A @ current_A.T|
                # frozen_A: [r_frozen, in_features]
                # current_A: [r_current, in_features]
                # Result: [r_frozen, r_current]
                ortho_product = torch.mm(frozen_A, current_A.T)
                ortho_loss = ortho_loss + torch.abs(ortho_product).sum()
                num_pairs += 1

        metrics = {
            'ortho_loss': ortho_loss.item(),
            'num_ortho_pairs': num_pairs,
        }

        return ortho_loss, metrics

    def get_state_dict(self) -> Dict[str, Any]:
        """Get state dict for saving."""
        state = {
            'method': self.method,
            'lora_rank': self.lora_rank,
            'lora_alpha': self.lora_alpha,
            'scaling_factor_init': self.scaling_factor_init,
            'normalize_lora': self.normalize_lora,
            'current_task_idx': self.current_task_idx,
            'scaling_factors': self.scaling_factors.copy(),
            'lora_module_names': self.lora_module_names.copy(),
            'frozen_task_loras': {},
        }

        # Save frozen LoRA params
        for task_idx, task_lora in self.frozen_task_loras.items():
            state['frozen_task_loras'][task_idx] = {
                'lora_A': {k: v.cpu() for k, v in task_lora.lora_A.items()},
                'lora_B': {k: v.cpu() for k, v in task_lora.lora_B.items()},
                'scaling_factor': task_lora.scaling_factor,
            }

        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load state dict."""
        self.method = state.get('method', self.method)
        self.lora_rank = state.get('lora_rank', self.lora_rank)
        self.lora_alpha = state.get('lora_alpha', self.lora_alpha)
        self.scaling = self.lora_alpha / self.lora_rank
        self.scaling_factor_init = state.get('scaling_factor_init', self.scaling_factor_init)
        self.normalize_lora = state.get('normalize_lora', self.normalize_lora)
        self.current_task_idx = state.get('current_task_idx', 0)
        self.scaling_factors = state.get('scaling_factors', {})
        self.lora_module_names = state.get('lora_module_names', [])

        # Load frozen LoRA params
        self.frozen_task_loras = {}
        for task_idx, task_data in state.get('frozen_task_loras', ).items():
            task_idx = int(task_idx)  # JSON keys are strings
            self.frozen_task_loras[task_idx] = TaskLoRAParams(
                task_idx=task_idx,
                lora_A=task_data['lora_A'],
                lora_B=task_data['lora_B'],
                scaling_factor=task_data.get('scaling_factor', 1.0),
                is_frozen=True
            )

    def save(self, path: str) -> None:
        """Save manager state to file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save(self.get_state_dict(), path)
        print(f"[MultiLoRA] Saved state to {path}")

    def load(self, path: str) -> None:
        """Load manager state from file."""
        if os.path.exists(path):
            state = torch.load(path, map_location='cpu')
            self.load_state_dict(state)
            print(f"[MultiLoRA] Loaded state from {path}: {len(self.frozen_task_loras)} frozen tasks")
        else:
            print(f"[MultiLoRA] Warning: State file not found: {path}")


def extract_lora_params_from_model(model: nn.Module) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Extract LoRA A and B parameters from a PEFT model.

    Args:
        model: The model (potentially FSDP-wrapped) containing LoRA layers

    Returns:
        Dict mapping module names to {'A': tensor, 'B': tensor}
    """
    lora_params = {}

    try:
        for name, module in model.named_modules():
            # Check for PEFT-style LoRA layers
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                lora_a = None
                lora_b = None

                try:
                    # Handle different LoRA implementations
                    if isinstance(module.lora_A, nn.ModuleDict):
                        # PEFT style with adapters
                        for adapter_name in module.lora_A.keys():
                            if hasattr(module.lora_A[adapter_name], 'weight'):
                                lora_a = module.lora_A[adapter_name].weight.data.clone()
                            if hasattr(module.lora_B[adapter_name], 'weight'):
                                lora_b = module.lora_B[adapter_name].weight.data.clone()
                            break
                    elif isinstance(module.lora_A, nn.Linear):
                        lora_a = module.lora_A.weight.data.clone()
                        lora_b = module.lora_B.weight.data.clone()
                    elif isinstance(module.lora_A, nn.Parameter):
                        lora_a = module.lora_A.data.clone()
                        lora_b = module.lora_B.data.clone()
                    elif hasattr(module.lora_A, 'default'):
                        # Another PEFT format
                        if hasattr(module.lora_A.default, 'weight'):
                            lora_a = module.lora_A.default.weight.data.clone()
                        if hasattr(module.lora_B.default, 'weight'):
                            lora_b = module.lora_B.default.weight.data.clone()

                    if lora_a is not None or lora_b is not None:
                        lora_params[name] = {'A': lora_a, 'B': lora_b}
                except Exception as e:
                    # Skip this module if we can't access its parameters
                    continue
    except Exception as e:
        print(f"[MultiLoRA] Warning: Error extracting LoRA params: {e}")

    return lora_params


def reinitialize_lora_for_new_task(model: nn.Module) -> None:
    """
    Reinitialize LoRA parameters for a new task.
    Uses Kaiming initialization for A and zeros for B (standard LoRA init).

    Args:
        model: The model containing LoRA layers
    """
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            try:
                # Reinitialize based on the LoRA implementation type
                if isinstance(module.lora_A, nn.ModuleDict):
                    for adapter_name in module.lora_A.keys():
                        if hasattr(module.lora_A[adapter_name], 'weight'):
                            nn.init.kaiming_uniform_(module.lora_A[adapter_name].weight, a=math.sqrt(5))
                        if hasattr(module.lora_B[adapter_name], 'weight'):
                            nn.init.zeros_(module.lora_B[adapter_name].weight)
                elif isinstance(module.lora_A, nn.Linear):
                    nn.init.kaiming_uniform_(module.lora_A.weight, a=math.sqrt(5))
                    nn.init.zeros_(module.lora_B.weight)
                elif isinstance(module.lora_A, nn.Parameter):
                    nn.init.kaiming_uniform_(module.lora_A, a=math.sqrt(5))
                    nn.init.zeros_(module.lora_B)
                elif hasattr(module.lora_A, 'default'):
                    if hasattr(module.lora_A.default, 'weight'):
                        nn.init.kaiming_uniform_(module.lora_A.default.weight, a=math.sqrt(5))
                    if hasattr(module.lora_B.default, 'weight'):
                        nn.init.zeros_(module.lora_B.default.weight)
            except Exception as e:
                print(f"[MultiLoRA] Warning: Could not reinitialize {name}: {e}")
                continue

    print(f"[MultiLoRA] Reinitialized LoRA parameters for new task")
