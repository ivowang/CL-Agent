# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor with Continual Learning Support
"""

import itertools
import logging
import os
from typing import Tuple, Dict, Any, Optional

import torch
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor

from peft import PeftModel

# Import CL loss functions
from ragen.cl_methods.loss_functions import compute_olora_loss, get_cl_loss_fn


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )
        
        # Continual Learning support
        self._cl_config: Dict[str, Any] = {}
        self._frozen_lora_params: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
        self._cl_method_name: str = 'naive'

        # SD-LoRA: Trainable scaling factors for each task
        # These are nn.Parameters that will be updated during training
        self._scaling_factors_params: Optional[nn.ParameterDict] = None
        self._scaling_factors_optimizer: Optional[torch.optim.Optimizer] = None

        # HiDE-Prompt: Prompt-based continual learning
        self._hideprompt_enabled = False
        self._last_hidden_states: Optional[torch.Tensor] = None  # Store for feature extraction
        
    def set_cl_config(self, cl_config: Dict[str, Any]) -> None:
        """
        Set continual learning configuration.

        Args:
            cl_config: CL configuration dict containing:
                - method: CL method name ('naive', 'olora', 'sdlora', etc.)
                - lambda_ortho: Weight for orthogonal loss (O-LoRA)
                - lambda_l2: Weight for L2 regularization (O-LoRA)
                - current_task_idx: Current task index
                - frozen_lora_params: Frozen LoRA params from previous tasks (optional)
                - scaling_factors: Scaling factors for each task (SD-LoRA)
                - all_frozen_lora_params: All frozen LoRA params by task (SD-LoRA)
        """
        self._cl_config = cl_config
        self._cl_method_name = cl_config.get('method_name', cl_config.get('method', 'naive'))
        self._frozen_lora_params = cl_config.get('frozen_lora_params', None)

        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        if rank == 0:
            print(f"[CL Actor] Set config: method={self._cl_method_name}, "
                  f"task_idx={cl_config.get('current_task_idx', 0)}, "
                  f"has_frozen_params={self._frozen_lora_params is not None}")
            # Log SD-LoRA specific info
            if self._cl_method_name == 'sdlora':
                scaling_factors = cl_config.get('scaling_factors', {})
                all_frozen = cl_config.get('all_frozen_lora_params', {})
                print(f"[CL Actor] SD-LoRA: scaling_factors={scaling_factors}, "
                      f"num_frozen_tasks={len(all_frozen)}")

        # Store original LoRA state for restoration after validation
        self._original_lora_state = None
        self._frozen_loras_applied = False

        # Initialize SD-LoRA scaling factors as trainable parameters
        if self._cl_method_name == 'sdlora':
            self._init_sdlora_scaling_factors(cl_config, rank)

        # Initialize HiDE-Prompt if needed
        if self._cl_method_name == 'hideprompt':
            self._hideprompt_enabled = True
            if rank == 0:
                print(f"[CL Actor] HiDE-Prompt enabled: prompt_length={cl_config.get('prompt_length', 5)}, "
                      f"reg_weight={cl_config.get('reg_weight', 0.1)}")

        # Check if we need to reinitialize LoRA for new task (O-LoRA and SD-LoRA)
        current_task_idx = cl_config.get('current_task_idx', 0)
        reinit_lora = cl_config.get('reinit_lora_per_task', False)

        # Only reinitialize for O-LoRA and SD-LoRA, and only for task > 0
        if reinit_lora and self._cl_method_name in ['olora', 'sdlora'] and current_task_idx > 0:
            # Check if this is a new task (not a refresh of the same task)
            prev_task_idx = getattr(self, '_prev_task_idx', -1)
            if current_task_idx != prev_task_idx:
                self._reinitialize_lora_for_new_task()
                self._prev_task_idx = current_task_idx
        else:
            # Track task index even for baseline
            self._prev_task_idx = current_task_idx

    def _init_sdlora_scaling_factors(self, cl_config: Dict[str, Any], rank: int) -> None:
        """
        Initialize SD-LoRA scaling factors as trainable nn.Parameters.

        For SD-LoRA, we need trainable scaling factors for:
        - Current task's scaling factor
        - All previous tasks' scaling factors (which are also updated during training)
        """
        current_task_idx = cl_config.get('current_task_idx', 0)
        scaling_factors_init = cl_config.get('scaling_factors', {})
        scaling_factor_default = cl_config.get('scaling_factor_init', 0.8)

        device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

        # Create ParameterDict for scaling factors
        if self._scaling_factors_params is None:
            self._scaling_factors_params = nn.ParameterDict()

        # Initialize scaling factors for all tasks up to current
        for task_idx in range(current_task_idx + 1):
            param_name = f"task_{task_idx}"
            if param_name not in self._scaling_factors_params:
                # Get initial value from config or use default
                init_value = scaling_factors_init.get(task_idx, scaling_factor_default)
                if isinstance(init_value, torch.Tensor):
                    init_value = init_value.item()

                # Create trainable parameter
                param = nn.Parameter(torch.tensor([init_value], dtype=torch.float32, device=device))
                self._scaling_factors_params[param_name] = param

                if rank == 0:
                    print(f"[SD-LoRA] Initialized scaling factor for task {task_idx}: {init_value}")

        # Create optimizer for scaling factors if not exists
        if self._scaling_factors_optimizer is None and len(self._scaling_factors_params) > 0:
            # Use same learning rate as actor optimizer, but can be adjusted
            lr = 0.001  # Default learning rate for scaling factors
            self._scaling_factors_optimizer = torch.optim.Adam(
                self._scaling_factors_params.parameters(),
                lr=lr
            )
            if rank == 0:
                print(f"[SD-LoRA] Created optimizer for {len(self._scaling_factors_params)} scaling factors")

    def _get_scaling_factor(self, task_idx: int) -> torch.Tensor:
        """Get the scaling factor for a specific task."""
        if self._scaling_factors_params is None:
            return torch.tensor([1.0], device=torch.cuda.current_device())

        param_name = f"task_{task_idx}"
        if param_name in self._scaling_factors_params:
            return self._scaling_factors_params[param_name]
        else:
            # Return default value if not found
            default_value = self._cl_config.get('scaling_factor_init', 0.8)
            return torch.tensor([default_value], device=torch.cuda.current_device())

    def _update_sdlora_scaling_factors(self, metrics: Dict[str, Any]) -> None:
        """
        Update SD-LoRA scaling factors using their dedicated optimizer.

        This method:
        1. Steps the scaling factors optimizer (if gradients exist)
        2. Logs the current scaling factor values
        3. Zeros the gradients for the next iteration
        """
        if self._scaling_factors_params is None:
            return

        if self._scaling_factors_optimizer is not None:
            # Check if any scaling factor has gradients
            has_grad = any(
                param.grad is not None
                for param in self._scaling_factors_params.values()
            )

            if has_grad:
                # Clip gradients for stability
                torch.nn.utils.clip_grad_norm_(
                    self._scaling_factors_params.values(),
                    max_norm=1.0
                )
                # Step the optimizer
                self._scaling_factors_optimizer.step()
                # Zero gradients
                self._scaling_factors_optimizer.zero_grad()

        # Log scaling factor values
        for param_name, param in self._scaling_factors_params.items():
            task_idx = param_name.replace("task_", "")
            metrics[f"sdlora/scaling_factor_task{task_idx}"] = param.item()
            # Also log gradient if available
            if param.grad is not None:
                metrics[f"sdlora/scaling_factor_grad_task{task_idx}"] = param.grad.item()

    def _reinitialize_lora_for_new_task(self) -> None:
        """Reinitialize LoRA parameters for a new task using Kaiming initialization."""
        from ragen.cl_methods.multi_lora import reinitialize_lora_for_new_task

        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        if rank == 0:
            print(f"[CL Actor] Reinitializing LoRA for new task {self._cl_config.get('current_task_idx', 0)}")

        try:
            with FSDP.summon_full_params(self.actor_module):
                reinitialize_lora_for_new_task(self.actor_module)
            if rank == 0:
                print(f"[CL Actor] Successfully reinitialized LoRA")
        except Exception as e:
            if rank == 0:
                print(f"[CL Actor] Error reinitializing LoRA: {e}")

    def _apply_frozen_loras_for_training(self) -> bool:
        """
        Apply frozen LoRA parameters from all previous tasks for training forward pass.

        For O-LoRA and SD-LoRA, the forward pass should combine all tasks' LoRA outputs.
        This method temporarily adds frozen LoRA contributions to the current LoRA parameters.

        For SD-LoRA: The scaling factors are trainable nn.Parameters that participate in the
        computation graph. The gradients will flow through them during backpropagation.

        IMPORTANT: This modifies the model's LoRA weights in-place. Call _remove_frozen_loras_after_training()
        after the forward pass to restore the original state before gradient computation.

        Returns:
            True if frozen LoRAs were applied, False otherwise
        """
        if self._cl_method_name not in ['olora', 'sdlora']:
            return False

        all_frozen_lora_params = self._cl_config.get('all_frozen_lora_params', {})
        if not all_frozen_lora_params:
            return False

        current_task_idx = self._cl_config.get('current_task_idx', 0)
        if current_task_idx == 0:
            return False  # No frozen LoRAs for first task

        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        # Get scaling factors for SD-LoRA
        normalize_lora = self._cl_config.get('normalize_lora', True)

        # Store original LoRA state before modification (for training)
        self._training_original_lora_state = {}

        # For SD-LoRA, we need to track the scaling factor contributions separately
        # so that gradients can flow through them
        self._sdlora_scaling_contributions = []

        try:
            # We need to access the model within FSDP context
            with FSDP.summon_full_params(self.actor_module):
                # Find all LoRA layers in the model
                for name, module in self.actor_module.named_modules():
                    if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                        # Store original state
                        if isinstance(module.lora_A, nn.ModuleDict):
                            for adapter_name in module.lora_A.keys():
                                key = f"{name}.{adapter_name}"
                                if hasattr(module.lora_A[adapter_name], 'weight'):
                                    self._training_original_lora_state[f"{key}.A"] = module.lora_A[adapter_name].weight.data.clone()
                                if hasattr(module.lora_B[adapter_name], 'weight'):
                                    self._training_original_lora_state[f"{key}.B"] = module.lora_B[adapter_name].weight.data.clone()
                        elif hasattr(module.lora_A, 'default'):
                            key = f"{name}.default"
                            if hasattr(module.lora_A.default, 'weight'):
                                self._training_original_lora_state[f"{key}.A"] = module.lora_A.default.weight.data.clone()
                            if hasattr(module.lora_B.default, 'weight'):
                                self._training_original_lora_state[f"{key}.B"] = module.lora_B.default.weight.data.clone()

                # Apply frozen LoRA params from all previous tasks
                for task_idx, task_params in all_frozen_lora_params.items():
                    frozen_A = task_params.get('A', {})
                    frozen_B = task_params.get('B', {})

                    # Get scaling factor for this task
                    if self._cl_method_name == 'sdlora':
                        # Use trainable scaling factor - this keeps the gradient flow
                        scale_param = self._get_scaling_factor(task_idx)
                        scale = scale_param.item()  # Get scalar value for weight modification
                    else:
                        scale = 1.0  # O-LoRA uses uniform scaling

                    # Add frozen LoRA params to current model
                    for module_name, frozen_A_param in frozen_A.items():
                        frozen_B_param = frozen_B.get(module_name)
                        if frozen_B_param is None:
                            continue

                        # Find the corresponding module in the model
                        for name, module in self.actor_module.named_modules():
                            if module_name in name and hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                                # Move frozen params to the same device
                                device = None
                                if isinstance(module.lora_A, nn.ModuleDict):
                                    for adapter_name in module.lora_A.keys():
                                        if hasattr(module.lora_A[adapter_name], 'weight'):
                                            device = module.lora_A[adapter_name].weight.device
                                            break
                                elif hasattr(module.lora_A, 'default') and hasattr(module.lora_A.default, 'weight'):
                                    device = module.lora_A.default.weight.device

                                if device is None:
                                    continue

                                frozen_A_tensor = frozen_A_param.to(device)
                                frozen_B_tensor = frozen_B_param.to(device)

                                # Apply normalization for SD-LoRA
                                if self._cl_method_name == 'sdlora' and normalize_lora:
                                    A_norm = torch.norm(frozen_A_tensor)
                                    B_norm = torch.norm(frozen_B_tensor)
                                    if A_norm > 1e-6 and B_norm > 1e-6:
                                        scale_normalized = scale / (A_norm.item() * B_norm.item())
                                    else:
                                        scale_normalized = scale
                                else:
                                    scale_normalized = scale

                                # Add to current LoRA weights
                                # Note: We use no_grad here because the scaling factor gradient
                                # will be computed separately in _compute_sdlora_scaling_loss
                                with torch.no_grad():
                                    if isinstance(module.lora_A, nn.ModuleDict):
                                        for adapter_name in module.lora_A.keys():
                                            if hasattr(module.lora_A[adapter_name], 'weight'):
                                                module.lora_A[adapter_name].weight.data += frozen_A_tensor * scale_normalized
                                            if hasattr(module.lora_B[adapter_name], 'weight'):
                                                module.lora_B[adapter_name].weight.data += frozen_B_tensor * scale_normalized
                                    elif hasattr(module.lora_A, 'default'):
                                        if hasattr(module.lora_A.default, 'weight'):
                                            module.lora_A.default.weight.data += frozen_A_tensor * scale_normalized
                                        if hasattr(module.lora_B.default, 'weight'):
                                            module.lora_B.default.weight.data += frozen_B_tensor * scale_normalized
                                break

            self._frozen_loras_applied_for_training = True
            return True

        except Exception as e:
            if rank == 0:
                print(f"[CL Training] Error applying frozen LoRAs: {e}")
            self._training_original_lora_state = None
            return False

    def _remove_frozen_loras_after_training(self) -> bool:
        """
        Remove frozen LoRA parameters and restore original state after training forward pass.

        Returns:
            True if restoration was successful, False otherwise
        """
        if not getattr(self, '_frozen_loras_applied_for_training', False):
            return False

        if self._training_original_lora_state is None:
            return False

        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        try:
            with FSDP.summon_full_params(self.actor_module):
                # Restore original LoRA state
                for name, module in self.actor_module.named_modules():
                    if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                        if isinstance(module.lora_A, nn.ModuleDict):
                            for adapter_name in module.lora_A.keys():
                                key = f"{name}.{adapter_name}"
                                if f"{key}.A" in self._training_original_lora_state and hasattr(module.lora_A[adapter_name], 'weight'):
                                    module.lora_A[adapter_name].weight.data.copy_(self._training_original_lora_state[f"{key}.A"])
                                if f"{key}.B" in self._training_original_lora_state and hasattr(module.lora_B[adapter_name], 'weight'):
                                    module.lora_B[adapter_name].weight.data.copy_(self._training_original_lora_state[f"{key}.B"])
                        elif hasattr(module.lora_A, 'default'):
                            key = f"{name}.default"
                            if f"{key}.A" in self._training_original_lora_state and hasattr(module.lora_A.default, 'weight'):
                                module.lora_A.default.weight.data.copy_(self._training_original_lora_state[f"{key}.A"])
                            if f"{key}.B" in self._training_original_lora_state and hasattr(module.lora_B.default, 'weight'):
                                module.lora_B.default.weight.data.copy_(self._training_original_lora_state[f"{key}.B"])

            self._frozen_loras_applied_for_training = False
            self._training_original_lora_state = None
            return True

        except Exception as e:
            if rank == 0:
                print(f"[CL Training] Error restoring LoRA state: {e}")
            return False

    def _apply_frozen_loras_for_validation(self) -> bool:
        """
        Apply frozen LoRA parameters from all previous tasks for validation.

        For O-LoRA: Add all frozen LoRA params to the current model
        For SD-LoRA: Add all frozen LoRA params weighted by scaling factors

        This modifies the model's LoRA weights in-place. Call _remove_frozen_loras_after_validation()
        to restore the original state.

        Returns:
            True if frozen LoRAs were applied, False otherwise
        """
        if self._cl_method_name not in ['olora', 'sdlora']:
            return False

        all_frozen_lora_params = self._cl_config.get('all_frozen_lora_params', {})
        if not all_frozen_lora_params:
            return False

        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        if rank == 0:
            print(f"[CL Validation] Applying frozen LoRAs from {len(all_frozen_lora_params)} previous tasks")

        # Get scaling factors for SD-LoRA
        scaling_factors = self._cl_config.get('scaling_factors', {})
        normalize_lora = self._cl_config.get('normalize_lora', True)

        # Store original LoRA state before modification
        self._original_lora_state = {}

        try:
            # We need to access the model within FSDP context
            with FSDP.summon_full_params(self.actor_module):
                # Find all LoRA layers in the model
                for name, module in self.actor_module.named_modules():
                    if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                        # Store original state
                        if isinstance(module.lora_A, nn.ModuleDict):
                            for adapter_name in module.lora_A.keys():
                                key = f"{name}.{adapter_name}"
                                if hasattr(module.lora_A[adapter_name], 'weight'):
                                    self._original_lora_state[f"{key}.A"] = module.lora_A[adapter_name].weight.data.clone()
                                if hasattr(module.lora_B[adapter_name], 'weight'):
                                    self._original_lora_state[f"{key}.B"] = module.lora_B[adapter_name].weight.data.clone()
                        elif hasattr(module.lora_A, 'default'):
                            key = f"{name}.default"
                            if hasattr(module.lora_A.default, 'weight'):
                                self._original_lora_state[f"{key}.A"] = module.lora_A.default.weight.data.clone()
                            if hasattr(module.lora_B.default, 'weight'):
                                self._original_lora_state[f"{key}.B"] = module.lora_B.default.weight.data.clone()

                # Apply frozen LoRA params from all previous tasks
                for task_idx, task_params in all_frozen_lora_params.items():
                    frozen_A = task_params.get('A', {})
                    frozen_B = task_params.get('B', {})

                    # Get scaling factor for this task
                    if self._cl_method_name == 'sdlora':
                        scale = scaling_factors.get(task_idx, 1.0)
                    else:
                        scale = 1.0  # O-LoRA uses uniform scaling

                    if rank == 0:
                        print(f"[CL Validation] Applying task {task_idx} LoRA with scale={scale}")

                    # Add frozen LoRA params to current model
                    for module_name, frozen_A_param in frozen_A.items():
                        frozen_B_param = frozen_B.get(module_name)
                        if frozen_B_param is None:
                            continue

                        # Find the corresponding module in the model
                        for name, module in self.actor_module.named_modules():
                            if module_name in name and hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                                # Move frozen params to the same device
                                device = None
                                if isinstance(module.lora_A, nn.ModuleDict):
                                    for adapter_name in module.lora_A.keys():
                                        if hasattr(module.lora_A[adapter_name], 'weight'):
                                            device = module.lora_A[adapter_name].weight.device
                                            break
                                elif hasattr(module.lora_A, 'default') and hasattr(module.lora_A.default, 'weight'):
                                    device = module.lora_A.default.weight.device

                                if device is None:
                                    continue

                                frozen_A_tensor = frozen_A_param.to(device)
                                frozen_B_tensor = frozen_B_param.to(device)

                                # Apply normalization for SD-LoRA
                                if self._cl_method_name == 'sdlora' and normalize_lora:
                                    A_norm = torch.norm(frozen_A_tensor)
                                    B_norm = torch.norm(frozen_B_tensor)
                                    if A_norm > 1e-6 and B_norm > 1e-6:
                                        scale_normalized = scale / (A_norm * B_norm)
                                    else:
                                        scale_normalized = scale
                                else:
                                    scale_normalized = scale

                                # Add to current LoRA weights
                                if isinstance(module.lora_A, nn.ModuleDict):
                                    for adapter_name in module.lora_A.keys():
                                        if hasattr(module.lora_A[adapter_name], 'weight'):
                                            module.lora_A[adapter_name].weight.data += frozen_A_tensor * scale_normalized
                                        if hasattr(module.lora_B[adapter_name], 'weight'):
                                            module.lora_B[adapter_name].weight.data += frozen_B_tensor * scale_normalized
                                elif hasattr(module.lora_A, 'default'):
                                    if hasattr(module.lora_A.default, 'weight'):
                                        module.lora_A.default.weight.data += frozen_A_tensor * scale_normalized
                                    if hasattr(module.lora_B.default, 'weight'):
                                        module.lora_B.default.weight.data += frozen_B_tensor * scale_normalized
                                break

            self._frozen_loras_applied = True
            if rank == 0:
                print(f"[CL Validation] Successfully applied frozen LoRAs")
            return True

        except Exception as e:
            if rank == 0:
                print(f"[CL Validation] Error applying frozen LoRAs: {e}")
            self._original_lora_state = None
            return False

    def _remove_frozen_loras_after_validation(self) -> bool:
        """
        Remove frozen LoRA parameters and restore original state after validation.

        Returns:
            True if restoration was successful, False otherwise
        """
        if not self._frozen_loras_applied or self._original_lora_state is None:
            return False

        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        if rank == 0:
            print(f"[CL Validation] Restoring original LoRA state")

        try:
            with FSDP.summon_full_params(self.actor_module):
                # Restore original LoRA state
                for name, module in self.actor_module.named_modules():
                    if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                        if isinstance(module.lora_A, nn.ModuleDict):
                            for adapter_name in module.lora_A.keys():
                                key = f"{name}.{adapter_name}"
                                if f"{key}.A" in self._original_lora_state and hasattr(module.lora_A[adapter_name], 'weight'):
                                    module.lora_A[adapter_name].weight.data.copy_(self._original_lora_state[f"{key}.A"])
                                if f"{key}.B" in self._original_lora_state and hasattr(module.lora_B[adapter_name], 'weight'):
                                    module.lora_B[adapter_name].weight.data.copy_(self._original_lora_state[f"{key}.B"])
                        elif hasattr(module.lora_A, 'default'):
                            key = f"{name}.default"
                            if f"{key}.A" in self._original_lora_state and hasattr(module.lora_A.default, 'weight'):
                                module.lora_A.default.weight.data.copy_(self._original_lora_state[f"{key}.A"])
                            if f"{key}.B" in self._original_lora_state and hasattr(module.lora_B.default, 'weight'):
                                module.lora_B.default.weight.data.copy_(self._original_lora_state[f"{key}.B"])

            self._frozen_loras_applied = False
            self._original_lora_state = None
            if rank == 0:
                print(f"[CL Validation] Successfully restored original LoRA state")
            return True

        except Exception as e:
            if rank == 0:
                print(f"[CL Validation] Error restoring LoRA state: {e}")
            return False

    def _compute_sdlora_scaling_loss(
        self,
        current_task_idx: int,
        all_frozen_lora_params: Dict[int, Dict[str, Any]]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute SD-LoRA scaling factor loss.

        This method computes a loss that makes the scaling factors participate in the
        computation graph, allowing them to be updated via backpropagation.

        The loss has two components:
        1. Regularization loss: Encourages scaling factors to stay near a target value
        2. Magnitude loss: Encourages scaling factors to have appropriate magnitudes
           based on the frozen LoRA contributions

        Args:
            current_task_idx: Current task index
            all_frozen_lora_params: All frozen LoRA params from previous tasks

        Returns:
            Tuple of (loss tensor, metrics dict)
        """
        device = torch.cuda.current_device()

        if self._scaling_factors_params is None or len(self._scaling_factors_params) == 0:
            return torch.tensor(0.0, device=device, requires_grad=False), {
                'cl/sdlora_loss': 0.0,
                'cl/total_loss': 0.0,
            }

        # Get configuration
        target_scaling = self._cl_config.get('scaling_factor_init', 0.8)
        lambda_scaling_reg = self._cl_config.get('lambda_scaling_reg', 0.01)
        normalize_lora = self._cl_config.get('normalize_lora', True)

        # Initialize loss components
        scaling_reg_loss = torch.tensor(0.0, device=device)
        scaling_magnitude_loss = torch.tensor(0.0, device=device)

        scaling_factor_values = {}

        # Compute regularization loss for all scaling factors
        for param_name, param in self._scaling_factors_params.items():
            task_idx = int(param_name.replace("task_", ""))
            scaling_factor_values[task_idx] = param.item()

            # Regularization: (scaling_factor - target)^2
            # This keeps scaling factors from drifting too far
            reg_term = (param - target_scaling) ** 2
            scaling_reg_loss = scaling_reg_loss + reg_term.squeeze()

        # Compute magnitude loss based on frozen LoRA norms
        # This encourages scaling factors to be proportional to the LoRA magnitudes
        for task_idx, task_params in all_frozen_lora_params.items():
            if task_idx >= current_task_idx:
                continue  # Only consider previous tasks

            frozen_A = task_params.get('A', {})
            frozen_B = task_params.get('B', {})

            # Get the scaling factor for this task
            scale_param = self._get_scaling_factor(task_idx)

            # Compute average LoRA magnitude for this task
            total_norm = 0.0
            num_modules = 0
            for module_name, frozen_A_param in frozen_A.items():
                frozen_B_param = frozen_B.get(module_name)
                if frozen_B_param is None:
                    continue

                A_norm = torch.norm(frozen_A_param.to(device)).item()
                B_norm = torch.norm(frozen_B_param.to(device)).item()

                if normalize_lora and A_norm > 1e-6 and B_norm > 1e-6:
                    # Normalized contribution
                    total_norm += 1.0  # After normalization, contribution is ~1
                else:
                    total_norm += A_norm * B_norm
                num_modules += 1

            if num_modules > 0:
                avg_norm = total_norm / num_modules
                # Magnitude loss: encourage scaling factor to be proportional to avg_norm
                # This is a soft constraint that helps balance contributions
                target_scale = min(1.0, avg_norm) if not normalize_lora else target_scaling
                magnitude_term = (scale_param - target_scale) ** 2
                scaling_magnitude_loss = scaling_magnitude_loss + magnitude_term.squeeze()

        # Compute the total SD-LoRA loss
        total_sdlora_loss = lambda_scaling_reg * (scaling_reg_loss + 0.1 * scaling_magnitude_loss)

        # Build metrics
        metrics = {
            'cl/sdlora_reg_loss': scaling_reg_loss.detach().item(),
            'cl/sdlora_magnitude_loss': scaling_magnitude_loss.detach().item(),
            'cl/sdlora_total_loss': total_sdlora_loss.detach().item(),
            'cl/total_loss': total_sdlora_loss.detach().item(),
            'cl/current_task': current_task_idx,
            'cl/num_frozen_tasks': len(all_frozen_lora_params),
            'cl/num_scaling_factors': len(self._scaling_factors_params),
        }

        # Log individual scaling factors
        for task_idx, sf_value in scaling_factor_values.items():
            metrics[f'cl/scaling_factor_task{task_idx}'] = sf_value

        return total_sdlora_loss, metrics

    def _compute_hideprompt_contrastive_loss(self) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute HiDE-Prompt contrastive regularization loss.

        This encourages the current batch features to be orthogonal to stored
        features from previous tasks, preventing catastrophic forgetting.

        Returns:
            Tuple of (loss tensor, metrics dict)
        """
        from ragen.cl_methods.trajectory_storage import compute_contrastive_loss

        device = torch.cuda.current_device()

        # Get configuration
        reg_weight = self._cl_config.get('reg_weight', 0.1)
        temperature = self._cl_config.get('temperature', 0.8)
        stored_means = self._cl_config.get('stored_means', None)

        # Check if we have stored hidden states from the forward pass
        if self._last_hidden_states is None:
            # No features available, return zero loss
            return torch.tensor(0.0, device=device, requires_grad=False), {
                'cl/contrastive_loss': 0.0,
                'cl/total_loss': 0.0,
            }

        # Extract features from last hidden states
        # Use the last token's hidden state as the feature representation
        current_features = self._last_hidden_states[:, -1, :]  # (batch_size, hidden_dim)

        # Move stored means to device if available
        if stored_means is not None and stored_means.numel() > 0:
            stored_means = stored_means.to(device)
        else:
            stored_means = None

        # Compute contrastive loss
        contrastive_loss = compute_contrastive_loss(
            current_features=current_features,
            stored_means=stored_means,
            temperature=temperature,
            reg_weight=reg_weight,
        )

        metrics = {
            'cl/contrastive_loss': contrastive_loss.item(),
            'cl/total_loss': contrastive_loss.item(),
            'cl/num_stored_means': stored_means.shape[0] if stored_means is not None else 0,
        }

        return contrastive_loss, metrics

    def _compute_cl_loss(self) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute continual learning loss based on the configured method.

        For O-LoRA: Computes orthogonal loss between current task's LoRA and ALL previous tasks' LoRAs
        For SD-LoRA: No explicit loss, but logs metrics for scaling factors
        For HiDE-Prompt: Computes contrastive regularization loss

        Returns:
            Tuple of (cl_loss_tensor, metrics_dict)
        """
        if self._cl_method_name == 'naive' or self._cl_method_name == 'baseline':
            # No CL loss for naive/baseline method
            return torch.tensor(0.0, device=torch.cuda.current_device(), requires_grad=False), {'cl/total_loss': 0.0}

        current_task_idx = self._cl_config.get('current_task_idx', 0)

        # HiDE-Prompt handling
        if self._cl_method_name == 'hideprompt':
            return self._compute_hideprompt_contrastive_loss()

        # Get all frozen LoRA params from all previous tasks (new multi-LoRA format)
        all_frozen_lora_params = self._cl_config.get('all_frozen_lora_params', {})

        # SD-LoRA handling
        if self._cl_method_name == 'sdlora':
            # Compute SD-LoRA scaling factor loss
            # This loss makes scaling factors participate in the computation graph
            return self._compute_sdlora_scaling_loss(current_task_idx, all_frozen_lora_params)

        # O-LoRA and other methods that require frozen params
        if current_task_idx == 0:
            # First task - no orthogonal constraint
            return torch.tensor(0.0, device=torch.cuda.current_device(), requires_grad=False), {'cl/total_loss': 0.0}

        # Check if we have frozen params (either new format or old format)
        has_frozen_params = (
            (all_frozen_lora_params and len(all_frozen_lora_params) > 0) or
            (self._frozen_lora_params is not None and len(self._frozen_lora_params) > 0)
        )
        if not has_frozen_params:
            return torch.tensor(0.0, device=torch.cuda.current_device(), requires_grad=False), {'cl/total_loss': 0.0}

        # Build CL config for the loss function
        # Include all_frozen_lora_params for computing orthogonal loss against ALL previous tasks
        cl_loss_config = {
            'lambda_ortho': self._cl_config.get('lambda_ortho', 0.5),
            'lambda_l2': self._cl_config.get('lambda_l2', 0.0),
            'current_task_idx': current_task_idx,
            'all_frozen_lora_params': all_frozen_lora_params,
        }

        # Get the appropriate loss function
        cl_loss_fn = get_cl_loss_fn(self._cl_method_name)
        
        # Compute CL loss
        # NOTE: For FSDP, we DON'T use summon_full_params because:
        # 1. The LoRA parameters can be accessed directly since they are on the current device
        # 2. Using summon_full_params inside the training loop can cause memory issues
        # 3. The frozen params are already on CPU and will be moved to GPU as needed
        try:
            cl_loss, metrics = cl_loss_fn(
                self.actor_module,
                cl_loss_config,
                self._frozen_lora_params,
                device=torch.cuda.current_device(),
            )
            return cl_loss, metrics
        except Exception as e:
            # If there's an error, log it and return zero loss
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
            else:
                rank = 0
            if rank == 0:
                print(f"[CL Warning] Error computing CL loss: {e}, returning zero loss")
            return torch.tensor(0.0, device=torch.cuda.current_device(), requires_grad=False), {'cl/total_loss': 0.0}

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False, is_training=True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        # For O-LoRA and SD-LoRA during training, apply frozen LoRAs before forward pass
        frozen_loras_applied_for_training = False
        if is_training and self._cl_method_name in ['olora', 'sdlora']:
            frozen_loras_applied_for_training = self._apply_frozen_loras_for_training()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None, self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                inplace_backward = True
                if calculate_entropy:
                    inplace_backward = False
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled, inplace_backward=inplace_backward)

                # compute entropy
                if calculate_entropy:
                    entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    output_hidden_states=self._hideprompt_enabled and is_training,  # Request hidden states for HiDE-Prompt
                )  # prevent model thinks we are generating

                # Store hidden states for HiDE-Prompt contrastive loss
                if self._hideprompt_enabled and is_training and hasattr(output, 'hidden_states') and output.hidden_states is not None:
                    # Get the last layer's hidden states
                    self._last_hidden_states = output.hidden_states[-1]  # (batch_size, seq_len, hidden_dim)

                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                if calculate_entropy:
                    entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            # Restore original LoRA state after forward pass (for training with frozen LoRAs)
            if frozen_loras_applied_for_training:
                self._remove_frozen_loras_after_training()

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False, no_lora=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        # Check if this is a validation call - if so, apply frozen LoRAs for O-LoRA/SD-LoRA
        is_validation = data.meta_info.get("validate", False)
        frozen_loras_applied = False
        if is_validation:
            frozen_loras_applied = self._apply_frozen_loras_for_validation()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        is_peft_model = not no_lora and isinstance(self.actor_module._fsdp_wrapped_module, PeftModel)
        if is_peft_model:
            print(f"[INFO] Actor is a PeftModel")
            with FSDP.summon_full_params(self.actor_module):
                self.actor_module.merge_adapter()
            print(f"[INFO] Merged adapter actor")

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy, is_training=False)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)

        if is_peft_model:
            print(f"[INFO] Unmerging adapter actor")
            with FSDP.summon_full_params(self.actor_module):
                self.actor_module.unmerge_adapter()
            print(f"[INFO] Unmerged adapter actor")

        # Restore original LoRA state after validation
        if frozen_loras_applied:
            self._remove_frozen_loras_after_validation()

        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages", "response_mask"]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(torch.cuda.current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(torch.cuda.current_device())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    response_mask = data["response_mask"]
                    # response_mask = attention_mask[:, -response_length:]
                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy)

                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode,
                    )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    # Compute CL loss if configured
                    cl_loss, cl_metrics = self._compute_cl_loss()
                    total_loss = policy_loss + cl_loss
                    
                    # Update metrics with CL loss info
                    if cl_metrics:
                        for k, v in cl_metrics.items():
                            if k not in metrics:  # Only add once per update
                                metrics[k] = v

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = total_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = total_loss / self.gradient_accumulation
                    loss.backward()

                    data = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    }
                    if entropy_coeff != 0:
                        data["actor/entropy_loss"] = entropy_loss.detach().item()
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()

                # Update SD-LoRA scaling factors using their dedicated optimizer
                if self._cl_method_name == 'sdlora':
                    self._update_sdlora_scaling_factors(metrics)

                data = {"actor/grad_norm": grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
