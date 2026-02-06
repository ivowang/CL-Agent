"""
Continual Learning Methods for RAGEN

This module provides a pluggable framework for different continual learning algorithms.
Each method handles:
- Task initialization and transition
- Additional loss computation (e.g., orthogonal loss for O-LoRA)
- Checkpoint management for task-specific parameters
- Multi-LoRA management for per-task LoRA adapters

Available methods:
- naive: Shared LoRA across all tasks (baseline)
- olora: Orthogonal Low-Rank Adaptation (O-LoRA) - per-task LoRA with orthogonal constraints
- sdlora: Scalable Decomposed Low-Rank Adaptation (SD-LoRA) - per-task LoRA with scaling factors
- hideprompt: HiDE-Prompt - prompt-based continual learning with contrastive regularization
- moe: Mixture of Experts - independent LoRA per task with mixed sampling
- mix: Multi-task interleaved training (all tasks simultaneously)
"""

from .registry import CL_METHODS, register_cl_method, get_cl_method
from .base import BaseCLMethod
from .naive import NaiveCLMethod
from .olora import OLoRACLMethod
from .sdlora import SDLoRACLMethod
from .hideprompt import HiDEPromptCLMethod
from .l2p import L2PCLMethod
from .moe import MoECLMethod
from .mix import MixCLMethod
from .multi_lora import (
    MultiLoRAManager,
    TaskLoRAParams,
    extract_lora_params_from_model,
    reinitialize_lora_for_new_task,
)
from .loss_functions import (
    compute_olora_loss,
    compute_sdlora_loss,
    compute_frozen_lora_output,
    get_cl_loss_fn,
    load_frozen_lora_params_from_checkpoint,
    reinitialize_lora_params,
)

__all__ = [
    'CL_METHODS',
    'register_cl_method',
    'get_cl_method',
    'BaseCLMethod',
    'NaiveCLMethod',
    'OLoRACLMethod',
    'SDLoRACLMethod',
    'HiDEPromptCLMethod',
    'L2PCLMethod',
    'MoECLMethod',
    'MixCLMethod',
    # Multi-LoRA management
    'MultiLoRAManager',
    'TaskLoRAParams',
    'extract_lora_params_from_model',
    'reinitialize_lora_for_new_task',
    # Loss functions
    'compute_olora_loss',
    'compute_sdlora_loss',
    'compute_frozen_lora_output',
    'get_cl_loss_fn',
    'load_frozen_lora_params_from_checkpoint',
    'reinitialize_lora_params',
]
