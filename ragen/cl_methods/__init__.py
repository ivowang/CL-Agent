"""
Continual Learning Methods for RAGEN

This module provides a pluggable framework for different continual learning algorithms.
Each method handles:
- Task initialization and transition
- Additional loss computation (e.g., orthogonal loss for O-LoRA)
- Checkpoint management for task-specific parameters

Available methods:
- naive: Shared LoRA across all tasks (baseline)
- olora: Orthogonal Low-Rank Adaptation (O-LoRA)
"""

from .registry import CL_METHODS, register_cl_method, get_cl_method
from .base import BaseCLMethod
from .naive import NaiveCLMethod
from .olora import OLoRACLMethod
from .loss_functions import (
    compute_olora_loss,
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
    # Loss functions
    'compute_olora_loss',
    'get_cl_loss_fn',
    'load_frozen_lora_params_from_checkpoint',
    'reinitialize_lora_params',
]

