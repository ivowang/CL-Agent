"""
Continual Learning Loss Functions

This module provides loss functions for different CL methods.
These functions are designed to be called from the worker during actor updates.
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


def compute_olora_loss(
    model: nn.Module,
    cl_config: Dict[str, Any],
    frozen_lora_params: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute O-LoRA orthogonal loss and L2 regularization.
    
    O-LoRA ensures new LoRA parameters are orthogonal to frozen parameters
    from previous tasks to minimize catastrophic forgetting.
    
    Loss components:
    1. Orthogonal loss: L_ortho = λ_ortho * Σ|A_frozen @ A_new.T|
    2. L2 regularization: L_l2 = λ_l2 * Σ||A_new||_2
    
    Args:
        model: The model containing LoRA layers
        cl_config: CL configuration dict containing:
            - lambda_ortho: Weight for orthogonal loss
            - lambda_l2: Weight for L2 regularization
            - current_task_idx: Current task index
        frozen_lora_params: Dict of frozen LoRA parameters from previous tasks
            Structure: {module_name: {'A': tensor, 'B': tensor}}
        device: Device to compute on
        
    Returns:
        Tuple of (total_cl_loss, metrics_dict)
    """
    lambda_ortho = cl_config.get('lambda_ortho', 0.5)
    lambda_l2 = cl_config.get('lambda_l2', 0.0)
    current_task_idx = cl_config.get('current_task_idx', 0)
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Skip for first task (no frozen params)
    if current_task_idx == 0 or frozen_lora_params is None or len(frozen_lora_params) == 0:
        return torch.tensor(0.0, device=device, requires_grad=False), {
            'cl/ortho_loss': 0.0,
            'cl/l2_loss': 0.0,
            'cl/total_loss': 0.0,
        }
    
    ortho_loss = torch.tensor(0.0, device=device)
    l2_loss = torch.tensor(0.0, device=device)
    
    # Collect current LoRA parameters
    current_lora_params = _extract_lora_params(model)
    
    # Compute orthogonal loss
    # Use torch.no_grad() for frozen params to avoid unnecessary computation
    for name, frozen_params in frozen_lora_params.items():
        if name in current_lora_params:
            current_a = current_lora_params[name]['A']
            
            if frozen_params.get('A') is not None and current_a is not None:
                # Move frozen param to device with no_grad to save memory
                with torch.no_grad():
                    frozen_a = frozen_params['A'].to(device, non_blocking=True)
                
                # Orthogonal loss: |frozen_A @ current_A.T|
                # frozen_A: [r_frozen, in_features]
                # current_A: [r_new, in_features]
                ortho_product = torch.mm(frozen_a, current_a.T)
                ortho_loss = ortho_loss + torch.abs(ortho_product).sum()
                
                # Explicitly delete to free memory
                del frozen_a
    
    # Compute L2 regularization on current LoRA params
    if lambda_l2 > 0:
        for name, params in current_lora_params.items():
            if params['A'] is not None:
                l2_loss = l2_loss + torch.norm(params['A'], p=2)
            if params['B'] is not None:
                l2_loss = l2_loss + torch.norm(params['B'], p=2)
    
    # Total CL loss
    total_cl_loss = lambda_ortho * ortho_loss + lambda_l2 * l2_loss
    
    metrics = {
        'cl/ortho_loss': ortho_loss.detach().item(),
        'cl/l2_loss': l2_loss.detach().item(),
        'cl/total_loss': total_cl_loss.detach().item(),
        'cl/lambda_ortho': lambda_ortho,
        'cl/lambda_l2': lambda_l2,
    }
    
    return total_cl_loss, metrics


def _extract_lora_params(model: nn.Module) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Extract LoRA A and B parameters from a model.
    Works with both FSDP-wrapped and regular models.
    
    Returns:
        Dict mapping module names to {'A': tensor, 'B': tensor}
    """
    lora_params = {}
    
    # For FSDP wrapped models, we iterate directly - FSDP handles the parameter access
    module_to_search = model
    
    try:
        for name, module in module_to_search.named_modules():
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
                                lora_a = module.lora_A[adapter_name].weight
                            if hasattr(module.lora_B[adapter_name], 'weight'):
                                lora_b = module.lora_B[adapter_name].weight
                            break
                    elif isinstance(module.lora_A, nn.Linear):
                        lora_a = module.lora_A.weight
                        lora_b = module.lora_B.weight
                    elif isinstance(module.lora_A, nn.Parameter):
                        lora_a = module.lora_A
                        lora_b = module.lora_B
                    elif hasattr(module.lora_A, 'default'):
                        # Another PEFT format
                        if hasattr(module.lora_A.default, 'weight'):
                            lora_a = module.lora_A.default.weight
                        if hasattr(module.lora_B.default, 'weight'):
                            lora_b = module.lora_B.default.weight
                    
                    if lora_a is not None or lora_b is not None:
                        lora_params[name] = {'A': lora_a, 'B': lora_b}
                except Exception as e:
                    # Skip this module if we can't access its parameters
                    # This can happen with FSDP sharded parameters
                    continue
    except Exception as e:
        print(f"[CL Warning] Error extracting LoRA params: {e}")
    
    return lora_params


def load_frozen_lora_params_from_checkpoint(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Load frozen LoRA parameters from a checkpoint.
    
    This is used to load LoRA parameters from previous tasks for
    computing the orthogonal loss.
    
    Args:
        checkpoint_path: Path to the checkpoint directory
        device: Device to load tensors to
        
    Returns:
        Dict mapping module names to {'A': tensor, 'B': tensor}
    """
    if device is None:
        device = torch.device('cpu')
    
    frozen_params = {}
    
    if not os.path.exists(checkpoint_path):
        print(f"[O-LoRA] Checkpoint path not found: {checkpoint_path}")
        return frozen_params
    
    # Find the actor checkpoint file - try multiple possible locations
    possible_paths = [
        os.path.join(checkpoint_path, 'actor'),
        os.path.join(checkpoint_path, 'actor_model'),
        checkpoint_path,  # Sometimes the checkpoint is directly in the path
    ]
    
    actor_path = None
    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            # Check if there are model files
            files = os.listdir(path)
            if any(f.endswith('.pt') or f.endswith('.bin') or f.endswith('.safetensors') for f in files):
                actor_path = path
                break
    
    if actor_path is None:
        print(f"[O-LoRA] No valid actor checkpoint found in {checkpoint_path}")
        return frozen_params
    
    # Load model weights - look for model files with different formats
    model_files = []
    for f in os.listdir(actor_path):
        if (f.startswith('model_') or f.startswith('pytorch_model') or 
            f.startswith('adapter_model')) and (f.endswith('.pt') or f.endswith('.bin')):
            model_files.append(f)
    
    # Also check for safetensors
    safetensor_files = [f for f in os.listdir(actor_path) if f.endswith('.safetensors')]
    
    if not model_files and not safetensor_files:
        print(f"[O-LoRA] No model files found in {actor_path}")
        print(f"[O-LoRA] Available files: {os.listdir(actor_path)}")
        return frozen_params
    
    # Prefer .pt/.bin files
    if model_files:
        model_file = sorted(model_files)[0]
        model_path = os.path.join(actor_path, model_file)
        
        try:
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
        except TypeError:
            # Older PyTorch version doesn't support weights_only
            state_dict = torch.load(model_path, map_location=device)
    elif safetensor_files:
        try:
            from safetensors.torch import load_file
            model_file = sorted(safetensor_files)[0]
            model_path = os.path.join(actor_path, model_file)
            state_dict = load_file(model_path, device=str(device))
        except ImportError:
            print("[O-LoRA] safetensors not available, cannot load checkpoint")
            return frozen_params
    
    try:
        # Extract LoRA parameters from state dict
        for key, value in state_dict.items():
            # Handle different naming conventions
            if 'lora_A' in key or 'lora_a' in key:
                # Get the base module name
                if 'lora_A' in key:
                    base_name = key.rsplit('.lora_A', 1)[0]
                else:
                    base_name = key.rsplit('.lora_a', 1)[0]
                    
                # Normalize base name (remove adapter suffix if present)
                base_name = base_name.replace('.default', '').replace('.adapter', '')
                
                if base_name not in frozen_params:
                    frozen_params[base_name] = {'A': None, 'B': None}
                frozen_params[base_name]['A'] = value.clone().to(device)
                
            elif 'lora_B' in key or 'lora_b' in key:
                if 'lora_B' in key:
                    base_name = key.rsplit('.lora_B', 1)[0]
                else:
                    base_name = key.rsplit('.lora_b', 1)[0]
                    
                base_name = base_name.replace('.default', '').replace('.adapter', '')
                
                if base_name not in frozen_params:
                    frozen_params[base_name] = {'A': None, 'B': None}
                frozen_params[base_name]['B'] = value.clone().to(device)
        
        print(f"[O-LoRA] Loaded {len(frozen_params)} frozen LoRA modules from {model_path}")
        
    except Exception as e:
        print(f"[O-LoRA] Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
    
    return frozen_params


def reinitialize_lora_params(model: nn.Module) -> None:
    """
    Reinitialize LoRA parameters for a new task.
    Uses Kaiming initialization for A and zeros for B (standard LoRA init).
    
    Args:
        model: The model containing LoRA layers
    """
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
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
    
    print(f"[O-LoRA] Reinitialized LoRA parameters for new task")


def get_cl_loss_fn(method_name: str):
    """
    Get the CL loss function for a specific method.
    
    Args:
        method_name: Name of the CL method ('baseline', 'naive', 'olora', etc.)
        
    Returns:
        The loss function for the method
    """
    # No-op loss function for baseline/naive
    def no_cl_loss(model, config, frozen, device=None):
        return torch.tensor(0.0), {'cl/total_loss': 0.0}
    
    loss_functions = {
        'baseline': no_cl_loss,
        'naive': no_cl_loss,
        'olora': compute_olora_loss,
    }
    
    if method_name not in loss_functions:
        print(f"[Warning] Unknown CL method: {method_name}, using baseline (no CL loss)")
        return loss_functions['baseline']
    
    return loss_functions[method_name]

