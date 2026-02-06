"""
Learning to Prompt (L2P) Continual Learning Method.

This implementation adapts L2P to the CL-Agent framework for decoder-only LLMs.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional

from .base import BaseCLMethod, CLMethodConfig
from .registry import register_cl_method


@dataclass
class L2PConfig(CLMethodConfig):
    name: str = "l2p"

    # Prompt pool configuration
    pool_size: int = 10
    prompt_length: int = 10
    top_k: int = 4
    embedding_key: str = "mean"  # mean | max | mean_max | cls
    prompt_init: str = "uniform"
    prompt_key: bool = True
    prompt_key_init: str = "uniform"
    use_prompt_mask: bool = False

    # Pull constraint
    pull_constraint_coeff: float = 1.0


@register_cl_method("l2p")
class L2PCLMethod(BaseCLMethod):
    """
    L2P Continual Learning Method.

    Only prompt pool parameters are trained; the backbone is frozen.
    """

    def __init__(self, config: L2PConfig):
        super().__init__(config)
        self.config: L2PConfig = config
        self.log_info(
            "Initialized L2P with pool_size={}, prompt_length={}, top_k={}, embedding_key={}, "
            "use_prompt_mask={}, pull_constraint_coeff={}".format(
                config.pool_size,
                config.prompt_length,
                config.top_k,
                config.embedding_key,
                config.use_prompt_mask,
                config.pull_constraint_coeff,
            )
        )

    def get_cl_loss_config(self) -> Dict[str, Any]:
        """Return config for L2P pull loss computation in the worker."""
        return {
            "method_name": "l2p",
            "current_task_idx": self.current_task_idx,
            "has_cl_loss": True,
            "pool_size": self.config.pool_size,
            "prompt_length": self.config.prompt_length,
            "top_k": self.config.top_k,
            "embedding_key": self.config.embedding_key,
            "prompt_init": self.config.prompt_init,
            "prompt_key": self.config.prompt_key,
            "prompt_key_init": self.config.prompt_key_init,
            "use_prompt_mask": self.config.use_prompt_mask,
            "pull_constraint_coeff": self.config.pull_constraint_coeff,
        }

    def get_method_info(self) -> Dict[str, Any]:
        """Return information about the method."""
        return {
            "name": "l2p",
            "description": "Learning to Prompt for Continual Learning",
            "has_cl_loss": True,
            "pool_size": self.config.pool_size,
            "prompt_length": self.config.prompt_length,
            "top_k": self.config.top_k,
            "embedding_key": self.config.embedding_key,
            "use_prompt_mask": self.config.use_prompt_mask,
            "pull_constraint_coeff": self.config.pull_constraint_coeff,
            "current_task": self.current_task_idx,
            "task_history": self.task_history,
        }
