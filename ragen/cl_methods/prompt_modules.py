"""
Prompt modules for HiDE-Prompt continual learning.

This module implements prompt-based continual learning components adapted for
decoder-only LLMs (like Llama/Qwen) in the CL-Agent framework.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any


class EPrompt(nn.Module):
    """
    E-Prompt (Expert Prompt) module for continual learning.

    Adapted from HiDE-Prompt for decoder-only LLMs. Each task gets its own prompt
    that is prepended to the input embeddings (prefix-tuning style).

    Args:
        num_tasks: Number of tasks in the continual learning sequence
        prompt_length: Length of each prompt (number of tokens)
        embed_dim: Embedding dimension of the model
        prompt_init: Initialization method ('uniform' or 'zero')
        device: Device to place the prompts on
    """

    def __init__(
        self,
        num_tasks: int,
        prompt_length: int = 5,
        embed_dim: int = 768,
        prompt_init: str = 'uniform',
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.num_tasks = num_tasks
        self.prompt_length = prompt_length
        self.embed_dim = embed_dim
        self.prompt_init = prompt_init

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        # Create prompt pool: (num_tasks, prompt_length, embed_dim)
        # Each task gets one prompt of length prompt_length
        prompt_pool_shape = (num_tasks, prompt_length, embed_dim)

        if prompt_init == 'zero':
            self.prompt_pool = nn.Parameter(torch.zeros(prompt_pool_shape, device=device))
        elif prompt_init == 'uniform':
            self.prompt_pool = nn.Parameter(torch.randn(prompt_pool_shape, device=device))
            nn.init.uniform_(self.prompt_pool, -1, 1)
        else:
            raise ValueError(f"Unknown prompt_init: {prompt_init}")

    def forward(
        self,
        input_embeds: torch.Tensor,
        task_id: int,
        prompt_momentum: float = 0.0,
    ) -> torch.Tensor:
        """
        Prepend task-specific prompt to input embeddings.

        Args:
            input_embeds: Input embeddings of shape (batch_size, seq_len, embed_dim)
            task_id: Current task ID (0-indexed)
            prompt_momentum: Momentum for prompt averaging (0.0 = no momentum)

        Returns:
            Embeddings with prompt prepended: (batch_size, prompt_length + seq_len, embed_dim)
        """
        batch_size = input_embeds.shape[0]
        device = input_embeds.device

        # Get the prompt for the current task
        if prompt_momentum > 0 and task_id > 0:
            # Apply momentum: blend current task prompt with average of previous prompts
            with torch.no_grad():
                prev_prompts_mean = self.prompt_pool[:task_id].mean(dim=0, keepdim=True)  # (1, prompt_length, embed_dim)
            current_prompt = self.prompt_pool[task_id:task_id+1]  # (1, prompt_length, embed_dim)
            task_prompt = (1 - prompt_momentum) * current_prompt + prompt_momentum * prev_prompts_mean
        else:
            task_prompt = self.prompt_pool[task_id:task_id+1]  # (1, prompt_length, embed_dim)

        # Move prompt to the same device as input_embeds
        task_prompt = task_prompt.to(device)

        # Expand to batch size: (batch_size, prompt_length, embed_dim)
        task_prompt = task_prompt.expand(batch_size, -1, -1)

        # Prepend prompt to input embeddings
        output_embeds = torch.cat([task_prompt, input_embeds], dim=1)

        return output_embeds

    def get_prompt(self, task_id: int) -> torch.Tensor:
        """Get the prompt for a specific task."""
        return self.prompt_pool[task_id]

    def set_prompt(self, task_id: int, prompt: torch.Tensor) -> None:
        """Set the prompt for a specific task."""
        with torch.no_grad():
            self.prompt_pool[task_id] = prompt


class PromptEmbeddingWrapper(nn.Module):
    """
    Wrapper around the model's embedding layer to inject prompts.

    This wrapper intercepts the embedding layer's forward pass and prepends
    task-specific prompts to the input embeddings.
    """

    def __init__(
        self,
        original_embedding: nn.Module,
        e_prompt: EPrompt,
    ):
        super().__init__()
        self.original_embedding = original_embedding
        self.e_prompt = e_prompt
        self.current_task_id = 0
        self.prompt_momentum = 0.0
        self.enabled = True

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with prompt injection.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)

        Returns:
            Embeddings with prompt prepended if enabled
        """
        # Get original embeddings
        input_embeds = self.original_embedding(input_ids)

        # Inject prompt if enabled
        if self.enabled and self.e_prompt is not None:
            input_embeds = self.e_prompt(
                input_embeds,
                task_id=self.current_task_id,
                prompt_momentum=self.prompt_momentum,
            )

        return input_embeds

    def set_task(self, task_id: int, prompt_momentum: float = 0.0) -> None:
        """Set the current task ID and prompt momentum."""
        self.current_task_id = task_id
        self.prompt_momentum = prompt_momentum

    def enable_prompt(self) -> None:
        """Enable prompt injection."""
        self.enabled = True

    def disable_prompt(self) -> None:
        """Disable prompt injection."""
        self.enabled = False


def inject_prompts_into_model(
    model: nn.Module,
    num_tasks: int,
    prompt_length: int = 5,
    embed_dim: Optional[int] = None,
    prompt_init: str = 'uniform',
    device: Optional[torch.device] = None,
) -> EPrompt:
    """
    Inject E-Prompt into a model by wrapping its embedding layer.

    Args:
        model: The model to inject prompts into
        num_tasks: Number of tasks
        prompt_length: Length of each prompt
        embed_dim: Embedding dimension (auto-detected if None)
        prompt_init: Initialization method
        device: Device to place prompts on

    Returns:
        The created EPrompt module
    """
    # Find the embedding layer
    embedding_layer = None
    embedding_attr_name = None

    # Common embedding layer names in decoder-only LLMs
    possible_names = ['embed_tokens', 'wte', 'word_embeddings', 'embeddings']

    for name in possible_names:
        if hasattr(model, name):
            embedding_layer = getattr(model, name)
            embedding_attr_name = name
            break

    if embedding_layer is None:
        # Try to find it in model.model (for some architectures)
        if hasattr(model, 'model'):
            for name in possible_names:
                if hasattr(model.model, name):
                    embedding_layer = getattr(model.model, name)
                    embedding_attr_name = name
                    break

    if embedding_layer is None:
        raise ValueError("Could not find embedding layer in model")

    # Auto-detect embed_dim if not provided
    if embed_dim is None:
        if hasattr(embedding_layer, 'embedding_dim'):
            embed_dim = embedding_layer.embedding_dim
        elif hasattr(embedding_layer, 'weight'):
            embed_dim = embedding_layer.weight.shape[1]
        else:
            raise ValueError("Could not auto-detect embed_dim")

    # Create E-Prompt module
    e_prompt = EPrompt(
        num_tasks=num_tasks,
        prompt_length=prompt_length,
        embed_dim=embed_dim,
        prompt_init=prompt_init,
        device=device,
    )

    # Wrap the embedding layer
    wrapped_embedding = PromptEmbeddingWrapper(embedding_layer, e_prompt)

    # Replace the original embedding layer
    if hasattr(model, embedding_attr_name):
        setattr(model, embedding_attr_name, wrapped_embedding)
    elif hasattr(model, 'model') and hasattr(model.model, embedding_attr_name):
        setattr(model.model, embedding_attr_name, wrapped_embedding)

    return e_prompt


def remove_prompts_from_model(model: nn.Module) -> None:
    """
    Remove prompt injection from a model by unwrapping its embedding layer.

    Args:
        model: The model to remove prompts from
    """
    # Common embedding layer names
    possible_names = ['embed_tokens', 'wte', 'word_embeddings', 'embeddings']

    for name in possible_names:
        if hasattr(model, name):
            layer = getattr(model, name)
            if isinstance(layer, PromptEmbeddingWrapper):
                setattr(model, name, layer.original_embedding)
                return

    # Try model.model
    if hasattr(model, 'model'):
        for name in possible_names:
            if hasattr(model.model, name):
                layer = getattr(model.model, name)
                if isinstance(layer, PromptEmbeddingWrapper):
                    setattr(model.model, name, layer.original_embedding)
                    return
