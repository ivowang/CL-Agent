"""
L2P prompt pool utilities for continual learning.

This module implements the prompt pool, instance-wise prompt selection,
and helper utilities to build prompt-prepended inputs for decoder-only LLMs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class L2PPromptPoolConfig:
    pool_size: int = 10
    prompt_length: int = 10
    top_k: int = 4
    embedding_key: str = "mean"  # mean | max | mean_max | cls
    prompt_init: str = "uniform"  # uniform | normal | zero
    prompt_key: bool = True
    prompt_key_init: str = "uniform"  # uniform | zero
    use_prompt_mask: bool = False


def _get_base_model(module: nn.Module) -> nn.Module:
    if hasattr(module, "_fsdp_wrapped_module"):
        return module._fsdp_wrapped_module
    return module


def get_input_embedding_layer(module: nn.Module) -> nn.Module:
    base = _get_base_model(module)
    if hasattr(base, "get_input_embeddings"):
        return base.get_input_embeddings()
    if hasattr(base, "model") and hasattr(base.model, "get_input_embeddings"):
        return base.model.get_input_embeddings()
    raise ValueError("Could not find input embedding layer for L2P.")


def _l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x * torch.rsqrt(torch.clamp((x * x).sum(dim=dim, keepdim=True), min=eps))


class L2PPromptPool(nn.Module):
    """
    L2P prompt pool with key-value matching and top-k prompt selection.

    Prompts are prepended to token embeddings for downstream inference.
    """

    def __init__(self, config: L2PPromptPoolConfig, embed_dim: int):
        super().__init__()
        self.config = config
        self.embed_dim = embed_dim

        pool_shape = (config.pool_size, config.prompt_length, embed_dim)
        if config.prompt_init == "zero":
            prompt_pool = torch.zeros(pool_shape)
        elif config.prompt_init == "normal":
            prompt_pool = torch.randn(pool_shape)
        elif config.prompt_init == "uniform":
            prompt_pool = torch.empty(pool_shape)
            nn.init.uniform_(prompt_pool, -1.0, 1.0)
        else:
            raise ValueError(f"Unknown prompt_init: {config.prompt_init}")

        self.prompt_pool = nn.Parameter(prompt_pool)

        if config.prompt_key:
            key_shape = (config.pool_size, embed_dim)
            if config.prompt_key_init == "zero":
                prompt_key = torch.zeros(key_shape)
            elif config.prompt_key_init == "uniform":
                prompt_key = torch.empty(key_shape)
                nn.init.uniform_(prompt_key, -1.0, 1.0)
            else:
                raise ValueError(f"Unknown prompt_key_init: {config.prompt_key_init}")
            self.prompt_key = nn.Parameter(prompt_key)
        else:
            self.prompt_key = None

    def _compute_query(
        self,
        input_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = torch.ones(
                input_embeds.shape[:2], device=input_embeds.device, dtype=torch.long
            )
        mask = attention_mask.unsqueeze(-1).to(input_embeds.dtype)
        denom = mask.sum(dim=1).clamp(min=1.0)
        mean = (input_embeds * mask).sum(dim=1) / denom

        if self.config.embedding_key == "mean":
            return mean
        if self.config.embedding_key == "max":
            masked = input_embeds.masked_fill(mask == 0, -1e4)
            return masked.max(dim=1).values
        if self.config.embedding_key == "mean_max":
            masked = input_embeds.masked_fill(mask == 0, -1e4)
            max_val = masked.max(dim=1).values
            return max_val + 2.0 * mean
        if self.config.embedding_key == "cls":
            pad_len = (attention_mask == 0).sum(dim=1)
            indices = pad_len.clamp(max=input_embeds.shape[1] - 1)
            return input_embeds[torch.arange(input_embeds.shape[0]), indices]
        raise ValueError(f"Unknown embedding_key: {self.config.embedding_key}")

    def _get_prompt_keys(self) -> torch.Tensor:
        if self.prompt_key is not None:
            return self.prompt_key
        return self.prompt_pool.mean(dim=1)

    def select_prompts(
        self,
        input_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        task_idx: int,
        train: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            selected_prompts: (bs, top_k * prompt_length, embed_dim)
            reduce_sim: scalar tensor for pull loss
            prompt_idx: (bs, top_k) prompt indices
        """
        batch_size = input_embeds.shape[0]
        top_k = min(self.config.top_k, self.config.pool_size)

        query = _l2_normalize(self._compute_query(input_embeds, attention_mask))
        keys = _l2_normalize(self._get_prompt_keys())

        use_prompt_mask = self.config.use_prompt_mask and train and task_idx >= 0
        if use_prompt_mask:
            start = task_idx * top_k
            end = (task_idx + 1) * top_k
            if end <= self.config.pool_size:
                prompt_idx = torch.arange(start, end, device=input_embeds.device)
                prompt_idx = prompt_idx.unsqueeze(0).repeat(batch_size, 1)
            else:
                use_prompt_mask = False

        if not use_prompt_mask:
            sim = torch.matmul(query, keys.t())  # (bs, pool_size)
            _, prompt_idx = torch.topk(sim, k=top_k, dim=1, largest=True, sorted=True)

        selected_keys = keys[prompt_idx]  # (bs, top_k, embed_dim)
        reduce_sim = (selected_keys * query.unsqueeze(1)).sum() / batch_size

        prompts = self.prompt_pool[prompt_idx]  # (bs, top_k, prompt_length, embed_dim)
        prompts = prompts.reshape(batch_size, top_k * self.config.prompt_length, self.embed_dim)

        return prompts, reduce_sim, prompt_idx

    def build_prompted_inputs(
        self,
        input_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        prompt_embeds: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Insert prompts after left padding and return new inputs.
        """
        batch_size, seq_len, embed_dim = input_embeds.shape
        prompt_len = prompt_embeds.shape[1]
        device = input_embeds.device

        pad_lens = (attention_mask == 0).sum(dim=1)
        shifted_position_ids = position_ids + attention_mask * prompt_len

        prompt_positions = torch.arange(prompt_len, device=device, dtype=position_ids.dtype)

        embeds = []
        attention = []
        positions = []

        for i in range(batch_size):
            pad_len = int(pad_lens[i].item())
            if pad_len > 0:
                left_embeds = input_embeds[i, :pad_len]
                left_attention = attention_mask.new_zeros((pad_len,))
                left_positions = position_ids.new_zeros((pad_len,))
            else:
                left_embeds = input_embeds.new_empty((0, embed_dim))
                left_attention = attention_mask.new_empty((0,))
                left_positions = position_ids.new_empty((0,))

            right_embeds = input_embeds[i, pad_len:]
            right_attention = attention_mask[i, pad_len:]
            right_positions = shifted_position_ids[i, pad_len:]

            embeds.append(torch.cat([left_embeds, prompt_embeds[i], right_embeds], dim=0))
            attention.append(
                torch.cat([left_attention, attention_mask.new_ones((prompt_len,)), right_attention], dim=0)
            )
            positions.append(torch.cat([left_positions, prompt_positions, right_positions], dim=0))

        new_embeds = torch.stack(embeds, dim=0)
        new_attention = torch.stack(attention, dim=0)
        new_position_ids = torch.stack(positions, dim=0)

        return new_embeds, new_attention, new_position_ids


def prepare_l2p_inputs(
    model: nn.Module,
    prompt_pool: L2PPromptPool,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    task_idx: int,
    train: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    embedding_layer = get_input_embedding_layer(model)
    input_embeds = embedding_layer(input_ids)
    prompt_embeds, reduce_sim, prompt_idx = prompt_pool.select_prompts(
        input_embeds=input_embeds,
        attention_mask=attention_mask,
        task_idx=task_idx,
        train=train,
    )
    new_embeds, new_attention, new_position_ids = prompt_pool.build_prompted_inputs(
        input_embeds=input_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        prompt_embeds=prompt_embeds,
    )
    return new_embeds, new_attention, new_position_ids, reduce_sim, prompt_idx
