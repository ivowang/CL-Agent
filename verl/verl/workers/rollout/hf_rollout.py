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
Rollout with huggingface models.
TODO: refactor this class. Currently, it will hang when using FSDP HybridShard. We should actually create a single
GPU model. Then, get full state_dict and bind the state_dict to the single GPU model. Then, use the single GPU model
to perform generation.
"""

import contextlib
import math

import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import GenerationConfig

from verl import DataProto
from verl.utils.device import get_device_name, get_torch_device
from verl.utils.torch_functional import get_response_mask

from .base import BaseRollout
from ragen.cl_methods.l2p_prompt import prepare_l2p_inputs

__all__ = ["HFRollout"]


class HFRollout(BaseRollout):
    def __init__(self, module: nn.Module, config, model_config=None, device_mesh=None):
        super().__init__(config=config, model_config=model_config, device_mesh=device_mesh)
        self.module = module

    async def resume(self, tags: list[str]):
        return

    async def update_weights(self, weights, **kwargs):
        # HF rollout shares the actor module in this backend; no weight sync needed.
        for _ in weights:
            pass
        return

    async def release(self):
        return

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        batch_size = prompts.batch.batch_size[0]
        micro_batch_size = self.config.get("micro_batch_size", None)
        if micro_batch_size is None or micro_batch_size <= 0:
            num_chunks = 1
        else:
            num_chunks = max(math.ceil(batch_size / micro_batch_size), 1)
        batch_prompts = prompts.chunk(chunks=num_chunks)
        output = [self._generate_minibatch(p) for p in batch_prompts]
        output = DataProto.concat(output)
        return output

    @torch.no_grad()
    def _generate_minibatch(self, prompts: DataProto) -> DataProto:
        # make sampling args can be overridden by inputs
        do_sample = prompts.meta_info.get("do_sample", self.config.do_sample)
        is_validate = prompts.meta_info.get("validate", False)

        temperature = prompts.meta_info.get("temperature", self.config.temperature)
        response_length = prompts.meta_info.get("response_length", self.config.response_length)
        top_p = prompts.meta_info.get("top_p", self.config.get("top_p", 1.0))
        top_k = max(0, prompts.meta_info.get("top_k", self.config.get("top_k", 0)))  # to be compatible with vllm

        if not do_sample:
            # do_sample==False -> greedy decoding
            kwargs = {
                "do_sample": False,
                "num_beams": 1,
            }
        elif is_validate:
            # do validate and do sample -> use val_kwargs
            kwargs = {
                "do_sample": True,
                "num_beams": 1,
                "top_k": max(0, self.config.val_kwargs.top_k),  # to be compatible with vllm
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "num_return_sequences": 1,  # if validate, already repeat in ray_trainer
            }
        else:
            # do_sample -> use rollout config
            kwargs = {
                "do_sample": True,
                "num_beams": 1,
                "top_p": top_p,
                "top_k": top_k,
                "temperature": temperature,
                # already repeat in ray_trainer
                # https://github.com/volcengine/verl/blob/2fdfbdcba6f2e076f64bc47922d8fe6cf7dc7da5/verl/trainer/ppo/ray_trainer.py#L1117
                "num_return_sequences": 1,
            }

        # make config according to generate mode
        generation_config = GenerationConfig(**kwargs)

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        prompt_length = idx.size(1)
        attention_mask = prompts.batch["attention_mask"]  # left-padded attention_mask
        position_ids = prompts.batch["position_ids"]

        orig_idx = idx
        orig_attention_mask = attention_mask
        orig_position_ids = position_ids
        orig_prompt_length = prompt_length
        l2p_prompt_len = 0
        pad_lens = None

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]
        pad_token_id = prompts.meta_info["pad_token_id"]

        self.module.eval()
        param_ctx = contextlib.nullcontext()

        if isinstance(self.module, FSDP):
            # recurse need to set to False according to https://github.com/pytorch/pytorch/issues/100069
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
        with param_ctx, torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            if hasattr(self.module, "l2p_prompt_pool") or (
                hasattr(self.module, "_fsdp_wrapped_module")
                and hasattr(self.module._fsdp_wrapped_module, "l2p_prompt_pool")
            ):
                prompt_pool = (
                    self.module.l2p_prompt_pool
                    if hasattr(self.module, "l2p_prompt_pool")
                    else self.module._fsdp_wrapped_module.l2p_prompt_pool
                )
                pad_lens = (attention_mask == 0).sum(dim=1)
                l2p_prompt_len = prompt_pool.config.prompt_length * min(
                    prompt_pool.config.top_k, prompt_pool.config.pool_size
                )

                inputs_embeds, attention_mask, position_ids, _, _ = prepare_l2p_inputs(
                    model=self.module,
                    prompt_pool=prompt_pool,
                    input_ids=idx,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    task_idx=prompts.meta_info.get("current_task_idx", 0),
                    train=not is_validate,
                )
                # Build input_ids with prompt slots so sequence length matches inputs_embeds.
                if l2p_prompt_len > 0:
                    idx_with_prompts = idx.new_full(
                        (idx.size(0), idx.size(1) + l2p_prompt_len),
                        fill_value=pad_token_id,
                    )
                    for i in range(idx.size(0)):
                        pad_len = int(pad_lens[i].item())
                        if pad_len > 0:
                            idx_with_prompts[i, :pad_len] = idx[i, :pad_len]
                        idx_with_prompts[i, pad_len + l2p_prompt_len :] = idx[i, pad_len:]
                    idx = idx_with_prompts

                output = self.module.generate(
                    input_ids=idx,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    do_sample=do_sample,
                    max_new_tokens=response_length,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                    generation_config=generation_config,
                    output_scores=False,  # this is potentially very large
                    return_dict_in_generate=True,
                    use_cache=True,
                )
            else:
                output = self.module.generate(
                    input_ids=idx,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    do_sample=do_sample,
                    max_new_tokens=response_length,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                    generation_config=generation_config,
                    output_scores=False,  # this is potentially very large
                    return_dict_in_generate=True,
                    use_cache=True,
                )

        # TODO: filter out the seq with no answers like ds-chat
        seq = output.sequences
        if l2p_prompt_len > 0 and pad_lens is not None:
            seq_no_prompt = seq.new_empty((seq.size(0), seq.size(1) - l2p_prompt_len))
            for i in range(seq.size(0)):
                pad_len = int(pad_lens[i].item())
                seq_no_prompt[i] = torch.cat(
                    [seq[i, :pad_len], seq[i, pad_len + l2p_prompt_len :]], dim=0
                )
            seq = seq_no_prompt
            idx = orig_idx
            attention_mask = orig_attention_mask
            position_ids = orig_position_ids
            prompt_length = orig_prompt_length
        generated_batch_size = seq.size(0)  # bs * num_return_sequences

        # huggingface generate will stop generating when all the batch reaches [EOS].
        # We have to pad to response_length
        sequence_length = prompt_length + self.config.response_length
        delta_length = sequence_length - seq.shape[1]

        if delta_length > 0:
            delta_tokens = torch.ones(size=(generated_batch_size, delta_length), device=seq.device, dtype=seq.dtype)
            delta_tokens = pad_token_id * delta_tokens
            seq = torch.cat((seq, delta_tokens), dim=1)
        assert seq.shape[1] == sequence_length

        # make necessary reputations if num_return_sequences > 1
        num_return_sequences = kwargs.get("num_return_sequences", 1)
        if num_return_sequences > 1:
            position_ids = position_ids.repeat_interleave(num_return_sequences, dim=0)
            attention_mask = attention_mask.repeat_interleave(num_return_sequences, dim=0)

        prompt = seq[:, :prompt_length]  # (generated_batch_size, prompt_length)
        response = seq[:, prompt_length:]  # (generated_batch_size, response_length)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(generated_batch_size, 1)

        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": prompt,
                "responses": response,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=generated_batch_size,
        )

        # empty cache before compute old_log_prob
        get_torch_device().empty_cache()

        self.module.train()
        return DataProto(batch=batch)
