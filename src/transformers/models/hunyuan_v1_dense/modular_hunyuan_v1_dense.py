# coding=utf-8
# Copyright (C) 2025 THL A29 Limited, a Tencent company and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch HunYuanDenseV1 model."""

import math
import warnings
from typing import Optional, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_13
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.utils.import_utils import is_torch_fx_available

from ...generation import GenerationMixin
from ...processing_utils import Unpack
from ...utils import TransformersKwargs
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaForSequenceClassification,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
    rotate_half,
)

from .configuration_hunyuan_v1_dense import HunYuanDenseV1Config

logger = logging.get_logger(__name__)

class HunYuanDenseV1RMSNorm(LlamaRMSNorm):
    pass

class HunYuanDenseV1MLP(LlamaMLP):
    def __init__(self, config: HunYuanDenseV1Config, layer_idx=None, is_shared_mlp=False):
        super().__init__(config)
        self.layer_idx = layer_idx

class HunYuanDenseV1Attention(LlamaAttention):
    def __init__(self, config: HunYuanDenseV1Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.q_norm = HunYuanDenseV1RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = HunYuanDenseV1RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape

# class HunYuanSdpaAttention(HunYuanDenseV1Attention):
#     """
#     HunYuan attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
#     `HunYuanDenseV1Attention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt
#     to SDPA API.
#     """
#     pass

# HUNYUAN_ATTENTION_CLASSES = {
#     "eager": HunYuanDenseV1Attention,
#     "sdpa": HunYuanSdpaAttention,
# }


class HunYuanDenseV1DecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: HunYuanDenseV1Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = HunYuanDenseV1Attention(config=config, layer_idx=layer_idx)

        self.mlp = HunYuanDenseV1MLP(config, layer_idx=layer_idx, is_shared_mlp=False)
        if config.norm_type == "hf_rms" or config.norm_type == "rms":
            self.input_layernorm = HunYuanDenseV1RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = HunYuanDenseV1RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        elif config.norm_type == "fused" or config.norm_type == "torch_nn":
            self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            assert False, "other norm_type are not supported"

class HunYuanDenseV1PreTrainedModel(PreTrainedModel):
    config_class = HunYuanDenseV1Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["HunYuanDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class HunYuanDenseV1Model(HunYuanDenseV1PreTrainedModel):
    def __init__(self, config: HunYuanDenseV1Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [HunYuanDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = HunYuanDenseV1RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class HunYuanDenseV1ForCausalLM(LlamaForCausalLM):
    pass

class HunYuanDenseV1ForSequenceClassification(LlamaForSequenceClassification):
    pass

__all__ = [
    "HunYuanDenseV1ForCausalLM",
    "HunYuanDenseV1Model",
    "HunYuanDenseV1PreTrainedModel",
    "HunYuanDenseV1ForSequenceClassification",
]
