# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half

from kvpress.presses.scorer_press import ScorerPress
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeAttention
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

from kvpress.utils import get_prerope_query_states


@dataclass
class SnapKVPress(ScorerPress):
    """
    SnapKV (https://arxiv.org/abs/2404.14469) use the attention of the latest window_size tokens to estimate the
    importance of the previous KV pairs. We use the default settings from:
    https://github.com/FasterDecoding/SnapKV/blob/main/snapkv/monkeypatch/snapkv_utils.py#L24
    """

    compression_ratio: float = 0.0
    window_size: int = 32
    kernel_size: int = 7

    @staticmethod
    def compute_window_attention(module, hidden_states, keys, window_size, position_embeddings):
        """
        Compute trailing-window queries over a full KV cache.

        ``hidden_states`` may contain either the full prefill or only the trailing
        window. ``keys`` always contains the complete prefix + window cache.
        """

        hidden_length = hidden_states.shape[1]
        total_length = keys.shape[2]
        num_heads = module.config.num_attention_heads
        head_dim = module.head_dim
        num_key_value_groups = num_heads // module.config.num_key_value_heads
        if hidden_length < window_size:
            raise ValueError(f"Need {window_size} query states, got {hidden_length}")
        if total_length <= window_size:
            raise ValueError(f"Cache length {total_length} must be greater than window size {window_size}")

        # # Get last window_size queries
        # if hasattr(module, "q_proj"):
        #     query_states = module.q_proj(hidden_states[:, -window_size:])
        # elif hasattr(module, "qkv_proj"):
        #     qkv = module.qkv_proj(hidden_states[:, -window_size:])
        #     query_states = qkv[..., : num_heads * head_dim]
        # else:
        #     raise NotImplementedError(f"SnapKV not yet implemented for {module.__class__}.")

        # query_states = query_states.view(bsz, window_size, num_heads, head_dim).transpose(1, 2)

        # if isinstance(module, (Qwen3MoeAttention, Qwen3Attention)):
        #     query_states = module.q_norm(query_states)

        # Get last window_size queries
        query_states = get_prerope_query_states(module, hidden_states[:, -window_size:])

        # Apply RoPE
        cos, sin = position_embeddings
        cos, sin = cos[:, -window_size:], sin[:, -window_size:]
        query_states = (query_states * cos.unsqueeze(1)) + (rotate_half(query_states) * sin.unsqueeze(1))

        # Compute attention for the historical tokens and protect the window.
        key_states = repeat_kv(keys, num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
        prefix_length = total_length - window_size
        key_positions = torch.arange(total_length, device=attn_weights.device)
        query_positions = prefix_length + torch.arange(window_size, device=attn_weights.device)
        causal_mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
        attn_weights.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = attn_weights[..., :-window_size]

        return attn_weights

    def _score_with_window(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
        window_size: int,
    ) -> torch.Tensor:

        bsz, num_key_value_heads, total_length, _ = keys.shape
        num_key_value_groups = module.config.num_attention_heads // num_key_value_heads

        if total_length <= window_size:
            raise ValueError("Cache length should be greater than the window size")

        if attentions is not None:
            attn_weights = attentions[..., -window_size:, :-window_size]
        else:
            attn_weights = self.compute_window_attention(
                module, hidden_states, keys, window_size, kwargs["position_embeddings"]
            )

        scores = attn_weights.mean(dim=-2)
        scores = F.avg_pool1d(scores, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)

        # Average per group (https://github.com/FasterDecoding/SnapKV/issues/22)
        scores = scores.view(bsz, num_key_value_heads, num_key_value_groups, total_length - window_size)
        scores = scores.mean(2)

        # Add back the observation window. Use max score to make sure the window is not pruned.
        scores = F.pad(scores, (0, window_size), value=scores.max().item())

        return scores

    def score_from_window(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
        window_size: int | None = None,
    ) -> torch.Tensor:
        """Score a full cache from only its trailing observation-window states."""

        window_size = hidden_states.shape[1] if window_size is None else window_size
        return self._score_with_window(module, hidden_states, keys, values, attentions, kwargs, window_size)

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        return self._score_with_window(
            module,
            hidden_states,
            keys,
            values,
            attentions,
            kwargs,
            self.window_size,
        )
