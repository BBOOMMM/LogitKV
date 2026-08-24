# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LogitKV: downstream-Fisher-aware KV-cache compression."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F
from transformers import DynamicCache, PreTrainedModel, QuantizedCache
from transformers.models.llama.modeling_llama import rotate_half

from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress
from kvpress.utils import get_prerope_query_states

logger = logging.getLogger(__name__)

LogitKVScoreMode = Literal["separable", "coupled_diag", "coupled_full"]
LOGITKV_SCORE_MODES = ("separable", "coupled_diag", "coupled_full")


def _output_projection_by_head(module: nn.Module) -> torch.Tensor:
    """Return ``W_O`` as ``[num_attention_heads, head_dim, hidden_size]``."""

    if not hasattr(module, "o_proj"):
        raise NotImplementedError(f"LogitKV requires an o_proj layer, but {module.__class__.__name__} has none")

    num_heads = module.config.num_attention_heads
    head_dim = module.head_dim
    projection = module.o_proj.weight.transpose(0, 1)
    expected_rows = num_heads * head_dim
    if projection.shape[0] != expected_rows:
        raise ValueError(
            f"Unexpected o_proj shape {tuple(module.o_proj.weight.shape)}: "
            f"expected an input dimension of {expected_rows}"
        )
    return projection.reshape(num_heads, head_dim, projection.shape[-1])


def fisher_quadratic_sensitivity(
    values: torch.Tensor,
    output_grad: torch.Tensor,
    module: nn.Module,
    fisher_window: int = 32,
) -> torch.Tensor:
    """Compute ``Q = mean_t((g_t^T (V_i W_O))^2)`` without forming ``V W_O``."""

    if values.ndim != 4:
        raise ValueError(f"values must be rank 4, got shape {tuple(values.shape)}")
    if output_grad.ndim != 3:
        raise ValueError(f"output_grad must be rank 3, got shape {tuple(output_grad.shape)}")
    if fisher_window <= 0:
        raise ValueError("fisher_window must be positive")

    batch_size, num_kv_heads, _, head_dim = values.shape
    if output_grad.shape[0] != batch_size:
        raise ValueError("values and output_grad must have the same batch size")
    if output_grad.shape[1] == 0:
        raise ValueError("output_grad must contain at least one token")

    num_heads = module.config.num_attention_heads
    if num_heads % num_kv_heads != 0:
        raise ValueError(f"{num_heads} attention heads cannot be grouped into {num_kv_heads} KV heads")
    if module.head_dim != head_dim:
        raise ValueError(f"Value head dimension {head_dim} does not match module.head_dim {module.head_dim}")

    num_key_value_groups = num_heads // num_kv_heads
    window = min(fisher_window, output_grad.shape[1])
    grad_window = output_grad[:, -window:]
    output_projection = _output_projection_by_head(module)

    # Compute each query head separately to avoid the [B, H, K, hidden_size]
    # V @ W_O intermediate. Squaring/averaging in float32 is important for the
    # small Fisher values produced by bf16/fp16 models.
    quadratic_by_head = []
    for head_idx in range(num_heads):
        kv_head_idx = head_idx // num_key_value_groups
        head_projection = output_projection[head_idx].to(device=grad_window.device, dtype=grad_window.dtype)
        grad_head = torch.matmul(grad_window, head_projection.transpose(-1, -2))
        head_values = values[:, kv_head_idx].to(device=grad_head.device, dtype=grad_head.dtype)
        dot = torch.matmul(head_values, grad_head.transpose(-1, -2))
        quadratic_by_head.append(dot.float().square().mean(dim=-1))

    quadratic = torch.stack(quadratic_by_head, dim=1)
    return quadratic.reshape(batch_size, num_kv_heads, num_key_value_groups, values.shape[2]).mean(dim=2)


def fisher_rms_sensitivity(
    values: torch.Tensor,
    output_grad: torch.Tensor,
    module: nn.Module,
    fisher_window: int = 32,
    fisher_eps: float = 1e-12,
) -> torch.Tensor:
    """Compute LogitKV's RMS downstream empirical-Fisher sensitivity.

    This evaluates

    ``sqrt(mean_t((g_t^T (V_i W_O))^2) + fisher_eps)``

    without materializing ``V W_O``. For grouped-query attention, the quadratic
    sensitivities of the query heads sharing one KV head are averaged before the
    square root, matching CriticalKV's query-group aggregation.

    Parameters
    ----------
    values:
        Cached values with shape ``[batch, num_kv_heads, seq_len, head_dim]``.
    output_grad:
        Gradient of the sampled final-token log-probability with respect to the
        attention layer output, shaped ``[batch, query_len, hidden_size]``.
    module:
        Attention module owning ``o_proj``.
    fisher_window:
        Number of trailing layer-output positions used for the empirical Fisher.
    fisher_eps:
        Non-negative stabilizer added to ``Q`` inside the square root.
    """

    if fisher_eps < 0:
        raise ValueError("fisher_eps must be non-negative")
    quadratic = fisher_quadratic_sensitivity(values, output_grad, module, fisher_window)
    return torch.sqrt(quadratic.clamp_min(0.0) + fisher_eps)


def coupled_fisher_quadratic_sensitivity(
    keys: torch.Tensor,
    values: torch.Tensor,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    output_grad: torch.Tensor,
    module: nn.Module,
    mode: Literal["coupled_diag", "coupled_full"],
    fisher_window: int = 32,
) -> torch.Tensor:
    """Compute an attention-gradient-coupled empirical-Fisher quadratic.

    For a KV token ``i`` shared by a group of query heads, define its
    value-path contribution at observation-window position ``t`` as

    ``c[t, i] = sum_h A[h, t, i] * g[t]^T (V[i] W_O[h])``.

    ``coupled_diag`` returns ``sum_t c[t, i]^2`` (a position-diagonal
    approximation), while ``coupled_full`` returns ``(sum_t c[t, i])^2``.
    Query heads sharing a KV head are summed before squaring because evicting
    one KV token removes their contributions jointly.

    The computation streams over KV/query heads. It therefore never retains a
    full ``[batch, heads, window, context]`` attention tensor.
    """

    if mode not in ("coupled_diag", "coupled_full"):
        raise ValueError(f"Unsupported coupled LogitKV score mode: {mode}")
    if keys.ndim != 4 or values.ndim != 4:
        raise ValueError("keys and values must be rank 4")
    if keys.shape != values.shape:
        raise ValueError(f"keys and values must have identical shapes, got {keys.shape} and {values.shape}")
    if hidden_states.ndim != 3 or output_grad.ndim != 3:
        raise ValueError("hidden_states and output_grad must be rank 3")
    if fisher_window <= 0:
        raise ValueError("fisher_window must be positive")
    if not isinstance(position_embeddings, (tuple, list)) or len(position_embeddings) != 2:
        raise ValueError("position_embeddings must be a (cos, sin) pair")

    batch_size, num_kv_heads, total_length, head_dim = values.shape
    if hidden_states.shape[0] != batch_size or output_grad.shape[0] != batch_size:
        raise ValueError("keys, values, hidden_states, and output_grad must have the same batch size")
    if hidden_states.shape[1] == 0 or output_grad.shape[1] == 0:
        raise ValueError("hidden_states and output_grad must contain at least one token")
    if module.head_dim != head_dim:
        raise ValueError(f"Value head dimension {head_dim} does not match module.head_dim {module.head_dim}")

    num_heads = module.config.num_attention_heads
    if num_heads % num_kv_heads != 0:
        raise ValueError(f"{num_heads} attention heads cannot be grouped into {num_kv_heads} KV heads")
    num_key_value_groups = num_heads // num_kv_heads
    window = min(fisher_window, hidden_states.shape[1], output_grad.shape[1])
    if total_length < window:
        raise ValueError(f"Cache length {total_length} is shorter than coupled Fisher window {window}")

    hidden_window = hidden_states[:, -window:]
    grad_window = output_grad[:, -window:]
    query_states = get_prerope_query_states(module, hidden_window)
    cos, sin = position_embeddings
    cos = cos[:, -window:].to(device=query_states.device, dtype=query_states.dtype)
    sin = sin[:, -window:].to(device=query_states.device, dtype=query_states.dtype)
    query_states = (query_states * cos.unsqueeze(1)) + (rotate_half(query_states) * sin.unsqueeze(1))

    output_projection = _output_projection_by_head(module)
    prefix_length = total_length - window
    key_positions = torch.arange(total_length, device=query_states.device)
    query_positions = prefix_length + torch.arange(window, device=query_states.device)
    causal_mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
    scale = 1.0 / math.sqrt(head_dim)

    quadratic_by_kv_head = []
    for kv_head_idx in range(num_kv_heads):
        head_values = values[:, kv_head_idx]
        head_keys = keys[:, kv_head_idx]
        # [batch, context, window]. Accumulating in float32 keeps cancellation
        # in coupled_full and small diagonal contributions numerically stable.
        position_contribution = torch.zeros(
            batch_size,
            total_length,
            window,
            device=values.device,
            dtype=torch.float32,
        )
        first_query_head = kv_head_idx * num_key_value_groups
        for head_idx in range(first_query_head, first_query_head + num_key_value_groups):
            query_head = query_states[:, head_idx]
            attention_logits = torch.matmul(query_head, head_keys.transpose(-1, -2)) * scale
            attention_logits.masked_fill_(causal_mask.unsqueeze(0), float("-inf"))
            attention_weights = torch.softmax(attention_logits, dim=-1, dtype=torch.float32)

            head_projection = output_projection[head_idx].to(device=grad_window.device, dtype=grad_window.dtype)
            grad_head = torch.matmul(grad_window, head_projection.transpose(-1, -2))
            dot = torch.matmul(
                head_values.to(device=grad_head.device, dtype=grad_head.dtype),
                grad_head.transpose(-1, -2),
            )
            position_contribution.add_(attention_weights.transpose(-1, -2) * dot.float())

        if mode == "coupled_diag":
            quadratic_by_kv_head.append(position_contribution.square().sum(dim=-1))
        else:
            quadratic_by_kv_head.append(position_contribution.sum(dim=-1).square())

    return torch.stack(quadratic_by_kv_head, dim=1)


def assert_root_squared_ranking(
    base_scores: torch.Tensor,
    fisher_quadratic: torch.Tensor,
    k: int,
    fisher_eps: float = 0.0,
    score_root: torch.Tensor | None = None,
    layer_idx: int | None = None,
) -> bool:
    """Log whether ``A * sqrt(Q)`` and ``A^2 * Q`` select the same top-k set."""

    if base_scores.shape != fisher_quadratic.shape:
        raise ValueError(
            f"base_scores and fisher_quadratic must have identical shapes, got "
            f"{tuple(base_scores.shape)} and {tuple(fisher_quadratic.shape)}"
        )
    if fisher_eps < 0:
        raise ValueError("fisher_eps must be non-negative")
    if not 0 <= k <= base_scores.shape[-1]:
        raise ValueError(f"k must be in [0, {base_scores.shape[-1]}], got {k}")

    quadratic = fisher_quadratic.float().clamp_min(0.0) + fisher_eps
    expected_root = base_scores.float() * torch.sqrt(quadratic)
    if score_root is None:
        score_root = expected_root
    elif not torch.allclose(score_root.float(), expected_root, rtol=1e-5, atol=1e-8):
        raise RuntimeError("LogitKV score is not A * sqrt(Q + fisher_eps)")
    # Keep this expression independent of score_root so the check catches a
    # missing root or an epsilon placed outside sqrt in the actual scorer.
    score_squared = base_scores.float().square() * quadratic
    # Independent float32 evaluations can swap nearly tied tokens within the
    # selected Top-K ordering. Compression only depends on membership, so do
    # not treat those harmless ordering differences as a failed sanity check.
    root_topk = score_root.float().topk(k, dim=-1, sorted=False).indices
    squared_topk = score_squared.topk(k, dim=-1, sorted=False).indices
    root_selected = torch.zeros_like(base_scores, dtype=torch.bool)
    squared_selected = torch.zeros_like(base_scores, dtype=torch.bool)
    root_selected.scatter_(-1, root_topk, True)
    squared_selected.scatter_(-1, squared_topk, True)
    if not torch.equal(root_selected, squared_selected):
        mismatch_count = torch.logical_xor(root_selected, squared_selected).sum().item()
        logger.warning(
            "LogitKV root/squared Top-K selections differ by %d membership entries "
            "(layer=%s, keep=%d/%d, compression_ratio=%.6f); "
            "continuing with the root-form LogitKV score",
            mismatch_count,
            "unknown" if layer_idx is None else layer_idx,
            k,
            base_scores.shape[-1],
            1 - k / base_scores.shape[-1],
        )
        return False
    return True


@dataclass
class _LayerState:
    module: nn.Module
    cache: DynamicCache
    keys: torch.Tensor
    values: torch.Tensor
    base_scores: torch.Tensor
    hidden_states: torch.Tensor | None = None
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None


@dataclass
class LogitKVPress(BasePress):
    """Memory-efficient online LogitKV with downstream Fisher sensitivity.

    The context is split into a detached no-grad prefix and a differentiable
    trailing Fisher window. ``fisher_positions`` trailing logit positions each
    draw ``fisher_labels`` independent labels. Their per-layer Fisher quadratic
    forms are averaged before the square root. ``score_mode`` selects the legacy
    separable attention-times-Fisher score or one of two position-wise coupled
    formulations. Coupled scores can optionally be smoothed across neighboring
    KV positions with ``coupled_kernel_size``. The full cache is then compressed
    with CriticalKV's Stage-1 safeguard and per-head Top-K rule.
    """

    press: ScorerPress
    fisher_window: int = 32
    fisher_positions: int = 1
    fisher_labels: int = 1
    score_mode: LogitKVScoreMode = "separable"
    coupled_kernel_size: int = 1
    first_stage_ratio: float = 0.5
    attention_eps: float = 0.0
    fisher_eps: float = 1e-12
    sanity_check: bool = True
    fisher_seed: int | None = None
    profile: bool = False

    _states: dict[int, _LayerState] = field(default_factory=dict, init=False, repr=False)
    _quadratic_sums: dict[int, torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    _gradient_counts: dict[int, int] = field(default_factory=dict, init=False, repr=False)
    _scores: dict[int, torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    _active: bool = field(default=False, init=False, repr=False)
    _total_length: int = field(default=0, init=False, repr=False)
    _gradient_handles: list = field(default_factory=list, init=False, repr=False)
    _profile_values: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    last_sampled_token_ids: torch.Tensor | None = field(default=None, init=False, repr=False)
    last_ranking_check_passed: bool | None = field(default=None, init=False, repr=False)
    last_full_cache_length: int | None = field(default=None, init=False, repr=False)
    last_profile: dict[str, float | int | str] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.press, ScorerPress):
            raise TypeError("LogitKVPress requires a ScorerPress as input")
        self._validate_fisher_configuration()
        self._validate_score_mode()
        if not 0 <= self.first_stage_ratio <= 1:
            raise ValueError("first_stage_ratio must be between 0 and 1")
        if self.attention_eps < 0:
            raise ValueError("attention_eps must be non-negative")
        if self.fisher_eps < 0:
            raise ValueError("fisher_eps must be non-negative")

    @property
    def _fisher_probe_count(self) -> int:
        return self.fisher_positions * self.fisher_labels

    def _validate_fisher_configuration(self) -> None:
        if self.fisher_window <= 0:
            raise ValueError("fisher_window must be positive")
        if self.fisher_positions <= 0:
            raise ValueError("fisher_positions must be positive")
        if self.fisher_positions > self.fisher_window:
            raise ValueError("fisher_positions must not exceed fisher_window")
        if self.fisher_labels <= 0:
            raise ValueError("fisher_labels must be positive")

    def _validate_score_mode(self) -> None:
        if self.score_mode not in LOGITKV_SCORE_MODES:
            raise ValueError(f"score_mode must be one of {', '.join(LOGITKV_SCORE_MODES)}, got {self.score_mode!r}")
        if self.score_mode != "separable" and self.attention_eps != 0:
            raise ValueError("attention_eps must be 0 for coupled LogitKV score modes")
        if self.coupled_kernel_size <= 0 or self.coupled_kernel_size % 2 == 0:
            raise ValueError("coupled_kernel_size must be a positive odd integer")
        if self.score_mode == "separable" and self.coupled_kernel_size != 1:
            raise ValueError("coupled_kernel_size only applies to coupled LogitKV score modes")

    @property
    def compression_ratio(self) -> float:
        return self.press.compression_ratio

    @compression_ratio.setter
    def compression_ratio(self, value: float) -> None:
        if not 0 <= value < 1:
            raise ValueError("compression_ratio must be between 0 and 1")
        self.press.compression_ratio = value

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        raise RuntimeError("LogitKV compression is finalized after the complete prefill, not per attention layer")

    @staticmethod
    def _cache_tensors(cache: DynamicCache, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(cache, QuantizedCache):
            raise NotImplementedError("LogitKV does not yet support QuantizedCache")
        if not isinstance(cache, DynamicCache):
            raise NotImplementedError(f"LogitKV currently requires DynamicCache, got {cache.__class__.__name__}")
        if layer_idx >= len(cache.layers):
            raise RuntimeError(f"Cache layer {layer_idx} is not initialized")
        cache_layer = cache.layers[layer_idx]
        if not getattr(cache_layer, "is_initialized", False):
            raise RuntimeError(f"Cache layer {layer_idx} is not initialized")
        return cache_layer.keys, cache_layer.values

    def _capture_window_attention(self, module: nn.Module, inputs: tuple, kwargs: dict, output: tuple):
        layer_idx = module.layer_idx
        if layer_idx in self._states:
            raise RuntimeError(f"Attention layer {layer_idx} ran more than once during the LogitKV window forward")
        if not isinstance(output, (tuple, list)) or len(output) < 1:
            raise RuntimeError("LogitKV expected the attention module to return a tuple")

        hidden_states = kwargs["hidden_states"]
        cache = kwargs.get("past_key_values", kwargs.get("past_key_value"))
        if cache is None:
            raise ValueError("LogitKV requires use_cache=True and an explicit DynamicCache")
        keys, values = self._cache_tensors(cache, layer_idx)
        if hidden_states.shape[1] != self.fisher_window:
            raise ValueError(f"Expected {self.fisher_window} window queries, got {hidden_states.shape[1]}")
        if keys.shape[2] != self._total_length:
            raise ValueError(f"Expected {self._total_length} full-cache keys, got {keys.shape[2]}")

        attentions = output[1] if len(output) > 1 else None
        # The prefill ran with autograd enabled, but the decoding cache itself
        # must not keep the forward graph alive after Fisher gradients are done.
        keys = keys.detach()
        values = values.detach()
        cache.layers[layer_idx].keys = keys
        cache.layers[layer_idx].values = values
        with torch.no_grad():
            if hasattr(self.press, "score_from_window"):
                base_scores = self.press.score_from_window(
                    module,
                    hidden_states.detach(),
                    keys,
                    values,
                    attentions,
                    kwargs,
                    window_size=self.fisher_window,
                ).float()
            else:
                base_scores = self.press.score(
                    module,
                    hidden_states.detach(),
                    keys,
                    values,
                    attentions,
                    kwargs,
                ).float()

        coupled_hidden_states = None
        coupled_position_embeddings = None
        if self.score_mode != "separable":
            position_embeddings = kwargs.get("position_embeddings")
            if not isinstance(position_embeddings, (tuple, list)) or len(position_embeddings) != 2:
                raise ValueError("Coupled LogitKV requires attention position_embeddings=(cos, sin)")
            coupled_hidden_states = hidden_states.detach()
            coupled_position_embeddings = tuple(embedding.detach() for embedding in position_embeddings)

        layer_output = output[0]
        if not layer_output.requires_grad:
            raise RuntimeError("Window attention output is not differentiable")

        expected_shape = keys.shape[:3]
        if base_scores.shape != expected_shape:
            raise ValueError(f"Base scorer returned {tuple(base_scores.shape)} for cache shape {tuple(expected_shape)}")
        if not torch.isfinite(base_scores).all():
            raise FloatingPointError(f"Base scorer produced non-finite values at layer {layer_idx}")
        if (base_scores < 0).any():
            raise ValueError("LogitKV requires a non-negative attention-based scorer")

        self._states[layer_idx] = _LayerState(
            module=module,
            cache=cache,
            keys=keys,
            values=values,
            base_scores=base_scores,
            hidden_states=coupled_hidden_states,
            position_embeddings=coupled_position_embeddings,
        )
        self._gradient_handles.append(layer_output.register_hook(self._make_gradient_hook(layer_idx)))
        return output

    @staticmethod
    def _logits_from_output(output) -> torch.Tensor:
        if hasattr(output, "logits"):
            return output.logits
        if isinstance(output, (tuple, list)) and output:
            return output[0]
        raise RuntimeError("LogitKV could not find final logits in the model output")

    def _sample_log_probabilities(self, logits: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Sample Fisher labels independently at each trailing probe position."""

        if logits.shape[1] < self.fisher_positions:
            raise ValueError(f"Need {self.fisher_positions} trailing logits for Fisher probes, got {logits.shape[1]}")
        probe_logits = logits[:, -self.fisher_positions :, :].float()
        if not probe_logits.requires_grad:
            raise RuntimeError(
                "Fisher probe logits do not require gradients. Run LogitKV inside its context manager "
                "and do not wrap the model call in an un-overridable inference context."
            )
        probabilities = torch.softmax(probe_logits.detach(), dim=-1)
        if not torch.isfinite(probabilities).all():
            raise FloatingPointError("Fisher probe probabilities contain non-finite values")

        generator = None
        if self.fisher_seed is not None:
            generator = torch.Generator(device=probabilities.device)
            generator.manual_seed(self.fisher_seed)
        batch_size, num_positions, vocab_size = probabilities.shape
        sampled = torch.multinomial(
            probabilities.reshape(batch_size * num_positions, vocab_size),
            num_samples=self.fisher_labels,
            replacement=True,
            generator=generator,
        ).reshape(batch_size, num_positions, self.fisher_labels)
        self.last_sampled_token_ids = sampled.detach().cpu()
        sampled_log_probabilities = torch.log_softmax(probe_logits, dim=-1).gather(-1, sampled)
        return tuple(
            sampled_log_probabilities[:, position_idx, label_idx].mean()
            for position_idx in range(num_positions)
            for label_idx in range(self.fisher_labels)
        )

    @staticmethod
    def _normal_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
        if isinstance(tensor, torch.Tensor) and torch.is_inference(tensor):
            return tensor.clone()
        return tensor

    @staticmethod
    def detach_cache(cache: DynamicCache) -> None:
        """Detach every initialized DynamicCache layer in-place."""

        if isinstance(cache, QuantizedCache) or not isinstance(cache, DynamicCache):
            raise NotImplementedError("LogitKV currently supports non-quantized DynamicCache only")
        for cache_layer in cache.layers:
            if getattr(cache_layer, "is_initialized", False):
                cache_layer.keys = cache_layer.keys.detach()
                cache_layer.values = cache_layer.values.detach()

    def _sync_for_profile(self, device: torch.device) -> None:
        if self.profile and device.type == "cuda":
            torch.cuda.synchronize(device)

    def _make_gradient_hook(self, layer_idx: int):
        def gradient_hook(output_grad: torch.Tensor) -> torch.Tensor:
            state = self._states[layer_idx]
            self._sync_for_profile(output_grad.device)
            started_at = time.perf_counter()
            with torch.no_grad():
                if self.score_mode == "separable":
                    fisher_quadratic = fisher_quadratic_sensitivity(
                        values=state.values,
                        output_grad=output_grad,
                        module=state.module,
                        fisher_window=self.fisher_window,
                    )
                else:
                    if state.hidden_states is None or state.position_embeddings is None:
                        raise RuntimeError(f"Layer {layer_idx} is missing coupled LogitKV attention state")
                    fisher_quadratic = coupled_fisher_quadratic_sensitivity(
                        keys=state.keys,
                        values=state.values,
                        hidden_states=state.hidden_states,
                        position_embeddings=state.position_embeddings,
                        output_grad=output_grad,
                        module=state.module,
                        mode=self.score_mode,
                        fisher_window=self.fisher_window,
                    )
                if layer_idx not in self._quadratic_sums:
                    self._quadratic_sums[layer_idx] = fisher_quadratic
                    self._gradient_counts[layer_idx] = 1
                else:
                    self._quadratic_sums[layer_idx].add_(fisher_quadratic)
                    self._gradient_counts[layer_idx] += 1

                gradient_count = self._gradient_counts[layer_idx]
                if gradient_count > self._fisher_probe_count:
                    raise RuntimeError(
                        f"Layer {layer_idx} received {gradient_count} Fisher gradients; "
                        f"expected {self._fisher_probe_count}"
                    )

                # Fisher matrices (and therefore their quadratic forms) are
                # averaged before applying the score-mode-specific transform.
                if gradient_count == self._fisher_probe_count:
                    fisher_quadratic_mean = self._quadratic_sums[layer_idx] / self._fisher_probe_count
                    if self.score_mode == "separable":
                        fisher_rms = torch.sqrt(fisher_quadratic_mean.clamp_min(0.0) + self.fisher_eps)
                        ranking_base = state.base_scores + self.attention_eps
                        scores = ranking_base * fisher_rms
                    else:
                        # Attention is already coupled position-wise inside Q.
                        # Rank directly by Q: sqrt and a fixed epsilon are
                        # mathematically unnecessary here and collapse tiny
                        # coupled values into large FP32 tie groups.
                        scores = fisher_quadratic_mean.clamp_min(0.0)
                        if self.coupled_kernel_size > 1:
                            scores = F.avg_pool1d(
                                scores,
                                kernel_size=self.coupled_kernel_size,
                                stride=1,
                                padding=self.coupled_kernel_size // 2,
                            )
                    if not torch.isfinite(scores).all():
                        raise FloatingPointError(f"LogitKV produced non-finite scores at layer {layer_idx}")

                    n_kept = int(self._total_length * (1 - self.compression_ratio))
                    if self.sanity_check and self.score_mode == "separable":
                        layer_ranking_check_passed = assert_root_squared_ranking(
                            ranking_base,
                            fisher_quadratic_mean,
                            n_kept,
                            self.fisher_eps,
                            score_root=scores,
                            layer_idx=layer_idx,
                        )
                        if self.last_ranking_check_passed is None:
                            self.last_ranking_check_passed = layer_ranking_check_passed
                        else:
                            self.last_ranking_check_passed &= layer_ranking_check_passed
                    elif self.sanity_check:
                        # Coupled modes use Q itself, so no monotonic score
                        # transform remains to cross-check.
                        self.last_ranking_check_passed = True
                    self._scores[layer_idx] = scores.detach()

            self._sync_for_profile(output_grad.device)
            self._profile_values["score_seconds"] += time.perf_counter() - started_at
            return output_grad

        return gradient_hook

    def _compress_cache_from_scores(self) -> None:
        incomplete = {
            layer_idx: self._gradient_counts.get(layer_idx, 0)
            for layer_idx in self._states
            if self._gradient_counts.get(layer_idx, 0) != self._fisher_probe_count
        }
        if incomplete:
            raise RuntimeError(
                f"LogitKV received incomplete Fisher gradients {incomplete}; "
                f"expected {self._fisher_probe_count} per layer"
            )
        if self._states.keys() != self._scores.keys():
            missing = sorted(self._states.keys() - self._scores.keys())
            raise RuntimeError(f"LogitKV did not receive gradients for layers {missing}")

        for layer_idx in sorted(self._states):
            state = self._states[layer_idx]
            keys = state.keys
            values = state.values
            scores = self._scores[layer_idx].clone()
            q_len = keys.shape[2]
            n_kept = int(q_len * (1 - self.compression_ratio))

            first_stage_budget = int((1 - self.compression_ratio) * q_len * self.first_stage_ratio)
            if first_stage_budget:
                top_base_indices = state.base_scores.topk(first_stage_budget, dim=-1, sorted=True).indices
                scores.scatter_(-1, top_base_indices, torch.finfo(scores.dtype).max)

            kept_indices = scores.topk(n_kept, dim=-1).indices
            gather_indices = kept_indices.unsqueeze(-1).expand(-1, -1, -1, state.module.head_dim)
            cache_layer = state.cache.layers[layer_idx]
            cache_layer.keys = keys.gather(2, gather_indices).contiguous().detach()
            cache_layer.values = values.gather(2, gather_indices).contiguous().detach()

    def prefill(
        self,
        model: PreTrainedModel,
        input_ids: torch.Tensor,
        cache: DynamicCache,
        *,
        output_attentions: bool = False,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ):
        """Run prefix no-grad + Fisher-window backward and compress the full cache."""

        if self.compression_ratio == 0:
            with torch.no_grad():
                return model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=cache,
                    output_attentions=output_attentions,
                    use_cache=True,
                    logits_to_keep=1,
                )
        if self._active:
            raise RuntimeError("A LogitKVPress instance cannot run two prefills at once")
        if not hasattr(model, "model") or not hasattr(model.model, "layers"):
            raise NotImplementedError("LogitKV expects a model with model.layers[*].self_attn")
        if not isinstance(cache, DynamicCache) or isinstance(cache, QuantizedCache):
            raise NotImplementedError("LogitKV currently supports non-quantized DynamicCache only")
        if cache.get_seq_length() != 0:
            raise ValueError("LogitKV prefill requires an empty DynamicCache")
        self._validate_fisher_configuration()
        self._validate_score_mode()
        if self.attention_eps < 0:
            raise ValueError("attention_eps must be non-negative")

        input_ids = self._normal_tensor(input_ids)
        attention_mask = self._normal_tensor(attention_mask)
        position_ids = self._normal_tensor(position_ids)
        total_length = input_ids.shape[1]
        if total_length <= self.fisher_window:
            raise ValueError(f"Context length {total_length} must be greater than fisher_window {self.fisher_window}")
        prefix_length = total_length - self.fisher_window
        prefix_ids = input_ids[:, :prefix_length]
        window_ids = input_ids[:, prefix_length:]
        prefix_attention_mask = attention_mask[:, :prefix_length] if attention_mask is not None else None
        prefix_position_ids = position_ids[..., :prefix_length] if position_ids is not None else None
        window_position_ids = position_ids[..., prefix_length:] if position_ids is not None else None

        parameters = list(model.parameters())
        parameter_requires_grad = [parameter.requires_grad for parameter in parameters]
        forward_hooks = []
        device = input_ids.device
        self._states.clear()
        self._quadratic_sums.clear()
        self._gradient_counts.clear()
        self._scores.clear()
        self._gradient_handles.clear()
        self._profile_values = {"score_seconds": 0.0}
        self.last_profile = {}
        self.last_sampled_token_ids = None
        self.last_ranking_check_passed = None
        self.last_full_cache_length = None
        self._total_length = total_length
        self._active = True

        if self.profile and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        self._sync_for_profile(device)
        total_started_at = time.perf_counter()
        try:
            for parameter in parameters:
                parameter.requires_grad_(False)

            with torch.inference_mode(False):
                self._sync_for_profile(device)
                phase_started_at = time.perf_counter()
                with torch.no_grad():
                    model(
                        input_ids=prefix_ids,
                        attention_mask=prefix_attention_mask,
                        position_ids=prefix_position_ids,
                        past_key_values=cache,
                        output_attentions=False,
                        use_cache=True,
                        logits_to_keep=1,
                    )
                self._sync_for_profile(device)
                self._profile_values["prefix_seconds"] = time.perf_counter() - phase_started_at
                if cache.get_seq_length() != prefix_length:
                    raise RuntimeError(f"Prefix cache length is {cache.get_seq_length()}, expected {prefix_length}")
                self.detach_cache(cache)

                with torch.no_grad():
                    suffix_embeds = model.get_input_embeddings()(window_ids)
                suffix_embeds = suffix_embeds.detach().requires_grad_(True)

                for layer in model.model.layers:
                    forward_hooks.append(
                        layer.self_attn.register_forward_hook(self._capture_window_attention, with_kwargs=True)
                    )

                self._sync_for_profile(device)
                phase_started_at = time.perf_counter()
                with torch.enable_grad():
                    outputs = model(
                        inputs_embeds=suffix_embeds,
                        attention_mask=attention_mask,
                        position_ids=window_position_ids,
                        past_key_values=cache,
                        output_attentions=output_attentions,
                        use_cache=True,
                        logits_to_keep=self.fisher_positions,
                    )
                self._sync_for_profile(device)
                self._profile_values["suffix_forward_seconds"] = time.perf_counter() - phase_started_at

                if cache.get_seq_length() != total_length:
                    raise RuntimeError(f"Full cache length is {cache.get_seq_length()}, expected {total_length}")
                self.last_full_cache_length = cache.get_seq_length()
                if len(self._states) != len(model.model.layers):
                    raise RuntimeError(
                        f"LogitKV captured {len(self._states)} of {len(model.model.layers)} attention layers"
                    )

                probes = self._sample_log_probabilities(self._logits_from_output(outputs))
                self._sync_for_profile(device)
                phase_started_at = time.perf_counter()
                for sample_idx, probe in enumerate(probes):
                    probe.backward(retain_graph=sample_idx < self._fisher_probe_count - 1)
                    suffix_embeds.grad = None
                self._sync_for_profile(device)
                self._profile_values["backward_with_score_seconds"] = time.perf_counter() - phase_started_at

                phase_started_at = time.perf_counter()
                self._compress_cache_from_scores()
                self.detach_cache(cache)
                self._sync_for_profile(device)
                self._profile_values["compression_seconds"] = time.perf_counter() - phase_started_at

            self._sync_for_profile(device)
            self._profile_values["backward_seconds"] = max(
                self._profile_values["backward_with_score_seconds"] - self._profile_values["score_seconds"],
                0.0,
            )
            self.last_profile = {
                **self._profile_values,
                "fisher_positions": self.fisher_positions,
                "fisher_labels": self.fisher_labels,
                "fisher_probe_count": self._fisher_probe_count,
                "score_mode": self.score_mode,
                "coupled_kernel_size": self.coupled_kernel_size,
                "total_seconds": time.perf_counter() - total_started_at,
            }
            if self.profile and device.type == "cuda":
                self.last_profile["peak_memory_bytes"] = torch.cuda.max_memory_allocated(device)
            return outputs
        finally:
            for hook in forward_hooks:
                hook.remove()
            for hook in self._gradient_handles:
                hook.remove()
            for parameter, requires_grad in zip(parameters, parameter_requires_grad):
                parameter.requires_grad_(requires_grad)
            self._states.clear()
            self._quadratic_sums.clear()
            self._gradient_counts.clear()
            self._scores.clear()
            self._gradient_handles.clear()
            self._active = False

    def __call__(self, model: PreTrainedModel):
        """Reject the ordinary one-shot press API, which cannot split the context."""

        raise RuntimeError("LogitKVPress requires press.prefill(model, input_ids, cache); the pipeline handles this")
