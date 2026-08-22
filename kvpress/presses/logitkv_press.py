# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LogitKV: downstream-Fisher-aware KV-cache compression."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator

import torch
from torch import nn
from transformers import DynamicCache, PreTrainedModel, QuantizedCache

from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress

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


def assert_root_squared_ranking(
    base_scores: torch.Tensor,
    fisher_quadratic: torch.Tensor,
    k: int,
    fisher_eps: float = 0.0,
    score_root: torch.Tensor | None = None,
) -> None:
    """Check that ``A * sqrt(Q)`` and ``A^2 * Q`` produce the same top-k ranking."""

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
    root_ranking = score_root.topk(k, dim=-1, sorted=True).indices
    squared_ranking = score_squared.topk(k, dim=-1, sorted=True).indices
    if not torch.equal(root_ranking, squared_ranking):
        mismatch_count = (root_ranking != squared_ranking).sum().item()
        raise RuntimeError(
            "LogitKV root/squared ranking sanity check failed: "
            f"{mismatch_count} top-k positions differ"
        )


@dataclass
class _LayerCapture:
    module: nn.Module
    cache: DynamicCache
    keys: torch.Tensor
    values: torch.Tensor
    base_scores: torch.Tensor
    layer_output: torch.Tensor


@dataclass
class LogitKVPress(BasePress):
    """CriticalKV-style two-stage compression using downstream Fisher sensitivity.

    Unlike ordinary :class:`ScorerPress` implementations, LogitKV cannot compress
    a layer as soon as that layer finishes: its score depends on the final model
    logits. Attention hooks therefore capture the base score and layer output,
    and a model-level hook performs one empirical-Fisher backward followed by
    in-place cache compression after the full-cache prefill completes.
    """

    press: ScorerPress
    fisher_window: int = 32
    first_stage_ratio: float = 0.5
    fisher_eps: float = 1e-12
    sanity_check: bool = True
    fisher_seed: int | None = None

    _captures: dict[int, _LayerCapture] = field(default_factory=dict, init=False, repr=False)
    _active: bool = field(default=False, init=False, repr=False)
    _completed: bool = field(default=False, init=False, repr=False)
    last_sampled_token_ids: torch.Tensor | None = field(default=None, init=False, repr=False)
    last_ranking_check_passed: bool | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.press, ScorerPress):
            raise TypeError("LogitKVPress requires a ScorerPress as input")
        if self.fisher_window <= 0:
            raise ValueError("fisher_window must be positive")
        if not 0 <= self.first_stage_ratio <= 1:
            raise ValueError("first_stage_ratio must be between 0 and 1")
        if self.fisher_eps < 0:
            raise ValueError("fisher_eps must be non-negative")

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
            raise NotImplementedError(
                f"LogitKV currently requires DynamicCache, got {cache.__class__.__name__}"
            )
        if layer_idx >= len(cache.layers):
            raise RuntimeError(f"Cache layer {layer_idx} is not initialized")
        cache_layer = cache.layers[layer_idx]
        if not getattr(cache_layer, "is_initialized", False):
            raise RuntimeError(f"Cache layer {layer_idx} is not initialized")
        return cache_layer.keys, cache_layer.values

    def _capture_attention(self, module: nn.Module, inputs: tuple, kwargs: dict, output: tuple):
        if self._completed:
            return output

        layer_idx = module.layer_idx
        if layer_idx in self._captures:
            raise RuntimeError(f"Attention layer {layer_idx} ran more than once during one LogitKV prefill")
        if not isinstance(output, (tuple, list)) or len(output) < 1:
            raise RuntimeError("LogitKV expected the attention module to return a tuple")

        hidden_states = kwargs["hidden_states"]
        cache = kwargs.get("past_key_values", kwargs.get("past_key_value"))
        if cache is None:
            raise ValueError("LogitKV requires use_cache=True and an explicit DynamicCache")
        keys, values = self._cache_tensors(cache, layer_idx)
        if keys.shape[2] != hidden_states.shape[1]:
            raise ValueError(
                "LogitKV requires a full-cache prefill into an empty cache; "
                f"layer {layer_idx} saw {hidden_states.shape[1]} queries and {keys.shape[2]} cached keys"
            )

        attentions = output[1] if len(output) > 1 else None
        # The prefill ran with autograd enabled, but the decoding cache itself
        # must not keep the forward graph alive after Fisher gradients are done.
        keys = keys.detach()
        values = values.detach()
        cache.layers[layer_idx].keys = keys
        cache.layers[layer_idx].values = values
        with torch.no_grad():
            base_scores = self.press.score(
                module,
                hidden_states.detach(),
                keys,
                values,
                attentions,
                kwargs,
            ).float()

        layer_output = output[0]
        if not layer_output.requires_grad:
            layer_output.requires_grad_(True)

        expected_shape = keys.shape[:3]
        if base_scores.shape != expected_shape:
            raise ValueError(
                f"Base scorer returned {tuple(base_scores.shape)} for cache shape {tuple(expected_shape)}"
            )
        if not torch.isfinite(base_scores).all():
            raise FloatingPointError(f"Base scorer produced non-finite values at layer {layer_idx}")
        if (base_scores < 0).any():
            raise ValueError("LogitKV requires a non-negative attention-based scorer")

        self._captures[layer_idx] = _LayerCapture(
            module=module,
            cache=cache,
            keys=keys,
            values=values,
            base_scores=base_scores,
            layer_output=layer_output,
        )
        return output

    @staticmethod
    def _logits_from_output(output) -> torch.Tensor:
        if hasattr(output, "logits"):
            return output.logits
        if isinstance(output, (tuple, list)) and output:
            return output[0]
        raise RuntimeError("LogitKV could not find final logits in the model output")

    def _sample_log_probability(self, logits: torch.Tensor) -> torch.Tensor:
        final_logits = logits[:, -1, :].float()
        if not final_logits.requires_grad:
            raise RuntimeError(
                "Final logits do not require gradients. Run LogitKV inside its context manager "
                "and do not wrap the model call in an un-overridable inference context."
            )
        probabilities = torch.softmax(final_logits.detach(), dim=-1)
        if not torch.isfinite(probabilities).all():
            raise FloatingPointError("Final-token probabilities contain non-finite values")

        generator = None
        if self.fisher_seed is not None:
            generator = torch.Generator(device=probabilities.device)
            generator.manual_seed(self.fisher_seed)
        sampled = torch.multinomial(probabilities, num_samples=1, generator=generator)
        self.last_sampled_token_ids = sampled.detach().cpu()
        return torch.log_softmax(final_logits, dim=-1).gather(-1, sampled).sum()

    @staticmethod
    def _prepare_model_inputs(model: nn.Module, inputs: tuple, kwargs: dict) -> tuple[tuple, dict]:
        """Clone inference tensors so autograd is allowed to save them for backward."""

        def make_normal(tensor):
            if isinstance(tensor, torch.Tensor) and torch.is_inference(tensor):
                return tensor.clone()
            return tensor

        inputs = tuple(make_normal(value) for value in inputs)
        kwargs = {name: make_normal(value) for name, value in kwargs.items()}
        return inputs, kwargs

    def _compress_capture(self, capture: _LayerCapture, output_grad: torch.Tensor) -> None:
        keys = capture.keys.detach()
        values = capture.values.detach()
        base_scores = capture.base_scores
        q_len = keys.shape[2]
        n_kept = int(q_len * (1 - self.compression_ratio))

        fisher_quadratic = fisher_quadratic_sensitivity(
            values=values,
            output_grad=output_grad.detach(),
            module=capture.module,
            fisher_window=self.fisher_window,
        )
        fisher_rms = torch.sqrt(fisher_quadratic.clamp_min(0.0) + self.fisher_eps)
        scores = base_scores * fisher_rms
        if not torch.isfinite(scores).all():
            raise FloatingPointError(f"LogitKV produced non-finite scores at layer {capture.module.layer_idx}")

        if self.sanity_check:
            assert_root_squared_ranking(
                base_scores,
                fisher_quadratic,
                n_kept,
                self.fisher_eps,
                score_root=scores,
            )
            self.last_ranking_check_passed = True

        # CriticalKV Stage 1: safeguard a fixed fraction of the base scorer's
        # top tokens before ranking the remaining budget with LogitKV scores.
        first_stage_budget = int((1 - self.compression_ratio) * q_len * self.first_stage_ratio)
        if first_stage_budget:
            top_base_indices = base_scores.topk(first_stage_budget, dim=-1, sorted=True).indices
            scores.scatter_(-1, top_base_indices, torch.finfo(scores.dtype).max)

        kept_indices = scores.topk(n_kept, dim=-1).indices
        gather_indices = kept_indices.unsqueeze(-1).expand(-1, -1, -1, capture.module.head_dim)
        compressed_keys = keys.gather(2, gather_indices).contiguous()
        compressed_values = values.gather(2, gather_indices).contiguous()

        cache_layer = capture.cache.layers[capture.module.layer_idx]
        cache_layer.keys = compressed_keys
        cache_layer.values = compressed_values

    def _finish_prefill(self, model: nn.Module, inputs: tuple, kwargs: dict, output):
        if self._completed:
            return output
        if not self._captures:
            raise RuntimeError("LogitKV did not capture any attention layers")
        expected_layer_count = len(model.model.layers)
        if len(self._captures) != expected_layer_count:
            raise RuntimeError(
                f"LogitKV captured {len(self._captures)} of {expected_layer_count} attention layers"
            )

        captures = [self._captures[layer_idx] for layer_idx in sorted(self._captures)]
        logits = self._logits_from_output(output)
        sampled_log_probability = self._sample_log_probability(logits)
        layer_outputs = [capture.layer_output for capture in captures]

        gradients = torch.autograd.grad(
            sampled_log_probability,
            layer_outputs,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )
        for capture, output_grad in zip(captures, gradients):
            self._compress_capture(capture, output_grad)

        self._captures.clear()
        self._completed = True
        return output

    @contextmanager
    def __call__(self, model: PreTrainedModel) -> Generator:
        """Capture a full prefill, run one Fisher backward, and compress its cache."""

        if self.compression_ratio == 0:
            yield
            return
        if self._active:
            raise RuntimeError("A LogitKVPress instance cannot be active in two contexts at once")
        if not hasattr(model, "model") or not hasattr(model.model, "layers"):
            raise NotImplementedError("LogitKV expects a model with model.layers[*].self_attn")

        hooks = []
        parameters = list(model.parameters())
        parameter_requires_grad = [parameter.requires_grad for parameter in parameters]
        self._captures.clear()
        self._active = True
        self._completed = False
        self.last_sampled_token_ids = None
        self.last_ranking_check_passed = None
        try:
            # Parameter gradients are not part of LogitKV's empirical Fisher.
            # Freezing them avoids retaining the graph before the first captured
            # attention output; all original flags are restored below.
            for parameter in parameters:
                parameter.requires_grad_(False)
            hooks.append(model.register_forward_pre_hook(self._prepare_model_inputs, with_kwargs=True))
            for layer in model.model.layers:
                hooks.append(layer.self_attn.register_forward_hook(self._capture_attention, with_kwargs=True))
            hooks.append(model.register_forward_hook(self._finish_prefill, with_kwargs=True))

            # Pipeline.forward uses no_grad and the efficiency evaluator uses
            # inference_mode. Explicitly disabling both here is required for the
            # single empirical-Fisher backward.
            with torch.inference_mode(False), torch.enable_grad():
                yield
        finally:
            for hook in hooks:
                hook.remove()
            for parameter, requires_grad in zip(parameters, parameter_requires_grad):
                parameter.requires_grad_(requires_grad)
            self._captures.clear()
            self._active = False
