# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LogitKV: downstream-Fisher-aware KV-cache compression."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import torch
from torch import nn
from transformers import DynamicCache, PreTrainedModel, QuantizedCache

from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress

logger = logging.getLogger(__name__)


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
            "continuing with A * sqrt(Q + fisher_eps)",
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


@dataclass
class LogitKVPress(BasePress):
    """Memory-efficient online LogitKV with downstream Fisher sensitivity.

    The context is split into a detached no-grad prefix and a differentiable
    trailing Fisher window. Each attention-output gradient hook immediately
    computes its layer's LogitKV score, after which the full cache is compressed
    with CriticalKV's Stage-1 safeguard and per-head Top-K rule.
    """

    press: ScorerPress
    fisher_window: int = 32
    first_stage_ratio: float = 0.5
    fisher_eps: float = 1e-12
    sanity_check: bool = True
    fisher_seed: int | None = None
    profile: bool = False

    _states: dict[int, _LayerState] = field(default_factory=dict, init=False, repr=False)
    _scores: dict[int, torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    _active: bool = field(default=False, init=False, repr=False)
    _total_length: int = field(default=0, init=False, repr=False)
    _gradient_handles: list = field(default_factory=list, init=False, repr=False)
    _profile_values: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    last_sampled_token_ids: torch.Tensor | None = field(default=None, init=False, repr=False)
    last_ranking_check_passed: bool | None = field(default=None, init=False, repr=False)
    last_full_cache_length: int | None = field(default=None, init=False, repr=False)
    last_profile: dict[str, float | int] = field(default_factory=dict, init=False, repr=False)

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

        layer_output = output[0]
        if not layer_output.requires_grad:
            raise RuntimeError("Window attention output is not differentiable")

        expected_shape = keys.shape[:3]
        if base_scores.shape != expected_shape:
            raise ValueError(
                f"Base scorer returned {tuple(base_scores.shape)} for cache shape {tuple(expected_shape)}"
            )
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
        return torch.log_softmax(final_logits, dim=-1).gather(-1, sampled).mean()

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
                fisher_quadratic = fisher_quadratic_sensitivity(
                    values=state.values,
                    output_grad=output_grad,
                    module=state.module,
                    fisher_window=self.fisher_window,
                )
                fisher_rms = torch.sqrt(fisher_quadratic.clamp_min(0.0) + self.fisher_eps)
                scores = state.base_scores * fisher_rms
                if not torch.isfinite(scores).all():
                    raise FloatingPointError(f"LogitKV produced non-finite scores at layer {layer_idx}")

                n_kept = int(self._total_length * (1 - self.compression_ratio))
                if self.sanity_check:
                    layer_ranking_check_passed = assert_root_squared_ranking(
                        state.base_scores,
                        fisher_quadratic,
                        n_kept,
                        self.fisher_eps,
                        score_root=scores,
                        layer_idx=layer_idx,
                    )
                    if self.last_ranking_check_passed is None:
                        self.last_ranking_check_passed = layer_ranking_check_passed
                    else:
                        self.last_ranking_check_passed &= layer_ranking_check_passed
                self._scores[layer_idx] = scores.detach()

            self._sync_for_profile(output_grad.device)
            self._profile_values["score_seconds"] += time.perf_counter() - started_at
            return output_grad

        return gradient_hook

    def _compress_cache_from_scores(self) -> None:
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

        input_ids = self._normal_tensor(input_ids)
        attention_mask = self._normal_tensor(attention_mask)
        position_ids = self._normal_tensor(position_ids)
        total_length = input_ids.shape[1]
        if total_length <= self.fisher_window:
            raise ValueError(
                f"Context length {total_length} must be greater than fisher_window {self.fisher_window}"
            )
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
                        logits_to_keep=1,
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

                probe = self._sample_log_probability(self._logits_from_output(outputs))
                self._sync_for_profile(device)
                phase_started_at = time.perf_counter()
                probe.backward()
                self._sync_for_profile(device)
                self._profile_values["backward_with_score_seconds"] = time.perf_counter() - phase_started_at
                suffix_embeds.grad = None

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
            self._scores.clear()
            self._gradient_handles.clear()
            self._active = False

    def __call__(self, model: PreTrainedModel):
        """Reject the ordinary one-shot press API, which cannot split the context."""

        raise RuntimeError("LogitKVPress requires press.prefill(model, input_ids, cache); the pipeline handles this")
