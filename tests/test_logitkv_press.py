from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM

from kvpress.presses.logitkv_press import (
    LogitKVPress,
    assert_root_squared_ranking,
    fisher_rms_sensitivity,
)
from kvpress.presses.snapkv_press import SnapKVPress


class DummyAttention(nn.Module):
    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        hidden_size = num_heads * head_dim
        self.config = SimpleNamespace(
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            hidden_size=hidden_size,
        )
        self.head_dim = head_dim
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)


def test_fisher_rms_matches_explicit_vwo_reference_with_gqa():
    torch.manual_seed(0)
    batch_size, num_heads, num_kv_heads = 2, 4, 2
    seq_len, grad_len, head_dim, fisher_window = 7, 5, 3, 3
    module = DummyAttention(num_heads, num_kv_heads, head_dim)
    values = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)
    output_grad = torch.randn(batch_size, grad_len, num_heads * head_dim)
    fisher_eps = 1e-9

    actual = fisher_rms_sensitivity(values, output_grad, module, fisher_window, fisher_eps)

    num_groups = num_heads // num_kv_heads
    repeated_values = values.repeat_interleave(num_groups, dim=1)
    projection = module.o_proj.weight.transpose(0, 1).reshape(num_heads, head_dim, -1)
    explicit_vwo = torch.einsum("bhkd,hdm->bhkm", repeated_values, projection)
    explicit_dot = torch.einsum("bhkm,bwm->bhkw", explicit_vwo, output_grad[:, -fisher_window:])
    explicit_q = explicit_dot.float().square().mean(dim=-1)
    explicit_q = explicit_q.reshape(batch_size, num_kv_heads, num_groups, seq_len).mean(dim=2)
    expected = torch.sqrt(explicit_q.clamp_min(0.0) + fisher_eps)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_multiple_fisher_samples_average_quadratics_before_root():
    press = LogitKVPress(
        SnapKVPress(compression_ratio=0.5),
        fisher_window=4,
        fisher_samples=2,
        fisher_eps=1e-12,
        sanity_check=False,
    )
    base_scores = torch.tensor([[[2.0, 3.0]]])
    press._states[0] = SimpleNamespace(
        values=torch.empty(1, 1, 2, 1),
        module=SimpleNamespace(layer_idx=0),
        base_scores=base_scores,
    )
    press._total_length = 2
    press._profile_values = {"score_seconds": 0.0}
    first_quadratic = torch.tensor([[[1.0, 9.0]]])
    second_quadratic = torch.tensor([[[9.0, 1.0]]])
    expected_quadratic = (first_quadratic + second_quadratic) / 2
    hook = press._make_gradient_hook(0)

    with patch(
        "kvpress.presses.logitkv_press.fisher_quadratic_sensitivity",
        side_effect=[first_quadratic, second_quadratic],
    ):
        hook(torch.zeros(1, press.fisher_window, 1))
        assert 0 not in press._scores
        hook(torch.zeros(1, press.fisher_window, 1))

    expected_scores = base_scores * torch.sqrt(expected_quadratic + press.fisher_eps)
    torch.testing.assert_close(press._scores[0], expected_scores)
    assert press._gradient_counts[0] == press.fisher_samples


def test_root_and_squared_scores_have_identical_topk_ranking():
    torch.manual_seed(1)
    attention_scores = torch.rand(2, 3, 19) + 0.01
    fisher_q = torch.rand(2, 3, 19)
    fisher_eps = 1e-12
    k = 8

    assert_root_squared_ranking(attention_scores, fisher_q, k, fisher_eps)

    root_scores = attention_scores * torch.sqrt(fisher_q + fisher_eps)
    squared_scores = attention_scores.square() * (fisher_q + fisher_eps)
    assert torch.equal(root_scores.topk(k).indices, squared_scores.topk(k).indices)

    # At realistic score magnitudes, independent float32 evaluations can
    # reorder nearly tied items while retaining exactly the same Top-K set.
    generator = torch.Generator().manual_seed(434)
    long_attention_scores = torch.rand(1, 1, 512, generator=generator)
    long_fisher_q = torch.rand(1, 1, 512, generator=generator) * 1e-8
    long_k = int(long_attention_scores.shape[-1] * 0.4)
    long_root_scores = long_attention_scores * torch.sqrt(long_fisher_q + fisher_eps)
    long_squared_scores = long_attention_scores.square() * (long_fisher_q + fisher_eps)
    root_indices = long_root_scores.topk(long_k, sorted=True).indices
    squared_indices = long_squared_scores.topk(long_k, sorted=True).indices
    assert not torch.equal(root_indices, squared_indices)
    assert torch.equal(root_indices.sort().values, squared_indices.sort().values)
    assert_root_squared_ranking(long_attention_scores, long_fisher_q, long_k, fisher_eps)

    # A true selection mismatch is logged but must not abort compression; the
    # production path continues with the root-form score.
    close_attention_scores = torch.ones(1, 1, 2)
    close_fisher_q = torch.tensor([[[1.0, 0.99999]]])
    perturbed_root_scores = torch.sqrt(close_fisher_q + fisher_eps)
    perturbed_root_scores[..., 0] -= 6e-6
    perturbed_root_scores[..., 1] += 6e-6
    with patch("kvpress.presses.logitkv_press.logger.warning") as ranking_warning:
        check_passed = assert_root_squared_ranking(
            close_attention_scores,
            close_fisher_q,
            1,
            fisher_eps,
            score_root=perturbed_root_scores,
            layer_idx=7,
        )
    assert check_passed is False
    ranking_warning.assert_called_once()

    try:
        assert_root_squared_ranking(
            attention_scores,
            fisher_q,
            k,
            fisher_eps,
            score_root=attention_scores * fisher_q,
        )
    except RuntimeError as error:
        assert "sqrt" in str(error)
    else:
        raise AssertionError("The sanity check did not detect a missing square root")


def _capture_attention_outputs(model, cache, *, input_ids=None, inputs_embeds=None):
    layer_outputs = {}
    hooks = []

    def capture(module, inputs, kwargs, output):
        layer_outputs[module.layer_idx] = output[0]

    for layer in model.model.layers:
        hooks.append(layer.self_attn.register_forward_hook(capture, with_kwargs=True))
    outputs = model(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
    )
    for hook in hooks:
        hook.remove()
    probe = outputs.logits[:, -1].log_softmax(dim=-1)[:, 7].mean()
    gradients = torch.autograd.grad(probe, [layer_outputs[index] for index in sorted(layer_outputs)])
    return outputs.logits.detach(), gradients


def test_split_forward_logits_and_window_gradients_match_full_graph():
    torch.manual_seed(2)
    total_length, window = 48, 12
    config = LlamaConfig(
        vocab_size=61,
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config).eval()
    model.requires_grad_(False)
    input_ids = torch.randint(0, config.vocab_size, (1, total_length))

    full_embeds = model.get_input_embeddings()(input_ids).detach().requires_grad_(True)
    full_logits, full_gradients = _capture_attention_outputs(
        model,
        DynamicCache(),
        inputs_embeds=full_embeds,
    )

    split_cache = DynamicCache()
    with torch.no_grad():
        model(
            input_ids=input_ids[:, :-window],
            past_key_values=split_cache,
            use_cache=True,
            logits_to_keep=1,
        )
    LogitKVPress.detach_cache(split_cache)
    window_embeds = model.get_input_embeddings()(input_ids[:, -window:]).detach().requires_grad_(True)
    split_logits, split_gradients = _capture_attention_outputs(
        model,
        split_cache,
        inputs_embeds=window_embeds,
    )

    torch.testing.assert_close(split_logits, full_logits, rtol=1e-4, atol=1e-5)
    for full_gradient, split_gradient in zip(full_gradients, split_gradients):
        torch.testing.assert_close(split_gradient, full_gradient[:, -window:], rtol=1e-4, atol=1e-5)


def test_window_aware_snapkv_matches_full_prefill_score():
    torch.manual_seed(3)
    total_length, window = 48, 12
    config = LlamaConfig(
        vocab_size=61,
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (1, total_length))
    scorer = SnapKVPress(compression_ratio=0.5, window_size=window, kernel_size=5)

    def run_and_score(cache, current_ids, from_window):
        scores = []

        def score_hook(module, inputs, kwargs, output):
            cache_layer = cache.layers[module.layer_idx]
            if from_window:
                score = scorer.score_from_window(
                    module,
                    kwargs["hidden_states"],
                    cache_layer.keys,
                    cache_layer.values,
                    output[1],
                    kwargs,
                    window_size=window,
                )
            else:
                score = scorer.score(
                    module,
                    kwargs["hidden_states"],
                    cache_layer.keys,
                    cache_layer.values,
                    output[1],
                    kwargs,
                )
            scores.append(score)

        hook = model.model.layers[0].self_attn.register_forward_hook(score_hook, with_kwargs=True)
        with torch.no_grad():
            model(input_ids=current_ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
        hook.remove()
        return scores[0]

    full_score = run_and_score(DynamicCache(), input_ids, from_window=False)
    split_cache = DynamicCache()
    with torch.no_grad():
        model(
            input_ids=input_ids[:, :-window],
            past_key_values=split_cache,
            use_cache=True,
            logits_to_keep=1,
        )
    window_score = run_and_score(split_cache, input_ids[:, -window:], from_window=True)
    torch.testing.assert_close(window_score, full_score, rtol=1e-5, atol=1e-6)


def test_128_token_prefill_and_compressed_cache_decode_on_cpu():
    torch.manual_seed(4)
    config = LlamaConfig(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
    )
    model = LlamaForCausalLM(config).eval()
    press = LogitKVPress(
        SnapKVPress(compression_ratio=0.5, window_size=32, kernel_size=5),
        fisher_window=32,
        fisher_samples=3,
        first_stage_ratio=0.5,
        fisher_seed=0,
    )

    # The real pipeline runs under no_grad and the efficiency script under
    # inference_mode; LogitKV's context must locally re-enable autograd.
    with torch.inference_mode():
        cache = DynamicCache()
        input_ids = torch.randint(0, config.vocab_size, (1, 128))
        parameter_storages = {parameter.untyped_storage().data_ptr() for parameter in model.parameters()}
        saved_activation_shapes = []

        def record_saved_tensor(tensor):
            if tensor.untyped_storage().data_ptr() not in parameter_storages:
                saved_activation_shapes.append(tuple(tensor.shape))
            return tensor

        with patch("torch.autograd.backward", wraps=torch.autograd.backward) as fisher_backward:
            with torch.autograd.graph.saved_tensors_hooks(record_saved_tensor, lambda tensor: tensor):
                outputs = press.prefill(model, input_ids, cache)

    assert outputs.logits.shape == (1, press.fisher_samples, config.vocab_size)
    assert fisher_backward.call_count == press.fisher_samples
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert all(parameter.grad is None for parameter in model.parameters())
    assert press.last_sampled_token_ids.shape == (1, press.fisher_samples)
    sampled_token_ids = press.last_sampled_token_ids.clone()
    assert press.last_ranking_check_passed is True
    assert press.last_full_cache_length == 128
    assert (1, 128, config.intermediate_size) not in saved_activation_shapes
    assert (1, press.fisher_window, config.intermediate_size) in saved_activation_shapes
    assert set(press.last_profile) >= {
        "prefix_seconds",
        "suffix_forward_seconds",
        "backward_with_score_seconds",
        "backward_seconds",
        "score_seconds",
        "compression_seconds",
        "total_seconds",
    }
    assert press.last_profile["fisher_samples"] == press.fisher_samples
    for layer in cache.layers:
        assert layer.keys.shape == (1, config.num_key_value_heads, 64, config.head_dim)
        assert layer.values.shape == (1, config.num_key_value_heads, 64, config.head_dim)
        assert not layer.keys.requires_grad
        assert not layer.values.requires_grad

    selected_keys = [layer.keys.clone() for layer in cache.layers]
    with torch.inference_mode():
        repeated_cache = DynamicCache()
        press.prefill(model, input_ids, repeated_cache)
    torch.testing.assert_close(press.last_sampled_token_ids, sampled_token_ids)
    for expected_keys, repeated_layer in zip(selected_keys, repeated_cache.layers):
        torch.testing.assert_close(repeated_layer.keys, expected_keys)

    with torch.inference_mode():
        decoded = model(
            input_ids=torch.randint(0, config.vocab_size, (1, 1)),
            past_key_values=cache,
            position_ids=torch.tensor([[128]]),
            use_cache=True,
            logits_to_keep=1,
        )
    assert decoded.logits.shape == (1, 1, config.vocab_size)
    assert cache.get_seq_length() == 65
