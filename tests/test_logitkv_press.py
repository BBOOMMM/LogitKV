from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM

from kvpress.presses.logitkv_press import (
    LogitKVPress,
    assert_root_squared_ranking,
    coupled_fisher_quadratic_sensitivity,
    fisher_rms_sensitivity,
    value_projection_l2_quadratic,
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
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
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


def test_fisher_rms_ignores_zero_gradient_positions():
    torch.manual_seed(1)
    module = DummyAttention(num_heads=1, num_kv_heads=1, head_dim=3)
    values = torch.randn(1, 1, 4, 3)
    output_grad = torch.zeros(1, 4, 3)
    output_grad[:, 1] = torch.tensor([1.0, -2.0, 0.5])

    actual = fisher_rms_sensitivity(values, output_grad, module, fisher_window=4)
    expected = fisher_rms_sensitivity(values, output_grad[:, 1:2], module, fisher_window=4)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_value_projection_l2_quadratic_matches_explicit_reference_with_gqa():
    torch.manual_seed(14)
    module = DummyAttention(num_heads=4, num_kv_heads=2, head_dim=3)
    values = torch.randn(2, 2, 5, 3)
    projection = module.o_proj.weight.transpose(0, 1).reshape(4, 3, -1)
    repeated_values = values.repeat_interleave(2, dim=1)
    explicit = torch.einsum("bhkd,hdm->bhkm", repeated_values, projection).square().sum(dim=-1)
    expected = explicit.reshape(2, 2, 2, 5).mean(dim=2)

    torch.testing.assert_close(value_projection_l2_quadratic(values, module), expected, rtol=1e-5, atol=1e-6)


def test_coupled_fisher_modes_match_explicit_attention_reference_with_gqa():
    torch.manual_seed(11)
    batch_size, num_heads, num_kv_heads = 2, 4, 2
    total_length, window, head_dim = 7, 3, 2
    module = DummyAttention(num_heads, num_kv_heads, head_dim)
    keys = torch.randn(batch_size, num_kv_heads, total_length, head_dim)
    values = torch.randn_like(keys)
    hidden_states = torch.randn(batch_size, window, num_heads * head_dim)
    output_grad = torch.randn_like(hidden_states)
    position_embeddings = (
        torch.ones(batch_size, window, head_dim),
        torch.zeros(batch_size, window, head_dim),
    )

    query_states = module.q_proj(hidden_states).view(batch_size, window, num_heads, head_dim).transpose(1, 2)
    repeated_keys = keys.repeat_interleave(num_heads // num_kv_heads, dim=1)
    attention_logits = torch.einsum("bhwd,bhkd->bhwk", query_states, repeated_keys) / head_dim**0.5
    key_positions = torch.arange(total_length)
    query_positions = total_length - window + torch.arange(window)
    causal_mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
    attention_logits.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    attention = torch.softmax(attention_logits, dim=-1)

    repeated_values = values.repeat_interleave(num_heads // num_kv_heads, dim=1)
    projection = module.o_proj.weight.transpose(0, 1).reshape(num_heads, head_dim, -1)
    explicit_vwo = torch.einsum("bhkd,hdm->bhkm", repeated_values, projection)
    dot = torch.einsum("bhkm,bwm->bhkw", explicit_vwo, output_grad)
    contribution = attention.transpose(-1, -2) * dot
    contribution = contribution.reshape(
        batch_size,
        num_kv_heads,
        num_heads // num_kv_heads,
        total_length,
        window,
    ).sum(dim=2)

    expected = {
        "coupled_diag": contribution.square().sum(dim=-1),
        "coupled_full": contribution.sum(dim=-1).square(),
    }
    for mode, expected_quadratic in expected.items():
        actual = coupled_fisher_quadratic_sensitivity(
            keys,
            values,
            hidden_states,
            position_embeddings,
            output_grad,
            module,
            mode,
            window,
        )
        torch.testing.assert_close(actual, expected_quadratic, rtol=1e-5, atol=1e-6)


def test_multiple_fisher_labels_average_quadratics_before_squared_score():
    press = LogitKVPress(
        SnapKVPress(compression_ratio=0.5),
        fisher_window=4,
        fisher_positions=1,
        fisher_labels=2,
        attention_eps=0.25,
        fisher_eps=1e-12,
        sanity_check=False,
    )
    base_scores = torch.tensor([[[0.0, 3.0]]])
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

    expected_scores = (base_scores + press.attention_eps).square() * (expected_quadratic + press.fisher_eps)
    torch.testing.assert_close(press._scores[0], expected_scores)
    assert press._gradient_counts[0] == press._fisher_probe_count


def test_fisher_position_max_averages_labels_then_preserves_each_positions_peak():
    press = LogitKVPress(
        SnapKVPress(compression_ratio=0.5),
        fisher_window=4,
        fisher_positions=2,
        fisher_labels=2,
        fisher_position_aggregation="max",
        sanity_check=False,
    )
    press._states[0] = SimpleNamespace(
        values=torch.empty(1, 1, 2, 1),
        module=SimpleNamespace(layer_idx=0),
        base_scores=torch.ones(1, 1, 2),
    )
    press._total_length = 2
    press._profile_values = {"score_seconds": 0.0}
    quadratics = [
        torch.tensor([[[1.0, 9.0]]]),
        torch.tensor([[[3.0, 5.0]]]),
        torch.tensor([[[10.0, 1.0]]]),
        torch.tensor([[[6.0, 3.0]]]),
    ]

    with patch(
        "kvpress.presses.logitkv_press.fisher_quadratic_sensitivity",
        side_effect=quadratics,
    ):
        hook = press._make_gradient_hook(0)
        for _ in quadratics:
            hook(torch.zeros(1, press.fisher_window, 1))

    # Position means are [2, 7] and [8, 2], hence position-max is [8, 7].
    expected_scores = torch.tensor([[[8.0, 7.0]]]) + press.fisher_eps
    torch.testing.assert_close(press._scores[0], expected_scores)


def test_coupled_modes_rank_by_average_quadratic_without_attention_or_root():
    base_scores = torch.tensor([[[100.0, 0.01]]])
    first_quadratic = torch.tensor([[[1.0, 9.0]]])
    second_quadratic = torch.tensor([[[9.0, 1.0]]])
    expected_quadratic = (first_quadratic + second_quadratic) / 2

    for mode in ("coupled_diag", "coupled_full"):
        press = LogitKVPress(
            SnapKVPress(compression_ratio=0.5),
            fisher_window=4,
            fisher_positions=1,
            fisher_labels=2,
            score_mode=mode,
            fisher_eps=1e-12,
            sanity_check=True,
        )
        press._states[0] = SimpleNamespace(
            keys=torch.empty(1, 1, 2, 1),
            values=torch.empty(1, 1, 2, 1),
            hidden_states=torch.empty(1, press.fisher_window, 1),
            position_embeddings=(torch.empty(1), torch.empty(1)),
            module=SimpleNamespace(layer_idx=0),
            base_scores=base_scores,
        )
        press._total_length = 2
        press._profile_values = {"score_seconds": 0.0}
        hook = press._make_gradient_hook(0)

        with (
            patch(
                "kvpress.presses.logitkv_press.coupled_fisher_quadratic_sensitivity",
                side_effect=[first_quadratic.clone(), second_quadratic.clone()],
            ),
            patch("kvpress.presses.logitkv_press.assert_root_squared_ranking") as ranking_check,
        ):
            hook(torch.zeros(1, press.fisher_window, 1))
            hook(torch.zeros(1, press.fisher_window, 1))

        torch.testing.assert_close(press._scores[0], expected_quadratic, rtol=0, atol=0)
        ranking_check.assert_not_called()
        assert press.last_ranking_check_passed is True


def test_adaptive_allocation_sets_headwise_mask_without_criticalkv_state():
    press = LogitKVPress(
        SnapKVPress(compression_ratio=0.5),
        allocation_mode="adaptive",
        first_stage_ratio=0.0,
        alpha_safeguard=0.0,
    )
    module = SimpleNamespace()
    state = SimpleNamespace(
        module=module,
        base_scores=torch.ones(1, 2, 4),
    )
    scores = torch.tensor(
        [[[10.0, 9.0, 8.0, 7.0], [1.0, 2.0, 3.0, 4.0]]]
    )

    press._set_adaptive_mask(state, scores, n_kept=2)

    batch_indices, head_indices, seq_indices = module.masked_key_indices
    assert batch_indices.numel() == 4
    assert torch.equal(head_indices, torch.ones(4, dtype=torch.long))
    assert torch.equal(seq_indices, torch.arange(4))


def test_coupled_kernel_pooling_spreads_quadratic_scores_to_neighbors():
    press = LogitKVPress(
        SnapKVPress(compression_ratio=0.4),
        fisher_window=4,
        fisher_positions=1,
        fisher_labels=1,
        score_mode="coupled_diag",
        coupled_kernel_size=3,
        sanity_check=False,
    )
    press._states[0] = SimpleNamespace(
        keys=torch.empty(1, 1, 5, 1),
        values=torch.empty(1, 1, 5, 1),
        hidden_states=torch.empty(1, press.fisher_window, 1),
        position_embeddings=(torch.empty(1), torch.empty(1)),
        module=SimpleNamespace(layer_idx=0),
        base_scores=torch.zeros(1, 1, 5),
    )
    press._total_length = 5
    press._profile_values = {"score_seconds": 0.0}
    quadratic = torch.tensor([[[0.0, 0.0, 9.0, 0.0, 0.0]]])

    with patch(
        "kvpress.presses.logitkv_press.coupled_fisher_quadratic_sensitivity",
        return_value=quadratic,
    ):
        press._make_gradient_hook(0)(torch.zeros(1, press.fisher_window, 1))

    expected = torch.tensor([[[0.0, 3.0, 3.0, 3.0, 0.0]]])
    torch.testing.assert_close(press._scores[0], expected, rtol=0, atol=0)


def test_coupled_max_pooling_propagates_peak_without_averaging_it_down():
    press = LogitKVPress(
        SnapKVPress(compression_ratio=0.4),
        fisher_window=4,
        score_mode="coupled_diag",
        coupled_kernel_size=3,
        coupled_pooling="max",
        sanity_check=False,
    )
    press._states[0] = SimpleNamespace(
        keys=torch.empty(1, 1, 5, 1),
        values=torch.empty(1, 1, 5, 1),
        hidden_states=torch.empty(1, press.fisher_window, 1),
        position_embeddings=(torch.empty(1), torch.empty(1)),
        module=SimpleNamespace(layer_idx=0),
        base_scores=torch.zeros(1, 1, 5),
    )
    press._total_length = 5
    press._profile_values = {"score_seconds": 0.0}
    quadratic = torch.tensor([[[0.0, 0.0, 9.0, 0.0, 0.0]]])

    with patch(
        "kvpress.presses.logitkv_press.coupled_fisher_quadratic_sensitivity",
        return_value=quadratic,
    ):
        press._make_gradient_hook(0)(torch.zeros(1, press.fisher_window, 1))

    expected = torch.tensor([[[0.0, 9.0, 9.0, 9.0, 0.0]]])
    torch.testing.assert_close(press._scores[0], expected, rtol=0, atol=0)


def test_coupled_attention_prior_softly_reweights_downstream_scores():
    press = LogitKVPress(
        SnapKVPress(compression_ratio=0.4),
        fisher_window=4,
        score_mode="coupled_diag",
        coupled_attention_power=1.0,
        sanity_check=False,
    )
    press._states[0] = SimpleNamespace(
        keys=torch.empty(1, 1, 3, 1),
        values=torch.empty(1, 1, 3, 1),
        hidden_states=torch.empty(1, press.fisher_window, 1),
        position_embeddings=(torch.empty(1), torch.empty(1)),
        module=SimpleNamespace(layer_idx=0),
        base_scores=torch.tensor([[[1.0, 2.0, 3.0]]]),
    )
    press._total_length = 3
    press._profile_values = {"score_seconds": 0.0}
    quadratic = torch.tensor([[[1.0, 4.0, 1.0]]])

    with patch(
        "kvpress.presses.logitkv_press.coupled_fisher_quadratic_sensitivity",
        return_value=quadratic,
    ):
        press._make_gradient_hook(0)(torch.zeros(1, press.fisher_window, 1))

    torch.testing.assert_close(press._scores[0], torch.tensor([[[0.5, 4.0, 1.5]]]))


def test_score_mode_validation_rejects_invalid_mode_and_coupled_attention_epsilon():
    try:
        LogitKVPress(SnapKVPress(), fisher_position_aggregation="median")
    except ValueError as error:
        assert "fisher_position_aggregation" in str(error)
    else:
        raise AssertionError("Invalid Fisher position aggregation was accepted")

    try:
        LogitKVPress(SnapKVPress(), fisher_backward_mode="batched")
    except ValueError as error:
        assert "fisher_backward_mode" in str(error)
    else:
        raise AssertionError("Invalid Fisher backward mode was accepted")

    try:
        LogitKVPress(SnapKVPress(), fisher_sketches=0)
    except ValueError as error:
        assert "fisher_sketches" in str(error)
    else:
        raise AssertionError("Non-positive Fisher sketch count was accepted")

    for invalid_label_count in (0, -2):
        try:
            LogitKVPress(SnapKVPress(), fisher_labels=invalid_label_count)
        except ValueError as error:
            assert "fisher_labels" in str(error)
        else:
            raise AssertionError("Invalid Fisher label count was accepted")

    try:
        LogitKVPress(SnapKVPress(), fisher_labels=-1, fisher_backward_mode="serial")
    except ValueError as error:
        assert "label_sketch" in str(error)
    else:
        raise AssertionError("All-label Fisher mode accepted serial backward")

    try:
        LogitKVPress(SnapKVPress(), score_mode="coupled_diag", coupled_pooling="median")
    except ValueError as error:
        assert "coupled_pooling" in str(error)
    else:
        raise AssertionError("Invalid coupled pooling mode was accepted")

    try:
        LogitKVPress(SnapKVPress(), score_mode="unknown")
    except ValueError as error:
        assert "score_mode" in str(error)
    else:
        raise AssertionError("Invalid score mode was accepted")

    try:
        LogitKVPress(SnapKVPress(), score_mode="coupled_diag", coupled_key_weight=-1)
    except ValueError as error:
        assert "coupled_key_weight" in str(error)
    else:
        raise AssertionError("Negative coupled key-path weight was accepted")

    try:
        LogitKVPress(SnapKVPress(), score_mode="coupled_diag", attention_eps=1e-4)
    except ValueError as error:
        assert "attention_eps" in str(error)
    else:
        raise AssertionError("Coupled score mode accepted a nonzero attention_eps")

    for invalid_kernel_size in (0, 2):
        try:
            LogitKVPress(
                SnapKVPress(),
                score_mode="coupled_diag",
                coupled_kernel_size=invalid_kernel_size,
            )
        except ValueError as error:
            assert "coupled_kernel_size" in str(error)
        else:
            raise AssertionError("Invalid coupled kernel size was accepted")

    try:
        LogitKVPress(SnapKVPress(), score_mode="separable", coupled_kernel_size=3)
    except ValueError as error:
        assert "only applies" in str(error)
    else:
        raise AssertionError("Separable mode accepted coupled pooling")


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
    # production path continues with the squared-form score.
    close_attention_scores = torch.ones(1, 1, 2)
    close_fisher_q = torch.tensor([[[1.0, 0.99999]]])
    perturbed_squared_scores = close_fisher_q + fisher_eps
    perturbed_squared_scores[..., 0] -= 6e-6
    perturbed_squared_scores[..., 1] += 6e-6
    with patch("kvpress.presses.logitkv_press.logger.warning") as ranking_warning:
        check_passed = assert_root_squared_ranking(
            close_attention_scores,
            close_fisher_q,
            1,
            fisher_eps,
            score_squared=perturbed_squared_scores,
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
            score_squared=attention_scores * fisher_q,
        )
    except RuntimeError as error:
        assert "squared score" in str(error)
    else:
        raise AssertionError("The sanity check did not detect an invalid squared score")


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


def test_coupled_score_modes_complete_split_prefill_and_cache_compression_on_cpu():
    torch.manual_seed(12)
    config = LlamaConfig(
        vocab_size=41,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    model = LlamaForCausalLM(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 32))

    for mode in ("coupled_diag", "coupled_full"):
        cache = DynamicCache()
        press = LogitKVPress(
            SnapKVPress(compression_ratio=0.5, window_size=8, kernel_size=5),
            fisher_window=8,
            fisher_positions=1,
            fisher_labels=1,
            score_mode=mode,
            fisher_seed=0,
        )
        with torch.inference_mode():
            outputs = press.prefill(model, input_ids, cache)

        assert outputs.logits.shape == (1, 1, config.vocab_size)
        assert cache.get_seq_length() == 16
        assert press.last_profile["score_mode"] == mode
        assert press.last_profile["separable_score_form"] == "squared"
        assert press.last_profile["fisher_eps"] == press.fisher_eps
        assert press.last_profile["coupled_kernel_size"] == 1
        assert press.last_ranking_check_passed is True


def test_fisher_and_snapkv_windows_are_independent():
    torch.manual_seed(13)
    config = LlamaConfig(
        vocab_size=41,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    model = LlamaForCausalLM(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 32))

    for fisher_window, snapkv_window, expected_prefill_window in ((4, 12, 12), (12, 4, 12)):
        cache = DynamicCache()
        press = LogitKVPress(
            SnapKVPress(compression_ratio=0.5, window_size=snapkv_window, kernel_size=5),
            fisher_window=fisher_window,
            fisher_positions=1,
            fisher_labels=1,
            fisher_seed=0,
        )

        with patch.object(
            press.press,
            "score_from_window",
            wraps=press.press.score_from_window,
        ) as score_from_window:
            with torch.inference_mode():
                press.prefill(model, input_ids, cache)

        assert cache.get_seq_length() == 16
        assert score_from_window.call_args.kwargs["window_size"] == snapkv_window
        assert press.last_profile["fisher_window"] == fisher_window
        assert press.last_profile["base_score_window"] == snapkv_window
        assert press.last_profile["prefill_window"] == expected_prefill_window


def test_top_fisherposition_selects_highest_labels_per_position():
    press = LogitKVPress(
        SnapKVPress(),
        fisher_window=2,
        fisher_positions=2,
        fisher_labels=2,
        fisherlabel_samplemode="top_fisherposition",
    )
    logits = torch.tensor(
        [[[1.0, 5.0, 2.0, 4.0], [8.0, 3.0, 7.0, 0.0]]],
        requires_grad=True,
    )

    probes = press._sample_log_probabilities(logits)

    torch.testing.assert_close(
        press.last_sampled_token_ids,
        torch.tensor([[[1, 3], [0, 2]]]),
    )
    assert len(probes) == 4


def test_label_sketch_reduces_backward_probes_and_reuses_fisher_seed():
    logits = torch.empty(1, 2, 4)
    label_objectives = torch.arange(6.0, requires_grad=True)

    def make_probes():
        press = LogitKVPress(
            SnapKVPress(),
            fisher_window=2,
            fisher_positions=2,
            fisher_labels=3,
            fisherlabel_samplemode="top_fisherposition",
            fisher_backward_mode="label_sketch",
            fisher_sketches=2,
            fisher_seed=17,
        )
        with patch.object(
            press,
            "_sample_log_probabilities",
            return_value=tuple(label_objectives.unbind()),
        ):
            probes = press._make_fisher_probes(logits)
        return press, probes

    first_press, first_probes = make_probes()
    second_press, second_probes = make_probes()

    assert first_press._fisher_probe_count == 4
    assert len(first_probes) == first_press.fisher_positions * first_press.fisher_sketches
    assert len(first_probes) < first_press.fisher_positions * first_press.fisher_labels
    torch.testing.assert_close(torch.stack(first_probes), torch.stack(second_probes))

    for probe in first_probes:
        gradient = torch.autograd.grad(probe, label_objectives, retain_graph=True)[0]
        nonzero = gradient[gradient != 0]
        assert nonzero.numel() > 0
        expected_magnitude = torch.full_like(nonzero, 1 / first_press.fisher_labels**0.5)
        torch.testing.assert_close(nonzero.abs(), expected_magnitude)


def test_all_label_sketch_has_exact_categorical_fisher_covariance_in_expectation():
    logits = torch.tensor([[[0.3, -0.2]]], requires_grad=True)
    press = LogitKVPress(
        SnapKVPress(),
        fisher_window=1,
        fisher_positions=1,
        fisher_labels=-1,
        fisher_backward_mode="label_sketch",
        fisher_sketches=4,
        fisher_seed=5,
    )
    sign_bits = [
        torch.tensor([[0, 0]], dtype=torch.int8),
        torch.tensor([[0, 1]], dtype=torch.int8),
        torch.tensor([[1, 0]], dtype=torch.int8),
        torch.tensor([[1, 1]], dtype=torch.int8),
    ]

    with patch("torch.randint", side_effect=sign_bits):
        probes = press._make_fisher_probes(logits)

    gradients = []
    for probe in probes:
        gradients.append(torch.autograd.grad(probe, logits, retain_graph=True)[0].flatten())
    estimated_fisher = torch.stack([gradient.outer(gradient) for gradient in gradients]).mean(dim=0)

    probabilities = torch.softmax(logits.detach().flatten(), dim=0)
    expected_fisher = torch.diag(probabilities) - probabilities.outer(probabilities)
    torch.testing.assert_close(estimated_fisher, expected_fisher, rtol=1e-5, atol=1e-6)
    assert press.last_sampled_token_ids is None


def test_label_sketch_prefill_uses_positions_times_sketches_backwards():
    torch.manual_seed(14)
    config = LlamaConfig(
        vocab_size=41,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    model = LlamaForCausalLM(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 32))
    press = LogitKVPress(
        SnapKVPress(compression_ratio=0.5, window_size=8, kernel_size=5),
        fisher_window=8,
        fisher_positions=2,
        fisher_labels=3,
        fisher_backward_mode="label_sketch",
        fisher_sketches=2,
        fisher_seed=17,
    )

    with patch("torch.autograd.backward", wraps=torch.autograd.backward) as backward:
        with torch.inference_mode():
            press.prefill(model, input_ids, DynamicCache())

    assert backward.call_count == 4
    assert backward.call_count == press.fisher_positions * press.fisher_sketches
    assert press.last_profile["fisher_backward_mode"] == "label_sketch"
    assert press.last_profile["fisher_sketches"] == 2
    assert press.last_profile["fisher_label_probe_count"] == 6
    assert press.last_profile["fisher_probe_count"] == 4


def test_all_label_sketch_prefill_uses_full_vocabulary_without_extra_backwards():
    torch.manual_seed(15)
    config = LlamaConfig(
        vocab_size=41,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    model = LlamaForCausalLM(config).eval()
    press = LogitKVPress(
        SnapKVPress(compression_ratio=0.5, window_size=8, kernel_size=5),
        fisher_window=8,
        fisher_positions=2,
        fisher_labels=-1,
        fisher_backward_mode="label_sketch",
        fisher_sketches=2,
        fisher_seed=19,
    )

    with patch("torch.autograd.backward", wraps=torch.autograd.backward) as backward:
        with torch.inference_mode():
            press.prefill(
                model,
                torch.randint(0, config.vocab_size, (1, 32)),
                DynamicCache(),
            )

    assert backward.call_count == 4
    assert press.last_sampled_token_ids is None
    assert press.last_profile["fisher_labels_per_position_used"] == config.vocab_size
    assert press.last_profile["fisherlabel_samplemode_effective"] == "all"
    assert press.last_profile["fisher_label_probe_count"] == 2 * config.vocab_size
    assert press.last_profile["fisher_probe_count"] == 4


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
        fisher_positions=3,
        fisher_labels=2,
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

    assert outputs.logits.shape == (1, press.fisher_positions, config.vocab_size)
    assert fisher_backward.call_count == press._fisher_probe_count
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert all(parameter.grad is None for parameter in model.parameters())
    assert press.last_sampled_token_ids.shape == (1, press.fisher_positions, press.fisher_labels)
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
    assert press.last_profile["fisher_positions"] == press.fisher_positions
    assert press.last_profile["fisher_labels"] == press.fisher_labels
    assert press.last_profile["fisher_probe_count"] == press._fisher_probe_count
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
