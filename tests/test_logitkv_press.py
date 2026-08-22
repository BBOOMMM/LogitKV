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


def test_128_token_prefill_and_compressed_cache_decode_on_cpu():
    torch.manual_seed(2)
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
        first_stage_ratio=0.5,
        fisher_seed=0,
    )

    # The real pipeline runs under no_grad and the efficiency script under
    # inference_mode; LogitKV's context must locally re-enable autograd.
    with torch.inference_mode():
        cache = DynamicCache()
        input_ids = torch.randint(0, config.vocab_size, (1, 128))
        with patch("torch.autograd.grad", wraps=torch.autograd.grad) as fisher_backward:
            with press(model):
                outputs = model(
                    input_ids=input_ids,
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )

    assert outputs.logits.shape == (1, 1, config.vocab_size)
    assert fisher_backward.call_count == 1
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert all(parameter.grad is None for parameter in model.parameters())
    assert press.last_sampled_token_ids.shape == (1, 1)
    assert press.last_ranking_check_passed is True
    for layer in cache.layers:
        assert layer.keys.shape == (1, config.num_key_value_heads, 64, config.head_dim)
        assert layer.values.shape == (1, config.num_key_value_heads, 64, config.head_dim)
        assert not layer.keys.requires_grad
        assert not layer.values.requires_grad

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
