"""Tests for 12-attention. Reference: PyTorch (float64) with identical weights copied in.

Tolerances:
  - attention vs F.scaled_dot_product_attention:            1e-12 absolute
  - multi-head attention and per-head weights vs nn.MultiheadAttention:   1e-12 absolute
  - full block vs nn.TransformerEncoderLayer(norm_first=True, GELU-tanh):  1e-12 absolute
  - layer norm vs F.layer_norm, GELU vs F.gelu(approximate="tanh"):        1e-12 absolute
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from model import (
    attention,
    causal_mask,
    gelu,
    init_block,
    layer_norm,
    multi_head_attention,
    sinusoidal_positions,
    transformer_block,
)

B, T, D, H, D_FF = 2, 7, 16, 4, 32


@pytest.fixture
def setup():
    rng = np.random.default_rng(0)
    return rng.normal(size=(B, T, D)), init_block(D, H, D_FF, rng)


def torch_layer(p):
    """nn.TransformerEncoderLayer holding exactly our weights (PyTorch stores W transposed)."""
    layer = torch.nn.TransformerEncoderLayer(
        D, H, D_FF, dropout=0.0, activation=lambda u: F.gelu(u, approximate="tanh"),
        batch_first=True, norm_first=True, dtype=torch.float64)
    t = lambda a: torch.tensor(a, dtype=torch.float64)                      # noqa: E731
    with torch.no_grad():
        sa = layer.self_attn
        sa.in_proj_weight.copy_(t(np.concatenate([p["Wq"].T, p["Wk"].T, p["Wv"].T])))
        sa.in_proj_bias.copy_(t(np.concatenate([p["bq"], p["bk"], p["bv"]])))
        sa.out_proj.weight.copy_(t(p["Wo"].T))
        sa.out_proj.bias.copy_(t(p["bo"]))
        layer.linear1.weight.copy_(t(p["W1"].T))
        layer.linear1.bias.copy_(t(p["b1"]))
        layer.linear2.weight.copy_(t(p["W2"].T))
        layer.linear2.bias.copy_(t(p["b2"]))
        for i, norm in ((1, layer.norm1), (2, layer.norm2)):
            norm.weight.copy_(t(p[f"ln{i}_g"]))
            norm.bias.copy_(t(p[f"ln{i}_b"]))
    return layer.eval()


@pytest.mark.parametrize("causal", [False, True])
def test_attention_matches_pytorch(causal):
    Q, K, V = np.random.default_rng(1).normal(size=(3, B, H, T, 8))
    ours, weights = attention(Q, K, V, causal_mask(T) if causal else None)
    ref = F.scaled_dot_product_attention(*(torch.tensor(a) for a in (Q, K, V)), is_causal=causal)
    np.testing.assert_allclose(ours, ref.numpy(), rtol=0, atol=1e-12)
    np.testing.assert_allclose(weights.sum(-1), 1.0, atol=1e-12)
    if causal:
        assert np.all(weights[..., causal_mask(T)] == 0)


@pytest.mark.parametrize("causal", [False, True])
def test_multi_head_attention_matches_pytorch(setup, causal):
    x, p = setup
    mha = torch_layer(p).self_attn
    mask = causal_mask(T) if causal else None
    ours, weights = multi_head_attention(x, p, H, mask)
    xt = torch.tensor(x)
    ref, ref_w = mha(xt, xt, xt, attn_mask=None if mask is None else torch.tensor(mask),
                     average_attn_weights=False)
    np.testing.assert_allclose(ours, ref.detach().numpy(), rtol=0, atol=1e-12)
    np.testing.assert_allclose(weights, ref_w.detach().numpy(), rtol=0, atol=1e-12)


@pytest.mark.parametrize("causal", [False, True])
def test_block_matches_pytorch_encoder_layer(setup, causal):
    x, p = setup
    ours, _ = transformer_block(x, p, H, causal)
    ref = torch_layer(p)(torch.tensor(x), src_mask=torch.tensor(causal_mask(T)) if causal
                         else None, is_causal=causal)
    np.testing.assert_allclose(ours, ref.detach().numpy(), rtol=0, atol=1e-12)


def test_layer_norm_and_gelu_match_pytorch(setup):
    x, p = setup
    ref = F.layer_norm(torch.tensor(x), (D,), torch.tensor(p["ln1_g"]), torch.tensor(p["ln1_b"]))
    np.testing.assert_allclose(layer_norm(x, p["ln1_g"], p["ln1_b"]), ref.numpy(), atol=1e-12)
    np.testing.assert_allclose(gelu(x), F.gelu(torch.tensor(x), approximate="tanh").numpy(),
                               rtol=0, atol=1e-12)


def test_sinusoidal_positions():
    pe = sinusoidal_positions(50, 8)
    t, i = 13, 2                                     # dimension pair (4, 5)
    assert pe[t, 2 * i] == pytest.approx(np.sin(t / 10000 ** (2 * i / 8)), abs=1e-15)
    assert pe[t, 2 * i + 1] == pytest.approx(np.cos(t / 10000 ** (2 * i / 8)), abs=1e-15)
    np.testing.assert_allclose(pe[:, 0::2] ** 2 + pe[:, 1::2] ** 2, 1.0, atol=1e-12)


def test_block_without_positions_is_permutation_equivariant(setup):
    x, p = setup
    perm = np.random.default_rng(2).permutation(T)
    out, _ = transformer_block(x, p, H)
    out_perm, _ = transformer_block(x[:, perm], p, H)
    np.testing.assert_allclose(out_perm, out[:, perm], rtol=0, atol=1e-12)


def test_readme_examples_run(run_readme):
    assert run_readme(__file__) >= 4
