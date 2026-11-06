"""Scaled dot-product attention and one pre-norm transformer block (forward pass). NumPy only.

attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
multi-head:  split d_model into h heads, attend in each, concatenate, project with W_o
block (pre-norm, as in GPT-2 and later):
    x = x + MultiHeadAttention(LayerNorm(x))
    x = x + MLP(LayerNorm(x)),    MLP(u) = GELU(u W_1 + b_1) W_2 + b_2
Shapes: x is (batch, T, d_model). Weights are stored so that layers compute x @ W + b.
"""

import numpy as np


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)          # unchanged result, no overflow
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def causal_mask(T):
    """True where position i may NOT look at position j (j in the future of i)."""
    return np.triu(np.ones((T, T), dtype=bool), k=1)


def attention(Q, K, V, mask=None, scale=True):
    """Q: (..., T_q, d_k), K: (..., T_k, d_k), V: (..., T_k, d_v). Returns output and weights."""
    scores = Q @ np.swapaxes(K, -1, -2)              # (..., T_q, T_k): every query . every key
    if scale:
        scores = scores / np.sqrt(Q.shape[-1])
    if mask is not None:
        scores = np.where(mask, -np.inf, scores)     # e^-inf = 0: masked keys get no weight
    weights = softmax(scores, axis=-1)               # each row sums to 1
    return weights @ V, weights                      # each output: weighted average of values


def layer_norm(x, gamma, beta, eps=1e-5):
    """Normalise each token's vector to mean 0, variance 1, then scale and shift."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)              # population variance (ddof = 0)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def gelu(x):
    """GELU, tanh approximation: 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def sinusoidal_positions(T, d):
    """PE[t, 2i] = sin(t / 10000^(2i/d)),  PE[t, 2i+1] = cos(t / 10000^(2i/d))."""
    t = np.arange(T)[:, None]
    freq = 1 / 10000 ** (np.arange(0, d, 2) / d)
    pe = np.zeros((T, d))
    pe[:, 0::2], pe[:, 1::2] = np.sin(t * freq), np.cos(t * freq)
    return pe


def init_block(d_model, n_heads, d_ff, rng):
    assert d_model % n_heads == 0

    def dense(n_in, n_out):
        return rng.normal(0, 1 / np.sqrt(n_in), (n_in, n_out)), rng.normal(0, 0.02, n_out)

    p = {}
    for name in ("q", "k", "v", "o"):
        p["W" + name], p["b" + name] = dense(d_model, d_model)
    p["W1"], p["b1"] = dense(d_model, d_ff)
    p["W2"], p["b2"] = dense(d_ff, d_model)
    for i in (1, 2):
        p[f"ln{i}_g"], p[f"ln{i}_b"] = 1 + rng.normal(0, 0.1, d_model), rng.normal(0, 0.1, d_model)
    return p


def multi_head_attention(x, p, n_heads, mask=None):
    """x: (B, T, d). Returns output (B, T, d) and per-head weights (B, h, T, T)."""
    B, T, d = x.shape
    d_head = d // n_heads

    def split(z):                                    # (B, T, d) -> (B, h, T, d_head)
        return z.reshape(B, T, n_heads, d_head).transpose(0, 2, 1, 3)

    Q, K, V = (split(x @ p["W" + c] + p["b" + c]) for c in "qkv")
    heads, weights = attention(Q, K, V, mask)        # attend within each head separately
    concat = heads.transpose(0, 2, 1, 3).reshape(B, T, d)
    return concat @ p["Wo"] + p["bo"], weights


def transformer_block(x, p, n_heads, causal=False):
    """Pre-norm block: each sub-layer reads a normalised copy and adds its result back."""
    mask = causal_mask(x.shape[1]) if causal else None
    attn_out, weights = multi_head_attention(layer_norm(x, p["ln1_g"], p["ln1_b"]), p, n_heads,
                                             mask)
    x = x + attn_out                                 # residual connection 1
    h = gelu(layer_norm(x, p["ln2_g"], p["ln2_b"]) @ p["W1"] + p["b1"])
    x = x + h @ p["W2"] + p["b2"]                    # residual connection 2
    return x, weights
