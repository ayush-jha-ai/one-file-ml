# 12 · Attention and a transformer block

Let every position in a sequence look at every other position and pull in information in proportion to how relevant it is. Attention is the core operation of transformers, the architecture behind modern language models. This concept builds one transformer block and checks it against PyTorch's.

![Left: six keys placed around a circle and a query arrow that rotates while growing longer. An orange output point, the weighted average of the values, is joined to each key by a line whose thickness shows its weight. With a short query the output stays near the centre; with a long one it snaps to the best-matching key. Right: the six attention weights as bars, going from nearly uniform to one dominant bar.](demo.gif)

## Intuition

Think of a dictionary where lookups are fuzzy. You arrive with a *query*. Every stored item has a *key* (what it is about) and a *value* (what it contains). You compare your query with every key, turn the match scores into weights that add up to 1, and take the weighted average of the values. If one key matches far better than the rest, this is an ordinary lookup; if several match, you get a blend. In a transformer, every token produces its own query, key and value, so each token gathers information from the tokens most relevant to it.

## The maths

**One query.** Given a query $\mathbf{q} \in \mathbb{R}^{d_k}$, keys $\mathbf{k}_1, \dots, \mathbf{k}_T$ and values $\mathbf{v}_1, \dots, \mathbf{v}_T$:

$$s_j = \frac{\mathbf{q}\cdot\mathbf{k}_j}{\sqrt{d_k}}, \qquad w_j = \frac{e^{s_j}}{\sum_{l} e^{s_l}}, \qquad \text{output} = \sum_j w_j\,\mathbf{v}_j.$$

The weights are positive and sum to 1, so the output is a weighted average of the values. It always lies inside the region spanned by them, which is what the orange point in the animation shows.

**All queries at once.** Stack the $T$ token vectors as rows of $X$. Learned matrices produce queries, keys and values: $Q = XW_q$, $K = XW_k$, $V = XW_v$ (plus biases). Then

$$\text{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V,$$

where softmax acts on each row. Entry $(i, j)$ of $QK^\top$ is $\mathbf{q}_i\cdot\mathbf{k}_j$, so row $i$ holds token $i$'s scores against every token.

**Why divide by $\sqrt{d_k}$.** Suppose the entries of $\mathbf{q}$ and $\mathbf{k}$ are independent with mean 0 and variance 1. Each product $q_ik_i$ has mean 0 and variance $\mathbb{E}[q_i^2]\,\mathbb{E}[k_i^2] = 1$. A sum of $d_k$ independent terms has variance equal to the sum of their variances, so $\operatorname{Var}(\mathbf{q}\cdot\mathbf{k}) = d_k$. Without scaling, the scores spread out as $d_k$ grows, and softmax gives almost all the weight to the largest score. The softmax derivative is $\partial w_i/\partial s_j = w_i(\mathbb{1}[i=j] - w_j)$, which is close to 0 when the weights are close to one-hot, so learning stalls. Dividing by $\sqrt{d_k}$ brings the variance back to 1 for any $d_k$ (example 2).

**Causal mask.** When predicting the next token, position $i$ must not see positions $j > i$. Set those scores to $-\infty$ before the softmax. Then $e^{-\infty} = 0$, so future positions get exactly zero weight and the remaining weights still sum to 1.

**Multiple heads.** One attention pattern per token is limiting, since a token may need to look at several others for different reasons. Split the $d$ dimensions of $Q$, $K$ and $V$ into $h$ groups of $d/h$, run attention separately in each group (*head*), concatenate the results, and mix them with a final matrix $W_o$. The total cost is the same as one head of size $d$.

**Layer normalisation.** For each token's vector $\mathbf{x} \in \mathbb{R}^d$, with mean $\mu$ and variance $\sigma^2$ over its $d$ entries:

$$\operatorname{LN}(\mathbf{x}) = \boldsymbol\gamma\odot\frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \boldsymbol\beta.$$

This keeps the inputs to each sub-layer on a fixed scale. ($\epsilon = 10^{-5}$ avoids dividing by zero; $\boldsymbol\gamma$ and $\boldsymbol\beta$ are learned.)

**The block.** In the pre-norm arrangement used by GPT-2 and most later models:

$$X \leftarrow X + \operatorname{MultiHead}(\operatorname{LN}_1(X)), \qquad X \leftarrow X + \operatorname{GELU}(\operatorname{LN}_2(X)W_1 + \mathbf{b}_1)W_2 + \mathbf{b}_2.$$

Attention mixes information *between* tokens. The two-layer network (MLP) then transforms each token *on its own*. Each sub-layer's result is *added* to its input (a *residual connection*). The derivative of $\mathbf{x} + f(\mathbf{x})$ is $I + \partial f/\partial\mathbf{x}$, so gradients always have a direct path back through the identity, which is what lets dozens of blocks be stacked. GELU is $x\Phi(x)$, with $\Phi$ the standard normal CDF. The code uses the common approximation $\tfrac12x\big(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3))\big)$.

**Positions.** Nothing above refers to position. Permute the rows of $X$ and every step (projections, attention, LN, MLP) produces the same rows, permuted the same way (example 1). Order must be supplied explicitly, for example by adding sinusoidal position vectors:

$$PE_{t,2i} = \sin\!\big(t/10000^{2i/d}\big), \qquad PE_{t,2i+1} = \cos\!\big(t/10000^{2i/d}\big).$$

**Cost.** $QK^\top$ is a $T\times T$ matrix for every head, so time grows as $T^2d$ and memory as $T^2$ per head. Doubling the context length quadruples both.

## Code walkthrough

| Equation | Code | Lines |
|---|---|---|
| softmax with the max subtracted | `z - z.max(axis=axis, keepdims=True)` | [15](model.py#L15) |
| causal mask: $j > i$ | `np.triu(np.ones((T, T), dtype=bool), k=1)` | [22](model.py#L22) |
| $QK^\top$ | `Q @ np.swapaxes(K, -1, -2)` | [27](model.py#L27) |
| $/\sqrt{d_k}$ | `scores / np.sqrt(Q.shape[-1])` | [29](model.py#L29) |
| masked scores $\to -\infty$ | `np.where(mask, -np.inf, scores)` | [31](model.py#L31) |
| row-wise softmax, then $\times V$ | `softmax(scores, axis=-1)`, `weights @ V` | [32-33](model.py#L32-L33) |
| $\operatorname{LN}(\mathbf{x}) = \boldsymbol\gamma\odot(\mathbf{x}-\mu)/\sqrt{\sigma^2+\epsilon} + \boldsymbol\beta$ | `gamma * (x - mean) / np.sqrt(var + eps) + beta` | [38-40](model.py#L38-L40) |
| GELU (tanh form) | `0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))` | [45](model.py#L45) |
| $1/10000^{2i/d}$, then sin and cos | `1 / 10000 ** (np.arange(0, d, 2) / d)` | [51-53](model.py#L51-L53) |
| split $d$ into $h$ heads of $d/h$ | `z.reshape(B, T, n_heads, d_head).transpose(0, 2, 1, 3)` | [79](model.py#L79) |
| $Q = XW_q + \mathbf{b}_q$ (and $K$, $V$) | `x @ p["W" + c] + p["b" + c] for c in "qkv"` | [81](model.py#L81) |
| concatenate heads, then $W_o$ | `heads.transpose(0, 2, 1, 3).reshape(B, T, d)`, `concat @ p["Wo"] + p["bo"]` | [83-84](model.py#L83-L84) |
| $X \leftarrow X + \operatorname{MultiHead}(\operatorname{LN}_1(X))$ | `x = x + attn_out` | [92](model.py#L92) |
| $X \leftarrow X + \operatorname{GELU}(\operatorname{LN}_2(X)W_1 + \mathbf{b}_1)W_2 + \mathbf{b}_2$ | `gelu(layer_norm(x, ...) @ p["W1"] + p["b1"])`, `x + h @ p["W2"] + p["b2"]` | [93-94](model.py#L93-L94) |

The tests copy identical weights into PyTorch (float64) and require agreement to `1e-12`:
- attention against `F.scaled_dot_product_attention`, with and without the causal mask;
- multi-head attention against `nn.MultiheadAttention`, including every head's weights;
- the full block against `nn.TransformerEncoderLayer(norm_first=True)` with GELU;
- layer norm and GELU against PyTorch's own.

They also check the position-vector formula and permutation equivariance.

## Where it breaks

Run these from inside this folder.

**1. Without positions, word order is invisible.** Reverse the input tokens and the block's outputs come back exactly reversed: the model computes the same thing for "dog bites man" and "man bites dog", just listed in a different order. Adding position vectors breaks the symmetry.

```python
import numpy as np
from model import init_block, sinusoidal_positions, transformer_block

rng = np.random.default_rng(0)
params = init_block(16, 4, 32, rng)
x = rng.normal(size=(1, 6, 16))
out, _ = transformer_block(x, params, n_heads=4)
out_rev, _ = transformer_block(x[:, ::-1], params, n_heads=4)
print(np.abs(out_rev - out[:, ::-1]).max())           # 5.6e-16: same rows, reversed

pe = sinusoidal_positions(6, 16)
out_pe, _ = transformer_block(x + pe, params, n_heads=4)
out_pe_rev, _ = transformer_block(x[:, ::-1] + pe, params, n_heads=4)
print(np.abs(out_pe_rev - out_pe[:, ::-1]).max())     # 1.30: order now matters
assert np.allclose(out_rev, out[:, ::-1], atol=1e-12) and not np.allclose(out_pe_rev,
                                                                          out_pe[:, ::-1])
```

**2. Without the $\sqrt{d_k}$, attention collapses as dimensions grow.** For random queries and 10 random keys, the average largest weight stays near 0.31 with scaling, at every size. Without scaling it climbs to 0.86 at $d_k = 64$ and 0.95 at $d_k = 1024$: nearly one-hot, where softmax gradients vanish.

```python
import numpy as np
from model import attention

for d_k in (4, 64, 1024):
    rng = np.random.default_rng(1)
    Q, K = rng.normal(size=(500, 1, d_k)), rng.normal(size=(500, 10, d_k))
    V = np.broadcast_to(np.eye(10), (500, 10, 10))
    top = [attention(Q, K, V, scale=s)[1].max(axis=-1).mean() for s in (True, False)]
    print(d_k, np.round(top, 3))     # 4: [0.298 0.484]  64: [0.317 0.855]  1024: [0.313 0.95]
assert top[0] < 0.35 and top[1] > 0.9
```

**3. Attention cannot count.** Its output is a weighted *average*, and repeating every item in the context does not change any average. One copy each of tokens `a` and `b` gives exactly the same output as three copies of each. Telling "a few" from "many" needs something extra, such as position information or a dedicated token to attend to.

```python
import numpy as np
from model import attention

a, b = np.array([1.0, 0, 0, 0]), np.array([0, 1.0, 0, 0])
query = np.array([[0.3, 0.3, 0, 0]])
once = np.stack([a, b])
thrice = np.stack([a, a, a, b, b, b])
out_once, _ = attention(query, once, once)
out_thrice, _ = attention(query, thrice, thrice)
print(out_once, out_thrice)              # [[0.5 0.5 0. 0.]] [[0.5 0.5 0. 0.]]
assert np.allclose(out_once, out_thrice, atol=1e-15)
```

**4. Forgetting the causal mask leaks the future.** Change only the last token and, without the mask, the outputs at all earlier positions change too (by up to 0.68). A next-token model trained like this can simply read the answer. With the mask, earlier outputs are exactly unchanged.

```python
import numpy as np
from model import init_block, transformer_block

rng = np.random.default_rng(0)
params = init_block(16, 4, 32, rng)
x = rng.normal(size=(1, 6, 16))
x_changed = x.copy()
x_changed[0, 5] = rng.normal(size=16)                 # replace only the last token
for causal in (False, True):
    a, _ = transformer_block(x, params, 4, causal=causal)
    b, _ = transformer_block(x_changed, params, 4, causal=causal)
    print(causal, np.abs(a[0, :5] - b[0, :5]).max())  # False: 0.68   True: 0.0
assert np.abs(a[0, :5] - b[0, :5]).max() == 0.0
```

A subtle trap when testing this: adding the same constant to every feature of a token changes nothing at all. Layer normalisation subtracts each token's mean, so that change disappears before attention sees it.

## Common mistakes

- **"Attention weights explain the model's decision."** They show where information was gathered from in one head of one layer. The values, the other heads, the MLPs and the residual stream all change what that information becomes. High weight is not the same as high influence.
- **"Queries, keys and values are different inputs."** In self-attention they are three different learned projections of the *same* token vectors. Only in cross-attention (for example, a decoder reading an encoder's output) do keys and values come from a different sequence.
- **"Multi-head attention costs $h$ times as much."** Each head works in $d/h$ dimensions, so the total is the same as one full-width head. What grows is the number of separate attention patterns.
- **"The transformer knows word order."** Only through the position information added to it (example 1). Remove that and the block treats the input as an unordered set.
- **"Long contexts are only slower."** The $T\times T$ score matrix also has to be stored. A context of 100,000 tokens needs $10^{10}$ scores per head per layer, which is why long-context models use memory-saving attention algorithms.

## Further reading

- Bahdanau, Cho and Bengio, "Neural machine translation by jointly learning to align and translate", *ICLR* (2015): attention in sequence models.
- Vaswani et al., "Attention is all you need", *NeurIPS* (2017): the transformer, scaled dot-product attention, multi-head attention and sinusoidal positions.
- Ba, Kiros and Hinton, "Layer normalization", arXiv:1607.06450 (2016).
- Hendrycks and Gimpel, "Gaussian error linear units (GELUs)", arXiv:1606.08415 (2016).
- Xiong et al., "On layer normalization in the transformer architecture", *ICML* (2020): why the pre-norm arrangement trains more stably.
- Radford et al., "Language models are unsupervised multitask learners", OpenAI technical report (2019): GPT-2, which uses the pre-norm block.
