# 05 · k-nearest neighbours

Predict a new point's label by looking up the most similar examples you have already seen. There is no training step, and the method makes almost no assumptions about the shape of the data, which makes it the baseline every other method should beat.

![Left: three classes of points, with the background coloured by the class k-NN would predict. At k = 1 the regions are jagged islands around single points; as k grows to 250 they smooth into three large regions. Right: training error starts at zero and rises, while leave-one-out error falls, flattens and rises again.](demo.gif)

## Intuition

To classify a new point, find the k training points closest to it and let them vote. With k = 1 you copy the label of the single nearest point. The model reproduces the training set perfectly, but it also copies every mislabelled or unusual point, so its boundaries are jagged. With larger k, each prediction averages over more neighbours. Boundaries become smooth and stable, but if k is too large the neighbourhood reaches into other classes and the majority class takes over. Choosing k is choosing how local "similar" means. Everything depends on the distance measure, so features must be on comparable scales.

## The maths

**Distance.** Training data are $(\mathbf{x}_i, y_i)$ for $i = 1, \dots, n$ with $\mathbf{x}_i \in \mathbb{R}^d$. The Euclidean distance between a query $\mathbf{q}$ and $\mathbf{x}_i$ is

$$\lVert\mathbf{q} - \mathbf{x}_i\rVert = \sqrt{\sum_{j=1}^d (q_j - x_{ij})^2}.$$

**Neighbourhood.** Sort the training points by distance to $\mathbf{q}$ and let $N_k(\mathbf{q})$ be the indices of the first $k$. Only the order matters, and squaring preserves order for non-negative numbers, so we can rank by squared distance and never take the square root.

**Prediction.** For classification with classes $c = 1, \dots, C$, estimate the probability of each class as the fraction of neighbours that have it, then predict the most common:

$$\hat{P}(y = c \mid \mathbf{q}) = \frac{1}{k}\sum_{i \in N_k(\mathbf{q})} \mathbb{1}[y_i = c], \qquad \hat{y} = \arg\max_c \hat{P}(y = c \mid \mathbf{q}).$$

$\mathbb{1}[\cdot]$ is 1 when the condition holds and 0 otherwise. For regression, average the targets: $\hat{y} = \tfrac1k\sum_{i\in N_k(\mathbf{q})} y_i$.

**All distances with one matrix product.** For $m$ queries and $n$ training points, looping over all pairs is slow in Python. Expand the square:

$$\lVert\mathbf{q} - \mathbf{x}\rVert^2 = (\mathbf{q} - \mathbf{x})^\top(\mathbf{q} - \mathbf{x}) = \lVert\mathbf{q}\rVert^2 + \lVert\mathbf{x}\rVert^2 - 2\,\mathbf{q}^\top\mathbf{x}.$$

Stacking queries as rows of $Q$ and training points as rows of $X$, the cross terms for every pair form the single product $QX^\top$. Rounding can make the result very slightly negative when $\mathbf{q} \approx \mathbf{x}$, so it is clipped at 0.

**Why it works, and the cost.** As $n$ grows with $k$ growing too, but much more slowly ($k \to \infty$, $k/n \to 0$), the neighbours of $\mathbf{q}$ crowd ever closer to it. Their labels then become a sample from $P(y \mid \mathbf{q})$ itself, so $\hat{P}$ converges to the true class probabilities. Even 1-NN has a guarantee. Cover and Hart (1967) proved that, as $n \to \infty$, its error rate is at most twice the lowest error any classifier could achieve (the *Bayes error*). The cost is at prediction time: each query computes $n$ distances in $d$ dimensions, which is $O(nd)$ work and requires keeping the whole training set.

**The curse of dimensionality.** How local is a neighbourhood? Take points spread uniformly in the unit cube $[0, 1]^d$. A smaller cube with edge length $\ell$ has volume $\ell^d$, so it contains a fraction $\ell^d$ of the data. To capture a fraction $r$ we need

$$\ell = r^{1/d}.$$

With $d = 2$, the nearest 1% of the data lies in a square of edge $0.01^{1/2} = 0.1$, which is genuinely local. With $d = 10$, it needs edge $0.01^{1/10} \approx 0.63$, covering most of the range of every feature. "Nearest" stops meaning "similar". Example 1 below shows the consequence.

## Code walkthrough

| Equation | Code | Lines |
|---|---|---|
| $\lVert\mathbf{q}\rVert^2 + \lVert\mathbf{x}\rVert^2 - 2\mathbf{q}^\top\mathbf{x}$ for all pairs | `(A**2).sum(axis=1)[:, None] + (B**2).sum(axis=1)[None, :] - 2 * A @ B.T` | [18](model.py#L18) |
| clip rounding error at 0 | `np.maximum(sq, 0.0)` | [19](model.py#L19) |
| labels as $0, \dots, C-1$ | `np.unique(self.y, return_inverse=True)` | [34](model.py#L34) |
| leave-one-out: a point is not its own neighbour | `np.fill_diagonal(d2, np.inf)` | [46](model.py#L46) |
| $N_k(\mathbf{q})$: first $k$ after sorting by distance | `np.argsort(d2, axis=1, kind="stable")[:, : self.k]` | [47](model.py#L47) |
| distances, only for the $k$ returned | `np.sqrt(np.take_along_axis(d2, idx, axis=1))` | [48](model.py#L48) |
| $\hat{P}(y = c \mid \mathbf{q}) = \frac1k\sum_{i\in N_k}\mathbb{1}[y_i = c]$ | `np.bincount(row, minlength=n_classes)` then `counts / self.k` | [53-56](model.py#L53-L56) |
| regression: $\frac1k\sum_{i\in N_k} y_i$ | `self.y[idx].mean(axis=1)` | [61](model.py#L61) |
| $\hat{y} = \arg\max_c \hat{P}$ | `np.argmax(self.predict_proba(...), axis=1)` | [63](model.py#L63) |

Two tie rules are defined explicitly, because they change the answer:
- **Equal distances:** the stable sort keeps training-set order.
- **Equal votes:** `argmax` returns the first maximum, so the smallest label wins.

scikit-learn's brute-force search makes the same choices. The tests confirm identical predictions, vote fractions and neighbour indices for k = 1, 4, 7 and 15, distances to `1e-10`, regression to `1e-10`, and leave-one-out error identical to scikit-learn's `cross_val_score` with `LeaveOneOut`.

## Where it breaks

**1. High dimensions: everything is equally far away.** For 1,000 uniform points, the gap between the farthest and the nearest point, relative to the nearest, is 135 in 2 dimensions and 0.11 in 1,000. With so little contrast, the "nearest" neighbours are barely nearer than random points. In practice the damage comes from irrelevant features. A two-feature problem that k-NN solves at 95% accuracy drops to 65% when 10 pure-noise features are appended, and to 53% with 50 of them.

```python
import numpy as np
from model import leave_one_out_error, pairwise_sq_distances

rng = np.random.default_rng(0)
contrast = {}
for d in (2, 10, 100, 1000):
    X, q = rng.uniform(size=(1000, d)), rng.uniform(size=(1, d))
    dist = np.sqrt(pairwise_sq_distances(q, X))[0]
    contrast[d] = (dist.max() - dist.min()) / dist.min()
print(contrast)                        # {2: 134.6, 10: 3.48, 100: 0.39, 1000: 0.11}

rng = np.random.default_rng(0)
X_useful = rng.normal(size=(400, 2))
y = (X_useful[:, 0] * X_useful[:, 1] > 0).astype(int)        # label: same-sign quadrants
acc = {}
for n_noise in (0, 10, 50):
    X = np.column_stack([X_useful, rng.normal(size=(400, n_noise))])
    acc[n_noise] = 1 - leave_one_out_error(X, y, k=5)
print(acc)                             # {0: 0.95, 10: 0.6475, 50: 0.535}
assert contrast[1000] < 0.2 < 100 < contrast[2] and acc[0] > 0.9 and acc[50] < 0.6
```

**2. Units decide the answer.** Distance adds raw squared differences, so a feature measured in thousands drowns out one measured in units. Here the label depends only on feature 0 (spread about 1), and feature 1 is irrelevant noise with spread about 1,000. Unscaled, k-NN does worse than a coin toss. Dividing each feature by its standard deviation gives 98%.

```python
import numpy as np
from model import leave_one_out_error

rng = np.random.default_rng(0)
a, b = rng.normal(size=400), 1000 * rng.normal(size=400)
y = (a > 0).astype(int)                                      # only feature a matters
X = np.column_stack([a, b])
raw = 1 - leave_one_out_error(X, y, k=5)
scaled = 1 - leave_one_out_error(X / X.std(axis=0), y, k=5)
print(raw, scaled)                                           # 0.485  0.98
assert raw < 0.55 and scaled > 0.95
```

**3. No extrapolation.** A k-NN prediction is an average of training targets, so it can never leave their range. Trained on $y = 2x$ for $x \in [0, 1]$, it predicts 1.92 at $x = 2$ and also 1.92 at $x = 10$, where the truth is 20.

```python
import numpy as np
from model import KNN

x = np.linspace(0, 1, 50)[:, None]
model = KNN(5, task="regress").fit(x, 2 * x[:, 0])
print(model.predict(np.array([[2.0], [10.0]])))              # [1.918 1.918]
assert np.allclose(model.predict(np.array([[2.0], [10.0]])), 1.918, atol=1e-3)
```

**4. Large k with a rare class: the minority disappears.** With 10% positives, a neighbourhood of 101 points almost always has a negative majority, so k-NN never predicts positive. Accuracy is 90.6%, exactly the share of negatives, and recall is zero. Small k finds 79% of positives.

```python
import numpy as np
from model import KNN

rng = np.random.default_rng(0)
y = (rng.random(500) < 0.1).astype(int)
X = rng.normal(size=(500, 2)) + 2.0 * y[:, None]             # positives shifted by 2 std
for k in (5, 101):
    pred = KNN(k).fit(X, y).predict(X, exclude_self=True)
    print(k, (pred == y).mean(), pred[y == 1].mean())        # 5: 0.964 0.79   101: 0.906 0.0
assert pred.sum() == 0 and np.isclose((pred == y).mean(), 1 - y.mean())
```

## Common mistakes

- **"k = 1 is perfect: zero training error."** Each training point is its own nearest neighbour, so training error at k = 1 is always 0 (unless identical points have different labels). It says nothing about new data. Measure error on held-out points, or with leave-one-out as the animation does.
- **"k-NN has no parameters, so it cannot overfit."** It stores all $n$ training points, which act as its parameters, and k controls how closely it follows them. At k = 1 it overfits as hard as any model can.
- **"Use an odd k to avoid ties."** That only works for two classes. With three or more, an odd k can still give tied votes (for example 2-2-1 with k = 5), so you need an explicit tie rule like the one in `predict`.
- **"Scaling features is optional cleaning."** It changes the model. Every choice of scaling defines a different distance, so it is as much a modelling decision as k (example 2).
- **"k-NN is fast because it does no training."** The work moves to prediction: $O(nd)$ per query. Tree and hashing indexes help in low dimensions, but in high dimensions they lose most of their advantage, for the same reason as example 1.

## Further reading

- Fix and Hodges, "Discriminatory analysis. Nonparametric discrimination: consistency properties", USAF School of Aviation Medicine technical report (1951): the first description of the nearest-neighbour rule.
- Cover and Hart, "Nearest neighbor pattern classification", *IEEE Transactions on Information Theory* 13(1) (1967): the bound of twice the Bayes error.
- Hastie, Tibshirani and Friedman, *The Elements of Statistical Learning*, 2nd edition (2009), sections 2.3 and 2.5, and chapter 13: k-NN and the curse of dimensionality.
- Beyer, Goldstein, Ramakrishnan and Shaft, "When is 'nearest neighbor' meaningful?", *International Conference on Database Theory* (1999): distance concentration in high dimensions.
