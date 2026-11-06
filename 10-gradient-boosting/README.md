# 10 · Gradient boosting

Build a strong model by adding many small trees one at a time, each trained to correct the mistakes of the ones before it. Gradient-boosted trees are the method of choice for tabular data and win most machine-learning competitions on it.

![Left: noisy points from a wave, and the ensemble's prediction, a step function that starts flat and gains detail every round. Right: the residuals (data minus prediction) for the current round, and the small four-leaf tree fitted to them. The residuals shrink towards zero as rounds pass; by round 100 the ensemble is chasing individual noisy points.](demo.gif)

## Intuition

Start with a constant guess, such as the average. Look at what is still wrong: the *residuals*. Fit a small tree to the residuals and add a fraction of it to the model, so the model moves towards the data where it was too low or too high. Repeat. Each tree is weak on its own, but every one targets what is left over. Random forests reduce error by averaging independent trees; boosting reduces it by stacking dependent ones. For losses other than squared error, "what is still wrong" becomes "which direction lowers the loss", which is the gradient.

## The maths

**The model.** After $M$ rounds the prediction is a sum:

$$F_M(\mathbf{x}) = F_0 + \nu\sum_{m=1}^{M} h_m(\mathbf{x}),$$

where $F_0$ is a constant, each $h_m$ is a small regression tree, and $0 < \nu \le 1$ is the *learning rate* (or *shrinkage*).

**Gradient descent on the predictions.** Let $\ell(y, F)$ be the loss for one example and $L = \sum_i \ell(y_i, F(\mathbf{x}_i))$. Treat the $n$ numbers $F(\mathbf{x}_1), \dots, F(\mathbf{x}_n)$ as if they were the parameters. Gradient descent would move each one downhill, by an amount proportional to the *pseudo-residual*

$$r_i = -\frac{\partial \ell(y_i, F)}{\partial F}\bigg|_{F = F_{m-1}(\mathbf{x}_i)}.$$

But a list of $n$ numbers says nothing about new points. So we fit a regression tree to the pairs $(\mathbf{x}_i, r_i)$. The tree is a function that approximates the downhill direction and can be evaluated anywhere. That is the *gradient* in gradient boosting.

**Leaf values.** The tree's leaves split the data into regions $R_1, \dots, R_J$. In each region, choose the constant $\gamma_j$ that best lowers the loss there, then update

$$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \nu\,\gamma_{j(\mathbf{x})},$$

where $j(\mathbf{x})$ is the leaf that $\mathbf{x}$ falls into.

**Squared error.** $\ell = \tfrac12(y - F)^2$, so $-\partial\ell/\partial F = y - F$, the ordinary residual. In leaf $R_j$, minimise $\sum_{i\in R_j}\tfrac12(y_i - F_i - \gamma)^2$. Setting the derivative $-\sum(y_i - F_i - \gamma)$ to zero gives the mean residual in the leaf:

$$\gamma_j = \frac{1}{|R_j|}\sum_{i\in R_j} r_i.$$

The best constant to start from is $F_0 = \bar{y}$.

**Log loss (binary classification).** With $y \in \{0, 1\}$ and $F$ the log-odds, $p = \sigma(F) = 1/(1 + e^{-F})$, the loss from concept 02 is $\ell = \log(1 + e^F) - yF$. Its first two derivatives are

$$\frac{\partial\ell}{\partial F} = \sigma(F) - y = p - y, \qquad \frac{\partial^2\ell}{\partial F^2} = \sigma(F)\big(1 - \sigma(F)\big) = p(1 - p).$$

So $r_i = y_i - p_i$. The leaf value has no closed form, so take one Newton step. Expand each loss to second order around $\gamma = 0$:

$$\sum_{i\in R_j}\ell(y_i, F_i + \gamma) \approx \sum_{i\in R_j}\Big[\ell(y_i, F_i) - r_i\gamma + \tfrac12 p_i(1-p_i)\gamma^2\Big].$$

Setting the derivative with respect to $\gamma$ to zero gives

$$\gamma_j = \frac{\sum_{i\in R_j} r_i}{\sum_{i\in R_j} p_i(1 - p_i)}.$$

The best constant to start from solves $\sum_i(\sigma(F_0) - y_i) = 0$, so $\sigma(F_0) = \bar{y}$ and $F_0 = \log\frac{\bar{y}}{1 - \bar{y}}$.

**Shrinkage.** Each tree is fitted to the residuals greedily and is only an approximation, so taking the full step tends to overshoot and fit noise. Scaling every step by $\nu$ (0.1 is typical) makes each round count for less, so more rounds are needed but the result is usually better on new data (example 2).

**The splits.** Each tree is a regression tree on $r$, chosen to minimise the children's total squared error. The squared error of $m$ values is $\sum r^2 - (\sum r)^2/m$ (derived in concept 08), so cumulative sums give every candidate split in one pass. scikit-learn's default criterion, `friedman_mse`, maximises $\frac{n_L n_R}{n}(\bar{r}_L - \bar{r}_R)^2$. That is algebraically the same reduction in squared error, so both choose the same split.

## Code walkthrough

| Equation | Code | Lines |
|---|---|---|
| $\sigma(z) = \tfrac12\big(1 + \tanh(z/2)\big)$, identical to $1/(1+e^{-z})$ without overflow | `0.5 * (1 + np.tanh(0.5 * z))` | [18](model.py#L18) |
| children's total squared error, $\sum r^2 - (\sum r)^2/m$ per side | `sse = (s2 - s**2 / n_left) + (...)` | [29](model.py#L29) |
| $F_0 = \bar{y}$ (squared error) | `y.mean()` | [76](model.py#L76) |
| $F_0 = \log\frac{\bar{y}}{1-\bar{y}}$ (log loss) | `np.log(y.mean() / (1 - y.mean()))` | [78](model.py#L78) |
| $r_i = y_i - F_i$ or $y_i - p_i$ | `r = y - (p if p is not None else F)` | [83](model.py#L83) |
| fit a tree to $(\mathbf{x}_i, r_i)$ | `grow(X, r, np.arange(len(y)), 0, ...)` | [85](model.py#L85) |
| $\gamma_j = $ mean residual in the leaf | `r[rows].mean()` | [89](model.py#L89) |
| $\gamma_j = \sum r_i / \sum p_i(1-p_i)$ | `hess = np.sum(p[rows] * (1 - p[rows]))`, `r[rows].sum() / hess` | [91-92](model.py#L91-L92) |
| $F_m = F_{m-1} + \nu\gamma_{j(\mathbf{x})}$ | `F + self.learning_rate * gamma[apply(tree, X)]` | [93](model.py#L93) |
| classify: $F > 0 \iff p > 0.5$ | `(F > 0).astype(int)` | [111](model.py#L111) |

The tests check:
- **Regression:** against scikit-learn's `GradientBoostingRegressor` at *every one of 100 rounds* (to `1e-10`), with minimum leaf sizes 5, 10 and 20.
- **Classification:** `GradientBoostingClassifier` checked the same way, including probabilities.
- **Ties:** where small leaves let scikit-learn break an exact tie differently, every differing node is checked to have exactly equal squared error.
- **Pseudo-residuals:** both losses checked against $-\partial\ell/\partial F$ from PyTorch autograd (to `1e-12`).

## Where it breaks

**1. Too many rounds fit the noise.** Boosting keeps reducing training error for as long as you let it. On noisy moons with $\nu = 1$, accuracy on new data peaks after the first round (85.8%) and falls to 80.2% by round 500, while training accuracy reaches 100%. The number of rounds must be chosen on held-out data (early stopping).

```python
import numpy as np
from model import GradientBoosting
from sklearn.datasets import make_moons

X, y = make_moons(1000, noise=0.4, random_state=0)
model = GradientBoosting("log_loss", 500, learning_rate=1.0, max_depth=3).fit(X[:400], y[:400])
acc = [np.mean((F > 0) == y[400:]) for F in model.staged_decision_function(X[400:])]
train_acc = np.mean(model.predict(X[:400]) == y[:400])
print(int(np.argmax(acc)) + 1, max(acc), acc[-1], train_acc)     # 1 0.858 0.802 1.0
assert train_acc == 1.0 and max(acc) - acc[-1] > 0.05
```

**2. Learning rate and rounds trade off.** A smaller learning rate takes smaller steps, so each round overfits less and the decline is slower. On the same data, $\nu = 0.1$ keeps 84.8% after 50 rounds, where $\nu = 1$ has already dropped to 81.8%, and 82.3% after 500 rounds. The usual recipe is a small $\nu$ and early stopping.

```python
import numpy as np
from model import GradientBoosting
from sklearn.datasets import make_moons

X, y = make_moons(1000, noise=0.4, random_state=0)
for lr in (1.0, 0.1):
    model = GradientBoosting("log_loss", 500, learning_rate=lr, max_depth=3).fit(X[:400], y[:400])
    acc = [np.mean((F > 0) == y[400:]) for F in model.staged_decision_function(X[400:])]
    print(lr, acc[49], acc[499])                 # 1.0: 0.818 0.802    0.1: 0.848 0.823
assert acc[49] > 0.84
```

**3. Squared loss chases outliers.** The pseudo-residual for squared error is the raw error, so one corrupted label (50 instead of about 0) produces a huge residual. Round after round, trees spend their splits on it. Error against the true curve rises 6,000-fold, from 0.0033 to 19.8. Losses with bounded gradients (absolute error, Huber) limit the pull of any one point.

```python
import numpy as np
from model import GradientBoosting

rng = np.random.default_rng(0)
x = rng.uniform(0, 1, (200, 1))
y = np.sin(2 * np.pi * x[:, 0]) + rng.normal(0, 0.1, 200)
grid = np.linspace(0, 1, 500)[:, None]
truth = np.sin(2 * np.pi * grid[:, 0])

clean = GradientBoosting("squared_error", 100, 0.1, 3).fit(x, y)
y_bad = y.copy()
y_bad[np.argmin(np.abs(x[:, 0] - 0.5))] = 50.0               # one corrupted label
dirty = GradientBoosting("squared_error", 100, 0.1, 3).fit(x, y_bad)
err_clean, err_dirty = (np.mean((m.predict(grid) - truth) ** 2) for m in (clean, dirty))
print(err_clean, err_dirty)                                   # 0.0033  19.8
assert err_dirty > 1000 * err_clean
```

**4. No extrapolation.** The model is a sum of trees, and every tree is constant beyond the last split, so predictions go flat outside the training range. Trained on $y = 3x$ for $x \in [0, 1]$, it predicts 3.0 at $x = 2$ and at $x = 10$. This matters for trends over time, where tomorrow is always outside the training range.

```python
import numpy as np
from model import GradientBoosting

x = np.linspace(0, 1, 100)[:, None]
model = GradientBoosting("squared_error", 200, 0.1, 3).fit(x, 3 * x[:, 0])
print(model.predict(np.array([[1.0], [2.0], [10.0]])))       # [3.0 3.0 3.0]
assert np.allclose(model.predict(np.array([[2.0], [10.0]])), 3.0, atol=1e-3)
```

## Common mistakes

- **"Boosting and random forests are basically the same."** Forests grow deep trees independently and average them, which mainly reduces variance. Boosting grows shallow trees one after another, each depending on the last, which mainly reduces bias. They need opposite tuning: forests want deep trees and many of them; boosting wants shallow trees and a carefully chosen number of rounds.
- **"The trees are fitted to the labels."** Only in round 1, and only for squared loss. Every tree is fitted to pseudo-residuals, the negative gradient of the loss at the current predictions. For classification these are $y - p$, not the class labels.
- **"A learning rate of 1 is fastest, so best."** It reaches low training error in fewer rounds but usually generalises worse (example 2). Shrinkage is a form of regularisation.
- **"More rounds always help, as with forests."** Forests plateau; boosting eventually overfits (example 1). The number of rounds is a hyperparameter to tune with held-out data.
- **"Gradient boosting means gradient descent on the tree's parameters."** The gradient is taken with respect to the model's *predictions*, not any parameters. The tree is the device that turns $n$ gradient values into a function that also works on new inputs.

## Further reading

- Friedman, "Greedy function approximation: a gradient boosting machine", *Annals of Statistics* 29(5) (2001): the algorithm implemented here, including the Newton leaf values.
- Friedman, "Stochastic gradient boosting", *Computational Statistics & Data Analysis* 38(4) (2002): row subsampling.
- Freund and Schapire, "A decision-theoretic generalization of on-line learning and an application to boosting", *Journal of Computer and System Sciences* 55(1) (1997): AdaBoost, the algorithm that started boosting.
- Chen and Guestrin, "XGBoost: a scalable tree boosting system", *Proceedings of KDD* (2016): second-order boosting with regularised leaves, used in practice.
- Hastie, Tibshirani and Friedman, *The Elements of Statistical Learning*, 2nd edition (2009), chapter 10.
