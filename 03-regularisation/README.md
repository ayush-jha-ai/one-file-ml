# 03 · Regularisation: L1 versus L2

Add a penalty on the size of the weights so the model cannot chase noise. L2 (ridge) shrinks every weight a little; L1 (lasso) sets some weights to exactly zero, which also chooses which features to keep.

![Left: loss contours meeting a shrinking circle (L2) and a shrinking diamond (L1). The L1 solution lands on a corner, where one weight is exactly zero. Right: ten weights on the diabetes dataset; as the budget shrinks, lasso switches weights off one by one while ridge only shrinks them.](demo.gif)

## Intuition

A model with many weights can fit the training data's noise as well as its signal. Regularisation adds a cost for large weights, so the model only keeps a large weight if it reduces the error enough to pay for it. L2 charges the *square* of each weight. That cost becomes negligible near zero, so weights get small but never reach zero. L1 charges the *absolute value*, which costs the same per unit however small the weight already is. So it is often cheapest to switch a weight off completely. Geometrically, the L1 penalty's contour is a diamond with corners on the axes, and the best fit usually touches it at a corner.

## The maths

**Two penalised problems.** Both use the same data term, so the only difference is the penalty. With $\alpha \ge 0$:

$$\text{ridge: } \min_{\mathbf{w}, b}\ \frac{1}{2n}\lVert\mathbf{y} - X\mathbf{w} - b\rVert^2 + \frac{\alpha}{2}\lVert\mathbf{w}\rVert_2^2, \qquad \text{lasso: } \min_{\mathbf{w}, b}\ \frac{1}{2n}\lVert\mathbf{y} - X\mathbf{w} - b\rVert^2 + \alpha\lVert\mathbf{w}\rVert_1,$$

where $\lVert\mathbf{w}\rVert_2^2 = \sum_j w_j^2$ and $\lVert\mathbf{w}\rVert_1 = \sum_j |w_j|$. The $\tfrac12$ factors only make the derivatives tidy.

**Removing the intercept.** $b$ is not penalised. Setting the derivative of either objective with respect to $b$ to zero gives $\sum_i (y_i - \mathbf{x}_i^\top\mathbf{w} - b) = 0$, so

$$b = \bar{y} - \bar{\mathbf{x}}^\top\mathbf{w}.$$

Substituting back replaces every $x_{ij}$ by $x_{ij} - \bar{x}_j$ and every $y_i$ by $y_i - \bar{y}$. So we centre the data, solve for $\mathbf{w}$ with no intercept, and recover $b$ at the end. From here on, $X$ and $\mathbf{y}$ mean the centred versions.

**Ridge has a closed form.** The gradient is $-\tfrac1n X^\top(\mathbf{y} - X\mathbf{w}) + \alpha\mathbf{w}$. Setting it to zero:

$$\left(\tfrac{1}{n}X^\top X + \alpha I\right)\mathbf{w} = \tfrac{1}{n}X^\top\mathbf{y}.$$

$\tfrac1n X^\top X$ has eigenvalues $\lambda_k \ge 0$, so the matrix on the left has eigenvalues $\lambda_k + \alpha > 0$. It is always invertible when $\alpha > 0$, even with duplicated features (concept 01, example 4). In the eigenvector basis, ridge multiplies each component of the unpenalised solution by

$$\frac{\lambda_k}{\lambda_k + \alpha},$$

which is less than 1 but never 0. Ridge shrinks every direction and switches none off. The directions with small $\lambda_k$ (the ones the data barely constrains) are shrunk most.

**Lasso has no closed form, because $|w|$ has a corner at 0.** But the problem in one weight at a time, with the others held fixed, can be solved exactly. Let $\mathbf{r}_{-j} = \mathbf{y} - \sum_{k \ne j}\mathbf{x}_k w_k$ be the residual ignoring feature $j$. Expanding the square, the objective as a function of $w_j$ alone is

$$f(w_j) = \frac{z_j}{2}w_j^2 - \rho_j w_j + \alpha|w_j| + \text{const}, \qquad z_j = \frac{\mathbf{x}_j^\top\mathbf{x}_j}{n}, \quad \rho_j = \frac{\mathbf{x}_j^\top\mathbf{r}_{-j}}{n}.$$

Minimise it case by case:

- If $w_j > 0$, then $|w_j| = w_j$ and $f'(w_j) = z_jw_j - \rho_j + \alpha = 0$ gives $w_j = (\rho_j - \alpha)/z_j$. This is positive only when $\rho_j > \alpha$.
- If $w_j < 0$, then $|w_j| = -w_j$ and the same step gives $w_j = (\rho_j + \alpha)/z_j$. This is negative only when $\rho_j < -\alpha$.
- If $|\rho_j| \le \alpha$, neither case is consistent, so the minimum is at the corner, $w_j = 0$.

All three cases are one formula, the *soft-thresholding* operator $S$:

$$w_j = \frac{S(\rho_j, \alpha)}{z_j}, \qquad S(\rho, \alpha) = \operatorname{sign}(\rho)\max(|\rho| - \alpha,\ 0).$$

*Coordinate descent* cycles through $j = 1, \dots, d$ applying this update until nothing changes. Each update can only lower the objective, and the objective is convex, so this converges to the global minimum. In code, $\mathbf{r}_{-j}$ is never formed: with the full residual $\mathbf{r}$, $\mathbf{r}_{-j} = \mathbf{r} + \mathbf{x}_jw_j$, so $\rho_j = \mathbf{x}_j^\top\mathbf{r}/n + z_jw_j$.

**Why L1 gives exact zeros and L2 does not.** Near $w = 0$, the L2 penalty's slope is $\alpha w$, which vanishes, so any small benefit from the data outweighs it. The L1 penalty's slope is $\pm\alpha$ however close $w$ is to 0, so a feature has to earn at least $\alpha$ of correlation with the residual ($|\rho_j| > \alpha$) before it is allowed to be non-zero.

**When does lasso select nothing?** With $\mathbf{w} = \mathbf{0}$ the residual is $\mathbf{y}$, so $\rho_j = \mathbf{x}_j^\top\mathbf{y}/n$. Every weight stays at zero exactly when every $|\rho_j| \le \alpha$:

$$\alpha_{\max} = \max_j \frac{|\mathbf{x}_j^\top\mathbf{y}|}{n}.$$

**Penalty and ball are two views of one problem.** For each $\alpha$ there is a radius $t$ such that the penalised problem has the same solution as minimising the data term subject to $\lVert\mathbf{w}\rVert \le t$. The left panel of the animation draws that ball: a circle for L2, a diamond for L1. The solution is where the smallest loss contour touches the ball, and the diamond's corners stick out towards the contours.

## Code walkthrough

| Equation | Code | Lines |
|---|---|---|
| centre: $\bar{\mathbf{x}}$, $\bar{y}$ | `X.mean(axis=0), y.mean()` | [18](model.py#L18) |
| $(\tfrac1n X^\top X + \alpha I)\mathbf{w} = \tfrac1n X^\top\mathbf{y}$ | `np.linalg.solve(Xc.T @ Xc / n + alpha * np.eye(d), Xc.T @ yc / n)` | [27](model.py#L27) |
| $b = \bar{y} - \bar{\mathbf{x}}^\top\mathbf{w}$ | `y_mean - x_mean @ w` | [28](model.py#L28), [58](model.py#L58) |
| $S(\rho, \alpha) = \operatorname{sign}(\rho)\max(\lvert\rho\rvert - \alpha, 0)$ | `np.sign(rho) * np.maximum(np.abs(rho) - alpha, 0.0)` | [33](model.py#L33) |
| $z_j = \mathbf{x}_j^\top\mathbf{x}_j/n$ | `(Xc**2).sum(axis=0) / n` | [41](model.py#L41) |
| $\rho_j = \mathbf{x}_j^\top\mathbf{r}/n + z_jw_j$ | `Xc[:, j] @ r / n + z[j] * w[j]` | [51](model.py#L51) |
| $w_j = S(\rho_j, \alpha)/z_j$ | `soft_threshold(rho, alpha) / z[j]` | [52](model.py#L52) |
| keep $\mathbf{r} = \mathbf{y} - X\mathbf{w}$ current | `r -= Xc[:, j] * (w_new - w[j])` | [53](model.py#L53) |
| stop when a full sweep changes nothing | `largest_change < tol` | [56](model.py#L56) |
| $\alpha_{\max} = \max_j\lvert\mathbf{x}_j^\top\mathbf{y}\rvert/n$ | `np.abs(Xc.T @ yc).max() / len(yc)` | [64](model.py#L64) |

Updating the residual in place (line 53) makes each coordinate update cost $O(n)$ instead of the $O(nd)$ needed to recompute $X\mathbf{w}$.

The tests check ridge against scikit-learn's `Ridge(alpha = n·α)` (to `1e-10`) and lasso against `Lasso(alpha = α)` (to `1e-8`), including that both select exactly the same features. They also check the lasso solution against the optimality conditions derived above (to `1e-9`), independently of any library, and that $\alpha_{\max}$ is exact.

## Where it breaks

**1. Correlated features: lasso's choice is arbitrary.** Two features that both measure the same thing are interchangeable, so lasso tends to keep one and drop the other, and which one it keeps can flip when a few rows change. Ridge splits the weight between them and stays put. On four bootstrap resamples of the same 100 rows:

```python
import numpy as np
from model import lasso, ridge

rng = np.random.default_rng(0)
x = rng.normal(size=100)
X = np.column_stack([x + rng.normal(0, 0.05, 100), x + rng.normal(0, 0.05, 100)])
y = 2 * x + rng.normal(0, 0.5, 100)

lasso_w1, ridge_w1 = [], []
for seed in range(4):
    rows = np.random.default_rng(seed).integers(0, 100, 100)   # resample rows
    lasso_w1.append(lasso(X[rows], y[rows], 0.1)[0][0])
    ridge_w1.append(ridge(X[rows], y[rows], 0.1)[0][0])
print(np.round(lasso_w1, 2))     # [0.13 1.85 0.65 1.18]: anywhere from "ignore" to "everything"
print(np.round(ridge_w1, 2))     # [0.91 0.96 0.92 0.95]
assert np.ptp(lasso_w1) > 1.5 and np.ptp(ridge_w1) < 0.1
```

**2. The penalty depends on units.** The same $\alpha$ costs a feature measured in kilometres 1,000 times more, per unit of effect, than the same feature in metres, because its weight must be 1,000 times larger. Change one feature's units and lasso drops it. Standardise features before penalising.

```python
import numpy as np
from model import lasso

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 3))
y = X @ np.array([1.0, 1.0, 1.0]) + rng.normal(0, 1, 200)
print(lasso(X, y, 0.3)[0])                  # [0.58 0.59 0.70]: all three kept

X_km = X.copy()
X_km[:, 0] /= 1000                          # feature 0 now in km instead of m
print(lasso(X_km, y, 0.3)[0])               # [0.   0.58 0.64]: feature 0 dropped
assert np.all(lasso(X, y, 0.3)[0] != 0) and lasso(X_km, y, 0.3)[0][0] == 0
```

**3. Lasso also shrinks the features it keeps.** The soft threshold subtracts $\alpha$ from every surviving $\rho_j$, so the true weights 5, 3, 2 come out as 4.40, 2.54, 1.47. If you want lasso only for *selection*, refit ordinary least squares on the selected features.

```python
import numpy as np
from model import lasso

rng = np.random.default_rng(0)
X = rng.normal(size=(500, 10))
y = X @ np.r_[5.0, 3.0, 2.0, np.zeros(7)] + rng.normal(0, 1, 500)

w, _ = lasso(X, y, 0.5)
print(np.round(w[:3], 2))                                  # [4.4  2.54 1.47]
keep = w != 0
A = np.column_stack([X[:, keep], np.ones(500)])
refit = np.linalg.lstsq(A, y, rcond=None)[0]
print(np.round(refit[:3], 2))                              # [4.94 2.98 1.96]
assert np.all(np.abs(w[:3] - [5, 3, 2]) > 0.4) and np.all(np.abs(refit[:3] - [5, 3, 2]) < 0.1)
```

**4. More features than examples: lasso keeps at most $n$.** With 20 examples and 100 features, of which 30 truly matter, lasso never selects more than 20, however small $\alpha$ is. Here only 3 of its picks are real.

```python
import numpy as np
from model import lasso

rng = np.random.default_rng(0)
X = rng.normal(size=(20, 100))                              # n = 20 rows, 100 features
y = X @ np.r_[np.ones(30), np.zeros(70)]                    # 30 features matter, no noise
for alpha in (0.1, 0.01):
    w, _ = lasso(X, y, alpha, tol=1e-10)
    print(alpha, np.count_nonzero(w), np.count_nonzero(w[:30]))   # 18 3, then 19 3
    assert np.count_nonzero(w) <= 20 and np.count_nonzero(w[:30]) == 3
```

Coordinate descent slows down as $\alpha \to 0$. At $\alpha = 0.001$, stopping after 10,000 sweeps leaves 21 non-zero weights. That is more than the solution can have, so the count itself tells you the solver had not converged. With 100,000 sweeps it settles at 19.

The elastic net (both penalties at once) was designed to fix examples 1 and 4: its L2 part makes correlated features share weight and lifts the limit of $n$ selected features.

## Common mistakes

- **"The same $\alpha$ means the same strength for L1 and L2."** They are on different scales: one charges $\tfrac\alpha2 w^2$, the other $\alpha|w|$. To compare them, match something meaningful, such as the size of the constraint ball (as in the animation) or validation error. Never match the raw $\alpha$.
- **"Lasso set a weight to zero, so that feature is irrelevant."** It may be a near-copy of a feature that was kept (example 1). A zero means "not needed *given the others*", and it can change with a different sample.
- **"Penalise the intercept too; it is just another weight."** Then adding 100 to every $y$ no longer adds 100 to every prediction, and the result depends on where you put zero on the $y$ scale. Centre the data, or leave $b$ out of the penalty.
- **"Regularisation is only about overfitting."** It also turns problems with no unique answer into problems with exactly one. Ridge solves the duplicated-feature case from concept 01, where the closed form failed.
- **"Lasso is the Bayesian answer with a Laplace prior, so the zeros are what the posterior says."** Lasso is the posterior *mode* under that prior. The posterior *mean* is never exactly zero. Exact zeros come from taking the mode, not from the prior itself.

## Further reading

- Hoerl and Kennard, "Ridge regression: biased estimation for nonorthogonal problems", *Technometrics* 12(1) (1970).
- Tibshirani, "Regression shrinkage and selection via the lasso", *Journal of the Royal Statistical Society, Series B* 58(1) (1996).
- Zou and Hastie, "Regularization and variable selection via the elastic net", *Journal of the Royal Statistical Society, Series B* 67(2) (2005).
- Friedman, Hastie and Tibshirani, "Regularization paths for generalized linear models via coordinate descent", *Journal of Statistical Software* 33(1) (2010): the algorithm used here, as implemented in `glmnet`.
- Hastie, Tibshirani and Friedman, *The Elements of Statistical Learning*, 2nd edition (2009), section 3.4: shrinkage methods and the ridge eigenvalue view.
- Hastie, Tibshirani and Wainwright, *Statistical Learning with Sparsity: The Lasso and Generalizations* (2015).
