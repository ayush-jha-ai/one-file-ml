# 07 · Principal component analysis

Find the few directions along which the data vary most, and describe each point by its position along them. It is used to compress data, to plot high-dimensional data in 2D, and to remove redundant features before another model.

![Left: a cloud of points and a line through its centre that rotates through 180 degrees; each point is joined to its projection on the line. Right: the variance of the projected points and the average squared distance to the line, plotted against the angle. Variance peaks at exactly the angle where distance is smallest, and the two always add up to the same total.](demo.gif)

## Intuition

Picture a cloud of points shaped like a cigar. To describe each point with one number instead of two, you would measure its position along the length of the cigar, because that direction captures most of how the points differ. That direction is the first principal component. PCA finds it by asking which line through the centre makes the projected points as spread out as possible. That is the same line that keeps the points closest to it, so the least information is lost. The second component is the best direction at right angles to the first, and so on.

## The maths

**Centre and covariance.** Data $\mathbf{x}_1, \dots, \mathbf{x}_n \in \mathbb{R}^d$ with mean $\bar{\mathbf{x}}$. Let $X_c$ be the $n\times d$ matrix whose rows are $\mathbf{x}_i - \bar{\mathbf{x}}$. The sample covariance matrix is

$$C = \frac{1}{n-1}X_c^\top X_c.$$

Entry $C_{jk}$ is the covariance between features $j$ and $k$, and the diagonal entries are the variances.

**Variance along a direction.** Take a unit vector $\mathbf{u}$ ($\mathbf{u}^\top\mathbf{u} = 1$). The position of centred point $i$ along $\mathbf{u}$ is $z_i = \mathbf{u}^\top(\mathbf{x}_i - \bar{\mathbf{x}})$. These positions have mean 0, so their variance is

$$\frac{1}{n-1}\sum_i z_i^2 = \frac{1}{n-1}\sum_i \mathbf{u}^\top(\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})^\top\mathbf{u} = \mathbf{u}^\top C\,\mathbf{u}.$$

**Maximise it.** We want the $\mathbf{u}$ that maximises $\mathbf{u}^\top C\mathbf{u}$ subject to $\mathbf{u}^\top\mathbf{u} = 1$. Without the constraint, scaling $\mathbf{u}$ up would make the variance as large as we like. Use a Lagrange multiplier $\lambda$: maximise $\mathbf{u}^\top C\mathbf{u} - \lambda(\mathbf{u}^\top\mathbf{u} - 1)$. Because $C$ is symmetric, the gradient of $\mathbf{u}^\top C\mathbf{u}$ is $2C\mathbf{u}$. Setting the gradient of the whole expression to zero:

$$2C\mathbf{u} - 2\lambda\mathbf{u} = \mathbf{0} \quad\Longrightarrow\quad C\mathbf{u} = \lambda\mathbf{u}.$$

So $\mathbf{u}$ must be an eigenvector of $C$. Its variance is then $\mathbf{u}^\top C\mathbf{u} = \lambda\,\mathbf{u}^\top\mathbf{u} = \lambda$, its eigenvalue. To maximise, take the eigenvector with the largest eigenvalue. Repeating with the extra constraint of being perpendicular to the directions already chosen gives the next eigenvectors in order of eigenvalue. $C$ is symmetric, so its eigenvectors are orthogonal.

**Maximum variance is minimum reconstruction error.** Projecting a centred point $\mathbf{x}$ onto the line along $\mathbf{u}$ gives $(\mathbf{u}^\top\mathbf{x})\mathbf{u}$. The leftover $\mathbf{x} - (\mathbf{u}^\top\mathbf{x})\mathbf{u}$ is perpendicular to $\mathbf{u}$, so by Pythagoras

$$\lVert\mathbf{x}\rVert^2 = (\mathbf{u}^\top\mathbf{x})^2 + \lVert\mathbf{x} - (\mathbf{u}^\top\mathbf{x})\mathbf{u}\rVert^2.$$

Sum over all points and divide by $n-1$. The left side is the total variance, which does not depend on $\mathbf{u}$. The first term on the right is the variance along $\mathbf{u}$, and the second is the average squared distance from the points to the line. Their sum is fixed, so making one as large as possible makes the other as small as possible. This is the right-hand panel of the animation. With $k$ components the same argument gives

$$\sum_i\lVert\mathbf{x}_i - \hat{\mathbf{x}}_i\rVert^2 = (n-1)\sum_{j > k}\lambda_j.$$

The error is exactly the variance in the directions you dropped.

**Computing it with the SVD.** Any matrix can be written $X_c = USV^\top$ with $U$ and $V$ having orthonormal columns and $S$ diagonal with entries $s_1 \ge s_2 \ge \dots \ge 0$. Then

$$C = \frac{1}{n-1}X_c^\top X_c = \frac{1}{n-1}VSU^\top USV^\top = V\,\frac{S^2}{n-1}\,V^\top.$$

That is an eigendecomposition: the principal components are the columns of $V$ (rows of $V^\top$), with variances $\lambda_j = s_j^2/(n-1)$. Working on $X_c$ directly avoids forming $X_c^\top X_c$, which squares the condition number and loses precision.

**Using the components.** With the top $k$ components as the rows of $V_k^\top$, a point's coordinates are $\mathbf{z} = V_k^\top(\mathbf{x} - \bar{\mathbf{x}})$, and its reconstruction is $\hat{\mathbf{x}} = \bar{\mathbf{x}} + V_k\mathbf{z}$. The fraction of variance kept by component $j$ is $\lambda_j / \sum_l \lambda_l$.

## Code walkthrough

| Equation | Code | Lines |
|---|---|---|
| sign convention: $\mathbf{u}$ and $-\mathbf{u}$ are both eigenvectors | `np.argmax(np.abs(components), axis=1)` and the sign flip | [14](model.py#L14) |
| $\bar{\mathbf{x}}$ and $X_c$ | `X.mean(axis=0)`, `X - self.mean_` | [27-28](model.py#L27-L28) |
| $X_c = USV^\top$ | `np.linalg.svd(Xc, full_matrices=False)` | [30](model.py#L30) |
| components = top $k$ rows of $V^\top$ | `fix_signs(Vt[:k])` | [31](model.py#L31) |
| $\lambda_j = s_j^2/(n-1)$ | `S**2 / (n - 1)` | [32](model.py#L32) |
| $\lambda_j/\sum_l\lambda_l$ | `variances[:k] / variances.sum()` | [34](model.py#L34) |
| $\mathbf{z} = V_k^\top(\mathbf{x} - \bar{\mathbf{x}})$ | `(X - self.mean_) @ self.components_.T` | [39](model.py#L39) |
| $\hat{\mathbf{x}} = \bar{\mathbf{x}} + V_k\mathbf{z}$ | `Z @ self.components_ + self.mean_` | [43](model.py#L43) |
| $C = X_c^\top X_c/(n-1)$, then eigenvectors | `Xc.T @ Xc / (len(X) - 1)`, `np.linalg.eigh(C)` | [55-56](model.py#L55-L56) |
| variance along $\mathbf{u}$: $\frac{1}{n-1}\sum_i z_i^2 = \mathbf{u}^\top C\mathbf{u}$ | `z @ z / (len(X) - 1)` | [66](model.py#L66) |

The tests check against scikit-learn's `PCA(svd_solver="full")` on the wine dataset for 1, 2, 5 and 13 components:
- components to `1e-10`, with the same signs;
- explained variance and ratios to `1e-10` relative;
- `transform` and `inverse_transform` to `1e-9`.

They also check:
- that the SVD and covariance routes agree;
- that the components are orthonormal;
- that reconstruction error equals $(n-1)\sum_{j>k}\lambda_j$ (to `1e-9`);
- that no random direction out of 2,000 has more variance than the first component.

## Where it breaks

**1. Units decide the components.** PCA maximises variance, and variance depends on units. In the wine dataset, proline is measured in numbers around 750 (standard deviation 314), while the other 12 features have standard deviations of 14 or less. On raw data the first component is almost exactly the proline axis and "explains" 99.8% of the variance. After standardising, the first component mixes many features and explains 36%.

```python
import numpy as np
from model import PCA
from sklearn.datasets import load_wine

X, _ = load_wine(return_X_y=True)
raw = PCA(2).fit(X)
print(raw.explained_variance_ratio_, np.abs(raw.components_[0, 12]))   # [0.998 0.002]  0.9998

X_std = (X - X.mean(axis=0)) / X.std(axis=0)
std = PCA(2).fit(X_std)
print(std.explained_variance_ratio_)                                    # [0.362 0.192]
assert raw.explained_variance_ratio_[0] > 0.99 and std.explained_variance_ratio_[0] < 0.4
```

**2. Curved structure.** Points on a circle are one-dimensional: a single angle describes each one. But no straight line captures that, so PCA reports two components of nearly equal importance (52% and 48%) and no way to compress to one number without losing half the information.

```python
import numpy as np
from model import PCA

angle = np.random.default_rng(0).uniform(0, 2 * np.pi, 500)
circle = np.column_stack([np.cos(angle), np.sin(angle)])
ratio = PCA().fit(circle).explained_variance_ratio_
print(ratio)                                    # [0.517 0.483]
assert abs(ratio[0] - ratio[1]) < 0.05
```

**3. The direction with the most variance is not always the one that matters.** Here two classes are separated along a direction with little variance, while an unrelated feature varies a lot. The first component explains 96% of the variance, and splitting on it classifies 53% of points correctly, barely better than guessing. The second component, with 4% of the variance, classifies 100%. PCA never looks at labels.

```python
import numpy as np
from model import PCA

rng = np.random.default_rng(0)
y = np.repeat([0, 1], 200)
X = np.column_stack([rng.normal(0, 5, 400),                         # high variance, no signal
                     np.where(y == 1, 1.0, -1.0) + rng.normal(0, 0.3, 400)])   # the signal
pca = PCA(2).fit(X)
Z = pca.transform(X)
acc = [max(np.mean((Z[:, j] > 0) == y), np.mean((Z[:, j] < 0) == y)) for j in (0, 1)]
print(pca.explained_variance_ratio_, acc)       # [0.958 0.042]  [0.53, 1.0]
assert acc[0] < 0.6 and acc[1] == 1.0
```

**4. One outlier can rotate everything.** Variance squares distances, so a single far-away point has a large pull. Adding one point to 100 rotates the first component from 28 degrees to 75 degrees.

```python
import numpy as np
from model import PCA

rng = np.random.default_rng(0)
x = rng.normal(size=100)
X = np.column_stack([x, 0.5 * x + rng.normal(0, 0.2, 100)])


def angle(data):
    u = PCA(1).fit(data).components_[0]
    return np.degrees(np.arctan2(u[1], u[0]))


with_outlier = np.vstack([X, [[0.0, 15.0]]])
print(angle(X), angle(with_outlier))            # 27.8  74.8
assert angle(with_outlier) - angle(X) > 40
```

## Common mistakes

- **"PCA selects the most important features."** Each component is a *combination* of all the original features. It does not pick a subset, and the combination can be hard to interpret.
- **"Components with small variance are noise and can be dropped."** Often true, but not guaranteed. Example 3 has all the signal in the smallest component. Check against the task before discarding.
- **"PCA needs the data to be Gaussian."** The maximum-variance and minimum-error results above hold for any data. Gaussianity matters only if you want to read the components as independent sources or use the variances in likelihood calculations.
- **"Forgetting to centre is harmless."** Without centring, the first direction found points from the origin to the data's mean instead of along its spread. The SVD of uncentred data is a different decomposition.
- **"The sign of a component means something."** $\mathbf{u}$ and $-\mathbf{u}$ are equally valid. Libraries pick a convention (here: the largest entry is positive), so compare components up to sign.

## Further reading

- Pearson, "On lines and planes of closest fit to systems of points in space", *Philosophical Magazine* 2(11) (1901): the minimum-distance view.
- Hotelling, "Analysis of a complex of statistical variables into principal components", *Journal of Educational Psychology* 24 (1933): the maximum-variance view and the name.
- Jolliffe, *Principal Component Analysis*, 2nd edition (2002).
- Shlens, "A tutorial on principal component analysis", arXiv:1404.1100 (2014).
- Strang, *Introduction to Linear Algebra*: eigenvalues of symmetric matrices and the singular value decomposition.
