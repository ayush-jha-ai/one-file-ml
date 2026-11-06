# 04 · The bias-variance trade-off

Split a model's expected error into three parts: being systematically wrong (bias), being sensitive to which data you happened to get (variance), and noise nothing can remove. This is the standard language for underfitting and overfitting.

![Left: 25 polynomial fits, each trained on a different noisy sample of the same curve, with their average. As the degree rises from 0 to 12 the average fit closes in on the truth while the individual fits spread out. Right: exact bias squared falling, variance rising, and their sum making a U shape above the noise floor.](demo.gif)

## Intuition

Imagine you could collect your training set many times over, each time with different noise, and train a fresh model on each one. *Bias* asks: on average across all those models, how far is the prediction from the truth? *Variance* asks: how much do the models disagree with each other? A straight line through a wave has high bias (every line misses the wave) and low variance (every line is roughly the same). A degree-12 polynomial has almost no bias (on average it traces the wave) but high variance (each one wiggles to its own noise). The best model sits between the two.

## The maths

**Setup.** The data come from $y = f(x) + \varepsilon$, where $f$ is a fixed unknown function and $\varepsilon$ is noise with $\mathbb{E}[\varepsilon] = 0$ and $\operatorname{Var}(\varepsilon) = \sigma^2$. A training set $D$ is a random sample of such pairs. Training on $D$ gives a prediction function $\hat{f}_D$. Different training sets give different $\hat{f}_D$, so at a fixed input $x$ the prediction $\hat{f}_D(x)$ is a random variable.

**The quantity we care about.** Fix $x$, draw a fresh test label $y = f(x) + \varepsilon$ that is independent of $D$, and ask for the expected squared error over both the training set and the test noise:

$$\mathbb{E}_{D,\varepsilon}\big[(y - \hat{f}_D(x))^2\big].$$

**Step 1: separate the noise.** Write $f$ for $f(x)$ and $\hat{f}$ for $\hat{f}_D(x)$. Substituting $y = f + \varepsilon$ and expanding,

$$\mathbb{E}\big[(f + \varepsilon - \hat{f})^2\big] = \mathbb{E}[\varepsilon^2] + \mathbb{E}\big[(f - \hat{f})^2\big] + 2\,\mathbb{E}\big[\varepsilon\,(f - \hat{f})\big].$$

$\varepsilon$ is independent of $D$, so the last expectation factorises into $\mathbb{E}[\varepsilon]\cdot\mathbb{E}[f - \hat{f}] = 0 \cdot \mathbb{E}[f - \hat{f}] = 0$. And $\mathbb{E}[\varepsilon^2] = \sigma^2$.

**Step 2: separate bias from variance.** Let $\bar{f} = \mathbb{E}_D[\hat{f}]$ be the average prediction over training sets. Add and subtract it:

$$\mathbb{E}\big[(f - \hat{f})^2\big] = \mathbb{E}\big[(f - \bar{f} + \bar{f} - \hat{f})^2\big] = (f - \bar{f})^2 + \mathbb{E}\big[(\bar{f} - \hat{f})^2\big] + 2(f - \bar{f})\,\mathbb{E}\big[\bar{f} - \hat{f}\big].$$

$f - \bar{f}$ is a constant, so it comes out of the expectation, and $\mathbb{E}[\bar{f} - \hat{f}] = \bar{f} - \bar{f} = 0$. Putting the steps together:

$$\boxed{\ \mathbb{E}\big[(y - \hat{f}_D(x))^2\big] = \underbrace{\big(f(x) - \bar{f}(x)\big)^2}_{\text{bias}^2} + \underbrace{\mathbb{E}_D\big[(\hat{f}_D(x) - \bar{f}(x))^2\big]}_{\text{variance}} + \underbrace{\sigma^2}_{\text{noise}}\ }$$

Averaging over test points $x$ gives the numbers in the animation. No assumption was made about the model; the derivation only needs squared error and test noise independent of the training set.

**An exact formula for linear models.** Polynomial least squares (with or without a ridge penalty) predicts by a fixed linear combination of the training labels: $\hat{f}(x) = \mathbf{h}(x)^\top\mathbf{y}$, where $\mathbf{h}(x)$ depends only on the training inputs, not the labels. Keep the training inputs fixed and redraw only the noise, so that $\mathbf{y} = \mathbf{f} + \boldsymbol\varepsilon$ with $\operatorname{Cov}(\boldsymbol\varepsilon) = \sigma^2 I$. Then

$$\bar{f}(x) = \mathbb{E}\big[\mathbf{h}(x)^\top(\mathbf{f} + \boldsymbol\varepsilon)\big] = \mathbf{h}(x)^\top\mathbf{f}, \qquad \operatorname{Var}\big(\hat{f}(x)\big) = \operatorname{Var}\Big(\textstyle\sum_i h_i\varepsilon_i\Big) = \sigma^2\lVert\mathbf{h}(x)\rVert^2,$$

using the fact that the variance of a sum of independent terms is the sum of their variances. The right-hand panel of the animation uses these exact values, so it has no simulation noise.

**Variance grows with the number of parameters.** For ordinary least squares with $p$ coefficients and $n \ge p$ points, stack the vectors $\mathbf{h}(x_i)^\top$ for the training points into the *hat matrix* $H = \Phi(\Phi^\top\Phi)^{-1}\Phi^\top$, where $\Phi$ is the $n\times p$ feature matrix. $H$ is symmetric and $H^2 = H$, so $\sum_i\lVert\mathbf{h}(x_i)\rVert^2 = \operatorname{tr}(HH^\top) = \operatorname{tr}(H) = \operatorname{tr}\big((\Phi^\top\Phi)^{-1}\Phi^\top\Phi\big) = p$. So the average variance at the training points is

$$\frac{1}{n}\sum_{i=1}^n \operatorname{Var}\big(\hat{f}(x_i)\big) = \frac{\sigma^2 p}{n}.$$

Each extra parameter adds $\sigma^2/n$ of variance, however useless the parameter is. More data divides it down.

## Code walkthrough

| Equation | Code | Lines |
|---|---|---|
| $y = f(x) + \varepsilon$, $\varepsilon \sim N(0, \sigma^2)$ | `true_function(x) + rng.normal(0, noise, len(x))` | [21](model.py#L21) |
| polynomial features (Legendre basis, same span as $1, x, \dots, x^p$) | `np.polynomial.legendre.legvander(2 * np.asarray(x) - 1, degree)` | [27](model.py#L27) |
| least squares; minimum-norm solution when $p > n$ | `np.linalg.lstsq(Phi, y, rcond=None)` | [38](model.py#L38) |
| ridge: $(\tfrac1n\Phi^\top\Phi + \alpha I)\mathbf{c} = \tfrac1n\Phi^\top\mathbf{y}$ | `np.linalg.solve(Phi.T @ Phi / n + alpha * np.eye(p), Phi.T @ y / n)` | [40](model.py#L40) |
| $\bar{f}(x) = \mathbb{E}_D[\hat{f}_D(x)]$, estimated by simulation | `preds.mean(axis=0)` | [60](model.py#L60) |
| bias² $= (\bar{f} - f)^2$ | `np.mean((mean_pred - f) ** 2)` | [61](model.py#L61) |
| variance $= \mathbb{E}_D[(\hat{f}_D - \bar{f})^2]$ | `np.mean(preds.var(axis=0))` | [62](model.py#L62) |
| total $=$ bias² $+$ variance $+ \sigma^2$ | `bias_sq + variance + noise**2` | [64](model.py#L64) |
| rows $\mathbf{h}(x)^\top$ of the linear map $\mathbf{y} \mapsto \hat{f}$ | `features(x_test, degree) @ M` | [72-75](model.py#L72-L75) |
| exact $\bar{f} = \mathbf{h}^\top\mathbf{f}$, so bias² $= (\mathbf{h}^\top\mathbf{f} - f)^2$ | `(H @ true_function(x_train) - true_function(x_test)) ** 2` | [86](model.py#L86) |
| exact variance $= \sigma^2\lVert\mathbf{h}(x)\rVert^2$ | `noise**2 * np.sum(H**2, axis=1)` | [87](model.py#L87) |

Legendre polynomials are used instead of raw powers $x^k$ because they span exactly the same functions but stay well conditioned. With raw powers on $[0, 1]$, $x^{11}$ and $x^{12}$ are nearly identical columns and the fits lose precision.

The tests check:
- **Fits:** polynomial fits against scikit-learn's `PolynomialFeatures` + `LinearRegression` and `Ridge` (to `1e-8`), and the minimum-norm interpolating fit (to `1e-8`).
- **Identities:** the decomposition identity (to `1e-12`), and the $\sigma^2 p/n$ identity (to `1e-10`).
- **Simulation:** simulated variance and bias² against the exact formulas, within stated statistical tolerances.
- **Ridge:** that a stronger penalty lowers variance and raises bias.

## Where it breaks

Run these from inside this folder. All four use the sine curve, $\sigma = 0.3$, and evenly spaced training inputs unless stated.

**1. Double descent: past the interpolation point, error comes back down.** The textbook U-curve says error rises forever with complexity. With 20 points, error peaks at about 2.6 million at degree 19, where 20 coefficients must pass exactly through 20 noisy points. Past that, least squares picks the *smallest* coefficients that still interpolate, and adding features gives it more freedom to find a smooth interpolant. By degree 300 the error is 0.138, close to the best U-curve value of 0.113.

```python
import numpy as np
from model import exact_bias_variance

x_train = (np.arange(20) + 0.5) / 20
x_test = np.linspace(0, 1, 400)
total = {}
for degree in (3, 19, 300):
    bias_sq, var = exact_bias_variance(x_train, x_test, degree, noise=0.3)
    total[degree] = bias_sq.mean() + var.mean() + 0.3**2
print(total)                    # {3: 0.113, 19: 2589629.5, 300: 0.138}
assert total[19] > 1e6 and total[300] < 1.25 * total[3]
```

**2. Random inputs: the "average" error is set by rare disasters.** When the training inputs are random, now and then they cluster and leave a gap, and a degree-8 polynomial swings wildly across it. Over 1,000 training sets of 20 points the median error is 0.40, but the mean is about 1,400, and the worst 1% of training sets contribute 95% of it. The variance in the decomposition is real, but a single number hides that it comes from a heavy tail.

```python
import numpy as np
from model import simulate, true_function

x_test = np.linspace(0, 1, 400)
preds = simulate(8, 20, 0.3, x_test, n_datasets=1000)       # x drawn at random each time
err = np.mean((preds - true_function(x_test)) ** 2, axis=1) + 0.3**2
worst_share = np.sort(err)[-10:].sum() / err.sum()
print(np.median(err), err.mean(), worst_share)               # 0.40  1389  0.95
assert np.median(err) < 0.5 and err.mean() > 1000 and worst_share > 0.9
```

**3. More data cannot fix bias.** A straight line through a sine wave misses by the same amount however many points you give it. From 20 to 2,000 points, variance falls by a factor of 100, but bias² stays at 0.198, so the error never drops below $0.198 + 0.09$.

```python
import numpy as np
from model import exact_bias_variance

x_test = np.linspace(0, 1, 400)
for n in (20, 200, 2000):
    x_train = (np.arange(n) + 0.5) / n
    bias_sq, var = exact_bias_variance(x_train, x_test, degree=1, noise=0.3)
    print(n, round(bias_sq.mean(), 4), round(var.mean(), 5))  # bias² 0.1979 → 0.1978; var 0.009 → 0.00009
    assert bias_sq.mean() > 0.19
```

**4. "Overfitting" depends on the data size, not just the model.** Degree 9 on 20 points has total error 0.160. Give the same model 1,000 points and its variance falls to almost nothing, leaving 0.091, just above the noise floor of 0.09. The right complexity is a property of the model *and* the amount of data.

```python
import numpy as np
from model import exact_bias_variance

x_test = np.linspace(0, 1, 400)
for n in (20, 1000):
    x_train = (np.arange(n) + 0.5) / n
    bias_sq, var = exact_bias_variance(x_train, x_test, degree=9, noise=0.3)
    print(n, round(bias_sq.mean() + var.mean() + 0.09, 3))     # 0.160, then 0.091
assert bias_sq.mean() + var.mean() < 0.002
```

## Common mistakes

- **"Bias and variance are properties of one trained model."** They are properties of a *training procedure*, averaged over the training sets it could have seen. One fitted model has an error; it does not have a variance.
- **"High variance means the data are noisy."** Noise is $\sigma^2$, which no model can remove. Variance is how much *the model* changes with the sample. A degree-0 model on very noisy data has low variance.
- **"Pick the complexity with the lowest training error."** Training error falls every time you add a parameter, because the extra parameter can only help the fit. Use held-out data or cross-validation, which estimate the test error that the decomposition describes.
- **"The trade-off is a law: more complexity always eventually means more error."** Example 1 shows error falling again past the interpolation point. The decomposition always holds, but the U shape is a common pattern, not a theorem.
- **"This decomposition works for any loss."** The derivation used squared error. For classification error (0 or 1 per example) there is no equivalent clean three-way split, and extra variance can even reduce the error.

## Further reading

- Geman, Bienenstock and Doursat, "Neural networks and the bias/variance dilemma", *Neural Computation* 4(1) (1992): the paper that made this decomposition standard in machine learning.
- Hastie, Tibshirani and Friedman, *The Elements of Statistical Learning*, 2nd edition (2009), chapter 7: the decomposition, the hat matrix and effective degrees of freedom.
- Bishop, *Pattern Recognition and Machine Learning* (2006), section 3.2: the same derivation with polynomial regression on a sine curve.
- Belkin, Hsu, Ma and Mandal, "Reconciling modern machine-learning practice and the classical bias-variance trade-off", *Proceedings of the National Academy of Sciences* 116(32) (2019): double descent.
- Hastie, Montanari, Rosset and Tibshirani, "Surprises in high-dimensional ridgeless least squares interpolation", *Annals of Statistics* 50(2) (2022): when and why minimum-norm interpolation generalises.
