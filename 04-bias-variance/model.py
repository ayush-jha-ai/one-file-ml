"""Measure bias and variance by refitting a model on many fresh training sets. NumPy only.

Setup: y = f(x) + noise, noise ~ N(0, sigma^2), f known (we choose it).
Model: polynomial of a given degree, fitted by (optionally ridge-penalised) least squares.

At each test point x, with the prediction y_hat(x) varying over training sets D:
  E_D,eps[(y - y_hat)^2] = (E_D[y_hat] - f)^2  +  Var_D(y_hat)  +  sigma^2
                            bias^2                variance         noise
"""

import numpy as np


def true_function(x):
    return np.sin(2 * np.pi * x)


def sample_dataset(rng, n, noise, x=None):
    """n points with x ~ Uniform(0, 1) (or the given fixed x) and y = f(x) + noise."""
    x = rng.uniform(0, 1, n) if x is None else x
    return x, true_function(x) + rng.normal(0, noise, len(x))


def features(x, degree):
    """Legendre polynomials P_0..P_degree of 2x - 1. They span the same functions as
    1, x, ..., x^degree but are far better conditioned on [0, 1]."""
    return np.polynomial.legendre.legvander(2 * np.asarray(x) - 1, degree)


def fit(x, y, degree, alpha=0.0):
    """Least squares on the features, plus (alpha/2)||c||^2 if alpha > 0.

    With alpha = 0 and more features than points, lstsq returns the minimum-norm
    solution among all those that fit the points exactly.
    """
    Phi = features(x, degree)
    if alpha == 0:
        return np.linalg.lstsq(Phi, y, rcond=None)[0]
    n, p = Phi.shape
    return np.linalg.solve(Phi.T @ Phi / n + alpha * np.eye(p), Phi.T @ y / n)


def predict(coef, x):
    return features(x, len(coef) - 1) @ coef


def simulate(degree, n, noise, x_test, n_datasets=500, alpha=0.0, seed=0, x_fixed=None):
    """Fit one model per fresh training set. Returns predictions, shape (n_datasets, n_test)."""
    rng = np.random.default_rng(seed)
    preds = np.empty((n_datasets, len(x_test)))
    for k in range(n_datasets):
        x, y = sample_dataset(rng, n, noise, x_fixed)
        preds[k] = predict(fit(x, y, degree, alpha), x_test)
    return preds


def decompose(preds, x_test, noise):
    """Average bias^2, variance and expected test error over the test points."""
    f = true_function(x_test)
    mean_pred = preds.mean(axis=0)                  # E_D[y_hat(x)]
    bias_sq = np.mean((mean_pred - f) ** 2)
    variance = np.mean(preds.var(axis=0))           # Var_D(y_hat(x)), ddof = 0
    return {"bias_sq": bias_sq, "variance": variance, "noise": noise**2,
            "total": bias_sq + variance + noise**2}


def smoother_matrix(x_train, x_test, degree, alpha=0.0):
    """H with y_hat(x_test) = H y: predictions are a fixed linear map of the training y."""
    Phi = features(x_train, degree)
    n, p = Phi.shape
    if alpha == 0:
        M = np.linalg.pinv(Phi)                     # coef = pinv(Phi) y
    else:
        M = np.linalg.solve(Phi.T @ Phi / n + alpha * np.eye(p), Phi.T / n)
    return features(x_test, degree) @ M             # row i is h(x_test_i)^T


def exact_bias_variance(x_train, x_test, degree, noise, alpha=0.0):
    """Exact bias^2 and variance at each test point for a FIXED design x_train.

    y = f + eps with E[eps] = 0 and Cov(eps) = sigma^2 I, and y_hat = H y, so
      E[y_hat]   = H f                ->  bias^2   = (H f - f_test)^2
      Var(y_hat) = sigma^2 ||h(x)||^2 ->  variance = sigma^2 * row sums of H^2
    """
    H = smoother_matrix(x_train, x_test, degree, alpha)
    bias_sq = (H @ true_function(x_train) - true_function(x_test)) ** 2
    return bias_sq, noise**2 * np.sum(H**2, axis=1)
