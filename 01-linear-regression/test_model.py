"""Tests for 01-linear-regression. References: scikit-learn and PyTorch, float64.

Tolerances:
  - closed form vs scikit-learn:        1e-10 absolute (both are exact solvers)
  - gradient descent vs scikit-learn:   1e-8 absolute after 2000 steps
  - analytic gradient vs PyTorch:       1e-10 absolute
"""

import numpy as np
import pytest
import torch
from model import (
    LinearRegression,
    gradients,
    max_stable_learning_rate,
    mse,
    normal_equation,
)
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression as SkLinearRegression


@pytest.fixture
def data():
    X, y = make_regression(n_samples=200, n_features=5, noise=10.0, random_state=0)
    return X, y


def test_normal_equation_matches_sklearn(data):
    X, y = data
    ref = SkLinearRegression().fit(X, y)
    w, b = normal_equation(X, y)
    np.testing.assert_allclose(w, ref.coef_, rtol=0, atol=1e-10)
    assert b == pytest.approx(ref.intercept_, abs=1e-10)


def test_gradient_descent_matches_sklearn(data):
    X, y = data
    ref = SkLinearRegression().fit(X, y)
    lr = 0.5 * max_stable_learning_rate(X)
    model = LinearRegression(learning_rate=lr, n_iters=2000).fit(X, y)
    np.testing.assert_allclose(model.coef_, ref.coef_, rtol=0, atol=1e-8)
    assert model.intercept_ == pytest.approx(ref.intercept_, abs=1e-8)
    np.testing.assert_allclose(model.predict(X), ref.predict(X), rtol=0, atol=1e-8)


def test_gradients_match_pytorch_autograd(data):
    X, y = data
    rng = np.random.default_rng(1)
    w, b = rng.normal(size=X.shape[1]), 0.7

    w_t = torch.tensor(w, requires_grad=True)
    b_t = torch.tensor(b, dtype=torch.float64, requires_grad=True)
    loss = torch.mean((torch.tensor(X) @ w_t + b_t - torch.tensor(y)) ** 2)
    loss.backward()

    grad_w, grad_b = gradients(X, y, w, b)
    assert mse(y, X @ w + b) == pytest.approx(loss.item(), abs=1e-10)
    np.testing.assert_allclose(grad_w, w_t.grad.numpy(), rtol=0, atol=1e-10)
    assert grad_b == pytest.approx(b_t.grad.item(), abs=1e-10)


def test_loss_never_increases_below_half_the_bound(data):
    X, y = data
    lr = 0.5 * max_stable_learning_rate(X)
    losses = LinearRegression(learning_rate=lr, n_iters=500).fit(X, y).losses
    assert np.all(np.diff(losses) <= 1e-9)


@pytest.mark.parametrize("factor, converges", [(0.99, True), (1.01, False)])
def test_learning_rate_bound_is_exact(data, factor, converges):
    X, y = data
    lr = factor * max_stable_learning_rate(X)
    losses = LinearRegression(learning_rate=lr, n_iters=3000).fit(X, y).losses
    best = SkLinearRegression().fit(X, y)
    best_loss = mse(y, best.predict(X))
    if converges:
        assert losses[-1] == pytest.approx(best_loss, rel=1e-8)
    else:
        assert losses[-1] > 1e6 * losses[0]


def test_collinear_features_gd_finds_minimum_norm_solution():
    # Column 2 is an exact copy of column 1, so infinitely many (w1, w2) fit equally well.
    rng = np.random.default_rng(0)
    x = rng.normal(size=100)
    X = np.column_stack([x, x])
    y = 3 * x + rng.normal(scale=0.1, size=100)
    ref = SkLinearRegression().fit(X, y)  # scikit-learn returns the minimum-norm solution
    model = LinearRegression(learning_rate=0.1, n_iters=2000).fit(X, y)
    np.testing.assert_allclose(model.coef_, ref.coef_, rtol=0, atol=1e-8)
    assert model.coef_[0] == pytest.approx(model.coef_[1], abs=1e-12)


def test_readme_examples_run(run_readme):
    assert run_readme(__file__) >= 4
