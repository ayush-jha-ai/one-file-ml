"""Tests for 03-regularisation. Reference: scikit-learn Ridge and Lasso, float64.

Tolerances:
  - ridge vs scikit-learn Ridge(alpha = n * alpha):          1e-10 absolute
  - lasso vs scikit-learn Lasso(alpha, tol=1e-14):           1e-8 absolute
  - lasso optimality (subgradient) conditions:               1e-9 absolute
"""

import numpy as np
import pytest
from model import centre, lasso, lasso_alpha_max, objective, path, ridge, soft_threshold
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso, Ridge


@pytest.fixture
def data():
    X, y = load_diabetes(return_X_y=True)
    return X, y


@pytest.mark.parametrize("alpha", [1e-3, 0.1, 1.0, 10.0])
def test_ridge_matches_sklearn(data, alpha):
    X, y = data
    ref = Ridge(alpha=len(y) * alpha).fit(X, y)
    w, b = ridge(X, y, alpha)
    np.testing.assert_allclose(w, ref.coef_, rtol=0, atol=1e-10)
    assert b == pytest.approx(ref.intercept_, abs=1e-10)


@pytest.mark.parametrize("alpha", [0.01, 0.1, 0.5, 1.0])
def test_lasso_matches_sklearn(data, alpha):
    X, y = data
    ref = Lasso(alpha=alpha, tol=1e-14, max_iter=1_000_000).fit(X, y)
    w, b = lasso(X, y, alpha)
    np.testing.assert_allclose(w, ref.coef_, rtol=0, atol=1e-8)
    assert b == pytest.approx(ref.intercept_, abs=1e-8)
    np.testing.assert_array_equal(w != 0, ref.coef_ != 0)   # identical selected features


@pytest.mark.parametrize("alpha", [0.05, 0.3])
def test_lasso_satisfies_optimality_conditions(data, alpha):
    # At the lasso minimum, with c_j = x_j^T r / n (r = centred residual):
    #   w_j != 0  =>  c_j = alpha * sign(w_j)        w_j == 0  =>  |c_j| <= alpha
    X, y = data
    w, b = lasso(X, y, alpha)
    Xc, yc, _, _ = centre(X, y)
    c = Xc.T @ (yc - Xc @ w) / len(y)
    active = w != 0
    np.testing.assert_allclose(c[active], alpha * np.sign(w[active]), rtol=0, atol=1e-9)
    assert np.all(np.abs(c[~active]) <= alpha + 1e-9)


def test_lasso_objective_not_beaten_by_sklearn(data):
    X, y = data
    ref = Lasso(alpha=0.2, tol=1e-14, max_iter=1_000_000).fit(X, y)
    w, b = lasso(X, y, 0.2)
    ours = objective(X, y, w, b, 0.2, "l1")
    theirs = objective(X, y, ref.coef_, ref.intercept_, 0.2, "l1")
    assert ours <= theirs + 1e-12


def test_alpha_max_is_exact(data):
    X, y = data
    a = lasso_alpha_max(X, y)
    assert np.count_nonzero(lasso(X, y, a)[0]) == 0
    assert np.count_nonzero(lasso(X, y, 0.999 * a)[0]) == 1


def test_soft_threshold():
    rho = np.array([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0])
    np.testing.assert_array_equal(soft_threshold(rho, 1.0), [-2, 0, 0, 0, 0, 0, 2])


def test_ridge_never_zeroes_lasso_does(data):
    X, y = data
    alphas = np.geomspace(1e-3, 1.0, 12)
    assert np.all(path(X, y, alphas, "l2") != 0)
    assert np.count_nonzero(path(X, y, alphas, "l1")[-1]) < X.shape[1]


def test_readme_examples_run(run_readme):
    assert run_readme(__file__) >= 4
