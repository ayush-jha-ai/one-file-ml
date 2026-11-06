"""Tests for 04-bias-variance. Reference: scikit-learn, plus exact closed-form variance.

Tolerances:
  - polynomial fits vs scikit-learn (LinearRegression / Ridge):    1e-8 absolute
  - bias^2 + variance = mean squared deviation from f (identity):   1e-12 absolute
  - mean hat-matrix variance = sigma^2 p / n (identity):            1e-10 relative
  - Monte Carlo variance vs exact formula, 4000 training sets:      5% relative
    (sample variance has relative standard error sqrt(2/4000) = 2.2% per point;
     averaged over 50 points it is far smaller, so 5% is a loose but honest bound)
  - Monte Carlo bias^2 vs exact formula: about 4 standard errors of the estimate
"""

import numpy as np
import pytest
from model import (
    decompose,
    exact_bias_variance,
    features,
    fit,
    predict,
    sample_dataset,
    simulate,
    true_function,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

X_TEST = np.linspace(0, 1, 50)


@pytest.fixture
def dataset():
    return sample_dataset(np.random.default_rng(0), 30, 0.3)


@pytest.mark.parametrize("degree", [1, 3, 5])
def test_polynomial_fit_matches_sklearn(dataset, degree):
    # Legendre and monomial bases span the same functions, so predictions must agree.
    x, y = dataset
    ref = make_pipeline(PolynomialFeatures(degree), LinearRegression()).fit(x[:, None], y)
    ours = predict(fit(x, y, degree), X_TEST)
    np.testing.assert_allclose(ours, ref.predict(X_TEST[:, None]), rtol=0, atol=1e-8)


def test_ridge_fit_matches_sklearn(dataset):
    x, y = dataset
    alpha = 0.01
    ref = Ridge(alpha=len(x) * alpha, fit_intercept=False).fit(features(x, 9), y)
    np.testing.assert_allclose(fit(x, y, 9, alpha), ref.coef_, rtol=0, atol=1e-8)


def test_min_norm_fit_matches_sklearn_when_more_features_than_points(dataset):
    x, y = dataset
    Phi = features(x, 59)                                        # 60 features, 30 points
    ref = LinearRegression(fit_intercept=False).fit(Phi, y)
    coef = fit(x, y, 59)
    np.testing.assert_allclose(Phi @ coef, y, atol=1e-8)          # interpolates the data
    np.testing.assert_allclose(coef, ref.coef_, rtol=0, atol=1e-8)


def test_decomposition_identity():
    preds = simulate(5, 20, 0.3, X_TEST, n_datasets=200)
    r = decompose(preds, X_TEST, 0.3)
    mse_to_f = np.mean((preds - true_function(X_TEST)) ** 2)
    assert r["bias_sq"] + r["variance"] == pytest.approx(mse_to_f, abs=1e-12)


def test_expected_error_matches_monte_carlo():
    # Draw a fresh noisy y at each test point for every fitted model, average (y - y_hat)^2.
    rng = np.random.default_rng(1)
    preds = simulate(3, 20, 0.3, X_TEST, n_datasets=4000)
    y_new = true_function(X_TEST) + rng.normal(0, 0.3, preds.shape)
    mc = np.mean((y_new - preds) ** 2)
    assert mc == pytest.approx(decompose(preds, X_TEST, 0.3)["total"], rel=0.02)


def test_hat_matrix_trace_gives_sigma_squared_p_over_n():
    x = (np.arange(20) + 0.5) / 20
    for degree in (2, 5, 9):
        _, v = exact_bias_variance(x, x, degree, noise=0.3)
        assert v.mean() == pytest.approx(0.09 * (degree + 1) / 20, rel=1e-10)


@pytest.mark.parametrize("degree, alpha", [(3, 0.0), (9, 0.0), (9, 1e-3)])
def test_monte_carlo_variance_matches_exact(degree, alpha):
    x = (np.arange(20) + 0.5) / 20
    preds = simulate(degree, 20, 0.3, X_TEST, n_datasets=4000, alpha=alpha, x_fixed=x)
    bias_sq, var = exact_bias_variance(x, X_TEST, degree, 0.3, alpha)
    assert preds.var(axis=0).mean() == pytest.approx(var.mean(), rel=0.05)
    # The mean of K predictions is f + b + e with e ~ N(0, v / K) (predictions are linear in
    # Gaussian noise). So E[mean_x (b + e)^2] = B + V/K, where B = mean b^2, V = mean v.
    # Subtract V/K. The remainder has standard deviation at most 2 sqrt(B V / K) + sqrt(2) V / K
    # (the worst case is e perfectly correlated across x). Allow 4 of those.
    k = len(preds)
    B, V = bias_sq.mean(), var.mean()
    sim_bias_sq = np.mean((preds.mean(axis=0) - true_function(X_TEST)) ** 2) - V / k
    assert sim_bias_sq == pytest.approx(B, abs=4 * (2 * np.sqrt(B * V / k) + np.sqrt(2) * V / k))


def test_ridge_trades_bias_for_variance():
    x = (np.arange(20) + 0.5) / 20
    weak = decompose(simulate(9, 20, 0.3, X_TEST, alpha=1e-6, x_fixed=x), X_TEST, 0.3)
    strong = decompose(simulate(9, 20, 0.3, X_TEST, alpha=1e-1, x_fixed=x), X_TEST, 0.3)
    assert strong["variance"] < weak["variance"] and strong["bias_sq"] > weak["bias_sq"]


def test_readme_examples_run(run_readme):
    assert run_readme(__file__) >= 4
