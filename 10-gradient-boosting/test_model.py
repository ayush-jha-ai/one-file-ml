"""Tests for 10-gradient-boosting. Reference: scikit-learn GradientBoosting*, PyTorch autograd.

scikit-learn's boosting is deterministic without subsampling, and its "friedman_mse"
split criterion picks the same split as plain squared error. It stores X as float32 and
breaks exactly tied splits by a random feature order. So data are cast to float32 values,
exact agreement is required where nodes are large enough for ties to be absent, and
elsewhere every disagreement is checked to be an exact tie.

Tolerances:
  - staged predictions vs scikit-learn (all 100 rounds):        1e-10 absolute
  - pseudo-residuals vs -dL/dF from PyTorch autograd:            1e-12 absolute
  - squared error of tied alternative splits:                    1e-10 absolute
"""

import numpy as np
import pytest
import torch
from model import GradientBoosting
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor


def as_float32(a):
    return np.asarray(a).astype(np.float32).astype(np.float64)


@pytest.fixture(scope="module")
def reg_data():
    X, y = make_regression(400, 5, noise=10.0, random_state=0)
    return as_float32(X), y, as_float32(np.random.default_rng(1).normal(size=(300, 5)))


@pytest.fixture(scope="module")
def clf_data():
    X, y = make_classification(500, 6, n_informative=4, flip_y=0.1, random_state=0)
    return as_float32(X), y, as_float32(np.random.default_rng(1).normal(size=(300, 6)))


@pytest.mark.parametrize("min_leaf", [5, 10, 20])
def test_regressor_matches_sklearn_at_every_round(reg_data, min_leaf):
    X, y, Q = reg_data
    model = GradientBoosting("squared_error", 100, 0.1, 3, min_samples_leaf=min_leaf).fit(X, y)
    ref = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3,
                                    min_samples_leaf=min_leaf, random_state=0).fit(X, y)
    for ours, theirs in zip(model.staged_decision_function(Q), ref.staged_predict(Q),
                            strict=True):
        np.testing.assert_allclose(ours, theirs, rtol=0, atol=1e-10)


def test_classifier_matches_sklearn_at_every_round(clf_data):
    X, y, Q = clf_data
    model = GradientBoosting("log_loss", 100, 0.1, 3, min_samples_leaf=20).fit(X, y)
    ref = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,
                                     min_samples_leaf=20, random_state=0).fit(X, y)
    for ours, theirs in zip(model.staged_decision_function(Q),
                            ref.staged_decision_function(Q), strict=True):
        np.testing.assert_allclose(ours, theirs.ravel(), rtol=0, atol=1e-10)
    np.testing.assert_allclose(model.predict_proba(Q), ref.predict_proba(Q), rtol=0, atol=1e-10)


def test_first_tree_disagreements_are_exact_ties(clf_data):
    # With 5-point leaves, scikit-learn's first tree differs from ours. Check every
    # differing node: both splits must leave exactly the same squared error.
    X, y, _ = clf_data
    ours = GradientBoosting("log_loss", 1, 0.1, 3, min_samples_leaf=5).fit(X, y).trees[0][0]
    ref = GradientBoostingClassifier(n_estimators=1, max_depth=3, min_samples_leaf=5,
                                     random_state=0).fit(X, y).estimators_[0, 0].tree_
    r = y - y.mean()                                   # first-round pseudo-residuals

    def sse(rows, j, t):
        left = rows & (X[:, j] <= t)
        return sum(np.sum((r[s] - r[s].mean()) ** 2) for s in (left, rows & ~left) if s.any())

    ties, stack = 0, [(0, ours, np.ones(len(y), bool))]
    while stack:
        i, node, rows = stack.pop()
        if ref.children_left[i] == -1 or "leaf" in node:
            continue
        f, t = ref.feature[i], ref.threshold[i]
        if f != node["feature"] or abs(t - node["threshold"]) > 1e-6:
            assert sse(rows, f, t) == pytest.approx(sse(rows, node["feature"], node["threshold"]),
                                                    abs=1e-10)
            ties += 1
            continue
        left = rows & (X[:, f] <= node["threshold"])
        stack += [(ref.children_left[i], node["left"], left),
                  (ref.children_right[i], node["right"], rows & ~left)]
    assert ties >= 1


@pytest.mark.parametrize("loss", ["squared_error", "log_loss"])
def test_pseudo_residuals_are_negative_gradients(loss):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 50).astype(float)
    F = torch.tensor(rng.normal(size=50), requires_grad=True)
    yt = torch.tensor(y)
    if loss == "squared_error":
        total = torch.sum((yt - F) ** 2 / 2)
        ours = y - F.detach().numpy()
    else:
        total = torch.sum(torch.nn.functional.softplus(F) - yt * F)   # log(1 + e^F) - y F
        ours = y - 1 / (1 + np.exp(-F.detach().numpy()))
    total.backward()
    np.testing.assert_allclose(ours, -F.grad.numpy(), rtol=0, atol=1e-12)


def test_training_error_never_increases_for_squared_loss(reg_data):
    X, y, _ = reg_data
    model = GradientBoosting("squared_error", 60, 0.5, 2).fit(X, y)
    mse = [np.mean((F - y) ** 2) for F in model.staged_decision_function(X)]
    assert np.all(np.diff(mse) <= 1e-9)


def test_readme_examples_run(run_readme):
    assert run_readme(__file__) >= 4
