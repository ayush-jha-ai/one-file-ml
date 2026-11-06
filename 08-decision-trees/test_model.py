"""Tests for 08-decision-trees. Reference: scikit-learn DecisionTree*.

scikit-learn stores X as float32 and, when two splits are exactly tied, picks one by
visiting features in a random order. So:
  - data are cast to float32 values first, so both implementations see identical numbers;
  - up to depth 4 on tie-free data the trees must agree exactly;
  - for full-depth trees, every node where the two trees choose different splits must be
    an exact tie (equal weighted child impurity to 1e-12), which is checked node by node.

Tolerances:
  - predictions and class probabilities (depth <= 4):     exact
  - regression predictions (depth <= 4):                   1e-10 absolute
  - feature importances (depth <= 4):                      1e-12 absolute
  - impurity of tied alternative splits:                   1e-12 absolute
"""

import numpy as np
import pytest
from model import DecisionTree, best_split, impurity
from sklearn.datasets import load_breast_cancer, make_classification, make_regression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def as_float32(a):
    return np.asarray(a).astype(np.float32).astype(np.float64)


@pytest.fixture
def clf_data():
    X, y = make_classification(600, 6, n_informative=4, n_redundant=0, n_classes=3,
                               flip_y=0.05, random_state=0)
    Q = np.random.default_rng(1).normal(size=(500, 6))
    return as_float32(X), y, as_float32(Q)


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_classifier_matches_sklearn(clf_data, criterion, depth):
    X, y, Q = clf_data
    ref = DecisionTreeClassifier(criterion=criterion, max_depth=depth, random_state=0).fit(X, y)
    model = DecisionTree(criterion, max_depth=depth).fit(X, y)
    np.testing.assert_array_equal(model.predict(Q), ref.predict(Q))
    np.testing.assert_array_equal(model.predict_proba(Q), ref.predict_proba(Q))
    np.testing.assert_allclose(model.feature_importances_, ref.feature_importances_, atol=1e-12)
    assert model.n_leaves() == ref.get_n_leaves()


@pytest.mark.parametrize("depth", [2, 4])
def test_regressor_matches_sklearn(depth):
    X, y = make_regression(400, 6, noise=10.0, random_state=0)
    X = as_float32(X)
    Q = as_float32(np.random.default_rng(1).normal(size=(200, 6)))
    ref = DecisionTreeRegressor(max_depth=depth, random_state=0).fit(X, y)
    model = DecisionTree("squared_error", max_depth=depth).fit(X, y)
    np.testing.assert_allclose(model.predict(Q), ref.predict(Q), rtol=0, atol=1e-10)
    np.testing.assert_allclose(model.feature_importances_, ref.feature_importances_, atol=1e-12)


def split_score(X, y, j, threshold, n_classes):
    left = X[:, j] <= threshold
    counts = [np.bincount(y[m], minlength=n_classes) for m in (left, ~left)]
    return sum(c.sum() * impurity(c, "gini") for c in counts) / len(y)


@pytest.mark.parametrize("dataset", ["synthetic", "breast_cancer"])
def test_every_disagreement_with_sklearn_is_an_exact_tie(clf_data, dataset):
    if dataset == "synthetic":
        X, y, _ = clf_data
    else:
        X, y = load_breast_cancer(return_X_y=True)
        X = as_float32(X)
    ref = DecisionTreeClassifier(random_state=0).fit(X, y).tree_
    ours = DecisionTree("gini").fit(X, y)
    n_classes, ties, agreements = len(np.unique(y)), 0, 0
    stack = [(0, ours.root, np.ones(len(y), bool))]
    while stack:
        i, node, rows = stack.pop()
        if ref.children_left[i] == -1 or "feature" not in node:
            assert ref.children_left[i] == -1 and "feature" not in node   # both leaves
            continue
        f, t = ref.feature[i], float(np.float32(ref.threshold[i]))
        same = f == node["feature"] and np.isclose(t, node["threshold"], rtol=0, atol=1e-6)
        if not same:                     # different split: it must score exactly the same
            a = split_score(X[rows], y[rows], f, t, n_classes)
            b = split_score(X[rows], y[rows], node["feature"], node["threshold"], n_classes)
            assert a == pytest.approx(b, abs=1e-12)
            ties += 1
            continue                     # subtrees differ from here on
        agreements += 1
        go_left = rows & (X[:, f] <= node["threshold"])
        stack += [(ref.children_left[i], node["left"], go_left),
                  (ref.children_right[i], node["right"], rows & ~go_left)]
    assert agreements > 5


def test_best_split_is_optimal_by_brute_force():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(60, 3)), rng.integers(0, 3, 60)
    j, t, score = best_split(X, y, "gini", 3, min_leaf=1)
    for f in range(3):
        values = np.unique(X[:, f])
        for thr in (values[:-1] + values[1:]) / 2:
            assert split_score(X, y, f, thr, 3) >= score - 1e-12


def test_full_tree_fits_training_data(clf_data):
    X, y, _ = clf_data
    assert np.mean(DecisionTree("gini").fit(X, y).predict(X) == y) == 1.0


def test_readme_examples_run(run_readme):
    assert run_readme(__file__) >= 4
