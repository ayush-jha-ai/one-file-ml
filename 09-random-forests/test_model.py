"""Tests for 09-random-forests. Reference: scikit-learn, exactly where possible, else statistically.

A forest uses random numbers (bootstrap rows, feature subsets) that cannot be made to
match scikit-learn's draw for draw. So:
  - with bootstrap off and all features allowed, one tree must equal scikit-learn's
    DecisionTreeClassifier exactly (on float32-valued, tie-free data, depth <= 4);
  - the full forest must match scikit-learn's accuracy statistically.

Tolerances:
  - single tree vs DecisionTreeClassifier:                    exact predictions/probabilities
  - out-of-bag fraction vs e^-1, 100 trees x 1000 rows:       0.006 (4 binomial std. errors)
  - mean test accuracy vs scikit-learn, 3 seeds each:         0.02 absolute (seed-to-seed std.
                                                              is about 0.004 for both)
  - out-of-bag accuracy vs scikit-learn's oob_score_:         0.03 absolute
"""

import numpy as np
import pytest
from model import RandomForest, predict_tree
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


@pytest.fixture(scope="module")
def data():
    X, y = make_classification(1200, 12, n_informative=6, n_redundant=2, flip_y=0.05,
                               random_state=1)
    return X[:600], y[:600], X[600:], y[600:]


@pytest.mark.parametrize("depth", [2, 4])
def test_one_tree_without_randomness_equals_sklearn_tree(depth):
    X, y = make_classification(600, 6, n_informative=4, n_redundant=0, n_classes=3,
                               flip_y=0.05, random_state=0)
    X = X.astype(np.float32).astype(float)
    Q = np.random.default_rng(1).normal(size=(500, 6)).astype(np.float32).astype(float)
    forest = RandomForest(1, max_features=None, max_depth=depth, bootstrap=False).fit(X, y)
    ref = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X, y)
    np.testing.assert_array_equal(forest.predict(Q), ref.predict(Q))
    np.testing.assert_array_equal(forest.predict_proba(Q), ref.predict_proba(Q))


def test_out_of_bag_fraction_is_one_over_e():
    X, y = make_classification(1000, 4, random_state=0)
    forest = RandomForest(100, max_depth=1, seed=0).fit(X, y)
    # P(row not drawn in n draws with replacement) = (1 - 1/n)^n -> e^-1.
    assert 1 - forest.in_bag.mean() == pytest.approx(np.exp(-1), abs=0.006)


def test_accuracy_matches_sklearn_statistically(data):
    X_train, y_train, X_test, y_test = data
    ours = [np.mean(RandomForest(60, seed=s).fit(X_train, y_train).predict(X_test) == y_test)
            for s in range(3)]
    ref = [RandomForestClassifier(60, random_state=s).fit(X_train, y_train).score(X_test, y_test)
           for s in range(3)]
    assert np.mean(ours) == pytest.approx(np.mean(ref), abs=0.02)


def test_oob_score_matches_sklearn_statistically(data):
    X_train, y_train, _, _ = data
    ours = RandomForest(100, seed=0).fit(X_train, y_train).oob_score()
    ref = RandomForestClassifier(100, oob_score=True, random_state=0).fit(X_train, y_train)
    assert ours == pytest.approx(ref.oob_score_, abs=0.03)


def test_forest_probability_is_average_of_trees(data):
    X_train, y_train, X_test, _ = data
    forest = RandomForest(10, seed=0).fit(X_train, y_train)
    manual = np.mean([predict_tree(t, X_test) for t in forest.trees], axis=0)
    np.testing.assert_allclose(forest.predict_proba(X_test), manual, rtol=0, atol=1e-15)
    np.testing.assert_allclose(forest.predict_proba(X_test).sum(axis=1), 1.0, atol=1e-12)


def test_forest_beats_single_tree(data):
    X_train, y_train, X_test, y_test = data
    tree = RandomForest(1, max_features=None, bootstrap=False).fit(X_train, y_train)
    forest = RandomForest(100, seed=0).fit(X_train, y_train)
    assert np.mean(forest.predict(X_test) == y_test) > np.mean(tree.predict(X_test) == y_test)


def test_readme_examples_run(run_readme):
    assert run_readme(__file__) >= 3
