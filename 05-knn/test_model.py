"""Tests for 05-knn. Reference: scikit-learn KNeighbors* with algorithm="brute".

Tolerances:
  - predicted labels, neighbour indices, vote fractions:   exact
  - neighbour distances:                                   1e-10 absolute
  - regression predictions:                                1e-10 absolute
"""

import numpy as np
import pytest
from model import KNN, leave_one_out_error, pairwise_sq_distances
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


@pytest.fixture
def data():
    X, y = make_classification(400, 5, n_informative=3, n_redundant=0, n_classes=3,
                               n_clusters_per_class=1, random_state=0)
    queries = np.random.default_rng(1).normal(size=(200, 5))
    return X, y, queries


@pytest.mark.parametrize("k", [1, 4, 7, 15])
def test_classifier_matches_sklearn(data, k):
    X, y, Q = data
    ref = KNeighborsClassifier(k, algorithm="brute").fit(X, y)
    model = KNN(k).fit(X, y)
    np.testing.assert_array_equal(model.predict(Q), ref.predict(Q))
    np.testing.assert_array_equal(model.predict_proba(Q), ref.predict_proba(Q))
    ref_dist, ref_idx = ref.kneighbors(Q)
    idx, dist = model.kneighbors(Q)
    np.testing.assert_array_equal(idx, ref_idx)
    np.testing.assert_allclose(dist, ref_dist, rtol=0, atol=1e-10)


def test_regressor_matches_sklearn():
    X, y = make_regression(300, 4, noise=5.0, random_state=0)
    Q = np.random.default_rng(1).normal(size=(100, 4))
    ref = KNeighborsRegressor(6, algorithm="brute").fit(X, y)
    np.testing.assert_allclose(KNN(6, "regress").fit(X, y).predict(Q), ref.predict(Q),
                               rtol=0, atol=1e-10)


def test_leave_one_out_matches_sklearn(data):
    X, y, _ = data
    ref = 1 - cross_val_score(KNeighborsClassifier(5, algorithm="brute"), X, y,
                              cv=LeaveOneOut()).mean()
    assert leave_one_out_error(X, y, 5) == pytest.approx(ref, abs=1e-12)


def test_distance_expansion_matches_direct():
    rng = np.random.default_rng(0)
    A, B = rng.normal(size=(30, 6)), rng.normal(size=(40, 6))
    direct = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    np.testing.assert_allclose(pairwise_sq_distances(A, B), direct, rtol=0, atol=1e-12)
    assert np.all(pairwise_sq_distances(A, A) >= 0)          # clipping keeps it non-negative


def test_one_neighbour_memorises_training_set(data):
    X, y, _ = data
    assert np.mean(KNN(1).fit(X, y).predict(X) == y) == 1.0


def test_tied_vote_goes_to_smallest_label():
    X = np.array([[0.0], [1.0], [10.0], [11.0]])
    y = np.array([2, 2, 1, 1])
    # k = 4: two votes each for labels 1 and 2. scikit-learn also picks the smaller label.
    assert KNN(4).fit(X, y).predict(np.array([[5.0]]))[0] == 1
    ref = KNeighborsClassifier(4, algorithm="brute").fit(X, y)
    assert ref.predict(np.array([[5.0]]))[0] == 1


def test_readme_examples_run(run_readme):
    assert run_readme(__file__) >= 4
