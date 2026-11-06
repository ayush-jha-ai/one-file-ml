"""Tests for 06-kmeans. Reference: scikit-learn KMeans(algorithm="lloyd") from the same start.

Tolerances:
  - centres vs scikit-learn from identical initial centres:     1e-10 absolute
  - labels vs scikit-learn:                                       exact
  - inertia vs scikit-learn:                                      1e-8 relative
  - k-means++ sampling probabilities vs D^2 formula:              1e-12 absolute
"""

import numpy as np
import pytest
from model import assign, best_of, inertia, kmeans, kmeans_plus_plus_probabilities, update
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


@pytest.fixture
def data():
    X, _ = make_blobs(500, centers=4, cluster_std=1.5, random_state=0)
    return X


@pytest.mark.parametrize("seed", range(5))
def test_lloyd_matches_sklearn_from_same_start(data, seed):
    X = data
    init = X[np.random.default_rng(seed).choice(len(X), 4, replace=False)]
    ref = KMeans(4, init=init, n_init=1, algorithm="lloyd", tol=0, max_iter=300).fit(X)
    centres, labels, history = kmeans(X, 4, init=init)
    np.testing.assert_allclose(centres, ref.cluster_centers_, rtol=0, atol=1e-10)
    np.testing.assert_array_equal(labels, ref.labels_)
    assert history[-1][2] == pytest.approx(ref.inertia_, rel=1e-8)


@pytest.mark.parametrize("init", ["k-means++", "random"])
def test_inertia_never_increases(data, init):
    _, _, history = kmeans(data, 6, init=init, seed=3)
    assert np.all(np.diff([h[2] for h in history]) <= 1e-9)


def test_each_half_step_is_optimal(data):
    X = data
    centres = X[:4].copy()
    labels = assign(X, centres)
    # Assign step: no point can lower its cost by switching cluster.
    d2 = ((X[:, None, :] - centres[None]) ** 2).sum(-1)
    assert np.all(d2[np.arange(len(X)), labels] <= d2.min(axis=1) + 1e-12)
    # Update step: the mean minimises the sum of squared distances, so nudging it hurts.
    new = update(X, labels, centres)
    base = inertia(X, new, labels)
    for delta in (np.array([1e-3, 0.0]), np.array([0.0, -1e-3])):
        assert inertia(X, new + delta, labels) > base


def test_kmeans_plus_plus_probabilities():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [0.0, 2.0]])
    p = kmeans_plus_plus_probabilities(X, X[[0]])
    np.testing.assert_allclose(p, np.array([0, 1, 9, 4]) / 14, rtol=0, atol=1e-12)


def test_kmeans_plus_plus_draws_follow_d_squared():
    # First centre fixed at x0; the second is drawn with probabilities 0, 1/14, 9/14, 4/14.
    X = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [0.0, 2.0]])
    rng = np.random.default_rng(0)
    p = kmeans_plus_plus_probabilities(X, X[[0]])
    draws = rng.choice(4, size=20_000, p=p)
    freq = np.bincount(draws, minlength=4) / 20_000
    # Binomial standard error is at most sqrt(0.25 / 20000) = 0.0035; allow 4 of them.
    np.testing.assert_allclose(freq, p, atol=0.014)


def test_empty_cluster_is_refilled():
    X = np.array([[0.0], [0.1], [0.2], [10.0]])
    centres = np.array([[0.1], [10.0], [100.0]])     # third centre attracts no points
    _, labels, _ = kmeans(X, 3, init=centres)
    assert len(np.unique(labels)) == 3


def test_best_of_is_at_least_as_good_as_one_run(data):
    single = kmeans(data, 4, seed=0)[2][-1][2]
    assert best_of(data, 4, n_init=10, seed=0)[2][-1][2] <= single + 1e-9


def test_readme_examples_run(run_readme):
    assert run_readme(__file__) >= 4
