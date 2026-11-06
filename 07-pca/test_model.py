"""Tests for 07-pca. Reference: scikit-learn PCA(svd_solver="full"), plus algebraic identities.

The wine features have variances up to about 1e5, so absolute tolerances are set
relative to that scale.

Tolerances:
  - components vs scikit-learn (same sign convention):         1e-10 absolute (unit vectors)
  - explained variance vs scikit-learn:                         1e-10 relative
  - transform / inverse_transform vs scikit-learn:              1e-9 absolute
  - SVD route vs covariance-eigendecomposition route:           1e-9 absolute
  - reconstruction error = (n - 1) * sum of dropped variances:  1e-9 relative
"""

import numpy as np
import pytest
from model import PCA, pca_by_covariance, variance_along
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA as SkPCA


@pytest.fixture
def X():
    return load_wine(return_X_y=True)[0]


@pytest.mark.parametrize("k", [1, 2, 5, 13])
def test_matches_sklearn(X, k):
    ref = SkPCA(k, svd_solver="full").fit(X)
    model = PCA(k).fit(X)
    np.testing.assert_allclose(model.components_, ref.components_, rtol=0, atol=1e-10)
    np.testing.assert_allclose(model.explained_variance_, ref.explained_variance_, rtol=1e-10)
    np.testing.assert_allclose(model.explained_variance_ratio_, ref.explained_variance_ratio_,
                               rtol=1e-10)
    np.testing.assert_allclose(model.transform(X), ref.transform(X), rtol=0, atol=1e-9)
    Z = model.transform(X)
    np.testing.assert_allclose(model.inverse_transform(Z), ref.inverse_transform(Z),
                               rtol=0, atol=1e-9)


@pytest.mark.parametrize("standardise", [False, True])
def test_svd_and_covariance_routes_agree(X, standardise):
    if standardise:
        X = (X - X.mean(0)) / X.std(0)
    V, variances = pca_by_covariance(X, 5)
    model = PCA(5).fit(X)
    np.testing.assert_allclose(V, model.components_, rtol=0, atol=1e-9)
    np.testing.assert_allclose(variances, model.explained_variance_, rtol=1e-9)


def test_components_are_orthonormal(X):
    V = PCA(6).fit(X).components_
    np.testing.assert_allclose(V @ V.T, np.eye(6), rtol=0, atol=1e-12)


@pytest.mark.parametrize("k", [1, 3, 8])
def test_reconstruction_error_equals_dropped_variance(X, k):
    full = PCA().fit(X)
    dropped = full.explained_variance_[k:].sum()
    assert PCA(k).fit(X).reconstruction_error(X) == pytest.approx((len(X) - 1) * dropped,
                                                                  rel=1e-9)


def test_first_component_maximises_variance():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3)) @ np.array([[3.0, 0, 0], [1.0, 1.0, 0], [0, 0.5, 0.2]])
    model = PCA(1).fit(X)
    best = variance_along(X, model.components_[0])
    assert best == pytest.approx(model.explained_variance_[0], rel=1e-12)
    for u in rng.normal(size=(2000, 3)):                     # no random direction does better
        assert variance_along(X, u) <= best + 1e-9


def test_readme_examples_run(run_readme):
    assert run_readme(__file__) >= 4
