"""Tests for 02-logistic-regression. References: scikit-learn and PyTorch, float64.

Tolerances:
  - Newton's method vs scikit-learn (newton-cg, tol=1e-14):   1e-10 absolute
  - gradient descent vs Newton's method after 1000 steps:     1e-10 absolute
  - loss, gradient and Hessian vs PyTorch autograd:            1e-10 absolute
"""

import numpy as np
import pytest
import torch
from model import LogisticRegression, gradients, hessian, log_loss, sigmoid
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression as SkLogisticRegression


@pytest.fixture
def data():
    X, y = make_classification(n_samples=300, n_features=4, n_informative=3, n_redundant=0,
                               flip_y=0.1, random_state=0)
    return X, y.astype(float)


def sklearn_reference(X, y, lam):
    # scikit-learn minimises 0.5 ||w||^2 + C sum(loss_i); ours is mean(loss_i) + 0.5 lam ||w||^2.
    # Dividing scikit-learn's objective by n C shows they agree when C = 1 / (n lam).
    kwargs = {"penalty": None} if lam == 0 else {"C": 1 / (len(y) * lam)}
    return SkLogisticRegression(solver="newton-cg", tol=1e-14, max_iter=10_000, **kwargs).fit(X, y)


@pytest.mark.parametrize("lam", [0.0, 0.01, 0.3])
def test_newton_matches_sklearn(data, lam):
    X, y = data
    ref = sklearn_reference(X, y, lam)
    model = LogisticRegression(lam=lam, method="newton", n_iters=25).fit(X, y)
    np.testing.assert_allclose(model.coef_, ref.coef_[0], rtol=0, atol=1e-10)
    assert model.intercept_ == pytest.approx(ref.intercept_[0], abs=1e-10)
    np.testing.assert_allclose(model.predict_proba(X), ref.predict_proba(X)[:, 1], atol=1e-10)


@pytest.mark.parametrize("lam", [0.0, 0.01])
def test_gradient_descent_matches_newton(data, lam):
    X, y = data
    newton = LogisticRegression(lam=lam, method="newton", n_iters=25).fit(X, y)
    gd = LogisticRegression(lam=lam, method="gd", learning_rate=1.0, n_iters=1000).fit(X, y)
    np.testing.assert_allclose(gd.coef_, newton.coef_, rtol=0, atol=1e-10)
    assert gd.intercept_ == pytest.approx(newton.intercept_, abs=1e-10)


def test_loss_gradient_hessian_match_pytorch(data):
    X, y = data
    lam = 0.05
    rng = np.random.default_rng(1)
    w, b = rng.normal(size=X.shape[1]), -0.3

    def torch_loss(theta):
        z = torch.tensor(X) @ theta[:-1] + theta[-1]
        bce = torch.nn.functional.binary_cross_entropy_with_logits(z, torch.tensor(y))
        return bce + 0.5 * lam * theta[:-1] @ theta[:-1]

    theta = torch.tensor(np.append(w, b), requires_grad=True)
    loss = torch_loss(theta)
    loss.backward()
    H_ref = torch.autograd.functional.hessian(torch_loss, theta.detach()).numpy()

    grad_w, grad_b = gradients(X, y, w, b, lam)
    assert log_loss(X, y, w, b, lam) == pytest.approx(loss.item(), abs=1e-10)
    np.testing.assert_allclose(np.append(grad_w, grad_b), theta.grad.numpy(), rtol=0, atol=1e-10)
    np.testing.assert_allclose(hessian(X, w, b, lam), H_ref, rtol=0, atol=1e-10)


def test_sigmoid_is_stable_and_matches_pytorch():
    z = np.array([-1000.0, -40.0, -1.0, 0.0, 1.0, 40.0, 1000.0])
    with np.errstate(over="raise"):                # any overflow would raise here
        p = sigmoid(z)
    np.testing.assert_allclose(p, torch.sigmoid(torch.tensor(z)).numpy(), rtol=1e-12, atol=0)


def test_newton_converges_quadratically(data):
    X, y = data
    losses = LogisticRegression(method="newton", n_iters=8).fit(X, y).losses
    gaps = losses[:-1] - losses[-1]
    # Quadratic convergence: the gap roughly squares each step, so it is tiny by step 4.
    assert gaps[1] < 1e-2 and gaps[3] < 1e-10


def test_readme_examples_run(run_readme):
    assert run_readme(__file__) >= 4
