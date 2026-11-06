"""Tests for 11-neural-networks. References: PyTorch autograd and scikit-learn MLPClassifier.

Tolerances:
  - scalar autograd gradients vs PyTorch:                        1e-12 absolute
  - scalar-autograd network vs vectorised network (same weights):  1e-12 absolute
  - vectorised loss and gradients vs PyTorch:                     1e-12 absolute
  - weights after 50 full-batch steps vs scikit-learn MLP:        1e-12 absolute
  - backprop vs central finite differences (h = 1e-6):           1e-7 absolute
"""

import numpy as np
import pytest
import torch
from model import Value, forward, init_params, loss_and_grads, train
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier


@pytest.fixture
def data():
    X, y = make_classification(300, 5, n_informative=4, n_redundant=0, n_classes=3,
                               random_state=0)
    return X, y, init_params([5, 8, 6, 3], np.random.default_rng(0))


def test_scalar_autograd_matches_pytorch():
    a, b, c = 0.7, -1.3, 2.1
    va, vb, vc = Value(a), Value(b), Value(c)
    out = ((va * vb + vc).tanh() * va - (vb**2).exp() / vc + (va * va).log() + vc.relu()) * 3
    out.backward()
    ta, tb, tc = (torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in (a, b, c))
    ref = (torch.tanh(ta * tb + tc) * ta - torch.exp(tb**2) / tc + torch.log(ta * ta)
           + torch.relu(tc)) * 3
    ref.backward()
    assert out.data == pytest.approx(ref.item(), abs=1e-12)
    for v, t in ((va, ta), (vb, tb), (vc, tc)):
        assert v.grad == pytest.approx(t.grad.item(), abs=1e-12)


def test_reused_value_sums_gradients_over_paths():
    x = Value(3.0)
    y = x * x + x                      # x is used three times: dy/dx = 2x + 1 = 7
    y.backward()
    assert x.grad == 7.0


def test_scalar_network_equals_vectorised_network(data):
    X, y, params = data
    X, y = X[:4], y[:4]
    loss_vec, grads_vec = loss_and_grads(params, X, y)
    # Rebuild the same network one scalar at a time.
    P = [[[Value(w) for w in row] for row in W] for W, _ in params]
    B = [[Value(v) for v in b] for _, b in params]
    total = Value(0.0)
    for x_row, label in zip(X, y, strict=True):
        a = [Value(v) for v in x_row]
        for layer, (W, b) in enumerate(zip(P, B, strict=True)):
            z = [sum((a[i] * W[i][j] for i in range(len(a))), b[j]) for j in range(len(b))]
            a = z if layer == len(P) - 1 else [zj.tanh() for zj in z]
        exps = [zj.exp() for zj in a]
        total = total - (exps[label] / sum(exps[1:], exps[0])).log()
    loss = total / len(y)
    loss.backward()
    assert loss.data == pytest.approx(loss_vec, abs=1e-12)
    for layer, (dW, db) in enumerate(grads_vec):
        np.testing.assert_allclose([[w.grad for w in row] for row in P[layer]], dW, atol=1e-12)
        np.testing.assert_allclose([v.grad for v in B[layer]], db, atol=1e-12)


@pytest.mark.parametrize("activation", ["tanh", "relu"])
def test_vectorised_backprop_matches_pytorch(data, activation):
    X, y, params = data
    loss, grads = loss_and_grads(params, X, y, activation, alpha=0.3)
    tp = [[torch.tensor(W, requires_grad=True), torch.tensor(b, requires_grad=True)]
          for W, b in params]
    A = torch.tensor(X)
    for layer, (W, b) in enumerate(tp):
        Z = A @ W + b
        A = Z if layer == len(tp) - 1 else (torch.tanh(Z) if activation == "tanh" else
                                            torch.relu(Z))
    ref = (torch.nn.functional.cross_entropy(A, torch.tensor(y))
           + 0.3 / (2 * len(y)) * sum((W**2).sum() for W, _ in tp))
    ref.backward()
    assert loss == pytest.approx(ref.item(), abs=1e-12)
    for (dW, db), (W, b) in zip(grads, tp, strict=True):
        np.testing.assert_allclose(dW, W.grad.numpy(), rtol=0, atol=1e-12)
        np.testing.assert_allclose(db, b.grad.numpy(), rtol=0, atol=1e-12)


@pytest.mark.parametrize("activation", ["tanh", "relu"])
def test_training_matches_sklearn_mlp(data, activation):
    # Plain full-batch gradient descent in scikit-learn: SGD, one batch, no momentum.
    X, y, params = data
    ref = MLPClassifier((8, 6), activation=activation, solver="sgd", learning_rate_init=0.1,
                        momentum=0, nesterovs_momentum=False, batch_size=len(y), alpha=0.3,
                        shuffle=False, random_state=0)
    ref.partial_fit(X, y, classes=[0, 1, 2])
    ref.coefs_ = [W.copy() for W, _ in params]                 # start from our weights
    ref.intercepts_ = [b.copy() for _, b in params]
    for _ in range(50):
        ref.partial_fit(X, y)
    ours, _ = train([[W.copy(), b.copy()] for W, b in params], X, y, 0.1, 50, activation, 0.3)
    for (W, b), W_ref, b_ref in zip(ours, ref.coefs_, ref.intercepts_, strict=True):
        np.testing.assert_allclose(W, W_ref, rtol=0, atol=1e-12)
        np.testing.assert_allclose(b, b_ref, rtol=0, atol=1e-12)
    np.testing.assert_allclose(forward(ours, X, activation)[0][-1], ref.predict_proba(X),
                               rtol=0, atol=1e-12)


def test_backprop_matches_finite_differences(data):
    X, y, params = data
    _, grads = loss_and_grads(params, X, y)
    rng = np.random.default_rng(1)
    for layer in range(len(params)):
        W = params[layer][0]
        for _ in range(5):
            i, j = rng.integers(W.shape[0]), rng.integers(W.shape[1])
            old = W[i, j]
            W[i, j] = old + 1e-6
            up = loss_and_grads(params, X, y)[0]
            W[i, j] = old - 1e-6
            down = loss_and_grads(params, X, y)[0]
            W[i, j] = old
            assert grads[layer][0][i, j] == pytest.approx((up - down) / 2e-6, abs=1e-7)


def test_readme_examples_run(run_readme):
    assert run_readme(__file__) >= 4
