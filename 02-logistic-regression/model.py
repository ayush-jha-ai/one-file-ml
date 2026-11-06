"""Binary logistic regression, trained by gradient descent or Newton's method. NumPy only.

Model:     p = sigmoid(X w + b),   sigmoid(z) = 1 / (1 + exp(-z))
Loss:      L = -(1/n) sum_i [ y_i log p_i + (1 - y_i) log(1 - p_i) ]  +  (lam/2) ||w||^2
Gradients: dL/dw = (1/n) X^T (p - y) + lam w,      dL/db = (1/n) sum_i (p_i - y_i)
Hessian:   H = (1/n) A^T S A + lam * diag(1, ..., 1, 0),   A = [X | 1],  S = diag(p (1 - p))

lam = 0 is plain maximum likelihood. scikit-learn's C corresponds to lam = 1 / (n C).
"""

import numpy as np


def sigmoid(z):
    """1 / (1 + e^-z), written so that neither branch can overflow."""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1 / (1 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])                      # z < 0, so e^z is small and safe
    out[~pos] = ez / (1 + ez)
    return out


def log_loss(X, y, w, b, lam=0.0):
    """Mean binary cross-entropy plus the L2 penalty, computed from the logits z.

    Uses log(1 + e^z) - y z, which equals -[y log p + (1 - y) log(1 - p)] but
    never takes the log of 0 when p rounds to exactly 0 or 1.
    """
    z = X @ w + b
    return np.mean(np.logaddexp(0, z) - y * z) + 0.5 * lam * w @ w


def gradients(X, y, w, b, lam=0.0):
    n = len(y)
    error = sigmoid(X @ w + b) - y            # p - y, shape (n,)
    grad_w = X.T @ error / n + lam * w        # (1/n) X^T (p - y) + lam w
    grad_b = error.mean()                     # (1/n) sum (p - y)
    return grad_w, grad_b


def hessian(X, w, b, lam=0.0):
    """Second derivatives with respect to theta = (w, b)."""
    A = np.column_stack([X, np.ones(len(X))])
    p = sigmoid(A[:, :-1] @ w + b)
    s = p * (1 - p)                           # diagonal of S
    H = (A.T * s) @ A / len(X)                # (1/n) A^T S A without building S
    H[:-1, :-1] += lam * np.eye(X.shape[1])   # the penalty does not touch b
    return H


class LogisticRegression:
    """method="gd": fixed-step gradient descent. method="newton": Newton's method."""

    def __init__(self, lam=0.0, method="gd", learning_rate=0.5, n_iters=1000):
        self.lam = lam
        self.method = method
        self.learning_rate = learning_rate
        self.n_iters = n_iters

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        w, b = np.zeros(X.shape[1]), 0.0
        self.history = [(w.copy(), b, log_loss(X, y, w, b, self.lam))]

        for _ in range(self.n_iters):
            grad_w, grad_b = gradients(X, y, w, b, self.lam)
            if self.method == "gd":
                w = w - self.learning_rate * grad_w
                b = b - self.learning_rate * grad_b
            else:
                # Newton: jump to the minimum of the local quadratic, theta -= H^-1 g
                step = np.linalg.solve(hessian(X, w, b, self.lam), np.append(grad_w, grad_b))
                w, b = w - step[:-1], b - step[-1]
            self.history.append((w.copy(), b, log_loss(X, y, w, b, self.lam)))

        self.coef_, self.intercept_ = w, b
        return self

    def decision_function(self, X):
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    def predict_proba(self, X):
        """Probability of class 1 for each row."""
        return sigmoid(self.decision_function(X))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    @property
    def losses(self):
        return np.array([loss for _, _, loss in self.history])
