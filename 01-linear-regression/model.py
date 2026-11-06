"""Linear regression trained by batch gradient descent. NumPy only.

Model:     y_hat = X w + b
Loss:      L(w, b) = (1/n) * sum_i (y_hat_i - y_i)^2          (mean squared error)
Gradients: dL/dw = (2/n) X^T (y_hat - y),   dL/db = (2/n) sum_i (y_hat_i - y_i)
Update:    w <- w - lr * dL/dw,             b <- b - lr * dL/db
"""

import numpy as np


def mse(y, y_hat):
    """Mean squared error: the average of the squared residuals."""
    return np.mean((y_hat - y) ** 2)


def gradients(X, y, w, b):
    """Gradient of the MSE with respect to the weights w and the intercept b."""
    n = len(y)
    residual = X @ w + b - y              # e = y_hat - y, shape (n,)
    grad_w = (2 / n) * X.T @ residual     # dL/dw = (2/n) X^T e, shape (d,)
    grad_b = (2 / n) * residual.sum()     # dL/db = (2/n) sum(e)
    return grad_w, grad_b


def add_bias_column(X):
    """[X | 1]: folds the intercept into the weights, so theta = (w, b)."""
    return np.column_stack([X, np.ones(len(X))])


def normal_equation(X, y):
    """Closed-form minimiser: solve (A^T A) theta = A^T y with A = [X | 1]."""
    A = add_bias_column(X)
    theta = np.linalg.solve(A.T @ A, A.T @ y)
    return theta[:-1], theta[-1]


def max_stable_learning_rate(X):
    """Gradient descent on this loss converges if and only if lr < 2 / lambda_max(H).

    H = (2/n) A^T A is the Hessian of the loss. It is constant because the loss
    is quadratic, so this bound is exact rather than a rule of thumb.
    """
    A = add_bias_column(X)
    H = (2 / len(X)) * A.T @ A
    return 2 / np.linalg.eigvalsh(H).max()


class LinearRegression:
    """Fits w and b by repeating the gradient descent update n_iters times."""

    def __init__(self, learning_rate=0.1, n_iters=1000, w0=None, b0=0.0):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.w0 = w0          # starting weights; None means all zeros
        self.b0 = b0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        w = np.zeros(X.shape[1]) if self.w0 is None else np.array(self.w0, dtype=float)
        b = float(self.b0)

        # history[k] holds (w, b, loss) after k updates; history[0] is the start.
        self.history = [(w.copy(), b, mse(y, X @ w + b))]
        for _ in range(self.n_iters):
            grad_w, grad_b = gradients(X, y, w, b)
            w = w - self.learning_rate * grad_w
            b = b - self.learning_rate * grad_b
            self.history.append((w.copy(), b, mse(y, X @ w + b)))

        self.coef_, self.intercept_ = w, b
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    @property
    def losses(self):
        return np.array([loss for _, _, loss in self.history])
