"""L2 (ridge) and L1 (lasso) regularised linear regression. NumPy only.

Both minimise the same data term plus a different penalty on the weights w:

  ridge:  (1/2n) ||y - X w - b||^2 + (alpha/2) ||w||_2^2      closed form
  lasso:  (1/2n) ||y - X w - b||^2 +  alpha    ||w||_1        coordinate descent

The intercept b is never penalised. Centring X and y removes it from the problem;
it is recovered at the end as b = mean(y) - mean(X) @ w.
scikit-learn equivalents: Ridge(alpha=n * alpha), Lasso(alpha=alpha).
"""

import numpy as np


def centre(X, y):
    """Subtract column means. Returns centred data and the means."""
    x_mean, y_mean = X.mean(axis=0), y.mean()
    return X - x_mean, y - y_mean, x_mean, y_mean


def ridge(X, y, alpha):
    """Solve (Xc^T Xc / n + alpha I) w = Xc^T yc / n, the zero-gradient condition."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    Xc, yc, x_mean, y_mean = centre(X, y)
    n, d = Xc.shape
    w = np.linalg.solve(Xc.T @ Xc / n + alpha * np.eye(d), Xc.T @ yc / n)
    return w, y_mean - x_mean @ w


def soft_threshold(rho, alpha):
    """S(rho, alpha) = sign(rho) * max(|rho| - alpha, 0): shrink towards 0, stop at 0."""
    return np.sign(rho) * np.maximum(np.abs(rho) - alpha, 0.0)


def lasso(X, y, alpha, n_sweeps=10_000, tol=1e-12, w0=None):
    """Cyclic coordinate descent: minimise exactly over one weight at a time."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    Xc, yc, x_mean, y_mean = centre(X, y)
    n, d = Xc.shape
    z = (Xc**2).sum(axis=0) / n                 # z_j = x_j^T x_j / n
    w = np.zeros(d) if w0 is None else np.array(w0, float)
    r = yc - Xc @ w                             # residual, kept up to date

    for _ in range(n_sweeps):
        largest_change = 0.0
        for j in range(d):
            if z[j] == 0:                       # constant column: weight stays 0
                continue
            # rho_j = x_j^T (residual with feature j's contribution added back) / n
            rho = Xc[:, j] @ r / n + z[j] * w[j]
            w_new = soft_threshold(rho, alpha) / z[j]
            r -= Xc[:, j] * (w_new - w[j])      # update the residual in O(n)
            largest_change = max(largest_change, abs(w_new - w[j]))
            w[j] = w_new
        if largest_change < tol:
            break
    return w, y_mean - x_mean @ w


def lasso_alpha_max(X, y):
    """Smallest alpha at which every lasso weight is exactly zero: max_j |x_j^T yc| / n."""
    Xc, yc, _, _ = centre(np.asarray(X, float), np.asarray(y, float))
    return np.abs(Xc.T @ yc).max() / len(yc)


def objective(X, y, w, b, alpha, penalty):
    """The quantity each solver minimises, for checking optimality."""
    data_term = np.sum((y - X @ w - b) ** 2) / (2 * len(y))
    if penalty == "l1":
        return data_term + alpha * np.abs(w).sum()
    return data_term + 0.5 * alpha * w @ w


def path(X, y, alphas, penalty):
    """Weights for each alpha. Lasso warm-starts from the previous solution."""
    weights, w = [], None
    for alpha in alphas:
        w, _ = lasso(X, y, alpha, w0=w) if penalty == "l1" else ridge(X, y, alpha)
        weights.append(w.copy())
    return np.array(weights)
