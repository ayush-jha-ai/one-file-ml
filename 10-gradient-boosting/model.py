"""Gradient boosting with regression trees. NumPy only.

F_0(x) = best constant.  For m = 1..M:
  r_i = -dL(y_i, F)/dF at F = F_{m-1}(x_i)          (pseudo-residuals: the downhill direction)
  fit a small regression tree to r                  (its leaves split the data into regions)
  in each leaf R_j pick the value gamma_j that best lowers the loss there
  F_m(x) = F_{m-1}(x) + learning_rate * gamma_{leaf(x)}

squared_error: L = (y - F)^2 / 2,  r = y - F,      gamma = mean of r in the leaf
log_loss:      L = log(1 + e^F) - y F (y in {0,1}), r = y - sigmoid(F),
               gamma = sum r / sum p (1 - p)        (one Newton step in the leaf)
"""

import numpy as np


def sigmoid(z):
    return 0.5 * (1 + np.tanh(0.5 * z))       # equals 1 / (1 + e^-z), never overflows


def best_split(X, r, min_leaf):
    """Split minimising the children's total squared error of r. Returns (j, threshold)."""
    n, best, best_sse = len(r), None, np.inf
    for j in range(X.shape[1]):
        order = np.argsort(X[:, j], kind="stable")
        x, rs = X[order, j], r[order]
        s, s2 = np.cumsum(rs), np.cumsum(rs**2)
        n_left = np.arange(1, n + 1)
        sse = (s2 - s**2 / n_left) + ((s2[-1] - s2) - (s[-1] - s) ** 2 / np.maximum(n - n_left, 1))
        valid = (x[:-1] < x[1:]) & (n_left[:-1] >= min_leaf) & (n - n_left[:-1] >= min_leaf)
        if np.any(valid):
            i = np.flatnonzero(valid)[np.argmin(sse[:-1][valid])]
            if sse[i] < best_sse:
                best, best_sse = (j, (x[i] + x[i + 1]) / 2), sse[i]
    return best


def grow(X, r, rows, depth, max_depth, min_leaf, leaves):
    """Regression tree on residuals r. Leaves are numbered in `leaves` (lists of row indices)."""
    split = None
    if depth < max_depth and len(rows) >= 2 and np.ptp(r[rows]) > 0:
        split = best_split(X[rows], r[rows], min_leaf)
    if split is None:
        leaves.append(rows)
        return {"leaf": len(leaves) - 1}
    j, t = split
    go_left = X[rows, j] <= t
    return {"feature": j, "threshold": t,
            "left": grow(X, r, rows[go_left], depth + 1, max_depth, min_leaf, leaves),
            "right": grow(X, r, rows[~go_left], depth + 1, max_depth, min_leaf, leaves)}


def apply(node, X):
    """Leaf number for every row of X."""
    if "leaf" in node:
        return np.full(len(X), node["leaf"])
    go_left = X[:, node["feature"]] <= node["threshold"]
    out = np.empty(len(X), dtype=int)
    out[go_left] = apply(node["left"], X[go_left])
    out[~go_left] = apply(node["right"], X[~go_left])
    return out


class GradientBoosting:
    def __init__(self, loss="squared_error", n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_leaf=1):
        self.loss = loss
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        if self.loss == "squared_error":
            self.init_ = y.mean()                                 # argmin_c sum (y - c)^2
        else:
            self.init_ = np.log(y.mean() / (1 - y.mean()))        # log-odds of the base rate
        F = np.full(len(y), self.init_)
        self.trees = []
        for _ in range(self.n_estimators):
            p = sigmoid(F) if self.loss == "log_loss" else None
            r = y - (p if p is not None else F)                   # negative gradient
            leaves = []
            tree = grow(X, r, np.arange(len(y)), 0, self.max_depth, self.min_samples_leaf, leaves)
            gamma = np.empty(len(leaves))
            for k, rows in enumerate(leaves):
                if self.loss == "squared_error":
                    gamma[k] = r[rows].mean()
                else:
                    hess = np.sum(p[rows] * (1 - p[rows]))        # second derivative of the loss
                    gamma[k] = r[rows].sum() / hess if hess > 1e-150 else 0.0
            F = F + self.learning_rate * gamma[apply(tree, X)]
            self.trees.append((tree, gamma))
        return self

    def staged_decision_function(self, X):
        """F_m(x) after each round m = 1..M, as a generator."""
        X = np.asarray(X, dtype=float)
        F = np.full(len(X), self.init_)
        for tree, gamma in self.trees:
            F = F + self.learning_rate * gamma[apply(tree, X)]
            yield F

    def decision_function(self, X):
        *_, F = self.staged_decision_function(X)
        return F

    def predict(self, X):
        F = self.decision_function(X)
        return F if self.loss == "squared_error" else (F > 0).astype(int)

    def predict_proba(self, X):
        p = sigmoid(self.decision_function(X))
        return np.column_stack([1 - p, p])
