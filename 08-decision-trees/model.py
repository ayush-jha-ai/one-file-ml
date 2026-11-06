"""CART decision trees for classification (Gini or entropy) and regression (squared error).

Grow greedily: at each node, try every feature and every threshold between two
consecutive distinct values, keep the split with the lowest weighted child impurity,
and recurse until a stopping rule fires. Leaves predict class proportions or the mean.
"""

import numpy as np


def impurity(counts, criterion):
    """Impurity of nodes from class counts, shape (..., n_classes)."""
    n = counts.sum(axis=-1, keepdims=True)
    p = counts / np.maximum(n, 1)
    if criterion == "gini":
        return 1 - np.sum(p**2, axis=-1)                      # chance two draws disagree
    logp = np.log2(np.where(p > 0, p, 1))                     # 0 log 0 is taken as 0
    return -np.sum(p * logp, axis=-1)                         # entropy, in bits


def best_split(X, y, criterion, n_classes, min_leaf):
    """Return (feature, threshold, weighted child impurity) of the best split, or None."""
    n, best = len(y), None
    for j in range(X.shape[1]):
        order = np.argsort(X[:, j], kind="stable")
        x, ys = X[order, j], y[order]
        # Candidate split after position i means: left = first i+1 points, right = the rest.
        if criterion == "squared_error":
            s, s2 = np.cumsum(ys), np.cumsum(ys**2)
            n_left = np.arange(1, n + 1)
            # sum (y - mean)^2 = sum y^2 - (sum y)^2 / m, for every left child at once
            sse_left = s2 - s**2 / n_left
            n_right = n - n_left
            sse_right = (s2[-1] - s2) - (s[-1] - s) ** 2 / np.maximum(n_right, 1)
            score = (sse_left + sse_right) / n                # weighted mean of child MSEs
        else:
            left = np.cumsum(np.eye(n_classes)[ys], axis=0)   # class counts on the left
            right = left[-1] - left
            n_left = np.arange(1, n + 1)
            score = (n_left * impurity(left, criterion)
                     + (n - n_left) * impurity(right, criterion)) / n
        # Only split between distinct values, and keep at least min_leaf points per side.
        valid = (x[:-1] < x[1:]) & (n_left[:-1] >= min_leaf) & (n - n_left[:-1] >= min_leaf)
        if not np.any(valid):
            continue
        i = np.flatnonzero(valid)[np.argmin(score[:-1][valid])]
        if best is None or score[i] < best[2]:
            best = (j, (x[i] + x[i + 1]) / 2, score[i])
    return best


class DecisionTree:
    """criterion: "gini", "entropy" (classification) or "squared_error" (regression)."""

    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.criterion = criterion
        self.max_depth = max_depth if max_depth is not None else np.inf
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.regression = self.criterion == "squared_error"
        if self.regression:
            y = np.asarray(y, dtype=float)
        else:
            self.classes_, y = np.unique(y, return_inverse=True)
        self.n_features = X.shape[1]
        self.importances = np.zeros(self.n_features)
        self.n_total = len(y)
        self.root = self._grow(X, y, depth=0)
        total = self.importances.sum()
        self.feature_importances_ = self.importances / total if total > 0 else self.importances
        return self

    def _node_impurity(self, y):
        if self.regression:
            return np.var(y)
        return impurity(np.bincount(y, minlength=len(self.classes_)), self.criterion)

    def _grow(self, X, y, depth):
        if self.regression:
            value = np.array([y.mean()])
        else:
            value = np.bincount(y, minlength=len(self.classes_)) / len(y)
        node = {"value": value, "n": len(y)}
        node_imp = self._node_impurity(y)
        if depth >= self.max_depth or len(y) < self.min_samples_split or node_imp <= 1e-12:
            return node                                        # leaf
        split = best_split(X, y, self.criterion, None if self.regression
                           else len(self.classes_), self.min_samples_leaf)
        if split is None:
            return node
        j, threshold, child_imp = split
        # Importance: impurity decrease, weighted by the share of all samples at this node.
        self.importances[j] += len(y) / self.n_total * (node_imp - child_imp)
        go_left = X[:, j] <= threshold
        node.update(feature=j, threshold=threshold,
                    left=self._grow(X[go_left], y[go_left], depth + 1),
                    right=self._grow(X[~go_left], y[~go_left], depth + 1))
        return node

    def _leaf(self, x):
        node = self.root
        while "feature" in node:
            node = node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
        return node["value"]

    def predict_proba(self, X):
        return np.array([self._leaf(x) for x in np.asarray(X, dtype=float)])

    def predict(self, X):
        out = self.predict_proba(X)
        return out[:, 0] if self.regression else self.classes_[np.argmax(out, axis=1)]

    def depth(self, node=None):
        node = node or self.root
        return 0 if "feature" not in node else 1 + max(self.depth(node["left"]),
                                                       self.depth(node["right"]))

    def n_leaves(self, node=None):
        node = node or self.root
        return 1 if "feature" not in node else self.n_leaves(node["left"]) + self.n_leaves(
            node["right"])
