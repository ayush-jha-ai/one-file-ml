"""Random forest classifier: bagged CART trees with a random feature subset at every split.

For b = 1..B:
  draw a bootstrap sample (n rows with replacement) and grow a deep tree on it, where each
  split may only consider m randomly chosen features (m = sqrt(d) by default).
Predict by averaging the trees' class probabilities.
Rows not drawn for tree b are "out of bag" for it and give a free validation estimate.
NumPy only. The tree is a compact Gini version of 08-decision-trees.
"""

import numpy as np


def gini(counts):
    p = counts / np.maximum(counts.sum(axis=-1, keepdims=True), 1)
    return 1 - np.sum(p**2, axis=-1)


def best_split(X, y, n_classes, features, min_leaf):
    """Lowest weighted Gini over the given features. Returns (feature, threshold, score)."""
    n, best = len(y), None
    for j in features:
        order = np.argsort(X[:, j], kind="stable")
        x = X[order, j]
        left = np.cumsum(np.eye(n_classes)[y[order]], axis=0)  # class counts left of each cut
        n_left = np.arange(1, n + 1)
        score = (n_left * gini(left) + (n - n_left) * gini(left[-1] - left)) / n
        valid = (x[:-1] < x[1:]) & (n_left[:-1] >= min_leaf) & (n - n_left[:-1] >= min_leaf)
        if np.any(valid):
            i = np.flatnonzero(valid)[np.argmin(score[:-1][valid])]
            if best is None or score[i] < best[2]:
                best = (j, (x[i] + x[i + 1]) / 2, score[i])
    return best


def grow(X, y, n_classes, depth, settings, rng):
    counts = np.bincount(y, minlength=n_classes)
    node = {"value": counts / len(y)}
    max_depth, max_features, min_leaf = settings
    if depth >= max_depth or gini(counts) <= 1e-12 or len(y) < 2:
        return node
    # The random-forest twist: each split looks at a fresh random subset of features.
    features = rng.choice(X.shape[1], size=max_features, replace=False)
    split = best_split(X, y, n_classes, features, min_leaf)
    if split is None:
        return node
    j, threshold, _ = split
    go_left = X[:, j] <= threshold
    node.update(feature=j, threshold=threshold,
                left=grow(X[go_left], y[go_left], n_classes, depth + 1, settings, rng),
                right=grow(X[~go_left], y[~go_left], n_classes, depth + 1, settings, rng))
    return node


def predict_tree(node, X):
    """Class probabilities for every row, routing whole groups of rows down at once."""
    if "feature" not in node:
        return np.tile(node["value"], (len(X), 1))
    go_left = X[:, node["feature"]] <= node["threshold"]
    out = np.empty((len(X), len(node["value"])))
    out[go_left] = predict_tree(node["left"], X[go_left])
    out[~go_left] = predict_tree(node["right"], X[~go_left])
    return out


class RandomForest:
    def __init__(self, n_trees=100, max_features="sqrt", max_depth=None, bootstrap=True,
                 min_samples_leaf=1, seed=0):
        self.n_trees = n_trees
        self.max_features = max_features
        self.max_depth = np.inf if max_depth is None else max_depth
        self.bootstrap = bootstrap
        self.min_samples_leaf = min_samples_leaf
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_, y = np.unique(y, return_inverse=True)
        n, d = X.shape
        m = {"sqrt": int(np.sqrt(d)), None: d}.get(self.max_features, self.max_features)
        settings = (self.max_depth, max(1, m), self.min_samples_leaf)
        rng = np.random.default_rng(self.seed)
        self.trees, self.in_bag = [], []
        for _ in range(self.n_trees):
            rows = rng.integers(0, n, n) if self.bootstrap else np.arange(n)
            self.trees.append(grow(X[rows], y[rows], len(self.classes_), 0, settings, rng))
            self.in_bag.append(np.bincount(rows, minlength=n) > 0)
        self.in_bag = np.array(self.in_bag)                     # (n_trees, n)
        self.X_train, self.y_train = X, y
        return self

    def predict_proba(self, X, n_trees=None):
        """Average of the first n_trees trees' probabilities (all trees by default)."""
        X = np.asarray(X, dtype=float)
        trees = self.trees[: n_trees or len(self.trees)]
        return np.mean([predict_tree(t, X) for t in trees], axis=0)

    def predict(self, X, n_trees=None):
        return self.classes_[np.argmax(self.predict_proba(X, n_trees), axis=1)]

    def oob_score(self):
        """Accuracy on each training row using only the trees that never saw it."""
        votes = np.zeros((len(self.y_train), len(self.classes_)))
        for tree, seen in zip(self.trees, self.in_bag, strict=True):
            votes[~seen] += predict_tree(tree, self.X_train[~seen])
        has_vote = votes.sum(axis=1) > 0                       # rows that were OOB at least once
        return np.mean(np.argmax(votes[has_vote], axis=1) == self.y_train[has_vote])
