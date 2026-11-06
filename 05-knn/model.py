"""k-nearest neighbours for classification and regression. NumPy only.

Training stores the data. To predict at a query point x:
  1. compute the distance from x to every training point
  2. take the k closest
  3. classification: majority vote of their labels; regression: mean of their targets
"""

import numpy as np


def pairwise_sq_distances(A, B):
    """||a_i - b_j||^2 for every pair, using ||a||^2 + ||b||^2 - 2 a.b (one matrix product).

    Rounding can make the expansion slightly negative for (near-)identical points,
    so clip at 0.
    """
    sq = (A**2).sum(axis=1)[:, None] + (B**2).sum(axis=1)[None, :] - 2 * A @ B.T
    return np.maximum(sq, 0.0)


class KNN:
    """task="classify" (majority vote) or task="regress" (mean of neighbours)."""

    def __init__(self, k=5, task="classify"):
        self.k = k
        self.task = task

    def fit(self, X, y):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y)
        if self.task == "classify":
            # Store labels as 0..C-1 so votes can be counted with bincount.
            self.classes_, self.y_index = np.unique(self.y, return_inverse=True)
        return self

    def kneighbors(self, X, exclude_self=False):
        """Indices and distances of the k nearest training points, nearest first.

        Ties in distance are broken by training-set order (stable sort).
        exclude_self=True assumes X is the training set and skips each point itself,
        which gives leave-one-out predictions.
        """
        d2 = pairwise_sq_distances(np.asarray(X, dtype=float), self.X)
        if exclude_self:
            np.fill_diagonal(d2, np.inf)
        idx = np.argsort(d2, axis=1, kind="stable")[:, : self.k]
        return idx, np.sqrt(np.take_along_axis(d2, idx, axis=1))

    def predict_proba(self, X, exclude_self=False):
        """Fraction of the k neighbours in each class, shape (n_queries, n_classes)."""
        idx, _ = self.kneighbors(X, exclude_self)
        votes = self.y_index[idx]                              # (n_queries, k)
        n_classes = len(self.classes_)
        counts = np.stack([np.bincount(row, minlength=n_classes) for row in votes])
        return counts / self.k

    def predict(self, X, exclude_self=False):
        if self.task == "regress":
            idx, _ = self.kneighbors(X, exclude_self)
            return self.y[idx].mean(axis=1)
        # argmax returns the first maximum, so a tied vote goes to the smallest label.
        return self.classes_[np.argmax(self.predict_proba(X, exclude_self), axis=1)]


def leave_one_out_error(X, y, k, task="classify"):
    """Predict each training point from the other n-1, and average the error."""
    model = KNN(k, task).fit(X, y)
    pred = model.predict(X, exclude_self=True)
    if task == "regress":
        return np.mean((pred - y) ** 2)
    return np.mean(pred != y)
