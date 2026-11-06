"""Principal component analysis via the SVD (and, for comparison, the covariance matrix).

Centre the data:  Xc = X - mean.  Sample covariance:  C = Xc^T Xc / (n - 1).
Principal components = eigenvectors of C, ordered by eigenvalue (variance along them).
The SVD Xc = U S V^T gives them directly: rows of V^T, with variances S^2 / (n - 1).
"""

import numpy as np


def fix_signs(components):
    """An eigenvector times -1 is still an eigenvector. Make each component's largest
    absolute entry positive, so results are reproducible."""
    largest = np.argmax(np.abs(components), axis=1)
    signs = np.sign(components[np.arange(len(components)), largest])
    return components * signs[:, None]


class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        k = self.n_components or min(n, d)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        # Economy SVD: U is n x r, S has r values (largest first), Vt is r x d, r = min(n, d).
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.components_ = fix_signs(Vt[:k])                    # (k, d), orthonormal rows
        variances = S**2 / (n - 1)                              # eigenvalues of C
        self.explained_variance_ = variances[:k]
        self.explained_variance_ratio_ = variances[:k] / variances.sum()
        return self

    def transform(self, X):
        """Coordinates along each component: z = V_k^T (x - mean)."""
        return (np.asarray(X, dtype=float) - self.mean_) @ self.components_.T

    def inverse_transform(self, Z):
        """Back to the original space: x_hat = mean + V_k z."""
        return Z @ self.components_ + self.mean_

    def reconstruction_error(self, X):
        """Total squared distance from each point to its projection."""
        X = np.asarray(X, dtype=float)
        return np.sum((X - self.inverse_transform(self.transform(X))) ** 2)


def pca_by_covariance(X, k):
    """Same answer from the eigendecomposition of C, as a cross-check of the SVD route."""
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0)
    C = Xc.T @ Xc / (len(X) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(C)              # ascending order
    order = np.argsort(eigenvalues)[::-1][:k]
    return fix_signs(eigenvectors[:, order].T), eigenvalues[order]


def variance_along(X, direction):
    """Sample variance of the data projected onto a unit vector u: u^T C u."""
    u = np.asarray(direction, dtype=float)
    u = u / np.linalg.norm(u)
    z = (X - X.mean(axis=0)) @ u
    return z @ z / (len(X) - 1)
