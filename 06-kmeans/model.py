"""k-means clustering: Lloyd's algorithm with k-means++ initialisation. NumPy only.

Objective (inertia):  J = sum_i || x_i - mu_{c_i} ||^2,  c_i = cluster of point i
Lloyd's algorithm alternates two steps, each of which can only lower J:
  assign:  c_i  <- argmin_j || x_i - mu_j ||^2           (best cluster for each point)
  update:  mu_j <- mean of the points with c_i = j        (best centre for each cluster)
"""

import numpy as np


def sq_distances(X, centres):
    """||x_i - mu_j||^2 for every point and centre, shape (n, k)."""
    return ((X[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)


def inertia(X, centres, labels):
    return np.sum((X - centres[labels]) ** 2)


def kmeans_plus_plus_probabilities(X, centres):
    """D^2 weighting: probability of picking x_i is D(x_i)^2 / sum_j D(x_j)^2,
    where D(x) is the distance from x to its nearest centre chosen so far."""
    d2 = sq_distances(X, centres).min(axis=1)
    return d2 / d2.sum()


def kmeans_plus_plus(X, k, rng):
    """First centre uniformly at random; each further centre by D^2 sampling."""
    centres = X[[rng.integers(len(X))]]
    for _ in range(k - 1):
        i = rng.choice(len(X), p=kmeans_plus_plus_probabilities(X, centres))
        centres = np.vstack([centres, X[i]])
    return centres


def assign(X, centres):
    return np.argmin(sq_distances(X, centres), axis=1)


def update(X, labels, centres):
    """Each centre moves to the mean of its points. An empty cluster takes the point
    currently farthest from its own centre, so k clusters always survive."""
    labels = labels.copy()                          # the refill below must not leak out
    new = np.array([X[labels == j].mean(axis=0) if np.any(labels == j) else centres[j]
                    for j in range(len(centres))])
    for j in np.flatnonzero(np.bincount(labels, minlength=len(centres)) == 0):
        far = np.argmax(np.sum((X - new[labels]) ** 2, axis=1))
        new[j], labels[far] = X[far], j
    return new


def kmeans(X, k, init="k-means++", seed=0, max_iter=300, tol=0.0):
    """Returns (centres, labels, history). history[t] = (centres, labels, inertia).

    init: "k-means++", "random" (k distinct data points), or an array of starting centres.
    Stops when the assignments stop changing or the centres move less than tol.
    """
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    if isinstance(init, str) and init == "random":
        centres = X[rng.choice(len(X), k, replace=False)]
    elif isinstance(init, str):
        centres = kmeans_plus_plus(X, k, rng)
    else:
        centres = np.array(init, dtype=float)

    labels = assign(X, centres)
    history = [(centres.copy(), labels.copy(), inertia(X, centres, labels))]
    for _ in range(max_iter):
        new_centres = update(X, labels, centres)
        new_labels = assign(X, new_centres)
        shift = np.sum((new_centres - centres) ** 2)
        centres, converged = new_centres, np.array_equal(new_labels, labels) or shift <= tol
        labels = new_labels
        history.append((centres.copy(), labels.copy(), inertia(X, centres, labels)))
        if converged:
            break
    return centres, labels, history


def best_of(X, k, n_init=10, init="k-means++", seed=0):
    """Run k-means n_init times from different seeds and keep the lowest inertia."""
    runs = [kmeans(X, k, init=init, seed=seed + r) for r in range(n_init)]
    return min(runs, key=lambda run: run[2][-1][2])
