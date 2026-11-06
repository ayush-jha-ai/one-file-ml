# 06 · k-means clustering

Split unlabelled data into k groups so that each point is close to its group's centre. This is the simplest clustering method and the one most others are compared against.

![Left: three clusters of points with three centres that all start inside the bottom-left cluster. Each iteration first recolours every point by its nearest centre, then moves each centre to the mean of its points; within four iterations each centre sits in its own cluster. Right: the objective falling at every step.](demo.gif)

## Intuition

Place k centres anywhere. Then repeat two steps. First, give every point to its nearest centre. Second, move every centre to the average position of the points it was given. Each step can only make the total squared distance from points to their centres smaller, so the process settles down quickly. Where it settles depends on where the centres started: a bad start can leave two centres sharing one real cluster. That is why k-means is usually run several times with spread-out starting centres (k-means++), keeping the best result.

## The maths

**Objective.** Data $\mathbf{x}_1, \dots, \mathbf{x}_n \in \mathbb{R}^d$. Choose $k$ centres $\boldsymbol\mu_1, \dots, \boldsymbol\mu_k$ and an assignment $c_i \in \{1, \dots, k\}$ for each point, to minimise the *inertia*

$$J(c, \boldsymbol\mu) = \sum_{i=1}^n \lVert\mathbf{x}_i - \boldsymbol\mu_{c_i}\rVert^2.$$

Minimising over both at once is hard: finding the global minimum is NP-hard even for $k = 2$. But minimising over either one with the other held fixed is easy.

**Assign step (centres fixed).** Each term of $J$ involves only one $c_i$, so each point can be optimised on its own:

$$c_i \leftarrow \arg\min_{j} \lVert\mathbf{x}_i - \boldsymbol\mu_j\rVert^2.$$

The set of points nearest to $\boldsymbol\mu_j$ is its *Voronoi cell*. The boundary between two cells is where $\lVert\mathbf{x} - \boldsymbol\mu_a\rVert^2 = \lVert\mathbf{x} - \boldsymbol\mu_b\rVert^2$. Expanding both sides cancels $\lVert\mathbf{x}\rVert^2$ and leaves an equation linear in $\mathbf{x}$, so cell boundaries are straight lines (planes in higher dimensions). These are the coloured regions in the animation.

**Update step (assignments fixed).** Now $J$ splits into one sum per cluster. For cluster $j$ with point set $C_j$ and $n_j$ points, minimise $g(\boldsymbol\mu) = \sum_{i \in C_j}\lVert\mathbf{x}_i - \boldsymbol\mu\rVert^2$. Its gradient is $-2\sum_{i\in C_j}(\mathbf{x}_i - \boldsymbol\mu)$. Setting that to zero gives $n_j\boldsymbol\mu = \sum_{i \in C_j}\mathbf{x}_i$, so

$$\boldsymbol\mu_j \leftarrow \frac{1}{n_j}\sum_{i \in C_j}\mathbf{x}_i,$$

the mean of the cluster. This is the minimum, not just a stationary point. Writing $\bar{\mathbf{x}}$ for the mean, the cross term $2(\bar{\mathbf{x}} - \boldsymbol\mu)^\top\sum_i(\mathbf{x}_i - \bar{\mathbf{x}})$ vanishes, which leaves

$$\sum_{i\in C_j}\lVert\mathbf{x}_i - \boldsymbol\mu\rVert^2 = \sum_{i\in C_j}\lVert\mathbf{x}_i - \bar{\mathbf{x}}\rVert^2 + n_j\lVert\bar{\mathbf{x}} - \boldsymbol\mu\rVert^2.$$

This is smallest exactly when $\boldsymbol\mu = \bar{\mathbf{x}}$.

**Why it always stops.** Each step solves its sub-problem exactly, so $J$ never increases. There are finitely many ways to assign $n$ points to $k$ clusters, $J$ is never negative, and the algorithm cannot return to an earlier assignment without $J$ having stayed the same. So it reaches a point where the assignments stop changing, in a finite number of steps. That point is a *local* minimum: no single step can improve it, but a different starting point might reach a lower $J$.

**k-means++ starting centres.** Pick the first centre uniformly at random from the data. Let $D(\mathbf{x})$ be the distance from $\mathbf{x}$ to the nearest centre chosen so far. Pick each further centre from the data with probability

$$P(\mathbf{x}_i) = \frac{D(\mathbf{x}_i)^2}{\sum_{l} D(\mathbf{x}_l)^2}.$$

Points far from the existing centres are much more likely to be picked, so the centres start spread out. Arthur and Vassilvitskii (2007) proved that the expected inertia straight after this seeding is at most $8(\ln k + 2)$ times the optimum, before Lloyd's iterations improve it further.

## Code walkthrough

| Equation | Code | Lines |
|---|---|---|
| $\lVert\mathbf{x}_i - \boldsymbol\mu_j\rVert^2$ for all $i, j$ | `((X[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)` | [14](model.py#L14) |
| $J = \sum_i\lVert\mathbf{x}_i - \boldsymbol\mu_{c_i}\rVert^2$ | `np.sum((X - centres[labels]) ** 2)` | [18](model.py#L18) |
| $D(\mathbf{x}_i)^2$ to the nearest chosen centre | `sq_distances(X, centres).min(axis=1)` | [24](model.py#L24) |
| $P(\mathbf{x}_i) = D(\mathbf{x}_i)^2/\sum_l D(\mathbf{x}_l)^2$ | `d2 / d2.sum()` | [25](model.py#L25) |
| first centre uniform, then $D^2$ sampling | `rng.integers(len(X))`, `rng.choice(len(X), p=...)` | [30-32](model.py#L30-L32) |
| assign: $c_i = \arg\min_j\lVert\mathbf{x}_i - \boldsymbol\mu_j\rVert^2$ | `np.argmin(sq_distances(X, centres), axis=1)` | [38](model.py#L38) |
| update: $\boldsymbol\mu_j = \frac{1}{n_j}\sum_{i\in C_j}\mathbf{x}_i$ | `X[labels == j].mean(axis=0)` | [45](model.py#L45) |
| empty cluster: restart at the worst-fitting point | `np.argmax(np.sum((X - new[labels]) ** 2, axis=1))` | [48](model.py#L48) |
| stop when assignments stop changing | `np.array_equal(new_labels, labels)` | [74](model.py#L74) |
| keep the best of several runs | `min(runs, key=lambda run: run[2][-1][2])` | [85](model.py#L85) |

The tests run scikit-learn's `KMeans(algorithm="lloyd")` from the same five sets of starting centres and get identical labels, centres to `1e-10`, and inertia to `1e-8` relative. They also check:
- inertia never increases, with both kinds of start;
- each half-step is optimal: no point would rather switch clusters, and nudging a centre off its mean raises $J$;
- the k-means++ probabilities equal the $D^2$ formula exactly, and the sampled frequencies match them;
- an empty cluster is refilled.

## Where it breaks

**1. Clusters that are not round blobs.** Cell boundaries are straight lines, so k-means can only cut space into convex pieces. Two interleaved half-moons cannot be separated that way. The best of ten runs agrees with the true moons on only 76% of points.

```python
import numpy as np
from model import best_of
from sklearn.datasets import make_moons

X, y = make_moons(400, noise=0.05, random_state=0)
_, labels, _ = best_of(X, 2)
agreement = max(np.mean(labels == y), np.mean(labels != y))   # cluster ids are arbitrary
print(agreement)                                               # 0.7625
assert agreement < 0.8
```

**2. Clusters of different sizes.** Inertia is a sum over points, so a wide cluster of 500 points costs far more than two tight clusters of 30. The lowest-inertia answer splits the big cluster in two and merges each small cluster into a half. Here k-means returns groups of 131, 126 and 303 points; the truth is 500, 30 and 30.

```python
import numpy as np
from model import best_of

rng = np.random.default_rng(0)
X = np.vstack([rng.normal([0, 0], 2.0, (500, 2)),     # one wide cluster
               rng.normal([5, 5], 0.3, (30, 2)),      # two small, tight ones
               rng.normal([5, -5], 0.3, (30, 2))])
centres, labels, _ = best_of(X, 3)
print(np.bincount(labels), centres.round(1))          # [131 126 303]: no centre near (5, 5)
assert np.bincount(labels).max() < 400 and not np.any(np.all(np.abs(centres - [5, 5]) < 1, axis=1))
```

**3. Local minima.** On 10 well-separated blobs, 50 runs from random starting points reach the best inertia (390) only once. The median run ends at 712 and the worst at 3,065, almost 8 times worse. k-means++ starts give a median of 413, much better but still not guaranteed. Always run several times and keep the best.

```python
import numpy as np
from model import kmeans
from sklearn.datasets import make_blobs

X, _ = make_blobs(600, centers=10, cluster_std=0.6, random_state=3)
rand = np.array([kmeans(X, 10, init="random", seed=s)[2][-1][2] for s in range(50)])
plus = np.array([kmeans(X, 10, init="k-means++", seed=s)[2][-1][2] for s in range(50)])
print(rand.min(), np.median(rand), rand.max())        # 390  712  3065
print(np.median(plus))                                # 413
assert rand.max() > 7 * rand.min() and np.median(plus) < np.median(rand)
```

**4. It always finds k clusters, even when there are none.** On points spread uniformly over a square, k = 4 returns four neat, equally sized groups. Inertia falls steadily with k whether or not real clusters exist (to 25% of the k = 1 value here, against 5% for data with four true clusters). There is no built-in signal that the clusters are not real.

```python
import numpy as np
from model import best_of
from sklearn.datasets import make_blobs

uniform = np.random.default_rng(0).uniform(size=(1000, 2))
blobs, _ = make_blobs(1000, centers=4, cluster_std=0.5, random_state=0)
for name, X in (("uniform", uniform), ("4 blobs", blobs)):
    j1, j4 = best_of(X, 1)[2][-1][2], best_of(X, 4)[2][-1][2]
    print(name, np.bincount(best_of(X, 4)[1]), round(j4 / j1, 3))
    # uniform [242 280 240 238] 0.246      4 blobs [248 250 252 250] 0.052
assert 0.2 < best_of(uniform, 4)[2][-1][2] / best_of(uniform, 1)[2][-1][2] < 0.3
```

## Common mistakes

- **"k-means finds the best clustering."** It finds a local minimum of one particular objective. Different starts give different answers (example 3), and the objective itself prefers round, equal-sized groups (examples 1 and 2).
- **"The k with the lowest inertia is the right k."** Inertia always falls as k grows, reaching 0 when every point has its own centre. Look for where the improvement levels off, or use a criterion that penalises extra clusters, and be sceptical (example 4).
- **"k-means and k-NN are related."** Only by the letter k. k-NN is supervised and its k is a number of neighbours. k-means is unsupervised and its k is a number of clusters.
- **"Cluster 0 in one run is cluster 0 in another."** The numbers are arbitrary labels. Two runs that find the same clusters can number them differently, so compare clusterings by which points are grouped together, not by label value.
- **"Scaling does not matter because there are no labels."** Distances drive every step, so a feature in large units dominates exactly as in k-NN. Standardise first unless the units are genuinely comparable.

## Further reading

- Lloyd, "Least squares quantization in PCM", *IEEE Transactions on Information Theory* 28(2) (1982), first circulated as a Bell Labs report in 1957: the alternating algorithm.
- MacQueen, "Some methods for classification and analysis of multivariate observations", *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability* (1967): the name k-means.
- Arthur and Vassilvitskii, "k-means++: the advantages of careful seeding", *Proceedings of the ACM-SIAM Symposium on Discrete Algorithms* (2007).
- Aloise, Deshpande, Hansen and Popat, "NP-hardness of Euclidean sum-of-squares clustering", *Machine Learning* 75(2) (2009).
- Bishop, *Pattern Recognition and Machine Learning* (2006), section 9.1: k-means as a limiting case of Gaussian mixtures.
