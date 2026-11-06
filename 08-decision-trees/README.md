# 08 · Decision trees

Predict by asking a sequence of yes/no questions about the features ("is income above 40,000?"), each chosen to make the groups it creates as pure as possible. Trees handle mixed feature types without scaling and are easy to read, and they are the building block of random forests and gradient boosting.

![Left: two interleaved classes, with the plane coloured by the tree's prediction. At depth 0 everything is one colour; each extra level adds horizontal and vertical cuts, until thin slivers surround single points. Right: training accuracy climbs to 99% while accuracy on new data peaks at 88% at depth 2 and falls to 82%.](demo.gif)

## Intuition

Start with all the training points in one group. Look at every feature and every place you could cut it, and pick the cut that leaves the two resulting groups most uniform in their labels. Then repeat inside each group, and keep going. Each cut is a question, and following the answers from the top leads to a final group (a *leaf*) whose majority label is the prediction. Grown fully, a tree keeps cutting until every leaf is pure. It then reproduces the training data perfectly, including its noise, so the depth must be limited.

## The maths

**Impurity of a node.** A node holds $m$ training points. Let $p_c$ be the fraction of them in class $c$. Two standard measures of how mixed the node is:

$$\text{Gini: } G = 1 - \sum_c p_c^2, \qquad \text{Entropy: } H = -\sum_c p_c\log_2 p_c.$$

Both are 0 for a pure node and largest when the classes are equally mixed. Gini has a direct meaning. Draw two points from the node at random, with replacement. They have the same class with probability $\sum_c p_c^2$, so $G$ is the probability that they disagree. In entropy, $0\log 0$ is taken as 0, its limit.

For regression the impurity is the variance of the targets in the node, $V = \tfrac1m\sum_i (y_i - \bar{y})^2$.

**Scoring a split.** A split sends the $m$ points into a left child ($m_L$ points, impurity $I_L$) and a right child ($m_R$ points, impurity $I_R$). Its score is the weighted average impurity of the children:

$$\text{score} = \frac{m_L}{m}I_L + \frac{m_R}{m}I_R.$$

The tree picks the split with the lowest score, which is the same as the largest *impurity decrease* $I - \text{score}$. For a numeric feature, the only cuts that make a difference lie between two consecutive distinct values, and the threshold is placed at their midpoint.

**Trying every cut quickly.** Sort the node's points by feature $j$. A cut after the $i$-th sorted point puts points $1, \dots, i$ on the left. The class counts on the left for every $i$ at once are a cumulative sum of one-hot label vectors. The right counts are the total minus the left. So all $m - 1$ cuts on a feature cost one sort and one pass.

For regression, use the identity

$$\sum_{i=1}^{m}(y_i - \bar{y})^2 = \sum_i y_i^2 - 2\bar{y}\sum_i y_i + m\bar{y}^2 = \sum_i y_i^2 - \frac{\left(\sum_i y_i\right)^2}{m},$$

which follows from $\bar{y} = \tfrac1m\sum_i y_i$. Cumulative sums of $y$ and $y^2$ then give the squared error of every possible left and right child in one pass.

**Leaves.** A node becomes a leaf when it is pure, when it reaches the maximum depth, or when it has too few points to split. It predicts the class proportions of its training points, or their mean for regression. These are the values that minimise the loss within the leaf: the mean minimises squared error, and the proportions minimise cross-entropy.

**Feature importance.** Every split on feature $j$ reduces impurity by $I - \text{score}$ at a node holding a fraction $m/n$ of all training points. The importance of $j$ is the sum of $\tfrac{m}{n}(I - \text{score})$ over all its splits, normalised so the importances add to 1.

**Greedy, not optimal.** Each split is chosen for its immediate improvement only. Finding the smallest tree that fits the data is NP-complete (Hyafil and Rivest, 1976), so every practical algorithm is greedy like this one. Example 3 shows what that costs.

## Code walkthrough

| Equation | Code | Lines |
|---|---|---|
| $G = 1 - \sum_c p_c^2$ | `1 - np.sum(p**2, axis=-1)` | [16](model.py#L16) |
| $H = -\sum_c p_c\log_2 p_c$, with $0\log 0 = 0$ | `np.log2(np.where(p > 0, p, 1))`, then `-np.sum(p * logp, ...)` | [17-18](model.py#L17-L18) |
| $\sum y^2 - (\sum y)^2/m$ for every left child, then every right child | `s2 - s**2 / n_left` and the line after | [32-34](model.py#L32-L34) |
| regression score: total squared error over $m$ | `(sse_left + sse_right) / n` | [35](model.py#L35) |
| left class counts for every cut (cumulative one-hot) | `np.cumsum(np.eye(n_classes)[ys], axis=0)` | [37](model.py#L37) |
| right counts = total − left | `left[-1] - left` | [38](model.py#L38) |
| $\frac{m_L}{m}I_L + \frac{m_R}{m}I_R$ | `(n_left * impurity(left, criterion)` and `(n - n_left) * impurity(right, criterion)) / n` | [40-41](model.py#L40-L41) |
| cut only between distinct values, with enough points each side | `(x[:-1] < x[1:]) & ...` | [43](model.py#L43) |
| threshold at the midpoint | `(x[i] + x[i + 1]) / 2` | [48](model.py#L48) |
| regression node impurity $V$ | `np.var(y)` | [78](model.py#L78) |
| leaf value: mean, or class proportions | `y.mean()`, `np.bincount(...) / len(y)` | [83-85](model.py#L83-L85) |
| stop: depth, size, or purity | `depth >= self.max_depth or ... or node_imp <= 1e-12` | [88](model.py#L88) |
| importance $\mathrel{+}= \frac{m}{n}(I - \text{score})$ | `len(y) / self.n_total * (node_imp - child_imp)` | [96](model.py#L96) |

The tests compare against scikit-learn's `DecisionTreeClassifier` and `DecisionTreeRegressor`, with one subtlety handled openly. When two splits are exactly tied, scikit-learn picks one by visiting features in a random order, and it stores data as 32-bit floats. So the tests:
- **Shallow trees (depth 1 to 4), tie-free data:** require identical predictions, identical probabilities, the same number of leaves, and importances to `1e-12`. Regression is checked to `1e-10`.
- **Full-depth trees:** walk both trees together on two datasets and require that every node where they choose different splits is an exact tie (equal score to `1e-12`). On the breast cancer data, the first such node splits on "worst texture" in one tree and "mean texture" in the other, with identical partitions.
- **Brute force:** check the chosen split against an exhaustive search.

## Where it breaks

**1. Grown fully, a tree memorises noise.** On noisy moons, a full tree is 100% accurate on its 300 training points and 82% on new points. Depth 2 gets 88% on new points. Two full trees trained on separate halves of the data disagree on 9% of the plane.

```python
import numpy as np
from model import DecisionTree
from sklearn.datasets import make_moons

X, y = make_moons(600, noise=0.35, random_state=0)
full = DecisionTree("gini").fit(X[:300], y[:300])
shallow = DecisionTree("gini", max_depth=2).fit(X[:300], y[:300])
for tree in (full, shallow):
    print(np.mean(tree.predict(X[:300]) == y[:300]), np.mean(tree.predict(X[300:]) == y[300:]))
# full: 1.0 0.817    depth 2: 0.873 0.88
assert np.mean(full.predict(X[:300]) == y[:300]) == 1.0
assert np.mean(full.predict(X[300:]) == y[300:]) < np.mean(shallow.predict(X[300:]) == y[300:])
```

**2. Diagonal boundaries need a staircase.** Every split is parallel to an axis. The rule "class 1 if $x_1 > x_2$" is one straight line, but a tree has to approximate it with steps: 4 leaves give 88% accuracy, and even the full tree with 43 leaves reaches only 98%. A model with a single linear boundary gets 100%.

```python
import numpy as np
from model import DecisionTree

rng = np.random.default_rng(0)
X, X_new = rng.uniform(size=(2000, 2)), rng.uniform(size=(5000, 2))
y, y_new = (X[:, 0] > X[:, 1]).astype(int), (X_new[:, 0] > X_new[:, 1]).astype(int)
for depth in (2, None):
    tree = DecisionTree("gini", max_depth=depth).fit(X, y)
    print(tree.n_leaves(), np.mean(tree.predict(X_new) == y_new))   # 4 0.882   43 0.982
assert tree.n_leaves() > 40 and np.mean(tree.predict(X_new) == y_new) < 0.99
```

**3. Greedy splitting is blind to interactions.** In XOR the label is 1 when exactly one of $x_1, x_2$ is positive. A depth-2 tree that cuts both at 0 is 100% accurate. But neither cut helps *on its own*: each half still holds 50% of each class. So the first cut is chosen by sampling noise, and it usually lands somewhere useless. Across 100 random training sets of 400 points, a depth-2 tree solves XOR only 8 times (median accuracy 55%). With four irrelevant features added, 3 times.

```python
import numpy as np
from model import DecisionTree


def xor(A):
    return ((A[:, 0] > 0) ^ (A[:, 1] > 0)).astype(int)


for n_features in (2, 6):                                # 2 relevant (+ 4 irrelevant)
    acc = []
    for seed in range(100):
        rng = np.random.default_rng(seed)
        X, X_new = rng.uniform(-1, 1, (400, n_features)), rng.uniform(-1, 1, (4000, n_features))
        tree = DecisionTree("gini", max_depth=2).fit(X, xor(X))
        acc.append(np.mean(tree.predict(X_new) == xor(X_new)))
    print(n_features, np.mean(np.array(acc) > 0.9), np.median(acc))   # 2: 0.08 0.55   6: 0.03 0.50
    assert np.mean(np.array(acc) > 0.9) < 0.1
```

**4. Importance rewards features with many distinct values, even pure noise.** A random row number gives a fully grown tree a new cut between almost every pair of points, and each cut shaves a little impurity off a noisy leaf. Here the label agrees with a real binary feature 80% of the time. Yet in a full tree, a column of row numbers with no information at all gets 62% of the importance, and the real feature gets 35%.

```python
import numpy as np
from model import DecisionTree

rng = np.random.default_rng(0)
signal = rng.integers(0, 2, 1000)
y = np.where(rng.random(1000) < 0.8, signal, 1 - signal)          # label = signal, 20% flipped
row_id = rng.permutation(1000).astype(float)                       # pure noise, all distinct
coin = rng.integers(0, 2, 1000).astype(float)                      # pure noise, two values
X = np.column_stack([signal, row_id, coin]).astype(float)
print(DecisionTree("gini").fit(X, y).feature_importances_)         # [0.353 0.618 0.029]
print(DecisionTree("gini", max_depth=3).fit(X, y).feature_importances_)   # [0.912 0.088 0.]
assert DecisionTree("gini").fit(X, y).feature_importances_[1] > 0.5
```

Importance computed on held-out data, such as permutation importance, does not have this bias, because cuts on noise do not help on new points.

## Common mistakes

- **"Trees need scaled features."** A split only asks whether a value is above a threshold, so any order-preserving change of units (scaling, logs) gives the same tree. This is one of the real advantages of trees over k-NN, PCA and regularised regression.
- **"The tree found the best questions to ask."** Each question is the best *given the ones above it*, chosen one at a time. Example 3 shows a case where the best pair of questions starts with a question that looks useless on its own.
- **"Gini and entropy give very different trees."** Both are 0 for pure nodes, peak at an even mix, and have similar shapes, so they usually choose the same or similar splits. Depth limits matter far more than the choice of impurity.
- **"High importance means the feature causes the outcome."** Importance measures how much the tree used a feature to reduce training impurity. It is biased towards features with many distinct values (example 4) and splits credit arbitrarily between correlated features.
- **"A deeper tree is more accurate."** Only on the training data. New-data accuracy usually peaks at a modest depth (example 1 and the animation), which is why trees are limited in depth or averaged into forests (concept 09).

## Further reading

- Breiman, Friedman, Olshen and Stone, *Classification and Regression Trees* (1984): CART, the algorithm implemented here.
- Quinlan, "Induction of decision trees", *Machine Learning* 1(1) (1986): ID3 and the entropy criterion.
- Hyafil and Rivest, "Constructing optimal binary decision trees is NP-complete", *Information Processing Letters* 5(1) (1976).
- Hastie, Tibshirani and Friedman, *The Elements of Statistical Learning*, 2nd edition (2009), section 9.2.
- Strobl, Boulesteix, Zeileis and Hothorn, "Bias in random forest variable importance measures: illustrations, sources and a solution", *BMC Bioinformatics* 8 (2007): the importance bias in example 4.
