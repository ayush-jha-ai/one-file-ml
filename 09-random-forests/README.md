# 09 · Random forests (bagging)

Train many deep decision trees, each on a random resample of the data and a random subset of features, then average their votes. A single deep tree is unstable; averaging hundreds of them gives one of the most reliable general-purpose classifiers available, with little tuning.

![Left: the plane shaded by the average vote of the trees so far. With one tree the regions are jagged blocks; as trees are added up to 300, the colours blend into a smooth probability surface. Right: accuracy on new data rises from 82.8% with one tree to about 86% within a few dozen trees, then stays flat.](demo.gif)

## Intuition

A fully grown tree fits its training data exactly, so a slightly different training set gives a noticeably different tree. Its errors are large but partly random. Averaging many independent noisy guesses cancels much of the noise, so we would like many trees trained on independent datasets. We only have one dataset, so each tree gets a *bootstrap sample*: n rows drawn at random with replacement. Trees grown this way are still similar, because the strongest features dominate every one. So each split is also restricted to a random handful of features, which makes the trees disagree more and the average better.

## The maths

**Bootstrap sample.** Draw $n$ rows from the $n$ training rows, uniformly and with replacement. A particular row is missed by one draw with probability $1 - \tfrac1n$, and by all $n$ independent draws with probability

$$\left(1 - \frac{1}{n}\right)^n \xrightarrow{\;n\to\infty\;} e^{-1} \approx 0.368.$$

(This is the standard limit $(1 + x/n)^n \to e^x$ with $x = -1$.) So each tree sees about 63.2% of the distinct rows, and the other 36.8% are *out of bag* for it.

**Why averaging helps: the variance of an average.** Let $T_1, \dots, T_B$ be the predictions of $B$ trees at some point $\mathbf{x}$. They come from the same random procedure, so each has the same variance $\sigma^2$ and every pair has the same correlation $\rho$, meaning $\operatorname{Cov}(T_a, T_b) = \rho\sigma^2$ for $a \ne b$. The variance of a sum is the sum of all variances and covariances. There are $B$ variance terms and $B(B-1)$ covariance terms, so

$$\operatorname{Var}\!\left(\frac{1}{B}\sum_{b=1}^B T_b\right) = \frac{1}{B^2}\Big(B\sigma^2 + B(B-1)\rho\sigma^2\Big) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2.$$

Two consequences:

- As $B$ grows, the second term vanishes, but the first does not. Adding trees can only remove the variance that the trees *do not share*, so accuracy levels off (the plateau in the animation).
- The floor $\rho\sigma^2$ is lowered by making the trees less correlated. That is the job of the random feature subset. At each split only $m$ of the $d$ features are candidates, $m = \lfloor\sqrt{d}\rfloor$ by default for classification. Then a dominant feature cannot be the first split of every tree.

**Averaging does not change bias.** The expected value of the average is $\mathbb{E}\big[\tfrac1B\sum_b T_b\big] = \mathbb{E}[T_1]$. Whatever systematic error each tree makes, the forest makes too. That is why forests use deep, low-bias trees, and why averaging shallow trees cannot repair a model that is too simple (example 1).

**Prediction.** The forest's probability for class $c$ is the average over trees of each tree's leaf proportion for $c$. The predicted class is the one with the highest average. This is scikit-learn's convention too.

**Out-of-bag estimate.** For each training row, average the votes of only the trees for which it was out of bag (about $0.368B$ of them), and compare with its label. This gives a validation estimate without a separate hold-out set.

## Code walkthrough

| Step | Code | Lines |
|---|---|---|
| Gini impurity $1 - \sum_c p_c^2$ | `1 - np.sum(p**2, axis=-1)` | [16](model.py#L16) |
| class counts left of every cut, and the weighted Gini score | `np.cumsum(np.eye(n_classes)[y[order]], axis=0)` | [25-27](model.py#L25-L27) |
| random subset of $m$ features at each split | `rng.choice(X.shape[1], size=max_features, replace=False)` | [43](model.py#L43) |
| $m = \lfloor\sqrt{d}\rfloor$ | `int(np.sqrt(d))` | [80](model.py#L80) |
| bootstrap sample: $n$ rows with replacement | `rng.integers(0, n, n)` | [85](model.py#L85) |
| remember which rows each tree saw | `np.bincount(rows, minlength=n) > 0` | [87](model.py#L87) |
| forest probability = mean of tree probabilities | `np.mean([predict_tree(t, X) for t in trees], axis=0)` | [96](model.py#L96) |
| out-of-bag votes: only trees that did not see the row | `votes[~seen] += predict_tree(tree, self.X_train[~seen])` | [105](model.py#L105) |
| out-of-bag accuracy | `np.argmax(votes[has_vote], axis=1) == self.y_train[has_vote]` | [107](model.py#L107) |

A forest uses random numbers that cannot be made to match scikit-learn's draw for draw. So the tests check what can be checked exactly, and the rest statistically, with stated tolerances:
- **Exact:** with bootstrapping off and all features allowed, one tree equals scikit-learn's `DecisionTreeClassifier`.
- **Out-of-bag fraction:** within 0.006 of $e^{-1}$.
- **Forest accuracy:** within 0.02 of scikit-learn's `RandomForestClassifier` over three seeds.
- **Out-of-bag accuracy:** within 0.03 of scikit-learn's `oob_score_`.

## Where it breaks

**1. Averaging cannot remove an error every tree shares.** A forest of 300 depth-1 trees (stumps) on XOR scores 52.6%. Each stump splits on one feature, and XOR has no signal in either feature alone, so every stump is wrong in the same way and the average is wrong too. Deep trees can represent the interaction, and the forest of those scores 100%.

```python
import numpy as np
from model import RandomForest

rng = np.random.default_rng(0)
X = rng.uniform(-1, 1, (1000, 2))
y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
for depth in (1, None):
    forest = RandomForest(300, max_depth=depth, seed=0).fit(X[:500], y[:500])
    print(depth, np.mean(forest.predict(X[500:]) == y[500:]))     # 1: 0.526   None: 1.0
    assert np.mean(forest.predict(X[500:]) == y[500:]) < 0.6 or depth is None
```

**2. More trees stop helping early.** By the variance formula, extra trees remove only the $(1-\rho)\sigma^2/B$ part. On this data, 10 trees reach 86.2% and 300 trees 85.6%: the same, within noise. More trees never make a forest worse in expectation, but past a point they only cost time and memory.

```python
import numpy as np
from model import RandomForest
from sklearn.datasets import make_moons

X, y = make_moons(1000, noise=0.3, random_state=1)
forest = RandomForest(300, seed=0).fit(X[:500], y[:500])
acc = {b: np.mean(forest.predict(X[500:], n_trees=b) == y[500:]) for b in (1, 10, 300)}
print(acc)                                    # {1: 0.828, 10: 0.862, 300: 0.856}
assert acc[10] > acc[1] and abs(acc[300] - acc[10]) < 0.01
```

**3. The out-of-bag estimate is pessimistic for small forests.** Each row's out-of-bag vote uses only the roughly 37% of trees that did not see it. With 5 trees, that is often one or two trees, which are much weaker than the full forest of 5. Out-of-bag accuracy is then 84.3%, against 88.0% on genuinely new data. With 100 trees the two agree (88.7% and 89.7%).

```python
import numpy as np
from model import RandomForest
from sklearn.datasets import make_moons

X, y = make_moons(600, noise=0.3, random_state=1)
for n_trees in (5, 100):
    forest = RandomForest(n_trees, seed=0).fit(X[:300], y[:300])
    print(n_trees, forest.oob_score(), np.mean(forest.predict(X[300:]) == y[300:]))
    # 5: 0.843 0.88     100: 0.887 0.897
small = RandomForest(5, seed=0).fit(X[:300], y[:300])
assert np.mean(small.predict(X[300:]) == y[300:]) - small.oob_score() > 0.03
```

## Common mistakes

- **"Random forests cannot overfit, so more trees fix everything."** Adding trees does not overfit, but each tree can, and their shared errors stay (examples 1 and 2). Depth limits and minimum leaf sizes still matter on noisy data.
- **"Bagging and random forests are the same."** Bagging resamples rows only. A random forest also restricts each split to a random subset of features, which lowers the correlation $\rho$ between trees and therefore the floor $\rho\sigma^2$.
- **"Each tree is trained on 100% of the data, shuffled."** Each tree sees a bootstrap sample, which contains about 63% of the distinct rows, some of them several times. The other 37% are what make the out-of-bag estimate possible.
- **"The forest probability is the fraction of trees voting for the class."** In this implementation, and in scikit-learn, it is the average of the trees' leaf *probabilities* (soft voting). With fully grown trees the two coincide because every leaf is pure; with depth limits they differ.
- **"The out-of-bag score is the same as a test score."** It is close for large forests, but each out-of-bag prediction uses fewer trees than the final model, so for small forests it understates accuracy (example 3).

## Further reading

- Breiman, "Bagging predictors", *Machine Learning* 24(2) (1996).
- Breiman, "Random forests", *Machine Learning* 45(1) (2001).
- Efron and Tibshirani, *An Introduction to the Bootstrap* (1993).
- Hastie, Tibshirani and Friedman, *The Elements of Statistical Learning*, 2nd edition (2009), chapters 8.7 and 15: bagging, and the $\rho\sigma^2 + (1-\rho)\sigma^2/B$ result.
- Louppe, "Understanding random forests: from theory to practice", PhD thesis, University of Liège, arXiv:1407.7502 (2014).
