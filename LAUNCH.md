# Launch kit

Every number below comes from a "where it breaks" example that runs in CI, so it can be quoted with confidence.

## Per concept

| # | Hook for a social post | GIF | Most surprising insight |
|---|---|---|---|
| 01 | Gradient descent in 80 lines of NumPy, with the exact learning rate where it stops working. | `01-linear-regression/demo.gif` | The learning-rate limit is a cliff, not a guideline: 0.99× the limit converges, 1.01× ends with a loss of 10^34. And with unscaled features, the loss can look finished while one weight is 11% off and the intercept has the wrong sign. |
| 02 | Logistic regression from scratch, where Newton's method finishes in 6 steps and gradient descent is still short after 400. | `02-logistic-regression/demo.gif` | On perfectly separable data there is no best model: the weights grow forever, about +1.1 in size for every tenfold increase in training steps. |
| 03 | Why L1 regularisation sets weights to exactly zero and L2 never does, in one picture. | `03-regularisation/demo.gif` | Measure one feature in kilometres instead of metres and lasso drops it entirely. Same data, same model, different units. |
| 04 | Bias and variance computed exactly, not hand-waved, for polynomials of degree 0 to 12. | `04-bias-variance/demo.gif` | The U-curve is not the whole story: with 20 points, error peaks at 2.6 million at degree 19, then falls back to 0.138 by degree 300, close to the best value of 0.113. |
| 05 | k-nearest neighbours in 72 lines, matching scikit-learn neighbour for neighbour. | `05-knn/demo.gif` | Add 10 columns of pure noise and k-NN's accuracy on an easy problem drops from 95% to 65%. In 1,000 dimensions the farthest point is only 11% farther away than the nearest. |
| 06 | k-means as two steps that each provably lower the same objective, watched one step at a time. | `06-kmeans/demo.gif` | On uniformly random points with no clusters at all, k-means confidently returns four tidy groups of almost equal size. |
| 07 | PCA from scratch: why the direction of maximum variance is also the line closest to the data. | `07-pca/demo.gif` | A component explaining 96% of the variance classified 53% of points correctly; the one explaining 4% classified 100%. |
| 08 | A decision tree that matches scikit-learn exactly, and a proof for every node where it does not. | `08-decision-trees/demo.gif` | A column of row numbers, pure noise, gets 62% of a fully grown tree's feature importance, more than the feature that actually predicts the label. |
| 09 | Random forests from scratch: why averaging trees works, and the floor it cannot get below. | `09-random-forests/demo.gif` | A forest of 300 one-split trees scores 52.6% on XOR, a coin toss. Averaging cannot remove an error every tree shares. |
| 10 | Gradient boosting that matches scikit-learn at every one of 100 rounds. | `10-gradient-boosting/demo.gif` | One corrupted label out of 200 makes squared-error boosting 6,000 times worse against the true curve. |
| 11 | Backpropagation written twice, once for single numbers and once for matrices, and both agree to 12 decimal places. | `11-neural-networks/demo.gif` | Start 16 hidden units with identical weights and they stay identical forever. The network reaches exactly the accuracy of a network with one hidden unit. |
| 12 | A transformer block in 95 lines of NumPy, matching PyTorch to 12 decimal places. | `12-attention/demo.gif` | Without position information, a transformer gives exactly the same outputs (reordered) for "dog bites man" and "man bites dog". It also cannot tell one copy of its context from three. |

## Show HN

**Title**

Show HN: Twelve core ML algorithms, each in one NumPy file, tested against sklearn and PyTorch

**First comment**

I built this because from-scratch ML material usually either stops at the maths or hides it inside library calls, so here each of twelve concepts, from linear regression to a transformer block, is under 150 lines of NumPy with every equation linked to the line that implements it. Each one is checked against scikit-learn or PyTorch, usually to 1e-10 or better, and where an exact match is impossible the tests prove why (tied tree splits are verified node by node; random forests are compared statistically). What I think is most different is the "where it breaks" section in each README: runnable examples, executed in CI, showing things like double descent reaching an error of 2.6 million, k-means finding clusters in pure noise, and a transformer that cannot tell word order without position encodings.
