# 11 · Neural networks and backpropagation

Stack simple functions (a weighted sum, then a bend) into layers, and train every weight at once with gradient descent. Backpropagation is the bookkeeping that makes the gradient cheap: it is the chain rule, applied in the right order. This file builds it twice, first one number at a time and then with matrices.

![Left: three interleaved spiral arms of points, with the plane coloured by the network's predicted class. Early on the regions are three rough wedges; as training continues the boundaries bend to follow each arm. Right: the loss on a log scale, flat for about 150 steps, then falling steeply towards 0.001.](demo.gif)

## Intuition

A neuron takes a weighted sum of its inputs and passes it through a bend, such as tanh. One layer of neurons can only draw simple boundaries, but feeding one layer's outputs into the next composes the bends, and the network can carve out shapes like the spiral above. To train it, we need to know how a small change to *each* weight would change the loss. Working outwards from each weight separately would repeat the same calculations millions of times. Backpropagation instead starts at the loss and works backwards once, passing each layer the sensitivity it needs from the layer above.

## The maths

### Part A: the chain rule on a graph of scalars

Any calculation can be drawn as a graph: inputs at the bottom, each node an operation on its parents, the loss at the top. Each operation knows its *local derivative*. For example:

| operation | value | local derivatives |
|---|---|---|
| $c = a + b$ | $a + b$ | $\partial c/\partial a = 1$, $\partial c/\partial b = 1$ |
| $c = ab$ | $ab$ | $\partial c/\partial a = b$, $\partial c/\partial b = a$ |
| $c = a^k$ | $a^k$ | $ka^{k-1}$ |
| $c = e^a$ | $e^a$ | $e^a$ |
| $c = \log a$ | $\log a$ | $1/a$ |
| $c = \tanh a$ | $\tanh a$ | $1 - \tanh^2 a$ |
| $c = \max(a, 0)$ | ReLU | $1$ if $a > 0$, else $0$ |

If the output $L$ depends on a node $v$, and $v$ feeds into nodes $c_1, \dots, c_m$, the multivariable chain rule says

$$\frac{\partial L}{\partial v} = \sum_{j=1}^{m}\frac{\partial L}{\partial c_j}\,\frac{\partial c_j}{\partial v}.$$

The sum matters: a value used in several places collects gradient from every use. For $y = x^2 + x$, written as $x \cdot x + x$, the three uses of $x$ contribute $x + x + 1 = 2x + 1$.

To compute $\partial L/\partial v$ for every node, we need $\partial L/\partial c_j$ for its children first. So: sort the nodes so that each comes after everything it depends on (a *topological order*), set $\partial L/\partial L = 1$, and visit the nodes in reverse. At each node, push its gradient times each local derivative into its parents. Every edge is visited once, so all the gradients together cost about as much as one forward evaluation.

### Part B: the same thing for whole layers

**Forward pass.** Stack the $n$ inputs as rows of $X$, so $A_0 = X$. Layer $l$ has a weight matrix $W_l$ and a bias vector $\mathbf{b}_l$:

$$Z_l = A_{l-1}W_l + \mathbf{b}_l, \qquad A_l = \phi(Z_l),$$

where $\phi$ (tanh or ReLU) acts on every entry. The last layer uses softmax instead, which turns each row of scores into probabilities:

$$P_{ik} = \frac{e^{Z_{ik}}}{\sum_j e^{Z_{ij}}}.$$

Subtracting the row's maximum from every score leaves $P$ unchanged (the factor $e^{-\max}$ cancels) and prevents overflow.

**Loss.** Mean cross-entropy, plus an optional L2 penalty that matches scikit-learn's convention. The code computes $\log P_{ik} = Z_{ik} - \log\sum_j e^{Z_{ij}}$ directly, so a probability that rounds to 0 never produces $\log 0$.

$$L = -\frac{1}{n}\sum_i \log P_{i,y_i} + \frac{\alpha}{2n}\sum_l\lVert W_l\rVert^2.$$

**Gradient at the output.** For one example, $-\log P_y = -z_y + \log\sum_j e^{z_j}$. Differentiate with respect to $z_k$:

$$\frac{\partial}{\partial z_k}\Big(-z_y + \log\textstyle\sum_j e^{z_j}\Big) = -\mathbb{1}[k = y] + \frac{e^{z_k}}{\sum_j e^{z_j}} = P_k - \mathbb{1}[k = y].$$

So, averaging over examples, $\partial L/\partial Z_L = (P - Y)/n$, where $Y$ is the one-hot label matrix. This is the same $p - y$ as in logistic regression (concept 02).

**Through a linear layer.** Write $G = \partial L/\partial Z$ for a layer with $Z = AW + \mathbf{b}$. Entry by entry, $Z_{ik} = \sum_j A_{ij}W_{jk} + b_k$. The weight $W_{jk}$ affects every $Z_{ik}$ (all rows $i$) with local derivative $A_{ij}$, so

$$\frac{\partial L}{\partial W_{jk}} = \sum_i A_{ij}G_{ik} = (A^\top G)_{jk}, \qquad \frac{\partial L}{\partial b_k} = \sum_i G_{ik}, \qquad \frac{\partial L}{\partial A_{ij}} = \sum_k G_{ik}W_{jk} = (GW^\top)_{ij}.$$

The L2 penalty adds $\tfrac{\alpha}{n}W$ to $\partial L/\partial W$.

**Through an activation.** $A = \phi(Z)$ acts entry by entry, so each entry has a single local derivative:

$$\frac{\partial L}{\partial Z} = \frac{\partial L}{\partial A}\odot\phi'(Z), \qquad \tanh'(z) = 1 - \tanh^2(z), \qquad \text{ReLU}'(z) = \mathbb{1}[z > 0].$$

Here $\odot$ is entry-by-entry multiplication. Alternate these two rules from the last layer to the first and every gradient is found. That is backpropagation.

**Starting weights.** All weights start at random, never all equal (example 1). Each is drawn uniformly from $[-a, a]$ with $a = \sqrt{6/(n_{\text{in}} + n_{\text{out}})}$ (Glorot initialisation). A uniform variable on $[-a, a]$ has variance $a^2/3 = 2/(n_{\text{in}} + n_{\text{out}})$, chosen so that signals keep roughly the same size going forwards and backwards through each layer (example 2).

## Code walkthrough

| Equation | Code | Lines |
|---|---|---|
| local derivatives of $+$ and $\times$ | `(1.0, 1.0)` and `(other.data, self.data)` | [28](model.py#L28), [32](model.py#L32) |
| $ka^{k-1}$, $e^a$, $1/a$ | `k * self.data ** (k - 1)`, `(e,)`, `(1 / self.data,)` | [35-42](model.py#L35-L42) |
| $1 - \tanh^2$ and $\mathbb{1}[a > 0]$ | `(1 - t**2,)`, `(float(self.data > 0),)` | [46](model.py#L46), [49](model.py#L49) |
| $\partial L/\partial L = 1$ | `self.grad = 1.0` | [74](model.py#L74) |
| $\partial L/\partial v \mathrel{+}= \partial L/\partial c \cdot \partial c/\partial v$ | `parent.grad += local * v.grad` | [77](model.py#L77) |
| $\phi'$ for tanh and ReLU | `1 - A**2`, `(Z > 0).astype(float)` | [82-83](model.py#L82-L83) |
| softmax with the max subtracted | `np.exp(Z - Z.max(axis=1, keepdims=True))` | [88](model.py#L88) |
| $a = \sqrt{6/(n_{\text{in}} + n_{\text{out}})}$ | `np.sqrt(6 / (fan_in + fan_out))` | [96](model.py#L96) |
| $Z_l = A_{l-1}W_l + \mathbf{b}_l$ | `As[-1] @ W + b` | [107](model.py#L107) |
| $\log P_{ik} = Z_{ik} - \log\sum_j e^{Z_{ij}}$, then $L$ | `Z - np.log(np.exp(Z).sum(axis=1, keepdims=True))`, `-np.mean(log_P[np.arange(n), y]) + ...` | [117-118](model.py#L117-L118) |
| $\partial L/\partial Z_L = (P - Y)/n$ | `dZ[np.arange(n), y] -= 1`, `dZ /= n` | [121-122](model.py#L121-L122) |
| $\partial L/\partial W = A^\top G + \frac{\alpha}{n}W$, $\partial L/\partial\mathbf{b} = \sum_i G_i$ | `As[l].T @ dZ + alpha / n * W, dZ.sum(axis=0)` | [126](model.py#L126) |
| $\partial L/\partial A = GW^\top$ | `dZ @ W.T` | [128](model.py#L128) |
| $\partial L/\partial Z = \partial L/\partial A \odot \phi'(Z)$ | `dA * ACTIVATIONS[activation][1](...)` | [129](model.py#L129) |
| gradient descent step | `W - lr * dW, b - lr * db` | [138](model.py#L138) |

The tests check five things:
- **Scalar engine:** against PyTorch autograd on an expression using every operation (to `1e-12`).
- **Two parts agree:** the scalar engine, used to build a whole network one number at a time, gives exactly the same loss and gradients as the vectorised code (to `1e-12`).
- **Vectorised gradients:** against PyTorch for tanh and ReLU, with the L2 term (to `1e-12`).
- **Training:** 50 steps of full-batch gradient descent give the same weights as scikit-learn's `MLPClassifier` started from the same weights (to `1e-12`).
- **Finite differences:** gradients agree with numerical estimates.

## Where it breaks

Run these from inside this folder. They use moons (`make_moons(400, noise=0.15, random_state=0)`) unless stated.

**1. Identical starting weights stay identical.** If two hidden units start with the same weights, they compute the same output and receive the same gradient, so every update keeps them equal. Sixteen such units are one unit in disguise: after 2,000 steps, merging them into a single unit (summing their outgoing weights) gives the same outputs to `1e-10`. They reach 88.75% accuracy; the same network with random starting weights reaches 98.5%.

```python
import numpy as np
from model import forward, init_params, predict, train
from sklearn.datasets import make_moons

X, y = make_moons(400, noise=0.15, random_state=0)
rng = np.random.default_rng(0)
w, v = rng.normal(size=(2, 1)), rng.normal(size=(1, 2))
same = [[np.repeat(w, 16, axis=1), np.zeros(16)], [np.repeat(v, 16, axis=0), np.zeros(2)]]
same, _ = train(same, X, y, lr=0.5, steps=2000)
merged = [[same[0][0][:, :1], same[0][1][:1]], [same[1][0].sum(axis=0, keepdims=True), same[1][1]]]
rand, _ = train(init_params([2, 16, 2], np.random.default_rng(0)), X, y, lr=0.5, steps=2000)

spread = np.ptp(same[0][0], axis=1).max()                       # how different the 16 units are
gap = np.abs(forward(same, X)[0][-1] - forward(merged, X)[0][-1]).max()
accs = [np.mean(predict(p, X) == y) for p in (same, rand)]
print(spread, gap, accs)                                        # 0.0  ~1e-16  [0.8875, 0.985]
assert spread < 1e-10 and gap < 1e-10 and accs[0] < 0.92 < accs[1]
```

**2. In a deep network the starting scale decides everything.** With 20 tanh layers of 32 units and Glorot weights, the first layer's gradient has norm 0.16. Halve every starting weight and each layer shrinks the signal further, both forwards and backwards. The gradient norm drops to $3.6\times10^{-7}$, and after 300 steps the loss has not moved from $\log 2 = 0.693$, the value for a coin toss.

```python
import numpy as np
from model import init_params, loss_and_grads, train
from sklearn.datasets import make_moons

X, y = make_moons(400, noise=0.15, random_state=0)
for scale in (1.0, 0.5):
    params = [[W * scale, b * 0] for W, b in init_params([2] + [32] * 20 + [2],
                                                         np.random.default_rng(0))]
    _, grads = loss_and_grads(params, X, y)
    _, history = train(params, X, y, lr=0.1, steps=300)
    print(scale, np.linalg.norm(grads[0][0]), history[-1])   # 1.0: 0.16 0.037   0.5: 3.6e-07 0.693
assert np.linalg.norm(grads[0][0]) < 1e-6 and abs(history[-1] - np.log(2)) < 1e-6
```

**3. Far from the data, a ReLU network is certain.** A ReLU network is piecewise linear, so far from the data the output scores grow linearly with distance, and softmax turns large score gaps into probabilities of essentially 1. Points ten or a hundred times outside the training range get a confidence of 1.000000. High confidence does not mean the input resembles anything seen in training.

```python
import numpy as np
from model import forward, init_params, train
from sklearn.datasets import make_moons

X, y = make_moons(400, noise=0.15, random_state=0)
params, _ = train(init_params([2, 32, 32, 2], np.random.default_rng(0)), X, y, lr=0.5,
                  steps=3000, activation="relu")
far = np.array([[50.0, -80.0], [-60.0, 40.0], [90.0, 90.0]])   # data lie within [-1.5, 2.5]
confidence = forward(params, far, "relu")[0][-1].max(axis=1)
print(confidence)                                              # [1. 1. 1.]
assert np.all(confidence > 0.999999)
```

**4. A large learning rate kills ReLU units.** If one big step pushes a ReLU unit's weights so that its input is negative for every training example, its output and its gradient are both zero from then on, so it never recovers. With learning rate 20, 66% of 64 hidden units end up dead, and accuracy is 75.5%, against 97% at learning rate 0.5.

```python
import numpy as np
from model import init_params, predict, train
from sklearn.datasets import make_moons

X, y = make_moons(400, noise=0.15, random_state=0)
for lr in (0.5, 20.0):
    params, _ = train(init_params([2, 64, 2], np.random.default_rng(0)), X, y, lr=lr, steps=300,
                      activation="relu")
    hidden_input = X @ params[0][0] + params[0][1]
    dead = np.mean((hidden_input > 0).sum(axis=0) == 0)       # never active on any example
    print(lr, dead, np.mean(predict(params, X, "relu") == y))  # 0.5: 0.05 0.97   20: 0.66 0.755
assert dead > 0.6
```

## Common mistakes

- **"Backpropagation is a training algorithm."** It only computes gradients. Gradient descent (or another optimiser) uses them to change the weights. The same backward pass serves plain gradient descent, momentum or Adam.
- **"Gradients of a reused value are overwritten."** They are summed over every use, which is why the scalar engine uses `+=` and why gradients must be reset before each new backward pass.
- **"Softmax and cross-entropy need separate derivatives."** Combined, their derivative with respect to the scores is simply $P - Y$. Computing them separately is slower and less accurate, because it divides by probabilities that can be tiny.
- **"Initialisation is a detail."** Equal starting weights make hidden units identical forever (example 1), and the wrong scale stops deep networks from learning at all (example 2).
- **"The output probabilities measure how sure the model should be."** They are trained to be right on the training data, not to be humble elsewhere (example 3).

## Further reading

- Rumelhart, Hinton and Williams, "Learning representations by back-propagating errors", *Nature* 323 (1986).
- Linnainmaa, "The representation of the cumulative rounding error of an algorithm as a Taylor expansion of the local rounding errors", MSc thesis, University of Helsinki (1970): reverse-mode automatic differentiation.
- Glorot and Bengio, "Understanding the difficulty of training deep feedforward neural networks", *Proceedings of AISTATS* (2010): the initialisation used here, and vanishing signals.
- Hein, Andriushchenko and Bitterwolf, "Why ReLU networks yield high-confidence predictions far away from the training data and how to mitigate the problem", *Proceedings of CVPR* (2019).
- Goodfellow, Bengio and Courville, *Deep Learning* (2016), chapter 6, especially section 6.5 on back-propagation.
- Baydin, Pearlmutter, Radul and Siskind, "Automatic differentiation in machine learning: a survey", *Journal of Machine Learning Research* 18 (2018).
