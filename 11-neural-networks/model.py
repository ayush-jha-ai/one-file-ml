"""Backpropagation twice: a scalar autograd engine, then a vectorised MLP. NumPy only.

Part A, Value: every arithmetic result remembers its inputs and the local derivative of
the operation. backward() visits nodes from the output back to the inputs and applies the
chain rule: d(out)/d(input) = sum over paths of products of local derivatives.

Part B, MLP: the same chain rule written for whole layers with matrices.
  forward:  A_0 = X,  Z_l = A_{l-1} W_l + b_l,  A_l = act(Z_l),  last layer softmax
  loss:     L = -(1/n) sum_i log p_{i, y_i}   (+ alpha/(2n) sum ||W_l||^2)
  backward: dZ_L = (P - Y)/n,   dW_l = A_{l-1}^T dZ_l,   db_l = sum_rows dZ_l,
            dA_{l-1} = dZ_l W_l^T,   dZ_{l-1} = dA_{l-1} * act'(Z_{l-1})
"""

import numpy as np

# ----------------------------------------------------------------- Part A: scalar autograd


class Value:
    def __init__(self, data, parents=(), local_grads=()):
        self.data = float(data)
        self.grad = 0.0
        self.parents = parents            # the Values this one was computed from
        self.local_grads = local_grads    # d(self)/d(parent) for each parent

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1.0, 1.0))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, k):                 # k is a plain number
        return Value(self.data**k, (self,), (k * self.data ** (k - 1),))

    def exp(self):
        e = np.exp(self.data)
        return Value(e, (self,), (e,))    # d/dx e^x = e^x

    def log(self):
        return Value(np.log(self.data), (self,), (1 / self.data,))

    def tanh(self):
        t = np.tanh(self.data)
        return Value(t, (self,), (1 - t**2,))

    def relu(self):
        return Value(max(self.data, 0.0), (self,), (float(self.data > 0),))

    __radd__, __rmul__ = __add__, __mul__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * other**-1

    def backward(self):
        # Order nodes so each comes after everything it depends on, then walk backwards.
        order, seen = [], set()

        def visit(v):
            if id(v) not in seen:
                seen.add(id(v))
                for p in v.parents:
                    visit(p)
                order.append(v)

        visit(self)
        self.grad = 1.0                                          # d(out)/d(out)
        for v in reversed(order):
            for parent, local in zip(v.parents, v.local_grads, strict=True):
                parent.grad += local * v.grad                    # chain rule, summed over paths

# --------------------------------------------------------------- Part B: vectorised MLP

ACTIVATIONS = {
    "tanh": (np.tanh, lambda Z, A: 1 - A**2),
    "relu": (lambda Z: np.maximum(Z, 0), lambda Z, A: (Z > 0).astype(float)),
}


def softmax(Z):
    E = np.exp(Z - Z.max(axis=1, keepdims=True))                # shifting leaves it unchanged
    return E / E.sum(axis=1, keepdims=True)


def init_params(sizes, rng):
    """Glorot-uniform weights (as scikit-learn), biases from the same range."""
    params = []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:], strict=True):
        bound = np.sqrt(6 / (fan_in + fan_out))
        params.append([rng.uniform(-bound, bound, (fan_in, fan_out)),
                       rng.uniform(-bound, bound, fan_out)])
    return params


def forward(params, X, activation="tanh"):
    """Returns the list of layer outputs A_0..A_L (softmax probabilities last) and Z's."""
    act = ACTIVATIONS[activation][0]
    As, Zs = [X], []
    for l, (W, b) in enumerate(params):
        Zs.append(As[-1] @ W + b)
        As.append(softmax(Zs[-1]) if l == len(params) - 1 else act(Zs[-1]))
    return As, Zs


def loss_and_grads(params, X, y, activation="tanh", alpha=0.0):
    """Mean cross-entropy (+ L2) and its gradient for every W and b, by backpropagation."""
    n = len(y)
    As, Zs = forward(params, X, activation)
    P, Z = As[-1], Zs[-1] - Zs[-1].max(axis=1, keepdims=True)
    log_P = Z - np.log(np.exp(Z).sum(axis=1, keepdims=True))   # log softmax: never log(0)
    loss = -np.mean(log_P[np.arange(n), y]) + alpha / (2 * n) * sum(
        np.sum(W**2) for W, _ in params)
    dZ = P.copy()
    dZ[np.arange(n), y] -= 1                                     # softmax + cross-entropy
    dZ /= n                                                      # dL/dZ_L = (P - Y) / n
    grads = [None] * len(params)
    for l in range(len(params) - 1, -1, -1):
        W, _ = params[l]
        grads[l] = [As[l].T @ dZ + alpha / n * W, dZ.sum(axis=0)]
        if l > 0:
            dA = dZ @ W.T                                        # back through the matmul
            dZ = dA * ACTIVATIONS[activation][1](Zs[l - 1], As[l])   # back through act
    return loss, grads


def train(params, X, y, lr=0.1, steps=1000, activation="tanh", alpha=0.0):
    """Full-batch gradient descent. Returns the updated params and the loss per step."""
    history = []
    for _ in range(steps):
        loss, grads = loss_and_grads(params, X, y, activation, alpha)
        params = [[W - lr * dW, b - lr * db] for (W, b), (dW, db) in zip(params, grads,
                                                                           strict=True)]
        history.append(loss)
    return params, history


def predict(params, X, activation="tanh"):
    return np.argmax(forward(params, X, activation)[0][-1], axis=1)
