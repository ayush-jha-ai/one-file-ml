"""Render demo.gif: gradient descent fitting a line, shown in data space and in (w, b) space."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from model import LinearRegression, max_stable_learning_rate, mse, normal_equation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for plotstyle
import plotstyle as ps  # noqa: E402

ps.apply()

# Data: a noisy line. x is not centred, so the loss valley is tilted and narrow.
rng = np.random.default_rng(0)
x = rng.uniform(0, 4, 40)
y = 1.5 * x + 2 + rng.normal(0, 0.6, 40)
X = x[:, None]

lr = 0.9 * max_stable_learning_rate(X)  # close to the limit: zig-zags across the valley
model = LinearRegression(learning_rate=lr, n_iters=300, w0=[-1.5], b0=-1.0).fit(X, y)
w_best, b_best = normal_equation(X, y)

# Show early steps one by one, later steps more sparsely: 72 frames = 6 s at 12 fps.
steps = np.unique(np.round(np.geomspace(1, 300, 66)).astype(int) - 1)
steps = np.concatenate([steps, np.full(72 - len(steps), steps[-1])])  # hold the end

# Loss surface on a grid of (w, b).
W, B = np.meshgrid(np.linspace(-2.2, 4.0, 160), np.linspace(-2.2, 5.5, 160))
Z = np.mean((W[..., None] * x + B[..., None] - y) ** 2, axis=-1)

fig, (ax_data, ax_loss) = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw={"wspace": 0.3})

ax_data.scatter(x, y, s=28, color=ps.INK_2, alpha=0.7, linewidths=0)
grid_x = np.array([-0.2, 4.2])
(fit_line,) = ax_data.plot([], [], color=ps.BLUE, lw=3)
ax_data.set(xlim=(-0.2, 4.2), ylim=(-6, 10), xlabel="x", ylabel="y")
ax_data.set_title("the line being fitted", loc="left")

levels = np.quantile(Z, np.linspace(0.002, 0.9, 14))
ax_loss.contour(W, B, Z, levels=levels, colors=ps.INK_2, linewidths=0.7, alpha=0.45)
ax_loss.plot(w_best, b_best, "*", ms=20, color=ps.INK, mec=ps.SURFACE, mew=1.5, zorder=5)
(path_line,) = ax_loss.plot([], [], color=ps.ORANGE, lw=2)
(point,) = ax_loss.plot([], [], "o", ms=8, color=ps.ORANGE, mec=ps.SURFACE, mew=1.5, zorder=4)
ax_loss.set(xlabel="slope w", ylabel="intercept b")
ax_loss.set_title("loss surface: path of (w, b)", loc="left")
ax_loss.grid(False)
caption = fig.text(0.02, 0.95, "", fontsize=16, color=ps.INK, ha="left")
best_loss = mse(y, X @ w_best + b_best)


def draw(frame):
    k = steps[frame]
    path = np.array([(w[0], b) for w, b, _ in model.history[: k + 1]])
    w, b, loss = model.history[k]
    fit_line.set_data(grid_x, w[0] * grid_x + b)
    path_line.set_data(path[:, 0], path[:, 1])
    point.set_data([w[0]], [b])
    caption.set_text(f"gradient descent   step {k:>3}   loss {loss:6.2f}   (best {best_loss:.2f})")
    return fit_line, path_line, point, caption


fig.subplots_adjust(left=0.09, right=0.98, bottom=0.15, top=0.8)
anim = FuncAnimation(fig, draw, frames=len(steps), blit=False)
ps.save_gif(anim, str(Path(__file__).with_name("demo.gif")), len(steps))
