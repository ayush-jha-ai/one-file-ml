"""Render demo.gif: the probability field and decision boundary forming as gradient descent runs."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for plotstyle
import plotstyle as ps  # noqa: E402

ps.apply()

# Two overlapping clouds of points, offset from the origin so the boundary has to travel.
rng = np.random.default_rng(0)
X = np.vstack([rng.normal([0.5, 1.0], 0.9, (60, 2)), rng.normal([2.5, 2.8], 0.9, (60, 2))])
y = np.r_[np.zeros(60), np.ones(60)]

N_STEPS = 400
gd = LogisticRegression(method="gd", learning_rate=0.3, n_iters=N_STEPS).fit(X, y)
newton = LogisticRegression(method="newton", n_iters=N_STEPS).fit(X, y)

steps = np.unique(np.round(np.geomspace(1, N_STEPS + 1, 70)).astype(int) - 1)
steps = np.concatenate([steps, np.full(84 - len(steps), steps[-1])])  # 84 frames = 7 s

gx, gy = np.meshgrid(np.linspace(-2, 5, 140), np.linspace(-1.5, 5.5, 140))
grid = np.column_stack([gx.ravel(), gy.ravel()])
# Diverging: class 0 blue, class 1 orange, neutral grey at p = 0.5.
cmap = LinearSegmentedColormap.from_list("prob", ["#c9dcf3", "#efefec", "#f9d4c3"])

fig, (ax, ax_loss) = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw={"wspace": 0.28})
field = ax.imshow(np.zeros_like(gx), extent=(-2, 5, -1.5, 5.5), origin="lower",
                  cmap=cmap, vmin=0, vmax=1, aspect="auto")
ax.scatter(*X[y == 0].T, s=26, color=ps.BLUE, edgecolor=ps.SURFACE, linewidth=0.8)
ax.scatter(*X[y == 1].T, s=26, color=ps.ORANGE, edgecolor=ps.SURFACE, linewidth=0.8)
(boundary,) = ax.plot([], [], color=ps.INK, lw=2.5)
ax.set(xlim=(-2, 5), ylim=(-1.5, 5.5), xlabel="$x_1$", ylabel="$x_2$")
ax.set_title("p(class 1) and the p = 0.5 line", loc="left")
ax.grid(False)

ax_loss.plot(np.arange(N_STEPS + 1), newton.losses, color=ps.AQUA, lw=2)
ax_loss.text(1.3, newton.losses[-1] - 0.045, "Newton's method: exact by step 6", fontsize=13)
(gd_curve,) = ax_loss.plot([], [], color=ps.BLUE, lw=2.5)
ax_loss.text(14, 0.6, "gradient descent", color=ps.INK, fontsize=13)
ax_loss.set(xscale="symlog", xlim=(0, N_STEPS), ylim=(0.2, 0.72), xlabel="step (log scale)",
            ylabel="cross-entropy loss")
ax_loss.set_title("loss", loc="left")
caption = fig.text(0.02, 0.95, "", fontsize=16)


def draw(frame):
    k = steps[frame]
    w, b, loss = gd.history[k]
    field.set_data((1 / (1 + np.exp(-(grid @ w + b)))).reshape(gx.shape))
    xs = np.array([-2.0, 5.0])
    if abs(w[1]) > 1e-9:                         # boundary: w1 x1 + w2 x2 + b = 0
        boundary.set_data(xs, -(w[0] * xs + b) / w[1])
    gd_curve.set_data(np.arange(k + 1), gd.losses[: k + 1])
    caption.set_text(f"logistic regression, gradient descent   step {k:>3}   loss {loss:.3f}")
    return field, boundary, gd_curve, caption


fig.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.8)
anim = FuncAnimation(fig, draw, frames=len(steps))
ps.save_gif(anim, str(Path(__file__).with_name("demo.gif")), len(steps))
