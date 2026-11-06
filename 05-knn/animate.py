"""Render demo.gif: k-NN decision regions as k grows from 1 (jagged) to 200 (oversmoothed)."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from model import KNN, leave_one_out_error

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for plotstyle
import plotstyle as ps  # noqa: E402

ps.apply()

# Three overlapping classes of 100 points each.
rng = np.random.default_rng(0)
centres = np.array([[0.0, 0.0], [2.2, 0.6], [1.0, 2.2]])
X = np.vstack([rng.normal(c, 0.9, (100, 2)) for c in centres])
y = np.repeat([0, 1, 2], 100)

KS = [1, 2, 3, 5, 7, 10, 15, 20, 30, 45, 60, 80, 110, 150, 200, 250]
FRAMES_EACH = 5                                                  # 80 frames = 6.7 s
train_err = [np.mean(KNN(k).fit(X, y).predict(X) != y) for k in KS]
loo_err = [leave_one_out_error(X, y, k) for k in KS]

gx, gy = np.meshgrid(np.linspace(-3, 5, 160), np.linspace(-3, 5, 160))
grid = np.column_stack([gx.ravel(), gy.ravel()])
regions = {k: KNN(k).fit(X, y).predict(grid).reshape(gx.shape) for k in KS}

tints = ListedColormap(["#b9d2f2", "#f8cdb9", "#bfe8d2"])       # light blue, orange, aqua
fig, (ax, ax_err) = plt.subplots(1, 2, figsize=(10.5, 4.8), gridspec_kw={"wspace": 0.3})
image = ax.imshow(regions[1], extent=(-3, 5, -3, 5), origin="lower", cmap=tints,
                  vmin=-0.5, vmax=2.5, interpolation="nearest")
for c, colour in enumerate((ps.BLUE, ps.ORANGE, ps.AQUA)):
    ax.scatter(*X[y == c].T, s=16, color=colour, edgecolor=ps.SURFACE, linewidth=0.5)
ax.set(xlim=(-3, 5), ylim=(-3, 5), xlabel="$x_1$", ylabel="$x_2$")
ax.set_title("predicted class everywhere", loc="left")
ax.grid(False)

(c_train,) = ax_err.plot([], [], color=ps.BLUE, lw=2.5, marker="o", ms=5, label="training error")
(c_loo,) = ax_err.plot([], [], color=ps.ORANGE, lw=2.5, marker="o", ms=5,
                       label="leave-one-out error")
ax_err.legend(loc="upper right", fontsize=12)
ax_err.set(xscale="log", xlim=(0.8, 320), ylim=(-0.02, 0.45), xlabel="k (log scale)",
           ylabel="error rate")
ax_err.set_title("error on seen vs unseen points", loc="left")
caption = fig.text(0.02, 0.95, "", fontsize=16)


def draw(frame):
    i = min(frame // FRAMES_EACH, len(KS) - 1)
    k = KS[i]
    image.set_data(regions[k])
    c_train.set_data(KS[: i + 1], train_err[: i + 1])
    c_loo.set_data(KS[: i + 1], loo_err[: i + 1])
    caption.set_text(f"k = {k:>3}   training error {train_err[i]:.3f}   "
                     f"leave-one-out error {loo_err[i]:.3f}")


fig.subplots_adjust(left=0.07, right=0.98, bottom=0.13, top=0.82)
n_frames = len(KS) * FRAMES_EACH
anim = FuncAnimation(fig, draw, frames=n_frames)
ps.save_gif(anim, str(Path(__file__).with_name("demo.gif")), n_frames)
