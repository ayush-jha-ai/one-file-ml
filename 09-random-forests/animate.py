"""Render demo.gif: averaging more and more random trees smooths the decision surface."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from model import RandomForest, predict_tree
from sklearn.datasets import make_moons

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for plotstyle
import plotstyle as ps  # noqa: E402

ps.apply()

X, y = make_moons(1000, noise=0.3, random_state=1)
X_train, y_train, X_test, y_test = X[:500], y[:500], X[500:], y[500:]
B_MAX = 300
forest = RandomForest(B_MAX, seed=0).fit(X_train, y_train)

gx, gy = np.meshgrid(np.linspace(-2, 3, 150), np.linspace(-1.6, 2.1, 150))
grid = np.column_stack([gx.ravel(), gy.ravel()])
grid_p = np.cumsum([predict_tree(t, grid)[:, 1] for t in forest.trees], axis=0)    # running sums
test_p = np.cumsum([predict_tree(t, X_test)[:, 1] for t in forest.trees], axis=0)
B = np.arange(1, B_MAX + 1)
acc = np.mean(((test_p / B[:, None]) > 0.5) == y_test, axis=1)

counts = np.unique(np.round(np.geomspace(1, B_MAX, 70)).astype(int))
counts = np.concatenate([counts, np.full(84 - len(counts), B_MAX)])              # 7 s

cmap = LinearSegmentedColormap.from_list("prob", ["#9fc2ee", "#efefec", "#f6b59a"])
fig, (ax, ax_acc) = plt.subplots(1, 2, figsize=(10.5, 4.8), gridspec_kw={"wspace": 0.28})
image = ax.imshow(np.zeros_like(gx), extent=(-2, 3, -1.6, 2.1), origin="lower", cmap=cmap,
                  vmin=0, vmax=1, aspect="auto")
for c, colour in enumerate((ps.BLUE, ps.ORANGE)):
    ax.scatter(*X_train[y_train == c].T, s=12, color=colour, edgecolor=ps.SURFACE, linewidth=0.4)
ax.set(xlim=(-2, 3), ylim=(-1.6, 2.1), xlabel="$x_1$", ylabel="$x_2$")
ax.set_title("average vote of the trees so far", loc="left", fontsize=14)
ax.grid(False)

ax_acc.plot(B, acc, color=ps.BLUE, lw=1, alpha=0.25)
(c_acc,) = ax_acc.plot([], [], color=ps.BLUE, lw=2.5)
ax_acc.axhline(acc[0], color=ps.INK_2, lw=1, ls=":")
ax_acc.text(1.1, acc[0] - 0.012, "one tree", color=ps.INK_2, fontsize=12)
ax_acc.set(xscale="log", xlim=(1, B_MAX), ylim=(0.78, 0.9), xlabel="number of trees (log scale)",
           ylabel="accuracy on new data")
ax_acc.set_title("gains stop after a few dozen trees", loc="left", fontsize=14)
caption = fig.text(0.02, 0.95, "", fontsize=16)


def draw(frame):
    b = counts[frame]
    image.set_data((grid_p[b - 1] / b).reshape(gx.shape))
    c_acc.set_data(B[:b], acc[:b])
    caption.set_text(f"random forest   {b:>3} tree{'s' if b > 1 else ' '}   "
                     f"accuracy on new data {acc[b - 1]:.1%}")


fig.subplots_adjust(left=0.07, right=0.98, bottom=0.13, top=0.82)
anim = FuncAnimation(fig, draw, frames=len(counts))
ps.save_gif(anim, str(Path(__file__).with_name("demo.gif")), len(counts))
