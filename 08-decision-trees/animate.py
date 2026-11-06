"""Render demo.gif: a decision tree grown one level at a time, from one region to many."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from model import DecisionTree
from sklearn.datasets import make_moons

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for plotstyle
import plotstyle as ps  # noqa: E402

ps.apply()

X, y = make_moons(600, noise=0.35, random_state=0)
X_train, y_train, X_test, y_test = X[:300], y[:300], X[300:], y[300:]

DEPTHS = list(range(0, 11))
FRAMES_EACH = 8                                              # 88 frames = 7.3 s
trees = [DecisionTree("gini", max_depth=d).fit(X_train, y_train) for d in DEPTHS]
train_acc = [np.mean(t.predict(X_train) == y_train) for t in trees]
test_acc = [np.mean(t.predict(X_test) == y_test) for t in trees]

gx, gy = np.meshgrid(np.linspace(-2, 3, 220), np.linspace(-1.6, 2.1, 220))
grid = np.column_stack([gx.ravel(), gy.ravel()])
regions = [t.predict(grid).reshape(gx.shape) for t in trees]

fig, (ax, ax_acc) = plt.subplots(1, 2, figsize=(10.5, 4.8), gridspec_kw={"wspace": 0.28})
image = ax.imshow(regions[0], extent=(-2, 3, -1.6, 2.1), origin="lower",
                  cmap=ListedColormap(["#b9d2f2", "#f8cdb9"]), vmin=-0.5, vmax=1.5,
                  interpolation="nearest", aspect="auto")
for c, colour in enumerate((ps.BLUE, ps.ORANGE)):
    ax.scatter(*X_train[y_train == c].T, s=14, color=colour, edgecolor=ps.SURFACE, linewidth=0.5)
ax.set(xlim=(-2, 3), ylim=(-1.6, 2.1), xlabel="$x_1$", ylabel="$x_2$")
ax.set_title("every split is a horizontal or vertical cut", loc="left", fontsize=14)
ax.grid(False)

(c_train,) = ax_acc.plot([], [], color=ps.BLUE, lw=2.5, marker="o", ms=5, label="training data")
(c_test,) = ax_acc.plot([], [], color=ps.ORANGE, lw=2.5, marker="o", ms=5, label="new data")
ax_acc.legend(loc="lower right", fontsize=12)
ax_acc.set(xlim=(-0.3, 10.3), ylim=(0.45, 1.02), xlabel="maximum depth", ylabel="accuracy",
           xticks=range(0, 11, 2))
ax_acc.set_title("accuracy", loc="left", fontsize=14)
caption = fig.text(0.02, 0.95, "", fontsize=16)


def draw(frame):
    i = min(frame // FRAMES_EACH, len(DEPTHS) - 1)
    image.set_data(regions[i])
    c_train.set_data(DEPTHS[: i + 1], train_acc[: i + 1])
    c_test.set_data(DEPTHS[: i + 1], test_acc[: i + 1])
    caption.set_text(f"decision tree   depth {DEPTHS[i]:>2}   {trees[i].n_leaves():>2} leaves   "
                     f"train {train_acc[i]:.0%}   new data {test_acc[i]:.0%}")


fig.subplots_adjust(left=0.07, right=0.98, bottom=0.13, top=0.82)
n_frames = len(DEPTHS) * FRAMES_EACH
anim = FuncAnimation(fig, draw, frames=n_frames)
ps.save_gif(anim, str(Path(__file__).with_name("demo.gif")), n_frames)
