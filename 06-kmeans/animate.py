"""Render demo.gif: Lloyd's algorithm alternating assign and update steps from a bad start."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from model import assign, inertia, kmeans

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for plotstyle
import plotstyle as ps  # noqa: E402

ps.apply()

rng = np.random.default_rng(1)
true_centres = np.array([[0.0, 0.0], [4.0, 0.5], [1.8, 3.6]])
X = np.vstack([rng.normal(c, 0.8, (120, 2)) for c in true_centres])
start = np.array([[-0.6, -0.4], [0.3, -0.6], [0.0, 0.5]])         # all three in one cluster
_, _, history = kmeans(X, 3, init=start)

# Frames: for each iteration show "assign" (colours change) then "update" (centres move).
stages = []                                                      # (centres, labels, text)
for t in range(1, len(history)):
    prev_centres, prev_labels, _ = history[t - 1]
    centres, labels, _ = history[t]
    stages.append((prev_centres, prev_labels, f"iteration {t}: assign each point"))
    stages.append((centres, prev_labels, f"iteration {t}: move centres to means"))
stages.append((history[-1][0], history[-1][1], "converged: nothing changes"))
FRAMES_EACH = max(2, min(10, 92 // len(stages)))
order = [s for s in stages for _ in range(FRAMES_EACH)]

colours = [ps.BLUE, ps.ORANGE, ps.AQUA]
tints = ListedColormap(["#b9d2f2", "#f8cdb9", "#bfe8d2"])
gx, gy = np.meshgrid(np.linspace(-3, 7, 200), np.linspace(-3, 6.5, 200))
grid = np.column_stack([gx.ravel(), gy.ravel()])

fig, (ax, ax_j) = plt.subplots(1, 2, figsize=(10.5, 4.8), gridspec_kw={"wspace": 0.28})
region = ax.imshow(np.zeros_like(gx), extent=(-3, 7, -3, 6.5), origin="lower", cmap=tints,
                   vmin=-0.5, vmax=2.5, interpolation="nearest", aspect="auto")
points = ax.scatter(*X.T, s=14, c=[colours[0]] * len(X), edgecolor=ps.SURFACE, linewidth=0.4)
centre_marks = ax.scatter(*start.T, s=260, marker="X", color=colours, edgecolor=ps.INK,
                          linewidth=1.5, zorder=5)
trails = [ax.plot([], [], color=ps.INK, lw=1.2, alpha=0.7)[0] for _ in range(3)]
ax.set(xlim=(-3, 7), ylim=(-3, 6.5), xlabel="$x_1$", ylabel="$x_2$")
ax.grid(False)
title = ax.set_title("", loc="left", fontsize=14)

J = [h[2] for h in history]
ax_j.plot(range(len(J)), J, color=ps.INK_2, lw=1, alpha=0.3)
(j_line,) = ax_j.plot([], [], color=ps.INK, lw=2.5, marker="o", ms=6)
ax_j.set(xlabel="iteration", ylabel="inertia  Σ ||x − μ||²", ylim=(0, None))
ax_j.set_title("objective after each iteration", loc="left", fontsize=14)
caption = fig.text(0.02, 0.95, "k-means: start with all three centres in one cluster", fontsize=16)


def draw(frame):
    centres, labels, text = order[frame]
    region.set_data(assign(grid, centres).reshape(gx.shape))
    points.set_color([colours[c] for c in labels])
    points.set_edgecolor(ps.SURFACE)
    centre_marks.set_offsets(centres)
    t = int(text.split()[1].rstrip(":")) if text.startswith("iteration") else len(history) - 1
    for j, trail in enumerate(trails):
        path = np.array([h[0][j] for h in history[: t + (0 if "assign" in text else 1)]])
        trail.set_data(path[:, 0], path[:, 1])
    title.set_text(text)
    shown = t if "assign" in text else t + 1
    j_line.set_data(range(min(shown, len(J))), J[: min(shown, len(J))])
    caption.set_text(f"k-means from a bad start   inertia {inertia(X, centres, labels):,.0f}")


fig.subplots_adjust(left=0.07, right=0.98, bottom=0.13, top=0.82)
anim = FuncAnimation(fig, draw, frames=len(order))
ps.save_gif(anim, str(Path(__file__).with_name("demo.gif")), len(order))
print(len(history) - 1, "iterations,", len(order), "frames")
