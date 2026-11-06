"""Render demo.gif: each round, a small tree fits what the ensemble still gets wrong."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from model import GradientBoosting, apply

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for plotstyle
import plotstyle as ps  # noqa: E402

ps.apply()

rng = np.random.default_rng(0)
x = np.sort(rng.uniform(0, 1, 120))
y = np.sin(2 * np.pi * x) + 0.6 * x + rng.normal(0, 0.15, 120)
X, grid = x[:, None], np.linspace(0, 1, 400)[:, None]

LR, ROUNDS = 0.3, 100
model = GradientBoosting("squared_error", ROUNDS, LR, max_depth=2).fit(X, y)
F_train = [np.full(len(y), model.init_)] + list(model.staged_decision_function(X))
F_grid = [np.full(len(grid), model.init_)] + list(model.staged_decision_function(grid))

shown = list(range(1, 13)) + [15, 20, 25, 30, 40, 50, 60, 80, 100]
order = [m for m in shown for _ in range(4)]                     # 84 frames = 7 s

fig, (ax, ax_r) = plt.subplots(1, 2, figsize=(10.5, 4.8), gridspec_kw={"wspace": 0.28})
ax.scatter(x, y, s=14, color=ps.INK_2, alpha=0.6, linewidths=0)
(fit,) = ax.plot([], [], color=ps.BLUE, lw=2.5)
ax.set(xlim=(0, 1), ylim=(-1.6, 2.2), xlabel="x", ylabel="y")
ax.set_title("ensemble prediction $F_m(x)$", loc="left", fontsize=14)

ax_r.axhline(0, color=ps.INK_2, lw=0.8)
resid = ax_r.scatter(x, np.zeros_like(x), s=14, color=ps.INK_2, alpha=0.6, linewidths=0)
(step,) = ax_r.plot([], [], color=ps.ORANGE, lw=2.5, drawstyle="steps-mid")
ax_r.set(xlim=(0, 1), ylim=(-1.6, 2.2), xlabel="x", ylabel="residual  y − F")
ax_r.set_title("residuals, and the new tree fitted to them", loc="left", fontsize=14)
caption = fig.text(0.02, 0.95, "", fontsize=16)


def draw(frame):
    m = order[frame]                                             # round m adds tree m
    tree, gamma = model.trees[m - 1]
    fit.set_data(grid[:, 0], F_grid[m])
    resid.set_offsets(np.column_stack([x, y - F_train[m - 1]]))  # what tree m was asked to fit
    step.set_data(grid[:, 0], gamma[apply(tree, grid)])
    mse = np.mean((y - F_train[m]) ** 2)
    caption.set_text(f"gradient boosting   round {m:>3}   training MSE {mse:.4f}   "
                     f"(each tree added × {LR})")


fig.subplots_adjust(left=0.07, right=0.98, bottom=0.13, top=0.82)
anim = FuncAnimation(fig, draw, frames=len(order))
ps.save_gif(anim, str(Path(__file__).with_name("demo.gif")), len(order))
