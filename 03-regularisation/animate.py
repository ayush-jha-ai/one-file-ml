"""Render demo.gif: L2 shrinks every weight, L1 sets weights exactly to zero, as alpha grows."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Polygon
from model import centre, lasso_alpha_max, path
from sklearn.datasets import load_diabetes

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for plotstyle
import plotstyle as ps  # noqa: E402

ps.apply()

# Left panel: two correlated, standardised features, so the geometry can be drawn.
rng = np.random.default_rng(0)
x1 = rng.normal(size=200)
x2 = 0.5 * x1 + np.sqrt(0.75) * rng.normal(size=200)
X2 = np.column_stack([x1, x2])
X2 = (X2 - X2.mean(0)) / X2.std(0)
y2 = X2 @ np.array([2.0, 0.6]) + rng.normal(0, 0.5, 200)
Xc2, yc2, _, _ = centre(X2, y2)

# Right panel: the ten diabetes features.
Xd, yd = load_diabetes(return_X_y=True)
names = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]

N = 84                                                   # 7 s at 12 fps


BUDGET = np.r_[np.linspace(1.0, 0.05, 67), np.full(N - 67, 0.05)]   # ball size, 100% -> 5%


def path_by_budget(X, y, penalty):
    """Weights whose norm (L2 for ridge, L1 for lasso) is BUDGET times the unpenalised norm.

    Equal ball sizes are a fairer comparison than equal alpha: the two penalties are
    on different scales. Compute a dense path, then interpolate alpha on log scale.
    """
    order = 2 if penalty == "l2" else 1
    dense = np.geomspace(1e-9, 1e3, 600) * lasso_alpha_max(X, y)   # 1e-9: unpenalised
    norms = np.linalg.norm(path(X, y, dense, penalty), ord=order, axis=1)
    ratio = norms / norms[0]                       # falls from 1 towards 0 as alpha grows
    log_alpha = np.interp(BUDGET, ratio[::-1], np.log(dense)[::-1])  # interp needs ascending x
    return path(X, y, np.exp(log_alpha), penalty)


l2_2d, l1_2d = path_by_budget(X2, y2, "l2"), path_by_budget(X2, y2, "l1")
l2_d, l1_d = path_by_budget(Xd, yd, "l2"), path_by_budget(Xd, yd, "l1")

fig, (ax, ax_bar) = plt.subplots(1, 2, figsize=(10.5, 4.8), gridspec_kw={"width_ratios": [1, 1.25]})

# Data-term contours over (w1, w2).
W1, W2 = np.meshgrid(np.linspace(-0.6, 2.6, 160), np.linspace(-1.2, 2.0, 160))
Wg = np.stack([W1.ravel(), W2.ravel()], 1)
Z = (((yc2[None, :] - Wg @ Xc2.T) ** 2).mean(1) / 2).reshape(W1.shape)
ax.contour(W1, W2, Z, levels=np.quantile(Z, np.linspace(0.01, 0.7, 10)),
           colors=ps.INK_2, linewidths=0.7, alpha=0.4)
ax.axhline(0, color=ps.INK_2, lw=0.8)
ax.axvline(0, color=ps.INK_2, lw=0.8)
ax.plot(*l2_2d[0], "*", ms=16, color=ps.INK, zorder=5)
ax.annotate("unpenalised", l2_2d[0], xytext=(4, 8), textcoords="offset points", fontsize=12)
circle = ax.add_patch(Circle((0, 0), 1, fill=False, ec=ps.BLUE, lw=2))
diamond = ax.add_patch(Polygon(np.zeros((4, 2)), closed=True, fill=False, ec=ps.ORANGE, lw=2))
(dot_l2,) = ax.plot([], [], "o", ms=9, color=ps.BLUE, mec=ps.SURFACE, mew=1.5, zorder=6)
(dot_l1,) = ax.plot([], [], "o", ms=9, color=ps.ORANGE, mec=ps.SURFACE, mew=1.5, zorder=6)
ax.set(xlim=(-0.6, 2.6), ylim=(-1.2, 2.0), xlabel="$w_1$", ylabel="$w_2$", aspect="equal")
ax.set_title("solution = where contours meet the ball", loc="left", fontsize=14)
ax.grid(False)

xs = np.arange(len(names))
bars_l2 = ax_bar.bar(xs - 0.2, l2_d[0], width=0.38, color=ps.BLUE, label="L2 (ridge)")
bars_l1 = ax_bar.bar(xs + 0.2, l1_d[0], width=0.38, color=ps.ORANGE, label="L1 (lasso)")
ax_bar.axhline(0, color=ps.INK_2, lw=0.8)
ax_bar.set_xticks(xs, names, fontsize=12)
ax_bar.set(ylim=(-900, 900), ylabel="weight")
ax_bar.legend(loc="upper left", ncol=2, fontsize=13)
ax_bar.set_title("diabetes data: 10 weights", loc="left", fontsize=14)
ax_bar.grid(axis="x", visible=False)
caption = fig.text(0.02, 0.95, "", fontsize=16)


def draw(k):
    w2, w1 = l2_2d[k], l1_2d[k]
    circle.set_radius(np.linalg.norm(w2))                # ball that just touches the L2 solution
    t = np.abs(w1).sum()
    diamond.set_xy([(t, 0), (0, t), (-t, 0), (0, -t)])   # L1 ball through the L1 solution
    dot_l2.set_data([w2[0]], [w2[1]])
    dot_l1.set_data([w1[0]], [w1[1]])
    for bar, h in zip(bars_l2, l2_d[k], strict=True):
        bar.set_height(h)
    for bar, h in zip(bars_l1, l1_d[k], strict=True):
        bar.set_height(h)
    zeros = int(np.sum(l1_d[k] == 0))
    caption.set_text(f"ball size {BUDGET[k]:4.0%} of unpenalised   weights at exactly 0:   "
                     f"lasso {zeros}/10   ridge {int(np.sum(l2_d[k] == 0))}/10")


fig.subplots_adjust(left=0.07, right=0.99, bottom=0.12, top=0.82, wspace=0.25)
anim = FuncAnimation(fig, draw, frames=N)
ps.save_gif(anim, str(Path(__file__).with_name("demo.gif")), N)
