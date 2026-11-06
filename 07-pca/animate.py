"""Render demo.gif: sweep a direction through 180 degrees; variance peaks where error is least."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from model import PCA, variance_along

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for plotstyle
import plotstyle as ps  # noqa: E402

ps.apply()

rng = np.random.default_rng(0)
theta = np.radians(60)                                      # main direction of the cloud
R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
X = (rng.normal(size=(150, 2)) * [2.2, 0.7]) @ R + np.array([1.0, 0.5])
mean = X.mean(axis=0)
Xc = X - mean
total = np.sum(Xc**2) / (len(X) - 1)                        # total variance, = var + error
pc1 = PCA(1).fit(X).components_[0]
pc1_angle = np.degrees(np.arctan2(pc1[1], pc1[0])) % 180

angles = np.linspace(0, 180, 85)                            # 85 frames = 7.1 s
dirs = np.column_stack([np.cos(np.radians(angles)), np.sin(np.radians(angles))])
var = np.array([variance_along(X, u) for u in dirs])
err = total - var                                           # Pythagoras, see README

fig, (ax, ax_c) = plt.subplots(1, 2, figsize=(10.5, 4.8), gridspec_kw={"wspace": 0.28})
ax.scatter(*X.T, s=14, color=ps.INK_2, alpha=0.6, linewidths=0)
(axis_line,) = ax.plot([], [], color=ps.BLUE, lw=2.5)
show = rng.choice(len(X), 40, replace=False)                # draw residuals for 40 points
residuals = [ax.plot([], [], color=ps.ORANGE, lw=1, alpha=0.8)[0] for _ in show]
(proj_pts,) = ax.plot([], [], "o", ms=4, color=ps.BLUE)
ax.set(xlim=(-5, 7), ylim=(-5, 6), xlabel="$x_1$", ylabel="$x_2$", aspect="equal")
ax.set_title("project onto a direction", loc="left")

ax_c.axhline(total, color=ps.INK_2, lw=1, ls=":")
ax_c.text(178, total * 1.03, "total variance", color=ps.INK_2, fontsize=12, ha="right")
ax_c.axvline(pc1_angle, color=ps.INK_2, lw=1, ls="--")
ax_c.text(pc1_angle - 2, total * 1.03, "1st component", color=ps.INK_2, fontsize=12,
          ha="right")
ax_c.plot(angles, var, color=ps.BLUE, lw=1, alpha=0.25)
ax_c.plot(angles, err, color=ps.ORANGE, lw=1, alpha=0.25)
(c_var,) = ax_c.plot([], [], color=ps.BLUE, lw=2.5, label="variance along the line")
(c_err,) = ax_c.plot([], [], color=ps.ORANGE, lw=2.5, label="squared distance to the line")
ax_c.legend(loc="center right", fontsize=12)
ax_c.set(xlim=(0, 180), ylim=(0, total * 1.12), xticks=range(0, 181, 45),
         xlabel="angle of the line (degrees)", ylabel="per point, averaged")
ax_c.set_title("the same angle wins both", loc="left")
caption = fig.text(0.02, 0.95, "", fontsize=16)


def draw(i):
    u = dirs[i]
    axis_line.set_data(mean[0] + np.array([-12, 12]) * u[0], mean[1] + np.array([-12, 12]) * u[1])
    proj = mean + np.outer(Xc[show] @ u, u)
    for line, p, q in zip(residuals, X[show], proj, strict=True):
        line.set_data([p[0], q[0]], [p[1], q[1]])
    proj_pts.set_data(proj[:, 0], proj[:, 1])
    c_var.set_data(angles[: i + 1], var[: i + 1])
    c_err.set_data(angles[: i + 1], err[: i + 1])
    caption.set_text(f"PCA   angle {angles[i]:5.1f}°   variance {var[i]:.2f}   "
                     f"distance² {err[i]:.2f}   sum {total:.2f}")


fig.subplots_adjust(left=0.06, right=0.98, bottom=0.13, top=0.82)
anim = FuncAnimation(fig, draw, frames=len(angles))
ps.save_gif(anim, str(Path(__file__).with_name("demo.gif")), len(angles))
