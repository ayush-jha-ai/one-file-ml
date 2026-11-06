"""Render demo.gif: polynomial degree rising from 0 to 12; bias falls, variance rises."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from model import exact_bias_variance, simulate, true_function

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for plotstyle
import plotstyle as ps  # noqa: E402

ps.apply()

N, NOISE, DEGREES, FRAMES_EACH = 20, 0.3, np.arange(13), 7     # 91 frames = 7.6 s
x_fixed = (np.arange(N) + 0.5) / N                             # same x, fresh noise each time
x_plot = np.linspace(0, 1, 300)

FLOOR = 1e-6                                                   # bottom of the log axis

# Left: real simulated fits. Right: exact bias^2 and variance (no simulation noise).
preds = {d: simulate(d, N, NOISE, x_plot, n_datasets=200, x_fixed=x_fixed) for d in DEGREES}
exact = [exact_bias_variance(x_fixed, x_plot, d, NOISE) for d in DEGREES]
bias_sq = np.array([b.mean() for b, _ in exact])
variance = np.array([v.mean() for _, v in exact])
total = bias_sq + variance + NOISE**2

fig, (ax, ax_err) = plt.subplots(1, 2, figsize=(10.5, 4.8), gridspec_kw={"wspace": 0.28})

fit_lines = [ax.plot([], [], color=ps.BLUE, lw=1, alpha=0.35)[0] for _ in range(25)]
(mean_line,) = ax.plot([], [], color=ps.ORANGE, lw=3, label="average fit")
ax.plot(x_plot, true_function(x_plot), color=ps.INK, lw=2, ls="--", label="truth f(x)")
ax.plot([], [], color=ps.BLUE, lw=1.5, label="25 fits, 25 datasets")
ax.set(xlim=(0, 1), ylim=(-2, 2), xlabel="x", ylabel="y")
ax.legend(loc="lower left", fontsize=12, handlelength=1.5)
ax.set_title("one model per training set", loc="left")

ax_err.axhline(NOISE**2, color=ps.INK_2, lw=1, ls=":")
ax_err.text(12.4, NOISE**2 * 0.55, "noise σ²", color=ps.INK_2, fontsize=12)
(c_bias,) = ax_err.plot([], [], color=ps.ORANGE, lw=2.5, marker="o", ms=5)
(c_var,) = ax_err.plot([], [], color=ps.BLUE, lw=2.5, marker="o", ms=5)
(c_tot,) = ax_err.plot([], [], color=ps.INK, lw=2.5, marker="o", ms=5)
labels = [ax_err.text(0, 0, s, fontsize=12, color=ps.INK) for s in ("bias²", "variance", "total")]
ax_err.set(xlim=(-0.5, 14.5), ylim=(FLOOR, 3), yscale="log", xlabel="polynomial degree",
           ylabel="expected squared error")
ax_err.set_xticks(range(0, 13, 2))
ax_err.set_title("error = bias² + variance + noise", loc="left")
caption = fig.text(0.02, 0.95, "", fontsize=16)


def draw(frame):
    d = DEGREES[min(frame // FRAMES_EACH, len(DEGREES) - 1)]
    for line, p in zip(fit_lines, preds[d][:25], strict=True):
        line.set_data(x_plot, p)
    mean_line.set_data(x_plot, preds[d].mean(axis=0))
    k = np.arange(d + 1)
    nudge = (1.0, 1 / 1.5, 1.5)                  # keep "variance" and "total" apart
    for curve, values, label, m in zip((c_bias, c_var, c_tot), (bias_sq, variance, total),
                                       labels, nudge, strict=True):
        shown = np.maximum(values[: d + 1], FLOOR)
        curve.set_data(k, shown)
        label.set_position((d + 0.35, max(shown[-1] * m, 1.6 * FLOOR)))
    labels[0].set_text("bias²" if bias_sq[d] > FLOOR else "bias² ≈ 0")
    caption.set_text(f"degree {d:>2}   bias² {bias_sq[d]:.1e}   variance {variance[d]:.3f}"
                     f"   total {total[d]:.3f}")


fig.subplots_adjust(left=0.07, right=0.98, bottom=0.13, top=0.82)
n_frames = len(DEGREES) * FRAMES_EACH
anim = FuncAnimation(fig, draw, frames=n_frames)
ps.save_gif(anim, str(Path(__file__).with_name("demo.gif")), n_frames)
