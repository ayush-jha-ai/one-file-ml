"""Render demo.gif: attention as a soft lookup. A rotating, lengthening query blends six values."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from model import attention

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for plotstyle
import plotstyle as ps  # noqa: E402

ps.apply()

# Six keys on a circle. Each value is placed at its key's position, so the output point
# (a weighted average of values) can be drawn in the same picture.
angles = np.radians([90, 150, 200, 260, 320, 20])
K = 1.6 * np.column_stack([np.cos(angles), np.sin(angles)])
V = K.copy()
names = [f"k{j + 1}" for j in range(6)]

N = 90                                                       # 7.5 s
phase = np.linspace(0, 2 * np.pi, N)
length = np.r_[np.linspace(0.3, 0.3, 20), np.linspace(0.3, 4.0, 50), np.full(20, 4.0)]
Q = (length * np.column_stack([np.cos(phase + 1.2), np.sin(phase + 1.2)]).T).T
outs, weights = zip(*(attention(q[None, :], K, V) for q in Q), strict=True)
outs = np.array([o[0] for o in outs])
weights = np.array([w[0] for w in weights])

fig, (ax, ax_w) = plt.subplots(1, 2, figsize=(10.5, 4.8), gridspec_kw={"wspace": 0.25})
links = [ax.plot([], [], color=ps.ORANGE, solid_capstyle="round")[0] for _ in range(6)]
ax.scatter(*K.T, s=110, color=ps.INK_2, zorder=3)
for (kx, ky), name in zip(K, names, strict=True):
    ax.annotate(name, (kx, ky), xytext=(1.18 * kx, 1.18 * ky), ha="center", va="center",
                fontsize=13, color=ps.INK)
query = ax.annotate("", xy=(0, 0), xytext=(0, 0),
                    arrowprops={"arrowstyle": "-|>", "color": ps.BLUE, "lw": 2.5})
q_label = ax.text(0, 0, "query", color=ps.BLUE, fontsize=13)
(trail,) = ax.plot([], [], color=ps.ORANGE, lw=1, alpha=0.5)
(out_pt,) = ax.plot([], [], "o", ms=12, color=ps.ORANGE, mec=ps.SURFACE, mew=2, zorder=4)
ax.set(xlim=(-2.4, 2.4), ylim=(-2.4, 2.4), aspect="equal", xticks=[], yticks=[])
ax.set_title("output = weighted average of values", loc="left", fontsize=14)
ax.grid(False)
for side in ("left", "bottom"):
    ax.spines[side].set_visible(False)

bars = ax_w.bar(names, np.zeros(6), color=ps.INK_2, width=0.6)
ax_w.set(ylim=(0, 1), ylabel="attention weight")
ax_w.set_title("softmax(q · k / √d)", loc="left", fontsize=14)
ax_w.grid(axis="x", visible=False)
ax_w.set_axisbelow(True)
caption = fig.text(0.02, 0.95, "", fontsize=16)


def draw(i):
    q, w = Q[i], weights[i]
    tip = q / np.linalg.norm(q) * min(0.5 + 0.4 * length[i], 2.0)   # drawn length, capped
    query.xy, query.xyann = (tip[0], tip[1]), (0.0, 0.0)
    q_label.set_position((tip[0] * 1.1 + 0.05, tip[1] * 1.1 + 0.05))
    for link, k, wj in zip(links, K, w, strict=True):
        link.set_data([outs[i, 0], k[0]], [outs[i, 1], k[1]])
        link.set_linewidth(0.3 + 7 * wj)
        link.set_alpha(min(1.0, 0.15 + wj))
    trail.set_data(outs[: i + 1, 0], outs[: i + 1, 1])
    out_pt.set_data([outs[i, 0]], [outs[i, 1]])
    top = int(np.argmax(w))
    for j, bar in enumerate(bars):
        bar.set_height(w[j])
        bar.set_color(ps.BLUE if j == top else ps.INK_2)
    caption.set_text(f"attention   query length {length[i]:.1f}   largest weight {w[top]:.2f}"
                     f"   (longer query = sharper lookup)")


fig.subplots_adjust(left=0.03, right=0.98, bottom=0.1, top=0.82)
anim = FuncAnimation(fig, draw, frames=N)
ps.save_gif(anim, str(Path(__file__).with_name("demo.gif")), N)
