"""Render demo.gif: a two-hidden-layer network learning to separate a three-armed spiral."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from model import init_params, loss_and_grads, predict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for plotstyle
import plotstyle as ps  # noqa: E402

ps.apply()

# Three interleaved spiral arms, 100 points each.
rng = np.random.default_rng(0)
r = np.tile(np.linspace(0.05, 1, 100), 3)
labels = np.repeat([0, 1, 2], 100)
theta = labels * 2 * np.pi / 3 + r * 4 + rng.normal(0, 0.2, 300)
X = np.column_stack([r * np.cos(theta), r * np.sin(theta)])

STEPS, LR = 6000, 0.5
params = init_params([2, 32, 32, 3], np.random.default_rng(1))
snap_at = set(np.unique(np.round(np.geomspace(1, STEPS, 72)).astype(int)) - 1)
snaps, losses = {}, []
for step in range(STEPS):
    loss, grads = loss_and_grads(params, X, labels)
    losses.append(loss)
    if step in snap_at:
        snaps[step] = [[W.copy(), b.copy()] for W, b in params]
    params = [[W - LR * dW, b - LR * db] for (W, b), (dW, db) in zip(params, grads, strict=True)]
steps = sorted(snaps)
steps += [steps[-1]] * (84 - len(steps))                         # hold the end: 84 frames

gx, gy = np.meshgrid(np.linspace(-1.15, 1.15, 160), np.linspace(-1.15, 1.15, 160))
grid = np.column_stack([gx.ravel(), gy.ravel()])
fig, (ax, ax_l) = plt.subplots(1, 2, figsize=(10.5, 4.8), gridspec_kw={"wspace": 0.28})
image = ax.imshow(np.zeros_like(gx), extent=(-1.15, 1.15, -1.15, 1.15), origin="lower",
                  cmap=ListedColormap(["#b9d2f2", "#f8cdb9", "#bfe8d2"]), vmin=-0.5, vmax=2.5,
                  interpolation="nearest")
for c, colour in enumerate((ps.BLUE, ps.ORANGE, ps.AQUA)):
    ax.scatter(*X[labels == c].T, s=14, color=colour, edgecolor=ps.SURFACE, linewidth=0.5)
ax.set(xlim=(-1.15, 1.15), ylim=(-1.15, 1.15), xlabel="$x_1$", ylabel="$x_2$")
ax.set_title("network 2 → 32 → 32 → 3 (tanh)", loc="left", fontsize=14)
ax.grid(False)

ax_l.plot(np.arange(1, STEPS + 1), losses, color=ps.BLUE, lw=1, alpha=0.25)
(curve,) = ax_l.plot([], [], color=ps.BLUE, lw=2.5)
ax_l.set(xscale="log", yscale="log", xlim=(1, STEPS), xlabel="gradient descent step (log scale)",
         ylabel="cross-entropy loss")
ax_l.set_title("loss, computed by backpropagation", loc="left", fontsize=14)
caption = fig.text(0.02, 0.95, "", fontsize=16)


def draw(frame):
    k = steps[frame]
    image.set_data(predict(snaps[k], grid).reshape(gx.shape))
    curve.set_data(np.arange(1, k + 2), losses[: k + 1])
    acc = np.mean(predict(snaps[k], X) == labels)
    caption.set_text(f"neural network   step {k + 1:>4}   loss {losses[k]:.3f}   "
                     f"training accuracy {acc:.0%}")


fig.subplots_adjust(left=0.07, right=0.98, bottom=0.13, top=0.82)
anim = FuncAnimation(fig, draw, frames=len(steps))
ps.save_gif(anim, str(Path(__file__).with_name("demo.gif")), len(steps))
