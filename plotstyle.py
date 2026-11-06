"""Shared look for every demo.gif: light background, large type, one colour system."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3, in order

FPS = 12
MAX_SECONDS = 8


def apply() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "text.color": INK,
        "font.size": 15,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "legend.frameon": False,
        "lines.linewidth": 2,
    })


def save_gif(anim, path: str, n_frames: int) -> None:
    """Save with Pillow and refuse to write anything longer than MAX_SECONDS."""
    from matplotlib.animation import PillowWriter

    assert n_frames / FPS <= MAX_SECONDS, f"{n_frames} frames is over {MAX_SECONDS}s"
    anim.save(path, writer=PillowWriter(fps=FPS), dpi=90)
