"""Consistent visual language for every figure in the project."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK = "#12212e"
MUTED = "#5b6b7a"
GRID = "#dfe5ea"
ACCENT = "#c8553d"          # delay / risk
ACCENT_2 = "#2f6f8f"        # volume / neutral
GOOD = "#3f8f6f"
SEQ = ["#2f6f8f", "#4d8ba6", "#7aa8ba", "#e3b23c", "#d98d3a", "#c8553d", "#8c2f2f"]

PALETTE = {"delay": ACCENT, "volume": ACCENT_2, "good": GOOD, "ink": INK}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.labelsize": 10,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.size": 10,
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
    })


def despine(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
