"""Shared chart styling so every chart in this repo -- regardless of which
script generated it -- reads as one system. Palette is the validated default
from this session's dataviz skill (categorical order passes CVD + normal-
vision checks in light mode; validated via scripts/validate_palette.js).
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
RED = "#e34948"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# One fixed color per strategy, reused across every chart in the repo so an
# entity's color never changes meaning between figures.
STRATEGY_COLORS = {
    "Q-Learning (no count)": BLUE,
    "Basic Strategy (no count)": ORANGE,
    "Hi-Lo Counting": AQUA,
    "DQN (count-aware)": YELLOW,
    "Hi-Lo, favorable rules": MAGENTA,
}

# Binary action heatmaps (Stand=0, Hit=1) -- a 2-class categorical, not a
# magnitude ramp, so two fixed hues rather than a red-green gradient.
ACTION_CMAP = ListedColormap([BLUE, ORANGE])
ACTION_LABELS = ["Stand", "Hit"]


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "axes.titlecolor": INK,
        "grid.color": GRIDLINE,
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def savefig(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"Saved {path}")
    plt.close()
