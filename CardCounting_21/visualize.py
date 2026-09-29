import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib.pyplot as plt
from common.plot_style import AQUA, BLUE, INK, INK_MUTED, ORANGE, RED, apply_style, savefig

apply_style()


def plot_bankroll(hilo_trajectory, flat_trajectory, save_path=None):
    plt.figure(figsize=(9, 5.5))
    plt.plot(flat_trajectory, color=ORANGE, linewidth=1.6, label="Flat bet (no counting)")
    plt.plot(hilo_trajectory, color=AQUA, linewidth=1.6, label="Hi-Lo spread bet")
    plt.title("Bankroll Over Hands: Hi-Lo Spread vs. Flat Bet")
    plt.xlabel("Hand #")
    plt.ylabel("Bankroll (units)")
    plt.legend(frameon=False)
    plt.grid(axis="y", alpha=0.6)
    if save_path:
        savefig(save_path)
    else:
        plt.show()
        plt.close()


def plot_true_count_distribution(counts, save_path=None):
    plt.figure(figsize=(9, 5))
    bins = np.arange(-8.5, 9.5, 1)
    plt.hist(counts, bins=bins, color=BLUE, edgecolor="#fcfcfb")
    plt.title("True Count Distribution Across Simulated Shoes")
    plt.xlabel("True Count (at start of hand)")
    plt.ylabel("Hands")
    plt.grid(axis="y", alpha=0.6)
    if save_path:
        savefig(save_path)
    else:
        plt.show()
        plt.close()


def plot_edge_vs_true_count(edge_by_count, save_path=None):
    counts = sorted(edge_by_count.keys())
    edges = [edge_by_count[c] * 100 for c in counts]
    plt.figure(figsize=(9, 5.5))
    plt.axhline(0, color=INK_MUTED, linewidth=1)
    plt.fill_between(counts, edges, 0, where=[e >= 0 for e in edges], color=BLUE, alpha=0.25, interpolate=True)
    plt.fill_between(counts, edges, 0, where=[e <= 0 for e in edges], color=RED, alpha=0.25, interpolate=True)
    plt.plot(counts, edges, color=INK, linewidth=1.5, marker="o", markersize=4)
    plt.title("Player Edge vs. True Count (basic strategy, no play deviations)")
    plt.xlabel("True Count")
    plt.ylabel("Player Edge")
    plt.gca().yaxis.set_major_formatter(lambda y, _: f"{y:.0f}%")
    plt.grid(axis="y", alpha=0.6)
    if save_path:
        savefig(save_path)
    else:
        plt.show()
        plt.close()
