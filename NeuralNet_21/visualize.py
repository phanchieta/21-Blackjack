import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib.pyplot as plt
import seaborn as sns
from common.dqn_model import normalize_obs
from common.plot_style import ACTION_CMAP, BASELINE, BLUE, apply_style, savefig

apply_style()


def _strategy_grid(model, usable_ace, true_count):
    grid = np.zeros((10, 10))
    model.eval()
    with torch.no_grad():
        for p_sum in range(12, 22):
            for d_card in range(1, 11):
                state = normalize_obs(p_sum, d_card, usable_ace, true_count)
                q = model(torch.FloatTensor(state))
                grid[p_sum - 12, d_card - 1] = torch.argmax(q).item()
    return grid


def _draw_heatmap(ax, grid, title):
    sns.heatmap(grid, annot=True, fmt=".0f", xticklabels=range(1, 11), yticklabels=range(12, 22),
                cmap=ACTION_CMAP, vmin=0, vmax=1, cbar=False, ax=ax, linewidths=1, linecolor="#fcfcfb")
    ax.set_title(title)
    ax.set_xlabel("Dealer Showing Card")


def plot_dqn_strategy(model, usable_ace=False, true_count=0.0, save_path=None):
    grid = _strategy_grid(model, usable_ace, true_count)
    plt.figure(figsize=(9, 7))
    ax = plt.gca()
    _draw_heatmap(ax, grid, f"DQN Strategy (Usable Ace: {usable_ace}, True Count: {true_count:+.0f})")
    ax.set_ylabel("Player Sum")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ACTION_CMAP.colors]
    ax.legend(handles, ["Stand", "Hit"], loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)

    if save_path:
        savefig(save_path)
    else:
        plt.show()
        plt.close()


def plot_dqn_strategy_by_count(model, usable_ace=False, true_counts=(-3, 0, 3), save_path=None):
    """The payoff visual: does the net's play actually shift with the count?"""
    fig, axes = plt.subplots(1, len(true_counts), figsize=(6 * len(true_counts), 6.5), sharey=True)
    for ax, tc in zip(axes, true_counts):
        grid = _strategy_grid(model, usable_ace, tc)
        _draw_heatmap(ax, grid, f"True Count {tc:+d}")
    axes[0].set_ylabel("Player Sum")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ACTION_CMAP.colors]
    fig.legend(handles, ["Stand", "Hit"], loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2, frameon=False)
    fig.suptitle(f"DQN Strategy Shift by True Count (Usable Ace: {usable_ace})", y=1.15, fontsize=13, fontweight="bold")

    if save_path:
        savefig(save_path)
    else:
        plt.show()
        plt.close()


def plot_training_curve(history, save_path=None):
    episodes, win_rates = zip(*history)
    plt.figure(figsize=(9, 5))
    plt.plot(episodes, win_rates, color=BLUE, linewidth=2)
    plt.axhline(win_rates[-1], color=BASELINE, linewidth=1, linestyle="--")
    plt.title("DQN Training Progress")
    plt.xlabel("Training Episode")
    plt.ylabel("Greedy Win Rate")
    plt.gca().yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    plt.grid(axis="y", alpha=0.6)

    if save_path:
        savefig(save_path)
    else:
        plt.show()
        plt.close()
