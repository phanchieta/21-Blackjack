import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib.pyplot as plt
import seaborn as sns
from common.plot_style import ACTION_CMAP, BASELINE, BLUE, apply_style, savefig

apply_style()


def plot_strategy(q_table, usable_ace=False, save_path=None):
    strategy = np.zeros((10, 10))  # Rows: Player sum (12-21), Cols: Dealer card (1-10)
    for player_sum in range(12, 22):
        for dealer_card in range(1, 11):
            state = (player_sum, dealer_card, usable_ace)
            if state in q_table:
                # 0 = Stand, 1 = Hit. We take the index of the max Q-value.
                strategy[player_sum - 12, dealer_card - 1] = np.argmax(q_table[state])

    plt.figure(figsize=(9, 7))
    ax = sns.heatmap(strategy, annot=True, fmt=".0f", xticklabels=range(1, 11), yticklabels=range(12, 22),
                      cmap=ACTION_CMAP, vmin=0, vmax=1, cbar=False, linewidths=1, linecolor="#fcfcfb")
    ax.set_title(f"Q-Learning Strategy (Usable Ace: {usable_ace})")
    ax.set_xlabel("Dealer Showing Card")
    ax.set_ylabel("Player Sum")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ACTION_CMAP.colors]
    ax.legend(handles, ["Stand", "Hit"], loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)

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
    plt.title("Q-Learning Training Progress")
    plt.xlabel("Training Episode")
    plt.ylabel("Greedy Win Rate")
    plt.gca().yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    plt.grid(axis="y", alpha=0.6)

    if save_path:
        savefig(save_path)
    else:
        plt.show()
        plt.close()
