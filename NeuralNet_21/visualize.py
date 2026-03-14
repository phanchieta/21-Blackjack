import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np

def plot_dqn_strategy(model, usable_ace=False):
    strategy = np.zeros((10, 10))
    model.eval()
    with torch.no_grad():
        for p_sum in range(12, 22):
            for d_card in range(1, 11):
                # State: (PlayerSum, DealerCard, Ace, Count)
                state = torch.FloatTensor([p_sum, d_card, float(usable_ace), 0])
                action = torch.argmax(model(state)).item()
                strategy[p_sum-12, d_card-1] = action

    sns.heatmap(strategy, annot=True, xticklabels=range(1, 11), yticklabels=range(12, 22), cmap="RdYlGn_r")
    plt.title(f"DQN Strategy (Ace: {usable_ace})")
    plt.show()