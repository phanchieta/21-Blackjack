"""Shared DQN pieces: the network architecture and a replay buffer.

Lives in common/ (not inside NeuralNet_21/) so compare.py can reconstruct the
model from a saved state_dict without importing NeuralNet_21 as a package.
"""
import numpy as np
import torch
import torch.nn as nn


class BlackjackNet(nn.Module):
    def __init__(self, input_dim=4, output_dim=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.fc(x)


class ReplayBuffer:
    """Fixed-capacity ring buffer of (state, action, reward, next_state, done)."""

    def __init__(self, capacity, state_dim=4):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.cursor = 0

    def push(self, state, action, reward, next_state, done):
        i = self.cursor
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = float(done)
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        idx = rng.integers(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.states[idx]),
            torch.from_numpy(self.actions[idx]),
            torch.from_numpy(self.rewards[idx]),
            torch.from_numpy(self.next_states[idx]),
            torch.from_numpy(self.dones[idx]),
        )

    def __len__(self):
        return self.size


def normalize_obs(player_sum, dealer_upcard, usable_ace, true_count):
    """Scale raw observation fields to roughly [-1, 1] for stabler training."""
    return (
        player_sum / 21.0,
        dealer_upcard / 10.0,
        float(usable_ace),
        max(-3.0, min(3.0, true_count)) / 3.0,
    )
