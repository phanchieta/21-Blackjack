import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.dqn_model import BlackjackNet, ReplayBuffer, normalize_obs


def _train_step(model, target_model, optimizer, buffer, batch_size, gamma, rng):
    states, actions, rewards, next_states, dones = buffer.sample(batch_size, rng)

    q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        # Double DQN: pick the next action with the online net, evaluate it with
        # the target net -- decouples action-selection from value-estimation,
        # which curbs the overestimation bias plain DQN is prone to.
        next_actions = torch.argmax(model(next_states), dim=1, keepdim=True)
        next_q = target_model(next_states).gather(1, next_actions).squeeze(1)
        targets = rewards + gamma * next_q * (1.0 - dones)

    loss = nn.functional.mse_loss(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(model, env, n_hands=500):
    model.eval()
    wins = 0
    reward = 0.0
    with torch.no_grad():
        for _ in range(n_hands):
            obs, _ = env.reset()
            done = False
            while not done:
                state = normalize_obs(*obs)
                q = model(torch.FloatTensor(state).unsqueeze(0))
                action = int(torch.argmax(q).item())
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            wins += reward > 0
    model.train()
    return wins / n_hands


def train_agent(env, model, episodes=150_000, buffer_capacity=50_000, batch_size=64,
                 train_freq=4, target_sync_every=500, gamma=1.0, lr=1e-3,
                 eval_every=None, eval_hands=500, verbose=True):
    """Real DQN: replay buffer + target network + Double DQN, trained every
    train_freq-th env step (not every step -- gradient updates dominate wall
    time far more than experience generation for a network this small, so
    decoupling the two is what makes a 150k-episode run affordable).

    gamma defaults to 1.0 (undiscounted), not the usual <1 -- a hand is a
    short, strictly episodic sequence (at most a handful of hit steps before
    terminating), and discounting within it has no real interpretation.
    Worse, it actively biases play: a "hit" that resolves 2 Bellman hops
    later than an immediate "stand" gets its eventual reward discounted by
    gamma^2, which systematically undervalues hitting relative to standing
    on exactly the close, multi-step decisions (e.g. stiff totals like 16)
    where that few-percent penalty is enough to flip the optimal action.
    """
    target_model = BlackjackNet()
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_capacity, state_dim=4)
    rng = np.random.default_rng()

    epsilon = 1.0
    epsilon_min = 0.01
    decay_horizon = int(episodes * 0.6)
    epsilon_decay = (epsilon_min / epsilon) ** (1 / decay_horizon)

    eval_every = eval_every or max(episodes // 50, 1)
    env_steps = 0
    train_steps = 0
    history = []  # [(episode, greedy_win_rate), ...]

    for ep in range(episodes):
        obs, _ = env.reset()
        state = normalize_obs(*obs)
        done = False

        while not done:
            if random.random() < epsilon:
                action = random.randint(0, 1)
            else:
                with torch.no_grad():
                    q = model(torch.FloatTensor(state).unsqueeze(0))
                    action = int(torch.argmax(q).item())

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = normalize_obs(*next_obs)

            buffer.push(state, action, reward, next_state, done)
            state = next_state
            env_steps += 1

            if len(buffer) >= batch_size and env_steps % train_freq == 0:
                _train_step(model, target_model, optimizer, buffer, batch_size, gamma, rng)
                train_steps += 1
                if train_steps % target_sync_every == 0:
                    target_model.load_state_dict(model.state_dict())

        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        if ep % eval_every == 0:
            history.append((ep, evaluate(model, env, n_hands=eval_hands)))
            if verbose:
                print(f"Episode {ep}/{episodes}  eps={epsilon:.3f}  win_rate={history[-1][1]:.1%}")

    history.append((episodes, evaluate(model, env, n_hands=max(eval_hands, 2000))))
    if verbose:
        print(f"Training finished over {episodes} hands. Final win_rate={history[-1][1]:.1%}")
    return history
