import random

import gymnasium as gym
import numpy as np
from load_save import save_checkpoint


def get_q_values(q_table, state):
    if state not in q_table:
        q_table[state] = [0.0, 0.0]  # Start with no knowledge
    return q_table[state]


def evaluate(q_table, env, n_hands=500):
    """Greedy (epsilon=0) win rate, for tracking a training curve. Doesn't
    mutate q_table -- unseen states just fall back to the neutral default."""
    wins = 0
    for _ in range(n_hands):
        state, info = env.reset()
        done = False
        reward = 0.0
        while not done:
            action = int(np.argmax(q_table.get(state, [0.0, 0.0])))
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        wins += reward > 0
    return wins / n_hands


# 3. Training
def train():
    # Setup Environment
    env = gym.make('Blackjack-v1', natural=False, sab=False)

    # Hyperparameters
    learning_rate = 0.05
    discount_factor = 0.95
    epsilon = 0.2  # 20% chance to explore (try random moves)
    epsilon_min = 0.01
    episodes = 500000
    eval_every = 10000
    # Reach epsilon_min by 60% of the way through training, so the back half
    # of training is mostly-greedy and the Q-table actually settles instead
    # of oscillating on noisy exploratory episodes right up to the end.
    decay_horizon = int(episodes * 0.6)
    epsilon_decay = (epsilon_min / epsilon) ** (1 / decay_horizon)

    # Initialize Q-table: {(state): [q_value_for_stand, q_value_for_hit]}
    q_table = {}
    history = []  # [(episode, greedy_win_rate), ...] -- for the training-curve chart
    print("Training...")
    for i in range(episodes):
        state, info = env.reset()
        done = False

        while not done:
            # Epsilon-Greedy Action Selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(get_q_values(q_table, state)))

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # q learning update
            old_q = get_q_values(q_table, state)[action]
            # On a terminal transition there is no real "next state" to bootstrap
            # from -- must zero this out, especially since Gymnasium's stand
            # observation is literally identical to the pre-stand state (player
            # hand and dealer upcard don't change when you stand), so without
            # this the update would bootstrap standing's value off the SAME
            # state's own (possibly-higher) hit-value.
            next_max = 0.0 if done else np.max(get_q_values(q_table, next_state))

            # Bellman Equation update
            new_q = old_q + learning_rate * (reward + (discount_factor * next_max) - old_q)
            q_table[state][action] = new_q

            state = next_state

        # Gradually reduce epsilon (decay) every episode so it actually reaches
        # epsilon_min well before training ends, instead of still exploring
        # ~12% of the time on the final episode.
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        if i % eval_every == 0:
            history.append((i, evaluate(q_table, env, n_hands=2000)))

    history.append((episodes, evaluate(q_table, env, n_hands=2000)))
    print(f"Training finished over {episodes} hands.")
    save_checkpoint(q_table, history, filename="checkpoints\\blackjack_brain.npy")
    print("Save complete.")
    return q_table, history
