from load_save import load_checkpoint
from train import train
from visualize import plot_strategy, plot_training_curve
import gymnasium as gym
import numpy as np

CHECKPOINT = "checkpoints\\blackjack_brain.npy"

print("Trying to load model..")
q_table, history = load_checkpoint(filename=CHECKPOINT)
if not q_table:
    q_table, history = train()
else:
    print("Model loaded!")

# Evaluation Loop
wins, losses, draws = 0, 0, 0
test_episodes = 10000

env = gym.make('Blackjack-v1', natural=False, sab=False)

for _ in range(test_episodes):
    state, info = env.reset()
    done = False
    reward = 0.0

    while not done:
        # No more epsilon! Always take the best move (Argmax)
        action = int(np.argmax(q_table.get(state, [0.0, 0.0])))
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    if reward > 0:
        wins += 1
    elif reward < 0:
        losses += 1
    else:
        draws += 1

print(f"--- Results after {test_episodes} games ---")
print(f"Win Rate:  {(wins/test_episodes)*100:.2f}%")
print(f"Loss Rate: {(losses/test_episodes)*100:.2f}%")
print(f"Draw Rate: {(draws/test_episodes)*100:.2f}%")
print(f"States learned: {len(q_table)}")

plot_strategy(q_table, usable_ace=False, save_path="results\\strategy_heatmap_hard.png")
plot_strategy(q_table, usable_ace=True, save_path="results\\strategy_heatmap_soft.png")
if history:
    plot_training_curve(history, save_path="results\\training_curve.png")
