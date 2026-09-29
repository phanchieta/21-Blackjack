import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.dqn_model import BlackjackNet, normalize_obs
from common.shoe_env import ShoeBlackjackEnv

from load_save import load_checkpoint, save_checkpoint
from train import train_agent
from visualize import plot_dqn_strategy, plot_dqn_strategy_by_count, plot_training_curve

CHECKPOINT = "checkpoints\\blackjack_dqn.pth"


def main():
    # 6-deck shoe with Hi-Lo true-count tracking -- this is what makes the 4th
    # input feature (true_count) real signal instead of the hardcoded 0 the
    # old infinite-deck version had no choice but to use.
    env = ShoeBlackjackEnv(num_decks=6, penetration=0.75, natural=True)
    model = BlackjackNet(input_dim=4, output_dim=2)

    history = load_checkpoint(model, filename=CHECKPOINT)
    if history is None:
        print("Training new Neural Net (DQN: replay buffer + target network + Double DQN)...")
        history = train_agent(env, model, episodes=400_000, buffer_capacity=100_000)
        save_checkpoint(model, history, filename=CHECKPOINT)
    else:
        print("Model loaded!")

    # Evaluation loop (win/loss/draw), mirroring Simple_21's report
    wins, losses, draws = 0, 0, 0
    test_hands = 10000
    model.eval()
    with torch.no_grad():
        for _ in range(test_hands):
            obs, _ = env.reset()
            done = False
            reward = 0.0
            while not done:
                state = torch.FloatTensor(normalize_obs(*obs)).unsqueeze(0)
                action = int(torch.argmax(model(state)).item())
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            if reward > 0:
                wins += 1
            elif reward < 0:
                losses += 1
            else:
                draws += 1

    print(f"--- Results after {test_hands} games (6-deck shoe) ---")
    print(f"Win Rate:  {(wins/test_hands)*100:.2f}%")
    print(f"Loss Rate: {(losses/test_hands)*100:.2f}%")
    print(f"Draw Rate: {(draws/test_hands)*100:.2f}%")

    print("\nVisualizing strategy...")
    plot_dqn_strategy(model, usable_ace=False, save_path="results\\strategy_heatmap_hard.png")
    plot_dqn_strategy(model, usable_ace=True, save_path="results\\strategy_heatmap_soft.png")
    plot_dqn_strategy_by_count(model, usable_ace=False, save_path="results\\strategy_by_true_count.png")
    if history:
        plot_training_curve(history, save_path="results\\training_curve.png")


if __name__ == "__main__":
    main()
