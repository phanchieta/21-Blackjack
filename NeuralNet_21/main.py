import gymnasium as gym
import torch
from train import BlackjackNet, train_agent
from load_save import save_checkpoint, load_checkpoint
from visualize import plot_dqn_strategy

def main():
    env = gym.make('Blackjack-v1')
    # Input: Sum, Dealer, Ace, Count = 4
    model = BlackjackNet(input_dim=4, output_dim=2)
    optimizer = torch.optim.Adam(model.parameters())

    if not load_checkpoint(model, optimizer):
        print("Training new Neural Net...")
        optimizer = train_agent(env, model, episodes=10000)
        save_checkpoint(model, optimizer)

    print("\nVisualizing Strategy...")
    plot_dqn_strategy(model, usable_ace=False)

if __name__ == "__main__":
    main()