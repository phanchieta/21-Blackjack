import os

import torch


def save_checkpoint(model, history, filename="checkpoints\\blackjack_dqn.pth"):
    torch.save({"model_state_dict": model.state_dict(), "history": history}, filename)
    print(f"Model saved to {filename}")


def load_checkpoint(model, filename="checkpoints\\blackjack_dqn.pth"):
    """Returns the saved training history on success (and loads weights into
    model in place), or None if there's nothing to load."""
    if os.path.exists(filename):
        checkpoint = torch.load(filename, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint: {filename}")
        return checkpoint["history"]
    print("No checkpoint found. Starting fresh.")
    return None
