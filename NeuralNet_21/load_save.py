import torch
import os

def save_checkpoint(model, optimizer, filename="checkpoints\\blackjack_dqn.pth"):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)
    print(f"Model saved to {filename}")

def load_checkpoint(model, optimizer, filename="checkpoints\\blackjack_dqn.pth"):
    if os.path.exists(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint: {filename}")
        return True
    print("No checkpoint found. Starting fresh.")
    return False