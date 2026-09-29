import numpy as np


def save_checkpoint(q_table, history, filename="checkpoints\\blackjack_brain.npy"):
    np.save(filename, {"q_table": q_table, "history": history})
    print(f"Model saved to {filename}")


def load_checkpoint(filename="checkpoints\\blackjack_brain.npy"):
    try:
        data = np.load(filename, allow_pickle=True).item()
        return data["q_table"], data["history"]
    except FileNotFoundError:
        print("No saved model found. Starting with a fresh brain.")
        return {}, []
