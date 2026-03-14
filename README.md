# 21
A repo for computer heuristics in 21
## Simple 21 - AI Blackjack Agent (Q-Learning)

A reinforcement learning project that trains an agent to play 21-Blackjack optimally using a **Q-Learning** algorithm.

### How it Works
Instead of hard-coding rules, this agent uses the **Bellman Equation** to learn from experience. It plays hundreds of thousands of hands, receiving a reward of `+1` for a win and `-1` for a loss.

### Training Progress
The agent uses an **Epsilon-Greedy** strategy:
1. **Exploration**: Initially, it takes random actions to discover the game's mechanics.
2. **Exploitation**: Over time, it "decays" its randomness and starts relying on its calculated Q-values to make the mathematically superior move.

### Results
After 500,000 episodes, the agent generates a strategy map that closely mirrors "Basic Strategy" used by professional players.

### Limitations
It doesn't remember the past. It treats every hand as if it's coming from a freshly shuffled infinite deck

### How to Run
1. Install dependencies: `pip install gymnasium numpy matplotlib seaborn`
2. Run the main script: `python main.py`

## Neural Net Blackjack
An AI that uses a Deep Neural Network to master Blackjack, considering player totals, dealer cards, and deck counting.

### Getting Started
1. Install dependencies:
   `pip install torch gymnasium matplotlib seaborn numpy`
2. Run the project:
   `python main.py`

### The AI Architecture
The model is a Feed-Forward Neural Network with 2 hidden layers (64 neurons each). It uses the **MSE Loss** of the Bellman Equation to optimize its decision-making.

### Limitations
To truly make the AI "count" cards, you would need to wrap the Gymnasium environment in a custom class that tracks the deck state (since the default Blackjack-v1 uses an infinite deck)