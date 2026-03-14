import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

class BlackjackNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BlackjackNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

def train_agent(env, model, episodes=5000):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    epsilon = 1.0
    epsilon_decay = 0.9995
    gamma = 0.95

    for ep in range(episodes):
        state, _ = env.reset()
        # Add a dummy 'count' of 0 for basic Gymnasium env (or your custom count logic)
        state = np.append(state, [0]) 
        done = False
        
        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            
            # Epsilon-greedy
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = torch.argmax(model(state_t)).item()

            next_state, reward, term, trunc, _ = env.step(action)
            next_state = np.append(next_state, [0]) # Update count here in custom env
            
            # Simple DQN Update
            target = reward
            if not (term or trunc):
                next_state_t = torch.FloatTensor(next_state).unsqueeze(0)
                target = reward + gamma * torch.max(model(next_state_t)).item()
            
            current_q = model(state_t)[0][action]
            loss = criterion(current_q, torch.tensor(target, dtype=torch.float32))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state
            done = term or trunc
        
        epsilon = max(0.01, epsilon * epsilon_decay)
        if ep % 500 == 0:
            print(f"Episode {ep} completed...")
    
    return optimizer