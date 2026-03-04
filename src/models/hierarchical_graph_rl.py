import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
import numpy as np
import os


# ======================================
# Graph Encoder (uses adjacency + features)
# ======================================
class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, adj):
        h = torch.matmul(adj, x)
        h = F.relu(self.fc1(h))
        z = self.fc2(h)
        return z.mean(dim=0)   # Graph-level embedding


# ======================================
# DQN Network
# ======================================
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# ======================================
# Replay Buffer
# ======================================
class ReplayBuffer:
    def __init__(self, capacity=5000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(args)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states),
            torch.tensor(actions),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)


# ======================================
# Reward Function
# ======================================
def compute_reward(adj_before, adj_after):

    connectivity_before = adj_before.sum()
    connectivity_after = adj_after.sum()

    reward = (connectivity_after - connectivity_before) * 0.1
    return reward


# ======================================
# Main Training
# ======================================
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_nodes = 10
    feature_dim = 8
    latent_dim = 16

    action_dim = 6
    # 0: Line
    # 1: Circle
    # 2: V
    # 3: Repair best edge
    # 4: Remove random edge
    # 5: Do nothing

    encoder = GraphEncoder(feature_dim, 32, latent_dim).to(device)

    policy_net = DQN(latent_dim, action_dim).to(device)
    target_net = DQN(latent_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    replay_buffer = ReplayBuffer()

    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.05

    episodes = 200
    batch_size = 32
    target_update = 10

    for episode in range(episodes):

        # Random initial graph
        x = torch.randn(num_nodes, feature_dim).to(device)
        adj = torch.randint(0, 2, (num_nodes, num_nodes)).float().to(device)
        adj = torch.triu(adj, 1)
        adj = adj + adj.T
        adj.fill_diagonal_(0)

        state = encoder(x, adj).detach()

        total_reward = 0

        for step in range(5):

            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                with torch.no_grad():
                    action = policy_net(state).argmax().item()

            adj_before = adj.clone()

            # ========= ACTION LOGIC =========
            if action == 3:
                # Repair: add random edge
                i, j = random.randint(0, num_nodes-1), random.randint(0, num_nodes-1)
                if i != j:
                    adj[i,j] = 1
                    adj[j,i] = 1

            elif action == 4:
                # Remove random edge
                i, j = random.randint(0, num_nodes-1), random.randint(0, num_nodes-1)
                if i != j:
                    adj[i,j] = 0
                    adj[j,i] = 0

            # Other actions (formation) are symbolic here

            reward = compute_reward(adj_before, adj)
            total_reward += reward

            next_state = encoder(x, adj).detach()
            done = step == 4

            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state

            # ========= TRAIN =========
            if len(replay_buffer) >= batch_size:

                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                states = states.to(device)
                next_states = next_states.to(device)
                actions = actions.to(device)
                rewards = rewards.to(device)
                dones = dones.to(device)

                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

                with torch.no_grad():
                    next_actions = policy_net(next_states).argmax(1)
                    next_q = target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
                    target = rewards + gamma * next_q * (1 - dones)

                loss = F.mse_loss(q_values, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")

    os.makedirs("data/models", exist_ok=True)
    torch.save(policy_net.state_dict(), "data/models/hierarchical_rl.pth")

    print("\n✅ PERFECT MIXED RL TRAINED")