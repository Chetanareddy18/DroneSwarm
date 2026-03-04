import os
import random
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =====================================================
# CONFIG
# =====================================================
NUM_EPISODES = 12000
MAX_DRONES = 35
state_dim = 3          # local state: degree, neighbor_mean_degree, global_density
action_dim = 4         # 0: connect, 1: disconnect, 2: repair, 3: idle
gamma = 0.95
lr = 1e-3
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05

# =====================================================
# SHARED POLICY NETWORK
# =====================================================
class DroneAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)


policy = DroneAgent().to(device)
optimizer = optim.Adam(policy.parameters(), lr=lr)
loss_fn = nn.MSELoss()

# =====================================================
# ENVIRONMENT
# =====================================================
def generate_graph(n):
    adj = torch.randint(0, 2, (n, n)).float()
    adj = torch.triu(adj, diagonal=1)
    adj = adj + adj.T
    adj.fill_diagonal_(0)
    return adj

def global_density(adj):
    n = adj.shape[0]
    return adj.sum() / (n * (n - 1))

def formation_score(adj):
    degrees = adj.sum(dim=1)
    variance = torch.var(degrees)
    connectivity = adj.sum()
    return connectivity - variance * 2

# =====================================================
# TRAINING LOOP
# =====================================================
for episode in range(NUM_EPISODES):

    n = random.choice([10, 15, 20, 25, 35])
    adj = generate_graph(n).to(device)

    score_before = formation_score(adj)

    for drone in range(n):

        degree = adj[drone].sum()
        neighbors = torch.where(adj[drone] == 1)[0]

        if len(neighbors) > 0:
            neighbor_mean = adj[neighbors].sum(dim=1).mean()
        else:
            neighbor_mean = torch.tensor(0.0).to(device)

        density = global_density(adj)

        state = torch.tensor(
            [degree, neighbor_mean, density],
            dtype=torch.float32
        ).to(device)

        # Epsilon-Greedy
        if random.random() < epsilon:
            action = random.randint(0, action_dim - 1)
        else:
            with torch.no_grad():
                q = policy(state)
                action = torch.argmax(q).item()

        # === ACTIONS ===
        if action == 0:
            target = random.randint(0, n - 1)
            if target != drone:
                adj[drone, target] = 1
                adj[target, drone] = 1

        elif action == 1:
            if degree > 0:
                target = neighbors[random.randint(0, len(neighbors) - 1)]
                adj[drone, target] = 0
                adj[target, drone] = 0

        elif action == 2:
            # simple repair: connect to lowest-degree node
            degrees = adj.sum(dim=1)
            target = torch.argmin(degrees)
            if target != drone:
                adj[drone, target] = 1
                adj[target, drone] = 1

        elif action == 3:
            pass

        # === REWARD ===
        score_after = formation_score(adj)
        reward = (score_after - score_before) * 0.02
        score_before = score_after

        next_degree = adj[drone].sum()
        next_neighbors = torch.where(adj[drone] == 1)[0]
        if len(next_neighbors) > 0:
            next_neighbor_mean = adj[next_neighbors].sum(dim=1).mean()
        else:
            next_neighbor_mean = torch.tensor(0.0).to(device)

        next_density = global_density(adj)

        next_state = torch.tensor(
            [next_degree, next_neighbor_mean, next_density],
            dtype=torch.float32
        ).to(device)

        with torch.no_grad():
            target_q = policy(next_state)
            target_value = reward + gamma * torch.max(target_q)

        current_q = policy(state)[action]

        loss = loss_fn(current_q, target_value)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    if episode % 500 == 0:
        print(f"Episode {episode} | Epsilon {round(epsilon,3)}")

# =====================================================
# SAVE MODEL
# =====================================================
BASE_PATH = r"C:\Users\Chetana\OneDrive\Desktop\DRONE_SWARM"
MODEL_DIR = os.path.join(BASE_PATH, "src", "models", "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

torch.save(policy.state_dict(), os.path.join(MODEL_DIR, "multi_agent_policy.pth"))

print("\n🔥 Multi-Agent Training Completed!")