import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from hierarchical_graph_rl import GraphEncoder, DQN
from graph_diffusion import GraphDiffusion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==========================================
# CONFIGURATION
# ==========================================
NUM_EPISODES = 15000
latent_dim = 16
action_dim = 6
feature_dim = 8
learning_rate = 1e-3
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05

# ==========================================
# PATHS
# ==========================================
BASE_PATH = r"C:\Users\Chetana\OneDrive\Desktop\DRONE_SWARM"
MODEL_DIR = os.path.join(BASE_PATH, "src", "models", "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

rl_save_path = os.path.join(MODEL_DIR, "hierarchical_rl.pth")
diffusion_path = os.path.join(MODEL_DIR, "diffusion_trained.pth")

# ==========================================
# LOAD DIFFUSION MODEL
# ==========================================
diffusion_model = GraphDiffusion(in_features=1, hidden_dim=128).to(device)
diffusion_model.load_state_dict(torch.load(diffusion_path, map_location=device))
diffusion_model.eval()
print("✅ Diffusion model loaded")

# ==========================================
# MODELS
# ==========================================
encoder = GraphEncoder(feature_dim, 32, latent_dim).to(device)
rl_model = DQN(latent_dim, action_dim).to(device)

optimizer = optim.Adam(rl_model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# ==========================================
# FORMATION FUNCTIONS
# ==========================================
def build_line(n):
    adj = torch.zeros((n, n))
    for i in range(n - 1):
        adj[i, i + 1] = 1
        adj[i + 1, i] = 1
    return adj

def build_circle(n):
    adj = build_line(n)
    adj[0, n - 1] = 1
    adj[n - 1, 0] = 1
    return adj

def build_v(n):
    adj = torch.zeros((n, n))
    mid = n // 2
    for i in range(1, mid + 1):
        adj[0, i] = 1
        adj[i, 0] = 1
    for i in range(mid + 1, n):
        adj[1, i] = 1
        adj[i, 1] = 1
    return adj

def formation_score(adj):
    degrees = adj.sum(dim=1)
    variance = torch.var(degrees)
    connectivity = adj.sum()
    return connectivity - variance * 2

# ==========================================
# RANDOM GRAPH GENERATOR (TRAINING ENV)
# ==========================================
def generate_random_graph(n):
    adj = torch.randint(0, 2, (n, n)).float()
    adj = torch.triu(adj, diagonal=1)
    adj = adj + adj.T
    adj.fill_diagonal_(0)
    return adj

# ==========================================
# TRAINING LOOP
# ==========================================
for episode in range(NUM_EPISODES):

    num_nodes = random.choice([10, 15, 20, 25, 35])
    adj = generate_random_graph(num_nodes).to(device)
    x = torch.randn(num_nodes, 8).to(device)

    state = encoder(x, adj).unsqueeze(0)

    # Epsilon-greedy action
    if random.random() < epsilon:
        action = random.randint(0, action_dim - 1)
    else:
        with torch.no_grad():
            q_values = rl_model(state)
            action = torch.argmax(q_values).item()

    adj_before = adj.clone()
    score_before = formation_score(adj_before)

    # ACTION EXECUTION
    if action == 0:
        adj = build_line(num_nodes).to(device)

    elif action == 1:
        adj = build_circle(num_nodes).to(device)

    elif action == 2:
        adj = build_v(num_nodes).to(device)

    elif action == 3:
        with torch.no_grad():
            adj = diffusion_model(adj.unsqueeze(0)).squeeze(0)

    elif action == 4:
        adj = (adj > 0.7).float()

    elif action == 5:
        pass

    score_after = formation_score(adj)
    reward = (score_after - score_before) * 0.05

    next_state = encoder(x, adj).unsqueeze(0)

    with torch.no_grad():
        target_q = rl_model(next_state)
        target_value = reward + gamma * torch.max(target_q)

    current_q = rl_model(state)[0, action]

    loss = loss_fn(current_q, target_value)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    if episode % 500 == 0:
        print(f"Episode {episode} | Reward: {round(reward.item(),4)} | Epsilon: {round(epsilon,3)}")

# ==========================================
# SAVE MODEL
# ==========================================
torch.save(rl_model.state_dict(), rl_save_path)
print("\n🔥 Training Completed & Model Saved!")