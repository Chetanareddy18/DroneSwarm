import os
import numpy as np
import torch

from hierarchical_graph_rl import GraphEncoder, DQN
from graph_diffusion import GraphDiffusion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ======================================
# PATHS
# ======================================
BASE_PATH = r"C:\Users\Chetana\OneDrive\Desktop\DRONE_SWARM"
GRAPH_DIR = os.path.join(BASE_PATH, "data", "graphs")
MODEL_DIR = os.path.join(BASE_PATH, "src", "models", "saved_models")

rl_path = os.path.join(MODEL_DIR, "hierarchical_rl.pth")
diffusion_path = os.path.join(MODEL_DIR, "diffusion_trained.pth")

# ======================================
# RL ARCHITECTURE (MUST MATCH TRAINING)
# ======================================
latent_dim = 16
action_dim = 6
feature_dim = 8  # MUST stay 8 (same as training)

# ======================================
# LOAD RL MODEL
# ======================================
rl_model = DQN(latent_dim, action_dim).to(device)
rl_model.load_state_dict(torch.load(rl_path, map_location=device))
rl_model.eval()
print("✅ RL model loaded")

# ======================================
# LOAD DIFFUSION MODEL (Scalable)
# ======================================
diffusion_model = GraphDiffusion(in_features=1, hidden_dim=128).to(device)
diffusion_model.load_state_dict(torch.load(diffusion_path, map_location=device))
diffusion_model.eval()
print("✅ Diffusion model loaded")

# ======================================
# FORMATION BUILDERS
# ======================================
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

# ======================================
# FORMATION EVALUATION
# ======================================
def formation_score(adj):
    degrees = adj.sum(dim=1)
    variance = torch.var(degrees)
    connectivity = adj.sum()
    return connectivity - variance * 2

# ======================================
# LOAD GRAPHS
# ======================================
graph_files = sorted([f for f in os.listdir(GRAPH_DIR) if f.endswith(".npz")])
print(f"Found {len(graph_files)} graph files")

for file in graph_files:

    print("\n==============================")
    print("Processing:", file)

    data = np.load(os.path.join(GRAPH_DIR, file))

    adj_np = data["adjacency"]
    num_nodes = adj_np.shape[0]

    adj = torch.tensor(adj_np, dtype=torch.float32)
    adj = (adj > 0).float()
    adj.fill_diagonal_(0)
    adj = adj.to(device)

    # ======================================
    # IMPORTANT FIX: FORCE 8-D FEATURES
    # (DO NOT USE node_features FROM FILE)
    # ======================================
    x = torch.randn(num_nodes, 8).to(device)

    # Encoder (must match training)
    encoder = GraphEncoder(feature_dim, 32, latent_dim).to(device)
    encoder.eval()

    # ======================================
    # RL DECISION
    # ======================================
    with torch.no_grad():
        state = encoder(x, adj).unsqueeze(0)
        q_values = rl_model(state)
        action = torch.argmax(q_values).item()

    print("Chosen Action:", action)

    adj_before = adj.clone()
    score_before = formation_score(adj_before)

    # ======================================
    # ACTION EXECUTION
    # ======================================
    if action == 0:
        print("→ Line formation")
        adj = build_line(num_nodes).to(device)

    elif action == 1:
        print("→ Circle formation")
        adj = build_circle(num_nodes).to(device)

    elif action == 2:
        print("→ V formation")
        adj = build_v(num_nodes).to(device)

    elif action == 3:
        print("→ Repair using Diffusion")
        with torch.no_grad():
            adj = diffusion_model(adj.unsqueeze(0)).squeeze(0)

    elif action == 4:
        print("→ Remove weak edges")
        adj = (adj > 0.7).float()

    elif action == 5:
        print("→ Do Nothing")

    # ======================================
    # REWARD (FORMATION-BASED)
    # ======================================
    score_after = formation_score(adj)
    reward = (score_after - score_before) * 0.05

    print("Formation Score Before:", round(score_before.item(), 4))
    print("Formation Score After:", round(score_after.item(), 4))
    print("Reward:", round(reward.item(), 4))

print("\nProcessing completed.")