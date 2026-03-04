import os
import json
import csv
import random
import numpy as np
import torch

# ==========================================================
# PATH CONFIG
# ==========================================================
BASE_PATH = r"C:\Users\Chetana\OneDrive\Desktop\DRONE_SWARM"
MODEL_DIR = os.path.join(BASE_PATH, "src", "models", "saved_models")
OUTPUT_DIR = os.path.join(BASE_PATH, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==========================================================
# IMPORT MODELS (MUST MATCH YOUR TRAIN FILES)
# ==========================================================
from graph_diffusion import GraphDiffusion
from train_multi_agent_rl import DroneAgent

# ==========================================================
# LOAD DIFFUSION MODEL
# ==========================================================
diffusion_model_path = os.path.join(MODEL_DIR, "diffusion_trained.pth")

diffusion = GraphDiffusion(in_features=1, hidden_dim=128).to(device)
diffusion.load_state_dict(torch.load(diffusion_model_path, map_location=device))
diffusion.eval()

# ==========================================================
# LOAD MULTI AGENT POLICY
# ==========================================================
policy_model_path = os.path.join(MODEL_DIR, "multi_agent_policy.pth")

policy = DroneAgent().to(device)
policy.load_state_dict(torch.load(policy_model_path, map_location=device))
policy.eval()

print("✅ Models loaded successfully")

# ==========================================================
# GRAPH UTILITIES
# ==========================================================
def generate_graph(n):
    adj = torch.randint(0, 2, (n, n)).float()
    adj = torch.triu(adj, diagonal=1)
    adj = adj + adj.T
    adj.fill_diagonal_(0)
    return adj

def formation_score(adj):
    degrees = adj.sum(dim=1)
    variance = torch.var(degrees)
    connectivity = adj.sum()
    return (connectivity - variance * 2).item()

def global_density(adj):
    n = adj.shape[0]
    return adj.sum() / (n * (n - 1))

# ==========================================================
# MULTI AGENT EXECUTION
# ==========================================================
def run_multi_agent(adj):
    n = adj.shape[0]

    for drone in range(n):

        degree = adj[drone].sum()
        neighbors = torch.where(adj[drone] == 1)[0]

        if len(neighbors) > 0:
            neighbor_mean = adj[neighbors].sum(dim=1).mean()
        else:
            neighbor_mean = torch.tensor(0.0, device=device)

        density = global_density(adj)

        state = torch.tensor(
            [degree, neighbor_mean, density],
            dtype=torch.float32,
            device=device
        )

        with torch.no_grad():
            q_values = policy(state)
            action = torch.argmax(q_values).item()

        # --- ACTIONS ---
        if action == 0:  # connect random
            target = random.randint(0, n - 1)
            if target != drone:
                adj[drone, target] = 1
                adj[target, drone] = 1

        elif action == 1:  # disconnect random neighbor
            if len(neighbors) > 0:
                idx = random.randint(0, len(neighbors) - 1)
                target = neighbors[idx]
                adj[drone, target] = 0
                adj[target, drone] = 0

        elif action == 2:  # repair lowest-degree node
            degrees = adj.sum(dim=1)
            target = torch.argmin(degrees)
            if target != drone:
                adj[drone, target] = 1
                adj[target, drone] = 1

        elif action == 3:  # idle
            pass

    return adj

# ==========================================================
# DIFFUSION REPAIR
# ==========================================================
def diffusion_repair(adj):
    adj_input = adj.unsqueeze(0).to(device)
    with torch.no_grad():
        repaired = diffusion(adj_input)
    repaired = repaired.squeeze(0)
    repaired = (repaired > 0.5).float()
    repaired.fill_diagonal_(0)
    return repaired

# ==========================================================
# 3D POSITION GENERATION (CIRCLE FORMATION)
# ==========================================================
def generate_positions(n):
    positions = []
    radius = 5.0

    for i in range(n):
        angle = 2 * np.pi * i / n
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = 2.0 + np.random.uniform(-0.5, 0.5)

        positions.append((float(x), float(y), float(z)))

    return positions

# ==========================================================
# MAIN PIPELINE
# ==========================================================
def main():

    NUM_DRONES = 100

    # 1️⃣ Generate initial graph
    adj = generate_graph(NUM_DRONES).to(device)

    initial_score = formation_score(adj)
    print("Initial Formation Score:", initial_score)

    # 2️⃣ Run Multi-Agent Control
    adj = run_multi_agent(adj)

    # 3️⃣ Run Diffusion Repair
    adj = diffusion_repair(adj)

    final_score = formation_score(adj)
    print("Final Formation Score:", final_score)

    # 4️⃣ Generate 3D positions
    positions = generate_positions(NUM_DRONES)

    # 5️⃣ Assign drone states
    drone_output = []
    for i in range(NUM_DRONES):

        state = "normal"

        if i == 0:
            state = "leader"

        if random.random() < 0.1:
            state = "failed"

        drone_output.append({
            "id": i,
            "x": positions[i][0],
            "y": positions[i][1],
            "z": positions[i][2],
            "state": state
        })

    # ======================================================
    # SAVE FILES
    # ======================================================

    # Drone JSON (Your Required Format)
    drone_json_path = os.path.join(OUTPUT_DIR, "drone_positions.json")
    with open(drone_json_path, "w") as f:
        json.dump(drone_output, f, indent=2)

    # CSV Summary
    csv_path = os.path.join(OUTPUT_DIR, "swarm_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["num_drones", "initial_score", "final_score"])
        writer.writerow([NUM_DRONES, initial_score, final_score])

    # JSON Summary
    summary_json_path = os.path.join(OUTPUT_DIR, "swarm_summary.json")
    summary = {
        "num_drones": NUM_DRONES,
        "initial_score": initial_score,
        "final_score": final_score
    }

    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n✅ All outputs saved in:", OUTPUT_DIR)


# ==========================================================
# RUN
# ==========================================================
if __name__ == "__main__":
    main()