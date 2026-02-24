import numpy as np
import os
import csv
import random

# ==========================
# PARAMETERS
# ==========================
NUM_DRONES = 35
AREA_SIZE = 100
MAX_SPEED = 2.0

BASE_RADIUS = 20
MAX_RADIUS = 50
MIN_RADIUS = 12

BATTERY_DRAIN = 0.4
LEADER_DRAIN = 0.8

EPISODES = 1
TIMESTEPS = 500

GRAPH_SAVE_EVERY = 5

# ==========================
# DIRS
# ==========================
BASE_DIR = "dataset_lvl5"
GRAPH_DIR = f"{BASE_DIR}/graphs"
os.makedirs(GRAPH_DIR, exist_ok=True)

CSV_PATH = f"{BASE_DIR}/global_states.csv"

# ==========================
# INIT CSV
# ==========================
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "alive_ratio",
        "avg_degree",
        "density",
        "isolated_nodes",
        "avg_battery",
        "leader_ratio",
        "comm_radius",
        "action",
        "next_alive_ratio",
        "next_density"
    ])

# ==========================
# HELPERS
# ==========================
def limit_speed(v):
    s = np.linalg.norm(v)
    return v if s < MAX_SPEED else v / s * MAX_SPEED

def build_graph(pos, alive, radius):
    N = len(pos)
    adj = np.zeros((N, N))
    for i in range(N):
        if not alive[i]:
            continue
        for j in range(N):
            if i != j and alive[j]:
                if np.linalg.norm(pos[i] - pos[j]) < radius:
                    adj[i, j] = 1
    return adj

def graph_stats(adj, alive):
    N = alive.sum()
    if N <= 1:
        return 0, 0, N
    degrees = adj.sum(axis=1)
    isolated = np.sum((degrees == 0) & (alive == 1))
    avg_degree = degrees.sum() / N
    density = adj.sum() / (N * (N - 1))
    return avg_degree, density, isolated

# ==========================
# MAIN LOOP
# ==========================
for ep in range(EPISODES):

    positions = np.random.rand(NUM_DRONES, 2) * AREA_SIZE
    velocities = np.random.randn(NUM_DRONES, 2) * 0.4

    battery = np.random.uniform(70, 100, NUM_DRONES)
    alive = np.ones(NUM_DRONES)

    leaders = np.zeros(NUM_DRONES)
    leaders[np.random.choice(NUM_DRONES, 5, replace=False)] = 1

    comm_radius = BASE_RADIUS

    for t in range(TIMESTEPS):

        # -------- BUILD GRAPH --------
        adj = build_graph(positions, alive, comm_radius)

        avg_degree, density, isolated = graph_stats(adj, alive)

        alive_ratio = alive.sum() / NUM_DRONES
        avg_battery = battery[alive == 1].mean() if alive.sum() else 0
        leader_ratio = leaders.sum() / NUM_DRONES

        # -------- POLICY (RULE BASED) --------
        if alive_ratio < 0.6:
            action = 4
        elif isolated > 6:
            action = 1
        elif density > 0.6:
            action = 2
        elif leader_ratio < 0.1:
            action = 3
        else:
            action = 0

        # -------- APPLY ACTION --------
        if action == 1:
            comm_radius = min(comm_radius + 3, MAX_RADIUS)
        elif action == 2:
            comm_radius = max(comm_radius - 3, MIN_RADIUS)
        elif action == 3:
            extra = np.random.choice(
                np.where(leaders == 0)[0], 1, replace=False
            )
            leaders[extra] = 1
        elif action == 4:
            comm_radius = MAX_RADIUS

        # -------- MOVE --------
        velocities += np.random.randn(NUM_DRONES, 2) * 0.05
        velocities = np.array([limit_speed(v) for v in velocities])
        positions += velocities
        positions = np.clip(positions, 0, AREA_SIZE)

        # -------- BATTERY --------
        battery -= BATTERY_DRAIN + leaders * LEADER_DRAIN
        dead = np.where(battery <= 0)[0]
        alive[dead] = 0

        # -------- NEXT STATE --------
        adj_next = build_graph(positions, alive, comm_radius)
        _, next_density, _ = graph_stats(adj_next, alive)
        next_alive_ratio = alive.sum() / NUM_DRONES

        # -------- SAVE GLOBAL STATE --------
        with open(CSV_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                alive_ratio,
                avg_degree,
                density,
                isolated,
                avg_battery,
                leader_ratio,
                comm_radius,
                action,
                next_alive_ratio,
                next_density
            ])

        # -------- SAVE GRAPH --------
        if t % GRAPH_SAVE_EVERY == 0:
            node_features = np.hstack([
                positions,
                velocities,
                battery[:, None],
                alive[:, None],
                leaders[:, None]
            ])
            np.savez(
                f"{GRAPH_DIR}/graph_{ep}_{t}.npz",
                node_features=node_features,
                adjacency=adj
            )

print("✅ DATASET GENERATION COMPLETE")
