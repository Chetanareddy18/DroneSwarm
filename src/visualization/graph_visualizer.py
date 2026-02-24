import numpy as np
import matplotlib.pyplot as plt
import os

# =========================
# PATH SETUP (IMPORTANT)
# =========================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

GRAPH_PATH = os.path.join(PROJECT_ROOT, "data", "graphs", "graph_002.npz")

# =========================
# LOAD GRAPH DATA
# =========================
g = np.load(GRAPH_PATH)

node_features = g["node_features"]   # (N, 3)
adjacency = g["adjacency"]           # (N, N)
alive = g["alive"]                   # (N,)

positions = node_features[:, :2]
num_drones = len(positions)

# =========================
# PLOT GRAPH
# =========================
plt.figure(figsize=(7, 7))

# Communication links
for i in range(num_drones):
    for j in range(i + 1, num_drones):
        if adjacency[i, j] == 1:
            plt.plot(
                [positions[i, 0], positions[j, 0]],
                [positions[i, 1], positions[j, 1]],
                color="gray",
                alpha=0.4
            )

# Drones
for i in range(num_drones):
    if alive[i] == 1:
        plt.scatter(positions[i, 0], positions[i, 1], c="blue", s=120)
        plt.text(positions[i, 0] + 2, positions[i, 1] + 2, f"D{i}")
    else:
        plt.scatter(positions[i, 0], positions[i, 1], c="red", s=120)

# Styling
plt.title("Drone Swarm Communication Graph")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.axis("equal")
plt.show()
