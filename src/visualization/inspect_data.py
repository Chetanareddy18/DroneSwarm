import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# =========================
# LOAD DATA
# =========================
positions = np.load("data/positions.npy")
adjacency = np.load("data/adjacency.npy")
battery = np.load("data/battery.npy")

T, N, _ = positions.shape

print("Timesteps:", T)
print("Number of drones:", N)

# =========================
# VISUALIZE ONE TIMESTEP
# =========================
t = 50  # change this to explore different times

pos = positions[t]
adj = adjacency[t]

# Create graph
G = nx.Graph()

for i in range(N):
    G.add_node(i, pos=(pos[i, 0], pos[i, 1]))

for i in range(N):
    for j in range(i + 1, N):
        if adj[i, j] == 1:
            G.add_edge(i, j)

# =========================
# PLOT
# =========================
plt.figure(figsize=(6, 6))

node_colors = ["red" if battery[t, i] <= 0 else "green" for i in range(N)]

nx.draw(
    G,
    pos=nx.get_node_attributes(G, "pos"),
    with_labels=True,
    node_color=node_colors,
    node_size=400,
)

plt.title(f"Drone Swarm at timestep {t}")
plt.show()
