import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import random

# ==========================
# PARAMETERS
# ==========================
NUM_DRONES = 35
AREA_SIZE = 100
MAX_SPEED = 1.8

NEIGHBOR_RADIUS = 25
MAX_DEGREE = 6
PACKET_LOSS = 0.25
LINK_DECAY = 0.02
LEADER_RATIO = 0.15

CENTER_FORCE = 0.004
DT = 1.0
SAVE_EVERY = 10
MAX_FRAMES = 300

# ==========================
# OUTPUT DIR (LEVEL-3)
# ==========================
GRAPH_DIR = "data/graphs_lvl3"
os.makedirs(GRAPH_DIR, exist_ok=True)

# ==========================
# INIT
# ==========================
positions = np.random.rand(NUM_DRONES, 2) * AREA_SIZE
velocities = np.random.randn(NUM_DRONES, 2) * 0.4

leaders = np.zeros(NUM_DRONES)
leaders[np.random.choice(NUM_DRONES, int(NUM_DRONES * LEADER_RATIO), replace=False)] = 1

link_age = {}
edge_artists = []

# ==========================
# HELPERS
# ==========================
def limit_speed(v):
    s = np.linalg.norm(v)
    return v if s < MAX_SPEED else v / s * MAX_SPEED

# ==========================
# GRAPH BUILD (STRESSED)
# ==========================
def build_graph(pos):
    N = len(pos)
    adj = np.zeros((N, N))

    for i in range(N):
        candidates = []

        for j in range(N):
            if i != j and np.linalg.norm(pos[i] - pos[j]) < NEIGHBOR_RADIUS:
                if random.random() > PACKET_LOSS:
                    candidates.append(j)

        random.shuffle(candidates)
        candidates = candidates[:MAX_DEGREE]

        for j in candidates:
            key = (i, j)
            age = link_age.get(key, 0)

            if random.random() > age:
                adj[i, j] = 1
                link_age[key] = age + LINK_DECAY
            else:
                link_age.pop(key, None)

    return adj

# ==========================
# UPDATE
# ==========================
def update(frame):
    global positions, velocities, edge_artists

    # random drift
    velocities += np.random.randn(NUM_DRONES, 2) * 0.05

    # weak centering (keep visible)
    center = np.array([AREA_SIZE / 2, AREA_SIZE / 2])
    velocities += CENTER_FORCE * (center - positions)

    velocities[:] = np.array([limit_speed(v) for v in velocities])
    positions[:] += velocities * DT
    positions[:] = np.clip(positions, 0, AREA_SIZE)

    adj = build_graph(positions)

    # ---------- CLEAR OLD EDGES ----------
    for artist in edge_artists:
        artist.remove()
    edge_artists = []

    # ---------- DRAW EDGES ----------
    for i in range(NUM_DRONES):
        for j in range(NUM_DRONES):
            if adj[i, j]:
                ln, = ax.plot(
                    [positions[i, 0], positions[j, 0]],
                    [positions[i, 1], positions[j, 1]],
                    linewidth=0.6,
                    alpha=0.4
                )
                edge_artists.append(ln)

    colors = ["red" if leaders[i] else "blue" for i in range(NUM_DRONES)]
    scat.set_offsets(positions)
    scat.set_color(colors)

    # ---------- SAVE DATA ----------
    if frame % SAVE_EVERY == 0:
        node_features = np.hstack([
            positions,
            velocities,
            leaders[:, None]
        ])

        np.savez(
            f"{GRAPH_DIR}/graph_{frame:04d}.npz",
            node_features=node_features,
            adjacency=adj
        )

        density = adj.sum() / (NUM_DRONES * (NUM_DRONES - 1))
        print(f"Frame {frame} | Density: {density:.3f}")

    return scat,

# ==========================
# VIS
# ==========================
fig, ax = plt.subplots()
ax.set_xlim(0, AREA_SIZE)
ax.set_ylim(0, AREA_SIZE)
ax.set_title("LEVEL-3: Dynamic Network Stress")

scat = ax.scatter(positions[:, 0], positions[:, 1], s=70)
ani = FuncAnimation(fig, update, frames=MAX_FRAMES, interval=40)
plt.show()
