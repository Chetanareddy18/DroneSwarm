import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# ==========================
# PARAMETERS
# ==========================
NUM_DRONES = 35
AREA_SIZE = 100
MAX_SPEED = 2.0

NEIGHBOR_RADIUS = 25
SEPARATION_DIST = 6

W_SEP = 1.2
W_ALIGN = 1.0
W_COH = 2.5
W_INERTIA = 0.6
DT = 1.0

SAVE_EVERY = 10
MAX_FRAMES = 300

# ==========================
# DATA DIR
# ==========================
GRAPH_DIR = "data/graphs"
os.makedirs(GRAPH_DIR, exist_ok=True)

# ==========================
# INITIALIZE
# ==========================
positions = np.random.rand(NUM_DRONES, 2) * AREA_SIZE
velocities = np.random.randn(NUM_DRONES, 2) * 0.3

def limit_speed(v):
    speed = np.linalg.norm(v)
    if speed > MAX_SPEED:
        return v * (MAX_SPEED / (speed + 1e-6))
    return v

# ==========================
# GRAPH BUILD
# ==========================
def build_graph(pos):
    N = len(pos)
    adj = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(pos[i] - pos[j])
            if d < NEIGHBOR_RADIUS:
                adj[i, j] = adj[j, i] = 1
    return adj

# ==========================
# VIS SETUP
# ==========================
fig, ax = plt.subplots()
ax.set_xlim(0, AREA_SIZE)
ax.set_ylim(0, AREA_SIZE)
ax.set_title("LEVEL-2: Drone Communication Graph")

scat = ax.scatter(positions[:, 0], positions[:, 1],
                  s=70, c="dodgerblue", zorder=3)

edge_lines = []

# ==========================
# UPDATE LOOP
# ==========================
def update(frame):
    global positions, velocities, edge_lines

    new_vel = np.zeros_like(velocities)

    for i in range(NUM_DRONES):
        diff = positions - positions[i]
        dist = np.linalg.norm(diff, axis=1)

        neighbors = (dist > 0) & (dist < NEIGHBOR_RADIUS)
        close = (dist > 0) & (dist < SEPARATION_DIST)

        sep = -np.sum(diff[close] / (dist[close][:, None] + 1e-5), axis=0) if np.any(close) else 0
        align = np.mean(velocities[neighbors], axis=0) if np.any(neighbors) else 0
        coh = np.mean(positions[neighbors], axis=0) - positions[i] if np.any(neighbors) else 0

        steering = W_SEP * sep + W_ALIGN * align + W_COH * coh
        vel = W_INERTIA * velocities[i] + (1 - W_INERTIA) * steering
        new_vel[i] = limit_speed(vel)

    velocities[:] = new_vel
    positions[:] += velocities * DT

    # bounce
    for i in range(NUM_DRONES):
        for d in range(2):
            if positions[i, d] < 0 or positions[i, d] > AREA_SIZE:
                velocities[i, d] *= -1

    # --------------------------
    # GRAPH + EDGES
    # --------------------------
    adj = build_graph(positions)

    for ln in edge_lines:
        ln.remove()
    edge_lines.clear()

    for i in range(NUM_DRONES):
        for j in range(i + 1, NUM_DRONES):
            if adj[i, j]:
                ln, = ax.plot(
                    [positions[i, 0], positions[j, 0]],
                    [positions[i, 1], positions[j, 1]],
                    color="gray", alpha=0.25, linewidth=0.8, zorder=1
                )
                edge_lines.append(ln)

    # save graph
    if frame % SAVE_EVERY == 0:
        node_features = np.hstack([
            positions,
            velocities,
            np.ones((NUM_DRONES, 1))
        ])
        np.savez(
            f"{GRAPH_DIR}/graph_{frame:04d}.npz",
            node_features=node_features,
            adjacency=adj
        )
        print(f"📦 Saved graph at frame {frame}")

    scat.set_offsets(positions)
    return scat, *edge_lines

ani = FuncAnimation(fig, update, frames=MAX_FRAMES, interval=40)
plt.show()
