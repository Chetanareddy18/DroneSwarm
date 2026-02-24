import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

# ==========================
# PARAMETERS
# ==========================
NUM_DRONES = 35
AREA_SIZE = 100
MAX_SPEED = 2.0

BASE_RADIUS = 18
MAX_RADIUS = 45

BATTERY_DRAIN = 0.4
LEADER_DRAIN = 0.8

HEAL_BATTERY = 40
MIN_HELPERS = 4

DT = 1.0
MAX_FRAMES = 350

# ==========================
# INIT
# ==========================
positions = np.random.rand(NUM_DRONES, 2) * AREA_SIZE
velocities = np.random.randn(NUM_DRONES, 2) * 0.3

battery = np.random.uniform(80, 100, NUM_DRONES)
alive = np.ones(NUM_DRONES)

leaders = np.zeros(NUM_DRONES)
leaders[np.random.choice(NUM_DRONES, 5, replace=False)] = 1

comm_radius = BASE_RADIUS
edge_artists = []

# ==========================
# HELPERS
# ==========================
def limit_speed(v):
    s = np.linalg.norm(v)
    return v if s < MAX_SPEED else v / s * MAX_SPEED

def build_graph(radius):
    adj = np.zeros((NUM_DRONES, NUM_DRONES))
    for i in range(NUM_DRONES):
        if not alive[i]:
            continue
        for j in range(NUM_DRONES):
            if i != j and alive[j]:
                if np.linalg.norm(positions[i] - positions[j]) < radius:
                    adj[i, j] = 1
    return adj

def get_global_state(adj):
    alive_ratio = alive.sum() / NUM_DRONES
    degrees = adj.sum(axis=1)
    avg_degree = degrees[alive == 1].mean() if alive.sum() else 0
    density = adj.sum() / (NUM_DRONES * (NUM_DRONES - 1))
    isolated = np.sum((degrees == 0) & (alive == 1))
    avg_battery = battery[alive == 1].mean() if alive.sum() else 0
    leader_ratio = leaders[alive == 1].sum() / max(alive.sum(), 1)
    return np.array([
        alive_ratio,
        avg_degree,
        density,
        isolated,
        avg_battery,
        leader_ratio
    ])

# ==========================
# RL POLICY (PLACEHOLDER)
# ==========================
def rl_policy(state):
    alive_ratio, avg_degree, density, isolated, avg_battery, leader_ratio = state

    if isolated > 3:
        return 1          # increase radius
    if avg_battery < 35:
        return 3          # heal
    if leader_ratio < 0.1:
        return 4          # elect leader
    if density > 0.5:
        return 2          # reduce radius
    return 0

# ==========================
# UPDATE
# ==========================
def update(frame):
    global positions, velocities, comm_radius, edge_artists

    # -------- MOVEMENT --------
    velocities += np.random.randn(NUM_DRONES, 2) * 0.05
    center = np.array([AREA_SIZE / 2, AREA_SIZE / 2])
    velocities += 0.01 * (center - positions)
    velocities[:] = np.array([limit_speed(v) for v in velocities])
    positions[:] += velocities * DT
    positions[:] = np.clip(positions, 0, AREA_SIZE)

    # -------- BATTERY --------
    battery[:] -= BATTERY_DRAIN + leaders * LEADER_DRAIN
    dead = np.where((battery <= 0) & (alive == 1))[0]
    alive[dead] = 0
    battery[dead] = 0

    # -------- GRAPH --------
    adj = build_graph(comm_radius)

    # -------- RL DECISION --------
    state = get_global_state(adj)
    action = rl_policy(state)

    if action == 1:
        comm_radius = min(comm_radius + 3, MAX_RADIUS)
    elif action == 2:
        comm_radius = max(comm_radius - 3, BASE_RADIUS)
    elif action == 3:
        for i in range(NUM_DRONES):
            if alive[i] == 0:
                helpers = np.sum(
                    (alive == 1) &
                    (np.linalg.norm(positions - positions[i], axis=1) < comm_radius)
                )
                if helpers >= MIN_HELPERS:
                    alive[i] = 1
                    battery[i] = HEAL_BATTERY
    elif action == 4:
        candidates = np.where((alive == 1) & (leaders == 0))[0]
        if len(candidates) > 0:
            leaders[random.choice(candidates)] = 1

    # -------- VIS --------
    for ln in edge_artists:
        ln.remove()
    edge_artists = []

    for i in range(NUM_DRONES):
        for j in range(NUM_DRONES):
            if adj[i, j]:
                ln, = ax.plot(
                    [positions[i, 0], positions[j, 0]],
                    [positions[i, 1], positions[j, 1]],
                    linewidth=0.5,
                    alpha=0.35
                )
                edge_artists.append(ln)

    colors = []
    for i in range(NUM_DRONES):
        if alive[i] == 0:
            colors.append("black")
        elif leaders[i]:
            colors.append("red")
        elif battery[i] < 40:
            colors.append("green")
        else:
            colors.append("blue")

    scat.set_offsets(positions)
    scat.set_color(colors)
    ax.set_title(
        f"LEVEL-5 Adaptive Intelligence | Action={action} | Radius={comm_radius}"
    )
    return scat,

# ==========================
# VIS
# ==========================
fig, ax = plt.subplots()
ax.set_xlim(0, AREA_SIZE)
ax.set_ylim(0, AREA_SIZE)

scat = ax.scatter(positions[:, 0], positions[:, 1], s=70)
ani = FuncAnimation(fig, update, frames=MAX_FRAMES, interval=40)
plt.show()
