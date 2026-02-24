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

BASE_RADIUS = 20
HEAL_RADIUS = 45

BATTERY_DRAIN = 0.5
LEADER_DRAIN = 1.0

SHOCK_INTERVAL = 70
SHOCK_SIZE = 4

CENTER_FORCE = 0.01
PANIC_FORCE = 0.08

HEAL_DELAY = 40
MIN_HELPERS = 4
REVIVE_BATTERY = 35

DT = 1.0
MAX_FRAMES = 400

# ==========================
# INIT
# ==========================
positions = np.random.rand(NUM_DRONES, 2) * AREA_SIZE
velocities = np.random.randn(NUM_DRONES, 2) * 0.3

battery = np.random.uniform(80, 100, NUM_DRONES)
alive = np.ones(NUM_DRONES)
death_time = np.full(NUM_DRONES, -1)

leaders = np.zeros(NUM_DRONES)
leaders[np.random.choice(NUM_DRONES, 5, replace=False)] = 1

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

# ==========================
# UPDATE
# ==========================
def update(frame):
    global edge_artists, battery, alive, death_time, leaders

    # -------- SHOCK FAILURES --------
    if frame % SHOCK_INTERVAL == 0 and frame > 0:
        victims = np.random.choice(
            np.where(alive == 1)[0],
            min(SHOCK_SIZE, int(alive.sum())),
            replace=False
        )
        alive[victims] = 0
        battery[victims] = 0
        death_time[victims] = frame
        print(f"⚡ Shock at frame {frame}: {victims}")

    # -------- BATTERY DRAIN --------
    battery[:] -= BATTERY_DRAIN + leaders * LEADER_DRAIN

    dying = np.where((battery <= 0) & (alive == 1))[0]
    alive[dying] = 0
    death_time[dying] = frame
    battery[dying] = 0

    # -------- GRAPH --------
    radius = HEAL_RADIUS if alive.sum() < 0.65 * NUM_DRONES else BASE_RADIUS
    adj = build_graph(radius)

    # -------- SELF-HEALING --------
    for i in range(NUM_DRONES):
        if alive[i] == 0 and death_time[i] >= 0:
            helpers = np.sum(
                (alive == 1) &
                (np.linalg.norm(positions - positions[i], axis=1) < HEAL_RADIUS)
            )
            if helpers >= MIN_HELPERS and frame - death_time[i] >= HEAL_DELAY:
                alive[i] = 1
                battery[i] = REVIVE_BATTERY
                death_time[i] = -1
                leaders[i] = 0
                print(f"🟢 Drone {i} self-healed at frame {frame}")

    # -------- MOVEMENT --------
    velocities[:] += np.random.randn(NUM_DRONES, 2) * 0.05
    center = np.array([AREA_SIZE / 2, AREA_SIZE / 2])
    velocities[:] += CENTER_FORCE * (center - positions)

    for i in range(NUM_DRONES):
        if alive[i] and adj[i].sum() == 0:
            nearest = np.mean(positions[alive == 1], axis=0)
            velocities[i] += PANIC_FORCE * (nearest - positions[i])

    velocities[:] = np.array([limit_speed(v) for v in velocities])
    positions[:] += velocities * DT
    positions[:] = np.clip(positions, 0, AREA_SIZE)

    # -------- CLEAR EDGES --------
    for ln in edge_artists:
        ln.remove()
    edge_artists = []

    # -------- DRAW EDGES --------
    for i in range(NUM_DRONES):
        for j in range(NUM_DRONES):
            if adj[i, j]:
                ln, = ax.plot(
                    [positions[i,0], positions[j,0]],
                    [positions[i,1], positions[j,1]],
                    linewidth=0.6,
                    alpha=0.4
                )
                edge_artists.append(ln)

    # -------- COLORS --------
    colors = []
    for i in range(NUM_DRONES):
        if alive[i] == 0:
            colors.append("black")
        elif battery[i] < 40:
            colors.append("green")
        elif leaders[i]:
            colors.append("red")
        elif adj[i].sum() == 0:
            colors.append("orange")
        else:
            colors.append("blue")

    scat.set_offsets(positions)
    scat.set_color(colors)

    return scat,

# ==========================
# VIS
# ==========================
fig, ax = plt.subplots()
ax.set_xlim(0, AREA_SIZE)
ax.set_ylim(0, AREA_SIZE)
ax.set_title("LEVEL-4: Visible Collapse & Self-Healing")

scat = ax.scatter(positions[:,0], positions[:,1], s=70)
ani = FuncAnimation(fig, update, frames=MAX_FRAMES, interval=40)
plt.show()
