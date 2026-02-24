import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ==========================
# FIXED RANDOMNESS (VERY IMPORTANT)
# ==========================
np.random.seed(42)

# ==========================
# PARAMETERS
# ==========================
NUM_DRONES = 35
AREA_SIZE = 100
MAX_SPEED = 2.0

NEIGHBOR_RADIUS = 30
SEPARATION_DIST = 6

W_SEP = 1.4
W_ALIGN = 1.0
W_COH = 1.6         
W_INERTIA = 0.7     

DT = 1.0

# ==========================
# INITIALIZE
# ==========================
positions = np.random.rand(NUM_DRONES, 2) * AREA_SIZE
velocities = np.random.randn(NUM_DRONES, 2) * 0.4


def limit_speed(v):
    speed = np.linalg.norm(v)
    if speed > MAX_SPEED:
        return v / (speed + 1e-6) * MAX_SPEED
    return v


# ==========================
# UPDATE FUNCTION
# ==========================
def update(frame):
    global positions, velocities

    new_velocities = np.zeros_like(velocities)

    for i in range(NUM_DRONES):
        diff = positions - positions[i]
        dist = np.linalg.norm(diff, axis=1)

        neighbors = (dist > 0) & (dist < NEIGHBOR_RADIUS)
        close = (dist > 0) & (dist < SEPARATION_DIST)

        # ----------------------
        # Separation
        # ----------------------
        sep = np.zeros(2)
        if np.any(close):
            sep = -np.sum(diff[close] / (dist[close][:, None] + 1e-5), axis=0)

        # ----------------------
        # Alignment
        # ----------------------
        align = np.zeros(2)
        if np.any(neighbors):
            mean_vel = np.mean(velocities[neighbors], axis=0)
            norm = np.linalg.norm(mean_vel)
            if norm > 0:
                align = mean_vel / norm

        # ----------------------
        # Cohesion
        # ----------------------
        coh = np.zeros(2)
        if np.any(neighbors):
            center = np.mean(positions[neighbors], axis=0)
            coh = center - positions[i]

        # ----------------------
        # Steering (SOFT)
        # ----------------------
        steering = (
            W_SEP * sep +
            W_ALIGN * align +
            W_COH * coh
        )

        # ----------------------
        # DAMPED VELOCITY UPDATE (KEY FIX)
        # ----------------------
        vel = W_INERTIA * velocities[i] + 0.25 * steering
        new_velocities[i] = limit_speed(vel)

    velocities[:] = new_velocities
    positions[:] += velocities * DT

    # ----------------------
    # SOFT WALL BOUNCE
    # ----------------------
    for i in range(NUM_DRONES):
        for d in range(2):
            if positions[i, d] < 0:
                positions[i, d] = 0
                velocities[i, d] *= -0.6
            elif positions[i, d] > AREA_SIZE:
                positions[i, d] = AREA_SIZE
                velocities[i, d] *= -0.6

    scat.set_offsets(positions)
    return scat,


# ==========================
# VISUALIZATION
# ==========================
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, AREA_SIZE)
ax.set_ylim(0, AREA_SIZE)
ax.set_title("STEP 1: Stable Drone Swarm Formation")

scat = ax.scatter(
    positions[:, 0],
    positions[:, 1],
    s=80,
    c="dodgerblue"
)

ani = FuncAnimation(fig, update, interval=60)
plt.show()
