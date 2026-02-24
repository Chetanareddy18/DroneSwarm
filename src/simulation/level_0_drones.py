import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --------------------------
# PARAMETERS
# --------------------------
NUM_DRONES = 20
AREA_SIZE = 100
SPEED = 1.5

# --------------------------
# INITIALIZE DRONES
# --------------------------
positions = np.random.rand(NUM_DRONES, 2) * AREA_SIZE
angles = np.random.rand(NUM_DRONES) * 2 * np.pi
velocities = np.column_stack((np.cos(angles), np.sin(angles))) * SPEED

# --------------------------
# PLOT SETUP
# --------------------------
fig, ax = plt.subplots()
ax.set_xlim(0, AREA_SIZE)
ax.set_ylim(0, AREA_SIZE)
ax.set_title("STEP 0: Basic Drone Motion")
scat = ax.scatter(positions[:, 0], positions[:, 1], s=50)

# --------------------------
# UPDATE FUNCTION
# --------------------------
def update(frame):
    global positions, velocities

    positions += velocities

    # Wall collision
    for i in range(NUM_DRONES):
        if positions[i, 0] <= 0 or positions[i, 0] >= AREA_SIZE:
            velocities[i, 0] *= -1
        if positions[i, 1] <= 0 or positions[i, 1] >= AREA_SIZE:
            velocities[i, 1] *= -1

    scat.set_offsets(positions)
    return scat,

# --------------------------
# ANIMATION
# --------------------------
ani = FuncAnimation(fig, update, frames=300, interval=50)
plt.show()
