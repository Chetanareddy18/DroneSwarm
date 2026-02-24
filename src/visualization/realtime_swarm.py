import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

# =========================
# LOAD DATA
# =========================
positions = np.load("../simulation/data/positions.npy")
alive = np.load("../simulation/data/alive.npy")

TIMESTEPS, NUM_DRONES, _ = positions.shape
COMM_RANGE = 120

# =========================
# FIGURE
# =========================
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim(0, 500)
ax.set_ylim(0, 500)
ax.set_title("Live Drone Swarm Radar View")

# =========================
# UPDATE FUNCTION
# =========================
def update(frame):
    ax.clear()
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 500)
    ax.set_title(f"Drone Swarm Radar – Time {frame}")

    pos = positions[frame]
    live = alive[frame]

    for i in range(NUM_DRONES):
        x, y = pos[i]

        # Dead drone
        if not live[i]:
            ax.scatter(x, y, c="red", s=60)
            continue

        # Drone body
        ax.scatter(x, y, c="lime", s=60)

        # Communication radius
        circle = Circle((x, y), COMM_RANGE, fill=False, alpha=0.15)
        ax.add_patch(circle)

        ax.text(x+3, y+3, f"D{i}", fontsize=7)

    # Draw links ONLY visually (no graph look)
    for i in range(NUM_DRONES):
        for j in range(i+1, NUM_DRONES):
            if live[i] and live[j]:
                dist = np.linalg.norm(pos[i] - pos[j])
                if dist <= COMM_RANGE:
                    ax.plot(
                        [pos[i, 0], pos[j, 0]],
                        [pos[i, 1], pos[j, 1]],
                        c="cyan",
                        alpha=0.4
                    )

# =========================
# ANIMATION
# =========================
ani = FuncAnimation(fig, update, frames=TIMESTEPS, interval=120)
plt.show()
