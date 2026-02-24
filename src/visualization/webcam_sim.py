import cv2
import numpy as np

N = 20
positions = np.random.rand(N, 2)
adj = np.zeros((N, N))

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    h, w, _ = frame.shape

    # Update random movement
    positions += np.random.uniform(-0.01, 0.01, (N, 2))
    positions = np.clip(positions, 0, 1)

    # Create adjacency
    for i in range(N):
        for j in range(N):
            if np.linalg.norm(positions[i] - positions[j]) < 0.15:
                adj[i][j] = 1
            else:
                adj[i][j] = 0

    # Draw edges
    for i in range(N):
        for j in range(i+1, N):
            if adj[i][j] == 1:
                x1 = int(positions[i][0] * w)
                y1 = int(positions[i][1] * h)
                x2 = int(positions[j][0] * w)
                y2 = int(positions[j][1] * h)
                cv2.line(frame, (x1,y1), (x2,y2), (255,0,0), 1)

    # Draw nodes
    for i in range(N):
        x = int(positions[i][0] * w)
        y = int(positions[i][1] * h)
        cv2.circle(frame, (x, y), 6, (0,255,0), -1)

    cv2.imshow("Drone Swarm Simulation (Live)", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
