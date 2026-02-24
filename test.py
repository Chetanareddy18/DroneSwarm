import numpy as np
alive = np.load("data/alive.npy")
print(alive.sum(axis=1))
