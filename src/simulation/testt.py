import numpy as np

g = np.load("data/graphs/graph_0040.npz")
adj = g["adjacency"]

print("Nodes:", adj.shape[0])
print("Possible links:", adj.size)
print("Actual links:", adj.sum())
print("Density:", adj.sum() / adj.size)
