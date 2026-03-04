import numpy as np
import torch


def build_graph(num_nodes=35, area_size=100, neighbor_radius=25):

    np.random.seed(42)

    positions = np.random.rand(num_nodes, 2) * area_size
    velocities = np.random.randn(num_nodes, 2)

    # Node features
    features = np.hstack([positions, velocities])
    x = torch.tensor(features, dtype=torch.float)

    # Adjacency matrix
    adj = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < neighbor_radius:
                    adj[i, j] = 1

    adj = torch.tensor(adj, dtype=torch.float)

    # Normalize adjacency
    I = torch.eye(num_nodes)
    adj = adj + I
    D = torch.diag(torch.pow(adj.sum(1), -0.5))
    adj_norm = torch.mm(torch.mm(D, adj), D)

    return x, adj, adj_norm