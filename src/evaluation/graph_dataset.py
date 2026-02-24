import os
import numpy as np
import torch
from torch_geometric.data import Data

GRAPH_DIR = "../../simulation/data/graphs"

def load_graphs():
    data_list = []

    files = sorted(os.listdir(GRAPH_DIR))
    for f in files:
        if f.endswith(".npz"):
            path = os.path.join(GRAPH_DIR, f)
            data = np.load(path)

            x = torch.tensor(data["node_features"], dtype=torch.float)
            adj = torch.tensor(data["adjacency"], dtype=torch.float)

            edge_index = adj.nonzero(as_tuple=False).t().contiguous()

            graph = Data(x=x, edge_index=edge_index)
            data_list.append(graph)

    return data_list
