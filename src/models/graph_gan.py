import torch
import torch.nn as nn

class GraphGenerator(nn.Module):
    def __init__(self, noise_dim=64, num_nodes=35):
        super(GraphGenerator, self).__init__()

        self.num_nodes = num_nodes
        self.fc = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_nodes * num_nodes),
            nn.Sigmoid()
        )

    def forward(self, z):
        adj = self.fc(z)
        adj = adj.view(-1, self.num_nodes, self.num_nodes)

        # Make symmetric (undirected graph)
        adj = (adj + adj.transpose(1,2)) / 2

        return adj
class GraphDiscriminator(nn.Module):
    def __init__(self, num_nodes=35):
        super(GraphDiscriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(num_nodes * num_nodes, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, adj):
        adj_flat = adj.view(adj.size(0), -1)
        return self.model(adj_flat)