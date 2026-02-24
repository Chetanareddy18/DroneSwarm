import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj_norm):
        x = torch.matmul(adj_norm, x)
        return self.linear(x)


class GraphAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout=0.3):
        super().__init__()
        self.gc1 = GraphConvolution(input_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)

    def encode(self, x, adj_norm):
        h = F.relu(self.gc1(x, adj_norm))
        h = self.dropout(h)
        return self.gc2(h, adj_norm)

    def decode(self, z):
        return torch.sigmoid(torch.matmul(z, z.t()))

    def forward(self, x, adj_norm):
        z = self.encode(x, adj_norm)
        adj_recon = self.decode(z)
        return adj_recon, z
