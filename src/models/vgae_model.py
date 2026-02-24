import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        return self.linear(adj @ x)


class VariationalGraphAutoEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim):
        super().__init__()

        self.gc1 = GraphConvolution(in_dim, hidden_dim)
        self.gc_mu = GraphConvolution(hidden_dim, latent_dim)
        self.gc_logvar = GraphConvolution(hidden_dim, latent_dim)

    def encode(self, x, adj):
        h = F.relu(self.gc1(x, adj))
        mu = self.gc_mu(h, adj)
        logvar = self.gc_logvar(h, adj)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # ❗ NO sigmoid here
        return z @ z.t()

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        adj_logits = self.decode(z)
        return adj_logits, mu, logvar
