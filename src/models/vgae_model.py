import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# =========================
# Adjacency Normalization
# =========================
def normalize_adj(adj):
    """
    Compute D^-1/2 (A + I) D^-1/2
    """
    device = adj.device
    adj = adj + torch.eye(adj.size(0)).to(device)

    deg = torch.sum(adj, dim=1)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0

    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    return D_inv_sqrt @ adj @ D_inv_sqrt


# =========================
# Graph Convolution Layer
# =========================
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        support = self.linear(x)
        return adj @ support


# =========================
# Variational Graph AutoEncoder
# =========================
class VariationalGraphAutoEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim, dropout=0.3):
        super().__init__()

        self.gc1 = GraphConvolution(in_dim, hidden_dim)
        self.gc_mu = GraphConvolution(hidden_dim, latent_dim)
        self.gc_logvar = GraphConvolution(hidden_dim, latent_dim)

        self.dropout = dropout

    def encode(self, x, adj):
        h = F.relu(self.gc1(x, adj))
        h = F.dropout(h, self.dropout, training=self.training)

        mu = self.gc_mu(h, adj)
        logvar = self.gc_logvar(h, adj)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # Inner product decoder (NO sigmoid)
        return z @ z.t()

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        adj_logits = self.decode(z)
        return adj_logits, mu, logvar


# =========================
# Loss Functions
# =========================
def loss_function(adj_logits, adj_true, mu, logvar):
    # Reconstruction loss
    recon_loss = F.binary_cross_entropy_with_logits(
        adj_logits, adj_true
    )

    # KL divergence
    kl_loss = -0.5 * torch.mean(
        torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    )

    return recon_loss + kl_loss


# =========================
# Example Training Script
# =========================
if __name__ == "__main__":

    # Example synthetic graph
    torch.manual_seed(42)

    num_nodes = 100
    input_dim = 16
    hidden_dim = 32
    latent_dim = 8

    # Random node features
    X = torch.randn(num_nodes, input_dim)

    # Random symmetric adjacency matrix
    A = torch.randint(0, 2, (num_nodes, num_nodes)).float()
    A = (A + A.t()) / 2
    A[A > 0] = 1
    A.fill_diagonal_(0)

    # Normalize adjacency for encoder
    A_norm = normalize_adj(A)

    # Initialize model
    model = VariationalGraphAutoEncoder(
        in_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        dropout=0.3
    )

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    epochs = 200

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        adj_logits, mu, logvar = model(X, A_norm)

        loss = loss_function(adj_logits, A, mu, logvar)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    print("\nTraining Finished!")