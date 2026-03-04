import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# Graph Convolution Layer (Size Independent)
# ==========================================
class GraphConv(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        """
        x: [B, N, F]
        adj: [B, N, N]
        """
        agg = torch.matmul(adj, x)   # Message passing
        return self.linear(agg)


# ==========================================
# Scalable Graph Diffusion Model
# ==========================================
class GraphDiffusion(nn.Module):
    def __init__(self, in_features=1, hidden_dim=128):
        super().__init__()

        # Now features are FIXED dimension (not num_nodes)
        self.gc1 = GraphConv(in_features, hidden_dim)
        self.gc2 = GraphConv(hidden_dim, hidden_dim)
        self.gc3 = GraphConv(hidden_dim, in_features)

        self.dropout = nn.Dropout(0.3)

    def forward(self, adj_noisy):
        """
        adj_noisy: [B, N, N]
        """

        B, N, _ = adj_noisy.shape

        # Convert adjacency to node features
        # Use node degree as fixed feature (size = 1)
        degree = adj_noisy.sum(dim=-1, keepdim=True)  # [B, N, 1]

        x = degree   # Now feature dim = 1 (constant)

        h = F.relu(self.gc1(x, adj_noisy))
        h = self.dropout(h)

        h = F.relu(self.gc2(h, adj_noisy))
        h = self.dropout(h)

        out = self.gc3(h, adj_noisy)

        # Convert node features back to adjacency via outer product
        out = out.squeeze(-1)  # [B, N]
        adj_reconstructed = torch.sigmoid(
            torch.matmul(out.unsqueeze(-1), out.unsqueeze(1))
        )  # [B, N, N]

        # Force symmetry
        adj_reconstructed = (adj_reconstructed + adj_reconstructed.transpose(1, 2)) / 2

        # Remove diagonal
        eye = torch.eye(N, device=adj_noisy.device)
        adj_reconstructed = adj_reconstructed * (1 - eye)

        return adj_reconstructed


# ==========================================
# Noise Function
# ==========================================
def add_noise(adj, noise_level=0.1):

    noise = torch.randn_like(adj) * noise_level
    noise = (noise + noise.transpose(-1, -2)) / 2

    adj_noisy = adj + noise
    adj_noisy = torch.clamp(adj_noisy, 0.0, 1.0)

    N = adj.size(-1)
    eye = torch.eye(N, device=adj.device)
    adj_noisy = adj_noisy * (1 - eye)

    return adj_noisy