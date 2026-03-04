import torch
from gae_model import GraphAutoEncoder
from graph_utils import build_graph

# Load Graph
x, adj, adj_norm = build_graph()

adj = adj.clone()
adj.fill_diagonal_(0)

# Model parameters
INPUT_DIM = 4
HIDDEN_DIM = 32
LATENT_DIM = 16

# Load model
model = GraphAutoEncoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)
model.load_state_dict(torch.load("gae_trained.pth"))
model.eval()

with torch.no_grad():
    adj_logits = model(x, adj_norm)
    adj_recon = torch.sigmoid(adj_logits)

    predicted = (adj_recon > 0.5).float()

    correct = (predicted == adj).sum().item()
    total = adj.numel()
    accuracy = correct / total

print("Test Accuracy:", round(accuracy, 4))

print("\nOriginal Adjacency Matrix:")
print(adj)

print("\nReconstructed Adjacency Matrix (Probabilities):")
print(adj_recon)