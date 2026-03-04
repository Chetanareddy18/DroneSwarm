import torch
from graph_utils import build_graph
from graph_diffusion import GraphDiffusion, add_noise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# Load Graph (any node count)
# ===============================
x, adj, adj_norm = build_graph()

adj = adj.clone().to(device)
adj.fill_diagonal_(0)

# Add batch dimension
adj = adj.unsqueeze(0)  # [1, N, N]

# ===============================
# Load Scalable Diffusion Model
# ===============================
model = GraphDiffusion(in_features=1, hidden_dim=128).to(device)
model.load_state_dict(torch.load("saved_models\diffusion_trained.pth", map_location=device))
model.eval()

# ===============================
# Add Noise
# ===============================
NOISE_LEVEL = 0.2
adj_noisy = add_noise(adj, noise_level=NOISE_LEVEL)

# ===============================
# Denoise
# ===============================
with torch.no_grad():
    adj_denoised = model(adj_noisy)

    predicted = (adj_denoised > 0.5).float()

    correct = (predicted == adj).sum().item()
    total = adj.numel()
    accuracy = correct / total

# ===============================
# Print Results
# ===============================
print("Test Accuracy:", round(accuracy, 4))

print("\nOriginal Adjacency:")
print(adj.squeeze(0))

print("\nNoisy Adjacency:")
print(adj_noisy.squeeze(0))

print("\nDenoised Adjacency (Probabilities):")
print(adj_denoised.squeeze(0))