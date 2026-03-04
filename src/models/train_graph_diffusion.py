import torch
import torch.optim as optim
import torch.nn as nn
from graph_utils import build_graph
from graph_diffusion import GraphDiffusion, add_noise

# ===============================
# Hyperparameters
# ===============================
NUM_EPOCHS = 200
LR = 0.001
NOISE_LEVEL = 0.2
HIDDEN_DIM = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Building graph...")

# ==========================================
# Load graph (can be ANY number of nodes)
# ==========================================
x, adj, adj_norm = build_graph()

adj = adj.clone().to(device)
adj.fill_diagonal_(0)

# Add batch dimension
adj = adj.unsqueeze(0)   # [1, N, N]

# ==========================================
# Scalable Diffusion Model
# ==========================================
model = GraphDiffusion(in_features=1, hidden_dim=HIDDEN_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

criterion = nn.BCELoss()

print("\nStarting Scalable Diffusion Training...\n")

# ==========================================
# TRAINING LOOP
# ==========================================
for epoch in range(NUM_EPOCHS):

    model.train()
    optimizer.zero_grad()

    # Add symmetric noise
    adj_noisy = add_noise(adj, noise_level=NOISE_LEVEL)

    # Forward pass
    adj_denoised = model(adj_noisy)

    # Loss
    loss = criterion(adj_denoised, adj)

    # Backprop
    loss.backward()
    optimizer.step()

    # Accuracy calculation
    with torch.no_grad():
        predicted = (adj_denoised > 0.5).float()
        correct = (predicted == adj).sum().item()
        total = adj.numel()
        accuracy = correct / total

    if epoch % 20 == 0:
        print(
            f"Epoch {epoch} | "
            f"Loss: {loss.item():.4f} | "
            f"Accuracy: {accuracy:.4f}"
        )

# ==========================================
# SAVE MODEL
# ==========================================
torch.save(model.state_dict(), "saved_models\diffusion_trained.pth")

print("\nScalable Diffusion Training Complete & Model Saved.")