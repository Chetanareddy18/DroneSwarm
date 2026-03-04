import torch
import torch.optim as optim
from gae_model import GraphAutoEncoder
from graph_utils import build_graph

# Hyperparameters
INPUT_DIM = 4
HIDDEN_DIM = 32
LATENT_DIM = 16
LR = 0.01
EPOCHS = 200

# Build Graph
x, adj, adj_norm = build_graph()

# Remove diagonal from training
adj = adj.clone()
adj.fill_diagonal_(0)

# Model
model = GraphAutoEncoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)
optimizer = optim.Adam(model.parameters(), lr=LR)

# Handle class imbalance
pos_weight = (adj == 0).sum() / (adj == 1).sum()
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

print("Training Started...\n")

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()

    # Forward
    adj_logits = model(x, adj_norm)

    # Loss
    loss = criterion(adj_logits, adj)

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Accuracy calculation
    with torch.no_grad():
        preds = torch.sigmoid(adj_logits)
        predicted = (preds > 0.5).float()

        correct = (predicted == adj).sum().item()
        total = adj.numel()
        accuracy = correct / total

    if epoch % 20 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f} | Accuracy: {accuracy:.4f}")

# Save model
torch.save(model.state_dict(), "gae_trained.pth")

print("\nTraining Finished & Model Saved.")