import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score

from gae_model import GraphAutoEncoder

# =========================
# PATHS
# =========================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

GRAPH_DIR = os.path.join(PROJECT_ROOT, "data", "graphs")
MODEL_DIR = os.path.join(PROJECT_ROOT, "data", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("💻 Using device:", device)

# =========================
# HYPERPARAMETERS
# =========================
INPUT_DIM = 3
HIDDEN_DIM = 16
LATENT_DIM = 8
LR = 0.01
EPOCHS = 50

# =========================
# NORMALIZATION
# =========================
def normalize_adj(adj):
    rowsum = adj.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = torch.diag(d_inv_sqrt)
    return D_inv_sqrt @ adj @ D_inv_sqrt

# =========================
# METRICS
# =========================
def link_metrics(adj_true, adj_pred):
    adj_true = adj_true.flatten().cpu().numpy()
    adj_pred = adj_pred.flatten().detach().cpu().numpy()

    adj_bin = (adj_pred > 0.5).astype(int)
    acc = (adj_true == adj_bin).mean()

    auc = roc_auc_score(adj_true, adj_pred)
    return acc, auc

# =========================
# LOAD DATA
# =========================
graph_files = sorted([f for f in os.listdir(GRAPH_DIR) if f.endswith(".npz")])
print(f"📦 Found {len(graph_files)} graph snapshots")

# =========================
# MODEL
# =========================
model = GraphAutoEncoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM).to(device)
optimizer = Adam(model.parameters(), lr=LR)

# =========================
# TRAINING
# =========================
for epoch in range(EPOCHS):
    total_loss = 0
    total_acc = 0
    total_auc = 0

    for file in graph_files:
        data = np.load(os.path.join(GRAPH_DIR, file))

        x = torch.tensor(data["node_features"], dtype=torch.float32).to(device)
        adj = torch.tensor(data["adjacency"], dtype=torch.float32).to(device)

        # Binary adjacency
        adj = (adj > 0).float()
        adj = adj + torch.eye(adj.size(0)).to(device)

        adj_norm = normalize_adj(adj)

        optimizer.zero_grad()
        adj_recon, _ = model(x, adj_norm)

        loss = F.binary_cross_entropy(adj_recon, adj)
        loss.backward()
        optimizer.step()

        acc, auc = link_metrics(adj, adj_recon)

        total_loss += loss.item()
        total_acc += acc
        total_auc += auc

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {total_loss/len(graph_files):.4f} | "
        f"Acc: {total_acc/len(graph_files):.4f} | "
        f"AUC: {total_auc/len(graph_files):.4f}"
    )

# =========================
# SAVE
# =========================
torch.save(model.state_dict(), os.path.join(MODEL_DIR, "gae_model.pth"))
print("✅ Training completed and model saved")
