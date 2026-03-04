import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score

from vgae_model import VariationalGraphAutoEncoder

# =====================================================
# PATH SETUP (MATCHES YOUR STRUCTURE)
# =====================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

GRAPH_DIR = os.path.abspath(os.path.join(
    CURRENT_DIR,
    "..",                # go from models -> src
    "simulation",
    "data",
    "graphs_lvl3"
))

MODEL_DIR = os.path.join(CURRENT_DIR, "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

print("GRAPH_DIR:", GRAPH_DIR)
print("Exists?", os.path.exists(GRAPH_DIR))

if not os.path.exists(GRAPH_DIR):
    raise FileNotFoundError(f"❌ Graph directory not found: {GRAPH_DIR}")

# =====================================================
# DEVICE
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("💻 Device:", device)

# =====================================================
# HYPERPARAMETERS
# =====================================================
IN_DIM = 5   # <-- FIXED
HIDDEN = 32
LATENT = 16
LR = 0.001
EPOCHS = 50


# =====================================================
# SAFE ADJ NORMALIZATION
# =====================================================
def normalize_adj(adj):
    adj = adj + torch.eye(adj.size(0)).to(adj.device)
    deg = adj.sum(1)
    deg[deg == 0] = 1
    deg_inv_sqrt = torch.pow(deg, -0.5)
    D = torch.diag(deg_inv_sqrt)
    return D @ adj @ D

# =====================================================
# LOAD FILES
# =====================================================
files = sorted([f for f in os.listdir(GRAPH_DIR) if f.endswith(".npz")])
print(f"📦 Training on {len(files)} graphs")

if len(files) == 0:
    raise ValueError("❌ No .npz files found in graphs_lvl3!")

# =====================================================
# MODEL
# =====================================================
model = VariationalGraphAutoEncoder(IN_DIM, HIDDEN, LATENT).to(device)
optimizer = Adam(model.parameters(), lr=LR)
bce_loss = torch.nn.BCEWithLogitsLoss()

# =====================================================
# TRAINING LOOP
# =====================================================
for epoch in range(EPOCHS):

    model.train()
    total_loss = 0
    aucs = []
    valid_graphs = 0

    for f in files:

        data = np.load(os.path.join(GRAPH_DIR, f))

        x = torch.tensor(data["node_features"], dtype=torch.float32).to(device)
        adj = torch.tensor(data["adjacency"], dtype=torch.float32).to(device)

        adj = (adj > 0).float()

        if adj.sum() == 0:
            continue

        adj_norm = normalize_adj(adj)

        optimizer.zero_grad()

        adj_logits, mu, logvar = model(x, adj_norm)

        logvar = torch.clamp(logvar, min=-10, max=10)

        recon = bce_loss(adj_logits, adj)

        kl = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - torch.exp(logvar)
        )

        loss = recon + kl

        if torch.isnan(loss):
            print(f"⚠️ NaN detected in {f}, skipping")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        valid_graphs += 1

        # AUC
        try:
            probs = torch.sigmoid(adj_logits)
            auc = roc_auc_score(
                adj.cpu().flatten().numpy(),
                probs.detach().cpu().flatten().numpy()
            )
            aucs.append(auc)
        except:
            pass

    if valid_graphs == 0:
        print("❌ No valid graphs processed!")
        break

    avg_loss = total_loss / valid_graphs
    avg_auc = np.mean(aucs) if aucs else float('nan')

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {avg_loss:.4f} "
        f"AUC: {avg_auc:.4f}"
    )

# =====================================================
# SAVE MODEL
# =====================================================
save_path = os.path.join(MODEL_DIR, "vgae_model.pth")
torch.save(model.state_dict(), save_path)

print(f"\n✅ Model saved to: {save_path}")
print("🎉 VGAE training complete.")