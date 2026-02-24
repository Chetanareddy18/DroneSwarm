import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score

from vgae_model import VariationalGraphAutoEncoder

# -------------------------
# PATHS
# -------------------------
ROOT = os.path.abspath(os.path.join(__file__, "..", "..", ".."))
GRAPH_DIR = os.path.join(ROOT, "data", "graphs")
MODEL_DIR = os.path.join(ROOT, "data", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------
# DEVICE
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("💻 Device:", device)

# -------------------------
# HYPERPARAMETERS
# -------------------------
IN_DIM = 3
HIDDEN = 32
LATENT = 16
LR = 0.005
EPOCHS = 50

# -------------------------
# UTILS
# -------------------------
def normalize_adj(adj):
    adj = adj + torch.eye(adj.size(0)).to(adj.device)
    deg = adj.sum(1)
    D = torch.diag(torch.pow(deg, -0.5))
    return D @ adj @ D

# -------------------------
# DATA
# -------------------------
files = sorted([f for f in os.listdir(GRAPH_DIR) if f.endswith(".npz")])
print(f"📦 Training on {len(files)} graphs")

# -------------------------
# MODEL
# -------------------------
model = VariationalGraphAutoEncoder(IN_DIM, HIDDEN, LATENT).to(device)
optimizer = Adam(model.parameters(), lr=LR)

bce_loss = torch.nn.BCEWithLogitsLoss()

# -------------------------
# TRAIN LOOP
# -------------------------
for epoch in range(EPOCHS):
    total_loss = 0
    aucs = []

    for f in files:
        data = np.load(os.path.join(GRAPH_DIR, f))
        x = torch.tensor(data["node_features"], dtype=torch.float32).to(device)
        adj = torch.tensor(data["adjacency"], dtype=torch.float32).to(device)

        adj = (adj > 0).float()
        adj_norm = normalize_adj(adj)

        optimizer.zero_grad()
        adj_logits, mu, logvar = model(x, adj_norm)

        # Reconstruction loss (stable)
        recon = bce_loss(adj_logits, adj)

        # KL divergence
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        loss = recon + kl
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # AUC (optional)
        try:
            probs = torch.sigmoid(adj_logits)
            auc = roc_auc_score(
                adj.cpu().flatten().numpy(),
                probs.detach().cpu().flatten().numpy()
            )
            aucs.append(auc)
        except:
            pass

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {total_loss/len(files):.4f} "
        f"AUC: {np.mean(aucs) if aucs else float('nan'):.4f}"
    )

torch.save(model.state_dict(), os.path.join(MODEL_DIR, "vgae_model.pth"))
print("✅ VGAE training complete")
                