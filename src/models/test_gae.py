import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score

from gae_model import GraphAutoEncoder

# =====================
# PATHS
# =====================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

GRAPH_DIR = os.path.join(PROJECT_ROOT, "data", "graphs")
MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "models", "gae_model.pth")

# =====================
# DEVICE
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("💻 Using device:", device)

# =====================
# HYPERPARAMS (MUST MATCH TRAINING)
# =====================
INPUT_DIM = 3
HIDDEN_DIM = 16
LATENT_DIM = 8

# =====================
# UTILS
# =====================
def normalize_adj(adj):
    rowsum = adj.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = torch.diag(d_inv_sqrt)
    return D_inv_sqrt @ adj @ D_inv_sqrt

# =====================
# LOAD DATA
# =====================
graph_files = sorted([f for f in os.listdir(GRAPH_DIR) if f.endswith(".npz")])

split_idx = int(0.7 * len(graph_files))
test_files = graph_files[split_idx:]

print(f"🧪 Testing on {len(test_files)} unseen graph snapshots")

# =====================
# LOAD MODEL
# =====================
model = GraphAutoEncoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# =====================
# TEST LOOP
# =====================
total_auc = 0
total_acc = 0

with torch.no_grad():
    for file in test_files:
        data = np.load(os.path.join(GRAPH_DIR, file))

        x = torch.tensor(data["node_features"], dtype=torch.float32).to(device)
        adj = torch.tensor(data["adjacency"], dtype=torch.float32).to(device)

        adj = (adj > 0).float()
        adj = adj + torch.eye(adj.size(0)).to(device)
        adj_norm = normalize_adj(adj)

        adj_recon, _ = model(x, adj_norm)

        y_true = adj.view(-1).cpu().numpy()
        y_score = adj_recon.view(-1).cpu().numpy()
        y_pred = (adj_recon > 0.5).float().view(-1).cpu().numpy()

        total_auc += roc_auc_score(y_true, y_score)
        total_acc += accuracy_score(y_true, y_pred)

# =====================
# RESULTS
# =====================
print(f"\n📊 TEST RESULTS")
print(f"Accuracy: {total_acc / len(test_files):.4f}")
print(f"AUC:      {total_auc / len(test_files):.4f}")
