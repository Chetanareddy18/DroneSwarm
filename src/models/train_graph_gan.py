import torch
import torch.optim as optim
import numpy as np
import os
from .graph_gan import GraphGenerator, GraphDiscriminator

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "simulation", "data", "adjacency_logs.npy")

real_data = np.load(DATA_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

real_data = np.load(DATA_PATH)
real_data = np.load("data/adjacency_logs.npy")
real_data = torch.tensor(real_data, dtype=torch.float32).to(device)

batch_size = 16
noise_dim = 64
epochs = 200

G = GraphGenerator(noise_dim).to(device)
D = GraphDiscriminator().to(device)

criterion = torch.nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=0.0002)
optimizer_D = optim.Adam(D.parameters(), lr=0.0002)

for epoch in range(epochs):

    z = torch.randn(batch_size, noise_dim).to(device)
    fake_graphs = G(z)

    real_batch = real_data[torch.randint(0, real_data.size(0), (batch_size,))]

    real_labels = torch.ones(batch_size, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)

    D_real = D(real_batch)
    D_fake = D(fake_graphs.detach())

    loss_D = criterion(D_real, real_labels) + criterion(D_fake, fake_labels)

    optimizer_D.zero_grad()
    loss_D.backward()
    optimizer_D.step()

    D_fake = D(fake_graphs)
    loss_G = criterion(D_fake, real_labels)

    optimizer_G.zero_grad()
    loss_G.backward()
    optimizer_G.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch} | D Loss: {loss_D.item():.4f} | G Loss: {loss_G.item():.4f}")

# Save model properly
os.makedirs("saved_models", exist_ok=True)
torch.save(G.state_dict(), "saved_models/graph_generator.pth")
print("Model saved successfully.")