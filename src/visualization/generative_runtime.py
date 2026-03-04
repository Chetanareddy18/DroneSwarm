import os
import torch
import numpy as np

# Correct import (no src prefix)
from models.graph_diffusion import GraphDiffusion


class GenerativeRuntime:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ✅ Correct path (based on your folder image)
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        model_path = os.path.join(BASE_DIR, "models", "saved_models", "diffusion_trained.pth")

        # Model init
        self.model = GraphDiffusion().to(self.device)

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print("✅ Generative model loaded")
        else:
            print("⚠️ Model not found, using random generator")

    def generate_topology(self, num_nodes=10):
        """
        Returns adjacency matrix
        """
        try:
            with torch.no_grad():
                noise = torch.randn((1, num_nodes, num_nodes)).to(self.device)
                output = self.model(noise).cpu().numpy()[0]

                # Make symmetric
                adj = (output + output.T) / 2

                # Threshold
                adj = (adj > 0.5).astype(int)

                np.fill_diagonal(adj, 0)
                return adj

        except Exception as e:
            print("Fallback:", e)
            return np.random.randint(0, 2, (num_nodes, num_nodes))