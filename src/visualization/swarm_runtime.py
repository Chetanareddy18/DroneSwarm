import os
import pandas as pd
import numpy as np
import re

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

DATASET_PATH = os.path.join(BASE_DIR, "simulation", "dataset_lvl5")
GRAPH_PATH = os.path.join(DATASET_PATH, "graphs")
CSV_PATH = os.path.join(DATASET_PATH, "global_states.csv")

AREA_MIN = 0
AREA_MAX = 100
RADIUS = 28

MAX_FAILURES = 5
MAX_FINAL_FAILURES = 2


class SwarmRuntime:

    def __init__(self):

        self.df = pd.read_csv(CSV_PATH)
        self.graph_steps = self._scan_graph_steps()

        # ✅ FIXED: required by dashboard slider
        self.total_steps = len(self.df)
        self.last_valid_step = self.total_steps - 1

        # Persistent simulation state
        self.adj = None
        self.failed_nodes = []
        self.heal_queue = []
        self.last_step = -1

    # ------------------------------------------------

    def _scan_graph_steps(self):

        steps = []
        pattern = re.compile(r"graph_0_(\d+)\.npz")

        for f in os.listdir(GRAPH_PATH):
            m = pattern.match(f)
            if m:
                steps.append(int(m.group(1)))

        steps.sort()
        return steps

    # ------------------------------------------------

    def _graph_for_step(self, step):

        valid = [s for s in self.graph_steps if s <= step]
        return valid[-1] if valid else self.graph_steps[0]

    # ------------------------------------------------

    def _enforce_arena(self, positions):

        positions[:, 0] = np.clip(positions[:, 0], AREA_MIN, AREA_MAX)
        positions[:, 1] = np.clip(positions[:, 1], AREA_MIN, AREA_MAX)
        return positions

    # ------------------------------------------------

    def _build_initial_graph(self, positions):

        N = len(positions)
        adj = np.zeros((N, N))

        for i in range(N):
            for j in range(i + 1, N):
                if np.linalg.norm(positions[i] - positions[j]) <= RADIUS:
                    adj[i, j] = adj[j, i] = 1

        return adj

    # ------------------------------------------------

    def _disconnect_one(self):

        if len(self.failed_nodes) >= MAX_FAILURES:
            return None

        N = len(self.adj)
        degrees = self.adj.sum(axis=1)

        candidates = [
            i for i in range(N)
            if degrees[i] > 1 and i not in self.failed_nodes
        ]

        if not candidates:
            return None

        node = candidates[len(self.failed_nodes) % len(candidates)]

        # remove all links
        self.adj[node] = 0
        self.adj[:, node] = 0

        self.failed_nodes.append(node)
        self.heal_queue.append(node)

        return node

    # ------------------------------------------------

    def _heal_one(self, positions):

        if not self.heal_queue:
            return None

        node = self.heal_queue.pop(0)

        distances = np.linalg.norm(
            positions - positions[node], axis=1
        )

        candidates = [i for i in range(len(positions)) if i != node]

        nearest = min(candidates, key=lambda x: distances[x])

        # reconnect
        self.adj[node, nearest] = 1
        self.adj[nearest, node] = 1

        if node in self.failed_nodes:
            self.failed_nodes.remove(node)

        return node

    # ------------------------------------------------

    def get_state(self, step=0):

        step = min(step, self.last_valid_step)
        row = self.df.iloc[step]

        graph_step = self._graph_for_step(step)
        graph_file = os.path.join(
            GRAPH_PATH,
            f"graph_0_{graph_step}.npz"
        )

        if not os.path.exists(graph_file):
            return None

        data = np.load(graph_file)
        node_features = data["node_features"].copy()

        positions = node_features[:, :2]
        positions = self._enforce_arena(positions)

        # Reset simulation if slider jumps
        if step != self.last_step + 1:
            self.adj = self._build_initial_graph(positions)
            self.failed_nodes = []
            self.heal_queue = []

        # ----------------------------------------
        # CONTROLLED FAILURE / HEALING CYCLE
        # ----------------------------------------

        cycle = step % 20
        healed_nodes = []

        # Gradual failure introduction
        if cycle in [2, 6, 10, 14, 18]:
            self._disconnect_one()

        # Gradual healing phase
        if cycle in [5, 9, 13, 17]:
            healed = self._heal_one(positions)
            if healed is not None:
                healed_nodes.append(healed)

        # Final guarantee: ≤ 2 failures
        if step == self.last_valid_step:
            while len(self.failed_nodes) > MAX_FINAL_FAILURES:
                healed = self._heal_one(positions)
                if healed is None:
                    break
                healed_nodes.append(healed)

        self.last_step = step

        return {
            "positions": positions,
            "adjacency": self.adj,
            "failed_nodes": list(self.failed_nodes),
            "healed_nodes": healed_nodes,
            "metrics": {
                "alive_ratio": float(row["alive_ratio"]),
                "density": float(row["density"]),
                "avg_degree": float(row["avg_degree"]),
                "isolated": int(row["isolated_nodes"]),
                "avg_battery": float(row["avg_battery"]),
                "leader_ratio": float(row["leader_ratio"])
            }
        }