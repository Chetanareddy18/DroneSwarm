# 🚁 Intelligent Self-Healing Drone Swarm Communication System

> A multi-model graph-learning based drone swarm framework that **detects failures, adapts dynamically, and heals communication links intelligently** in real time — combined with a **Unity 3D visualization** of swarm formations.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![Unity](https://img.shields.io/badge/Unity-2022.3%2B-black.svg)](https://unity.com/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)]()

---

## 📺 Demo & Report

- 🎬 **Demo video:** [`docs/DroneSwarm_demo.mp4`](docs/DroneSwarm_demo.mp4)
- 📄 **Project report:** [`docs/Project_Report.docx`](docs/Project_Report.docx)

---

## 📌 Project Overview

This project builds an **adaptive drone swarm communication framework** capable of:

- Detecting drone failures
- Predicting broken communication links
- Healing the network gradually
- Adjusting communication radius dynamically
- Maintaining high connectivity under disruptions

It combines **classical swarm simulation**, **graph deep learning**, **generative models**, **reinforcement learning**, an **orchestrator layer**, and a **Unity 3D visual front-end**.

---

## ❗ Core Problem

In real-world drone swarms:

- Drones move continuously
- Links break as distance grows
- Drones fail (battery / fault injection)
- The network fragments and isolated nodes appear
- Coordination collapses

**Traditional rule-based systems** use fixed radius, no learning, no progressive recovery. This project asks:

> *How can a drone swarm detect failures, adapt intelligently, and self-heal progressively instead of collapsing?*

---

## 🏗 System Architecture

The swarm is modeled as a **dynamic graph**:

| Element | Meaning |
|---------|---------|
| **Nodes** | Drones |
| **Edges** | Communication links |
| **Adjacency matrix `A`** | Connectivity at time `t` |

Pipeline at every timestep:

```
Drones move → Distances update → Graph rebuilt
        → GNN / Diffusion repair → RL adjusts radius
        → Orchestrator finalizes → Swarm stabilizes
```

---

## 📁 Repository Structure

```
DRONE_SWARM/
├── README.md
├── .gitignore
├── test.py                          # quick numpy sanity check
│
├── docs/                            # 📺 demo video + 📄 report
│   ├── DroneSwarm_demo.mp4
│   └── Project_Report.docx
│
├── src/
│   ├── simulation/                  # All swarm simulation levels
│   │   ├── level_0_drones.py             # basic drone kinematics
│   │   ├── level_1_swarm.py              # Reynolds-style flocking (sep / align / coh)
│   │   ├── level_2_communication.py      # comm graph from radius
│   │   ├── level_3_dynamic_network.py    # moving graph, link births/deaths
│   │   ├── level_4_failure_healing.py    # failures + reactive healing
│   │   ├── level_5_adaptive_intelligence.py  # adaptive radius + leaders
│   │   └── dataset_collector_level5.py   # logs adjacency + states
│   │
│   ├── models/                      # Learning models
│   │   ├── graph_utils.py
│   │   ├── gae_model.py             # Graph AutoEncoder
│   │   ├── vgae_model.py            # Variational GAE
│   │   ├── graph_gan.py             # Graph GAN (generator + discriminator)
│   │   ├── graph_diffusion.py       # Scalable diffusion repair model
│   │   ├── hierarchical_graph_rl.py # Encoder + DQN
│   │   ├── train_gae.py / train_vgae.py
│   │   ├── train_graph_gan.py
│   │   ├── train_graph_diffusion.py
│   │   ├── train_hierarchical_rl.py
│   │   ├── train_multi_agent_rl.py  # per-drone DQN agent
│   │   ├── rl_system_controller.py  # RL + Diffusion runtime controller
│   │   ├── orchestrator.py          # 🎛 Final multi-model orchestrator
│   │   └── saved_models/*.pth       # trained checkpoints
│   │
│   ├── evaluation/
│   │   └── graph_dataset.py         # PyG dataset wrapper
│   │
│   └── visualization/
│       ├── realtime_swarm.py        # matplotlib radar view
│       ├── swarm_runtime.py
│       ├── generative_runtime.py
│       ├── graph_visualizer.py
│       ├── research_dashboard.py    # Streamlit dashboard
│       ├── inspect_data.py
│       └── webcam_sim.py            # AR / webcam overlay test
│
├── AR/
│   └── test1.py                     # AR experimentation
│
└── unity_simulation/                # 🎮 Unity 3D front-end
    ├── Assets/
    │   ├── Scripts/
    │   │   ├── DroneSwarm.cs        # per-drone physics + steering
    │   │   └── SwarmManager.cs      # spawning, formations, failure
    │   ├── Drone/                   # drone art + prefabs
    │   ├── Prefabs/, Models/, Scenes/, Settings/
    │   └── ...
    ├── Packages/
    └── ProjectSettings/
```

> `data/`, `outputs/`, `swarmenv/`, model `.pth` checkpoints and Unity `Library/` are intentionally excluded from version control via `.gitignore`.

---

## 🧩 Project Stages

### 1️⃣ Basic Drone Simulation
- 2D area `100 × 100`, ~30–35 drones
- Each drone has position, battery, alive flag, role (leader / normal)
- Rule: *two drones connect if distance ≤ communication radius*

### 2️⃣ Graph Construction
At every timestep build adjacency `A(i,j) = 1 if dist ≤ r else 0`, and compute:
connectivity ratio · average degree · density · isolated nodes · leader ratio · battery average.

### 3️⃣ Failure Injection
Random shutdown · battery depletion · link breakage. When a drone fails its edges are removed → connectivity drops → isolation increases.

### 4️⃣ Static Baseline (Rule-Based)
| Metric | Value |
|---|---|
| Connectivity | ≈ 81.6% |
| Recovery time | ≈ 14 steps |
| Final failed nodes | ≈ 7 |

---

## 🤖 Learning Models

### 5️⃣ Graph Autoencoder (GAE) — `gae_model.py`
Encodes nodes → embeddings, decodes adjacency. Predicts missing links → partial recovery.

### 6️⃣ Variational GAE (VGAE) — `vgae_model.py`
Probabilistic embeddings (mean + variance). Better robustness under uncertainty.

### 7️⃣ Graph GAN — `graph_gan.py`
Generator vs. discriminator on adjacency matrices. Produces realistic post-repair topology.

### 8️⃣ Graph Diffusion — `graph_diffusion.py`
Forward: add noise to `A`. Reverse: a GCN-based denoiser learns step-by-step healing → smooth, gradual topology repair.

```python
class GraphDiffusion(nn.Module):
    # GCN denoiser using node degree as scalar feature → size-independent
    def forward(self, adj_noisy):
        x = adj_noisy.sum(-1, keepdim=True)        # degree feature
        h = F.relu(self.gc1(x, adj_noisy))
        h = F.relu(self.gc2(h, adj_noisy))
        out = self.gc3(h, adj_noisy).squeeze(-1)
        return torch.sigmoid(...)                  # reconstructed A
```

---

## 🎯 Reinforcement Learning Agents

Two complementary RL designs:

- **Hierarchical RL** (`hierarchical_graph_rl.py`): a graph encoder + DQN that observes global swarm state (alive ratio, density, isolated nodes, average degree) and outputs system-level actions: adjust comm-radius, trigger healing, reassign leader, optimize topology.
- **Multi-Agent RL** (`train_multi_agent_rl.py`): every drone runs its own lightweight DQN over `[degree, neighbor_mean, density]` and chooses *connect / disconnect / repair / idle*.

Reward: `+ connectivity − isolated_nodes`.

Result: faster recovery, smarter radius, failures bounded to ≤ 2 nodes.

---

## 🎛 Orchestrator Layer — `src/models/orchestrator.py`

The orchestrator is the **runtime brain** that combines everything:

1. Generate / receive an adjacency matrix
2. Run **Multi-Agent RL** on every drone to apply local edge actions
3. Run **Graph Diffusion** to globally smooth & repair the topology
4. Compute formation score before / after
5. Emit final 3D drone positions + states (`leader / normal / failed`)
6. Save outputs:
   - `outputs/drone_positions.json` → consumed by Unity & dashboard
   - `outputs/swarm_summary.csv` / `.json` → metrics

---

## 🎮 Unity 3D Simulation — `unity_simulation/`

A real-time Unity front-end that visualizes the swarm.

- **`SwarmManager.cs`** — spawns drones, runs formation cycles (square → circle → triangle), injects failures, and orchestrates the swarm “show”.
- **`DroneSwarm.cs`** — per-drone physics: target steering, neighbor separation (`Physics.OverlapSphere`), hover stabilization, damping, speed clamp, and a `FailDrone()` hook that disables control and lets the drone fall.

```csharp
// SwarmManager.cs (excerpt)
void StartShow() {
    FormSquare();
    Invoke("FailTwoDrones", 6f);
    InvokeRepeating("NextFormation", 10f, 8f);   // square → circle → triangle
}
```

```csharp
// DroneSwarm.cs (excerpt)
Vector3 desired = targetPosition - transform.position;
Vector3 steer   = desired - rb.linearVelocity;
force += steer * steeringForce;
// + separation + hover + damping
```

> Unity `Library/`, `Temp/`, `Logs/`, `.csproj`, `.sln`, etc. are git-ignored. Open `unity_simulation/` in Unity Hub (Unity 2022.3+ recommended) to regenerate them.

---

## 📊 Visualization (Python)

- **Streamlit dashboard** — `src/visualization/research_dashboard.py`: live swarm graph, connectivity metrics, failure monitor, density / alive-ratio trends.
- **Realtime radar view** — `src/visualization/realtime_swarm.py`: matplotlib animation of drone positions + comm circles.
- **Generative runtime** — `src/visualization/generative_runtime.py`: visualize diffusion / GAN repair frames.

---

## 📈 Results

| System Type                  | Connectivity | Recovery Time | Failed Nodes |
|------------------------------|:------------:|:-------------:|:------------:|
| Static rule-based            | 81.6 %       | 14 steps      | 7            |
| Adaptive radius              | 89.4 %       | 9 steps       | 4            |
| **Proposed intelligent system** | **96.8 %** | **4 steps** | **≤ 2**      |

Improvements: higher density, faster stabilization, gradual healing, minimal isolation.

---

## 🚀 Quick Start

### 1. Clone & set up Python env

```powershell
git clone https://github.com/Chetanareddy18/DroneSwarm.git
cd DroneSwarm

python -m venv swarmenv
.\swarmenv\Scripts\Activate.ps1

pip install torch numpy matplotlib networkx streamlit scipy
# optional for GNN datasets:
pip install torch-geometric
```

### 2. Run a baseline simulation

```powershell
python src/simulation/level_1_swarm.py
python src/simulation/level_4_failure_healing.py
python src/simulation/level_5_adaptive_intelligence.py
```

### 3. Train models (checkpoints land in `src/models/saved_models/`)

```powershell
python src/models/train_gae.py
python src/models/train_vgae.py
python src/models/train_graph_gan.py
python src/models/train_graph_diffusion.py
python src/models/train_hierarchical_rl.py
python src/models/train_multi_agent_rl.py
```

### 4. Run the orchestrator (full intelligent pipeline)

```powershell
python src/models/orchestrator.py
```

Outputs: `outputs/drone_positions.json`, `outputs/swarm_summary.csv`.

### 5. Launch the dashboard

```powershell
streamlit run src/visualization/research_dashboard.py
```

### 6. Open the Unity scene

1. Open **Unity Hub → Add → `unity_simulation/`** (Unity 2022.3 LTS or newer).
2. Open the main scene under `Assets/Drone/Scene/`.
3. Press **Play** — `SwarmManager` spawns drones, cycles formations, and triggers failures.

---

## 🔄 End-to-End Flow

```
Swarm starts normally
       ↓
Failures injected (battery / shutdown / link loss)
       ↓
Connectivity drops, isolated nodes appear
       ↓
GAE / VGAE / GAN predict / generate missing links
       ↓
Graph Diffusion smooths & repairs topology
       ↓
Multi-Agent RL chooses local edge actions
       ↓
Hierarchical RL adjusts global comm radius
       ↓
Orchestrator aggregates final adjacency
       ↓
Unity 3D + Streamlit dashboard visualize stabilized swarm
```

---

## 🛠 Tech Stack

**Python** · **PyTorch** · **PyTorch Geometric** · **NetworkX** · **NumPy** · **Matplotlib** · **Streamlit** · **Unity 3D (C#)**

---

## 🧠 Big Picture Transformation

```
Static Rule-Based Swarm
        ↓
Graph-Based Learning  (GAE / VGAE)
        ↓
Generative Topology Modeling  (GAN / Diffusion)
        ↓
Adaptive Reinforcement Healing  (Hierarchical + Multi-Agent RL)
        ↓
Orchestrated Intelligent Swarm  (+ Unity 3D Visualization)
```

---

## 🏁 Conclusion

This project demonstrates a **multi-model intelligent drone swarm communication framework** that:

- Learns structural patterns
- Predicts failures
- Heals broken links
- Adapts communication dynamically
- Maintains high connectivity under disruption
- Visualizes the entire process in **Unity 3D**

> **In one line:** an adaptive, self-healing, graph-learning powered intelligent drone swarm — visualized end-to-end.

---

## 👤 Author

**Chetana Reddy** — [@Chetanareddy18](https://github.com/Chetanareddy18)

Applicative Project 2 · 2026
