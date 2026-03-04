# 🚁 Intelligent Self-Healing Drone Swarm Communication System

> A multi-model graph-learning based drone swarm system that detects failures, adapts dynamically, and heals communication links intelligently under real-time disruptions.

---

## 📌 Project Overview

This project builds an adaptive drone swarm communication framework capable of:

- Detecting drone failures  
- Predicting broken communication links  
- Healing the network gradually  
- Adjusting communication radius dynamically  
- Maintaining high connectivity under disruptions  

Unlike traditional rule-based swarm systems, this approach integrates:

- Graph Neural Networks  
- Generative Models  
- Diffusion Models  
- Reinforcement Learning  
- Multi-model Orchestration  

---

## ❗ Core Problem

In real-world drone swarms:

- Drones move continuously  
- Communication links break when distance increases  
- Drones fail due to battery depletion or injected faults  
- The network becomes fragmented  
- Isolated drones appear  
- Swarm coordination collapses  

### Traditional Systems Limitations

- Fixed communication radius  
- Static rule-based formation  
- No intelligent failure recovery  
- All failures modeled statically  
- No gradual healing mechanism  

> The challenge:  
> How can a drone swarm detect failures, adapt intelligently, and self-heal progressively instead of collapsing?

---

## 🏗 System Architecture

The swarm is modeled as a **dynamic graph**:

- **Nodes** → Drones  
- **Edges** → Communication links  
- **Adjacency Matrix** → Connectivity representation  

At every timestep:

1. Drones move  
2. Distances update  
3. Links are recalculated  
4. Graph is reconstructed  
5. Learning models are applied  
6. Healing decisions are executed  

---

# 🧩 Project Stages

---

## 1️⃣ Stage 1 – Basic Drone Simulation

- 2D simulation (100 × 100 area)
- 30 drones
- Random initial placement
- Each drone has:
  - Position (x, y)
  - Battery level
  - Alive status
  - Role (leader / normal)

Communication rule:

```
Two drones connect if distance ≤ communication_radius
```

---

## 2️⃣ Stage 2 – Graph Construction

Adjacency matrix built at every timestep:

```
A(i, j) = 1 if distance ≤ r
A(i, j) = 0 otherwise
```

Metrics computed:

- Connectivity ratio
- Average node degree
- Density
- Isolated nodes
- Leader ratio
- Battery average

---

## 3️⃣ Stage 3 – Failure Injection

Failures occur gradually:

- Random drone shutdown
- Battery depletion
- Link breakage

When a drone fails:
- All its edges are removed
- Connectivity drops
- Isolation increases

---

## 4️⃣ Stage 4 – Static Baseline (Rule-Based System)

Without intelligence:

- Connectivity ≈ 81.6%
- Recovery time ≈ 14 steps
- Final failed nodes ≈ 7

---

# 🤖 Learning Models

---

## 5️⃣ Graph Autoencoder (GAE)

- Learns structural embeddings
- Encoder → Node embeddings
- Decoder → Reconstructs adjacency matrix

Improvement:
- Predicts missing links
- Partial connectivity recovery

---

## 6️⃣ Variational Graph Autoencoder (VGAE)

- Probabilistic node embeddings
- Learns mean and variance
- Handles uncertainty

Improvement:
- More stable recovery under failures
- Better robustness

---

## 7️⃣ Graph GAN

Adversarial graph learning:

- Generator → Generates adjacency
- Discriminator → Checks realism

Improvement:
- Produces realistic swarm topology
- Improves cohesion after repair

---

## 8️⃣ Graph Diffusion Model

Gradual healing mechanism:

1. Add noise to adjacency
2. Learn to remove noise step-by-step

Improvement:
- Smooth topology repair
- No abrupt structural jumps
- Progressive stabilization

---

# 🎯 Reinforcement Learning Agent

The RL agent observes:

- Alive ratio
- Density
- Isolated nodes
- Average degree

Actions:

- Adjust communication radius
- Trigger healing
- Reassign leader
- Optimize topology

Reward function:

```
Reward = + connectivity
Penalty = isolated_nodes
```

Goal:

> Maximize long-term swarm connectivity.

Impact:

- Faster recovery
- Smart radius adjustment
- Failures limited to ≤ 2 nodes

---

# 🎛 Orchestrator Layer

The orchestrator integrates:

- GAE
- VGAE
- GAN
- Diffusion
- Reinforcement Learning

Responsibilities:

- Aggregate model outputs
- Finalize adjacency decisions
- Update drone states
- Stabilize topology

This creates:

> A coordinated multi-layer intelligent swarm.

---

# 📊 Visualization

## Streamlit Dashboard

- Live swarm graph
- Connectivity metrics
- Failure monitor
- Density trends
- Alive ratio tracking

## Unity 3D Simulation

- Real-time swarm movement
- 3D topology visualization
- Dynamic link updates

---

# 📈 Results

| System Type              | Connectivity | Recovery Time | Failed Nodes |
|--------------------------|-------------|--------------|-------------|
| Static System            | 81.6%       | 14 steps     | 7           |
| Adaptive Radius System   | 89.4%       | 9 steps      | 4           |
| Proposed Intelligent System | 96.8%   | 4 steps      | ≤ 2         |

Improvements:

- Higher density
- Faster stabilization
- Gradual healing
- Minimal isolation

---

# 🔄 System Flow

1. Swarm starts normally  
2. Failures are injected  
3. Connectivity drops  
4. Graph models predict missing links  
5. Diffusion smooths topology  
6. RL adjusts communication radius  
7. Orchestrator aggregates decisions  
8. Swarm stabilizes  
9. Connectivity restored  
10. Failures minimized  

---

# 🛠 Tech Stack

- Python
- PyTorch
- PyTorch Geometric
- NetworkX
- NumPy
- Streamlit
- Unity 3D

---

# 🧠 Big Picture Transformation

```
Static Rule-Based Swarm
        ↓
Graph-Based Learning
        ↓
Generative Topology Modeling
        ↓
Adaptive Reinforcement Healing
        ↓
Orchestrated Intelligent Swarm
```

---

# 🏁 Conclusion

This project demonstrates a multi-model intelligent drone swarm communication framework that:

- Learns structural patterns  
- Predicts failures  
- Heals broken links  
- Adapts communication dynamically  
- Maintains high connectivity under disruption  

> In one line:  
> An adaptive, self-healing, graph-learning powered intelligent drone swarm system.
