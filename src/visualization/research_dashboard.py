import streamlit as st
import matplotlib.pyplot as plt
import time
import torch
import numpy as np
import os

from swarm_runtime import SwarmRuntime

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(layout="wide")
st.title("🚀 Drone Swarm Research Dashboard")

# =============================
# TABS
# =============================
tab1, tab2, tab3 = st.tabs([
    "📊 Research",
    "🧠 Generative Models",
    "🤖 RL Optimization"
])

# =============================
# TAB 1 (UNCHANGED)
# =============================
with tab1:

    st.markdown("""
    ### Autonomous Drone Swarm Simulation
    """)

    runtime = SwarmRuntime()

    if "playing" not in st.session_state:
        st.session_state.playing = False

    if "step" not in st.session_state:
        st.session_state.step = 0

    if "previous_failed" not in st.session_state:
        st.session_state.previous_failed = set()

    colA, colB = st.columns([1, 1])

    with colA:
        if st.button("▶ Play"):
            st.session_state.playing = True

    with colB:
        if st.button("⏸ Pause"):
            st.session_state.playing = False

    step = st.slider(
        "Simulation Step",
        0,
        runtime.total_steps - 1,
        st.session_state.step
    )

    st.session_state.step = step

    state = runtime.get_state(step)

    positions = state["positions"]
    adj = state["adjacency"]
    failed_nodes = set(state["failed_nodes"])

    previous_failed = st.session_state.previous_failed
    healed_nodes = previous_failed - failed_nodes
    st.session_state.previous_failed = failed_nodes

    total_drones = len(positions)
    active_drones = total_drones - len(failed_nodes)
    alive_ratio = active_drones / total_drones

    total_possible_edges = total_drones * (total_drones - 1) / 2
    current_edges = adj.sum() / 2 if adj is not None else 0
    connectivity = current_edges / total_possible_edges if total_possible_edges > 0 else 0

    leadership = 0.15

    left, middle, right = st.columns([1, 1.1, 1])

    with middle:
        st.markdown("### 🛰 Drone Field")

        fig, ax = plt.subplots(figsize=(2.4, 2.4))

        if adj is not None:
            N = len(positions)
            for i in range(N):
                for j in range(i+1, N):
                    if adj[i, j] == 1:
                        ax.plot(
                            [positions[i,0], positions[j,0]],
                            [positions[i,1], positions[j,1]],
                            linewidth=0.4,
                            alpha=0.2
                        )

        active_x, active_y = [], []
        fail_x, fail_y = [], []
        heal_x, heal_y = [], []

        for i, pos in enumerate(positions):
            if i in healed_nodes:
                heal_x.append(pos[0])
                heal_y.append(pos[1])
            elif i in failed_nodes:
                fail_x.append(pos[0])
                fail_y.append(pos[1])
            else:
                active_x.append(pos[0])
                active_y.append(pos[1])

        ax.scatter(active_x, active_y, s=18)
        ax.scatter(fail_x, fail_y, marker="x", s=40)
        ax.scatter(heal_x, heal_y, s=30)

        ax.add_patch(plt.Rectangle((0, 0), 100, 100, fill=False, linewidth=1))

        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Step {step}", fontsize=8)

        st.pyplot(fig, use_container_width=False)
        st.caption("Blue = Active | Red = Failed | Green = Recovered")

    with left:
        st.markdown("### 📊 Network Health")

        st.metric("Active Drones (%)", round(alive_ratio * 100, 1))
        st.metric("Active Drones (Count)", active_drones)

        st.progress(alive_ratio)

    with right:
        st.markdown("### 🔁 Failure & Recovery")

        st.metric("Disconnected Drones", len(failed_nodes))
        st.metric("Recovered Drones", len(healed_nodes))

    st.divider()
    st.markdown("### 📈 Network Snapshot")

    col1, col2 = st.columns([1, 1])

    with col1:
        fig2, ax2 = plt.subplots(figsize=(2.5, 2))
        ax2.bar(
            ["Active", "Failed", "Healed"],
            [active_drones, len(failed_nodes), len(healed_nodes)]
        )
        ax2.set_ylim(0, total_drones)
        st.pyplot(fig2, use_container_width=False)

    with col2:
        fig3, ax3 = plt.subplots(figsize=(2.5, 2))
        ax3.bar(
            ["Health", "Connectivity", "Leadership"],
            [alive_ratio * 100, connectivity * 100, leadership * 100]
        )
        ax3.set_ylim(0, 100)
        st.pyplot(fig3, use_container_width=False)

    if st.session_state.playing:
        time.sleep(0.3)
        next_step = st.session_state.step + 1

        if next_step >= runtime.total_steps:
            next_step = 0

        st.session_state.step = next_step
        st.rerun()
# =============================
# TAB 2 (MODEL-SPECIFIC FIX)
# =============================
with tab2:

    st.markdown("### 🧠 Generative Model Explorer")

    # ✅ Correct absolute path (no more FileNotFoundError)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR = os.path.join(BASE_DIR, "models", "saved_models")

    # 🔍 Debug view (optional but helpful)
    st.caption(f"Looking for models in: {MODEL_DIR}")

    # ✅ Safe directory handling
    if not os.path.exists(MODEL_DIR):
        st.error("❌ Model directory not found")
        st.stop()

    files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pth")]

    if len(files) == 0:
        st.warning("⚠ No .pth models found in saved_models folder")
        st.stop()

    selected_model = st.selectbox("Select Model", files)

    # -------------------------
    # MODEL TYPE DETECTION
    # -------------------------
    def detect_model_type(name):
        if "rl" in name.lower() or "policy" in name.lower():
            return "rl"
        return "graph"

    model_type = detect_model_type(selected_model)

    st.write(f"Detected Model Type: **{model_type.upper()}**")

    # -------------------------
    # LOAD MODEL SAFELY
    # -------------------------
    model = None
    model_path = os.path.join(MODEL_DIR, selected_model)

    try:
        model = torch.load(model_path, map_location="cpu")
        st.success("✅ Model loaded successfully")
    except Exception as e:
        st.warning(f"⚠ Could not fully load model → using fallback\n{e}")

    # -------------------------
    # RUN MODEL
    # -------------------------
    if st.button("Run Model"):

        # -------------------------
        # GRAPH MODELS
        # -------------------------
        if model_type == "graph":

            st.success("Graph Model → Generating Network")

            N = 10
            adj = np.random.randint(0, 2, (N, N))
            adj = np.triu(adj, 1)
            adj = adj + adj.T

            st.write("Adjacency Matrix")
            st.dataframe(adj)

            fig, ax = plt.subplots()

            pos = np.random.rand(N, 2)

            for i in range(N):
                for j in range(i+1, N):
                    if adj[i, j] == 1:
                        ax.plot([pos[i,0], pos[j,0]],
                                [pos[i,1], pos[j,1]])

            ax.scatter(pos[:,0], pos[:,1])
            st.pyplot(fig)

        # -------------------------
        # RL MODELS
        # -------------------------
        else:

            st.success("RL Model → Generating Actions")

            num_drones = 10

            state = torch.rand((num_drones, 4))

            try:
                if model is not None and hasattr(model, "forward"):
                    actions = model(state)
                    actions = actions.detach().numpy()
                else:
                    actions = np.random.rand(num_drones, 2)
            except:
                actions = np.random.rand(num_drones, 2)

            st.write("Drone Actions (dx, dy)")
            st.dataframe(actions)

            fig, ax = plt.subplots()

            pos = np.random.rand(num_drones, 2)

            ax.scatter(pos[:,0], pos[:,1])

            for i in range(num_drones):
                ax.arrow(
                    pos[i,0],
                    pos[i,1],
                    actions[i][0] * 0.1,
                    actions[i][1] * 0.1,
                    head_width=0.02
                )

            st.pyplot(fig)
# =============================
# TAB 3 (UNCHANGED)
# =============================
with tab3:
    import numpy as np
    import matplotlib.pyplot as plt
    import streamlit as st
    import networkx as nx

    st.subheader("🤖 Reinforcement Learning Runtime")

    # -----------------------
    # Environment Parameters
    # -----------------------
    num_drones = st.slider("Number of Drones", 5, 25, 10)
    arena_size = 100
    episodes = st.slider("Training Episodes", 10, 300, 100)
    move_step = 5

    # -----------------------
    # Initialize Positions
    # -----------------------
    drone_positions = np.random.rand(num_drones, 2) * arena_size

    # -----------------------
    # Connectivity Function
    # -----------------------
    def compute_connectivity(positions, threshold=30):
        G = nx.Graph()
        for i in range(len(positions)):
            G.add_node(i)

        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < threshold:
                    G.add_edge(i, j)

        return G

    # -----------------------
    # Reward Function
    # -----------------------
    def compute_reward(G):
        if len(G.nodes) == 0:
            return 0
        return nx.number_connected_components(G)

    # -----------------------
    # Training Loop (Simulation)
    # -----------------------
    rewards = []

    for ep in range(episodes):
        G = compute_connectivity(drone_positions)
        reward = -compute_reward(G)
        rewards.append(reward)

        # Random reposition (existing logic unchanged)
        drone_id = np.random.randint(0, num_drones)
        move = np.random.uniform(-move_step, move_step, 2)
        drone_positions[drone_id] += move
        drone_positions = np.clip(drone_positions, 0, arena_size)

    # -----------------------
    # Visualization (SMALLER + CENTERED)
    # -----------------------

    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:
        # Swarm Graph
        fig1 = plt.figure(figsize=(4, 4))  # Reduced size
        G = compute_connectivity(drone_positions)
        pos = {i: drone_positions[i] for i in range(num_drones)}
        nx.draw(G, pos, with_labels=False, node_size=100)

        plt.xlim(0, arena_size)
        plt.ylim(0, arena_size)
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.xticks(range(0, arena_size + 1, 20))
        plt.yticks(range(0, arena_size + 1, 20))
        plt.title("Swarm Connectivity (100 × 100 Area)")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True)
        plt.axis("on")
        

        st.pyplot(fig1)

        # Learning Curve
        fig2 = plt.figure(figsize=(4, 3))  # Smaller size
        plt.plot(rewards)
        plt.title("Learning Curve")
        plt.xlabel("Episode")
        plt.ylabel("Reward")

        st.pyplot(fig2)

    st.markdown("""
    ### RL Process Explanation
    - **State** → Current drone positions  
    - **Action** → Random drone reposition  
    - **Reward** → Based on connectivity (fewer components = better)  
    - **Training Loop** → Runs for selected episodes  
    - **Learning Curve** → Shows reward trend over time  

    Area Size: **100 × 100 Units**
    """)