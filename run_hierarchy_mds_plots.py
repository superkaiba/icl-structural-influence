#!/usr/bin/env python3
"""
Create MDS token trajectory plots for the 3-level hierarchical graph.
Uses MDS only (as per user preference) with expanded vocabulary (16 tokens)
and finer context length sampling for denser visualizations.
"""

import json
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from sklearn.manifold import MDS
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent / "src"))
from models import HookedLLM

# Import the 3-level hierarchy graph from the previous script
from run_hierarchy_and_semantic_experiments import HierarchicalGraph3Level


def collect_representations(model, tokenizer, graph, context_lengths, n_samples=50, layer_idx=-5):
    """Collect token representations at each context length."""
    token_reps = {ctx: {} for ctx in context_lengths}

    for ctx_len in context_lengths:
        print(f"    N={ctx_len}...", end=" ", flush=True)
        token_representations = defaultdict(list)

        for _ in range(n_samples):
            prompt, node_sequence = graph.generate_random_walk(length=ctx_len, return_nodes=True)
            tokens = tokenizer.encode(prompt, add_special_tokens=False)
            input_ids = torch.tensor([tokens]).to(model.device)

            with torch.no_grad():
                outputs = model(input_ids, output_hidden_states=True)
                hidden_states = outputs.hidden_states[layer_idx][0]

            token_texts = prompt.split()
            for pos, (node, token_text) in enumerate(zip(node_sequence, token_texts)):
                if pos < hidden_states.shape[0]:
                    rep = hidden_states[pos].cpu().float().numpy()
                    token_representations[token_text].append(rep)

        for token_text, reps in token_representations.items():
            if reps:
                token_reps[ctx_len][token_text] = np.mean(reps, axis=0)

        print(f"({len(token_reps[ctx_len])} tokens)")

    return token_reps


def prepare_data_matrix(token_reps, context_lengths):
    """Prepare data matrix for dimensionality reduction."""
    all_reps = []
    all_labels = []

    all_tokens = set()
    for ctx_len in context_lengths:
        all_tokens.update(token_reps[ctx_len].keys())

    for ctx_len in context_lengths:
        for token_text in sorted(all_tokens):
            if token_text in token_reps[ctx_len]:
                all_reps.append(token_reps[ctx_len][token_text])
                all_labels.append((token_text, ctx_len))

    return np.array(all_reps), all_labels


def apply_mds(data_matrix):
    """Apply MDS dimensionality reduction."""
    print(f"  Applying MDS...")
    reducer = MDS(n_components=2, random_state=42, n_init=4, max_iter=300,
                 normalized_stress='auto')
    embedded = reducer.fit_transform(data_matrix)
    return embedded


def create_trajectory_plot(ax, embedded, all_labels, graph, context_lengths,
                           method_name, color_by="super"):
    """
    Plot token trajectories with hierarchical coloring.

    color_by: "super" (Level 1) or "mid" (Level 2)
    """
    hierarchy = graph.get_hierarchy_labels()

    # Color schemes
    if color_by == "super":
        # Color by super-cluster (Level 1)
        colors = {0: '#e41a1c', 1: '#377eb8'}  # Red = Super_A, Blue = Super_B
        get_cluster = lambda t: hierarchy["level1"][graph.token_to_node[t]]
        legend_labels = {0: 'Super_A (8 tokens)', 1: 'Super_B (8 tokens)'}
    else:
        # Color by mid-cluster (Level 2) - 4 mid-clusters
        colors = {0: '#e41a1c', 1: '#ff7f00', 2: '#377eb8', 3: '#00bfff'}
        get_cluster = lambda t: hierarchy["level2"][graph.token_to_node[t]]
        legend_labels = {0: 'Mid_A1 (4 tokens)', 1: 'Mid_A2 (4 tokens)',
                         2: 'Mid_B1 (4 tokens)', 3: 'Mid_B2 (4 tokens)'}

    # Build token trajectories
    all_tokens = list(set(t for t, _ in all_labels))
    token_trajectories = {t: [] for t in all_tokens}

    for i, (token_text, ctx_len) in enumerate(all_labels):
        token_trajectories[token_text].append((ctx_len, embedded[i]))

    for token in token_trajectories:
        token_trajectories[token].sort(key=lambda x: x[0])

    # Plot trajectories
    for token_text in all_tokens:
        cluster_id = get_cluster(token_text)
        trajectory = token_trajectories[token_text]

        if len(trajectory) < 2:
            continue

        points = np.array([p[1] for p in trajectory])

        # Plot trajectory line
        ax.plot(points[:, 0], points[:, 1], '-',
               color=colors[cluster_id], alpha=0.5, linewidth=1.5)

        # Plot points with size by context
        for ctx_len, point in trajectory:
            size = 30 + (ctx_len / max(context_lengths)) * 120
            ax.scatter(point[0], point[1],
                      c=colors[cluster_id], s=size, alpha=0.7,
                      edgecolors='white', linewidths=0.3)

    # Add legend
    for cluster_id in sorted(colors.keys()):
        ax.scatter([], [], c=colors[cluster_id], s=80,
                  label=legend_labels[cluster_id])

    ax.legend(loc='upper left', fontsize=8)
    ax.set_xlabel("Dimension 1", fontsize=10)
    ax.set_ylabel("Dimension 2", fontsize=10)
    ax.set_title(f"{method_name}", fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)


def create_centroid_trajectory_plot(ax, embedded, all_labels, graph, context_lengths,
                                    method_name, level="super"):
    """Plot cluster centroid trajectories."""
    hierarchy = graph.get_hierarchy_labels()
    cmap = plt.cm.viridis
    norm = Normalize(vmin=min(context_lengths), vmax=max(context_lengths))

    if level == "super":
        colors = {0: '#e41a1c', 1: '#377eb8'}
        names = {0: 'Super_A', 1: 'Super_B'}
        get_cluster = lambda t: hierarchy["level1"][graph.token_to_node[t]]
        n_clusters = 2
    else:
        colors = {0: '#e41a1c', 1: '#ff7f00', 2: '#377eb8', 3: '#00bfff'}
        names = {0: 'Mid_A1', 1: 'Mid_A2', 2: 'Mid_B1', 3: 'Mid_B2'}
        get_cluster = lambda t: hierarchy["level2"][graph.token_to_node[t]]
        n_clusters = 4

    # Compute cluster centroids at each context length
    cluster_trajectories = {c: [] for c in range(n_clusters)}

    for ctx_len in context_lengths:
        cluster_points = defaultdict(list)

        for i, (token_text, ctx) in enumerate(all_labels):
            if ctx == ctx_len:
                cluster_id = get_cluster(token_text)
                cluster_points[cluster_id].append(embedded[i])

        for cluster_id in range(n_clusters):
            if cluster_points[cluster_id]:
                centroid = np.mean(cluster_points[cluster_id], axis=0)
                cluster_trajectories[cluster_id].append((ctx_len, centroid))

    # Plot centroid trajectories
    for cluster_id in range(n_clusters):
        trajectory = cluster_trajectories[cluster_id]
        if len(trajectory) < 2:
            continue

        points = np.array([p[1] for p in trajectory])
        ctxs = [p[0] for p in trajectory]

        # Plot trajectory line
        ax.plot(points[:, 0], points[:, 1], '-',
               color=colors[cluster_id], alpha=0.6, linewidth=2.5)

        # Plot points colored by context length
        for j, (ctx_len, point) in enumerate(trajectory):
            color = cmap(norm(ctx_len))
            size = 80 + (ctx_len / max(context_lengths)) * 150
            ax.scatter(point[0], point[1], c=[color], s=size,
                      edgecolors=colors[cluster_id], linewidths=2.5)

        # Label final position
        if len(points) > 0:
            ax.annotate(names[cluster_id],
                       xy=(points[-1, 0], points[-1, 1]),
                       fontsize=9, fontweight='bold', color=colors[cluster_id])

    ax.set_xlabel("Dimension 1", fontsize=10)
    ax.set_ylabel("Dimension 2", fontsize=10)
    level_name = "Super-Cluster" if level == "super" else "Mid-Cluster"
    ax.set_title(f"{method_name}\n({level_name} Centroids)", fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Context Length (N)', fontsize=9)


def main():
    print("=" * 70)
    print("3-LEVEL HIERARCHY: MDS TOKEN TRAJECTORY PLOTS")
    print("=" * 70)

    # Configuration - very long context to see mid-cluster separation
    model_name = "meta-llama/Llama-3.1-8B"
    # Fine-grained context lengths, especially at larger N
    context_lengths = [
        1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100,
        150, 200, 300, 400, 500, 750, 1000, 1250, 1500, 1750, 2000
    ]
    n_samples = 50
    layer_idx = -5
    output_dir = Path("results/hierarchy_mds_plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\nLoading model: {model_name}...")
    hooked_model = HookedLLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = hooked_model.model
    tokenizer = hooked_model.tokenizer
    model.eval()

    # Create 3-level hierarchy graph (now with 16 tokens)
    print("\nCreating 3-level hierarchical graph (16 tokens)...")
    graph = HierarchicalGraph3Level(seed=42)

    hierarchy = graph.get_hierarchy_labels()
    print("\nHierarchy structure (16 tokens, 4 per mid-cluster):")
    for node in range(graph.num_nodes):
        token = graph.node_to_token[node]
        super_c = hierarchy["super_names"][hierarchy["level1"][node]]
        mid_c = hierarchy["mid_names"][hierarchy["level2"][node]]
        print(f"  {token}: {super_c} -> {mid_c}")

    # Collect representations
    print("\nCollecting token representations...")
    token_reps = collect_representations(
        model, tokenizer, graph, context_lengths, n_samples, layer_idx
    )

    # Prepare data matrix
    data_matrix, all_labels = prepare_data_matrix(token_reps, context_lengths)
    print(f"\nData matrix shape: {data_matrix.shape}")
    print(f"  (16 tokens x 12 context lengths = 192 data points)")

    # Apply MDS only (per user preference)
    print("\nApplying MDS dimensionality reduction...")
    embedded_mds = apply_mds(data_matrix)

    # =========================================================================
    # Figure 1: Token trajectories colored by SUPER-CLUSTER (Level 1)
    # =========================================================================
    print("\nCreating Figure 1: Token trajectories by Super-Cluster (MDS)...")
    fig1, ax1 = plt.subplots(figsize=(12, 10))
    create_trajectory_plot(ax1, embedded_mds, all_labels, graph,
                          context_lengths, "MDS", color_by="super")
    fig1.suptitle("3-Level Hierarchy: Token Trajectories by Super-Cluster (MDS)\n" +
                  "(Red = Super_A, Blue = Super_B | 16 tokens, 12 context lengths)",
                  fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig1.savefig(output_dir / "hierarchy_trajectories_super.png", dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close(fig1)

    # =========================================================================
    # Figure 2: Token trajectories colored by MID-CLUSTER (Level 2)
    # =========================================================================
    print("Creating Figure 2: Token trajectories by Mid-Cluster (MDS)...")
    fig2, ax2 = plt.subplots(figsize=(12, 10))
    create_trajectory_plot(ax2, embedded_mds, all_labels, graph,
                          context_lengths, "MDS", color_by="mid")
    fig2.suptitle("3-Level Hierarchy: Token Trajectories by Mid-Cluster (MDS)\n" +
                  "(4 mid-clusters: A1, A2 under Super_A; B1, B2 under Super_B)",
                  fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig2.savefig(output_dir / "hierarchy_trajectories_mid.png", dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close(fig2)

    # =========================================================================
    # Figure 3: Super-cluster centroid trajectories
    # =========================================================================
    print("Creating Figure 3: Super-cluster centroid trajectories (MDS)...")
    fig3, ax3 = plt.subplots(figsize=(12, 10))
    create_centroid_trajectory_plot(ax3, embedded_mds, all_labels, graph,
                                    context_lengths, "MDS", level="super")
    fig3.suptitle("3-Level Hierarchy: Super-Cluster Centroid Trajectories (MDS)\n" +
                  "(Color gradient = Context length progression)",
                  fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig3.savefig(output_dir / "hierarchy_centroids_super.png", dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close(fig3)

    # =========================================================================
    # Figure 4: Mid-cluster centroid trajectories
    # =========================================================================
    print("Creating Figure 4: Mid-cluster centroid trajectories (MDS)...")
    fig4, ax4 = plt.subplots(figsize=(12, 10))
    create_centroid_trajectory_plot(ax4, embedded_mds, all_labels, graph,
                                    context_lengths, "MDS", level="mid")
    fig4.suptitle("3-Level Hierarchy: Mid-Cluster Centroid Trajectories (MDS)\n" +
                  "(Shows branching within Super-clusters)",
                  fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig4.savefig(output_dir / "hierarchy_centroids_mid.png", dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close(fig4)

    # =========================================================================
    # Figure 5: Combined Lee et al. style figure (MDS only)
    # =========================================================================
    print("Creating Figure 5: Combined Lee et al. style figure (MDS)...")
    fig5 = plt.figure(figsize=(18, 12))

    cmap = plt.cm.viridis
    norm = Normalize(vmin=min(context_lengths), vmax=max(context_lengths))

    # Panel A: All tokens colored by context length
    ax1 = fig5.add_subplot(2, 3, 1)
    for i, (token_text, ctx_len) in enumerate(all_labels):
        color = cmap(norm(ctx_len))
        super_cluster = hierarchy["level1"][graph.token_to_node[token_text]]
        marker = 'o' if super_cluster == 0 else 's'
        ax1.scatter(embedded_mds[i, 0], embedded_mds[i, 1],
                   c=[color], marker=marker, s=50, alpha=0.7,
                   edgecolors='white', linewidths=0.3)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1)
    cbar.set_label('Context Length (N)', fontsize=10)
    ax1.scatter([], [], marker='o', c='gray', s=60, label='Super_A (8 tokens)')
    ax1.scatter([], [], marker='s', c='gray', s=60, label='Super_B (8 tokens)')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_xlabel("MDS Dimension 1", fontsize=11)
    ax1.set_ylabel("MDS Dimension 2", fontsize=11)
    ax1.set_title("A. All Tokens\n(Color = Context Length)", fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Panel B: Token trajectories by super-cluster
    ax2 = fig5.add_subplot(2, 3, 2)
    create_trajectory_plot(ax2, embedded_mds, all_labels, graph,
                          context_lengths, "B. Token Trajectories", color_by="super")

    # Panel C: Token trajectories by mid-cluster
    ax3 = fig5.add_subplot(2, 3, 3)
    create_trajectory_plot(ax3, embedded_mds, all_labels, graph,
                          context_lengths, "C. Token Trajectories (Mid)", color_by="mid")

    # Panel D: Super-cluster centroids
    ax4 = fig5.add_subplot(2, 3, 4)
    create_centroid_trajectory_plot(ax4, embedded_mds, all_labels, graph,
                                    context_lengths, "D. Super-Cluster Centroids", level="super")

    # Panel E: Mid-cluster centroids
    ax5 = fig5.add_subplot(2, 3, 5)
    create_centroid_trajectory_plot(ax5, embedded_mds, all_labels, graph,
                                    context_lengths, "E. Mid-Cluster Centroids", level="mid")

    # Panel F: Interpretation
    ax6 = fig5.add_subplot(2, 3, 6)
    ax6.axis('off')

    interpretation_text = """
    LEE ET AL. STYLE MDS VISUALIZATION
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    HIERARCHY STRUCTURE (16 tokens):
    ─────────────────────────────────
              ALL
             /   \\
        Super_A  Super_B
         /  \\     /  \\
      M_A1  M_A2  M_B1  M_B2

    Tokens (4 per mid-cluster):
    • M_A1: crystal, marble, diamond, granite
    • M_A2: lantern, castle, beacon, fortress
    • M_B1: cloud, canvas, mist, fabric
    • M_B2: pillar, tunnel, column, passage

    EXPECTED STAGEWISE LEARNING:
    ─────────────────────────────
    1. EARLY (N=1-5): All tokens similar

    2. MIDDLE (N=10-30): Super-clusters separate
       (Super_A vs Super_B)

    3. LATE (N=50+): Mid-clusters separate
       (M_A1 vs M_A2 within Super_A)

    DATA: 16 tokens x 12 context lengths
    = 192 total MDS points
    """

    ax6.text(0.05, 0.95, interpretation_text, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    ax6.set_title("F. Interpretation", fontsize=12, fontweight='bold')

    plt.tight_layout()
    fig5.suptitle("3-Level Hierarchy: Lee et al. Style MDS Visualization\n" +
                  "Showing Hierarchical Branching Over Context Length (16 tokens, 12 context lengths)",
                  fontsize=14, fontweight='bold', y=1.02)
    fig5.savefig(output_dir / "hierarchy_lee_et_al_style.png", dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close(fig5)

    # =========================================================================
    # Figure 6: ZOOMED IN - Late context (N >= 30) with labels
    # =========================================================================
    print("Creating Figure 6: Zoomed-in late context with labels...")

    # Filter to late context points only
    late_ctx = [30, 50, 75, 100]
    late_indices = [i for i, (_, ctx) in enumerate(all_labels) if ctx in late_ctx]
    late_embedded = embedded_mds[late_indices]
    late_labels = [all_labels[i] for i in late_indices]

    fig6, axes6 = plt.subplots(1, 2, figsize=(18, 8))

    # Panel A: By super-cluster with labels
    ax = axes6[0]
    super_colors = {0: '#e41a1c', 1: '#377eb8'}

    for i, (token_text, ctx_len) in enumerate(late_labels):
        node = graph.token_to_node[token_text]
        super_cluster = hierarchy["level1"][node]
        color = super_colors[super_cluster]

        ax.scatter(late_embedded[i, 0], late_embedded[i, 1],
                  c=color, s=150, alpha=0.8,
                  edgecolors='black', linewidths=1)
        # Label with context length
        ax.annotate(f"N={ctx_len}", (late_embedded[i, 0], late_embedded[i, 1]),
                   fontsize=7, ha='center', va='bottom',
                   xytext=(0, 5), textcoords='offset points')

    ax.scatter([], [], c='#e41a1c', s=100, label='Super_A')
    ax.scatter([], [], c='#377eb8', s=100, label='Super_B')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlabel("MDS Dimension 1", fontsize=12)
    ax.set_ylabel("MDS Dimension 2", fontsize=12)
    ax.set_title("A. Late Context (N=30-100) by Super-Cluster\nLabeled by Context Length",
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Panel B: By mid-cluster with labels
    ax = axes6[1]
    mid_colors = {0: '#e41a1c', 1: '#ff7f00', 2: '#377eb8', 3: '#00bfff'}
    mid_names = {0: 'Mid_A1', 1: 'Mid_A2', 2: 'Mid_B1', 3: 'Mid_B2'}

    for i, (token_text, ctx_len) in enumerate(late_labels):
        node = graph.token_to_node[token_text]
        mid_cluster = hierarchy["level2"][node]
        color = mid_colors[mid_cluster]

        ax.scatter(late_embedded[i, 0], late_embedded[i, 1],
                  c=color, s=150, alpha=0.8,
                  edgecolors='black', linewidths=1)
        ax.annotate(f"N={ctx_len}", (late_embedded[i, 0], late_embedded[i, 1]),
                   fontsize=7, ha='center', va='bottom',
                   xytext=(0, 5), textcoords='offset points')

    for mid_id, name in mid_names.items():
        ax.scatter([], [], c=mid_colors[mid_id], s=100, label=name)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlabel("MDS Dimension 1", fontsize=12)
    ax.set_ylabel("MDS Dimension 2", fontsize=12)
    ax.set_title("B. Late Context (N=30-100) by Mid-Cluster\nLabeled by Context Length",
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig6.suptitle("Zoomed-In: Late Context Representations (N=30, 50, 75, 100)\n" +
                  "Should Show Clear Hierarchical Clustering",
                  fontsize=14, fontweight='bold', y=1.02)
    fig6.savefig(output_dir / "hierarchy_zoomed_late.png", dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close(fig6)

    # =========================================================================
    # Figure 7: Centroid trajectories with ALL labels
    # =========================================================================
    print("Creating Figure 7: Centroid trajectories with context length labels...")

    fig7, axes7 = plt.subplots(1, 2, figsize=(18, 8))

    # Compute centroids at each context length
    def compute_centroids(embedded, all_labels, graph, hierarchy, level="super"):
        if level == "super":
            get_cluster = lambda t: hierarchy["level1"][graph.token_to_node[t]]
            n_clusters = 2
        else:
            get_cluster = lambda t: hierarchy["level2"][graph.token_to_node[t]]
            n_clusters = 4

        centroids = {c: {} for c in range(n_clusters)}

        for ctx_len in context_lengths:
            cluster_points = defaultdict(list)
            for i, (token_text, ctx) in enumerate(all_labels):
                if ctx == ctx_len:
                    cluster_id = get_cluster(token_text)
                    cluster_points[cluster_id].append(embedded[i])

            for cluster_id in range(n_clusters):
                if cluster_points[cluster_id]:
                    centroids[cluster_id][ctx_len] = np.mean(cluster_points[cluster_id], axis=0)

        return centroids

    # Panel A: Super-cluster centroids with labels
    ax = axes7[0]
    super_centroids = compute_centroids(embedded_mds, all_labels, graph, hierarchy, "super")
    super_colors = {0: '#e41a1c', 1: '#377eb8'}
    super_names = {0: 'Super_A', 1: 'Super_B'}

    for cluster_id in [0, 1]:
        ctxs = sorted(super_centroids[cluster_id].keys())
        points = np.array([super_centroids[cluster_id][c] for c in ctxs])

        # Plot trajectory line
        ax.plot(points[:, 0], points[:, 1], '-', color=super_colors[cluster_id],
               linewidth=2, alpha=0.5)

        # Plot points with labels
        for j, ctx in enumerate(ctxs):
            ax.scatter(points[j, 0], points[j, 1], c=super_colors[cluster_id],
                      s=120, edgecolors='black', linewidths=1, zorder=10)
            ax.annotate(f"{ctx}", (points[j, 0], points[j, 1]),
                       fontsize=8, fontweight='bold', ha='center', va='bottom',
                       xytext=(0, 6), textcoords='offset points',
                       color=super_colors[cluster_id])

        # Label endpoint
        ax.annotate(super_names[cluster_id], (points[-1, 0], points[-1, 1]),
                   fontsize=11, fontweight='bold', ha='left', va='center',
                   xytext=(10, 0), textcoords='offset points',
                   color=super_colors[cluster_id])

    ax.set_xlabel("MDS Dimension 1", fontsize=12)
    ax.set_ylabel("MDS Dimension 2", fontsize=12)
    ax.set_title("A. Super-Cluster Centroid Trajectories\n(Numbers = Context Length N)",
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Panel B: Mid-cluster centroids with labels
    ax = axes7[1]
    mid_centroids = compute_centroids(embedded_mds, all_labels, graph, hierarchy, "mid")
    mid_colors = {0: '#e41a1c', 1: '#ff7f00', 2: '#377eb8', 3: '#00bfff'}
    mid_names = {0: 'M_A1', 1: 'M_A2', 2: 'M_B1', 3: 'M_B2'}

    for cluster_id in range(4):
        ctxs = sorted(mid_centroids[cluster_id].keys())
        points = np.array([mid_centroids[cluster_id][c] for c in ctxs])

        # Plot trajectory line
        ax.plot(points[:, 0], points[:, 1], '-', color=mid_colors[cluster_id],
               linewidth=2, alpha=0.5)

        # Plot points with labels
        for j, ctx in enumerate(ctxs):
            ax.scatter(points[j, 0], points[j, 1], c=mid_colors[cluster_id],
                      s=100, edgecolors='black', linewidths=1, zorder=10)
            # Only label key context lengths to avoid clutter
            if ctx in [1, 5, 10, 30, 100]:
                ax.annotate(f"{ctx}", (points[j, 0], points[j, 1]),
                           fontsize=7, fontweight='bold', ha='center', va='bottom',
                           xytext=(0, 5), textcoords='offset points',
                           color=mid_colors[cluster_id])

        # Label endpoint
        ax.annotate(mid_names[cluster_id], (points[-1, 0], points[-1, 1]),
                   fontsize=10, fontweight='bold', ha='left', va='center',
                   xytext=(8, 0), textcoords='offset points',
                   color=mid_colors[cluster_id])

    ax.set_xlabel("MDS Dimension 1", fontsize=12)
    ax.set_ylabel("MDS Dimension 2", fontsize=12)
    ax.set_title("B. Mid-Cluster Centroid Trajectories\n(Numbers = Context Length N)",
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig7.suptitle("Centroid Trajectories with Context Length Labels\n" +
                  "Shows How Clusters Evolve and Separate Over Context",
                  fontsize=14, fontweight='bold', y=1.02)
    fig7.savefig(output_dir / "hierarchy_centroids_labeled.png", dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close(fig7)

    # =========================================================================
    # Figure 8: Final snapshot at N=100 with token names
    # =========================================================================
    print("Creating Figure 8: Final snapshot at N=100 with token names...")

    # Filter to N=100 only
    final_indices = [i for i, (_, ctx) in enumerate(all_labels) if ctx == 100]
    final_embedded = embedded_mds[final_indices]
    final_labels = [all_labels[i] for i in final_indices]

    fig8, axes8 = plt.subplots(1, 2, figsize=(16, 7))

    # Panel A: By super-cluster
    ax = axes8[0]
    for i, (token_text, _) in enumerate(final_labels):
        node = graph.token_to_node[token_text]
        super_cluster = hierarchy["level1"][node]
        color = super_colors[super_cluster]
        marker = 'o' if super_cluster == 0 else 's'

        ax.scatter(final_embedded[i, 0], final_embedded[i, 1],
                  c=color, marker=marker, s=200, alpha=0.8,
                  edgecolors='black', linewidths=1.5)
        ax.annotate(token_text, (final_embedded[i, 0], final_embedded[i, 1]),
                   fontsize=9, ha='center', va='bottom',
                   xytext=(0, 8), textcoords='offset points',
                   fontweight='bold')

    ax.scatter([], [], c='#e41a1c', marker='o', s=150, label='Super_A')
    ax.scatter([], [], c='#377eb8', marker='s', s=150, label='Super_B')
    ax.legend(loc='upper left', fontsize=11)
    ax.set_xlabel("MDS Dimension 1", fontsize=12)
    ax.set_ylabel("MDS Dimension 2", fontsize=12)
    ax.set_title("A. Final Representations (N=100) by Super-Cluster",
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Panel B: By mid-cluster
    ax = axes8[1]
    markers = {0: 'o', 1: '^', 2: 's', 3: 'D'}

    for i, (token_text, _) in enumerate(final_labels):
        node = graph.token_to_node[token_text]
        mid_cluster = hierarchy["level2"][node]
        color = mid_colors[mid_cluster]
        marker = markers[mid_cluster]

        ax.scatter(final_embedded[i, 0], final_embedded[i, 1],
                  c=color, marker=marker, s=200, alpha=0.8,
                  edgecolors='black', linewidths=1.5)
        ax.annotate(token_text, (final_embedded[i, 0], final_embedded[i, 1]),
                   fontsize=9, ha='center', va='bottom',
                   xytext=(0, 8), textcoords='offset points',
                   fontweight='bold')

    for mid_id, name in mid_names.items():
        ax.scatter([], [], c=mid_colors[mid_id], marker=markers[mid_id],
                  s=150, label=name)
    ax.legend(loc='upper left', fontsize=11)
    ax.set_xlabel("MDS Dimension 1", fontsize=12)
    ax.set_ylabel("MDS Dimension 2", fontsize=12)
    ax.set_title("B. Final Representations (N=100) by Mid-Cluster",
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig8.suptitle("Final Token Representations at N=100\n" +
                  "Showing Learned Hierarchical Structure",
                  fontsize=14, fontweight='bold', y=1.02)
    fig8.savefig(output_dir / "hierarchy_final_n100.png", dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close(fig8)

    # =========================================================================
    # Figure 9: Mid-cluster centroid evolution with lines and sized points
    # =========================================================================
    print("Creating Figure 9: Mid-cluster evolution with trajectory lines...")

    mid_centroids = compute_centroids(embedded_mds, all_labels, graph, hierarchy, "mid")
    mid_colors = {0: '#e41a1c', 1: '#ff7f00', 2: '#377eb8', 3: '#00bfff'}
    mid_names = {0: 'Mid_A1\n(crystal, marble,\ndiamond, granite)',
                 1: 'Mid_A2\n(lantern, castle,\nbeacon, fortress)',
                 2: 'Mid_B1\n(cloud, canvas,\nmist, fabric)',
                 3: 'Mid_B2\n(pillar, tunnel,\ncolumn, passage)'}
    mid_short = {0: 'M_A1', 1: 'M_A2', 2: 'M_B1', 3: 'M_B2'}

    # Size scaling: N=1 -> small, N=100 -> large
    min_size, max_size = 80, 400
    size_scale = lambda ctx: min_size + (ctx / max(context_lengths)) * (max_size - min_size)

    # Find zoom region (final points: N >= 50)
    final_points_mid = []
    for cluster_id in range(4):
        for ctx in [50, 75, 100]:
            if ctx in mid_centroids[cluster_id]:
                final_points_mid.append(mid_centroids[cluster_id][ctx])
    final_points_mid = np.array(final_points_mid)

    # Calculate zoom bounds with padding
    zoom_pad = 0.15
    x_min, x_max = final_points_mid[:, 0].min(), final_points_mid[:, 0].max()
    y_min, y_max = final_points_mid[:, 1].min(), final_points_mid[:, 1].max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    zoom_xlim = (x_min - zoom_pad * x_range, x_max + zoom_pad * x_range)
    zoom_ylim = (y_min - zoom_pad * y_range, y_max + zoom_pad * y_range)

    fig9, ax = plt.subplots(figsize=(14, 12))

    for cluster_id in range(4):
        ctxs = sorted(mid_centroids[cluster_id].keys())
        points = np.array([mid_centroids[cluster_id][c] for c in ctxs])

        # Plot trajectory line (thick, semi-transparent)
        ax.plot(points[:, 0], points[:, 1], '-', color=mid_colors[cluster_id],
               linewidth=3, alpha=0.6, zorder=1)

        # Plot points with size proportional to context length
        for j, ctx in enumerate(ctxs):
            size = size_scale(ctx)
            ax.scatter(points[j, 0], points[j, 1], c=mid_colors[cluster_id],
                      s=size, edgecolors='black', linewidths=1.5, zorder=10, alpha=0.9)

            # Label with context length
            ax.annotate(f"{ctx}", (points[j, 0], points[j, 1]),
                       fontsize=9, fontweight='bold', ha='center', va='center',
                       color='white' if size > 200 else 'black',
                       zorder=11)

        # Label the cluster at the endpoint
        ax.annotate(mid_names[cluster_id], (points[-1, 0], points[-1, 1]),
                   fontsize=10, fontweight='bold', ha='left', va='center',
                   xytext=(15, 0), textcoords='offset points',
                   color=mid_colors[cluster_id],
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Add legend for point sizes
    for ctx_example in [1, 10, 50, 100]:
        ax.scatter([], [], c='gray', s=size_scale(ctx_example),
                  edgecolors='black', linewidths=1, label=f'N={ctx_example}')

    ax.legend(loc='upper left', fontsize=10, title='Point Size = Context Length')
    ax.set_xlabel("MDS Dimension 1", fontsize=13)
    ax.set_ylabel("MDS Dimension 2", fontsize=13)
    ax.set_title("Mid-Cluster Centroid Evolution Over Context Length\n" +
                "(Larger points = More context | Numbers inside = N)",
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add zoomed inset on bottom left
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins = ax.inset_axes([0.02, 0.02, 0.4, 0.4])  # [x, y, width, height] in axes coords

    for cluster_id in range(4):
        ctxs = sorted(mid_centroids[cluster_id].keys())
        points = np.array([mid_centroids[cluster_id][c] for c in ctxs])

        # Only plot late context points in zoom
        late_mask = [c >= 30 for c in ctxs]
        late_ctxs = [c for c in ctxs if c >= 30]
        late_points = points[[i for i, c in enumerate(ctxs) if c >= 30]]

        if len(late_points) > 1:
            axins.plot(late_points[:, 0], late_points[:, 1], '-', color=mid_colors[cluster_id],
                      linewidth=2, alpha=0.6)

        for j, ctx in enumerate(late_ctxs):
            idx = ctxs.index(ctx)
            size = size_scale(ctx) * 0.7  # Slightly smaller in inset
            axins.scatter(points[idx, 0], points[idx, 1], c=mid_colors[cluster_id],
                         s=size, edgecolors='black', linewidths=1, zorder=10, alpha=0.9)
            axins.annotate(f"{ctx}", (points[idx, 0], points[idx, 1]),
                          fontsize=7, fontweight='bold', ha='center', va='center',
                          color='white' if size > 150 else 'black', zorder=11)

        # Label cluster in inset
        if len(late_points) > 0:
            axins.annotate(mid_short[cluster_id], (late_points[-1, 0], late_points[-1, 1]),
                          fontsize=9, fontweight='bold', ha='left', va='center',
                          xytext=(8, 0), textcoords='offset points',
                          color=mid_colors[cluster_id])

    axins.set_xlim(zoom_xlim)
    axins.set_ylim(zoom_ylim)
    axins.set_title("Zoom: Final Points (N≥30)", fontsize=10, fontweight='bold')
    axins.grid(True, alpha=0.3)
    axins.set_facecolor('lightyellow')

    # Draw rectangle showing zoom region on main plot
    from matplotlib.patches import Rectangle
    rect = Rectangle((zoom_xlim[0], zoom_ylim[0]),
                     zoom_xlim[1] - zoom_xlim[0], zoom_ylim[1] - zoom_ylim[0],
                     linewidth=2, edgecolor='orange', facecolor='none', linestyle='--')
    ax.add_patch(rect)

    plt.tight_layout()
    fig9.savefig(output_dir / "hierarchy_midcluster_evolution.png", dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close(fig9)

    # =========================================================================
    # Figure 10: Same but with arrows showing direction
    # =========================================================================
    print("Creating Figure 10: Mid-cluster evolution with arrows...")

    fig10, ax = plt.subplots(figsize=(14, 12))

    for cluster_id in range(4):
        ctxs = sorted(mid_centroids[cluster_id].keys())
        points = np.array([mid_centroids[cluster_id][c] for c in ctxs])

        # Plot arrows between consecutive points
        for j in range(len(points) - 1):
            ax.annotate('', xy=(points[j+1, 0], points[j+1, 1]),
                       xytext=(points[j, 0], points[j, 1]),
                       arrowprops=dict(arrowstyle='->', color=mid_colors[cluster_id],
                                      lw=2, alpha=0.7))

        # Plot points with size proportional to context length
        for j, ctx in enumerate(ctxs):
            size = size_scale(ctx)
            ax.scatter(points[j, 0], points[j, 1], c=mid_colors[cluster_id],
                      s=size, edgecolors='black', linewidths=1.5, zorder=10, alpha=0.9)

            # Label with context length
            ax.annotate(f"{ctx}", (points[j, 0], points[j, 1]),
                       fontsize=9, fontweight='bold', ha='center', va='center',
                       color='white' if size > 200 else 'black',
                       zorder=11)

        # Label the cluster at the endpoint
        ax.annotate(mid_short[cluster_id], (points[-1, 0], points[-1, 1]),
                   fontsize=12, fontweight='bold', ha='left', va='center',
                   xytext=(15, 0), textcoords='offset points',
                   color=mid_colors[cluster_id],
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # Mark start point
        ax.scatter(points[0, 0], points[0, 1], c=mid_colors[cluster_id],
                  s=size_scale(ctxs[0]), marker='o', edgecolors='green',
                  linewidths=3, zorder=12)

    # Add legend
    for ctx_example in [1, 10, 50, 100]:
        ax.scatter([], [], c='gray', s=size_scale(ctx_example),
                  edgecolors='black', linewidths=1, label=f'N={ctx_example}')
    ax.scatter([], [], c='gray', s=100, edgecolors='green', linewidths=3, label='Start (N=1)')

    ax.legend(loc='upper left', fontsize=10, title='Point Size = Context Length')
    ax.set_xlabel("MDS Dimension 1", fontsize=13)
    ax.set_ylabel("MDS Dimension 2", fontsize=13)
    ax.set_title("Mid-Cluster Centroid Evolution with Direction Arrows\n" +
                "(Arrows show progression from low to high context)",
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add zoomed inset on bottom left
    axins = ax.inset_axes([0.02, 0.02, 0.4, 0.4])

    for cluster_id in range(4):
        ctxs = sorted(mid_centroids[cluster_id].keys())
        points = np.array([mid_centroids[cluster_id][c] for c in ctxs])
        late_ctxs = [c for c in ctxs if c >= 30]

        # Plot arrows for late context
        for j in range(len(ctxs) - 1):
            if ctxs[j] >= 30 or ctxs[j+1] >= 30:
                axins.annotate('', xy=(points[j+1, 0], points[j+1, 1]),
                              xytext=(points[j, 0], points[j, 1]),
                              arrowprops=dict(arrowstyle='->', color=mid_colors[cluster_id],
                                             lw=1.5, alpha=0.7))

        for j, ctx in enumerate(late_ctxs):
            idx = ctxs.index(ctx)
            size = size_scale(ctx) * 0.7
            axins.scatter(points[idx, 0], points[idx, 1], c=mid_colors[cluster_id],
                         s=size, edgecolors='black', linewidths=1, zorder=10, alpha=0.9)
            axins.annotate(f"{ctx}", (points[idx, 0], points[idx, 1]),
                          fontsize=7, fontweight='bold', ha='center', va='center',
                          color='white' if size > 150 else 'black', zorder=11)

        # Label in inset
        if late_ctxs:
            idx = ctxs.index(late_ctxs[-1])
            axins.annotate(mid_short[cluster_id], (points[idx, 0], points[idx, 1]),
                          fontsize=9, fontweight='bold', ha='left', va='center',
                          xytext=(8, 0), textcoords='offset points',
                          color=mid_colors[cluster_id])

    axins.set_xlim(zoom_xlim)
    axins.set_ylim(zoom_ylim)
    axins.set_title("Zoom: Final Points (N≥30)", fontsize=10, fontweight='bold')
    axins.grid(True, alpha=0.3)
    axins.set_facecolor('lightyellow')

    # Draw rectangle showing zoom region
    rect = Rectangle((zoom_xlim[0], zoom_ylim[0]),
                     zoom_xlim[1] - zoom_xlim[0], zoom_ylim[1] - zoom_ylim[0],
                     linewidth=2, edgecolor='orange', facecolor='none', linestyle='--')
    ax.add_patch(rect)

    plt.tight_layout()
    fig10.savefig(output_dir / "hierarchy_midcluster_arrows.png", dpi=300,
                 bbox_inches='tight', facecolor='white')
    plt.close(fig10)

    # =========================================================================
    # Figure 11: Super-cluster evolution (same style)
    # =========================================================================
    print("Creating Figure 11: Super-cluster evolution with arrows...")

    fig11, ax = plt.subplots(figsize=(14, 12))

    super_centroids = compute_centroids(embedded_mds, all_labels, graph, hierarchy, "super")
    super_colors = {0: '#e41a1c', 1: '#377eb8'}
    super_names = {0: 'Super_A\n(M_A1 + M_A2)', 1: 'Super_B\n(M_B1 + M_B2)'}
    super_short = {0: 'Super_A', 1: 'Super_B'}

    # Find zoom region for super-clusters
    final_points_super = []
    for cluster_id in [0, 1]:
        for ctx in [50, 75, 100]:
            if ctx in super_centroids[cluster_id]:
                final_points_super.append(super_centroids[cluster_id][ctx])
    final_points_super = np.array(final_points_super)

    zoom_pad_s = 0.2
    x_min_s, x_max_s = final_points_super[:, 0].min(), final_points_super[:, 0].max()
    y_min_s, y_max_s = final_points_super[:, 1].min(), final_points_super[:, 1].max()
    x_range_s = max(x_max_s - x_min_s, 0.1)
    y_range_s = max(y_max_s - y_min_s, 0.1)
    zoom_xlim_s = (x_min_s - zoom_pad_s * x_range_s, x_max_s + zoom_pad_s * x_range_s)
    zoom_ylim_s = (y_min_s - zoom_pad_s * y_range_s, y_max_s + zoom_pad_s * y_range_s)

    for cluster_id in [0, 1]:
        ctxs = sorted(super_centroids[cluster_id].keys())
        points = np.array([super_centroids[cluster_id][c] for c in ctxs])

        # Plot arrows between consecutive points
        for j in range(len(points) - 1):
            ax.annotate('', xy=(points[j+1, 0], points[j+1, 1]),
                       xytext=(points[j, 0], points[j, 1]),
                       arrowprops=dict(arrowstyle='->', color=super_colors[cluster_id],
                                      lw=3, alpha=0.7))

        # Plot points with size proportional to context length
        for j, ctx in enumerate(ctxs):
            size = size_scale(ctx)
            ax.scatter(points[j, 0], points[j, 1], c=super_colors[cluster_id],
                      s=size, edgecolors='black', linewidths=2, zorder=10, alpha=0.9)

            # Label with context length
            ax.annotate(f"{ctx}", (points[j, 0], points[j, 1]),
                       fontsize=10, fontweight='bold', ha='center', va='center',
                       color='white' if size > 200 else 'black',
                       zorder=11)

        # Label the cluster at the endpoint
        ax.annotate(super_names[cluster_id], (points[-1, 0], points[-1, 1]),
                   fontsize=12, fontweight='bold', ha='left', va='center',
                   xytext=(20, 0), textcoords='offset points',
                   color=super_colors[cluster_id],
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # Mark start point
        ax.scatter(points[0, 0], points[0, 1], c=super_colors[cluster_id],
                  s=size_scale(ctxs[0]), marker='o', edgecolors='green',
                  linewidths=3, zorder=12)

    # Add legend
    for ctx_example in [1, 10, 50, 100]:
        ax.scatter([], [], c='gray', s=size_scale(ctx_example),
                  edgecolors='black', linewidths=1, label=f'N={ctx_example}')
    ax.scatter([], [], c='gray', s=100, edgecolors='green', linewidths=3, label='Start (N=1)')

    ax.legend(loc='upper left', fontsize=10, title='Point Size = Context Length')
    ax.set_xlabel("MDS Dimension 1", fontsize=13)
    ax.set_ylabel("MDS Dimension 2", fontsize=13)
    ax.set_title("Super-Cluster Centroid Evolution with Direction Arrows\n" +
                "(Shows how the two super-clusters diverge over context)",
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add zoomed inset on bottom left
    axins = ax.inset_axes([0.02, 0.02, 0.4, 0.4])

    for cluster_id in [0, 1]:
        ctxs = sorted(super_centroids[cluster_id].keys())
        points = np.array([super_centroids[cluster_id][c] for c in ctxs])
        late_ctxs = [c for c in ctxs if c >= 30]

        # Plot arrows for late context
        for j in range(len(ctxs) - 1):
            if ctxs[j] >= 30 or ctxs[j+1] >= 30:
                axins.annotate('', xy=(points[j+1, 0], points[j+1, 1]),
                              xytext=(points[j, 0], points[j, 1]),
                              arrowprops=dict(arrowstyle='->', color=super_colors[cluster_id],
                                             lw=2, alpha=0.7))

        for j, ctx in enumerate(late_ctxs):
            idx = ctxs.index(ctx)
            size = size_scale(ctx) * 0.7
            axins.scatter(points[idx, 0], points[idx, 1], c=super_colors[cluster_id],
                         s=size, edgecolors='black', linewidths=1.5, zorder=10, alpha=0.9)
            axins.annotate(f"{ctx}", (points[idx, 0], points[idx, 1]),
                          fontsize=8, fontweight='bold', ha='center', va='center',
                          color='white' if size > 150 else 'black', zorder=11)

        # Label in inset
        if late_ctxs:
            idx = ctxs.index(late_ctxs[-1])
            axins.annotate(super_short[cluster_id], (points[idx, 0], points[idx, 1]),
                          fontsize=10, fontweight='bold', ha='left', va='center',
                          xytext=(10, 0), textcoords='offset points',
                          color=super_colors[cluster_id])

    axins.set_xlim(zoom_xlim_s)
    axins.set_ylim(zoom_ylim_s)
    axins.set_title("Zoom: Final Points (N≥30)", fontsize=10, fontweight='bold')
    axins.grid(True, alpha=0.3)
    axins.set_facecolor('lightyellow')

    # Draw rectangle showing zoom region
    rect = Rectangle((zoom_xlim_s[0], zoom_ylim_s[0]),
                     zoom_xlim_s[1] - zoom_xlim_s[0], zoom_ylim_s[1] - zoom_ylim_s[0],
                     linewidth=2, edgecolor='orange', facecolor='none', linestyle='--')
    ax.add_patch(rect)

    plt.tight_layout()
    fig11.savefig(output_dir / "hierarchy_supercluster_arrows.png", dpi=300,
                 bbox_inches='tight', facecolor='white')
    plt.close(fig11)

    # =========================================================================
    # Figure 12: Combined - Both levels showing branching with zoom insets
    # =========================================================================
    print("Creating Figure 12: Combined super + mid cluster evolution...")

    fig12, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Panel A: Super-clusters
    ax = axes[0]
    for cluster_id in [0, 1]:
        ctxs = sorted(super_centroids[cluster_id].keys())
        points = np.array([super_centroids[cluster_id][c] for c in ctxs])

        for j in range(len(points) - 1):
            ax.annotate('', xy=(points[j+1, 0], points[j+1, 1]),
                       xytext=(points[j, 0], points[j, 1]),
                       arrowprops=dict(arrowstyle='->', color=super_colors[cluster_id],
                                      lw=3, alpha=0.7))

        for j, ctx in enumerate(ctxs):
            size = size_scale(ctx)
            ax.scatter(points[j, 0], points[j, 1], c=super_colors[cluster_id],
                      s=size, edgecolors='black', linewidths=2, zorder=10, alpha=0.9)
            ax.annotate(f"{ctx}", (points[j, 0], points[j, 1]),
                       fontsize=9, fontweight='bold', ha='center', va='center',
                       color='white' if size > 200 else 'black', zorder=11)

        ax.annotate(['Super_A', 'Super_B'][cluster_id], (points[-1, 0], points[-1, 1]),
                   fontsize=14, fontweight='bold', ha='left', va='center',
                   xytext=(15, 0), textcoords='offset points',
                   color=super_colors[cluster_id])

    ax.set_xlabel("MDS Dimension 1", fontsize=12)
    ax.set_ylabel("MDS Dimension 2", fontsize=12)
    ax.set_title("A. Super-Cluster Evolution\n(First level of hierarchy)", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add zoom inset to Panel A
    axins_a = ax.inset_axes([0.02, 0.02, 0.35, 0.35])
    for cluster_id in [0, 1]:
        ctxs = sorted(super_centroids[cluster_id].keys())
        points = np.array([super_centroids[cluster_id][c] for c in ctxs])
        late_ctxs = [c for c in ctxs if c >= 30]

        for j in range(len(ctxs) - 1):
            if ctxs[j] >= 30 or ctxs[j+1] >= 30:
                axins_a.annotate('', xy=(points[j+1, 0], points[j+1, 1]),
                                xytext=(points[j, 0], points[j, 1]),
                                arrowprops=dict(arrowstyle='->', color=super_colors[cluster_id],
                                               lw=2, alpha=0.7))

        for ctx in late_ctxs:
            idx = ctxs.index(ctx)
            size = size_scale(ctx) * 0.6
            axins_a.scatter(points[idx, 0], points[idx, 1], c=super_colors[cluster_id],
                           s=size, edgecolors='black', linewidths=1, zorder=10, alpha=0.9)
            axins_a.annotate(f"{ctx}", (points[idx, 0], points[idx, 1]),
                            fontsize=7, fontweight='bold', ha='center', va='center',
                            color='white' if size > 120 else 'black', zorder=11)

    axins_a.set_xlim(zoom_xlim_s)
    axins_a.set_ylim(zoom_ylim_s)
    axins_a.set_title("Zoom", fontsize=9, fontweight='bold')
    axins_a.grid(True, alpha=0.3)
    axins_a.set_facecolor('lightyellow')
    rect_a = Rectangle((zoom_xlim_s[0], zoom_ylim_s[0]),
                       zoom_xlim_s[1] - zoom_xlim_s[0], zoom_ylim_s[1] - zoom_ylim_s[0],
                       linewidth=2, edgecolor='orange', facecolor='none', linestyle='--')
    ax.add_patch(rect_a)

    # Panel B: Mid-clusters
    ax = axes[1]
    for cluster_id in range(4):
        ctxs = sorted(mid_centroids[cluster_id].keys())
        points = np.array([mid_centroids[cluster_id][c] for c in ctxs])

        for j in range(len(points) - 1):
            ax.annotate('', xy=(points[j+1, 0], points[j+1, 1]),
                       xytext=(points[j, 0], points[j, 1]),
                       arrowprops=dict(arrowstyle='->', color=mid_colors[cluster_id],
                                      lw=2, alpha=0.7))

        for j, ctx in enumerate(ctxs):
            size = size_scale(ctx)
            ax.scatter(points[j, 0], points[j, 1], c=mid_colors[cluster_id],
                      s=size, edgecolors='black', linewidths=1.5, zorder=10, alpha=0.9)
            ax.annotate(f"{ctx}", (points[j, 0], points[j, 1]),
                       fontsize=8, fontweight='bold', ha='center', va='center',
                       color='white' if size > 200 else 'black', zorder=11)

        ax.annotate(mid_short[cluster_id], (points[-1, 0], points[-1, 1]),
                   fontsize=12, fontweight='bold', ha='left', va='center',
                   xytext=(12, 0), textcoords='offset points',
                   color=mid_colors[cluster_id])

    ax.set_xlabel("MDS Dimension 1", fontsize=12)
    ax.set_ylabel("MDS Dimension 2", fontsize=12)
    ax.set_title("B. Mid-Cluster Evolution\n(Second level - branching within super-clusters)", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add zoom inset to Panel B
    axins_b = ax.inset_axes([0.02, 0.02, 0.35, 0.35])
    for cluster_id in range(4):
        ctxs = sorted(mid_centroids[cluster_id].keys())
        points = np.array([mid_centroids[cluster_id][c] for c in ctxs])
        late_ctxs = [c for c in ctxs if c >= 30]

        for j in range(len(ctxs) - 1):
            if ctxs[j] >= 30 or ctxs[j+1] >= 30:
                axins_b.annotate('', xy=(points[j+1, 0], points[j+1, 1]),
                                xytext=(points[j, 0], points[j, 1]),
                                arrowprops=dict(arrowstyle='->', color=mid_colors[cluster_id],
                                               lw=1.5, alpha=0.7))

        for ctx in late_ctxs:
            idx = ctxs.index(ctx)
            size = size_scale(ctx) * 0.6
            axins_b.scatter(points[idx, 0], points[idx, 1], c=mid_colors[cluster_id],
                           s=size, edgecolors='black', linewidths=1, zorder=10, alpha=0.9)
            axins_b.annotate(f"{ctx}", (points[idx, 0], points[idx, 1]),
                            fontsize=6, fontweight='bold', ha='center', va='center',
                            color='white' if size > 120 else 'black', zorder=11)

    axins_b.set_xlim(zoom_xlim)
    axins_b.set_ylim(zoom_ylim)
    axins_b.set_title("Zoom", fontsize=9, fontweight='bold')
    axins_b.grid(True, alpha=0.3)
    axins_b.set_facecolor('lightyellow')
    rect_b = Rectangle((zoom_xlim[0], zoom_ylim[0]),
                       zoom_xlim[1] - zoom_xlim[0], zoom_ylim[1] - zoom_ylim[0],
                       linewidth=2, edgecolor='orange', facecolor='none', linestyle='--')
    ax.add_patch(rect_b)

    # Shared legend
    legend_elements = [plt.scatter([], [], c='gray', s=size_scale(ctx), edgecolors='black', linewidths=1)
                      for ctx in [1, 10, 50, 100]]
    fig12.legend(legend_elements, ['N=1', 'N=10', 'N=50', 'N=100'],
                loc='lower center', ncol=4, fontsize=11,
                title='Point Size = Context Length', bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    fig12.suptitle("Hierarchical Clustering Evolution: Lee et al. Stagewise Learning\n" +
                  "(Numbers inside points = Context length N | Yellow insets show zoom on final points)",
                  fontsize=15, fontweight='bold', y=1.02)
    fig12.savefig(output_dir / "hierarchy_evolution_combined.png", dpi=300,
                 bbox_inches='tight', facecolor='white')
    plt.close(fig12)

    print("\n" + "=" * 70)
    print("PLOTS COMPLETE")
    print("=" * 70)
    print(f"\nSaved to: {output_dir}")
    print("  - hierarchy_trajectories_super.png")
    print("  - hierarchy_trajectories_mid.png")
    print("  - hierarchy_centroids_super.png")
    print("  - hierarchy_centroids_mid.png")
    print("  - hierarchy_lee_et_al_style.png")
    print("  - hierarchy_zoomed_late.png")
    print("  - hierarchy_centroids_labeled.png")
    print("  - hierarchy_final_n100.png")
    print("  - hierarchy_midcluster_evolution.png (NEW)")
    print("  - hierarchy_midcluster_arrows.png (NEW)")
    print("  - hierarchy_supercluster_arrows.png (NEW)")
    print("  - hierarchy_evolution_combined.png (NEW)")


if __name__ == "__main__":
    main()
