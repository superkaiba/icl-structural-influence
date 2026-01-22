#!/usr/bin/env python3
"""
Create Lee et al. Figure 4b style MDS visualization.

Shows how individual token representations branch apart over context length,
with color indicating the "learning" progression (context length).
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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models import HookedLLM
from data.hierarchical_graph import HierarchicalGraph, HierarchicalGraphConfig


def collect_token_representations(model, tokenizer, graph, context_lengths,
                                   n_samples=50, layer_idx=-5):
    """
    Collect representations for each token at each context length.

    Returns:
        token_reps: dict[context_len][token_text] = mean_representation
        token_to_cluster: dict[token_text] = cluster_id
    """
    token_reps = {ctx: {} for ctx in context_lengths}
    token_to_cluster = {}

    # Map tokens to clusters
    for node_id in range(graph.num_nodes):
        token_text = graph.node_to_token[node_id]
        cluster_id = graph.get_cluster(node_id)
        token_to_cluster[token_text] = cluster_id

    for ctx_len in context_lengths:
        print(f"  Processing N={ctx_len}...")

        # Collect representations for each token
        token_representations = defaultdict(list)

        for sample_idx in range(n_samples):
            # Generate random walk
            prompt, node_sequence = graph.generate_random_walk(
                length=ctx_len,
                return_nodes=True
            )

            # Tokenize
            tokens = tokenizer.encode(prompt, add_special_tokens=False)
            input_ids = torch.tensor([tokens]).to(model.device)

            with torch.no_grad():
                outputs = model(input_ids, output_hidden_states=True)
                hidden_states = outputs.hidden_states[layer_idx][0]

            # Map representations to tokens
            token_texts = prompt.split()
            for pos, (node, token_text) in enumerate(zip(node_sequence, token_texts)):
                if pos < hidden_states.shape[0]:
                    rep = hidden_states[pos].cpu().float().numpy()
                    token_representations[token_text].append(rep)

        # Compute mean representation per token
        for token_text, reps in token_representations.items():
            if len(reps) > 0:
                token_reps[ctx_len][token_text] = np.mean(reps, axis=0)

    return token_reps, token_to_cluster


def compute_joint_mds(token_reps, context_lengths):
    """
    Compute MDS embedding for all tokens across all context lengths jointly.
    This ensures consistent embedding space.
    """
    # Collect all representations
    all_reps = []
    all_labels = []  # (token_text, context_len, cluster_id)

    # Get all unique tokens
    all_tokens = set()
    for ctx_len in context_lengths:
        all_tokens.update(token_reps[ctx_len].keys())

    for ctx_len in context_lengths:
        for token_text in all_tokens:
            if token_text in token_reps[ctx_len]:
                all_reps.append(token_reps[ctx_len][token_text])
                all_labels.append((token_text, ctx_len))

    # Run MDS
    all_reps = np.array(all_reps)
    print(f"  Running MDS on {len(all_reps)} points...")

    mds = MDS(n_components=2, random_state=42, n_init=4, max_iter=300, normalized_stress='auto')
    embedded = mds.fit_transform(all_reps)

    return embedded, all_labels


def find_branching_points(embedded, all_labels, token_to_cluster, context_lengths):
    """
    Find context lengths where clusters branch apart.
    """
    # Compute inter-cluster distance at each context length
    distances = {}

    for ctx_len in context_lengths:
        # Get points at this context length
        cluster_centroids = defaultdict(list)

        for i, (token_text, ctx) in enumerate(all_labels):
            if ctx == ctx_len and token_text in token_to_cluster:
                cluster_id = token_to_cluster[token_text]
                cluster_centroids[cluster_id].append(embedded[i])

        # Compute mean centroid per cluster
        centroids = {}
        for cluster_id, points in cluster_centroids.items():
            centroids[cluster_id] = np.mean(points, axis=0)

        # Compute inter-cluster distances
        if len(centroids) >= 2:
            inter_dists = []
            for c1 in centroids:
                for c2 in centroids:
                    if c1 < c2:
                        inter_dists.append(np.linalg.norm(centroids[c1] - centroids[c2]))
            distances[ctx_len] = np.mean(inter_dists)
        else:
            distances[ctx_len] = 0

    # Find max rate of change (branching point)
    ctx_sorted = sorted(distances.keys())
    max_rate = 0
    branching_ctx = ctx_sorted[0]

    for i in range(1, len(ctx_sorted)):
        rate = (distances[ctx_sorted[i]] - distances[ctx_sorted[i-1]]) / (ctx_sorted[i] - ctx_sorted[i-1])
        if rate > max_rate:
            max_rate = rate
            branching_ctx = ctx_sorted[i]

    return distances, branching_ctx


def create_mds_figure(embedded, all_labels, token_to_cluster, token_reps,
                      context_lengths, graph, output_dir):
    """
    Create Lee et al. Figure 4b style MDS visualization.
    """
    fig = plt.figure(figsize=(18, 16))

    # Color setup - gradient by context length
    cmap = plt.cm.viridis
    norm = Normalize(vmin=min(context_lengths), vmax=max(context_lengths))

    # Cluster colors for trajectory lines
    cluster_colors = {0: '#e41a1c', 1: '#377eb8', 2: '#4daf4a'}
    cluster_names = {0: 'Cluster A', 1: 'Cluster B', 2: 'Cluster C'}

    # =========================================================================
    # Main plot: MDS with color by context length
    # =========================================================================
    ax1 = fig.add_subplot(2, 2, 1)

    # Plot all points colored by context length
    for i, (token_text, ctx_len) in enumerate(all_labels):
        color = cmap(norm(ctx_len))
        cluster_id = token_to_cluster.get(token_text, 0)
        marker = ['o', 's', '^'][cluster_id]  # Different markers for clusters
        ax1.scatter(embedded[i, 0], embedded[i, 1],
                   c=[color], marker=marker, s=40, alpha=0.7,
                   edgecolors='white', linewidths=0.3)

    # Add colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1)
    cbar.set_label('Context Length (N)', fontsize=11)

    # Add legend for cluster markers
    for cluster_id in range(3):
        marker = ['o', 's', '^'][cluster_id]
        ax1.scatter([], [], marker=marker, c='gray', s=60,
                   label=cluster_names[cluster_id])
    ax1.legend(loc='upper left', fontsize=9)

    ax1.set_xlabel("MDS Dimension 1", fontsize=12)
    ax1.set_ylabel("MDS Dimension 2", fontsize=12)
    ax1.set_title("A. MDS of Token Representations\n(Color = Context Length)",
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # =========================================================================
    # Plot with token trajectories
    # =========================================================================
    ax2 = fig.add_subplot(2, 2, 2)

    # Get all unique tokens
    all_tokens = list(set(t for t, _ in all_labels))

    # Build token trajectories
    token_trajectories = {t: [] for t in all_tokens}
    for i, (token_text, ctx_len) in enumerate(all_labels):
        token_trajectories[token_text].append((ctx_len, embedded[i]))

    # Sort trajectories by context length
    for token in token_trajectories:
        token_trajectories[token].sort(key=lambda x: x[0])

    # Plot trajectories colored by cluster
    for token_text in all_tokens:
        if token_text not in token_to_cluster:
            continue
        cluster_id = token_to_cluster[token_text]
        trajectory = token_trajectories[token_text]

        if len(trajectory) < 2:
            continue

        # Extract points
        points = np.array([p[1] for p in trajectory])

        # Plot trajectory line
        ax2.plot(points[:, 0], points[:, 1], '-',
                color=cluster_colors[cluster_id], alpha=0.3, linewidth=1)

        # Plot points with size by context
        for ctx_len, point in trajectory:
            size = 20 + (ctx_len / max(context_lengths)) * 80
            ax2.scatter(point[0], point[1],
                       c=cluster_colors[cluster_id], s=size, alpha=0.6,
                       edgecolors='white', linewidths=0.3)

        # Label final position for some tokens
        if len(trajectory) > 0:
            final_point = trajectory[-1][1]
            # Only label a few tokens per cluster to avoid clutter
            if token_text in list(token_reps[context_lengths[-1]].keys())[:2]:
                ax2.annotate(token_text, xy=(final_point[0], final_point[1]),
                           fontsize=8, alpha=0.8, color=cluster_colors[cluster_id])

    # Add legend
    for cluster_id in range(3):
        ax2.plot([], [], '-o', color=cluster_colors[cluster_id],
                label=cluster_names[cluster_id], markersize=6)
    ax2.legend(loc='upper left', fontsize=9)

    ax2.set_xlabel("MDS Dimension 1", fontsize=12)
    ax2.set_ylabel("MDS Dimension 2", fontsize=12)
    ax2.set_title("B. Token Trajectories by Cluster\n(Larger = More Context)",
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # =========================================================================
    # Cluster centroid trajectories (cleaner view)
    # =========================================================================
    ax3 = fig.add_subplot(2, 2, 3)

    # Compute cluster centroids at each context length
    cluster_trajectories = {c: [] for c in range(3)}

    for ctx_len in context_lengths:
        cluster_points = defaultdict(list)

        for i, (token_text, ctx) in enumerate(all_labels):
            if ctx == ctx_len and token_text in token_to_cluster:
                cluster_id = token_to_cluster[token_text]
                cluster_points[cluster_id].append(embedded[i])

        for cluster_id in range(3):
            if cluster_points[cluster_id]:
                centroid = np.mean(cluster_points[cluster_id], axis=0)
                cluster_trajectories[cluster_id].append((ctx_len, centroid))

    # Plot centroid trajectories
    for cluster_id in range(3):
        trajectory = cluster_trajectories[cluster_id]
        if len(trajectory) < 2:
            continue

        points = np.array([p[1] for p in trajectory])
        ctxs = [p[0] for p in trajectory]

        # Plot trajectory
        ax3.plot(points[:, 0], points[:, 1], '-',
                color=cluster_colors[cluster_id], alpha=0.5, linewidth=2)

        # Plot points colored by context length
        for j, (ctx_len, point) in enumerate(trajectory):
            color = cmap(norm(ctx_len))
            size = 50 + (ctx_len / max(context_lengths)) * 150
            ax3.scatter(point[0], point[1], c=[color], s=size,
                       edgecolors=cluster_colors[cluster_id], linewidths=2)

        # Label start and end
        if len(points) > 0:
            ax3.annotate(f'{cluster_names[cluster_id]}\n(N={ctxs[0]})',
                        xy=(points[0, 0], points[0, 1]),
                        fontsize=8, alpha=0.7)
            ax3.annotate(f'{cluster_names[cluster_id]}\n(N={ctxs[-1]})',
                        xy=(points[-1, 0], points[-1, 1]),
                        fontsize=9, fontweight='bold', color=cluster_colors[cluster_id])

    # Find and mark branching points
    distances, branching_ctx = find_branching_points(embedded, all_labels, token_to_cluster, context_lengths)

    # Mark branching point on trajectory
    for cluster_id in range(3):
        trajectory = cluster_trajectories[cluster_id]
        for ctx_len, point in trajectory:
            if ctx_len == branching_ctx:
                ax3.scatter(point[0], point[1], c='black', s=200, marker='*',
                           zorder=10, label='Branching Point' if cluster_id == 0 else '')

    ax3.legend(loc='upper left', fontsize=9)
    ax3.set_xlabel("MDS Dimension 1", fontsize=12)
    ax3.set_ylabel("MDS Dimension 2", fontsize=12)
    ax3.set_title("C. Cluster Centroid Trajectories\n(Stars = Branching Point)",
                  fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Add colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax3)
    cbar.set_label('Context Length (N)', fontsize=11)

    # =========================================================================
    # Inter-cluster distance plot
    # =========================================================================
    ax4 = fig.add_subplot(2, 2, 4)

    ctx_sorted = sorted(distances.keys())
    dist_vals = [distances[c] for c in ctx_sorted]

    # Color points by context length
    colors = [cmap(norm(c)) for c in ctx_sorted]

    ax4.plot(ctx_sorted, dist_vals, 'k-', alpha=0.5, linewidth=1.5)
    ax4.scatter(ctx_sorted, dist_vals, c=colors, s=100, edgecolors='white', linewidths=1, zorder=5)

    # Mark branching point
    ax4.axvline(x=branching_ctx, color='red', linestyle='--', linewidth=2, alpha=0.7,
               label=f'Max Branching Rate (N={branching_ctx})')
    ax4.scatter([branching_ctx], [distances[branching_ctx]], c='red', s=200, marker='*', zorder=10)

    ax4.set_xlabel("Context Length (N)", fontsize=12)
    ax4.set_ylabel("Inter-Cluster Distance (MDS)", fontsize=12)
    ax4.set_title("D. Hierarchical Branching\n(Distance Between Cluster Centroids)",
                  fontsize=13, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    fig.suptitle("Lee et al. (2025) Figure 4b Replication: MDS of Token Representations\n" +
                 "Showing Hierarchical Branching Over Context Length",
                 fontsize=14, fontweight='bold', y=1.02)

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_dir / "mds_branching.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / "mds_branching.pdf", bbox_inches='tight', facecolor='white')

    print(f"\nFigure saved to: {output_dir / 'mds_branching.png'}")

    plt.close()

    return distances, branching_ctx


def main():
    print("=" * 80)
    print("LEE ET AL. FIGURE 4B: MDS BRANCHING VISUALIZATION")
    print("=" * 80)

    # Configuration
    model_name = "meta-llama/Llama-3.1-8B"
    context_lengths = [5, 10, 15, 20, 30, 50, 75, 100]
    n_samples = 50
    layer_idx = -5
    output_dir = Path("results/stagewise_learning")

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

    # Create graph
    print("\nCreating hierarchical graph...")
    config = HierarchicalGraphConfig(
        num_superclusters=3,
        nodes_per_cluster=5,
        p_intra_cluster=0.8,
        p_inter_cluster=0.1,
        seed=42
    )
    graph = HierarchicalGraph(config)

    # Print token assignments
    print("\nToken-to-Cluster Assignments:")
    for node_id in range(graph.num_nodes):
        token = graph.node_to_token[node_id]
        cluster = graph.get_cluster(node_id)
        print(f"  {token}: Cluster {['A', 'B', 'C'][cluster]}")

    # Collect representations
    print("\n" + "-" * 60)
    print("Collecting token representations...")
    print("-" * 60)

    token_reps, token_to_cluster = collect_token_representations(
        model, tokenizer, graph, context_lengths,
        n_samples=n_samples, layer_idx=layer_idx
    )

    # Compute joint MDS
    print("\n" + "-" * 60)
    print("Computing joint MDS embedding...")
    print("-" * 60)

    embedded, all_labels = compute_joint_mds(token_reps, context_lengths)

    # Create visualization
    print("\n" + "-" * 60)
    print("Creating MDS branching figure...")
    print("-" * 60)

    distances, branching_ctx = create_mds_figure(
        embedded, all_labels, token_to_cluster, token_reps,
        context_lengths, graph, output_dir
    )

    # Save results
    results = {
        "model": model_name,
        "context_lengths": context_lengths,
        "branching_ctx": branching_ctx,
        "distances": {str(k): v for k, v in distances.items()},
        "tokens_per_cluster": {
            "A": [t for t, c in token_to_cluster.items() if c == 0],
            "B": [t for t, c in token_to_cluster.items() if c == 1],
            "C": [t for t, c in token_to_cluster.items() if c == 2],
        }
    }

    with open(output_dir / "mds_branching_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("MDS BRANCHING ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nBranching point detected at: N={branching_ctx}")
    print(f"Results saved to: {output_dir}")

    return results


if __name__ == "__main__":
    main()
