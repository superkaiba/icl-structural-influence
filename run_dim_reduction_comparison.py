#!/usr/bin/env python3
"""
Compare dimensionality reduction methods for token trajectory visualization.
Methods: PCA, t-SNE, UMAP, MDS
"""

import json
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from collections import defaultdict

# Dimensionality reduction methods
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
import umap

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models import HookedLLM
from data.hierarchical_graph import HierarchicalGraph, HierarchicalGraphConfig


def collect_token_representations(model, tokenizer, graph, context_lengths,
                                   n_samples=50, layer_idx=-5):
    """Collect representations for each token at each context length."""
    token_reps = {ctx: {} for ctx in context_lengths}
    token_to_cluster = {}

    # Map tokens to clusters
    for node_id in range(graph.num_nodes):
        token_text = graph.node_to_token[node_id]
        cluster_id = graph.get_cluster(node_id)
        token_to_cluster[token_text] = cluster_id

    for ctx_len in context_lengths:
        print(f"  Collecting N={ctx_len}...")
        token_representations = defaultdict(list)

        for sample_idx in range(n_samples):
            prompt, node_sequence = graph.generate_random_walk(
                length=ctx_len, return_nodes=True
            )
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
            if len(reps) > 0:
                token_reps[ctx_len][token_text] = np.mean(reps, axis=0)

    return token_reps, token_to_cluster


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


def apply_dim_reduction(data_matrix, method="pca"):
    """Apply dimensionality reduction."""
    print(f"  Applying {method.upper()}...")

    if method == "pca":
        reducer = PCA(n_components=2, random_state=42)
        embedded = reducer.fit_transform(data_matrix)

    elif method == "tsne":
        # t-SNE needs perplexity < n_samples
        perplexity = min(30, len(data_matrix) - 1)
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity,
                      max_iter=1000, learning_rate='auto', init='pca')
        embedded = reducer.fit_transform(data_matrix)

    elif method == "umap":
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15,
                           min_dist=0.1, metric='euclidean')
        embedded = reducer.fit_transform(data_matrix)

    elif method == "mds":
        reducer = MDS(n_components=2, random_state=42, n_init=4, max_iter=300,
                     normalized_stress='auto')
        embedded = reducer.fit_transform(data_matrix)

    return embedded


def plot_single_method(ax, embedded, all_labels, token_to_cluster, context_lengths,
                       method_name, show_trajectories=True):
    """Plot token trajectories for a single method."""

    cmap = plt.cm.viridis
    norm = Normalize(vmin=min(context_lengths), vmax=max(context_lengths))
    cluster_colors = {0: '#e41a1c', 1: '#377eb8', 2: '#4daf4a'}
    cluster_names = {0: 'A', 1: 'B', 2: 'C'}

    # Build token trajectories
    all_tokens = list(set(t for t, _ in all_labels))
    token_trajectories = {t: [] for t in all_tokens}

    for i, (token_text, ctx_len) in enumerate(all_labels):
        token_trajectories[token_text].append((ctx_len, embedded[i]))

    for token in token_trajectories:
        token_trajectories[token].sort(key=lambda x: x[0])

    # Plot trajectories
    for token_text in all_tokens:
        if token_text not in token_to_cluster:
            continue
        cluster_id = token_to_cluster[token_text]
        trajectory = token_trajectories[token_text]

        if len(trajectory) < 2:
            continue

        points = np.array([p[1] for p in trajectory])

        # Plot trajectory line
        if show_trajectories:
            ax.plot(points[:, 0], points[:, 1], '-',
                   color=cluster_colors[cluster_id], alpha=0.4, linewidth=1.5)

        # Plot points with size by context
        for ctx_len, point in trajectory:
            size = 20 + (ctx_len / max(context_lengths)) * 100
            ax.scatter(point[0], point[1],
                      c=cluster_colors[cluster_id], s=size, alpha=0.7,
                      edgecolors='white', linewidths=0.3)

    # Add legend
    for cluster_id in range(3):
        ax.scatter([], [], c=cluster_colors[cluster_id], s=60,
                  label=f'Cluster {cluster_names[cluster_id]}')

    ax.legend(loc='upper left', fontsize=8)
    ax.set_xlabel("Dimension 1", fontsize=10)
    ax.set_ylabel("Dimension 2", fontsize=10)
    ax.set_title(f"{method_name}", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)


def plot_cluster_centroids(ax, embedded, all_labels, token_to_cluster, context_lengths,
                           method_name):
    """Plot cluster centroid trajectories."""

    cmap = plt.cm.viridis
    norm = Normalize(vmin=min(context_lengths), vmax=max(context_lengths))
    cluster_colors = {0: '#e41a1c', 1: '#377eb8', 2: '#4daf4a'}
    cluster_names = {0: 'Cluster A', 1: 'Cluster B', 2: 'Cluster C'}

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
        ax.plot(points[:, 0], points[:, 1], '-',
               color=cluster_colors[cluster_id], alpha=0.6, linewidth=2.5)

        # Plot points colored by context length
        for j, (ctx_len, point) in enumerate(trajectory):
            color = cmap(norm(ctx_len))
            size = 60 + (ctx_len / max(context_lengths)) * 140
            ax.scatter(point[0], point[1], c=[color], s=size,
                      edgecolors=cluster_colors[cluster_id], linewidths=2.5)

        # Label endpoints
        ax.annotate(f'{cluster_names[cluster_id]}',
                   xy=(points[-1, 0], points[-1, 1]),
                   fontsize=9, fontweight='bold', color=cluster_colors[cluster_id])

    ax.set_xlabel("Dimension 1", fontsize=10)
    ax.set_ylabel("Dimension 2", fontsize=10)
    ax.set_title(f"{method_name}\n(Cluster Centroids)", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Context Length (N)', fontsize=9)


def create_comparison_figure(token_reps, token_to_cluster, context_lengths, output_dir):
    """Create comparison figure with all four methods."""

    # Prepare data
    data_matrix, all_labels = prepare_data_matrix(token_reps, context_lengths)
    print(f"\nData matrix shape: {data_matrix.shape}")

    # Apply all methods
    methods = {
        'PCA': apply_dim_reduction(data_matrix, 'pca'),
        't-SNE': apply_dim_reduction(data_matrix, 'tsne'),
        'UMAP': apply_dim_reduction(data_matrix, 'umap'),
        'MDS': apply_dim_reduction(data_matrix, 'mds'),
    }

    # =========================================================================
    # Figure 1: Token trajectories comparison
    # =========================================================================
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 14))
    axes1 = axes1.flatten()

    for idx, (method_name, embedded) in enumerate(methods.items()):
        plot_single_method(axes1[idx], embedded, all_labels, token_to_cluster,
                          context_lengths, method_name)

    fig1.suptitle("Token Trajectories: Comparing Dimensionality Reduction Methods\n" +
                  "(Larger dots = More context, Colors = Graph clusters)",
                  fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig1.savefig(output_dir / "dim_reduction_comparison_trajectories.png",
                dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_dir / 'dim_reduction_comparison_trajectories.png'}")
    plt.close(fig1)

    # =========================================================================
    # Figure 2: Cluster centroid trajectories comparison
    # =========================================================================
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 14))
    axes2 = axes2.flatten()

    for idx, (method_name, embedded) in enumerate(methods.items()):
        plot_cluster_centroids(axes2[idx], embedded, all_labels, token_to_cluster,
                              context_lengths, method_name)

    fig2.suptitle("Cluster Centroid Trajectories: Comparing Methods\n" +
                  "(Color gradient = Context length progression)",
                  fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    fig2.savefig(output_dir / "dim_reduction_comparison_centroids.png",
                dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_dir / 'dim_reduction_comparison_centroids.png'}")
    plt.close(fig2)

    # =========================================================================
    # Figure 3: Combined detailed view
    # =========================================================================
    fig3 = plt.figure(figsize=(20, 16))

    # Row 1: All methods - token trajectories
    for idx, (method_name, embedded) in enumerate(methods.items()):
        ax = fig3.add_subplot(3, 4, idx + 1)
        plot_single_method(ax, embedded, all_labels, token_to_cluster,
                          context_lengths, method_name, show_trajectories=True)

    # Row 2: All methods - centroids only
    for idx, (method_name, embedded) in enumerate(methods.items()):
        ax = fig3.add_subplot(3, 4, idx + 5)
        plot_cluster_centroids(ax, embedded, all_labels, token_to_cluster,
                              context_lengths, method_name)

    # Row 3: Method explanations
    ax_info = fig3.add_subplot(3, 1, 3)
    ax_info.axis('off')

    info_text = """
    DIMENSIONALITY REDUCTION METHODS COMPARISON
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    PCA (Principal Component Analysis)                    t-SNE (t-Distributed Stochastic Neighbor Embedding)
    ─────────────────────────────────────                 ──────────────────────────────────────────────────────
    • Linear projection onto directions of max variance   • Non-linear, preserves LOCAL neighborhood structure
    • Fast, deterministic                                 • Good for visualizing clusters
    • Preserves global structure                          • Distances between clusters less meaningful
    • Best for: Quick overview, linear relationships      • Best for: Finding tight clusters

    UMAP (Uniform Manifold Approximation)                 MDS (Multi-Dimensional Scaling)
    ─────────────────────────────────────                 ─────────────────────────────────────
    • Non-linear, preserves local AND global structure    • Preserves pairwise DISTANCES
    • Fast, scales well                                   • If A-B close in 4096D → close in 2D
    • Good balance of local/global                        • Slower, but interpretable distances
    • Best for: Large datasets, complex manifolds         • Best for: When distances matter

    INTERPRETATION FOR OUR EXPERIMENT:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    • If clusters separate CONSISTENTLY across all methods → Robust finding
    • If only some methods show separation → Method-dependent artifact
    • Look for: Same-cluster tokens (same color) staying together as context increases
    """

    ax_info.text(0.02, 0.95, info_text, transform=ax_info.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    fig3.suptitle("Complete Comparison: PCA vs t-SNE vs UMAP vs MDS\n" +
                  "Lee et al. Figure 4b Style Token Trajectory Visualization",
                  fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()

    fig3.savefig(output_dir / "dim_reduction_full_comparison.png",
                dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_dir / 'dim_reduction_full_comparison.png'}")
    plt.close(fig3)

    return methods


def main():
    print("=" * 70)
    print("DIMENSIONALITY REDUCTION COMPARISON")
    print("PCA vs t-SNE vs UMAP vs MDS")
    print("=" * 70)

    # Configuration
    model_name = "meta-llama/Llama-3.1-8B"
    context_lengths = [5, 10, 15, 20, 30, 50, 75, 100]
    n_samples = 50
    layer_idx = -5
    output_dir = Path("results/dim_reduction_comparison")

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

    print("\nToken assignments:")
    for cluster_id in range(3):
        tokens = [graph.node_to_token[n] for n in range(graph.num_nodes)
                  if graph.get_cluster(n) == cluster_id]
        print(f"  Cluster {['A', 'B', 'C'][cluster_id]}: {tokens}")

    # Collect representations
    print("\n" + "-" * 50)
    print("Collecting token representations...")
    print("-" * 50)

    token_reps, token_to_cluster = collect_token_representations(
        model, tokenizer, graph, context_lengths,
        n_samples=n_samples, layer_idx=layer_idx
    )

    # Create comparison figures
    print("\n" + "-" * 50)
    print("Creating comparison figures...")
    print("-" * 50)

    methods = create_comparison_figure(
        token_reps, token_to_cluster, context_lengths, output_dir
    )

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")

    return methods


if __name__ == "__main__":
    main()
