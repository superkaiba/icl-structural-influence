#!/usr/bin/env python3
"""
Analyze whether there's structure WITHIN the hierarchy.

Key questions:
1. Do tokens with sibling edges cluster together within mid-clusters?
2. Do tokens with cross-super edges (granite, fabric) show different patterns?
3. Is there sub-structure within each mid-cluster?
"""

import json
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from collections import defaultdict
from sklearn.manifold import MDS
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))
from models import HookedLLM
from run_hierarchy_and_semantic_experiments import HierarchicalGraph3Level


def analyze_edge_structure(graph):
    """Analyze the edge structure to understand token connectivity."""
    print("=" * 70)
    print("EDGE STRUCTURE ANALYSIS")
    print("=" * 70)

    # Token mapping
    tokens = graph.vocabulary
    print(f"\nTokens: {tokens}")

    # Edge structure from the deterministic graph
    # Within-mid edges: fully connected within each cluster (6 edges per cluster = 24 total)
    # Sibling edges: (0,4), (1,5), (8,12), (9,13)
    # Cross-super edge: (3,11)

    sibling_edges = [(0, 4), (1, 5), (8, 12), (9, 13)]
    cross_super_edge = (3, 11)

    print("\n--- Token Edge Roles ---")
    edge_roles = {}
    for i, token in enumerate(tokens):
        roles = []
        # Check sibling edges
        for e in sibling_edges:
            if i in e:
                other = e[1] if e[0] == i else e[0]
                roles.append(f"sibling→{tokens[other]}")
        # Check cross-super edge
        if i in cross_super_edge:
            other = cross_super_edge[1] if cross_super_edge[0] == i else cross_super_edge[0]
            roles.append(f"cross-super→{tokens[other]}")

        edge_roles[token] = roles
        mid_cluster = i // 4
        cluster_names = ["M_A1", "M_A2", "M_B1", "M_B2"]
        print(f"  {token:10s} ({cluster_names[mid_cluster]}): {roles if roles else ['within-cluster only']}")

    return edge_roles


def compute_within_cluster_distances(token_reps, graph, context_lengths):
    """Compute pairwise distances within each mid-cluster."""
    tokens = graph.vocabulary
    mid_clusters = {
        "M_A1": tokens[0:4],   # crystal, marble, diamond, granite
        "M_A2": tokens[4:8],   # lantern, castle, beacon, fortress
        "M_B1": tokens[8:12],  # cloud, canvas, mist, fabric
        "M_B2": tokens[12:16], # pillar, tunnel, column, passage
    }

    # For each context length, compute all pairwise distances within each cluster
    within_cluster_distances = {cluster: {} for cluster in mid_clusters}

    for ctx_len in context_lengths:
        reps = token_reps.get(ctx_len, {})
        if not reps:
            continue

        for cluster_name, cluster_tokens in mid_clusters.items():
            pair_dists = {}
            for i, t1 in enumerate(cluster_tokens):
                for t2 in cluster_tokens[i+1:]:
                    if t1 in reps and t2 in reps:
                        dist = np.linalg.norm(reps[t1] - reps[t2])
                        pair_dists[f"{t1}-{t2}"] = dist
            within_cluster_distances[cluster_name][ctx_len] = pair_dists

    return within_cluster_distances, mid_clusters


def collect_representations(model, tokenizer, graph, context_lengths, n_samples=50, layer_idx=-5):
    """Collect token representations at each context length."""
    token_reps = {ctx: {} for ctx in context_lengths}

    for ctx_len in context_lengths:
        print(f"  N={ctx_len}...", end=" ", flush=True)
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


def main():
    print("=" * 70)
    print("WITHIN-CLUSTER STRUCTURE ANALYSIS")
    print("=" * 70)

    # Create graph and analyze edge structure
    graph = HierarchicalGraph3Level(seed=42)
    edge_roles = analyze_edge_structure(graph)

    # Load model
    print("\nLoading model...")
    hooked_model = HookedLLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = hooked_model.model
    tokenizer = hooked_model.tokenizer
    model.eval()

    # Use a subset of context lengths for analysis
    context_lengths = [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000]

    print("\nCollecting representations...")
    token_reps = collect_representations(model, tokenizer, graph, context_lengths, n_samples=50, layer_idx=-5)

    # Compute within-cluster distances
    print("\nComputing within-cluster pairwise distances...")
    within_dists, mid_clusters = compute_within_cluster_distances(token_reps, graph, context_lengths)

    # Analyze patterns
    print("\n" + "=" * 70)
    print("WITHIN-CLUSTER DISTANCE ANALYSIS")
    print("=" * 70)

    # Define which token pairs have special edges
    sibling_pairs = [
        ("crystal", "lantern"),   # Different clusters but sibling edge
        ("marble", "castle"),     # Different clusters but sibling edge
        ("cloud", "pillar"),      # Different clusters but sibling edge
        ("canvas", "tunnel"),     # Different clusters but sibling edge
    ]
    cross_super_pair = ("granite", "fabric")

    # Tokens involved in special edges (within their own cluster)
    tokens_with_sibling = {"crystal", "marble", "lantern", "castle", "cloud", "canvas", "pillar", "tunnel"}
    tokens_with_cross_super = {"granite", "fabric"}
    tokens_no_special = {"diamond", "beacon", "mist", "column", "fortress", "passage"}

    # Create visualization
    output_dir = Path("results/hierarchy_mds_plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Within-cluster distances over context length
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("M_A1 (crystal, marble, diamond, granite)",
                       "M_A2 (lantern, castle, beacon, fortress)",
                       "M_B1 (cloud, canvas, mist, fabric)",
                       "M_B2 (pillar, tunnel, column, passage)"),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']

    positions = {"M_A1": (1, 1), "M_A2": (1, 2), "M_B1": (2, 1), "M_B2": (2, 2)}

    for cluster_name, cluster_tokens in mid_clusters.items():
        row, col = positions[cluster_name]

        # Get all pairs in this cluster
        pairs = []
        for i, t1 in enumerate(cluster_tokens):
            for t2 in cluster_tokens[i+1:]:
                pairs.append(f"{t1}-{t2}")

        for idx, pair in enumerate(pairs):
            x_vals = []
            y_vals = []
            for ctx_len in context_lengths:
                if ctx_len in within_dists[cluster_name]:
                    if pair in within_dists[cluster_name][ctx_len]:
                        x_vals.append(ctx_len)
                        y_vals.append(within_dists[cluster_name][ctx_len][pair])

            # Determine line style based on edge roles
            t1, t2 = pair.split("-")
            if t1 in tokens_with_cross_super or t2 in tokens_with_cross_super:
                dash = "dash"
                width = 3
            elif t1 in tokens_with_sibling or t2 in tokens_with_sibling:
                dash = "dot"
                width = 2
            else:
                dash = "solid"
                width = 2

            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                mode='lines+markers',
                name=pair,
                line=dict(color=colors[idx % len(colors)], width=width, dash=dash),
                marker=dict(size=8),
                showlegend=(cluster_name == "M_A1"),  # Only show legend for first cluster
                legendgroup=pair,
            ), row=row, col=col)

    fig.update_layout(
        title="Within-Cluster Pairwise Distances Over Context Length<br>" +
              "<sup>Dashed: involves cross-super edge token | Dotted: involves sibling edge token | Solid: within-cluster only</sup>",
        width=1200,
        height=900,
    )

    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(title_text="Context Length (N)", type="log", row=i, col=j)
            fig.update_yaxes(title_text="L2 Distance", row=i, col=j)

    fig.write_html(output_dir / "within_cluster_distances.html")
    print(f"\nSaved: {output_dir / 'within_cluster_distances.html'}")

    # Plot 2: Compare tokens with special edges vs without
    fig2 = go.Figure()

    # For each context length, compute average distance for:
    # 1. Pairs where BOTH tokens have no special edges
    # 2. Pairs where at least one token has a sibling edge
    # 3. Pairs involving cross-super tokens

    avg_no_special = []
    avg_with_sibling = []
    avg_with_cross = []

    for ctx_len in context_lengths:
        no_special_dists = []
        sibling_dists = []
        cross_dists = []

        for cluster_name in mid_clusters:
            if ctx_len not in within_dists[cluster_name]:
                continue
            for pair, dist in within_dists[cluster_name][ctx_len].items():
                t1, t2 = pair.split("-")
                if t1 in tokens_with_cross_super or t2 in tokens_with_cross_super:
                    cross_dists.append(dist)
                elif t1 in tokens_with_sibling or t2 in tokens_with_sibling:
                    sibling_dists.append(dist)
                else:
                    no_special_dists.append(dist)

        avg_no_special.append(np.mean(no_special_dists) if no_special_dists else np.nan)
        avg_with_sibling.append(np.mean(sibling_dists) if sibling_dists else np.nan)
        avg_with_cross.append(np.mean(cross_dists) if cross_dists else np.nan)

    fig2.add_trace(go.Scatter(
        x=context_lengths, y=avg_no_special,
        mode='lines+markers',
        name='No special edges (diamond-beacon, etc.)',
        line=dict(color='#2ca02c', width=3),
        marker=dict(size=10),
    ))

    fig2.add_trace(go.Scatter(
        x=context_lengths, y=avg_with_sibling,
        mode='lines+markers',
        name='Has sibling edge (crystal, marble, etc.)',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=10),
    ))

    fig2.add_trace(go.Scatter(
        x=context_lengths, y=avg_with_cross,
        mode='lines+markers',
        name='Has cross-super edge (granite, fabric)',
        line=dict(color='#d62728', width=3),
        marker=dict(size=10),
    ))

    fig2.update_layout(
        title="Within-Cluster Distance by Token Edge Type<br>" +
              "<sup>Do tokens with external edges (sibling/cross-super) cluster differently?</sup>",
        xaxis_title="Context Length (N)",
        yaxis_title="Average Within-Cluster L2 Distance",
        xaxis_type="log",
        width=900,
        height=600,
    )

    fig2.write_html(output_dir / "within_cluster_by_edge_type.html")
    print(f"Saved: {output_dir / 'within_cluster_by_edge_type.html'}")

    # Plot 3: Cross-super tokens (granite, fabric) - how do they differ from their cluster-mates?
    print("\n" + "=" * 70)
    print("CROSS-SUPER TOKEN ANALYSIS")
    print("=" * 70)

    # granite is in M_A1, fabric is in M_B1
    # They have an edge between them, so they should be more similar to each other
    # than to other tokens in different super-clusters

    granite_to_others = {}
    fabric_to_others = {}
    granite_to_fabric = {}

    for ctx_len in context_lengths:
        reps = token_reps.get(ctx_len, {})
        if "granite" not in reps or "fabric" not in reps:
            continue

        # granite to other M_A1 tokens
        granite_dists = []
        for t in ["crystal", "marble", "diamond"]:
            if t in reps:
                granite_dists.append(np.linalg.norm(reps["granite"] - reps[t]))
        granite_to_others[ctx_len] = np.mean(granite_dists) if granite_dists else np.nan

        # fabric to other M_B1 tokens
        fabric_dists = []
        for t in ["cloud", "canvas", "mist"]:
            if t in reps:
                fabric_dists.append(np.linalg.norm(reps["fabric"] - reps[t]))
        fabric_to_others[ctx_len] = np.mean(fabric_dists) if fabric_dists else np.nan

        # granite to fabric (cross-super edge)
        granite_to_fabric[ctx_len] = np.linalg.norm(reps["granite"] - reps["fabric"])

    fig3 = go.Figure()

    fig3.add_trace(go.Scatter(
        x=list(granite_to_others.keys()),
        y=list(granite_to_others.values()),
        mode='lines+markers',
        name='granite → other M_A1 (within cluster)',
        line=dict(color='#e41a1c', width=3),
    ))

    fig3.add_trace(go.Scatter(
        x=list(fabric_to_others.keys()),
        y=list(fabric_to_others.values()),
        mode='lines+markers',
        name='fabric → other M_B1 (within cluster)',
        line=dict(color='#377eb8', width=3),
    ))

    fig3.add_trace(go.Scatter(
        x=list(granite_to_fabric.keys()),
        y=list(granite_to_fabric.values()),
        mode='lines+markers',
        name='granite ↔ fabric (cross-super edge)',
        line=dict(color='#4daf4a', width=3, dash='dash'),
    ))

    fig3.update_layout(
        title="Cross-Super Edge Tokens: granite and fabric<br>" +
              "<sup>These tokens have an edge between them despite being in different super-clusters</sup>",
        xaxis_title="Context Length (N)",
        yaxis_title="L2 Distance",
        xaxis_type="log",
        width=900,
        height=600,
    )

    fig3.write_html(output_dir / "cross_super_edge_analysis.html")
    print(f"Saved: {output_dir / 'cross_super_edge_analysis.html'}")

    # Print numerical summary
    print("\n--- Numerical Summary at Key Context Lengths ---")
    for ctx_len in [100, 1000, 5000, 10000]:
        if ctx_len in granite_to_others:
            print(f"\nN={ctx_len}:")
            print(f"  granite → M_A1 cluster-mates: {granite_to_others[ctx_len]:.2f}")
            print(f"  fabric → M_B1 cluster-mates: {fabric_to_others[ctx_len]:.2f}")
            print(f"  granite ↔ fabric (cross-super): {granite_to_fabric[ctx_len]:.2f}")

            # Ratio: is granite-fabric closer than expected for cross-super?
            avg_cluster = (granite_to_others[ctx_len] + fabric_to_others[ctx_len]) / 2
            ratio = granite_to_fabric[ctx_len] / avg_cluster
            print(f"  Ratio (cross-super / within-cluster): {ratio:.2f}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
