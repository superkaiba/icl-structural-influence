#!/usr/bin/env python3
"""
Detailed analysis of within-cluster structure with finer granularity.
Focus on the phase transition where cross-super edge tokens become similar.
"""

import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))
from models import HookedLLM
from run_hierarchy_and_semantic_experiments import HierarchicalGraph3Level


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
    print("DETAILED WITHIN-CLUSTER STRUCTURE ANALYSIS")
    print("=" * 70)

    # Create graph
    graph = HierarchicalGraph3Level(seed=42)
    tokens = graph.vocabulary

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

    # Fine-grained context lengths focusing on the transition
    context_lengths = (
        [10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000] +
        list(range(1500, 10001, 500))  # 1500, 2000, 2500, ..., 10000
    )

    print(f"\nCollecting representations at {len(context_lengths)} context lengths...")
    token_reps = collect_representations(model, tokenizer, graph, context_lengths, n_samples=50, layer_idx=-5)

    output_dir = Path("results/hierarchy_mds_plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Analysis 1: Cross-super edge (granite↔fabric) vs within-cluster
    # =========================================================================
    print("\n--- Cross-Super Edge Analysis ---")

    granite_to_cluster = []
    fabric_to_cluster = []
    granite_to_fabric = []
    sibling_distances = []  # Average of sibling-connected tokens
    ctx_lens_used = []

    for ctx_len in context_lengths:
        reps = token_reps.get(ctx_len, {})
        if "granite" not in reps or "fabric" not in reps:
            continue

        ctx_lens_used.append(ctx_len)

        # granite to its M_A1 cluster-mates
        g_dists = [np.linalg.norm(reps["granite"] - reps[t]) for t in ["crystal", "marble", "diamond"] if t in reps]
        granite_to_cluster.append(np.mean(g_dists) if g_dists else np.nan)

        # fabric to its M_B1 cluster-mates
        f_dists = [np.linalg.norm(reps["fabric"] - reps[t]) for t in ["cloud", "canvas", "mist"] if t in reps]
        fabric_to_cluster.append(np.mean(f_dists) if f_dists else np.nan)

        # granite to fabric (cross-super edge)
        granite_to_fabric.append(np.linalg.norm(reps["granite"] - reps["fabric"]))

        # Sibling edge distances (within same super-cluster)
        sib_dists = []
        for t1, t2 in [("crystal", "lantern"), ("marble", "castle"), ("cloud", "pillar"), ("canvas", "tunnel")]:
            if t1 in reps and t2 in reps:
                sib_dists.append(np.linalg.norm(reps[t1] - reps[t2]))
        sibling_distances.append(np.mean(sib_dists) if sib_dists else np.nan)

    # Compute ratios
    avg_within = [(g + f) / 2 for g, f in zip(granite_to_cluster, fabric_to_cluster)]
    ratio_cross_to_within = [c / w if w > 0 else np.nan for c, w in zip(granite_to_fabric, avg_within)]

    # Create comprehensive figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Absolute Distances",
            "Cross-Super / Within-Cluster Ratio",
            "Sibling Edge vs Within-Cluster",
            "All Distance Types Normalized"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # Panel 1: Absolute distances
    fig.add_trace(go.Scatter(
        x=ctx_lens_used, y=granite_to_cluster,
        mode='lines+markers', name='granite → M_A1 (within)',
        line=dict(color='#e41a1c', width=2),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=ctx_lens_used, y=fabric_to_cluster,
        mode='lines+markers', name='fabric → M_B1 (within)',
        line=dict(color='#377eb8', width=2),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=ctx_lens_used, y=granite_to_fabric,
        mode='lines+markers', name='granite ↔ fabric (cross-super)',
        line=dict(color='#4daf4a', width=3, dash='dash'),
    ), row=1, col=1)

    # Panel 2: Ratio
    fig.add_trace(go.Scatter(
        x=ctx_lens_used, y=ratio_cross_to_within,
        mode='lines+markers', name='Ratio',
        line=dict(color='#984ea3', width=3),
        marker=dict(size=10),
    ), row=1, col=2)

    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=1, col=2)
    fig.add_annotation(
        x=np.log10(5000), y=1.0, text="Ratio = 1 (cross-super = within-cluster)",
        showarrow=False, yshift=15, row=1, col=2
    )

    # Panel 3: Sibling edge analysis
    # Compare sibling edge distances (between mid-clusters) to within-cluster distances
    fig.add_trace(go.Scatter(
        x=ctx_lens_used, y=sibling_distances,
        mode='lines+markers', name='Sibling edges (cross mid-cluster)',
        line=dict(color='#ff7f00', width=2),
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=ctx_lens_used, y=avg_within,
        mode='lines+markers', name='Within mid-cluster (avg)',
        line=dict(color='#a65628', width=2),
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=ctx_lens_used, y=granite_to_fabric,
        mode='lines+markers', name='Cross-super edge',
        line=dict(color='#4daf4a', width=2, dash='dash'),
    ), row=2, col=1)

    # Panel 4: Normalized view (divide by within-cluster)
    sibling_ratio = [s / w if w > 0 else np.nan for s, w in zip(sibling_distances, avg_within)]
    cross_ratio = ratio_cross_to_within

    fig.add_trace(go.Scatter(
        x=ctx_lens_used, y=sibling_ratio,
        mode='lines+markers', name='Sibling / Within',
        line=dict(color='#ff7f00', width=2),
    ), row=2, col=2)

    fig.add_trace(go.Scatter(
        x=ctx_lens_used, y=cross_ratio,
        mode='lines+markers', name='Cross-Super / Within',
        line=dict(color='#4daf4a', width=2),
    ), row=2, col=2)

    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=2, col=2)

    fig.update_layout(
        title="Within-Hierarchy Structure: Edge-Type Distance Analysis<br>" +
              "<sup>Does the cross-super edge (granite↔fabric) create sub-structure?</sup>",
        width=1200,
        height=900,
        legend=dict(x=1.02, y=0.5),
    )

    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(title_text="Context Length (N)", type="log", row=i, col=j)

    fig.update_yaxes(title_text="L2 Distance", row=1, col=1)
    fig.update_yaxes(title_text="Ratio", row=1, col=2)
    fig.update_yaxes(title_text="L2 Distance", row=2, col=1)
    fig.update_yaxes(title_text="Ratio (normalized)", row=2, col=2)

    fig.write_html(output_dir / "within_hierarchy_structure_detailed.html")
    print(f"\nSaved: {output_dir / 'within_hierarchy_structure_detailed.html'}")

    # =========================================================================
    # Analysis 2: Do tokens with similar edge roles cluster together?
    # =========================================================================
    print("\n--- Token Role Clustering Analysis ---")

    # Group tokens by their edge role
    # Role 0: No special edges (diamond, beacon, fortress, mist, column, passage)
    # Role 1: Sibling edge only (crystal, marble, lantern, castle, cloud, canvas, pillar, tunnel)
    # Role 2: Cross-super edge (granite, fabric)

    role_0_tokens = ["diamond", "beacon", "fortress", "mist", "column", "passage"]
    role_1_tokens = ["crystal", "marble", "lantern", "castle", "cloud", "canvas", "pillar", "tunnel"]
    role_2_tokens = ["granite", "fabric"]

    # Compute average within-role distances
    role_0_dists = []
    role_1_dists = []
    role_2_dists = []

    for ctx_len in context_lengths:
        reps = token_reps.get(ctx_len, {})

        # Role 0 distances
        r0_dists = []
        for i, t1 in enumerate(role_0_tokens):
            for t2 in role_0_tokens[i+1:]:
                if t1 in reps and t2 in reps:
                    r0_dists.append(np.linalg.norm(reps[t1] - reps[t2]))
        role_0_dists.append(np.mean(r0_dists) if r0_dists else np.nan)

        # Role 1 distances
        r1_dists = []
        for i, t1 in enumerate(role_1_tokens):
            for t2 in role_1_tokens[i+1:]:
                if t1 in reps and t2 in reps:
                    r1_dists.append(np.linalg.norm(reps[t1] - reps[t2]))
        role_1_dists.append(np.mean(r1_dists) if r1_dists else np.nan)

        # Role 2 distance (only 2 tokens)
        if "granite" in reps and "fabric" in reps:
            role_2_dists.append(np.linalg.norm(reps["granite"] - reps["fabric"]))
        else:
            role_2_dists.append(np.nan)

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=context_lengths, y=role_0_dists,
        mode='lines+markers',
        name='No special edges (6 tokens)',
        line=dict(color='#2ca02c', width=3),
    ))

    fig2.add_trace(go.Scatter(
        x=context_lengths, y=role_1_dists,
        mode='lines+markers',
        name='Sibling edges (8 tokens)',
        line=dict(color='#1f77b4', width=3),
    ))

    fig2.add_trace(go.Scatter(
        x=context_lengths, y=role_2_dists,
        mode='lines+markers',
        name='Cross-super edge (2 tokens)',
        line=dict(color='#d62728', width=3),
    ))

    fig2.update_layout(
        title="Distance Between Tokens with Same Edge Role<br>" +
              "<sup>Do tokens with similar connectivity patterns cluster together?</sup>",
        xaxis_title="Context Length (N)",
        yaxis_title="Average Pairwise L2 Distance",
        xaxis_type="log",
        width=900,
        height=600,
    )

    fig2.write_html(output_dir / "token_role_clustering.html")
    print(f"Saved: {output_dir / 'token_role_clustering.html'}")

    # =========================================================================
    # Print summary statistics
    # =========================================================================
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Find the context length where cross-super ratio is closest to 1
    min_ratio_idx = np.nanargmin([abs(r - 1) for r in ratio_cross_to_within])
    min_ratio_ctx = ctx_lens_used[min_ratio_idx]
    min_ratio_val = ratio_cross_to_within[min_ratio_idx]

    print(f"\n1. CROSS-SUPER EDGE CONVERGENCE:")
    print(f"   The cross-super tokens (granite, fabric) become closest to their")
    print(f"   within-cluster distance at N={min_ratio_ctx} (ratio={min_ratio_val:.3f})")

    # Find where cross-super becomes closer than within-cluster
    crossings = [i for i, r in enumerate(ratio_cross_to_within) if r < 1.0]
    if crossings:
        first_crossing = ctx_lens_used[crossings[0]]
        print(f"\n2. PHASE TRANSITION:")
        print(f"   Cross-super distance < within-cluster distance first occurs at N={first_crossing}")
        print(f"   This means granite↔fabric become more similar than their own cluster-mates!")

    # Compare sibling vs cross-super at late context
    late_idx = -1  # Last context length
    print(f"\n3. EDGE TYPE HIERARCHY at N={ctx_lens_used[late_idx]}:")
    print(f"   Within mid-cluster: {avg_within[late_idx]:.2f}")
    print(f"   Sibling edges (within super): {sibling_distances[late_idx]:.2f}")
    print(f"   Cross-super edge: {granite_to_fabric[late_idx]:.2f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
