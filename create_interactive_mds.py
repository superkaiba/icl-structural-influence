#!/usr/bin/env python3
"""
Create interactive MDS plots using Plotly.
Allows zooming, panning, and hovering for details.
"""

import json
import sys
from pathlib import Path
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.manifold import MDS
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent / "src"))
from models import HookedLLM
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


def main():
    print("=" * 70)
    print("INTERACTIVE MDS PLOTS (Plotly)")
    print("=" * 70)

    # Configuration
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

    # Create graph
    print("\nCreating 3-level hierarchical graph (16 tokens)...")
    graph = HierarchicalGraph3Level(seed=42)
    hierarchy = graph.get_hierarchy_labels()

    # Collect representations
    print("\nCollecting token representations...")
    token_reps = collect_representations(
        model, tokenizer, graph, context_lengths, n_samples, layer_idx
    )

    # Prepare data matrix
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

    data_matrix = np.array(all_reps)
    print(f"\nData matrix shape: {data_matrix.shape}")

    # Apply MDS
    print("\nApplying MDS...")
    mds = MDS(n_components=2, random_state=42, n_init=4, max_iter=300, normalized_stress='auto')
    embedded = mds.fit_transform(data_matrix)

    # Compute centroids
    mid_colors = {0: '#e41a1c', 1: '#ff7f00', 2: '#377eb8', 3: '#00bfff'}
    mid_names = {0: 'M_A1', 1: 'M_A2', 2: 'M_B1', 3: 'M_B2'}
    mid_tokens = {
        0: ['crystal', 'marble', 'diamond', 'granite'],
        1: ['lantern', 'castle', 'beacon', 'fortress'],
        2: ['cloud', 'canvas', 'mist', 'fabric'],
        3: ['pillar', 'tunnel', 'column', 'passage']
    }

    super_colors = {0: '#e41a1c', 1: '#377eb8'}
    super_names = {0: 'Super_A', 1: 'Super_B'}

    # Compute mid-cluster centroids
    mid_centroids = {c: {} for c in range(4)}
    for ctx_len in context_lengths:
        cluster_points = defaultdict(list)
        for i, (token_text, ctx) in enumerate(all_labels):
            if ctx == ctx_len:
                mid_cluster = hierarchy["level2"][graph.token_to_node[token_text]]
                cluster_points[mid_cluster].append(embedded[i])
        for cluster_id in range(4):
            if cluster_points[cluster_id]:
                mid_centroids[cluster_id][ctx_len] = np.mean(cluster_points[cluster_id], axis=0)

    # Compute super-cluster centroids
    super_centroids = {c: {} for c in range(2)}
    for ctx_len in context_lengths:
        cluster_points = defaultdict(list)
        for i, (token_text, ctx) in enumerate(all_labels):
            if ctx == ctx_len:
                super_cluster = hierarchy["level1"][graph.token_to_node[token_text]]
                cluster_points[super_cluster].append(embedded[i])
        for cluster_id in range(2):
            if cluster_points[cluster_id]:
                super_centroids[cluster_id][ctx_len] = np.mean(cluster_points[cluster_id], axis=0)

    # =========================================================================
    # Interactive Plot 1: Mid-cluster Evolution
    # =========================================================================
    print("\nCreating interactive mid-cluster evolution plot...")

    fig1 = go.Figure()

    for cluster_id in range(4):
        ctxs = sorted(mid_centroids[cluster_id].keys())
        points = np.array([mid_centroids[cluster_id][c] for c in ctxs])

        # Trajectory line
        fig1.add_trace(go.Scatter(
            x=points[:, 0], y=points[:, 1],
            mode='lines',
            line=dict(color=mid_colors[cluster_id], width=3),
            name=f'{mid_names[cluster_id]} trajectory',
            legendgroup=mid_names[cluster_id],
            hoverinfo='skip'
        ))

        # Points with size by context
        sizes = [10 + (ctx / max(context_lengths)) * 40 for ctx in ctxs]
        hover_text = [f"{mid_names[cluster_id]}<br>N={ctx}<br>Tokens: {', '.join(mid_tokens[cluster_id])}"
                     for ctx in ctxs]

        fig1.add_trace(go.Scatter(
            x=points[:, 0], y=points[:, 1],
            mode='markers+text',
            marker=dict(
                size=sizes,
                color=mid_colors[cluster_id],
                line=dict(color='black', width=1)
            ),
            text=[str(ctx) for ctx in ctxs],
            textposition='middle center',
            textfont=dict(size=8, color='white'),
            name=f'{mid_names[cluster_id]} points',
            legendgroup=mid_names[cluster_id],
            hovertext=hover_text,
            hoverinfo='text'
        ))

    fig1.update_layout(
        title=dict(
            text='Mid-Cluster Centroid Evolution (Interactive)<br><sup>Scroll to zoom, drag to pan, hover for details</sup>',
            x=0.5
        ),
        xaxis_title='MDS Dimension 1',
        yaxis_title='MDS Dimension 2',
        hovermode='closest',
        legend=dict(x=1.02, y=1),
        width=1000,
        height=800
    )

    fig1.write_html(output_dir / "interactive_midcluster_evolution.html")
    print(f"  Saved: {output_dir / 'interactive_midcluster_evolution.html'}")

    # =========================================================================
    # Interactive Plot 2: Super-cluster Evolution
    # =========================================================================
    print("Creating interactive super-cluster evolution plot...")

    fig2 = go.Figure()

    for cluster_id in [0, 1]:
        ctxs = sorted(super_centroids[cluster_id].keys())
        points = np.array([super_centroids[cluster_id][c] for c in ctxs])

        # Trajectory line
        fig2.add_trace(go.Scatter(
            x=points[:, 0], y=points[:, 1],
            mode='lines',
            line=dict(color=super_colors[cluster_id], width=4),
            name=f'{super_names[cluster_id]} trajectory',
            legendgroup=super_names[cluster_id],
            hoverinfo='skip'
        ))

        # Points
        sizes = [12 + (ctx / max(context_lengths)) * 45 for ctx in ctxs]
        mid_clusters_in_super = ['M_A1, M_A2'] if cluster_id == 0 else ['M_B1, M_B2']
        hover_text = [f"{super_names[cluster_id]}<br>N={ctx}<br>Contains: {mid_clusters_in_super[0]}"
                     for ctx in ctxs]

        fig2.add_trace(go.Scatter(
            x=points[:, 0], y=points[:, 1],
            mode='markers+text',
            marker=dict(
                size=sizes,
                color=super_colors[cluster_id],
                line=dict(color='black', width=2)
            ),
            text=[str(ctx) for ctx in ctxs],
            textposition='middle center',
            textfont=dict(size=9, color='white'),
            name=f'{super_names[cluster_id]} points',
            legendgroup=super_names[cluster_id],
            hovertext=hover_text,
            hoverinfo='text'
        ))

    fig2.update_layout(
        title=dict(
            text='Super-Cluster Centroid Evolution (Interactive)<br><sup>Scroll to zoom, drag to pan, hover for details</sup>',
            x=0.5
        ),
        xaxis_title='MDS Dimension 1',
        yaxis_title='MDS Dimension 2',
        hovermode='closest',
        legend=dict(x=1.02, y=1),
        width=1000,
        height=800
    )

    fig2.write_html(output_dir / "interactive_supercluster_evolution.html")
    print(f"  Saved: {output_dir / 'interactive_supercluster_evolution.html'}")

    # =========================================================================
    # Interactive Plot 3: All Individual Tokens
    # =========================================================================
    print("Creating interactive all-tokens plot...")

    fig3 = go.Figure()

    # Group by token
    token_trajectories = defaultdict(list)
    for i, (token_text, ctx_len) in enumerate(all_labels):
        token_trajectories[token_text].append((ctx_len, embedded[i]))

    for token_text in sorted(token_trajectories.keys()):
        traj = sorted(token_trajectories[token_text], key=lambda x: x[0])
        ctxs = [t[0] for t in traj]
        points = np.array([t[1] for t in traj])

        mid_cluster = hierarchy["level2"][graph.token_to_node[token_text]]
        super_cluster = hierarchy["level1"][graph.token_to_node[token_text]]
        color = mid_colors[mid_cluster]

        # Trajectory line
        fig3.add_trace(go.Scatter(
            x=points[:, 0], y=points[:, 1],
            mode='lines',
            line=dict(color=color, width=1.5),
            name=token_text,
            legendgroup=mid_names[mid_cluster],
            showlegend=False,
            hoverinfo='skip',
            opacity=0.5
        ))

        # Points
        sizes = [5 + (ctx / max(context_lengths)) * 15 for ctx in ctxs]
        hover_text = [f"Token: {token_text}<br>N={ctx}<br>Cluster: {mid_names[mid_cluster]}<br>Super: {super_names[super_cluster]}"
                     for ctx in ctxs]

        fig3.add_trace(go.Scatter(
            x=points[:, 0], y=points[:, 1],
            mode='markers',
            marker=dict(
                size=sizes,
                color=color,
                line=dict(color='white', width=0.5)
            ),
            name=token_text,
            legendgroup=mid_names[mid_cluster],
            showlegend=True,
            hovertext=hover_text,
            hoverinfo='text'
        ))

    fig3.update_layout(
        title=dict(
            text='All Token Trajectories (Interactive)<br><sup>16 tokens x 13 context lengths | Scroll to zoom, hover for details</sup>',
            x=0.5
        ),
        xaxis_title='MDS Dimension 1',
        yaxis_title='MDS Dimension 2',
        hovermode='closest',
        legend=dict(x=1.02, y=1, font=dict(size=9)),
        width=1200,
        height=900
    )

    fig3.write_html(output_dir / "interactive_all_tokens.html")
    print(f"  Saved: {output_dir / 'interactive_all_tokens.html'}")

    # =========================================================================
    # Interactive Plot 4: Combined with Dropdown
    # =========================================================================
    print("Creating combined interactive plot with view selector...")

    fig4 = go.Figure()

    # Add all mid-cluster traces (visible by default)
    for cluster_id in range(4):
        ctxs = sorted(mid_centroids[cluster_id].keys())
        points = np.array([mid_centroids[cluster_id][c] for c in ctxs])
        sizes = [10 + (ctx / max(context_lengths)) * 40 for ctx in ctxs]

        fig4.add_trace(go.Scatter(
            x=points[:, 0], y=points[:, 1],
            mode='lines+markers+text',
            line=dict(color=mid_colors[cluster_id], width=3),
            marker=dict(size=sizes, color=mid_colors[cluster_id], line=dict(color='black', width=1)),
            text=[str(ctx) for ctx in ctxs],
            textposition='middle center',
            textfont=dict(size=8, color='white'),
            name=f'{mid_names[cluster_id]}',
            visible=True
        ))

    # Add super-cluster traces (hidden by default)
    for cluster_id in [0, 1]:
        ctxs = sorted(super_centroids[cluster_id].keys())
        points = np.array([super_centroids[cluster_id][c] for c in ctxs])
        sizes = [12 + (ctx / max(context_lengths)) * 45 for ctx in ctxs]

        fig4.add_trace(go.Scatter(
            x=points[:, 0], y=points[:, 1],
            mode='lines+markers+text',
            line=dict(color=super_colors[cluster_id], width=4),
            marker=dict(size=sizes, color=super_colors[cluster_id], line=dict(color='black', width=2)),
            text=[str(ctx) for ctx in ctxs],
            textposition='middle center',
            textfont=dict(size=9, color='white'),
            name=f'{super_names[cluster_id]}',
            visible=False
        ))

    # Add dropdown menu
    fig4.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=[
                    dict(
                        label="Mid-Clusters (4)",
                        method="update",
                        args=[{"visible": [True, True, True, True, False, False]},
                              {"title": "Mid-Cluster Evolution (M_A1, M_A2, M_B1, M_B2)"}]
                    ),
                    dict(
                        label="Super-Clusters (2)",
                        method="update",
                        args=[{"visible": [False, False, False, False, True, True]},
                              {"title": "Super-Cluster Evolution (Super_A, Super_B)"}]
                    ),
                    dict(
                        label="All Clusters",
                        method="update",
                        args=[{"visible": [True, True, True, True, True, True]},
                              {"title": "All Clusters: Mid + Super"}]
                    ),
                ],
                direction="down",
                showactive=True,
                x=0.02,
                xanchor="left",
                y=1.15,
                yanchor="top"
            )
        ],
        title=dict(
            text='Cluster Evolution (Interactive)<br><sup>Use dropdown to switch views | Scroll to zoom</sup>',
            x=0.5
        ),
        xaxis_title='MDS Dimension 1',
        yaxis_title='MDS Dimension 2',
        hovermode='closest',
        width=1000,
        height=800
    )

    fig4.write_html(output_dir / "interactive_combined.html")
    print(f"  Saved: {output_dir / 'interactive_combined.html'}")

    # =========================================================================
    # Interactive Plot 5: Individual tokens + centroids overlay
    # =========================================================================
    print("Creating interactive plot with individual tokens + centroids...")

    fig5 = go.Figure()

    # First add individual token trajectories (smaller, semi-transparent)
    for token_text in sorted(token_trajectories.keys()):
        traj = sorted(token_trajectories[token_text], key=lambda x: x[0])
        ctxs = [t[0] for t in traj]
        points = np.array([t[1] for t in traj])

        mid_cluster = hierarchy["level2"][graph.token_to_node[token_text]]
        color = mid_colors[mid_cluster]

        # Individual token trajectory
        fig5.add_trace(go.Scatter(
            x=points[:, 0], y=points[:, 1],
            mode='lines+markers',
            line=dict(color=color, width=1),
            marker=dict(size=6, color=color, opacity=0.5),
            name=f'{token_text}',
            legendgroup=f'tokens_{mid_names[mid_cluster]}',
            legendgrouptitle_text=f'{mid_names[mid_cluster]} tokens',
            hovertext=[f"Token: {token_text}<br>N={ctx}<br>Cluster: {mid_names[mid_cluster]}" for ctx in ctxs],
            hoverinfo='text',
            opacity=0.6
        ))

    # Then add cluster centroids (larger, bold)
    for cluster_id in range(4):
        ctxs = sorted(mid_centroids[cluster_id].keys())
        points = np.array([mid_centroids[cluster_id][c] for c in ctxs])
        sizes = [15 + (ctx / max(context_lengths)) * 35 for ctx in ctxs]

        # Centroid trajectory (thick line)
        fig5.add_trace(go.Scatter(
            x=points[:, 0], y=points[:, 1],
            mode='lines',
            line=dict(color=mid_colors[cluster_id], width=5),
            name=f'{mid_names[cluster_id]} CENTROID',
            legendgroup=f'centroid_{mid_names[cluster_id]}',
            hoverinfo='skip',
            opacity=0.9
        ))

        # Centroid points (large with labels)
        fig5.add_trace(go.Scatter(
            x=points[:, 0], y=points[:, 1],
            mode='markers+text',
            marker=dict(
                size=sizes,
                color=mid_colors[cluster_id],
                line=dict(color='black', width=2),
                symbol='diamond'
            ),
            text=[str(ctx) for ctx in ctxs],
            textposition='middle center',
            textfont=dict(size=8, color='white'),
            name=f'{mid_names[cluster_id]} centroid points',
            legendgroup=f'centroid_{mid_names[cluster_id]}',
            hovertext=[f"CENTROID: {mid_names[cluster_id]}<br>N={ctx}<br>Mean of 4 tokens:<br>{', '.join(mid_tokens[cluster_id])}" for ctx in ctxs],
            hoverinfo='text'
        ))

    fig5.update_layout(
        title=dict(
            text='Individual Tokens + Cluster Centroids<br><sup>Small circles = individual tokens | Large diamonds = cluster centroids (mean)</sup>',
            x=0.5
        ),
        xaxis_title='MDS Dimension 1',
        yaxis_title='MDS Dimension 2',
        hovermode='closest',
        legend=dict(x=1.02, y=1, font=dict(size=9)),
        width=1200,
        height=900
    )

    fig5.write_html(output_dir / "interactive_tokens_and_centroids.html")
    print(f"  Saved: {output_dir / 'interactive_tokens_and_centroids.html'}")

    # =========================================================================
    # Interactive Plot 6: Focus on single context length (slider)
    # =========================================================================
    print("Creating interactive plot with context length slider...")

    fig6 = go.Figure()

    # Create frames for each context length
    frames = []
    for ctx_len in context_lengths:
        frame_data = []

        # Individual tokens at this context length
        for token_text in sorted(token_trajectories.keys()):
            traj_dict = {t[0]: t[1] for t in token_trajectories[token_text]}
            if ctx_len in traj_dict:
                point = traj_dict[ctx_len]
                mid_cluster = hierarchy["level2"][graph.token_to_node[token_text]]

                frame_data.append(go.Scatter(
                    x=[point[0]], y=[point[1]],
                    mode='markers+text',
                    marker=dict(size=15, color=mid_colors[mid_cluster], line=dict(color='black', width=1)),
                    text=[token_text],
                    textposition='top center',
                    textfont=dict(size=10),
                    name=token_text,
                    hovertext=f"{token_text}<br>Cluster: {mid_names[mid_cluster]}",
                    hoverinfo='text'
                ))

        frames.append(go.Frame(data=frame_data, name=str(ctx_len)))

    # Initial data (first context length)
    ctx_len = context_lengths[0]
    for token_text in sorted(token_trajectories.keys()):
        traj_dict = {t[0]: t[1] for t in token_trajectories[token_text]}
        if ctx_len in traj_dict:
            point = traj_dict[ctx_len]
            mid_cluster = hierarchy["level2"][graph.token_to_node[token_text]]

            fig6.add_trace(go.Scatter(
                x=[point[0]], y=[point[1]],
                mode='markers+text',
                marker=dict(size=15, color=mid_colors[mid_cluster], line=dict(color='black', width=1)),
                text=[token_text],
                textposition='top center',
                textfont=dict(size=10),
                name=token_text,
                hovertext=f"{token_text}<br>Cluster: {mid_names[mid_cluster]}",
                hoverinfo='text'
            ))

    fig6.frames = frames

    # Get axis range from all data
    all_x = [embedded[i, 0] for i in range(len(embedded))]
    all_y = [embedded[i, 1] for i in range(len(embedded))]
    x_range = [min(all_x) - 5, max(all_x) + 5]
    y_range = [min(all_y) - 5, max(all_y) + 5]

    # Add slider
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Context Length N=", "font": {"size": 16}},
        pad={"t": 50},
        steps=[dict(
            method="animate",
            args=[[str(ctx)], {"frame": {"duration": 300, "redraw": True}, "mode": "immediate"}],
            label=str(ctx)
        ) for ctx in context_lengths]
    )]

    fig6.update_layout(
        title=dict(
            text='Token Positions at Each Context Length<br><sup>Use slider to change N | Watch how tokens separate into clusters</sup>',
            x=0.5
        ),
        xaxis=dict(title='MDS Dimension 1', range=x_range),
        yaxis=dict(title='MDS Dimension 2', range=y_range),
        sliders=sliders,
        showlegend=False,
        width=1000,
        height=800,
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=1.15,
            x=0.1,
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}]),
                dict(label="Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
            ]
        )]
    )

    fig6.write_html(output_dir / "interactive_slider.html")
    print(f"  Saved: {output_dir / 'interactive_slider.html'}")

    print("\n" + "=" * 70)
    print("INTERACTIVE PLOTS COMPLETE")
    print("=" * 70)
    print(f"\nSaved to: {output_dir}")
    print("  - interactive_midcluster_evolution.html  (centroid trajectories)")
    print("  - interactive_supercluster_evolution.html (centroid trajectories)")
    print("  - interactive_all_tokens.html (individual token trajectories)")
    print("  - interactive_combined.html (dropdown selector)")
    print("  - interactive_tokens_and_centroids.html (both individual + centroids)")
    print("  - interactive_slider.html (animated slider through context lengths)")
    print("\nOpen these HTML files in a browser to interact with the plots!")


if __name__ == "__main__":
    main()
