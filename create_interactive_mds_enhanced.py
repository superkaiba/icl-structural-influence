#!/usr/bin/env python3
"""
Enhanced interactive MDS plots with:
1. Fine-grained context length sampling
2. Perplexity tracking per token
3. Influence metrics (hierarchy distance ratios)
4. Multi-panel interactive slider plot
"""

import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent / "src"))
from models import HookedLLM
from run_hierarchy_and_semantic_experiments import HierarchicalGraph3Level


def collect_representations_with_perplexity(model, tokenizer, graph, context_lengths,
                                            n_samples=50, layer_idx=-5):
    """
    Collect token representations AND perplexity at each context length.

    Returns:
        token_reps: {ctx_len: {token: mean_representation}}
        token_perplexity: {ctx_len: {token: mean_perplexity}}
    """
    token_reps = {ctx: {} for ctx in context_lengths}
    token_perplexity = {ctx: {} for ctx in context_lengths}

    # Get token IDs for vocabulary
    vocab_token_ids = {}
    for word in graph.vocabulary:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if len(ids) == 1:
            vocab_token_ids[word] = ids[0]
        else:
            # Multi-token word, use first token
            vocab_token_ids[word] = ids[0]

    for ctx_len in context_lengths:
        print(f"    N={ctx_len}...", end=" ", flush=True)
        token_representations = defaultdict(list)
        token_nll = defaultdict(list)  # negative log likelihood

        for _ in range(n_samples):
            prompt, node_sequence = graph.generate_random_walk(length=ctx_len, return_nodes=True)
            tokens = tokenizer.encode(prompt, add_special_tokens=False)
            input_ids = torch.tensor([tokens]).to(model.device)

            with torch.no_grad():
                outputs = model(input_ids, output_hidden_states=True)
                hidden_states = outputs.hidden_states[layer_idx][0]
                logits = outputs.logits[0]  # [seq_len, vocab_size]

            token_texts = prompt.split()
            for pos, (node, token_text) in enumerate(zip(node_sequence, token_texts)):
                if pos < hidden_states.shape[0]:
                    # Representation
                    rep = hidden_states[pos].cpu().float().numpy()
                    token_representations[token_text].append(rep)

                    # Perplexity: compute NLL for predicting this token from previous position
                    if pos > 0 and pos - 1 < logits.shape[0]:
                        # Get logits from position pos-1 (predicting pos)
                        pred_logits = logits[pos - 1]
                        target_id = tokens[pos] if pos < len(tokens) else vocab_token_ids.get(token_text, 0)

                        # Compute negative log likelihood
                        log_probs = F.log_softmax(pred_logits, dim=-1)
                        nll = -log_probs[target_id].cpu().float().item()
                        token_nll[token_text].append(nll)

        # Average representations
        for token_text, reps in token_representations.items():
            if reps:
                token_reps[ctx_len][token_text] = np.mean(reps, axis=0)

        # Average perplexity (convert NLL to perplexity = exp(NLL))
        for token_text, nlls in token_nll.items():
            if nlls:
                mean_nll = np.mean(nlls)
                token_perplexity[ctx_len][token_text] = np.exp(mean_nll)

        print(f"({len(token_reps[ctx_len])} tokens, {len(token_perplexity[ctx_len])} with PPL)")

    return token_reps, token_perplexity


def compute_influence_metrics(token_reps, graph, context_lengths):
    """
    Compute influence metrics at each context length:
    - Within-mid distance (same mid-cluster)
    - Within-super distance (same super, diff mid)
    - Across-super distance (different super)
    - Ratios indicating hierarchy emergence
    """
    hierarchy = graph.get_hierarchy_labels()
    level1 = hierarchy["level1"]
    level2 = hierarchy["level2"]

    metrics = {
        "within_mid": [],
        "within_super": [],
        "across_super": [],
        "ratio_super_to_mid": [],      # across_super / within_mid
        "ratio_super_to_sibling": [],  # across_super / within_super
        "context_lengths": context_lengths,
    }

    for ctx_len in context_lengths:
        reps = token_reps[ctx_len]
        tokens = list(reps.keys())

        within_mid_dists = []
        within_super_dists = []
        across_super_dists = []

        for i, t1 in enumerate(tokens):
            for t2 in tokens[i+1:]:
                n1 = graph.token_to_node[t1]
                n2 = graph.token_to_node[t2]

                dist = np.linalg.norm(reps[t1] - reps[t2])

                if level2[n1] == level2[n2]:
                    within_mid_dists.append(dist)
                elif level1[n1] == level1[n2]:
                    within_super_dists.append(dist)
                else:
                    across_super_dists.append(dist)

        w_mid = np.mean(within_mid_dists) if within_mid_dists else 0
        w_super = np.mean(within_super_dists) if within_super_dists else 0
        a_super = np.mean(across_super_dists) if across_super_dists else 0

        metrics["within_mid"].append(w_mid)
        metrics["within_super"].append(w_super)
        metrics["across_super"].append(a_super)
        metrics["ratio_super_to_mid"].append(a_super / w_mid if w_mid > 0 else 0)
        metrics["ratio_super_to_sibling"].append(a_super / w_super if w_super > 0 else 0)

    return metrics


def compute_cluster_perplexity(token_perplexity, graph, context_lengths):
    """Compute average perplexity per cluster at each context length."""
    hierarchy = graph.get_hierarchy_labels()

    mid_perplexity = {c: [] for c in range(4)}
    super_perplexity = {c: [] for c in range(2)}

    for ctx_len in context_lengths:
        ppl = token_perplexity.get(ctx_len, {})

        # Mid-cluster perplexity
        mid_ppls = defaultdict(list)
        super_ppls = defaultdict(list)

        for token, p in ppl.items():
            if token in graph.token_to_node:
                node = graph.token_to_node[token]
                mid_cluster = hierarchy["level2"][node]
                super_cluster = hierarchy["level1"][node]
                mid_ppls[mid_cluster].append(p)
                super_ppls[super_cluster].append(p)

        for c in range(4):
            mid_perplexity[c].append(np.mean(mid_ppls[c]) if mid_ppls[c] else 0)
        for c in range(2):
            super_perplexity[c].append(np.mean(super_ppls[c]) if super_ppls[c] else 0)

    return mid_perplexity, super_perplexity


def compute_clustering_metrics(embedded, all_labels, graph, context_lengths):
    """
    Compute clustering quality metrics at each context length.

    Metrics computed:
    - Silhouette Score (mid-cluster): How well tokens cluster by mid-cluster
    - Silhouette Score (super-cluster): How well tokens cluster by super-cluster
    - Calinski-Harabasz Index: Ratio of between/within cluster variance
    - Davies-Bouldin Index: Average cluster similarity (lower = better)

    All metrics are computed on the 2D MDS embedding.
    """
    hierarchy = graph.get_hierarchy_labels()

    metrics = {
        "silhouette_mid": [],
        "silhouette_super": [],
        "calinski_harabasz": [],
        "davies_bouldin": [],
        "context_lengths": context_lengths,
    }

    for ctx_len in context_lengths:
        # Get points and labels for this context length
        points = []
        mid_labels = []
        super_labels = []

        for i, (token_text, ctx) in enumerate(all_labels):
            if ctx == ctx_len:
                points.append(embedded[i])
                node = graph.token_to_node[token_text]
                mid_labels.append(hierarchy["level2"][node])
                super_labels.append(hierarchy["level1"][node])

        if len(points) < 4:  # Need enough points for meaningful metrics
            metrics["silhouette_mid"].append(0)
            metrics["silhouette_super"].append(0)
            metrics["calinski_harabasz"].append(0)
            metrics["davies_bouldin"].append(0)
            continue

        X = np.array(points)
        mid_labels = np.array(mid_labels)
        super_labels = np.array(super_labels)

        # Silhouette scores (need at least 2 clusters with 2+ points each)
        try:
            sil_mid = silhouette_score(X, mid_labels)
        except:
            sil_mid = 0

        try:
            sil_super = silhouette_score(X, super_labels)
        except:
            sil_super = 0

        # Calinski-Harabasz (use mid-cluster labels)
        try:
            ch = calinski_harabasz_score(X, mid_labels)
        except:
            ch = 0

        # Davies-Bouldin (use mid-cluster labels)
        try:
            db = davies_bouldin_score(X, mid_labels)
        except:
            db = float('inf')

        metrics["silhouette_mid"].append(sil_mid)
        metrics["silhouette_super"].append(sil_super)
        metrics["calinski_harabasz"].append(ch)
        metrics["davies_bouldin"].append(db)

    return metrics


def main():
    print("=" * 70)
    print("ENHANCED INTERACTIVE MDS PLOTS")
    print("With Perplexity Tracking and Influence Metrics")
    print("=" * 70)

    # Configuration - FINE-GRAINED context lengths up to 10,000
    model_name = "meta-llama/Llama-3.1-8B"
    # Fine-grained at start, then jumps of 100 from 1000 onwards
    context_lengths = (
        [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500, 750] +
        list(range(1000, 10001, 100))  # 1000, 1100, 1200, ..., 10000
    )
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

    # Color schemes
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

    # Collect representations WITH perplexity
    print("\nCollecting token representations and perplexity...")
    token_reps, token_perplexity = collect_representations_with_perplexity(
        model, tokenizer, graph, context_lengths, n_samples, layer_idx
    )

    # Compute influence metrics
    print("\nComputing influence metrics...")
    influence_metrics = compute_influence_metrics(token_reps, graph, context_lengths)

    # Compute cluster perplexity
    print("Computing cluster perplexity...")
    mid_perplexity, super_perplexity = compute_cluster_perplexity(
        token_perplexity, graph, context_lengths
    )

    # Prepare data matrix for MDS
    print("\nPreparing data for MDS...")
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
    print(f"Data matrix shape: {data_matrix.shape}")

    # Apply MDS
    print("\nApplying MDS...")
    mds = MDS(n_components=2, random_state=42, n_init=4, max_iter=300, normalized_stress='auto')
    embedded = mds.fit_transform(data_matrix)

    # Build token trajectories
    token_trajectories = defaultdict(list)
    for i, (token_text, ctx_len) in enumerate(all_labels):
        token_trajectories[token_text].append((ctx_len, embedded[i]))

    # Compute centroids
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

    # Get axis ranges
    all_x = [embedded[i, 0] for i in range(len(embedded))]
    all_y = [embedded[i, 1] for i in range(len(embedded))]
    x_range = [min(all_x) - 5, max(all_x) + 5]
    y_range = [min(all_y) - 5, max(all_y) + 5]

    # Compute clustering quality metrics
    print("Computing clustering quality metrics...")
    clustering_metrics = compute_clustering_metrics(embedded, all_labels, graph, context_lengths)

    # =========================================================================
    # Interactive Plot: Multi-panel slider with MDS, Perplexity, Influence
    # =========================================================================
    print("\nCreating enhanced interactive slider plot...")

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Token Positions (MDS)',
            'Perplexity by Cluster',
            'Hierarchy Distance Ratios',
            'Distance by Level'
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # IMPORTANT: Fixed token order for consistent trace count across frames
    sorted_tokens = sorted(graph.vocabulary)  # Always 16 tokens

    # Create frames for each context length
    frames = []

    for frame_idx, ctx_len in enumerate(context_lengths):
        frame_data = []

        # =================================================================
        # Panel 1: MDS Token Positions (row=1, col=1)
        # FIXED: Always create 16 traces (one per token), use NaN for missing
        # =================================================================
        for token_text in sorted_tokens:
            traj_dict = {t[0]: t[1] for t in token_trajectories[token_text]}
            mid_cluster = hierarchy["level2"][graph.token_to_node[token_text]]
            ppl = token_perplexity.get(ctx_len, {}).get(token_text, 0)

            if ctx_len in traj_dict:
                point = traj_dict[ctx_len]
                x_val, y_val = [point[0]], [point[1]]
                text_val = [token_text]
            else:
                # Token not present at this context length - use NaN (invisible)
                x_val, y_val = [None], [None]
                text_val = [""]

            frame_data.append(go.Scatter(
                x=x_val, y=y_val,
                mode='markers+text',
                marker=dict(size=12, color=mid_colors[mid_cluster],
                           line=dict(color='black', width=1)),
                text=text_val,
                textposition='top center',
                textfont=dict(size=9),
                name=token_text,
                hovertext=f"{token_text}<br>Cluster: {mid_names[mid_cluster]}<br>PPL: {ppl:.2f}",
                hoverinfo='text',
                showlegend=False
            ))

        # =================================================================
        # Panel 2: Perplexity by Cluster (row=1, col=2) - LINE PLOT
        # =================================================================
        # Line plot showing perplexity over context length for each cluster
        ctx_subset = context_lengths[:frame_idx+1]

        for c in range(4):
            ppl_values = mid_perplexity[c][:frame_idx+1]
            frame_data.append(go.Scatter(
                x=ctx_subset,
                y=ppl_values,
                mode='lines+markers',
                line=dict(color=mid_colors[c], width=2),
                marker=dict(size=6, color=mid_colors[c]),
                name=mid_names[c],
                hovertext=[f"{mid_names[c]}<br>N={ctx}<br>PPL: {p:.2f}" for ctx, p in zip(ctx_subset, ppl_values)],
                hoverinfo='text',
                showlegend=False
            ))

        # =================================================================
        # Panel 3: Hierarchy Distance Ratios (row=2, col=1)
        # =================================================================
        # Show ratios up to current context length
        ctx_subset = context_lengths[:frame_idx+1]
        ratio_super_mid = influence_metrics["ratio_super_to_mid"][:frame_idx+1]
        ratio_super_sib = influence_metrics["ratio_super_to_sibling"][:frame_idx+1]

        frame_data.append(go.Scatter(
            x=ctx_subset, y=ratio_super_mid,
            mode='lines+markers',
            line=dict(color='#d62728', width=2),
            marker=dict(size=8, color='#d62728'),
            name='Cross-Super / Within-Mid',
            hovertext=[f"N={ctx}<br>Ratio: {r:.3f}" for ctx, r in zip(ctx_subset, ratio_super_mid)],
            hoverinfo='text',
            showlegend=False
        ))

        frame_data.append(go.Scatter(
            x=ctx_subset, y=ratio_super_sib,
            mode='lines+markers',
            line=dict(color='#9467bd', width=2),
            marker=dict(size=8, color='#9467bd'),
            name='Cross-Super / Within-Super',
            hovertext=[f"N={ctx}<br>Ratio: {r:.3f}" for ctx, r in zip(ctx_subset, ratio_super_sib)],
            hoverinfo='text',
            showlegend=False
        ))

        # Reference line at 1.0
        frame_data.append(go.Scatter(
            x=[context_lengths[0], context_lengths[-1]], y=[1, 1],
            mode='lines',
            line=dict(color='gray', width=1, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ))

        # =================================================================
        # Panel 4: Raw Distances (row=2, col=2)
        # =================================================================
        dist_within_mid = influence_metrics["within_mid"][:frame_idx+1]
        dist_within_super = influence_metrics["within_super"][:frame_idx+1]
        dist_across_super = influence_metrics["across_super"][:frame_idx+1]

        frame_data.append(go.Scatter(
            x=ctx_subset, y=dist_within_mid,
            mode='lines+markers',
            line=dict(color='green', width=2),
            marker=dict(size=6, color='green'),
            name='Within Mid',
            hovertext=[f"N={ctx}<br>Dist: {d:.1f}" for ctx, d in zip(ctx_subset, dist_within_mid)],
            hoverinfo='text',
            showlegend=False
        ))

        frame_data.append(go.Scatter(
            x=ctx_subset, y=dist_within_super,
            mode='lines+markers',
            line=dict(color='orange', width=2),
            marker=dict(size=6, color='orange'),
            name='Within Super',
            hovertext=[f"N={ctx}<br>Dist: {d:.1f}" for ctx, d in zip(ctx_subset, dist_within_super)],
            hoverinfo='text',
            showlegend=False
        ))

        frame_data.append(go.Scatter(
            x=ctx_subset, y=dist_across_super,
            mode='lines+markers',
            line=dict(color='red', width=2),
            marker=dict(size=6, color='red'),
            name='Across Super',
            hovertext=[f"N={ctx}<br>Dist: {d:.1f}" for ctx, d in zip(ctx_subset, dist_across_super)],
            hoverinfo='text',
            showlegend=False
        ))

        frames.append(go.Frame(data=frame_data, name=str(ctx_len)))

    # =================================================================
    # Initial data (first context length)
    # IMPORTANT: Must match frame structure exactly (16 tokens + other traces)
    # =================================================================
    ctx_len = context_lengths[0]

    # Panel 1: MDS positions - ALWAYS 16 traces (one per token)
    for token_text in sorted_tokens:
        traj_dict = {t[0]: t[1] for t in token_trajectories[token_text]}
        mid_cluster = hierarchy["level2"][graph.token_to_node[token_text]]
        ppl = token_perplexity.get(ctx_len, {}).get(token_text, 0)

        if ctx_len in traj_dict:
            point = traj_dict[ctx_len]
            x_val, y_val = [point[0]], [point[1]]
            text_val = [token_text]
        else:
            x_val, y_val = [None], [None]
            text_val = [""]

        fig.add_trace(go.Scatter(
            x=x_val, y=y_val,
            mode='markers+text',
            marker=dict(size=12, color=mid_colors[mid_cluster],
                       line=dict(color='black', width=1)),
            text=text_val,
            textposition='top center',
            textfont=dict(size=9),
            name=token_text,
            hovertext=f"{token_text}<br>Cluster: {mid_names[mid_cluster]}<br>PPL: {ppl:.2f}",
            hoverinfo='text',
            showlegend=False
        ), row=1, col=1)

    # Panel 2: Perplexity lines (4 traces, one per cluster)
    for c in range(4):
        fig.add_trace(go.Scatter(
            x=[context_lengths[0]],
            y=[mid_perplexity[c][0]],
            mode='lines+markers',
            line=dict(color=mid_colors[c], width=2),
            marker=dict(size=6, color=mid_colors[c]),
            name=mid_names[c],
            showlegend=True
        ), row=1, col=2)

    # Panel 3: Ratios
    fig.add_trace(go.Scatter(
        x=[context_lengths[0]], y=[influence_metrics["ratio_super_to_mid"][0]],
        mode='lines+markers',
        line=dict(color='#d62728', width=2),
        marker=dict(size=8),
        name='Cross/Within-Mid',
        showlegend=True
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=[context_lengths[0]], y=[influence_metrics["ratio_super_to_sibling"][0]],
        mode='lines+markers',
        line=dict(color='#9467bd', width=2),
        marker=dict(size=8),
        name='Cross/Within-Super',
        showlegend=True
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=[context_lengths[0], context_lengths[-1]], y=[1, 1],
        mode='lines',
        line=dict(color='gray', dash='dash'),
        showlegend=False
    ), row=2, col=1)

    # Panel 4: Distances
    fig.add_trace(go.Scatter(
        x=[context_lengths[0]], y=[influence_metrics["within_mid"][0]],
        mode='lines+markers',
        line=dict(color='green', width=2),
        name='Within Mid',
        showlegend=True
    ), row=2, col=2)

    fig.add_trace(go.Scatter(
        x=[context_lengths[0]], y=[influence_metrics["within_super"][0]],
        mode='lines+markers',
        line=dict(color='orange', width=2),
        name='Within Super',
        showlegend=True
    ), row=2, col=2)

    fig.add_trace(go.Scatter(
        x=[context_lengths[0]], y=[influence_metrics["across_super"][0]],
        mode='lines+markers',
        line=dict(color='red', width=2),
        name='Across Super',
        showlegend=True
    ), row=2, col=2)

    fig.frames = frames

    # Slider
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Context Length N=", "font": {"size": 18}},
        pad={"t": 60},
        steps=[dict(
            method="animate",
            args=[[str(ctx)], {"frame": {"duration": 200, "redraw": True}, "mode": "immediate"}],
            label=str(ctx)
        ) for ctx in context_lengths]
    )]

    # Update layout
    fig.update_layout(
        title=dict(
            text='Enhanced Interactive Analysis: MDS + Perplexity + Influence<br>' +
                 '<sup>Use slider to change context length | Watch hierarchy emerge</sup>',
            x=0.5,
            font=dict(size=16)
        ),
        sliders=sliders,
        width=1400,
        height=1000,
        legend=dict(x=1.02, y=0.5, font=dict(size=10)),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=1.08,
            x=0.1,
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, {"frame": {"duration": 300, "redraw": True}, "fromcurrent": True}]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
            ]
        )]
    )

    # Set axis properties with larger tick fonts - ALL METRICS ON LOG Y SCALE
    tick_font = dict(size=14)
    title_font = dict(size=14)

    # Panel 1: MDS (keep linear for spatial plot)
    fig.update_xaxes(title_text="MDS Dim 1", range=x_range, title_font=title_font,
                     tickfont=tick_font, row=1, col=1)
    fig.update_yaxes(title_text="MDS Dim 2", range=y_range, title_font=title_font,
                     tickfont=tick_font, row=1, col=1)

    # Panel 2: Perplexity (log-log)
    fig.update_xaxes(title_text="Context Length (N)", type="log", title_font=title_font,
                     tickfont=tick_font, row=1, col=2)
    fig.update_yaxes(title_text="Perplexity (log)", type="log", title_font=title_font,
                     tickfont=tick_font, row=1, col=2)

    # Panel 3: Ratios (log x, log y)
    fig.update_xaxes(title_text="Context Length (N)", type="log", title_font=title_font,
                     tickfont=dict(size=16), row=2, col=1)
    fig.update_yaxes(title_text="Distance Ratio (log)", type="log", title_font=title_font,
                     tickfont=tick_font, row=2, col=1)

    # Panel 4: Distances (log x, log y)
    fig.update_xaxes(title_text="Context Length (N)", type="log", title_font=title_font,
                     tickfont=dict(size=16), row=2, col=2)
    fig.update_yaxes(title_text="L2 Distance (log)", type="log", title_font=title_font,
                     tickfont=tick_font, row=2, col=2)

    fig.write_html(output_dir / "interactive_enhanced_slider.html")
    print(f"  Saved: {output_dir / 'interactive_enhanced_slider.html'}")

    # =========================================================================
    # Save metrics to JSON for later analysis
    # =========================================================================
    metrics_output = {
        "context_lengths": context_lengths,
        "influence_metrics": influence_metrics,
        "clustering_metrics": clustering_metrics,
        "mid_cluster_perplexity": {mid_names[c]: mid_perplexity[c] for c in range(4)},
        "super_cluster_perplexity": {super_names[c]: super_perplexity[c] for c in range(2)},
        "token_perplexity_by_context": {
            ctx: {t: float(p) for t, p in ppl.items()}
            for ctx, ppl in token_perplexity.items()
        }
    }

    with open(output_dir / "enhanced_metrics.json", "w") as f:
        json.dump(metrics_output, f, indent=2, default=lambda x: x if not hasattr(x, 'tolist') else x.tolist())
    print(f"  Saved: {output_dir / 'enhanced_metrics.json'}")

    # =========================================================================
    # Also create static summary plots
    # =========================================================================
    print("\nCreating static summary plots...")

    # Perplexity over context length
    fig_ppl = go.Figure()
    for c in range(4):
        fig_ppl.add_trace(go.Scatter(
            x=context_lengths, y=mid_perplexity[c],
            mode='lines+markers',
            line=dict(color=mid_colors[c], width=2),
            marker=dict(size=6),
            name=mid_names[c]
        ))

    fig_ppl.update_layout(
        title='Perplexity by Mid-Cluster Over Context Length',
        xaxis_title='Context Length (N)',
        yaxis_title='Mean Perplexity (log)',
        xaxis_type='log',
        yaxis_type='log',
        width=900,
        height=600
    )
    fig_ppl.write_html(output_dir / "perplexity_by_cluster.html")
    print(f"  Saved: {output_dir / 'perplexity_by_cluster.html'}")

    # Influence metrics over context length
    fig_inf = go.Figure()
    fig_inf.add_trace(go.Scatter(
        x=context_lengths, y=influence_metrics["ratio_super_to_mid"],
        mode='lines+markers',
        line=dict(color='#d62728', width=3),
        name='Cross-Super / Within-Mid'
    ))
    fig_inf.add_trace(go.Scatter(
        x=context_lengths, y=influence_metrics["ratio_super_to_sibling"],
        mode='lines+markers',
        line=dict(color='#9467bd', width=3),
        name='Cross-Super / Within-Super'
    ))
    fig_inf.add_hline(y=1, line_dash="dash", line_color="gray", annotation_text="Ratio = 1")

    fig_inf.update_layout(
        title='Hierarchy Influence Ratios Over Context Length<br>' +
              '<sup>Higher ratio = stronger hierarchy separation</sup>',
        xaxis_title='Context Length (N)',
        yaxis_title='Distance Ratio (log)',
        xaxis_type='log',
        yaxis_type='log',
        width=900,
        height=600
    )
    fig_inf.write_html(output_dir / "influence_ratios.html")
    print(f"  Saved: {output_dir / 'influence_ratios.html'}")

    # Clustering quality metrics over context length
    fig_clust = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Silhouette Score (higher = better clustering)',
                        'Davies-Bouldin Index (lower = better clustering)')
    )

    # Silhouette scores
    fig_clust.add_trace(go.Scatter(
        x=context_lengths, y=clustering_metrics["silhouette_mid"],
        mode='lines+markers',
        line=dict(color='#2ca02c', width=3),
        marker=dict(size=8),
        name='Silhouette (Mid-Cluster)'
    ), row=1, col=1)

    fig_clust.add_trace(go.Scatter(
        x=context_lengths, y=clustering_metrics["silhouette_super"],
        mode='lines+markers',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8),
        name='Silhouette (Super-Cluster)'
    ), row=1, col=1)

    fig_clust.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)

    # Davies-Bouldin index
    fig_clust.add_trace(go.Scatter(
        x=context_lengths, y=clustering_metrics["davies_bouldin"],
        mode='lines+markers',
        line=dict(color='#d62728', width=3),
        marker=dict(size=8),
        name='Davies-Bouldin'
    ), row=1, col=2)

    fig_clust.update_layout(
        title='Clustering Quality Metrics Over Context Length<br>' +
              '<sup>Silhouette: [-1,1], higher=better | Davies-Bouldin: [0,∞), lower=better</sup>',
        width=1200,
        height=500
    )

    fig_clust.update_xaxes(title_text="Context Length (N)", type="log", tickfont=dict(size=14))
    fig_clust.update_yaxes(title_text="Silhouette Score", row=1, col=1)  # Can't use log (negative values)
    fig_clust.update_yaxes(title_text="Davies-Bouldin Index (log)", type="log", row=1, col=2)

    fig_clust.write_html(output_dir / "clustering_quality.html")
    print(f"  Saved: {output_dir / 'clustering_quality.html'}")

    # Print summary of clustering metrics
    print("\n--- Clustering Quality Summary ---")
    best_sil_idx = np.argmax(clustering_metrics["silhouette_mid"])
    best_db_idx = np.argmin(clustering_metrics["davies_bouldin"])
    print(f"Best Silhouette (Mid) at N={context_lengths[best_sil_idx]}: {clustering_metrics['silhouette_mid'][best_sil_idx]:.3f}")
    print(f"Best Davies-Bouldin at N={context_lengths[best_db_idx]}: {clustering_metrics['davies_bouldin'][best_db_idx]:.3f}")

    print("\n" + "=" * 70)
    print("ENHANCED PLOTS COMPLETE")
    print("=" * 70)
    print(f"\nSaved to: {output_dir}")
    print("  - interactive_enhanced_slider.html (main multi-panel plot)")
    print("  - perplexity_by_cluster.html")
    print("  - influence_ratios.html")
    print("  - clustering_quality.html (NEW)")
    print("  - enhanced_metrics.json (raw data)")


if __name__ == "__main__":
    main()
