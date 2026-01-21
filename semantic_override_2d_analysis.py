#!/usr/bin/env python3
"""
2D Analysis: Semantic override across both context length (N) and layer depth.
Creates heatmap showing ratio at each (layer, N) combination.
"""

import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from pathlib import Path
from collections import defaultdict
import wandb
import json
import sys

sys.path.insert(0, str(Path('.') / 'src'))
from models import HookedLLM
from run_hierarchy_and_semantic_experiments import SemanticConflictGraph


def get_reps_all_layers(model, tokenizer, graph, ctx_len, n_samples=50, seed=None):
    """Get representations at ALL layers for given context length."""
    if seed is not None:
        np.random.seed(seed)

    n_layers = model.config.num_hidden_layers
    tokens_list = list(graph.vocabulary)

    if ctx_len == 0:
        # Pretrained: single token, no context
        reps = {layer: {} for layer in range(n_layers + 1)}
        for tok in tokens_list:
            ids = tokenizer.encode(tok, add_special_tokens=False)
            input_ids = torch.tensor([ids]).to(model.device)
            with torch.no_grad():
                out = model(input_ids, output_hidden_states=True)
                for layer_idx, hidden in enumerate(out.hidden_states):
                    reps[layer_idx][tok] = hidden[0, -1].cpu().float().numpy()
        return reps

    # With context: collect from random walks
    token_reps = {layer: defaultdict(list) for layer in range(n_layers + 1)}

    for _ in range(n_samples):
        prompt, nodes = graph.generate_random_walk(length=ctx_len, return_nodes=True)
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([tokens]).to(model.device)

        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            token_texts = prompt.split()

            for layer_idx, hidden in enumerate(outputs.hidden_states):
                for pos, (node, txt) in enumerate(zip(nodes, token_texts)):
                    if pos < hidden.shape[1]:
                        token_reps[layer_idx][txt].append(
                            hidden[0, pos].cpu().float().numpy()
                        )

    # Average
    avg_reps = {}
    for layer_idx in range(n_layers + 1):
        avg_reps[layer_idx] = {
            t: np.mean(r, axis=0)
            for t, r in token_reps[layer_idx].items() if r
        }

    return avg_reps


def compute_distances(reps, sem_pairs, graph_pairs):
    """Compute mean L2 distances."""
    sem_dists = [np.linalg.norm(reps[t1] - reps[t2])
                 for t1, t2 in sem_pairs if t1 in reps and t2 in reps]
    graph_dists = [np.linalg.norm(reps[t1] - reps[t2])
                   for t1, t2 in graph_pairs if t1 in reps and t2 in reps]
    return (np.mean(sem_dists) if sem_dists else np.nan,
            np.mean(graph_dists) if graph_dists else np.nan)


def main():
    print("=" * 70)
    print("2D ANALYSIS: SEMANTIC OVERRIDE (Layer x Context Length)")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    hooked = HookedLLM.from_pretrained(
        'meta-llama/Llama-3.1-8B',
        torch_dtype=torch.float16,
        device_map='auto'
    )
    model, tokenizer = hooked.model, hooked.tokenizer
    model.eval()

    n_layers = model.config.num_hidden_layers
    print(f"Model has {n_layers} layers (+ embedding layer)")

    # Create graph
    graph = SemanticConflictGraph(seed=42)

    # Define pairs
    sem_pairs = [(t1, t2) for t1, t2, _ in graph.get_semantic_pairs()]
    graph_pairs = []
    for cid, toks in graph.graph_clusters.items():
        for i, t1 in enumerate(toks):
            for t2 in toks[i+1:]:
                if graph.token_to_semantic_group[t1] != graph.token_to_semantic_group[t2]:
                    graph_pairs.append((t1, t2))

    # Context lengths to sample
    context_lengths = [0, 1, 2, 3, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000]

    n_trials = 5
    layers = list(range(n_layers + 1))  # 0 to 32 (embedding + 32 layers)

    # Results matrices
    ratio_matrix = np.zeros((len(layers), len(context_lengths)))
    sem_matrix = np.zeros((len(layers), len(context_lengths)))
    graph_matrix = np.zeros((len(layers), len(context_lengths)))
    std_matrix = np.zeros((len(layers), len(context_lengths)))

    print(f"\nCollecting data for {len(context_lengths)} context lengths x {len(layers)} layers")
    print(f"Trials per (N, layer): {n_trials}")

    for n_idx, ctx_len in enumerate(context_lengths):
        print(f"\nN={ctx_len}...", end=" ", flush=True)

        trial_ratios = {layer: [] for layer in layers}
        trial_sem = {layer: [] for layer in layers}
        trial_graph = {layer: [] for layer in layers}

        for trial in range(n_trials):
            reps = get_reps_all_layers(
                model, tokenizer, graph, ctx_len,
                n_samples=50, seed=42 + trial * 1000
            )

            for layer_idx in layers:
                l2_sem, l2_graph = compute_distances(reps[layer_idx], sem_pairs, graph_pairs)
                ratio = l2_sem / l2_graph if l2_graph > 0 else np.nan
                trial_ratios[layer_idx].append(ratio)
                trial_sem[layer_idx].append(l2_sem)
                trial_graph[layer_idx].append(l2_graph)

        # Aggregate
        for layer_idx in layers:
            ratio_matrix[layer_idx, n_idx] = np.nanmean(trial_ratios[layer_idx])
            sem_matrix[layer_idx, n_idx] = np.nanmean(trial_sem[layer_idx])
            graph_matrix[layer_idx, n_idx] = np.nanmean(trial_graph[layer_idx])
            std_matrix[layer_idx, n_idx] = np.nanstd(trial_ratios[layer_idx])

        # Summary for this N
        mean_ratio = np.nanmean([ratio_matrix[l, n_idx] for l in layers])
        print(f"mean ratio across layers: {mean_ratio:.3f}")

    # Find crossover points per layer
    crossover_per_layer = []
    for layer_idx in layers:
        crossover_n = None
        for n_idx, ctx_len in enumerate(context_lengths):
            if ratio_matrix[layer_idx, n_idx] > 1:
                crossover_n = ctx_len
                break
        crossover_per_layer.append(crossover_n)

    print("\n" + "=" * 70)
    print("CROSSOVER POINTS BY LAYER")
    print("=" * 70)
    for layer_idx, crossover_n in enumerate(crossover_per_layer):
        layer_name = "Embedding" if layer_idx == 0 else f"Layer {layer_idx - 1}"
        print(f"  {layer_name}: N={crossover_n}")

    # Create visualization
    print("\n" + "=" * 70)
    print("Creating visualization...")
    print("=" * 70)

    # Main heatmap
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Ratio (Semantic/Graph) by Layer and Context Length",
            "Crossover Point (N where ratio > 1) by Layer",
            "Semantic Distance by Layer and N",
            "Graph Distance by Layer and N"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.12,
        specs=[[{"type": "heatmap"}, {"type": "scatter"}],
               [{"type": "heatmap"}, {"type": "heatmap"}]]
    )

    # Custom colorscale: red (semantic wins) -> white (equal) -> blue (graph wins)
    colorscale = [
        [0.0, 'rgb(178,24,43)'],      # Dark red (ratio << 1)
        [0.25, 'rgb(239,138,98)'],    # Light red
        [0.5, 'rgb(255,255,255)'],    # White (ratio = 1)
        [0.75, 'rgb(103,169,207)'],   # Light blue
        [1.0, 'rgb(33,102,172)']      # Dark blue (ratio >> 1)
    ]

    # Panel 1: Ratio heatmap
    # Normalize ratio for colorscale: map [0, 2] to [0, 1]
    ratio_normalized = np.clip(ratio_matrix / 2, 0, 1)

    layer_labels = ['Emb'] + [str(i) for i in range(n_layers)]

    fig.add_trace(go.Heatmap(
        z=ratio_matrix,
        x=context_lengths,
        y=layer_labels,
        colorscale=colorscale,
        zmin=0, zmax=2,
        colorbar=dict(title="Ratio", x=0.45),
        hovertemplate="N=%{x}<br>Layer %{y}<br>Ratio=%{z:.3f}<extra></extra>",
    ), row=1, col=1)

    # Add contour line at ratio = 1
    fig.add_trace(go.Contour(
        z=ratio_matrix,
        x=context_lengths,
        y=list(range(len(layer_labels))),
        contours=dict(
            start=1, end=1, size=1,
            coloring='none',
            showlabels=True,
        ),
        line=dict(color='black', width=2),
        showscale=False,
        hoverinfo='skip',
    ), row=1, col=1)

    # Panel 2: Crossover point by layer
    valid_crossovers = [(i, c) for i, c in enumerate(crossover_per_layer) if c is not None]
    if valid_crossovers:
        fig.add_trace(go.Scatter(
            x=[c for _, c in valid_crossovers],
            y=[layer_labels[i] for i, _ in valid_crossovers],
            mode='markers+lines',
            marker=dict(size=10, color='#1f77b4'),
            line=dict(color='#1f77b4', width=2),
            name='Crossover N',
        ), row=1, col=2)

    # Panel 3: Semantic distance heatmap
    fig.add_trace(go.Heatmap(
        z=sem_matrix,
        x=context_lengths,
        y=layer_labels,
        colorscale='Reds',
        colorbar=dict(title="L2 Dist", x=1.0),
        hovertemplate="N=%{x}<br>Layer %{y}<br>Sem Dist=%{z:.2f}<extra></extra>",
    ), row=2, col=1)

    # Panel 4: Graph distance heatmap
    fig.add_trace(go.Heatmap(
        z=graph_matrix,
        x=context_lengths,
        y=layer_labels,
        colorscale='Blues',
        colorbar=dict(title="L2 Dist", x=1.02),
        hovertemplate="N=%{x}<br>Layer %{y}<br>Graph Dist=%{z:.2f}<extra></extra>",
    ), row=2, col=2)

    # Update axes
    fig.update_xaxes(title_text="Context Length (N)", type="log", row=1, col=1)
    fig.update_xaxes(title_text="Context Length (N) at Crossover", type="log", row=1, col=2)
    fig.update_xaxes(title_text="Context Length (N)", type="log", row=2, col=1)
    fig.update_xaxes(title_text="Context Length (N)", type="log", row=2, col=2)

    fig.update_yaxes(title_text="Layer", row=1, col=1)
    fig.update_yaxes(title_text="Layer", row=1, col=2)
    fig.update_yaxes(title_text="Layer", row=2, col=1)
    fig.update_yaxes(title_text="Layer", row=2, col=2)

    fig.update_layout(
        title="2D Semantic Override Analysis: Layer × Context Length<br>" +
              "<sup>Red = Semantic wins | White = Equal | Blue = Graph wins</sup>",
        width=1400,
        height=1100,
        showlegend=False,
    )

    output_dir = Path("results/unrelated_tokens_layerwise")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig.write_html(output_dir / "semantic_override_2d.html")
    print(f"Saved: {output_dir / 'semantic_override_2d.html'}")

    # Create line plot version showing trajectories
    fig2 = go.Figure()

    # Sample layers for clarity
    sample_layers = [0, 1, 4, 8, 12, 16, 20, 24, 28, 32]
    colors = px.colors.sample_colorscale('Viridis', len(sample_layers))

    for i, layer_idx in enumerate(sample_layers):
        layer_name = "Embedding" if layer_idx == 0 else f"Layer {layer_idx - 1}"
        fig2.add_trace(go.Scatter(
            x=context_lengths,
            y=ratio_matrix[layer_idx, :],
            mode='lines+markers',
            name=layer_name,
            line=dict(color=colors[i], width=2),
            marker=dict(size=6),
        ))

    fig2.add_hline(y=1.0, line_dash="dash", line_color="gray")

    fig2.update_xaxes(title_text="Context Length (N)", type="log")
    fig2.update_yaxes(title_text="Ratio (Semantic/Graph)")
    fig2.update_layout(
        title="Ratio Trajectories by Layer (N=0 to 2000)<br>" +
              "<sup>Above 1 = Graph wins | Below 1 = Semantic wins</sup>",
        width=1000,
        height=600,
        legend=dict(x=1.02, y=1),
    )

    fig2.write_html(output_dir / "semantic_override_2d_lines.html")
    print(f"Saved: {output_dir / 'semantic_override_2d_lines.html'}")

    # Save data
    results = {
        'context_lengths': context_lengths,
        'layers': layers,
        'layer_labels': layer_labels,
        'ratio_matrix': ratio_matrix.tolist(),
        'sem_matrix': sem_matrix.tolist(),
        'graph_matrix': graph_matrix.tolist(),
        'std_matrix': std_matrix.tolist(),
        'crossover_per_layer': crossover_per_layer,
        'n_trials': n_trials,
    }

    with open(output_dir / "2d_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {output_dir / '2d_analysis_results.json'}")

    # Log to W&B
    print("\nLogging to W&B...")

    run = wandb.init(
        project='icl-structural-influence',
        name='semantic-override-2d-analysis',
        tags=['semantic-override', '2d-analysis', 'layer-x-context'],
        config={
            'context_lengths': context_lengths,
            'n_layers': n_layers,
            'n_trials': n_trials,
        }
    )

    wandb.log({
        '2d_heatmap': wandb.Html(open(output_dir / 'semantic_override_2d.html').read()),
        '2d_lines': wandb.Html(open(output_dir / 'semantic_override_2d_lines.html').read()),
    })

    # Summary stats
    # Find earliest crossover (any layer)
    earliest_crossover = min([c for c in crossover_per_layer if c is not None], default=None)
    # Find layer that crosses over first
    first_layer_to_cross = None
    for layer_idx, c in enumerate(crossover_per_layer):
        if c == earliest_crossover:
            first_layer_to_cross = layer_idx
            break

    wandb.summary['earliest_crossover_N'] = earliest_crossover
    wandb.summary['first_layer_to_cross'] = first_layer_to_cross
    wandb.summary['max_ratio'] = float(np.nanmax(ratio_matrix))
    wandb.summary['max_ratio_layer'] = int(np.unravel_index(np.nanargmax(ratio_matrix), ratio_matrix.shape)[0])
    wandb.summary['max_ratio_N'] = context_lengths[int(np.unravel_index(np.nanargmax(ratio_matrix), ratio_matrix.shape)[1])]

    print(f"\nRun URL: {run.url}")
    wandb.finish()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Earliest crossover: N={earliest_crossover} (Layer {first_layer_to_cross})")
    print(f"Maximum ratio: {np.nanmax(ratio_matrix):.3f} at Layer {np.unravel_index(np.nanargmax(ratio_matrix), ratio_matrix.shape)[0]}, N={context_lengths[np.unravel_index(np.nanargmax(ratio_matrix), ratio_matrix.shape)[1]]}")
    print("=" * 70)


if __name__ == "__main__":
    main()
