#!/usr/bin/env python3
"""
Layer-wise analysis of semantic override at N=2000.
Shows at which layers pretrained semantics are overridden by graph structure.
"""

import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from collections import defaultdict
import wandb
import json
import sys

sys.path.insert(0, str(Path('.') / 'src'))
from models import HookedLLM
from run_hierarchy_and_semantic_experiments import SemanticConflictGraph


def get_pretrained_reps_all_layers(model, tokenizer, graph):
    """Get pretrained representations (no context) at ALL layers."""
    tokens = list(graph.vocabulary)
    n_layers = model.config.num_hidden_layers

    # reps[layer_idx][token] = representation
    reps = {layer: {} for layer in range(n_layers + 1)}  # +1 for embedding layer

    for tok in tokens:
        ids = tokenizer.encode(tok, add_special_tokens=False)
        input_ids = torch.tensor([ids]).to(model.device)
        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)
            # hidden_states[0] is embedding, [1] is after layer 0, etc.
            for layer_idx, hidden in enumerate(out.hidden_states):
                reps[layer_idx][tok] = hidden[0, -1].cpu().float().numpy()

    return reps


def collect_reps_all_layers(model, tokenizer, graph, ctx_len, n_samples=50, seed=None):
    """Collect representations at ALL layers for given context length."""
    if seed is not None:
        np.random.seed(seed)

    n_layers = model.config.num_hidden_layers

    # token_reps[layer_idx][token] = list of representations
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

    # Average representations per token per layer
    avg_reps = {}
    for layer_idx in range(n_layers + 1):
        avg_reps[layer_idx] = {
            t: np.mean(r, axis=0)
            for t, r in token_reps[layer_idx].items() if r
        }

    return avg_reps


def compute_distances(reps, sem_pairs, graph_pairs):
    """Compute mean L2 distances for semantic and graph pairs."""
    sem_dists = [np.linalg.norm(reps[t1] - reps[t2])
                 for t1, t2 in sem_pairs if t1 in reps and t2 in reps]
    graph_dists = [np.linalg.norm(reps[t1] - reps[t2])
                   for t1, t2 in graph_pairs if t1 in reps and t2 in reps]
    return (np.mean(sem_dists) if sem_dists else np.nan,
            np.mean(graph_dists) if graph_dists else np.nan)


def main():
    print("=" * 70)
    print("LAYER-WISE SEMANTIC OVERRIDE ANALYSIS (N=2000)")
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

    print(f"\nSemantic pairs: {len(sem_pairs)}")
    print(f"Graph pairs: {len(graph_pairs)}")

    # Parameters
    ctx_len = 2000
    n_trials = 10

    # Results storage
    results = {
        'layers': list(range(n_layers + 1)),
        'layer_names': ['Embedding'] + [f'Layer {i}' for i in range(n_layers)],
        'pretrained_ratio': [],
        'pretrained_sem': [],
        'pretrained_graph': [],
        'context_ratio_mean': [],
        'context_ratio_std': [],
        'context_sem_mean': [],
        'context_graph_mean': [],
    }

    # Get pretrained representations (no context)
    print("\nCollecting pretrained representations (no context)...")
    pretrained_reps = get_pretrained_reps_all_layers(model, tokenizer, graph)

    for layer_idx in range(n_layers + 1):
        l2_sem, l2_graph = compute_distances(pretrained_reps[layer_idx], sem_pairs, graph_pairs)
        ratio = l2_sem / l2_graph if l2_graph > 0 else np.nan
        results['pretrained_ratio'].append(float(ratio))
        results['pretrained_sem'].append(float(l2_sem))
        results['pretrained_graph'].append(float(l2_graph))
        print(f"  {results['layer_names'][layer_idx]}: ratio={ratio:.3f}")

    # Collect context representations with multiple trials
    print(f"\nCollecting representations at N={ctx_len} ({n_trials} trials)...")

    trial_ratios = {layer: [] for layer in range(n_layers + 1)}
    trial_sem = {layer: [] for layer in range(n_layers + 1)}
    trial_graph = {layer: [] for layer in range(n_layers + 1)}

    for trial in range(n_trials):
        print(f"  Trial {trial + 1}/{n_trials}...", end=" ", flush=True)

        context_reps = collect_reps_all_layers(
            model, tokenizer, graph, ctx_len,
            n_samples=50, seed=42 + trial * 1000
        )

        for layer_idx in range(n_layers + 1):
            l2_sem, l2_graph = compute_distances(context_reps[layer_idx], sem_pairs, graph_pairs)
            ratio = l2_sem / l2_graph if l2_graph > 0 else np.nan
            trial_ratios[layer_idx].append(ratio)
            trial_sem[layer_idx].append(l2_sem)
            trial_graph[layer_idx].append(l2_graph)

        print(f"done")

    # Aggregate results
    for layer_idx in range(n_layers + 1):
        results['context_ratio_mean'].append(float(np.nanmean(trial_ratios[layer_idx])))
        results['context_ratio_std'].append(float(np.nanstd(trial_ratios[layer_idx])))
        results['context_sem_mean'].append(float(np.nanmean(trial_sem[layer_idx])))
        results['context_graph_mean'].append(float(np.nanmean(trial_graph[layer_idx])))

    # Analysis
    print("\n" + "=" * 70)
    print("LAYER-WISE RESULTS")
    print("=" * 70)
    print(f"\n{'Layer':<12} {'Pretrained':<12} {'N=2000':<16} {'Change':<12}")
    print("-" * 52)

    for i, layer_name in enumerate(results['layer_names']):
        pre = results['pretrained_ratio'][i]
        ctx = results['context_ratio_mean'][i]
        std = results['context_ratio_std'][i]
        change = ctx - pre
        winner = "Graph" if ctx > 1 else "Semantic"
        print(f"{layer_name:<12} {pre:<12.3f} {ctx:.3f}±{std:.3f}     {change:+.3f} [{winner}]")

    # Find key layers
    # First layer where context ratio > 1
    first_override = None
    for i, r in enumerate(results['context_ratio_mean']):
        if r > 1:
            first_override = i
            break

    # Layer with maximum override
    max_override_idx = np.argmax(results['context_ratio_mean'])
    max_override_ratio = results['context_ratio_mean'][max_override_idx]

    # Layer with maximum change from pretrained
    changes = [c - p for c, p in zip(results['context_ratio_mean'], results['pretrained_ratio'])]
    max_change_idx = np.argmax(changes)
    max_change = changes[max_change_idx]

    print(f"\n--- KEY FINDINGS ---")
    print(f"First layer with override (ratio > 1): {results['layer_names'][first_override] if first_override else 'None'}")
    print(f"Maximum override: {results['layer_names'][max_override_idx]} (ratio={max_override_ratio:.3f})")
    print(f"Maximum change from pretrained: {results['layer_names'][max_change_idx]} (change={max_change:+.3f})")

    # Create visualization
    print("\n" + "=" * 70)
    print("Creating visualization...")
    print("=" * 70)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Ratio by Layer: Pretrained vs N=2000",
            "L2 Distances by Layer (N=2000)",
            "Change from Pretrained by Layer",
            "Ratio Difference (Context - Pretrained)"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    layers = results['layers']

    # Panel 1: Ratio comparison
    fig.add_trace(go.Scatter(
        x=layers, y=results['pretrained_ratio'],
        mode='lines+markers',
        name='Pretrained (N=0)',
        line=dict(color='#d62728', width=2),
        marker=dict(size=6),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=layers, y=results['context_ratio_mean'],
        mode='lines+markers',
        name='N=2000',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6),
        error_y=dict(type='data', array=results['context_ratio_std'], visible=True),
    ), row=1, col=1)

    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=1, col=1)

    # Panel 2: L2 distances at N=2000
    fig.add_trace(go.Scatter(
        x=layers, y=results['context_sem_mean'],
        mode='lines+markers',
        name='Semantic-same',
        line=dict(color='#d62728', width=2),
        marker=dict(size=6),
        showlegend=False,
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=layers, y=results['context_graph_mean'],
        mode='lines+markers',
        name='Graph-same',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6),
        showlegend=False,
    ), row=2, col=1)

    # Add both to panel 2
    fig.data[-2].update(showlegend=True)
    fig.data[-1].update(showlegend=True)

    # Actually put them in the right panel
    fig.data[-2].update(xaxis='x2', yaxis='y2')
    fig.data[-1].update(xaxis='x2', yaxis='y2')

    # Panel 3: Pretrained distances
    fig.add_trace(go.Scatter(
        x=layers, y=results['pretrained_sem'],
        mode='lines+markers',
        name='Semantic (pretrained)',
        line=dict(color='#d62728', width=2, dash='dot'),
        marker=dict(size=6),
        showlegend=False,
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=layers, y=results['pretrained_graph'],
        mode='lines+markers',
        name='Graph (pretrained)',
        line=dict(color='#1f77b4', width=2, dash='dot'),
        marker=dict(size=6),
        showlegend=False,
    ), row=2, col=1)

    # Panel 4: Change from pretrained
    fig.add_trace(go.Bar(
        x=layers, y=changes,
        name='Ratio change',
        marker_color=['#2ca02c' if c > 0 else '#d62728' for c in changes],
        showlegend=False,
    ), row=2, col=2)

    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=2)

    # Update axes
    fig.update_xaxes(title_text="Layer", row=1, col=1)
    fig.update_xaxes(title_text="Layer", row=1, col=2)
    fig.update_xaxes(title_text="Layer", row=2, col=1)
    fig.update_xaxes(title_text="Layer", row=2, col=2)

    fig.update_yaxes(title_text="Ratio (sem/graph)", row=1, col=1)
    fig.update_yaxes(title_text="L2 Distance", row=1, col=2)
    fig.update_yaxes(title_text="L2 Distance", row=2, col=1)
    fig.update_yaxes(title_text="Ratio Change", row=2, col=2)

    # Add layer region annotations
    fig.add_vrect(x0=-0.5, x1=10.5, fillcolor="rgba(255,0,0,0.05)",
                  line_width=0, row=1, col=1, annotation_text="Early",
                  annotation_position="top left")
    fig.add_vrect(x0=10.5, x1=21.5, fillcolor="rgba(255,255,0,0.05)",
                  line_width=0, row=1, col=1, annotation_text="Middle")
    fig.add_vrect(x0=21.5, x1=32.5, fillcolor="rgba(0,255,0,0.05)",
                  line_width=0, row=1, col=1, annotation_text="Late")

    fig.update_layout(
        title=f"Layer-wise Semantic Override Analysis (N={ctx_len}, {n_trials} trials)<br>" +
              f"<sup>First override: {results['layer_names'][first_override] if first_override else 'None'} | " +
              f"Max override: {results['layer_names'][max_override_idx]} (ratio={max_override_ratio:.2f})</sup>",
        width=1200,
        height=900,
        legend=dict(x=0.02, y=0.98),
    )

    output_dir = Path("results/unrelated_tokens_layerwise")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig.write_html(output_dir / "semantic_override_layerwise.html")
    print(f"Saved: {output_dir / 'semantic_override_layerwise.html'}")

    # Save JSON
    with open(output_dir / "layerwise_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {output_dir / 'layerwise_results.json'}")

    # Log to W&B
    print("\nLogging to W&B...")

    run = wandb.init(
        project='icl-structural-influence',
        name='semantic-override-layerwise',
        tags=['semantic-override', 'layerwise', 'N=2000'],
        config={
            'context_length': ctx_len,
            'n_trials': n_trials,
            'n_layers': n_layers,
            'n_samples_per_trial': 50,
        }
    )

    wandb.log({
        'layerwise_plot': wandb.Html(
            open(output_dir / 'semantic_override_layerwise.html').read()
        )
    })

    # Log table
    table = wandb.Table(columns=[
        'Layer', 'Layer_Name', 'Pretrained_Ratio', 'Context_Ratio_Mean',
        'Context_Ratio_Std', 'Change', 'Winner'
    ])
    for i in range(n_layers + 1):
        change = results['context_ratio_mean'][i] - results['pretrained_ratio'][i]
        winner = 'Graph' if results['context_ratio_mean'][i] > 1 else 'Semantic'
        table.add_data(
            i, results['layer_names'][i],
            results['pretrained_ratio'][i],
            results['context_ratio_mean'][i],
            results['context_ratio_std'][i],
            change, winner
        )
    wandb.log({'layerwise_table': table})

    # Summary
    wandb.summary['context_length'] = ctx_len
    wandb.summary['n_trials'] = n_trials
    wandb.summary['first_override_layer'] = first_override
    wandb.summary['max_override_layer'] = int(max_override_idx)
    wandb.summary['max_override_ratio'] = max_override_ratio
    wandb.summary['max_change_layer'] = int(max_change_idx)
    wandb.summary['max_change'] = max_change

    print(f"\nRun URL: {run.url}")
    wandb.finish()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Context length: N={ctx_len}")
    print(f"Trials: {n_trials}")
    print(f"\nFirst layer with override: {results['layer_names'][first_override] if first_override else 'None'}")
    print(f"Maximum override: {results['layer_names'][max_override_idx]} (ratio={max_override_ratio:.3f})")
    print(f"Maximum change: {results['layer_names'][max_change_idx]} (change={max_change:+.3f})")
    print("=" * 70)


if __name__ == "__main__":
    main()
