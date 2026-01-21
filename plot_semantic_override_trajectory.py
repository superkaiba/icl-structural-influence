#!/usr/bin/env python3
"""
Plot the full trajectory of semantic override:
- N=0: Pretrained (no context)
- N=1-50: Fine-grained early
- N=100-10000: Extended
"""

import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from collections import defaultdict
import wandb
import sys

sys.path.insert(0, str(Path('.') / 'src'))
from models import HookedLLM
from run_hierarchy_and_semantic_experiments import SemanticConflictGraph


def get_pretrained_distances(model, tokenizer, graph, layer_idx=-5):
    """Get distances with NO context (single token = pretrained)."""
    tokens = list(graph.vocabulary)

    reps = {}
    for tok in tokens:
        ids = tokenizer.encode(tok, add_special_tokens=False)
        input_ids = torch.tensor([ids]).to(model.device)
        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)
            rep = out.hidden_states[layer_idx][0, -1].cpu().float().numpy()
        reps[tok] = rep

    return reps


def collect_reps(model, tokenizer, graph, ctx_len, n_samples=50, layer_idx=-5):
    """Collect representations at a given context length."""
    token_reps = defaultdict(list)
    for _ in range(n_samples):
        prompt, nodes = graph.generate_random_walk(length=ctx_len, return_nodes=True)
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([tokens]).to(model.device)
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            hidden = outputs.hidden_states[layer_idx][0]
        token_texts = prompt.split()
        for pos, (node, txt) in enumerate(zip(nodes, token_texts)):
            if pos < hidden.shape[0]:
                token_reps[txt].append(hidden[pos].cpu().float().numpy())
    return {t: np.mean(r, axis=0) for t, r in token_reps.items() if r}


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
    print("SEMANTIC OVERRIDE TRAJECTORY")
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

    # Context lengths to test
    context_lengths = (
        [0] +  # Pretrained
        list(range(1, 11)) +  # Fine-grained early
        [15, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000] +
        list(range(1500, 10001, 500))
    )

    results = {
        'context_lengths': [],
        'l2_semantic': [],
        'l2_graph': [],
        'ratio': []
    }

    print(f"\nCollecting data for {len(context_lengths)} context lengths...")

    for n in context_lengths:
        print(f"  N={n}...", end=" ", flush=True)

        if n == 0:
            # Pretrained (no context)
            reps = get_pretrained_distances(model, tokenizer, graph)
        else:
            reps = collect_reps(model, tokenizer, graph, n, n_samples=50)

        l2_sem, l2_graph = compute_distances(reps, sem_pairs, graph_pairs)
        ratio = l2_sem / l2_graph if l2_graph and l2_graph > 0 else np.nan

        results['context_lengths'].append(n)
        results['l2_semantic'].append(l2_sem)
        results['l2_graph'].append(l2_graph)
        results['ratio'].append(ratio)

        winner = 'Graph' if ratio > 1 else 'Semantic'
        print(f"sem={l2_sem:.1f}, graph={l2_graph:.1f}, ratio={ratio:.3f} ({winner})")

    # Find key points
    # First stable crossover (3 consecutive graph wins)
    stable_crossover = None
    for i in range(len(results['ratio']) - 2):
        if all(r > 1 for r in results['ratio'][i:i+3]):
            stable_crossover = results['context_lengths'][i]
            break

    # Peak separation
    peak_idx = np.nanargmax(results['ratio'])
    peak_n = results['context_lengths'][peak_idx]
    peak_ratio = results['ratio'][peak_idx]

    print(f"\n--- KEY POINTS ---")
    print(f"Pretrained ratio: {results['ratio'][0]:.3f}")
    print(f"Stable crossover: N={stable_crossover}")
    print(f"Peak separation: N={peak_n}, ratio={peak_ratio:.3f}")

    # Create visualization
    print("\nCreating visualization...")

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "L2 Distance Over Context Length",
            "Ratio: Semantic/Graph (>1 = Graph Wins)",
            "Early Context (N=0-50) Detail",
            "Ratio with Key Points Marked"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # Use N=0.5 for plotting pretrained (so it shows on log scale)
    plot_n = [0.5 if n == 0 else n for n in results['context_lengths']]

    # Panel 1: L2 distances
    fig.add_trace(go.Scatter(
        x=plot_n, y=results['l2_semantic'],
        mode='lines+markers',
        name='Semantic-same (cat-dog)',
        line=dict(color='#d62728', width=2),
        marker=dict(size=6),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=plot_n, y=results['l2_graph'],
        mode='lines+markers',
        name='Graph-same (cat-computer)',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6),
    ), row=1, col=1)

    # Panel 2: Ratio
    fig.add_trace(go.Scatter(
        x=plot_n, y=results['ratio'],
        mode='lines+markers',
        name='Ratio',
        line=dict(color='#9467bd', width=3),
        marker=dict(size=8),
        showlegend=False,
    ), row=1, col=2)

    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=1, col=2)

    # Panel 3: Early detail (linear scale)
    early_mask = [n <= 50 for n in results['context_lengths']]
    early_n = [results['context_lengths'][i] for i in range(len(early_mask)) if early_mask[i]]
    early_sem = [results['l2_semantic'][i] for i in range(len(early_mask)) if early_mask[i]]
    early_graph = [results['l2_graph'][i] for i in range(len(early_mask)) if early_mask[i]]
    early_ratio = [results['ratio'][i] for i in range(len(early_mask)) if early_mask[i]]

    fig.add_trace(go.Scatter(
        x=early_n, y=early_ratio,
        mode='lines+markers',
        name='Ratio (early)',
        line=dict(color='#9467bd', width=2),
        marker=dict(size=8),
        showlegend=False,
    ), row=2, col=1)

    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=2, col=1)

    # Color points by winner
    colors = ['#d62728' if r < 1 else '#1f77b4' for r in early_ratio]
    fig.add_trace(go.Scatter(
        x=early_n, y=early_ratio,
        mode='markers',
        marker=dict(size=12, color=colors, line=dict(color='black', width=1)),
        showlegend=False,
    ), row=2, col=1)

    # Panel 4: Ratio with annotations
    fig.add_trace(go.Scatter(
        x=plot_n, y=results['ratio'],
        mode='lines+markers',
        line=dict(color='#9467bd', width=2),
        marker=dict(size=6),
        showlegend=False,
    ), row=2, col=2)

    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=2, col=2)

    # Mark key points
    # Pretrained
    fig.add_annotation(
        x=np.log10(0.5), y=results['ratio'][0],
        text=f"Pretrained<br>ratio={results['ratio'][0]:.2f}",
        showarrow=True, arrowhead=2,
        ax=40, ay=-40,
        row=2, col=2
    )

    # Stable crossover
    if stable_crossover:
        cross_idx = results['context_lengths'].index(stable_crossover)
        fig.add_vline(x=stable_crossover, line_dash="dot", line_color="green", row=2, col=2)
        fig.add_annotation(
            x=np.log10(stable_crossover), y=1.0,
            text=f"Stable crossover<br>N={stable_crossover}",
            showarrow=True, arrowhead=2,
            ax=0, ay=-50,
            row=2, col=2
        )

    # Peak
    fig.add_annotation(
        x=np.log10(peak_n), y=peak_ratio,
        text=f"Peak separation<br>N={peak_n}, ratio={peak_ratio:.2f}",
        showarrow=True, arrowhead=2,
        ax=-50, ay=-30,
        row=2, col=2
    )

    # Update axes
    fig.update_xaxes(title_text="Context Length (N)", type="log", row=1, col=1)
    fig.update_xaxes(title_text="Context Length (N)", type="log", row=1, col=2)
    fig.update_xaxes(title_text="Context Length (N)", row=2, col=1)  # Linear for early
    fig.update_xaxes(title_text="Context Length (N)", type="log", row=2, col=2)

    fig.update_yaxes(title_text="L2 Distance", type="log", row=1, col=1)
    fig.update_yaxes(title_text="Ratio (sem/graph)", row=1, col=2)
    fig.update_yaxes(title_text="Ratio (sem/graph)", row=2, col=1)
    fig.update_yaxes(title_text="Ratio (sem/graph)", row=2, col=2)

    fig.update_layout(
        title="When Does In-Context Learning Override Pretrained Semantics?<br>" +
              "<sup>Ratio > 1: Graph structure wins | Ratio < 1: Pretrained semantics win</sup>",
        width=1200,
        height=900,
        legend=dict(x=0.02, y=0.98),
    )

    output_dir = Path("results/semantic_override_trajectory")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig.write_html(output_dir / "semantic_override_trajectory.html")
    print(f"Saved: {output_dir / 'semantic_override_trajectory.html'}")

    # Log to W&B
    print("\nLogging to W&B...")

    run = wandb.init(
        project='icl-structural-influence',
        name='semantic-override-full-trajectory',
        tags=['semantic-override', 'pretrained', 'trajectory', 'crossover'],
        config={
            'context_lengths': results['context_lengths'],
            'max_context': 10000,
            'n_samples': 50,
        }
    )

    wandb.log({'semantic_override_trajectory': wandb.Html(open(output_dir / 'semantic_override_trajectory.html').read())})

    # Log table
    table = wandb.Table(columns=['N', 'L2_semantic', 'L2_graph', 'Ratio', 'Winner'])
    for i, n in enumerate(results['context_lengths']):
        winner = 'Graph' if results['ratio'][i] > 1 else 'Semantic'
        table.add_data(n, results['l2_semantic'][i], results['l2_graph'][i],
                       results['ratio'][i], winner)
    wandb.log({'trajectory_table': table})

    # Summary
    wandb.summary['pretrained_ratio'] = results['ratio'][0]
    wandb.summary['stable_crossover_N'] = stable_crossover
    wandb.summary['peak_separation_N'] = peak_n
    wandb.summary['peak_separation_ratio'] = peak_ratio
    wandb.summary['final_ratio'] = results['ratio'][-1]

    print(f"\nRun URL: {run.url}")
    wandb.finish()

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
