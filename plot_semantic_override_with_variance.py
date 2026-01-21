#!/usr/bin/env python3
"""
Plot semantic override trajectory with error bars from multiple trials.
Shows variance to demonstrate reliability of the crossover finding.
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


def collect_reps_single_trial(model, tokenizer, graph, ctx_len, n_samples=50, layer_idx=-5, seed=None):
    """Collect representations for a single trial."""
    if seed is not None:
        np.random.seed(seed)

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
    print("SEMANTIC OVERRIDE TRAJECTORY WITH VARIANCE")
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

    # Context lengths - focus on key points with multiple trials
    # Dense sampling at early N where variance is high
    context_lengths = (
        [0] +  # Pretrained
        list(range(1, 21)) +  # Fine-grained early (high variance region)
        [25, 30, 40, 50, 75, 100, 150, 200, 300, 500, 750, 1000] +
        [1500, 2000, 3000, 5000, 7500, 10000]
    )

    n_trials = 5  # Number of independent trials for error bars

    results = {
        'context_lengths': [],
        'ratio_mean': [],
        'ratio_std': [],
        'ratio_trials': [],
        'l2_sem_mean': [],
        'l2_sem_std': [],
        'l2_graph_mean': [],
        'l2_graph_std': [],
    }

    print(f"\nCollecting data for {len(context_lengths)} context lengths, {n_trials} trials each...")

    for n in context_lengths:
        print(f"  N={n}...", end=" ", flush=True)

        trial_ratios = []
        trial_sem = []
        trial_graph = []

        for trial in range(n_trials):
            if n == 0:
                # Pretrained (no context) - same for all trials
                reps = get_pretrained_distances(model, tokenizer, graph)
            else:
                reps = collect_reps_single_trial(
                    model, tokenizer, graph, n,
                    n_samples=50, seed=42 + trial * 1000
                )

            l2_sem, l2_graph = compute_distances(reps, sem_pairs, graph_pairs)
            ratio = l2_sem / l2_graph if l2_graph and l2_graph > 0 else np.nan

            trial_ratios.append(ratio)
            trial_sem.append(l2_sem)
            trial_graph.append(l2_graph)

        results['context_lengths'].append(n)
        results['ratio_mean'].append(np.nanmean(trial_ratios))
        results['ratio_std'].append(np.nanstd(trial_ratios))
        results['ratio_trials'].append(trial_ratios)
        results['l2_sem_mean'].append(np.nanmean(trial_sem))
        results['l2_sem_std'].append(np.nanstd(trial_sem))
        results['l2_graph_mean'].append(np.nanmean(trial_graph))
        results['l2_graph_std'].append(np.nanstd(trial_graph))

        winner = 'Graph' if results['ratio_mean'][-1] > 1 else 'Semantic'
        print(f"ratio={results['ratio_mean'][-1]:.3f}±{results['ratio_std'][-1]:.3f} ({winner})")

    # Find key points
    # First stable crossover (3 consecutive graph wins by mean)
    stable_crossover = None
    for i in range(len(results['ratio_mean']) - 2):
        if all(r > 1 for r in results['ratio_mean'][i:i+3]):
            stable_crossover = results['context_lengths'][i]
            break

    # Peak separation
    peak_idx = np.nanargmax(results['ratio_mean'])
    peak_n = results['context_lengths'][peak_idx]
    peak_ratio = results['ratio_mean'][peak_idx]

    # First statistically significant crossover (mean - std > 1)
    sig_crossover = None
    for i, (mean, std) in enumerate(zip(results['ratio_mean'], results['ratio_std'])):
        if mean - std > 1:
            sig_crossover = results['context_lengths'][i]
            break

    print(f"\n--- KEY POINTS ---")
    print(f"Pretrained ratio: {results['ratio_mean'][0]:.3f}")
    print(f"Stable crossover (3 consecutive): N={stable_crossover}")
    print(f"Statistically significant crossover (mean-std > 1): N={sig_crossover}")
    print(f"Peak separation: N={peak_n}, ratio={peak_ratio:.3f}")

    # Create visualization
    print("\nCreating visualization...")

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "L2 Distance with Error Bars",
            "Ratio: Semantic/Graph with 95% CI",
            "Early Context Detail (N=0-50)",
            "Variance Analysis"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # Use N=0.5 for plotting pretrained (so it shows on log scale)
    plot_n = [0.5 if n == 0 else n for n in results['context_lengths']]

    # Panel 1: L2 distances with error bars
    fig.add_trace(go.Scatter(
        x=plot_n, y=results['l2_sem_mean'],
        mode='lines+markers',
        name='Semantic-same (cat-dog)',
        line=dict(color='#d62728', width=2),
        marker=dict(size=6),
        error_y=dict(type='data', array=results['l2_sem_std'], visible=True),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=plot_n, y=results['l2_graph_mean'],
        mode='lines+markers',
        name='Graph-same (cat-computer)',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6),
        error_y=dict(type='data', array=results['l2_graph_std'], visible=True),
    ), row=1, col=1)

    # Panel 2: Ratio with confidence interval (approx 95% CI = 2*std)
    ci_95 = [2 * s for s in results['ratio_std']]

    fig.add_trace(go.Scatter(
        x=plot_n, y=results['ratio_mean'],
        mode='lines+markers',
        name='Ratio (mean)',
        line=dict(color='#9467bd', width=3),
        marker=dict(size=8),
        error_y=dict(type='data', array=ci_95, visible=True),
        showlegend=False,
    ), row=1, col=2)

    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=1, col=2)

    # Panel 3: Early detail (linear scale, N=0-50)
    early_mask = [n <= 50 for n in results['context_lengths']]
    early_n = [results['context_lengths'][i] for i in range(len(early_mask)) if early_mask[i]]
    early_ratio = [results['ratio_mean'][i] for i in range(len(early_mask)) if early_mask[i]]
    early_std = [results['ratio_std'][i] for i in range(len(early_mask)) if early_mask[i]]
    early_ci = [2 * s for s in early_std]

    fig.add_trace(go.Scatter(
        x=early_n, y=early_ratio,
        mode='lines+markers',
        name='Ratio (early)',
        line=dict(color='#9467bd', width=2),
        marker=dict(size=8),
        error_y=dict(type='data', array=early_ci, visible=True),
        showlegend=False,
    ), row=2, col=1)

    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=2, col=1)

    # Mark significant crossover
    if sig_crossover:
        fig.add_vline(x=sig_crossover, line_dash="dot", line_color="green", row=2, col=1)
        fig.add_annotation(
            x=sig_crossover, y=1.0,
            text=f"Significant at N={sig_crossover}",
            showarrow=True, arrowhead=2,
            ax=30, ay=-40,
            row=2, col=1
        )

    # Panel 4: Variance analysis - show std over context length
    fig.add_trace(go.Scatter(
        x=plot_n, y=results['ratio_std'],
        mode='lines+markers',
        name='Std Dev of Ratio',
        line=dict(color='#2ca02c', width=2),
        marker=dict(size=6),
        showlegend=False,
    ), row=2, col=2)

    # Coefficient of variation
    cv = [s / m if m > 0 else np.nan for s, m in zip(results['ratio_std'], results['ratio_mean'])]
    fig.add_trace(go.Scatter(
        x=plot_n, y=cv,
        mode='lines+markers',
        name='CV (std/mean)',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        marker=dict(size=6),
        yaxis='y2',
        showlegend=False,
    ), row=2, col=2)

    # Update axes
    fig.update_xaxes(title_text="Context Length (N)", type="log", row=1, col=1)
    fig.update_xaxes(title_text="Context Length (N)", type="log", row=1, col=2)
    fig.update_xaxes(title_text="Context Length (N)", row=2, col=1)  # Linear for early
    fig.update_xaxes(title_text="Context Length (N)", type="log", row=2, col=2)

    fig.update_yaxes(title_text="L2 Distance", type="log", row=1, col=1)
    fig.update_yaxes(title_text="Ratio (sem/graph)", row=1, col=2)
    fig.update_yaxes(title_text="Ratio (sem/graph)", row=2, col=1)
    fig.update_yaxes(title_text="Std Dev", row=2, col=2)

    fig.update_layout(
        title="Semantic Override Trajectory with Error Bars<br>" +
              f"<sup>n={n_trials} trials | Crossover: stable at N={stable_crossover}, significant at N={sig_crossover}</sup>",
        width=1200,
        height=900,
        legend=dict(x=0.02, y=0.98),
    )

    output_dir = Path("results/semantic_override_trajectory")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig.write_html(output_dir / "semantic_override_with_variance.html")
    print(f"Saved: {output_dir / 'semantic_override_with_variance.html'}")

    # Save numerical results
    import json
    with open(output_dir / "variance_results.json", "w") as f:
        json.dump({
            'context_lengths': [int(x) for x in results['context_lengths']],
            'ratio_mean': [float(x) for x in results['ratio_mean']],
            'ratio_std': [float(x) for x in results['ratio_std']],
            'l2_sem_mean': [float(x) for x in results['l2_sem_mean']],
            'l2_sem_std': [float(x) for x in results['l2_sem_std']],
            'l2_graph_mean': [float(x) for x in results['l2_graph_mean']],
            'l2_graph_std': [float(x) for x in results['l2_graph_std']],
            'n_trials': n_trials,
            'stable_crossover': stable_crossover,
            'sig_crossover': sig_crossover,
            'peak_n': int(peak_n),
            'peak_ratio': float(peak_ratio),
        }, f, indent=2)
    print(f"Saved: {output_dir / 'variance_results.json'}")

    # Log to W&B
    print("\nLogging to W&B...")

    run = wandb.init(
        project='icl-structural-influence',
        name='semantic-override-with-variance',
        tags=['semantic-override', 'variance', 'error-bars', 'trajectory'],
        config={
            'n_trials': n_trials,
            'context_lengths': results['context_lengths'],
            'max_context': 10000,
            'n_samples_per_trial': 50,
        }
    )

    wandb.log({
        'semantic_override_variance_plot': wandb.Html(
            open(output_dir / 'semantic_override_with_variance.html').read()
        )
    })

    # Log table with all trials
    table = wandb.Table(columns=[
        'N', 'Ratio_Mean', 'Ratio_Std', 'Ratio_CI95',
        'L2_Sem_Mean', 'L2_Sem_Std', 'L2_Graph_Mean', 'L2_Graph_Std',
        'Winner', 'Significant'
    ])
    for i, n in enumerate(results['context_lengths']):
        winner = 'Graph' if results['ratio_mean'][i] > 1 else 'Semantic'
        significant = 'Yes' if results['ratio_mean'][i] - results['ratio_std'][i] > 1 else 'No'
        table.add_data(
            n,
            results['ratio_mean'][i],
            results['ratio_std'][i],
            2 * results['ratio_std'][i],
            results['l2_sem_mean'][i],
            results['l2_sem_std'][i],
            results['l2_graph_mean'][i],
            results['l2_graph_std'][i],
            winner,
            significant
        )
    wandb.log({'variance_table': table})

    # Summary
    wandb.summary['n_trials'] = n_trials
    wandb.summary['pretrained_ratio_mean'] = results['ratio_mean'][0]
    wandb.summary['stable_crossover_N'] = stable_crossover
    wandb.summary['significant_crossover_N'] = sig_crossover
    wandb.summary['peak_separation_N'] = int(peak_n)
    wandb.summary['peak_separation_ratio'] = float(peak_ratio)
    wandb.summary['max_variance_N'] = results['context_lengths'][np.argmax(results['ratio_std'])]
    wandb.summary['max_variance_std'] = max(results['ratio_std'])

    print(f"\nRun URL: {run.url}")
    wandb.finish()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Pretrained ratio: {results['ratio_mean'][0]:.3f} ± {results['ratio_std'][0]:.3f}")
    print(f"Stable crossover: N={stable_crossover}")
    print(f"Statistically significant crossover: N={sig_crossover}")
    print(f"Peak separation: N={peak_n}, ratio={peak_ratio:.3f} ± {results['ratio_std'][peak_idx]:.3f}")
    print(f"Highest variance at: N={results['context_lengths'][np.argmax(results['ratio_std'])]}")
    print(f"  (std={max(results['ratio_std']):.3f})")
    print("=" * 70)


if __name__ == "__main__":
    main()
