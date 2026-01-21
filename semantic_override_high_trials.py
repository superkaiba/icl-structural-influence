#!/usr/bin/env python3
"""
Semantic override trajectory with many independent trials for robust statistics.
Focus on N=0 to N=3000 with dense sampling and high trial count.
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
    print("SEMANTIC OVERRIDE: HIGH-TRIAL ANALYSIS (N=0 to 3000)")
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

    # Context lengths: dense at early N, sparser later
    context_lengths = (
        [0] +  # Pretrained
        list(range(1, 51)) +  # Every 1 from 1-50
        list(range(55, 101, 5)) +  # Every 5 from 55-100
        list(range(125, 251, 25)) +  # Every 25 from 125-250
        list(range(300, 501, 50)) +  # Every 50 from 300-500
        list(range(600, 1001, 100)) +  # Every 100 from 600-1000
        list(range(1250, 3001, 250))  # Every 250 from 1250-3000
    )

    n_trials = 20  # More trials for better statistics

    results = {
        'context_lengths': [],
        'ratio_mean': [],
        'ratio_std': [],
        'ratio_sem': [],  # Standard error of mean
        'ratio_trials': [],
        'l2_sem_mean': [],
        'l2_sem_std': [],
        'l2_graph_mean': [],
        'l2_graph_std': [],
    }

    print(f"\nCollecting data for {len(context_lengths)} context lengths, {n_trials} trials each...")
    print(f"Total forward passes: ~{len(context_lengths) * n_trials * 50}")

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

        mean_ratio = np.nanmean(trial_ratios)
        std_ratio = np.nanstd(trial_ratios)
        sem_ratio = std_ratio / np.sqrt(n_trials)  # Standard error

        results['context_lengths'].append(n)
        results['ratio_mean'].append(float(mean_ratio))
        results['ratio_std'].append(float(std_ratio))
        results['ratio_sem'].append(float(sem_ratio))
        results['ratio_trials'].append([float(r) for r in trial_ratios])
        results['l2_sem_mean'].append(float(np.nanmean(trial_sem)))
        results['l2_sem_std'].append(float(np.nanstd(trial_sem)))
        results['l2_graph_mean'].append(float(np.nanmean(trial_graph)))
        results['l2_graph_std'].append(float(np.nanstd(trial_graph)))

        winner = 'Graph' if mean_ratio > 1 else 'Semantic'
        print(f"ratio={mean_ratio:.3f}±{std_ratio:.3f} (SEM={sem_ratio:.3f}) [{winner}]")

    # Statistical analysis
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)

    # Find key crossover points
    # 1. First time mean > 1
    first_mean_cross = None
    for i, r in enumerate(results['ratio_mean']):
        if r > 1:
            first_mean_cross = results['context_lengths'][i]
            break

    # 2. First stable crossover (5 consecutive graph wins)
    stable_5 = None
    for i in range(len(results['ratio_mean']) - 4):
        if all(r > 1 for r in results['ratio_mean'][i:i+5]):
            stable_5 = results['context_lengths'][i]
            break

    # 3. First statistically significant (mean - 2*SEM > 1, ~95% CI)
    sig_95 = None
    for i, (mean, sem) in enumerate(zip(results['ratio_mean'], results['ratio_sem'])):
        if mean - 2*sem > 1:
            sig_95 = results['context_lengths'][i]
            break

    # 4. First very significant (mean - 3*SEM > 1, ~99% CI)
    sig_99 = None
    for i, (mean, sem) in enumerate(zip(results['ratio_mean'], results['ratio_sem'])):
        if mean - 3*sem > 1:
            sig_99 = results['context_lengths'][i]
            break

    # 5. Peak separation
    peak_idx = np.nanargmax(results['ratio_mean'])
    peak_n = results['context_lengths'][peak_idx]
    peak_ratio = results['ratio_mean'][peak_idx]
    peak_sem = results['ratio_sem'][peak_idx]

    # 6. Count oscillations in early region (N=1-50)
    early_mask = [n <= 50 for n in results['context_lengths']]
    early_ratios = [results['ratio_mean'][i] for i in range(len(early_mask)) if early_mask[i] and results['context_lengths'][i] > 0]
    oscillations = sum(1 for i in range(1, len(early_ratios)) if (early_ratios[i] > 1) != (early_ratios[i-1] > 1))

    print(f"\nPretrained ratio: {results['ratio_mean'][0]:.4f}")
    print(f"\nCROSSOVER POINTS:")
    print(f"  First mean > 1: N={first_mean_cross}")
    print(f"  Stable (5 consecutive): N={stable_5}")
    print(f"  95% significant (mean - 2*SEM > 1): N={sig_95}")
    print(f"  99% significant (mean - 3*SEM > 1): N={sig_99}")
    print(f"\nPEAK SEPARATION:")
    print(f"  N={peak_n}, ratio={peak_ratio:.3f} ± {peak_sem:.3f} (SEM)")
    print(f"\nEARLY DYNAMICS (N=1-50):")
    print(f"  Oscillations across threshold: {oscillations}")

    # Create visualization
    print("\n" + "=" * 70)
    print("Creating visualization...")
    print("=" * 70)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f"Ratio Trajectory (n={n_trials} trials, ±2 SEM)",
            "Early Context Detail (N=0-100)",
            "Individual Trial Distribution",
            "Variance and Confidence Over N"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # Use N=0.5 for plotting pretrained (so it shows on log scale)
    plot_n = [0.5 if n == 0 else n for n in results['context_lengths']]

    # Panel 1: Main trajectory with 95% CI (2*SEM)
    ci_95 = [2 * s for s in results['ratio_sem']]

    fig.add_trace(go.Scatter(
        x=plot_n, y=results['ratio_mean'],
        mode='lines',
        name='Mean ratio',
        line=dict(color='#9467bd', width=3),
    ), row=1, col=1)

    # Confidence band
    upper = [m + c for m, c in zip(results['ratio_mean'], ci_95)]
    lower = [m - c for m, c in zip(results['ratio_mean'], ci_95)]

    fig.add_trace(go.Scatter(
        x=plot_n + plot_n[::-1],
        y=upper + lower[::-1],
        fill='toself',
        fillcolor='rgba(148, 103, 189, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% CI',
        showlegend=True,
    ), row=1, col=1)

    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=1, col=1)

    # Mark key points
    if sig_95:
        fig.add_vline(x=sig_95, line_dash="dot", line_color="green", row=1, col=1)
        idx_95 = results['context_lengths'].index(sig_95)
        fig.add_annotation(
            x=np.log10(sig_95), y=results['ratio_mean'][idx_95],
            text=f"95% sig: N={sig_95}",
            showarrow=True, arrowhead=2,
            ax=50, ay=-30,
            row=1, col=1
        )

    # Panel 2: Early detail (N=0-100, linear scale)
    early_mask = [n <= 100 for n in results['context_lengths']]
    early_n = [results['context_lengths'][i] for i in range(len(early_mask)) if early_mask[i]]
    early_ratio = [results['ratio_mean'][i] for i in range(len(early_mask)) if early_mask[i]]
    early_ci = [2 * results['ratio_sem'][i] for i in range(len(early_mask)) if early_mask[i]]

    fig.add_trace(go.Scatter(
        x=early_n, y=early_ratio,
        mode='lines+markers',
        name='Mean (early)',
        line=dict(color='#9467bd', width=2),
        marker=dict(size=5),
        error_y=dict(type='data', array=early_ci, visible=True, thickness=1),
        showlegend=False,
    ), row=1, col=2)

    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=1, col=2)

    # Color regions
    fig.add_vrect(x0=0, x1=first_mean_cross if first_mean_cross else 100,
                  fillcolor="rgba(214, 39, 40, 0.1)", line_width=0, row=1, col=2)

    # Panel 3: Violin/box plot at key context lengths
    key_ns = [1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 3000]
    for n in key_ns:
        if n in results['context_lengths']:
            idx = results['context_lengths'].index(n)
            trials = results['ratio_trials'][idx]
            fig.add_trace(go.Box(
                y=trials,
                name=f'N={n}',
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8,
                marker=dict(size=3),
                showlegend=False,
            ), row=2, col=1)

    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=2, col=1)

    # Panel 4: Variance analysis
    fig.add_trace(go.Scatter(
        x=plot_n, y=results['ratio_std'],
        mode='lines',
        name='Std Dev',
        line=dict(color='#2ca02c', width=2),
    ), row=2, col=2)

    fig.add_trace(go.Scatter(
        x=plot_n, y=results['ratio_sem'],
        mode='lines',
        name='SEM',
        line=dict(color='#ff7f0e', width=2),
    ), row=2, col=2)

    # Update axes
    fig.update_xaxes(title_text="Context Length (N)", type="log", row=1, col=1)
    fig.update_xaxes(title_text="Context Length (N)", row=1, col=2)
    fig.update_xaxes(title_text="Context Length", row=2, col=1)
    fig.update_xaxes(title_text="Context Length (N)", type="log", row=2, col=2)

    fig.update_yaxes(title_text="Ratio (sem/graph)", row=1, col=1)
    fig.update_yaxes(title_text="Ratio (sem/graph)", row=1, col=2)
    fig.update_yaxes(title_text="Ratio", row=2, col=1)
    fig.update_yaxes(title_text="Variance", row=2, col=2)

    fig.update_layout(
        title=f"Semantic Override Analysis: N=0 to 3000, {n_trials} Independent Trials<br>" +
              f"<sup>95% sig at N={sig_95} | Peak at N={peak_n} (ratio={peak_ratio:.2f})</sup>",
        width=1400,
        height=1000,
        legend=dict(x=0.02, y=0.98),
    )

    output_dir = Path("results/unrelated_tokens_layerwise")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig.write_html(output_dir / "semantic_override_high_trials.html")
    print(f"Saved: {output_dir / 'semantic_override_high_trials.html'}")

    # Save JSON
    with open(output_dir / "high_trials_results.json", "w") as f:
        json.dump({
            'context_lengths': results['context_lengths'],
            'ratio_mean': results['ratio_mean'],
            'ratio_std': results['ratio_std'],
            'ratio_sem': results['ratio_sem'],
            'ratio_trials': results['ratio_trials'],
            'l2_sem_mean': results['l2_sem_mean'],
            'l2_sem_std': results['l2_sem_std'],
            'l2_graph_mean': results['l2_graph_mean'],
            'l2_graph_std': results['l2_graph_std'],
            'n_trials': n_trials,
            'first_mean_cross': first_mean_cross,
            'stable_5_cross': stable_5,
            'sig_95_cross': sig_95,
            'sig_99_cross': sig_99,
            'peak_n': peak_n,
            'peak_ratio': peak_ratio,
            'peak_sem': peak_sem,
            'early_oscillations': oscillations,
        }, f, indent=2)
    print(f"Saved: {output_dir / 'high_trials_results.json'}")

    # Log to W&B
    print("\nLogging to W&B...")

    run = wandb.init(
        project='icl-structural-influence',
        name='semantic-override-high-trials',
        tags=['semantic-override', 'high-trials', 'statistics', 'n=0-3000'],
        config={
            'n_trials': n_trials,
            'max_context': 3000,
            'n_context_lengths': len(context_lengths),
            'n_samples_per_trial': 50,
        }
    )

    wandb.log({
        'semantic_override_high_trials_plot': wandb.Html(
            open(output_dir / 'semantic_override_high_trials.html').read()
        )
    })

    # Log detailed table
    table = wandb.Table(columns=[
        'N', 'Ratio_Mean', 'Ratio_Std', 'Ratio_SEM', 'CI_95_Lower', 'CI_95_Upper',
        'Winner', 'Sig_95', 'Sig_99'
    ])
    for i, n in enumerate(results['context_lengths']):
        mean = results['ratio_mean'][i]
        sem = results['ratio_sem'][i]
        winner = 'Graph' if mean > 1 else 'Semantic'
        sig_95_flag = 'Yes' if mean - 2*sem > 1 else 'No'
        sig_99_flag = 'Yes' if mean - 3*sem > 1 else 'No'
        table.add_data(
            n, mean, results['ratio_std'][i], sem,
            mean - 2*sem, mean + 2*sem,
            winner, sig_95_flag, sig_99_flag
        )
    wandb.log({'high_trials_table': table})

    # Summary
    wandb.summary['n_trials'] = n_trials
    wandb.summary['pretrained_ratio'] = results['ratio_mean'][0]
    wandb.summary['first_mean_crossover'] = first_mean_cross
    wandb.summary['stable_5_crossover'] = stable_5
    wandb.summary['sig_95_crossover'] = sig_95
    wandb.summary['sig_99_crossover'] = sig_99
    wandb.summary['peak_N'] = peak_n
    wandb.summary['peak_ratio'] = peak_ratio
    wandb.summary['peak_SEM'] = peak_sem
    wandb.summary['early_oscillations'] = oscillations

    print(f"\nRun URL: {run.url}")
    wandb.finish()

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Trials per context length: {n_trials}")
    print(f"Context lengths tested: {len(context_lengths)}")
    print(f"\nPRETRAINED BASELINE:")
    print(f"  Ratio = {results['ratio_mean'][0]:.4f} (semantic ~3x closer)")
    print(f"\nCROSSOVER ANALYSIS:")
    print(f"  First mean > 1:        N = {first_mean_cross}")
    print(f"  Stable (5 consecutive): N = {stable_5}")
    print(f"  95% significant:       N = {sig_95}")
    print(f"  99% significant:       N = {sig_99}")
    print(f"\nPEAK SEPARATION:")
    print(f"  N = {peak_n}, ratio = {peak_ratio:.3f} ± {peak_sem:.3f}")
    print(f"\nEARLY DYNAMICS:")
    print(f"  Oscillations (N=1-50): {oscillations} crossings")
    print("=" * 70)


if __name__ == "__main__":
    main()
