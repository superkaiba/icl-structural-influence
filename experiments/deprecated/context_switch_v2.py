#!/usr/bin/env python3
"""
Context Switching Experiment v2:
- Two RANDOM graphs (no semantic alignment)
- Fixed N1=1000 tokens from Graph1
- More trials for robust statistics
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


class RandomDualGraphExperiment:
    """Two graphs with SAME tokens but RANDOM (non-semantic) cluster assignments."""

    def __init__(self, seed=42):
        np.random.seed(seed)

        # Use 16 arbitrary tokens (no semantic relationship)
        self.tokens = [
            'alpha', 'beta', 'gamma', 'delta',
            'echo', 'foxtrot', 'golf', 'hotel',
            'india', 'juliet', 'kilo', 'lima',
            'mike', 'november', 'oscar', 'papa'
        ]

        # Graph 1: Random clustering
        shuffled1 = self.tokens.copy()
        np.random.shuffle(shuffled1)
        self.graph1_clusters = {
            0: shuffled1[0:4],
            1: shuffled1[4:8],
            2: shuffled1[8:12],
            3: shuffled1[12:16]
        }

        # Graph 2: Different random clustering
        shuffled2 = self.tokens.copy()
        np.random.shuffle(shuffled2)
        # Ensure it's actually different
        while self._same_clustering(shuffled1, shuffled2):
            np.random.shuffle(shuffled2)

        self.graph2_clusters = {
            0: shuffled2[0:4],
            1: shuffled2[4:8],
            2: shuffled2[8:12],
            3: shuffled2[12:16]
        }

        # Build adjacency for each graph
        self.graph1_adj = self._build_adjacency(self.graph1_clusters)
        self.graph2_adj = self._build_adjacency(self.graph2_clusters)

        # Token to cluster mapping
        self.token_to_cluster1 = {}
        for cid, toks in self.graph1_clusters.items():
            for t in toks:
                self.token_to_cluster1[t] = cid

        self.token_to_cluster2 = {}
        for cid, toks in self.graph2_clusters.items():
            for t in toks:
                self.token_to_cluster2[t] = cid

        # Count conflicts (tokens in same cluster in G1 but different in G2)
        g1_pairs = set(self._get_same_cluster_pairs(self.graph1_clusters))
        g2_pairs = set(self._get_same_cluster_pairs(self.graph2_clusters))
        conflicts = len(g1_pairs - g2_pairs)

        print("Graph 1 clusters (random):")
        for cid, toks in self.graph1_clusters.items():
            print(f"  Cluster {cid}: {toks}")

        print("\nGraph 2 clusters (random):")
        for cid, toks in self.graph2_clusters.items():
            print(f"  Cluster {cid}: {toks}")

        print(f"\nG1 same-cluster pairs: {len(g1_pairs)}")
        print(f"G2 same-cluster pairs: {len(g2_pairs)}")
        print(f"Conflicting pairs (same in G1, different in G2): {conflicts}")

    def _same_clustering(self, shuffle1, shuffle2):
        """Check if two shuffles produce the same clustering."""
        c1 = {frozenset(shuffle1[i:i+4]) for i in range(0, 16, 4)}
        c2 = {frozenset(shuffle2[i:i+4]) for i in range(0, 16, 4)}
        return c1 == c2

    def _get_same_cluster_pairs(self, clusters):
        """Get all pairs that are in the same cluster."""
        pairs = []
        for cid, toks in clusters.items():
            for i, t1 in enumerate(toks):
                for t2 in toks[i+1:]:
                    pairs.append((t1, t2) if t1 < t2 else (t2, t1))
        return pairs

    def _build_adjacency(self, clusters, p_in=0.9, p_out=0.1):
        """Build adjacency matrix with SBM structure."""
        adj = {}
        token_to_cluster = {}
        for cid, toks in clusters.items():
            for t in toks:
                token_to_cluster[t] = cid

        for t1 in self.tokens:
            adj[t1] = {}
            c1 = token_to_cluster[t1]
            for t2 in self.tokens:
                if t1 == t2:
                    continue
                c2 = token_to_cluster[t2]
                adj[t1][t2] = p_in if c1 == c2 else p_out

        return adj

    def generate_walk(self, adj, length, start=None):
        """Generate random walk on given graph."""
        if start is None:
            start = np.random.choice(self.tokens)

        walk = [start]
        current = start

        for _ in range(length - 1):
            neighbors = list(adj[current].keys())
            weights = [adj[current][n] for n in neighbors]
            weights = np.array(weights) / sum(weights)
            current = np.random.choice(neighbors, p=weights)
            walk.append(current)

        return walk

    def get_graph1_pairs(self):
        """Pairs that are same-cluster in Graph1."""
        return self._get_same_cluster_pairs(self.graph1_clusters)

    def get_graph2_pairs(self):
        """Pairs that are same-cluster in Graph2."""
        return self._get_same_cluster_pairs(self.graph2_clusters)


def collect_reps_from_context(model, tokenizer, walk1, walk2, layer_idx=-5):
    """
    Collect representations from context [walk1] + [walk2].
    Returns representations of tokens from walk2 portion only.
    """
    full_context = ' '.join(walk1 + walk2)
    tokens = tokenizer.encode(full_context, add_special_tokens=False)
    input_ids = torch.tensor([tokens]).to(model.device)

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        hidden = outputs.hidden_states[layer_idx][0]

    # Get representations from walk2 portion only
    n1 = len(walk1)
    token_reps = defaultdict(list)

    all_words = walk1 + walk2
    for pos, word in enumerate(all_words):
        if pos >= n1 and pos < hidden.shape[0]:
            token_reps[word].append(hidden[pos].cpu().float().numpy())

    return {t: np.mean(r, axis=0) for t, r in token_reps.items() if r}


def compute_pair_distances(reps, pairs):
    """Compute mean L2 distance for pairs."""
    dists = [np.linalg.norm(reps[t1] - reps[t2])
             for t1, t2 in pairs if t1 in reps and t2 in reps]
    return np.mean(dists) if dists else np.nan


def main():
    print("=" * 70)
    print("CONTEXT SWITCHING v2: Two Random Graphs, N1=1000, 50 trials")
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

    # Create experiment
    experiment = RandomDualGraphExperiment(seed=42)

    g1_pairs = experiment.get_graph1_pairs()
    g2_pairs = experiment.get_graph2_pairs()

    # Parameters
    n1 = 1000  # Fixed Graph1 context
    n2_values = [1, 2, 5, 10, 20, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000]
    n_trials = 50
    layer_idx = -5

    results = {
        'n1': n1,
        'n2_values': n2_values,
        'n_trials': n_trials,
        'data': {}
    }

    print(f"\nFixed N1 = {n1} (Graph1 context)")
    print(f"Varying N2 from {n2_values[0]} to {n2_values[-1]}")
    print(f"Trials per N2: {n_trials}")
    print("=" * 70)

    for n2 in n2_values:
        print(f"N2={n2}...", end=" ", flush=True)

        trial_g1_dists = []
        trial_g2_dists = []
        trial_ratios = []

        for trial in range(n_trials):
            np.random.seed(42 + trial * 1000 + n2)

            walk1 = experiment.generate_walk(experiment.graph1_adj, n1)
            walk2 = experiment.generate_walk(experiment.graph2_adj, n2)

            reps = collect_reps_from_context(model, tokenizer, walk1, walk2, layer_idx)

            if len(reps) < 2:
                continue

            g1_dist = compute_pair_distances(reps, g1_pairs)
            g2_dist = compute_pair_distances(reps, g2_pairs)

            if not np.isnan(g1_dist) and not np.isnan(g2_dist) and g2_dist > 0:
                trial_g1_dists.append(g1_dist)
                trial_g2_dists.append(g2_dist)
                trial_ratios.append(g1_dist / g2_dist)

        if trial_ratios:
            mean_ratio = np.mean(trial_ratios)
            std_ratio = np.std(trial_ratios)
            sem_ratio = std_ratio / np.sqrt(len(trial_ratios))

            results['data'][n2] = {
                'g1_dist_mean': float(np.mean(trial_g1_dists)),
                'g1_dist_std': float(np.std(trial_g1_dists)),
                'g2_dist_mean': float(np.mean(trial_g2_dists)),
                'g2_dist_std': float(np.std(trial_g2_dists)),
                'ratio_mean': float(mean_ratio),
                'ratio_std': float(std_ratio),
                'ratio_sem': float(sem_ratio),
                'n_valid_trials': len(trial_ratios),
                'ratio_trials': [float(r) for r in trial_ratios],
            }

            winner = "G2" if mean_ratio > 1 else "G1"
            sig = "*" if abs(mean_ratio - 1) > 2 * sem_ratio else ""
            print(f"ratio={mean_ratio:.3f}±{sem_ratio:.3f} [{winner}]{sig}")
        else:
            print("no valid trials")
            results['data'][n2] = None

    # Find crossover
    print("\n" + "=" * 70)
    print("CROSSOVER ANALYSIS")
    print("=" * 70)

    # First N2 where ratio > 1
    first_cross = None
    for n2 in n2_values:
        if results['data'][n2] and results['data'][n2]['ratio_mean'] > 1:
            first_cross = n2
            break

    # First N2 where ratio significantly > 1 (mean - 2*SEM > 1)
    sig_cross = None
    for n2 in n2_values:
        if results['data'][n2]:
            mean = results['data'][n2]['ratio_mean']
            sem = results['data'][n2]['ratio_sem']
            if mean - 2 * sem > 1:
                sig_cross = n2
                break

    # Stable crossover (3 consecutive)
    stable_cross = None
    for i, n2 in enumerate(n2_values[:-2]):
        if all(results['data'][n2_values[i+j]] and
               results['data'][n2_values[i+j]]['ratio_mean'] > 1
               for j in range(3)):
            stable_cross = n2
            break

    print(f"First crossover (ratio > 1): N2 = {first_cross}")
    print(f"Significant crossover (95% CI): N2 = {sig_cross}")
    print(f"Stable crossover (3 consecutive): N2 = {stable_cross}")
    if first_cross:
        print(f"Crossover ratio (N2/N1): {first_cross}/{n1} = {first_cross/n1:.3f}")

    # Visualization
    print("\n" + "=" * 70)
    print("Creating visualization...")
    print("=" * 70)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f"Ratio (G1-dist / G2-dist) vs N2 | N1={n1} fixed",
            "Individual Trial Distribution",
            "L2 Distances: G1-pairs vs G2-pairs",
            "Ratio with 95% CI"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    valid_n2 = [n2 for n2 in n2_values if results['data'][n2]]
    ratios = [results['data'][n2]['ratio_mean'] for n2 in valid_n2]
    ratio_stds = [results['data'][n2]['ratio_std'] for n2 in valid_n2]
    ratio_sems = [results['data'][n2]['ratio_sem'] for n2 in valid_n2]
    g1_dists = [results['data'][n2]['g1_dist_mean'] for n2 in valid_n2]
    g2_dists = [results['data'][n2]['g2_dist_mean'] for n2 in valid_n2]

    # Panel 1: Main ratio plot with error bars
    fig.add_trace(go.Scatter(
        x=valid_n2, y=ratios,
        mode='lines+markers',
        name='Ratio (mean)',
        line=dict(color='#9467bd', width=3),
        marker=dict(size=10),
        error_y=dict(type='data', array=ratio_sems, visible=True),
    ), row=1, col=1)

    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=1, col=1)

    if first_cross:
        fig.add_vline(x=first_cross, line_dash="dot", line_color="green", row=1, col=1)

    # Panel 2: Box plots of trial distributions
    for n2 in [10, 50, 100, 200, 500, 1000]:
        if results['data'].get(n2) and results['data'][n2].get('ratio_trials'):
            fig.add_trace(go.Box(
                y=results['data'][n2]['ratio_trials'],
                name=f'N2={n2}',
                boxpoints='outliers',
            ), row=1, col=2)

    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=1, col=2)

    # Panel 3: G1 and G2 distances
    fig.add_trace(go.Scatter(
        x=valid_n2, y=g1_dists,
        mode='lines+markers',
        name='G1-pair dist',
        line=dict(color='#d62728', width=2),
        marker=dict(size=6),
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=valid_n2, y=g2_dists,
        mode='lines+markers',
        name='G2-pair dist',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6),
    ), row=2, col=1)

    # Panel 4: Ratio with 95% CI band
    ci_upper = [r + 2*s for r, s in zip(ratios, ratio_sems)]
    ci_lower = [r - 2*s for r, s in zip(ratios, ratio_sems)]

    fig.add_trace(go.Scatter(
        x=valid_n2 + valid_n2[::-1],
        y=ci_upper + ci_lower[::-1],
        fill='toself',
        fillcolor='rgba(148, 103, 189, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% CI',
    ), row=2, col=2)

    fig.add_trace(go.Scatter(
        x=valid_n2, y=ratios,
        mode='lines',
        name='Mean',
        line=dict(color='#9467bd', width=2),
    ), row=2, col=2)

    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=2, col=2)

    # Update axes
    fig.update_xaxes(title_text="Graph2 Context Length (N2)", type="log", row=1, col=1)
    fig.update_xaxes(title_text="N2", row=1, col=2)
    fig.update_xaxes(title_text="Graph2 Context Length (N2)", type="log", row=2, col=1)
    fig.update_xaxes(title_text="Graph2 Context Length (N2)", type="log", row=2, col=2)

    fig.update_yaxes(title_text="Ratio (G1-dist / G2-dist)", row=1, col=1)
    fig.update_yaxes(title_text="Ratio", row=1, col=2)
    fig.update_yaxes(title_text="L2 Distance", row=2, col=1)
    fig.update_yaxes(title_text="Ratio", row=2, col=2)

    fig.update_layout(
        title=f"Context Switching: N1={n1} (Graph1) + N2 (Graph2)<br>" +
              f"<sup>Two random graphs, {n_trials} trials | Crossover at N2={first_cross} (ratio={first_cross/n1:.2f})</sup>",
        width=1200,
        height=900,
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
    )

    output_dir = Path("results/context_switch")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig.write_html(output_dir / "context_switch_v2.html")
    print(f"Saved: {output_dir / 'context_switch_v2.html'}")

    # Save JSON
    json_results = {
        'n1': n1,
        'n2_values': n2_values,
        'n_trials': n_trials,
        'data': {str(k): v for k, v in results['data'].items()},
        'first_crossover': first_cross,
        'sig_crossover': sig_cross,
        'stable_crossover': stable_cross,
        'crossover_ratio': first_cross / n1 if first_cross else None,
    }

    with open(output_dir / "context_switch_v2_results.json", "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"Saved: {output_dir / 'context_switch_v2_results.json'}")

    # Log to W&B
    print("\nLogging to W&B...")

    run = wandb.init(
        project='icl-structural-influence',
        name='context-switch-v2-random-graphs',
        tags=['context-switch', 'random-graphs', 'N1=1000', 'high-trials'],
        config={
            'n1': n1,
            'n2_values': n2_values,
            'n_trials': n_trials,
            'layer_idx': layer_idx,
        }
    )

    wandb.log({
        'context_switch_v2_plot': wandb.Html(open(output_dir / 'context_switch_v2.html').read())
    })

    wandb.summary['first_crossover_N2'] = first_cross
    wandb.summary['sig_crossover_N2'] = sig_cross
    wandb.summary['stable_crossover_N2'] = stable_cross
    wandb.summary['crossover_ratio'] = first_cross / n1 if first_cross else None
    wandb.summary['n1'] = n1

    print(f"\nRun URL: {run.url}")
    wandb.finish()

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Graph1 context: N1 = {n1}")
    print(f"Trials per N2: {n_trials}")
    print(f"\nCrossover points:")
    print(f"  First (ratio > 1):     N2 = {first_cross}")
    print(f"  Significant (95% CI):  N2 = {sig_cross}")
    print(f"  Stable (3 consecutive): N2 = {stable_cross}")
    if first_cross:
        print(f"\nTo override {n1} tokens of Graph1, need ~{first_cross} tokens of Graph2")
        print(f"Ratio: {first_cross/n1:.1%} of original context")
    print("=" * 70)


if __name__ == "__main__":
    main()
