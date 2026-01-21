#!/usr/bin/env python3
"""
Context Switching Experiment:
When we have [Walk from Graph1] + [Walk from Graph2] in context,
at what point do tokens in Graph2 stop being influenced by Graph1's structure?

Tests how quickly the model "forgets" the first graph structure when
a new conflicting structure is introduced.
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


class DualGraphExperiment:
    """Two graphs with SAME tokens but DIFFERENT cluster assignments."""

    def __init__(self, seed=42):
        np.random.seed(seed)

        # Use 12 tokens that can be clustered differently
        self.tokens = [
            'apple', 'orange', 'banana', 'grape',
            'hammer', 'wrench', 'drill', 'saw',
            'shirt', 'pants', 'jacket', 'shoes'
        ]

        # Graph 1: Cluster by semantic category (fruit, tools, clothes)
        self.graph1_clusters = {
            0: ['apple', 'orange', 'banana', 'grape'],    # Fruits
            1: ['hammer', 'wrench', 'drill', 'saw'],       # Tools
            2: ['shirt', 'pants', 'jacket', 'shoes']       # Clothes
        }

        # Graph 2: Random clustering (conflicts with semantic)
        np.random.shuffle(self.tokens)
        shuffled = self.tokens.copy()
        self.graph2_clusters = {
            0: shuffled[0:4],
            1: shuffled[4:8],
            2: shuffled[8:12]
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

        print("Graph 1 clusters (semantic):")
        for cid, toks in self.graph1_clusters.items():
            print(f"  Cluster {cid}: {toks}")

        print("\nGraph 2 clusters (random):")
        for cid, toks in self.graph2_clusters.items():
            print(f"  Cluster {cid}: {toks}")

    def _build_adjacency(self, clusters, p_in=0.9, p_out=0.1):
        """Build adjacency matrix with SBM structure."""
        adj = {}
        for t1 in self.tokens:
            adj[t1] = {}
            c1 = None
            for cid, toks in clusters.items():
                if t1 in toks:
                    c1 = cid
                    break

            for t2 in self.tokens:
                if t1 == t2:
                    continue
                c2 = None
                for cid, toks in clusters.items():
                    if t2 in toks:
                        c2 = cid
                        break

                # Higher weight for same cluster
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

    def generate_context_switch(self, n1, n2):
        """Generate context: [n1 tokens from graph1] + [n2 tokens from graph2]."""
        walk1 = self.generate_walk(self.graph1_adj, n1)
        walk2 = self.generate_walk(self.graph2_adj, n2)
        return walk1, walk2

    def get_graph1_pairs(self):
        """Pairs that are same-cluster in Graph1."""
        pairs = []
        for cid, toks in self.graph1_clusters.items():
            for i, t1 in enumerate(toks):
                for t2 in toks[i+1:]:
                    pairs.append((t1, t2))
        return pairs

    def get_graph2_pairs(self):
        """Pairs that are same-cluster in Graph2."""
        pairs = []
        for cid, toks in self.graph2_clusters.items():
            for i, t1 in enumerate(toks):
                for t2 in toks[i+1:]:
                    pairs.append((t1, t2))
        return pairs


def collect_reps_from_context(model, tokenizer, walk1, walk2, layer_idx=-5):
    """
    Collect representations from a context of [walk1] + [walk2].
    Returns representations of tokens as they appear in walk2 portion.
    """
    # Full context
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
        if pos >= n1 and pos < hidden.shape[0]:  # Only walk2 portion
            token_reps[word].append(hidden[pos].cpu().float().numpy())

    return {t: np.mean(r, axis=0) for t, r in token_reps.items() if r}


def compute_pair_distances(reps, pairs):
    """Compute mean L2 distance for pairs."""
    dists = [np.linalg.norm(reps[t1] - reps[t2])
             for t1, t2 in pairs if t1 in reps and t2 in reps]
    return np.mean(dists) if dists else np.nan


def main():
    print("=" * 70)
    print("CONTEXT SWITCHING EXPERIMENT")
    print("When does Graph2 structure override Graph1 in context?")
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

    # Create dual graph experiment
    experiment = DualGraphExperiment(seed=42)

    # Get pairs for each graph
    g1_pairs = experiment.get_graph1_pairs()
    g2_pairs = experiment.get_graph2_pairs()

    print(f"\nGraph1 same-cluster pairs: {len(g1_pairs)}")
    print(f"Graph2 same-cluster pairs: {len(g2_pairs)}")

    # Experiment parameters
    # Fix n1 (graph1 context) and vary n2 (graph2 context)
    n1_values = [100, 500, 1000]  # Different amounts of Graph1 context
    n2_values = [1, 5, 10, 20, 50, 100, 200, 500, 1000]  # Graph2 context lengths

    n_trials = 10
    layer_idx = -5  # Late layer

    results = {
        'n1_values': n1_values,
        'n2_values': n2_values,
        'data': {}  # data[n1][n2] = {'g1_dist': ..., 'g2_dist': ..., 'ratio': ...}
    }

    for n1 in n1_values:
        results['data'][n1] = {}
        print(f"\n{'='*50}")
        print(f"Graph1 context length: N1 = {n1}")
        print(f"{'='*50}")

        for n2 in n2_values:
            print(f"  N2={n2}...", end=" ", flush=True)

            trial_g1_dists = []
            trial_g2_dists = []

            for trial in range(n_trials):
                np.random.seed(42 + trial * 1000 + n1 * 100 + n2)

                walk1, walk2 = experiment.generate_context_switch(n1, n2)
                reps = collect_reps_from_context(model, tokenizer, walk1, walk2, layer_idx)

                g1_dist = compute_pair_distances(reps, g1_pairs)
                g2_dist = compute_pair_distances(reps, g2_pairs)

                trial_g1_dists.append(g1_dist)
                trial_g2_dists.append(g2_dist)

            mean_g1 = np.nanmean(trial_g1_dists)
            mean_g2 = np.nanmean(trial_g2_dists)
            ratio = mean_g1 / mean_g2 if mean_g2 > 0 else np.nan

            results['data'][n1][n2] = {
                'g1_dist_mean': float(mean_g1),
                'g1_dist_std': float(np.nanstd(trial_g1_dists)),
                'g2_dist_mean': float(mean_g2),
                'g2_dist_std': float(np.nanstd(trial_g2_dists)),
                'ratio_mean': float(ratio),
                'ratio_std': float(np.nanstd([g1/g2 if g2 > 0 else np.nan
                                               for g1, g2 in zip(trial_g1_dists, trial_g2_dists)])),
            }

            winner = "Graph2" if ratio > 1 else "Graph1"
            print(f"G1-dist={mean_g1:.3f}, G2-dist={mean_g2:.3f}, ratio={ratio:.3f} [{winner} wins]")

    # Analysis
    print("\n" + "=" * 70)
    print("CROSSOVER ANALYSIS")
    print("=" * 70)
    print("(ratio > 1 means Graph2 pairs are closer = Graph2 structure learned)")

    for n1 in n1_values:
        crossover = None
        for n2 in n2_values:
            if results['data'][n1][n2]['ratio_mean'] > 1:
                crossover = n2
                break
        print(f"  N1={n1}: Graph2 wins at N2={crossover}")

    # Create visualization
    print("\n" + "=" * 70)
    print("Creating visualization...")
    print("=" * 70)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Ratio (G1-dist / G2-dist) by N2 for different N1",
            "Crossover Point: N2 needed to override N1 tokens of Graph1",
            "Graph1 Pair Distances (should increase as Graph2 dominates)",
            "Graph2 Pair Distances (should decrease as Graph2 dominates)"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    colors = ['#d62728', '#1f77b4', '#2ca02c']

    # Panel 1: Ratio curves
    for i, n1 in enumerate(n1_values):
        ratios = [results['data'][n1][n2]['ratio_mean'] for n2 in n2_values]
        stds = [results['data'][n1][n2]['ratio_std'] for n2 in n2_values]

        fig.add_trace(go.Scatter(
            x=n2_values, y=ratios,
            mode='lines+markers',
            name=f'N1={n1}',
            line=dict(color=colors[i], width=2),
            marker=dict(size=8),
            error_y=dict(type='data', array=stds, visible=True),
        ), row=1, col=1)

    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=1, col=1)

    # Panel 2: Crossover points
    crossovers = []
    for n1 in n1_values:
        crossover = None
        for n2 in n2_values:
            if results['data'][n1][n2]['ratio_mean'] > 1:
                crossover = n2
                break
        crossovers.append(crossover if crossover else n2_values[-1])

    fig.add_trace(go.Bar(
        x=[f'N1={n1}' for n1 in n1_values],
        y=crossovers,
        marker_color=colors,
        text=[f'N2={c}' for c in crossovers],
        textposition='auto',
    ), row=1, col=2)

    # Add ratio line (N2_crossover / N1)
    ratios_crossover = [c / n1 for c, n1 in zip(crossovers, n1_values)]

    # Panel 3: G1 distances
    for i, n1 in enumerate(n1_values):
        g1_dists = [results['data'][n1][n2]['g1_dist_mean'] for n2 in n2_values]
        fig.add_trace(go.Scatter(
            x=n2_values, y=g1_dists,
            mode='lines+markers',
            name=f'N1={n1} (G1)',
            line=dict(color=colors[i], width=2),
            marker=dict(size=6),
            showlegend=False,
        ), row=2, col=1)

    # Panel 4: G2 distances
    for i, n1 in enumerate(n1_values):
        g2_dists = [results['data'][n1][n2]['g2_dist_mean'] for n2 in n2_values]
        fig.add_trace(go.Scatter(
            x=n2_values, y=g2_dists,
            mode='lines+markers',
            name=f'N1={n1} (G2)',
            line=dict(color=colors[i], width=2),
            marker=dict(size=6),
            showlegend=False,
        ), row=2, col=2)

    # Update axes
    fig.update_xaxes(title_text="Graph2 Context Length (N2)", type="log", row=1, col=1)
    fig.update_xaxes(title_text="Graph1 Context Length", row=1, col=2)
    fig.update_xaxes(title_text="Graph2 Context Length (N2)", type="log", row=2, col=1)
    fig.update_xaxes(title_text="Graph2 Context Length (N2)", type="log", row=2, col=2)

    fig.update_yaxes(title_text="Ratio (G1-dist / G2-dist)", row=1, col=1)
    fig.update_yaxes(title_text="N2 at Crossover", row=1, col=2)
    fig.update_yaxes(title_text="L2 Distance (G1 pairs)", row=2, col=1)
    fig.update_yaxes(title_text="L2 Distance (G2 pairs)", row=2, col=2)

    fig.update_layout(
        title="Context Switching: When does Graph2 override Graph1?<br>" +
              "<sup>Context = [N1 tokens from Graph1] + [N2 tokens from Graph2]</sup>",
        width=1200,
        height=900,
        legend=dict(x=0.02, y=0.98),
    )

    output_dir = Path("results/context_switch")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig.write_html(output_dir / "context_switch.html")
    print(f"Saved: {output_dir / 'context_switch.html'}")

    # Save JSON
    # Convert nested dict to JSON-serializable format
    json_results = {
        'n1_values': n1_values,
        'n2_values': n2_values,
        'data': {str(n1): {str(n2): v for n2, v in n1_data.items()}
                 for n1, n1_data in results['data'].items()},
        'crossovers': {str(n1): c for n1, c in zip(n1_values, crossovers)},
        'n_trials': n_trials,
    }

    with open(output_dir / "context_switch_results.json", "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"Saved: {output_dir / 'context_switch_results.json'}")

    # Log to W&B
    print("\nLogging to W&B...")

    run = wandb.init(
        project='icl-structural-influence',
        name='context-switch-experiment',
        tags=['context-switch', 'forgetting', 'graph-competition'],
        config={
            'n1_values': n1_values,
            'n2_values': n2_values,
            'n_trials': n_trials,
            'layer_idx': layer_idx,
        }
    )

    wandb.log({
        'context_switch_plot': wandb.Html(open(output_dir / 'context_switch.html').read())
    })

    # Summary
    for i, n1 in enumerate(n1_values):
        wandb.summary[f'crossover_N1={n1}'] = crossovers[i]
        wandb.summary[f'crossover_ratio_N1={n1}'] = crossovers[i] / n1

    wandb.summary['mean_crossover_ratio'] = np.mean(ratios_crossover)

    print(f"\nRun URL: {run.url}")
    wandb.finish()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("Crossover points (N2 needed for Graph2 to win):")
    for n1, crossover in zip(n1_values, crossovers):
        ratio = crossover / n1
        print(f"  N1={n1}: N2={crossover} (ratio={ratio:.2f})")
    print(f"\nMean crossover ratio (N2/N1): {np.mean(ratios_crossover):.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
