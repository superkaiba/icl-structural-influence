#!/usr/bin/env python3
"""
Context Switching v4:
Plot over TOTAL context length to show the transition at the switch point.
X-axis: total context length (N1 + N2)
Switch happens at x = 1000
"""

import numpy as np
import torch
import torch.nn.functional as F
import plotly.graph_objects as go
from pathlib import Path
from collections import defaultdict
import wandb
import json
import sys

sys.path.insert(0, str(Path('.') / 'src'))
from models import HookedLLM


class RandomDualGraph:
    """Two random graphs with same tokens, different clusters."""

    def __init__(self, seed=42):
        np.random.seed(seed)

        self.tokens = [
            'alpha', 'beta', 'gamma', 'delta',
            'echo', 'foxtrot', 'golf', 'hotel',
            'india', 'juliet', 'kilo', 'lima',
            'mike', 'november', 'oscar', 'papa'
        ]

        shuffled1 = self.tokens.copy()
        np.random.shuffle(shuffled1)
        self.graph1_clusters = {
            0: shuffled1[0:4], 1: shuffled1[4:8],
            2: shuffled1[8:12], 3: shuffled1[12:16]
        }

        shuffled2 = self.tokens.copy()
        np.random.shuffle(shuffled2)
        while self._same_clustering(shuffled1, shuffled2):
            np.random.shuffle(shuffled2)

        self.graph2_clusters = {
            0: shuffled2[0:4], 1: shuffled2[4:8],
            2: shuffled2[8:12], 3: shuffled2[12:16]
        }

        self.graph1_adj = self._build_adjacency(self.graph1_clusters)
        self.graph2_adj = self._build_adjacency(self.graph2_clusters)

    def _same_clustering(self, s1, s2):
        c1 = {frozenset(s1[i:i+4]) for i in range(0, 16, 4)}
        c2 = {frozenset(s2[i:i+4]) for i in range(0, 16, 4)}
        return c1 == c2

    def _build_adjacency(self, clusters, p_in=0.9, p_out=0.1):
        adj = {}
        t2c = {t: c for c, toks in clusters.items() for t in toks}
        for t1 in self.tokens:
            adj[t1] = {t2: (p_in if t2c[t1] == t2c[t2] else p_out)
                       for t2 in self.tokens if t1 != t2}
        return adj

    def generate_walk(self, adj, length):
        current = np.random.choice(self.tokens)
        walk = [current]
        for _ in range(length - 1):
            neighbors = list(adj[current].keys())
            weights = np.array([adj[current][n] for n in neighbors])
            current = np.random.choice(neighbors, p=weights/weights.sum())
            walk.append(current)
        return walk

    def get_pairs(self, clusters):
        pairs = []
        for toks in clusters.values():
            for i, t1 in enumerate(toks):
                for t2 in toks[i+1:]:
                    pairs.append((min(t1,t2), max(t1,t2)))
        return pairs


def collect_reps(model, tokenizer, walk1, walk2, layer_idx=-5):
    """Collect representations from walk2 portion."""
    full_context = ' '.join(walk1 + walk2)
    tokens = tokenizer.encode(full_context, add_special_tokens=False)
    input_ids = torch.tensor([tokens]).to(model.device)

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        hidden = outputs.hidden_states[layer_idx][0]

    n1 = len(walk1)
    token_reps = defaultdict(list)
    all_words = walk1 + walk2

    for pos, word in enumerate(all_words):
        if pos >= n1 and pos < hidden.shape[0]:
            token_reps[word].append(hidden[pos].cpu().float().numpy())

    return {t: np.mean(r, axis=0) for t, r in token_reps.items() if r}


def compute_pair_distances(reps, pairs):
    dists = [np.linalg.norm(reps[t1] - reps[t2])
             for t1, t2 in pairs if t1 in reps and t2 in reps]
    return np.mean(dists) if dists else np.nan


def main():
    print("=" * 70)
    print("CONTEXT SWITCHING v4: Plot over Total Context Length")
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

    # Parameters
    n1 = 1000  # Switch point
    n2_values = [0, 5, 10, 20, 50, 100, 200, 300, 500, 750, 1000]
    n_trials = 20
    graph_seeds = [42, 123, 456, 789, 1011]
    layer_idx = -5

    all_results = {}

    for graph_idx, graph_seed in enumerate(graph_seeds):
        print(f"\nGraph Pair {graph_idx + 1}/5 (seed={graph_seed})")

        graph = RandomDualGraph(seed=graph_seed)
        g1_pairs = graph.get_pairs(graph.graph1_clusters)
        g2_pairs = graph.get_pairs(graph.graph2_clusters)

        results = []

        for n2 in n2_values:
            total_len = n1 + n2
            print(f"  Total={total_len} (N1={n1}, N2={n2})...", end=" ", flush=True)

            if n2 == 0:
                # Only Graph1, no Graph2 yet - measure at end of Graph1
                trial_ratios = []
                for trial in range(n_trials):
                    np.random.seed(graph_seed * 1000 + trial)
                    walk1 = graph.generate_walk(graph.graph1_adj, n1)

                    # Get reps from last portion of walk1
                    full_context = ' '.join(walk1)
                    tokens = tokenizer.encode(full_context, add_special_tokens=False)
                    input_ids = torch.tensor([tokens]).to(model.device)

                    with torch.no_grad():
                        outputs = model(input_ids, output_hidden_states=True)
                        hidden = outputs.hidden_states[layer_idx][0]

                    # Use last 100 positions
                    token_reps = defaultdict(list)
                    for pos in range(max(0, len(walk1)-100), len(walk1)):
                        if pos < hidden.shape[0]:
                            token_reps[walk1[pos]].append(hidden[pos].cpu().float().numpy())

                    reps = {t: np.mean(r, axis=0) for t, r in token_reps.items() if r}

                    if len(reps) >= 2:
                        g1_dist = compute_pair_distances(reps, g1_pairs)
                        g2_dist = compute_pair_distances(reps, g2_pairs)
                        if g2_dist > 0:
                            trial_ratios.append(g1_dist / g2_dist)

                if trial_ratios:
                    results.append({
                        'total_len': total_len,
                        'n2': n2,
                        'ratio_mean': float(np.mean(trial_ratios)),
                        'ratio_std': float(np.std(trial_ratios)),
                        'ratio_sem': float(np.std(trial_ratios) / np.sqrt(len(trial_ratios))),
                    })
                    print(f"ratio={results[-1]['ratio_mean']:.3f}")
                else:
                    print("no valid")
            else:
                trial_ratios = []
                for trial in range(n_trials):
                    np.random.seed(graph_seed * 1000 + trial * 100 + n2)
                    walk1 = graph.generate_walk(graph.graph1_adj, n1)
                    walk2 = graph.generate_walk(graph.graph2_adj, n2)
                    reps = collect_reps(model, tokenizer, walk1, walk2, layer_idx)

                    if len(reps) >= 2:
                        g1_dist = compute_pair_distances(reps, g1_pairs)
                        g2_dist = compute_pair_distances(reps, g2_pairs)
                        if g2_dist > 0 and not np.isnan(g1_dist):
                            trial_ratios.append(g1_dist / g2_dist)

                if trial_ratios:
                    results.append({
                        'total_len': total_len,
                        'n2': n2,
                        'ratio_mean': float(np.mean(trial_ratios)),
                        'ratio_std': float(np.std(trial_ratios)),
                        'ratio_sem': float(np.std(trial_ratios) / np.sqrt(len(trial_ratios))),
                    })
                    print(f"ratio={results[-1]['ratio_mean']:.3f}")
                else:
                    print("no valid")

        all_results[graph_seed] = results

    # Create visualization
    print("\n" + "=" * 70)
    print("Creating visualization...")

    fig = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, (seed, results) in enumerate(all_results.items()):
        total_lens = [r['total_len'] for r in results]
        ratios = [r['ratio_mean'] for r in results]
        sems = [r['ratio_sem'] for r in results]

        fig.add_trace(go.Scatter(
            x=total_lens,
            y=ratios,
            mode='lines+markers',
            name=f'Graph pair {i+1}',
            line=dict(color=colors[i], width=2),
            marker=dict(size=8),
            error_y=dict(type='data', array=sems, visible=True),
        ))

    # Add vertical line at switch point
    fig.add_vline(x=n1, line_dash="dash", line_color="black", line_width=2,
                  annotation_text="Switch to Graph2", annotation_position="top")

    # Add horizontal line at ratio=1
    fig.add_hline(y=1.0, line_dash="dot", line_color="gray",
                  annotation_text="Crossover", annotation_position="right")

    # Add shaded regions
    fig.add_vrect(x0=0, x1=n1, fillcolor="rgba(255,0,0,0.05)",
                  line_width=0, annotation_text="Graph1 region",
                  annotation_position="top left")
    fig.add_vrect(x0=n1, x1=n1+max(n2_values), fillcolor="rgba(0,0,255,0.05)",
                  line_width=0, annotation_text="Graph2 region",
                  annotation_position="top right")

    fig.update_layout(
        title="Context Switching: Ratio over Total Context Length<br>" +
              f"<sup>Switch from Graph1 to Graph2 at position {n1} | 5 graph pairs × {n_trials} trials</sup>",
        xaxis_title="Total Context Length",
        yaxis_title="Ratio (G1-pair dist / G2-pair dist)",
        width=1000,
        height=600,
        legend=dict(x=0.02, y=0.98),
    )

    output_dir = Path("results/context_switch")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig.write_html(output_dir / "context_switch_v4_total_length.html")
    print(f"Saved: {output_dir / 'context_switch_v4_total_length.html'}")

    # Save JSON
    json_results = {
        'n1': n1,
        'n2_values': n2_values,
        'n_trials': n_trials,
        'graph_seeds': graph_seeds,
        'results': {str(k): v for k, v in all_results.items()},
    }

    with open(output_dir / "context_switch_v4_results.json", "w") as f:
        json.dump(json_results, f, indent=2)

    # Log to W&B
    print("\nLogging to W&B...")

    run = wandb.init(
        project='icl-structural-influence',
        name='context-switch-v4-total-length',
        tags=['context-switch', 'total-length', 'transition'],
        config={'n1': n1, 'n2_values': n2_values, 'n_trials': n_trials}
    )

    wandb.log({
        'total_length_plot': wandb.Html(open(output_dir / 'context_switch_v4_total_length.html').read())
    })

    print(f"\nRun URL: {run.url}")
    wandb.finish()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Switch point: position {n1}")
    print(f"Before switch (x={n1}): Ratio < 1 (Graph1 structure dominates)")
    print(f"After switch (x>{n1}): Ratio > 1 (Graph2 structure dominates)")
    print("=" * 70)


if __name__ == "__main__":
    main()
