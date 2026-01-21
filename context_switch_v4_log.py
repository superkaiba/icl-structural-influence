#!/usr/bin/env python3
"""
Context Switching v4 with LOG scale x-axis.
"""

import numpy as np
import torch
import plotly.graph_objects as go
from pathlib import Path
from collections import defaultdict
import wandb
import json
import sys

sys.path.insert(0, str(Path('.') / 'src'))
from models import HookedLLM


class RandomDualGraph:
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.tokens = ['alpha', 'beta', 'gamma', 'delta', 'echo', 'foxtrot', 'golf', 'hotel',
                       'india', 'juliet', 'kilo', 'lima', 'mike', 'november', 'oscar', 'papa']

        shuffled1 = self.tokens.copy()
        np.random.shuffle(shuffled1)
        self.graph1_clusters = {i: shuffled1[i*4:(i+1)*4] for i in range(4)}

        shuffled2 = self.tokens.copy()
        np.random.shuffle(shuffled2)
        while {frozenset(shuffled1[i*4:(i+1)*4]) for i in range(4)} == \
              {frozenset(shuffled2[i*4:(i+1)*4]) for i in range(4)}:
            np.random.shuffle(shuffled2)
        self.graph2_clusters = {i: shuffled2[i*4:(i+1)*4] for i in range(4)}

        self.graph1_adj = self._build_adjacency(self.graph1_clusters)
        self.graph2_adj = self._build_adjacency(self.graph2_clusters)

    def _build_adjacency(self, clusters, p_in=0.9, p_out=0.1):
        t2c = {t: c for c, toks in clusters.items() for t in toks}
        return {t1: {t2: (p_in if t2c[t1] == t2c[t2] else p_out)
                     for t2 in self.tokens if t1 != t2} for t1 in self.tokens}

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
        return [(min(t1,t2), max(t1,t2)) for toks in clusters.values()
                for i, t1 in enumerate(toks) for t2 in toks[i+1:]]


def collect_reps(model, tokenizer, walk1, walk2, layer_idx=-5):
    full_context = ' '.join(walk1 + walk2)
    tokens = tokenizer.encode(full_context, add_special_tokens=False)
    input_ids = torch.tensor([tokens]).to(model.device)

    with torch.no_grad():
        hidden = model(input_ids, output_hidden_states=True).hidden_states[layer_idx][0]

    n1, token_reps = len(walk1), defaultdict(list)
    for pos, word in enumerate(walk1 + walk2):
        if pos >= n1 and pos < hidden.shape[0]:
            token_reps[word].append(hidden[pos].cpu().float().numpy())
    return {t: np.mean(r, axis=0) for t, r in token_reps.items() if r}


def compute_pair_distances(reps, pairs):
    dists = [np.linalg.norm(reps[t1] - reps[t2]) for t1, t2 in pairs if t1 in reps and t2 in reps]
    return np.mean(dists) if dists else np.nan


def main():
    print("=" * 70)
    print("CONTEXT SWITCHING v4: LOG SCALE")
    print("=" * 70)

    hooked = HookedLLM.from_pretrained('meta-llama/Llama-3.1-8B', torch_dtype=torch.float16, device_map='auto')
    model, tokenizer = hooked.model, hooked.tokenizer
    model.eval()

    n1 = 1000
    n2_values = [0, 5, 10, 20, 50, 100, 200, 300, 500, 750, 1000]
    n_trials, graph_seeds = 20, [42, 123, 456, 789, 1011]

    all_results = {}

    for graph_idx, graph_seed in enumerate(graph_seeds):
        print(f"\nGraph Pair {graph_idx + 1}/5")
        graph = RandomDualGraph(seed=graph_seed)
        g1_pairs, g2_pairs = graph.get_pairs(graph.graph1_clusters), graph.get_pairs(graph.graph2_clusters)
        results = []

        for n2 in n2_values:
            total_len = n1 + n2
            print(f"  Total={total_len}...", end=" ")

            trial_ratios = []
            for trial in range(n_trials):
                np.random.seed(graph_seed * 1000 + trial * 100 + n2)
                walk1 = graph.generate_walk(graph.graph1_adj, n1)

                if n2 == 0:
                    full_context = ' '.join(walk1)
                    tokens = tokenizer.encode(full_context, add_special_tokens=False)
                    input_ids = torch.tensor([tokens]).to(model.device)
                    with torch.no_grad():
                        hidden = model(input_ids, output_hidden_states=True).hidden_states[-5][0]
                    token_reps = defaultdict(list)
                    for pos in range(max(0, len(walk1)-100), len(walk1)):
                        if pos < hidden.shape[0]:
                            token_reps[walk1[pos]].append(hidden[pos].cpu().float().numpy())
                    reps = {t: np.mean(r, axis=0) for t, r in token_reps.items() if r}
                else:
                    walk2 = graph.generate_walk(graph.graph2_adj, n2)
                    reps = collect_reps(model, tokenizer, walk1, walk2)

                if len(reps) >= 2:
                    g1_dist, g2_dist = compute_pair_distances(reps, g1_pairs), compute_pair_distances(reps, g2_pairs)
                    if g2_dist > 0 and not np.isnan(g1_dist):
                        trial_ratios.append(g1_dist / g2_dist)

            if trial_ratios:
                results.append({'total_len': total_len, 'n2': n2,
                                'ratio_mean': float(np.mean(trial_ratios)),
                                'ratio_sem': float(np.std(trial_ratios) / np.sqrt(len(trial_ratios)))})
                print(f"ratio={results[-1]['ratio_mean']:.3f}")
            else:
                print("no valid")

        all_results[graph_seed] = results

    # Create LOG SCALE visualization
    fig = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, (seed, results) in enumerate(all_results.items()):
        total_lens = [r['total_len'] for r in results]
        ratios = [r['ratio_mean'] for r in results]
        sems = [r['ratio_sem'] for r in results]

        fig.add_trace(go.Scatter(
            x=total_lens, y=ratios,
            mode='lines+markers',
            name=f'Graph pair {i+1}',
            line=dict(color=colors[i], width=2),
            marker=dict(size=8),
            error_y=dict(type='data', array=sems, visible=True),
        ))

    fig.add_vline(x=n1, line_dash="dash", line_color="black", line_width=2,
                  annotation_text="Switch to Graph2", annotation_position="top")
    fig.add_hline(y=1.0, line_dash="dot", line_color="gray",
                  annotation_text="Crossover", annotation_position="right")

    fig.update_layout(
        title=f"Context Switching: Ratio vs Total Context Length (LOG SCALE)<br>" +
              f"<sup>Switch from Graph1 to Graph2 at position {n1}</sup>",
        xaxis_title="Total Context Length (log scale)",
        yaxis_title="Ratio (G1-pair dist / G2-pair dist)",
        xaxis_type="log",
        width=1000,
        height=600,
        legend=dict(x=0.02, y=0.98),
    )

    output_dir = Path("results/context_switch")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_dir / "context_switch_v4_log.html")
    print(f"\nSaved: {output_dir / 'context_switch_v4_log.html'}")

    run = wandb.init(project='icl-structural-influence', name='context-switch-v4-log-scale',
                     tags=['context-switch', 'log-scale'])
    wandb.log({'log_scale_plot': wandb.Html(open(output_dir / 'context_switch_v4_log.html').read())})
    print(f"Run URL: {run.url}")
    wandb.finish()


if __name__ == "__main__":
    main()
