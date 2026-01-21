#!/usr/bin/env python3
"""
Context Switching Experiment v3:
- Linear x-axis
- Add perplexity tracking
- Single plot
- 5 different random graph pairs
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
        while self._same_clustering(shuffled1, shuffled2):
            np.random.shuffle(shuffled2)

        self.graph2_clusters = {
            0: shuffled2[0:4],
            1: shuffled2[4:8],
            2: shuffled2[8:12],
            3: shuffled2[12:16]
        }

        self.graph1_adj = self._build_adjacency(self.graph1_clusters)
        self.graph2_adj = self._build_adjacency(self.graph2_clusters)

    def _same_clustering(self, shuffle1, shuffle2):
        c1 = {frozenset(shuffle1[i:i+4]) for i in range(0, 16, 4)}
        c2 = {frozenset(shuffle2[i:i+4]) for i in range(0, 16, 4)}
        return c1 == c2

    def _build_adjacency(self, clusters, p_in=0.9, p_out=0.1):
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

    def generate_walk(self, adj, length):
        start = np.random.choice(self.tokens)
        walk = [start]
        current = start
        for _ in range(length - 1):
            neighbors = list(adj[current].keys())
            weights = np.array([adj[current][n] for n in neighbors])
            weights = weights / weights.sum()
            current = np.random.choice(neighbors, p=weights)
            walk.append(current)
        return walk

    def get_pairs(self, clusters):
        pairs = []
        for cid, toks in clusters.items():
            for i, t1 in enumerate(toks):
                for t2 in toks[i+1:]:
                    pairs.append((t1, t2) if t1 < t2 else (t2, t1))
        return pairs


def compute_perplexity(model, tokenizer, text):
    """Compute perplexity of text."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    input_ids = torch.tensor([tokens]).to(model.device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, :-1]  # Predict next token
        targets = input_ids[0, 1:]

        loss = F.cross_entropy(logits, targets, reduction='mean')
        perplexity = torch.exp(loss).item()

    return perplexity


def collect_reps_and_perplexity(model, tokenizer, walk1, walk2, layer_idx=-5):
    """Collect representations from walk2 portion and compute perplexity."""
    full_context = ' '.join(walk1 + walk2)
    tokens = tokenizer.encode(full_context, add_special_tokens=False)
    input_ids = torch.tensor([tokens]).to(model.device)

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        hidden = outputs.hidden_states[layer_idx][0]

        # Perplexity on walk2 portion
        n1_tokens = len(tokenizer.encode(' '.join(walk1), add_special_tokens=False))
        logits = outputs.logits[0]

        if n1_tokens < logits.shape[0] - 1:
            walk2_logits = logits[n1_tokens:-1]
            walk2_targets = input_ids[0, n1_tokens+1:]
            if len(walk2_targets) > 0:
                loss = F.cross_entropy(walk2_logits[:len(walk2_targets)], walk2_targets, reduction='mean')
                perplexity = torch.exp(loss).item()
            else:
                perplexity = np.nan
        else:
            perplexity = np.nan

    # Representations from walk2 portion
    n1 = len(walk1)
    token_reps = defaultdict(list)
    all_words = walk1 + walk2

    for pos, word in enumerate(all_words):
        if pos >= n1 and pos < hidden.shape[0]:
            token_reps[word].append(hidden[pos].cpu().float().numpy())

    reps = {t: np.mean(r, axis=0) for t, r in token_reps.items() if r}
    return reps, perplexity


def compute_pair_distances(reps, pairs):
    dists = [np.linalg.norm(reps[t1] - reps[t2])
             for t1, t2 in pairs if t1 in reps and t2 in reps]
    return np.mean(dists) if dists else np.nan


def main():
    print("=" * 70)
    print("CONTEXT SWITCHING v3: 5 Graph Pairs, Linear Scale, + Perplexity")
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
    n1 = 1000
    n2_values = [5, 10, 20, 50, 100, 150, 200, 300, 400, 500, 600, 800, 1000]
    n_trials = 20
    n_graph_pairs = 5
    layer_idx = -5

    graph_seeds = [42, 123, 456, 789, 1011]

    all_results = {}

    for graph_idx, graph_seed in enumerate(graph_seeds):
        print(f"\n{'='*50}")
        print(f"Graph Pair {graph_idx + 1}/5 (seed={graph_seed})")
        print(f"{'='*50}")

        graph = RandomDualGraph(seed=graph_seed)
        g1_pairs = graph.get_pairs(graph.graph1_clusters)
        g2_pairs = graph.get_pairs(graph.graph2_clusters)

        results = {'n2_values': n2_values, 'data': {}}

        for n2 in n2_values:
            print(f"  N2={n2}...", end=" ", flush=True)

            trial_ratios = []
            trial_perplexities = []

            for trial in range(n_trials):
                np.random.seed(graph_seed * 1000 + trial * 100 + n2)

                walk1 = graph.generate_walk(graph.graph1_adj, n1)
                walk2 = graph.generate_walk(graph.graph2_adj, n2)

                reps, ppl = collect_reps_and_perplexity(model, tokenizer, walk1, walk2, layer_idx)

                if len(reps) >= 2:
                    g1_dist = compute_pair_distances(reps, g1_pairs)
                    g2_dist = compute_pair_distances(reps, g2_pairs)

                    if not np.isnan(g1_dist) and not np.isnan(g2_dist) and g2_dist > 0:
                        trial_ratios.append(g1_dist / g2_dist)

                if not np.isnan(ppl) and ppl < 1000:  # Filter outliers
                    trial_perplexities.append(ppl)

            if trial_ratios:
                results['data'][n2] = {
                    'ratio_mean': float(np.mean(trial_ratios)),
                    'ratio_std': float(np.std(trial_ratios)),
                    'ratio_sem': float(np.std(trial_ratios) / np.sqrt(len(trial_ratios))),
                    'ppl_mean': float(np.mean(trial_perplexities)) if trial_perplexities else np.nan,
                    'ppl_std': float(np.std(trial_perplexities)) if trial_perplexities else np.nan,
                }
                print(f"ratio={results['data'][n2]['ratio_mean']:.3f}, ppl={results['data'][n2]['ppl_mean']:.1f}")
            else:
                results['data'][n2] = None
                print("no valid trials")

        all_results[graph_seed] = results

    # Create visualization
    print("\n" + "=" * 70)
    print("Creating visualization...")
    print("=" * 70)

    fig = go.Figure()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Plot ratio for each graph pair
    for i, (graph_seed, results) in enumerate(all_results.items()):
        valid_n2 = [n2 for n2 in n2_values if results['data'].get(n2)]
        ratios = [results['data'][n2]['ratio_mean'] for n2 in valid_n2]
        sems = [results['data'][n2]['ratio_sem'] for n2 in valid_n2]

        fig.add_trace(go.Scatter(
            x=valid_n2,
            y=ratios,
            mode='lines+markers',
            name=f'Graph pair {i+1}',
            line=dict(color=colors[i], width=2),
            marker=dict(size=6),
            error_y=dict(type='data', array=sems, visible=True),
        ))

    # Add threshold line
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                  annotation_text="Crossover (ratio=1)", annotation_position="right")

    fig.update_layout(
        title=f"Context Switching: Ratio (G1-dist / G2-dist) vs Graph2 Length<br>" +
              f"<sup>N1={n1} tokens from Graph1, then N2 tokens from Graph2 | {n_trials} trials × 5 graph pairs</sup>",
        xaxis_title="Graph2 Context Length (N2)",
        yaxis_title="Ratio (G1-pair dist / G2-pair dist)",
        width=900,
        height=600,
        legend=dict(x=0.98, y=0.98, xanchor='right'),
    )

    output_dir = Path("results/context_switch")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig.write_html(output_dir / "context_switch_v3_ratio.html")
    print(f"Saved: {output_dir / 'context_switch_v3_ratio.html'}")

    # Perplexity plot
    fig_ppl = go.Figure()

    for i, (graph_seed, results) in enumerate(all_results.items()):
        valid_n2 = [n2 for n2 in n2_values if results['data'].get(n2) and not np.isnan(results['data'][n2].get('ppl_mean', np.nan))]
        ppls = [results['data'][n2]['ppl_mean'] for n2 in valid_n2]

        fig_ppl.add_trace(go.Scatter(
            x=valid_n2,
            y=ppls,
            mode='lines+markers',
            name=f'Graph pair {i+1}',
            line=dict(color=colors[i], width=2),
            marker=dict(size=6),
        ))

    fig_ppl.update_layout(
        title=f"Perplexity on Graph2 Portion vs Graph2 Length<br>" +
              f"<sup>N1={n1} tokens from Graph1, then N2 tokens from Graph2</sup>",
        xaxis_title="Graph2 Context Length (N2)",
        yaxis_title="Perplexity",
        width=900,
        height=600,
        legend=dict(x=0.98, y=0.98, xanchor='right'),
    )

    fig_ppl.write_html(output_dir / "context_switch_v3_perplexity.html")
    print(f"Saved: {output_dir / 'context_switch_v3_perplexity.html'}")

    # Combined plot (ratio + perplexity on secondary axis)
    from plotly.subplots import make_subplots

    fig_combined = make_subplots(specs=[[{"secondary_y": True}]])

    # Average across graph pairs
    avg_ratios = []
    avg_ppls = []
    for n2 in n2_values:
        ratios_at_n2 = [all_results[s]['data'][n2]['ratio_mean']
                        for s in graph_seeds if all_results[s]['data'].get(n2)]
        ppls_at_n2 = [all_results[s]['data'][n2]['ppl_mean']
                      for s in graph_seeds if all_results[s]['data'].get(n2) and not np.isnan(all_results[s]['data'][n2].get('ppl_mean', np.nan))]
        avg_ratios.append(np.mean(ratios_at_n2) if ratios_at_n2 else np.nan)
        avg_ppls.append(np.mean(ppls_at_n2) if ppls_at_n2 else np.nan)

    fig_combined.add_trace(
        go.Scatter(x=n2_values, y=avg_ratios, name="Ratio (avg)",
                   line=dict(color='#9467bd', width=3), marker=dict(size=8)),
        secondary_y=False,
    )

    fig_combined.add_trace(
        go.Scatter(x=n2_values, y=avg_ppls, name="Perplexity (avg)",
                   line=dict(color='#2ca02c', width=3, dash='dash'), marker=dict(size=8)),
        secondary_y=True,
    )

    fig_combined.add_hline(y=1.0, line_dash="dot", line_color="gray", secondary_y=False)

    fig_combined.update_layout(
        title=f"Context Switching: Ratio and Perplexity (averaged over 5 graph pairs)<br>" +
              f"<sup>N1={n1} tokens from Graph1, then N2 tokens from Graph2</sup>",
        xaxis_title="Graph2 Context Length (N2)",
        width=900,
        height=600,
    )
    fig_combined.update_yaxes(title_text="Ratio (G1-dist / G2-dist)", secondary_y=False)
    fig_combined.update_yaxes(title_text="Perplexity", secondary_y=True)

    fig_combined.write_html(output_dir / "context_switch_v3_combined.html")
    print(f"Saved: {output_dir / 'context_switch_v3_combined.html'}")

    # Save JSON
    json_results = {
        'n1': n1,
        'n2_values': n2_values,
        'n_trials': n_trials,
        'n_graph_pairs': n_graph_pairs,
        'graph_seeds': graph_seeds,
        'results': {str(k): {str(n2): v for n2, v in r['data'].items()}
                    for k, r in all_results.items()},
        'avg_ratios': [float(r) if not np.isnan(r) else None for r in avg_ratios],
        'avg_ppls': [float(p) if not np.isnan(p) else None for p in avg_ppls],
    }

    with open(output_dir / "context_switch_v3_results.json", "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"Saved: {output_dir / 'context_switch_v3_results.json'}")

    # Log to W&B
    print("\nLogging to W&B...")

    run = wandb.init(
        project='icl-structural-influence',
        name='context-switch-v3-5-graphs',
        tags=['context-switch', '5-graphs', 'perplexity', 'linear-scale'],
        config={
            'n1': n1,
            'n2_values': n2_values,
            'n_trials': n_trials,
            'n_graph_pairs': n_graph_pairs,
        }
    )

    wandb.log({
        'ratio_plot': wandb.Html(open(output_dir / 'context_switch_v3_ratio.html').read()),
        'perplexity_plot': wandb.Html(open(output_dir / 'context_switch_v3_perplexity.html').read()),
        'combined_plot': wandb.Html(open(output_dir / 'context_switch_v3_combined.html').read()),
    })

    # Find average crossover
    first_cross = None
    for i, n2 in enumerate(n2_values):
        if not np.isnan(avg_ratios[i]) and avg_ratios[i] > 1:
            first_cross = n2
            break

    wandb.summary['avg_first_crossover'] = first_cross
    wandb.summary['n1'] = n1

    print(f"\nRun URL: {run.url}")
    wandb.finish()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Graph1 context: N1 = {n1}")
    print(f"Graph pairs tested: {n_graph_pairs}")
    print(f"Trials per (graph, N2): {n_trials}")
    print(f"\nAverage crossover (ratio > 1): N2 = {first_cross}")
    if first_cross:
        print(f"Ratio: {first_cross/n1:.1%} of Graph1 context")
    print("=" * 70)


if __name__ == "__main__":
    main()
