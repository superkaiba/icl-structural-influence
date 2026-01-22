#!/usr/bin/env python3
"""
Chain Experiment: Graph1 → Graph2 → Graph3 → Graph4 → Graph5
Plot distances for each graph's pairs across the entire context.
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


class MultiGraphChain:
    """5 different random graphs with same tokens."""

    def __init__(self, seed=42):
        np.random.seed(seed)

        self.tokens = ['alpha', 'beta', 'gamma', 'delta', 'echo', 'foxtrot', 'golf', 'hotel',
                       'india', 'juliet', 'kilo', 'lima', 'mike', 'november', 'oscar', 'papa']

        self.graphs = []
        self.clusters = []
        self.adjacencies = []
        self.pairs = []

        used_clusterings = set()

        for i in range(5):
            while True:
                shuffled = self.tokens.copy()
                np.random.shuffle(shuffled)
                clustering = frozenset(frozenset(shuffled[j*4:(j+1)*4]) for j in range(4))
                if clustering not in used_clusterings:
                    used_clusterings.add(clustering)
                    break

            clusters = {j: shuffled[j*4:(j+1)*4] for j in range(4)}
            self.clusters.append(clusters)
            self.adjacencies.append(self._build_adjacency(clusters))
            self.pairs.append(self._get_pairs(clusters))

            print(f"Graph {i+1} clusters:")
            for cid, toks in clusters.items():
                print(f"  {cid}: {toks}")
            print()

    def _build_adjacency(self, clusters, p_in=0.9, p_out=0.1):
        t2c = {t: c for c, toks in clusters.items() for t in toks}
        return {t1: {t2: (p_in if t2c[t1] == t2c[t2] else p_out)
                     for t2 in self.tokens if t1 != t2} for t1 in self.tokens}

    def _get_pairs(self, clusters):
        return [(min(t1, t2), max(t1, t2)) for toks in clusters.values()
                for i, t1 in enumerate(toks) for t2 in toks[i+1:]]

    def generate_walk(self, graph_idx, length):
        adj = self.adjacencies[graph_idx]
        current = np.random.choice(self.tokens)
        walk = [current]
        for _ in range(length - 1):
            neighbors = list(adj[current].keys())
            weights = np.array([adj[current][n] for n in neighbors])
            current = np.random.choice(neighbors, p=weights/weights.sum())
            walk.append(current)
        return walk

    def generate_chain(self, lengths):
        """Generate chain: [G1 walk] [G2 walk] [G3 walk] [G4 walk] [G5 walk]"""
        chain = []
        boundaries = [0]
        for i, length in enumerate(lengths):
            walk = self.generate_walk(i, length)
            chain.extend(walk)
            boundaries.append(boundaries[-1] + length)
        return chain, boundaries


def compute_distances_at_position(model, tokenizer, chain, position, window=50, layer_idx=-5):
    """Get representations at a specific position (using surrounding window)."""
    # Use context up to position + some window
    end_pos = min(position + window, len(chain))
    start_pos = max(0, position - window)

    context = ' '.join(chain[:end_pos])
    tokens = tokenizer.encode(context, add_special_tokens=False)
    input_ids = torch.tensor([tokens]).to(model.device)

    with torch.no_grad():
        hidden = model(input_ids, output_hidden_states=True).hidden_states[layer_idx][0]

    # Collect representations from window around position
    token_reps = defaultdict(list)
    for pos in range(start_pos, min(end_pos, hidden.shape[0])):
        token_reps[chain[pos]].append(hidden[pos].cpu().float().numpy())

    return {t: np.mean(r, axis=0) for t, r in token_reps.items() if r}


def compute_pair_distance(reps, pairs):
    dists = [np.linalg.norm(reps[t1] - reps[t2]) for t1, t2 in pairs if t1 in reps and t2 in reps]
    return np.mean(dists) if dists else np.nan


def main():
    print("=" * 70)
    print("CHAIN EXPERIMENT: G1 → G2 → G3 → G4 → G5")
    print("=" * 70)

    hooked = HookedLLM.from_pretrained('meta-llama/Llama-3.1-8B', torch_dtype=torch.float16, device_map='auto')
    model, tokenizer = hooked.model, hooked.tokenizer
    model.eval()

    # Create 5 graphs
    multi_graph = MultiGraphChain(seed=42)

    # Each graph gets 1000 tokens
    segment_length = 1000
    lengths = [segment_length] * 5
    total_length = sum(lengths)

    n_trials = 20

    # Sample positions throughout the chain
    sample_positions = list(range(50, total_length, 50))

    # Results: distance for each graph's pairs at each position
    results = {f'G{i+1}': {'positions': [], 'distances': [], 'stds': []} for i in range(5)}

    print(f"\nChain structure: {' → '.join([f'G{i+1}({l})' for i, l in enumerate(lengths)])}")
    print(f"Total length: {total_length}")
    print(f"Boundaries: {[i * segment_length for i in range(6)]}")
    print(f"\nSampling at {len(sample_positions)} positions...")

    for pos in sample_positions:
        print(f"  Position {pos}...", end=" ", flush=True)

        trial_distances = {f'G{i+1}': [] for i in range(5)}

        for trial in range(n_trials):
            try:
                np.random.seed(42 + trial * 1000 + pos)
                chain, boundaries = multi_graph.generate_chain(lengths)
                reps = compute_distances_at_position(model, tokenizer, chain, pos)

                for i in range(5):
                    dist = compute_pair_distance(reps, multi_graph.pairs[i])
                    if not np.isnan(dist):
                        trial_distances[f'G{i+1}'].append(dist)
            except RuntimeError as e:
                print(f"CUDA error at trial {trial}, skipping...", end=" ")
                torch.cuda.empty_cache()
                continue

        # Clear GPU memory periodically
        if pos % 500 == 0:
            torch.cuda.empty_cache()

        for i in range(5):
            key = f'G{i+1}'
            if trial_distances[key]:
                results[key]['positions'].append(pos)
                results[key]['distances'].append(float(np.mean(trial_distances[key])))
                results[key]['stds'].append(float(np.std(trial_distances[key])))

        # Print which graph region we're in
        region = pos // segment_length + 1
        print(f"in G{min(region, 5)} region")

    # Create visualization with LOG SCALE
    print("\n" + "=" * 70)
    print("Creating visualization...")

    fig = go.Figure()
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

    for i in range(5):
        key = f'G{i+1}'
        fig.add_trace(go.Scatter(
            x=results[key]['positions'],
            y=results[key]['distances'],
            mode='lines+markers',
            name=f'G{i+1}-pair distance',
            line=dict(color=colors[i], width=2),
            marker=dict(size=6),
        ))

    # Add vertical lines at boundaries
    for i in range(1, 5):
        boundary = i * segment_length
        fig.add_vline(x=boundary, line_dash="dash", line_color="gray", line_width=1,
                      annotation_text=f"G{i}→G{i+1}", annotation_position="top")

    # Add shaded regions for each graph
    for i in range(5):
        fig.add_vrect(
            x0=i * segment_length,
            x1=(i + 1) * segment_length,
            fillcolor=colors[i],
            opacity=0.1,
            line_width=0,
        )

    # Set layout with explicit axis types (direct assignment to avoid template override)
    fig.layout.title = f"Chain Experiment: Distance for Each Graph's Pairs Across Context<br><sup>Chain: G1({segment_length}) → G2({segment_length}) → G3({segment_length}) → G4({segment_length}) → G5({segment_length})</sup>"
    fig.layout.width = 1200
    fig.layout.height = 600
    fig.layout.legend = dict(x=1.02, y=1, xanchor='left')
    fig.layout.xaxis.type = 'linear'
    fig.layout.xaxis.title = 'Position in Context'
    fig.layout.yaxis.type = 'log'
    fig.layout.yaxis.title = 'Mean L2 Distance (log scale)'
    fig.layout.yaxis.range = [np.log10(10), np.log10(12)]  # 10 to 12 on log scale

    output_dir = Path("results/context_switch")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_dir / "chain_experiment.html")
    print(f"Saved: {output_dir / 'chain_experiment.html'}")

    # Also create linear x-scale version (keeps log y-scale)
    fig_linear = go.Figure(fig)
    fig_linear.update_layout(
        xaxis_type="linear",
        xaxis_title="Position in Context",
        title="Chain Experiment: Distance for Each Graph's Pairs Across Context (Linear X)<br>" +
              f"<sup>Chain: G1({segment_length}) → G2({segment_length}) → G3({segment_length}) → G4({segment_length}) → G5({segment_length})</sup>",
    )
    fig_linear.write_html(output_dir / "chain_experiment_linear.html")
    print(f"Saved: {output_dir / 'chain_experiment_linear.html'}")

    # Save JSON
    with open(output_dir / "chain_experiment_results.json", "w") as f:
        json.dump({
            'segment_length': segment_length,
            'total_length': total_length,
            'n_trials': n_trials,
            'results': results,
        }, f, indent=2)

    # Log to W&B
    print("\nLogging to W&B...")
    run = wandb.init(project='icl-structural-influence', name='chain-experiment-5-graphs',
                     tags=['chain', '5-graphs', 'distance-tracking'])
    wandb.log({
        'chain_log_scale': wandb.Html(open(output_dir / 'chain_experiment.html').read()),
        'chain_linear_scale': wandb.Html(open(output_dir / 'chain_experiment_linear.html').read()),
    })
    print(f"Run URL: {run.url}")
    wandb.finish()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("Each graph's pairs should have LOW distance when in that graph's region,")
    print("and HIGH distance in other regions.")
    print("=" * 70)


if __name__ == "__main__":
    main()
