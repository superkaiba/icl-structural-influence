#!/usr/bin/env python3
"""
Chain Perplexity Experiment: Track perplexity across G1 → G2 → G3 → G4 → G5 context.
"""

import numpy as np
import torch
import torch.nn.functional as F
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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


def compute_perplexity_at_position(model, tokenizer, chain, position, window=100):
    """Compute perplexity at a specific position using a window of context."""
    # Use context from (position - window) to position
    start_pos = max(0, position - window)
    context = ' '.join(chain[start_pos:position])

    if len(context.strip()) == 0:
        return np.nan

    tokens = tokenizer.encode(context, add_special_tokens=False)
    if len(tokens) < 2:
        return np.nan

    input_ids = torch.tensor([tokens]).to(model.device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, :-1]  # All but last
        targets = input_ids[0, 1:]  # All but first
        loss = F.cross_entropy(logits, targets, reduction='mean')
        perplexity = torch.exp(loss).item()

    return perplexity


def compute_local_perplexity(model, tokenizer, chain, position, context_length=50):
    """Compute perplexity for predicting tokens around a specific position."""
    # Get context ending at position
    start_pos = max(0, position - context_length)
    end_pos = min(len(chain), position + 10)  # Look ahead a bit

    context = ' '.join(chain[start_pos:end_pos])
    tokens = tokenizer.encode(context, add_special_tokens=False)

    if len(tokens) < 2:
        return np.nan

    input_ids = torch.tensor([tokens]).to(model.device)

    with torch.no_grad():
        outputs = model(input_ids)
        # Focus on the tokens around the position
        rel_pos = position - start_pos
        # Convert word position to approximate token position (rough estimate)
        approx_token_pos = min(rel_pos, len(tokens) - 2)

        logits = outputs.logits[0, approx_token_pos:approx_token_pos+5]
        targets = input_ids[0, approx_token_pos+1:approx_token_pos+6]

        if logits.shape[0] == 0 or targets.shape[0] == 0:
            return np.nan

        min_len = min(logits.shape[0], targets.shape[0])
        loss = F.cross_entropy(logits[:min_len], targets[:min_len], reduction='mean')
        perplexity = torch.exp(loss).item()

    return perplexity


def main():
    print("=" * 70)
    print("CHAIN PERPLEXITY EXPERIMENT: G1 → G2 → G3 → G4 → G5")
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

    n_trials = 10

    # Sample positions - denser at transitions
    regular_positions = list(range(100, total_length, 100))

    # Add dense sampling at transitions
    transition_positions = []
    for boundary in [1000, 2000, 3000, 4000]:
        transition_positions.extend(range(boundary - 20, boundary + 50, 5))

    sample_positions = sorted(set(regular_positions + transition_positions))
    sample_positions = [p for p in sample_positions if 50 < p < total_length - 10]

    print(f"\nChain structure: {' → '.join([f'G{i+1}({l})' for i, l in enumerate(lengths)])}")
    print(f"Total length: {total_length}")
    print(f"Boundaries: {[i * segment_length for i in range(6)]}")
    print(f"\nSampling perplexity at {len(sample_positions)} positions...")

    results = {
        'positions': [],
        'perplexities': [],
        'stds': [],
    }

    for pos in sample_positions:
        region = pos // segment_length + 1
        print(f"  Position {pos} (G{min(region, 5)})...", end=" ", flush=True)

        trial_perplexities = []

        for trial in range(n_trials):
            try:
                np.random.seed(42 + trial * 1000 + pos)
                chain, boundaries = multi_graph.generate_chain(lengths)
                ppl = compute_perplexity_at_position(model, tokenizer, chain, pos)

                if not np.isnan(ppl) and ppl < 1000:  # Filter outliers
                    trial_perplexities.append(ppl)
            except RuntimeError as e:
                print(f"CUDA error at trial {trial}, skipping...", end=" ")
                torch.cuda.empty_cache()
                continue

        # Clear GPU memory periodically
        if pos % 500 == 0:
            torch.cuda.empty_cache()

        if trial_perplexities:
            results['positions'].append(pos)
            results['perplexities'].append(float(np.mean(trial_perplexities)))
            results['stds'].append(float(np.std(trial_perplexities)))
            print(f"ppl={results['perplexities'][-1]:.2f}")
        else:
            print("no valid")

    # Create visualization
    print("\n" + "=" * 70)
    print("Creating visualization...")

    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

    # Main perplexity plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=results['positions'],
        y=results['perplexities'],
        mode='lines+markers',
        name='Perplexity',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6),
        error_y=dict(type='data', array=results['stds'], visible=True, thickness=1),
    ))

    # Add vertical lines at boundaries
    for i in range(1, 5):
        boundary = i * segment_length
        fig.add_vline(x=boundary, line_dash="dash", line_color="gray", line_width=2,
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

    fig.update_layout(
        title=f"Chain Experiment: Perplexity Across Context<br><sup>Chain: G1({segment_length}) → G2({segment_length}) → G3({segment_length}) → G4({segment_length}) → G5({segment_length})</sup>",
        xaxis_title="Position in Context",
        yaxis_title="Perplexity",
        width=1200,
        height=600,
        legend=dict(x=1.02, y=1, xanchor='left'),
    )

    output_dir = Path("results/context_switch")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_dir / "chain_perplexity.html")
    print(f"Saved: {output_dir / 'chain_perplexity.html'}")

    # Create zoomed transition plots
    fig_transitions = make_subplots(
        rows=2, cols=2,
        subplot_titles=['G1→G2 Transition', 'G2→G3 Transition',
                       'G3→G4 Transition', 'G4→G5 Transition'],
        horizontal_spacing=0.1,
        vertical_spacing=0.15,
    )

    transitions = [
        (900, 1100, 'G1→G2', 1000),
        (1900, 2100, 'G2→G3', 2000),
        (2900, 3100, 'G3→G4', 3000),
        (3900, 4100, 'G4→G5', 4000),
    ]

    for idx, (start, end, label, boundary) in enumerate(transitions):
        row, col = idx // 2 + 1, idx % 2 + 1

        # Filter data for this range
        mask = [(start <= p <= end) for p in results['positions']]
        positions = [p for p, m in zip(results['positions'], mask) if m]
        perplexities = [p for p, m in zip(results['perplexities'], mask) if m]
        stds = [s for s, m in zip(results['stds'], mask) if m]

        if positions:
            fig_transitions.add_trace(go.Scatter(
                x=positions,
                y=perplexities,
                mode='lines+markers',
                name=label,
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=8),
                error_y=dict(type='data', array=stds, visible=True),
                showlegend=False,
            ), row=row, col=col)

            # Add vertical line at boundary
            fig_transitions.add_vline(x=boundary, line_dash="dash", line_color="red",
                                     line_width=2, row=row, col=col)

    fig_transitions.update_layout(
        title="Perplexity at Graph Transitions",
        width=1000,
        height=700,
    )

    fig_transitions.write_html(output_dir / "chain_perplexity_transitions.html")
    print(f"Saved: {output_dir / 'chain_perplexity_transitions.html'}")

    # Save JSON
    with open(output_dir / "chain_perplexity_results.json", "w") as f:
        json.dump({
            'segment_length': segment_length,
            'total_length': total_length,
            'n_trials': n_trials,
            'results': results,
        }, f, indent=2)
    print(f"Saved: {output_dir / 'chain_perplexity_results.json'}")

    # Log to W&B
    print("\nLogging to W&B...")
    run = wandb.init(project='icl-structural-influence', name='chain-perplexity-experiment',
                     tags=['chain', '5-graphs', 'perplexity'])
    wandb.log({
        'chain_perplexity': wandb.Html(open(output_dir / 'chain_perplexity.html').read()),
        'chain_perplexity_transitions': wandb.Html(open(output_dir / 'chain_perplexity_transitions.html').read()),
    })
    print(f"Run URL: {run.url}")
    wandb.finish()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("Perplexity should spike at graph transitions as the model")
    print("encounters a new statistical structure that contradicts learned patterns.")
    print("=" * 70)


if __name__ == "__main__":
    main()
