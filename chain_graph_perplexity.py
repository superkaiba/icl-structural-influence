#!/usr/bin/env python3
"""
Chain Graph-Specific Perplexity: Measure prediction accuracy for each graph's structure.

For each graph Gi, we measure: "How much probability mass does the model put on
tokens that are in the SAME CLUSTER as the current token, according to Gi's clustering?"

High same-cluster probability = model has learned Gi's structure
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

        self.clusters = []
        self.adjacencies = []
        self.token_to_cluster = []  # For each graph, map token -> cluster id

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

            # Build token -> cluster mapping
            t2c = {t: c for c, toks in clusters.items() for t in toks}
            self.token_to_cluster.append(t2c)

            print(f"Graph {i+1} clusters:")
            for cid, toks in clusters.items():
                print(f"  {cid}: {toks}")
            print()

    def _build_adjacency(self, clusters, p_in=0.9, p_out=0.1):
        t2c = {t: c for c, toks in clusters.items() for t in toks}
        return {t1: {t2: (p_in if t2c[t1] == t2c[t2] else p_out)
                     for t2 in self.tokens if t1 != t2} for t1 in self.tokens}

    def get_same_cluster_tokens(self, graph_idx, token):
        """Get tokens in the same cluster as `token` according to graph `graph_idx`."""
        cluster_id = self.token_to_cluster[graph_idx][token]
        return self.clusters[graph_idx][cluster_id]

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
        chain = []
        boundaries = [0]
        for i, length in enumerate(lengths):
            walk = self.generate_walk(i, length)
            chain.extend(walk)
            boundaries.append(boundaries[-1] + length)
        return chain, boundaries


def compute_graph_specific_metrics(model, tokenizer, chain, position, multi_graph, window=50):
    """
    At a given position, compute for each graph:
    1. Same-cluster probability: P(next token is in same cluster | current token)
    2. Graph-specific perplexity: -log P averaged over same-cluster tokens
    """
    start_pos = max(0, position - window)
    context = ' '.join(chain[start_pos:position+1])

    tokens = tokenizer.encode(context, add_special_tokens=False)
    if len(tokens) < 2:
        return {f'G{i+1}': {'same_cluster_prob': np.nan, 'cross_cluster_prob': np.nan} for i in range(5)}

    input_ids = torch.tensor([tokens]).to(model.device)

    with torch.no_grad():
        outputs = model(input_ids)
        # Get logits for the last position (predicting next token)
        logits = outputs.logits[0, -1]  # Shape: [vocab_size]
        probs = F.softmax(logits, dim=-1)

    # Get current token (the one we're predicting FROM)
    current_word = chain[position]

    # For each graph, compute probability mass on same-cluster vs cross-cluster tokens
    results = {}

    for graph_idx in range(5):
        same_cluster_tokens = multi_graph.get_same_cluster_tokens(graph_idx, current_word)
        other_tokens = [t for t in multi_graph.tokens if t not in same_cluster_tokens]

        # Get token IDs for same-cluster and cross-cluster tokens
        same_cluster_prob = 0.0
        cross_cluster_prob = 0.0

        for word in same_cluster_tokens:
            if word != current_word:  # Exclude current token
                word_tokens = tokenizer.encode(word, add_special_tokens=False)
                if word_tokens:
                    # Use first token's probability
                    same_cluster_prob += probs[word_tokens[0]].item()

        for word in other_tokens:
            word_tokens = tokenizer.encode(word, add_special_tokens=False)
            if word_tokens:
                cross_cluster_prob += probs[word_tokens[0]].item()

        # Normalize to get ratio
        total_relevant_prob = same_cluster_prob + cross_cluster_prob
        if total_relevant_prob > 0:
            same_cluster_ratio = same_cluster_prob / total_relevant_prob
            cross_cluster_ratio = cross_cluster_prob / total_relevant_prob
        else:
            same_cluster_ratio = np.nan
            cross_cluster_ratio = np.nan

        results[f'G{graph_idx+1}'] = {
            'same_cluster_prob': same_cluster_prob,
            'cross_cluster_prob': cross_cluster_prob,
            'same_cluster_ratio': same_cluster_ratio,
        }

    return results


def main():
    print("=" * 70)
    print("CHAIN GRAPH-SPECIFIC PERPLEXITY: G1 → G2 → G3 → G4 → G5")
    print("=" * 70)
    print("Measuring: P(same-cluster token | current token) for each graph")
    print("=" * 70)

    hooked = HookedLLM.from_pretrained('meta-llama/Llama-3.1-8B', torch_dtype=torch.float16, device_map='auto')
    model, tokenizer = hooked.model, hooked.tokenizer
    model.eval()

    multi_graph = MultiGraphChain(seed=42)

    segment_length = 1000
    lengths = [segment_length] * 5
    total_length = sum(lengths)

    n_trials = 10

    # Sample positions every 50 tokens
    sample_positions = list(range(50, total_length, 50))

    print(f"\nChain structure: {' → '.join([f'G{i+1}({l})' for i, l in enumerate(lengths)])}")
    print(f"Sampling at {len(sample_positions)} positions...")

    # Results structure
    results = {f'G{i+1}': {'positions': [], 'same_cluster_ratios': [], 'stds': []} for i in range(5)}

    for pos in sample_positions:
        region = pos // segment_length + 1
        print(f"  Position {pos} (G{min(region, 5)})...", end=" ", flush=True)

        trial_results = {f'G{i+1}': [] for i in range(5)}

        for trial in range(n_trials):
            try:
                np.random.seed(42 + trial * 1000 + pos)
                chain, boundaries = multi_graph.generate_chain(lengths)
                metrics = compute_graph_specific_metrics(model, tokenizer, chain, pos, multi_graph)

                for i in range(5):
                    key = f'G{i+1}'
                    ratio = metrics[key]['same_cluster_ratio']
                    if not np.isnan(ratio):
                        trial_results[key].append(ratio)
            except RuntimeError as e:
                print(f"CUDA error, skipping...", end=" ")
                torch.cuda.empty_cache()
                continue

        if pos % 500 == 0:
            torch.cuda.empty_cache()

        for i in range(5):
            key = f'G{i+1}'
            if trial_results[key]:
                results[key]['positions'].append(pos)
                results[key]['same_cluster_ratios'].append(float(np.mean(trial_results[key])))
                results[key]['stds'].append(float(np.std(trial_results[key])))

        # Print active graph's same-cluster ratio
        active_key = f'G{min(region, 5)}'
        if trial_results[active_key]:
            print(f"G{min(region, 5)} ratio={np.mean(trial_results[active_key]):.3f}")
        else:
            print("no valid")

    # Create visualization
    print("\n" + "=" * 70)
    print("Creating visualization...")

    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

    # Create 4 subplots: G1&G2, G2&G3, G3&G4, G4&G5
    fig_pairs = make_subplots(
        rows=4, cols=1,
        subplot_titles=['G1 vs G2', 'G2 vs G3', 'G3 vs G4', 'G4 vs G5'],
        vertical_spacing=0.08,
        shared_xaxes=True,
    )

    pairs = [(0, 1), (1, 2), (2, 3), (3, 4)]  # Graph index pairs

    for row_idx, (g1_idx, g2_idx) in enumerate(pairs):
        row = row_idx + 1

        # Plot both graphs in this pair
        for graph_idx in [g1_idx, g2_idx]:
            key = f'G{graph_idx+1}'
            fig_pairs.add_trace(go.Scatter(
                x=results[key]['positions'],
                y=results[key]['same_cluster_ratios'],
                mode='lines+markers',
                name=f'G{graph_idx+1}',
                line=dict(color=colors[graph_idx], width=2),
                marker=dict(size=4),
                showlegend=(row_idx == 0),
            ), row=row, col=1)

        # Add baseline
        fig_pairs.add_hline(y=0.2, line_dash="dot", line_color="gray", line_width=1,
                           row=row, col=1)

        # Add vertical lines at all boundaries
        for i in range(1, 5):
            boundary = i * segment_length
            fig_pairs.add_vline(x=boundary, line_dash="dash", line_color="gray",
                               line_width=1, row=row, col=1)

        # Add shaded regions for the two graphs in this pair
        for graph_idx in [g1_idx, g2_idx]:
            fig_pairs.add_vrect(
                x0=graph_idx * segment_length,
                x1=(graph_idx + 1) * segment_length,
                fillcolor=colors[graph_idx],
                opacity=0.15,
                line_width=0,
                row=row, col=1,
            )

    fig_pairs.update_xaxes(range=[0, 5000])
    fig_pairs.update_layout(
        title="Graph-Specific Prediction: Same-Cluster Probability (Pairwise)<br>" +
              "<sup>Each subplot shows adjacent graph pairs across full context</sup>",
        width=1200,
        height=900,
        legend=dict(x=1.02, y=1, xanchor='left'),
    )
    fig_pairs.update_xaxes(title_text="Position in Context", row=4, col=1)

    output_dir = Path("results/context_switch")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_pairs.write_html(output_dir / "chain_graph_perplexity.html")
    print(f"Saved: {output_dir / 'chain_graph_perplexity.html'}")

    # Create zoomed transition plots
    fig_transitions = make_subplots(
        rows=2, cols=2,
        subplot_titles=['G1→G2 Transition', 'G2→G3 Transition',
                       'G3→G4 Transition', 'G4→G5 Transition'],
        horizontal_spacing=0.1,
        vertical_spacing=0.15,
    )

    transitions = [
        (900, 1100, 1000, 0, 1),  # boundary, old_graph_idx, new_graph_idx
        (1900, 2100, 2000, 1, 2),
        (2900, 3100, 3000, 2, 3),
        (3900, 4100, 4000, 3, 4),
    ]

    for idx, (start, end, boundary, old_idx, new_idx) in enumerate(transitions):
        row, col = idx // 2 + 1, idx % 2 + 1

        # Plot old and new graph's same-cluster ratios
        for graph_idx, graph_color in [(old_idx, colors[old_idx]), (new_idx, colors[new_idx])]:
            key = f'G{graph_idx+1}'
            mask = [(start <= p <= end) for p in results[key]['positions']]
            positions = [p for p, m in zip(results[key]['positions'], mask) if m]
            ratios = [r for r, m in zip(results[key]['same_cluster_ratios'], mask) if m]

            if positions:
                fig_transitions.add_trace(go.Scatter(
                    x=positions,
                    y=ratios,
                    mode='lines+markers',
                    name=f'G{graph_idx+1}',
                    line=dict(color=graph_color, width=2),
                    marker=dict(size=6),
                    showlegend=(idx == 0),
                ), row=row, col=col)

        fig_transitions.add_vline(x=boundary, line_dash="dash", line_color="black",
                                 line_width=2, row=row, col=col)

    fig_transitions.update_layout(
        title="Same-Cluster Ratio at Graph Transitions<br><sup>Old graph (leaving) vs New graph (entering)</sup>",
        width=1000,
        height=700,
    )

    fig_transitions.write_html(output_dir / "chain_graph_perplexity_transitions.html")
    print(f"Saved: {output_dir / 'chain_graph_perplexity_transitions.html'}")

    # Save JSON
    with open(output_dir / "chain_graph_perplexity_results.json", "w") as f:
        json.dump({
            'segment_length': segment_length,
            'total_length': total_length,
            'n_trials': n_trials,
            'results': results,
        }, f, indent=2)
    print(f"Saved: {output_dir / 'chain_graph_perplexity_results.json'}")

    # Log to W&B
    print("\nLogging to W&B...")
    run = wandb.init(project='icl-structural-influence', name='chain-graph-specific-perplexity',
                     tags=['chain', '5-graphs', 'graph-perplexity'])
    wandb.log({
        'graph_perplexity': wandb.Html(open(output_dir / 'chain_graph_perplexity.html').read()),
        'graph_perplexity_transitions': wandb.Html(open(output_dir / 'chain_graph_perplexity_transitions.html').read()),
    })
    print(f"Run URL: {run.url}")
    wandb.finish()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("For each graph Gi, we measure P(same-cluster token) / P(all 16 tokens)")
    print("- Random baseline: 3/15 ≈ 0.2 (3 same-cluster tokens out of 15 others)")
    print("- High ratio = model has learned that graph's clustering structure")
    print("- The ACTIVE graph should have highest ratio in its region")
    print("=" * 70)


if __name__ == "__main__":
    main()
