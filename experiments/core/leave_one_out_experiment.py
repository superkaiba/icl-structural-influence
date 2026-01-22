#!/usr/bin/env python3
"""
Leave-One-Out Influence Experiment for ICL Structure Learning.

This script identifies which context examples are most influential for:
1. Phase 1: Overriding pretraining semantics (semantic tokens)
2. Phase 2: Building graph structure (both token types)

Experiments:
A. Position-wise influence map
B. Token type analysis (bridge vs anchor)
C. Phase-specific influence comparison
D. Temporal influence dynamics (Lee et al. style)

Usage:
    python leave_one_out_experiment.py --use-semantic-tokens
    python leave_one_out_experiment.py --use-unrelated-tokens
    python leave_one_out_experiment.py --both
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import from local modules
import sys
sys.path.insert(0, str(Path(__file__).parent))
from run_hierarchy_and_semantic_experiments import SemanticConflictGraph


def load_model(model_name="meta-llama/Llama-3.1-8B", device="cuda"):
    """Load model with hook support."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    # Get number of layers
    if hasattr(model.config, 'num_hidden_layers'):
        n_layers = model.config.num_hidden_layers
    else:
        n_layers = 32  # Default for Llama

    print(f"Model loaded: {n_layers} layers")
    return model, tokenizer, n_layers


def get_token_representations(model, tokenizer, context_tokens, layer_idx):
    """
    Get representations for each unique token in context at specified layer.

    Returns:
        token_reps: dict mapping token -> representation tensor
    """
    # Build prompt from context tokens
    prompt = " ".join(context_tokens)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Hook to capture layer outputs
    representations = {}

    def hook_fn(module, input, output):
        # output is tuple, first element is hidden states
        if isinstance(output, tuple):
            representations['hidden'] = output[0].detach()
        else:
            representations['hidden'] = output.detach()

    # Register hook on target layer
    if layer_idx == 0:
        # Embedding layer
        handle = model.model.embed_tokens.register_forward_hook(hook_fn)
    else:
        # Transformer layer (0-indexed in model, but we use 1-indexed for layers after embedding)
        actual_layer = layer_idx - 1
        handle = model.model.layers[actual_layer].register_forward_hook(hook_fn)

    # Forward pass
    with torch.no_grad():
        model(**inputs)

    handle.remove()

    # Extract token representations
    # Map each context token to its representation
    hidden = representations['hidden'][0]  # Remove batch dim

    # Tokenize each word to find positions
    token_reps = {}
    current_pos = 0

    # Simple approach: use mean of subword tokens for each word
    for i, token in enumerate(context_tokens):
        word_tokens = tokenizer.encode(" " + token, add_special_tokens=False)
        n_subtokens = len(word_tokens)

        if current_pos + n_subtokens <= hidden.shape[0]:
            # Mean pool over subword tokens
            rep = hidden[current_pos:current_pos + n_subtokens].mean(dim=0)

            if token not in token_reps:
                token_reps[token] = []
            token_reps[token].append(rep)

            current_pos += n_subtokens

    # Average representations for repeated tokens
    for token in token_reps:
        token_reps[token] = torch.stack(token_reps[token]).mean(dim=0)

    return token_reps


def compute_pairwise_distances(token_reps, pairs):
    """Compute mean L2 distance for token pairs."""
    distances = []
    for t1, t2 in pairs:
        if t1 in token_reps and t2 in token_reps:
            dist = torch.norm(token_reps[t1] - token_reps[t2]).item()
            distances.append(dist)
    return np.mean(distances) if distances else 0.0


def compute_ratio(model, tokenizer, context_tokens, graph, layer_idx):
    """
    Compute semantic/graph distance ratio for given context.

    Returns:
        ratio: semantic_distance / graph_distance
        l2_sem: mean distance between semantic pairs
        l2_graph: mean distance between graph pairs
    """
    # Get representations
    token_reps = get_token_representations(model, tokenizer, context_tokens, layer_idx)

    # Define pairs
    sem_pairs = [(t1, t2) for t1, t2, _ in graph.get_semantic_pairs()]

    graph_pairs = []
    for cid, toks in graph.graph_clusters.items():
        for i, t1 in enumerate(toks):
            for t2 in toks[i+1:]:
                if graph.token_to_semantic_group[t1] != graph.token_to_semantic_group[t2]:
                    graph_pairs.append((t1, t2))

    # Compute distances
    l2_sem = compute_pairwise_distances(token_reps, sem_pairs)
    l2_graph = compute_pairwise_distances(token_reps, graph_pairs)

    # Use relative epsilon to bound ratio (max ~1000)
    if l2_graph == 0 or np.isnan(l2_sem) or np.isnan(l2_graph):
        ratio = np.nan
    else:
        eps = 1e-3
        min_graph = max(eps * l2_sem, 1e-10)
        l2_graph_bounded = max(l2_graph, min_graph)
        ratio = l2_sem / l2_graph_bounded

    return ratio, l2_sem, l2_graph


def identify_token_types(context_tokens, graph):
    """
    Identify bridge and anchor positions in context.

    Returns:
        bridge_positions: list of positions where cluster changes
        anchor_positions: list of positions within same cluster
        first_occurrences: dict mapping token -> first position
    """
    bridge_positions = []
    anchor_positions = []
    first_occurrences = {}

    prev_cluster = None
    for i, token in enumerate(context_tokens):
        cluster = graph.token_to_graph_cluster.get(token)

        # Track first occurrence
        if token not in first_occurrences:
            first_occurrences[token] = i

        # Classify position
        if prev_cluster is not None and cluster != prev_cluster:
            bridge_positions.append(i)
        else:
            anchor_positions.append(i)

        prev_cluster = cluster

    return bridge_positions, anchor_positions, first_occurrences


def compute_loo_influence(model, tokenizer, context_tokens, graph, layer_idx,
                          positions_to_test=None):
    """
    Compute leave-one-out influence for each position.

    Args:
        positions_to_test: list of positions to test (None = all)

    Returns:
        influences: dict mapping position -> influence value
        full_ratio: ratio with full context
    """
    # Compute full context ratio
    full_ratio, full_sem, full_graph = compute_ratio(
        model, tokenizer, context_tokens, graph, layer_idx
    )

    if positions_to_test is None:
        positions_to_test = list(range(len(context_tokens)))

    influences = {}
    for pos in positions_to_test:
        # Remove position
        loo_context = context_tokens[:pos] + context_tokens[pos+1:]

        if len(loo_context) < 2:  # Need at least 2 tokens
            influences[pos] = 0.0
            continue

        # Compute LOO ratio
        loo_ratio, _, _ = compute_ratio(model, tokenizer, loo_context, graph, layer_idx)

        # Influence = how much removing this position changes the ratio
        # Positive = removing hurts (position was helpful)
        # Negative = removing helps (position was harmful)
        influences[pos] = full_ratio - loo_ratio

    return influences, full_ratio


def run_experiment_a(model, tokenizer, graph, n_layers, context_lengths, n_trials,
                     output_dir, desc=""):
    """
    Experiment A: Position-wise influence map.

    Creates heatmap of influence by position × layer for each context length.
    """
    print(f"\n{'='*60}")
    print(f"EXPERIMENT A: Position-wise Influence Map {desc}")
    print(f"{'='*60}")

    results = {
        'context_lengths': context_lengths,
        'n_trials': n_trials,
        'n_layers': n_layers,
        'influence_by_N': {}
    }

    for N in context_lengths:
        print(f"\nContext length N={N}")

        # Store: layer -> position -> [influences across trials]
        layer_influences = defaultdict(lambda: defaultdict(list))
        bridge_influences = defaultdict(list)
        anchor_influences = defaultdict(list)

        for trial in tqdm(range(n_trials), desc=f"N={N} trials"):
            # Generate random walk
            context, _ = graph.generate_random_walk(length=N, return_nodes=True)
            context_tokens = context.split()

            # Identify token types
            bridge_pos, anchor_pos, _ = identify_token_types(context_tokens, graph)

            # Sample positions for large N
            if N > 100:
                positions_to_test = list(range(0, N, 5))  # Every 5th
            else:
                positions_to_test = list(range(N))

            # Test ALL layers for comprehensive analysis
            layers_to_test = list(range(n_layers + 1))  # 0 to n_layers inclusive

            for layer_idx in layers_to_test:
                influences, full_ratio = compute_loo_influence(
                    model, tokenizer, context_tokens, graph, layer_idx,
                    positions_to_test
                )

                for pos, inf in influences.items():
                    layer_influences[layer_idx][pos].append(inf)

                    # Track by token type
                    if pos in bridge_pos:
                        bridge_influences[layer_idx].append(inf)
                    elif pos in anchor_pos:
                        anchor_influences[layer_idx].append(inf)

        # Aggregate results
        results['influence_by_N'][N] = {
            'layer_position_influence': {
                layer: {pos: np.mean(infs) for pos, infs in pos_dict.items()}
                for layer, pos_dict in layer_influences.items()
            },
            'bridge_mean': {layer: np.mean(infs) for layer, infs in bridge_influences.items()},
            'anchor_mean': {layer: np.mean(infs) for layer, infs in anchor_influences.items()},
            'bridge_std': {layer: np.std(infs) if len(infs) > 1 else 0.0 for layer, infs in bridge_influences.items()},
            'anchor_std': {layer: np.std(infs) if len(infs) > 1 else 0.0 for layer, infs in anchor_influences.items()},
            'bridge_n': {layer: len(infs) for layer, infs in bridge_influences.items()},
            'anchor_n': {layer: len(infs) for layer, infs in anchor_influences.items()},
        }

        # Print summary
        for layer in sorted(bridge_influences.keys()):
            bridge_m = np.mean(bridge_influences[layer]) if bridge_influences[layer] else 0
            anchor_m = np.mean(anchor_influences[layer]) if anchor_influences[layer] else 0
            print(f"  Layer {layer}: Bridge={bridge_m:.4f}, Anchor={anchor_m:.4f}")

    return results


def run_experiment_d(model, tokenizer, graph, n_layers, context_lengths, n_trials,
                     output_dir, desc=""):
    """
    Experiment D: Temporal influence dynamics.

    Track how a fixed position's influence changes as context length grows.
    """
    print(f"\n{'='*60}")
    print(f"EXPERIMENT D: Temporal Influence Dynamics {desc}")
    print(f"{'='*60}")

    # Fixed positions to track
    fixed_positions = [5, 10, 20, 30, 50]

    results = {
        'context_lengths': context_lengths,
        'fixed_positions': fixed_positions,
        'temporal_influence': defaultdict(lambda: defaultdict(list))
    }

    # Test at late layer where structure is strongest
    layer_idx = n_layers  # Last layer

    for N in context_lengths:
        print(f"\nContext length N={N}")

        for trial in tqdm(range(n_trials), desc=f"N={N} trials"):
            # Generate random walk
            context, _ = graph.generate_random_walk(length=N, return_nodes=True)
            context_tokens = context.split()

            # Test fixed positions that exist in this context
            positions_to_test = [p for p in fixed_positions if p < N]

            if not positions_to_test:
                continue

            influences, _ = compute_loo_influence(
                model, tokenizer, context_tokens, graph, layer_idx,
                positions_to_test
            )

            for pos, inf in influences.items():
                results['temporal_influence'][pos][N].append(inf)

    # Aggregate
    aggregated = {
        pos: {N: np.mean(infs) for N, infs in n_dict.items()}
        for pos, n_dict in results['temporal_influence'].items()
    }
    results['aggregated'] = aggregated

    # Print summary
    print("\nTemporal Influence (position → N → mean influence):")
    for pos in sorted(aggregated.keys()):
        values = [f"N={N}:{v:.4f}" for N, v in sorted(aggregated[pos].items())]
        print(f"  Position {pos}: {', '.join(values)}")

    return results


def create_visualizations(results_semantic, results_unrelated, output_dir):
    """Generate comparison visualizations."""
    print("\nGenerating visualizations...")

    # Experiment A: Bridge vs Anchor comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Select interesting N values for visualization (phase transition region)
    display_Ns = [4, 7, 10, 20, 50, 100]

    for idx, (results, title) in enumerate([
        (results_semantic, "Semantic Tokens"),
        (results_unrelated, "Unrelated Tokens")
    ]):
        ax = axes[idx]

        if 'exp_a' in results:
            # Sort N values numerically
            all_Ns = sorted([int(k) for k in results['exp_a']['influence_by_N'].keys()])

            # Filter to display_Ns that exist in data
            Ns_to_plot = [n for n in display_Ns if n in all_Ns]

            for N in Ns_to_plot:
                N_str = str(N)
                data = results['exp_a']['influence_by_N'][N_str]

                # Skip if no bridge data (too few examples)
                if not data.get('bridge_mean'):
                    continue

                # Sort layers numerically
                layers = sorted([int(l) for l in data['bridge_mean'].keys()])
                bridge = [data['bridge_mean'][str(l)] for l in layers]
                anchor = [data['anchor_mean'].get(str(l), float('nan')) for l in layers]

                ax.plot(layers, bridge, 'o-', label=f'Bridge N={N}', alpha=0.7)
                ax.plot(layers, anchor, 's--', label=f'Anchor N={N}', alpha=0.7)

        ax.set_xlabel('Layer')
        ax.set_ylabel('Mean LOO Influence')
        ax.set_title(f'{title}\nBridge vs Anchor Token Influence')
        ax.legend(fontsize=8)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'bridge_vs_anchor_comparison.png', dpi=150)
    plt.savefig(output_dir / 'bridge_vs_anchor_comparison.pdf')
    print(f"Saved: {output_dir / 'bridge_vs_anchor_comparison.png'}")

    # Experiment D: Temporal dynamics
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (results, title) in enumerate([
        (results_semantic, "Semantic Tokens"),
        (results_unrelated, "Unrelated Tokens")
    ]):
        ax = axes[idx]

        if 'exp_d' in results and 'aggregated' in results['exp_d']:
            for pos_str, n_dict in results['exp_d']['aggregated'].items():
                pos = int(pos_str)
                # Sort N values numerically
                Ns = sorted([int(k) for k in n_dict.keys()])
                values = [n_dict[str(N)] for N in Ns]
                ax.plot(Ns, values, 'o-', label=f'Position {pos}', markersize=4)

        ax.set_xlabel('Context Length (N)')
        ax.set_ylabel('LOO Influence')
        ax.set_title(f'{title}\nTemporal Influence Dynamics')
        ax.legend(fontsize=8)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'temporal_dynamics_comparison.png', dpi=150)
    plt.savefig(output_dir / 'temporal_dynamics_comparison.pdf')
    print(f"Saved: {output_dir / 'temporal_dynamics_comparison.png'}")


def main():
    parser = argparse.ArgumentParser(description="Leave-One-Out Influence Experiment")
    parser.add_argument("--use-semantic-tokens", action="store_true",
                        help="Run with semantic tokens (cat, dog, bird...)")
    parser.add_argument("--use-unrelated-tokens", action="store_true",
                        help="Run with unrelated tokens (piano, river, hammer...)")
    parser.add_argument("--both", action="store_true",
                        help="Run with both token sets")
    parser.add_argument("--context-lengths", type=str, default="20,50,100,200",
                        help="Comma-separated context lengths")
    parser.add_argument("--n-trials", type=int, default=5,
                        help="Number of trials per condition")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B")
    args = parser.parse_args()

    # Parse context lengths
    context_lengths = [int(x) for x in args.context_lengths.split(",")]

    # Determine which token sets to run
    run_semantic = args.use_semantic_tokens or args.both
    run_unrelated = args.use_unrelated_tokens or args.both

    if not run_semantic and not run_unrelated:
        run_semantic = True  # Default to semantic

    # Setup output directory
    output_dir = Path("results/loo_experiments")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, tokenizer, n_layers = load_model(args.model)

    results_semantic = {}
    results_unrelated = {}

    # Run with semantic tokens
    if run_semantic:
        print("\n" + "="*70)
        print("RUNNING WITH SEMANTIC TOKENS")
        print("="*70)

        graph = SemanticConflictGraph(seed=42, use_semantic_tokens=True)

        sem_dir = output_dir / "semantic_tokens"
        sem_dir.mkdir(exist_ok=True)

        results_semantic['exp_a'] = run_experiment_a(
            model, tokenizer, graph, n_layers, context_lengths, args.n_trials,
            sem_dir, desc="(Semantic)"
        )

        results_semantic['exp_d'] = run_experiment_d(
            model, tokenizer, graph, n_layers, context_lengths, args.n_trials,
            sem_dir, desc="(Semantic)"
        )

        # Save results
        with open(sem_dir / "loo_results.json", "w") as f:
            json.dump(results_semantic, f, indent=2, default=str)
        print(f"Saved: {sem_dir / 'loo_results.json'}")

    # Run with unrelated tokens
    if run_unrelated:
        print("\n" + "="*70)
        print("RUNNING WITH UNRELATED TOKENS")
        print("="*70)

        graph = SemanticConflictGraph(seed=42, use_semantic_tokens=False)

        unrel_dir = output_dir / "unrelated_tokens"
        unrel_dir.mkdir(exist_ok=True)

        results_unrelated['exp_a'] = run_experiment_a(
            model, tokenizer, graph, n_layers, context_lengths, args.n_trials,
            unrel_dir, desc="(Unrelated)"
        )

        results_unrelated['exp_d'] = run_experiment_d(
            model, tokenizer, graph, n_layers, context_lengths, args.n_trials,
            unrel_dir, desc="(Unrelated)"
        )

        # Save results
        with open(unrel_dir / "loo_results.json", "w") as f:
            json.dump(results_unrelated, f, indent=2, default=str)
        print(f"Saved: {unrel_dir / 'loo_results.json'}")

    # Generate comparison visualizations if both were run
    if run_semantic and run_unrelated:
        create_visualizations(results_semantic, results_unrelated, output_dir)

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
