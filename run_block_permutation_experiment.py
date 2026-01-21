#!/usr/bin/env python3
"""
Block Permutation / Full Shuffle Experiment: Testing if Sequential Order Matters for ICL

Research Question:
    Can the model learn hierarchical graph structure from the same tokens
    presented in a different order?

Design:
    Control:    Natural random walk (respects graph structure)
    Treatment:  Block-permuted walk (same tokens, blocks shuffled)
             OR Full shuffle (all tokens randomly reordered)

Permutation Modes:
    - 'block': Shuffle block order, preserve local edges within blocks
    - 'full_shuffle': Completely randomize all token positions (destroys ALL structure)

Example (Block Permutation):
    Natural:   [A1 A2 A3] [B1 B2] [A4 A5] [C1 C2 C3]
    Permuted:  [C1 C2 C3] [A1 A2 A3] [B1 B2] [A4 A5]
    (Consecutive tokens within blocks are still graph neighbors)

Example (Full Shuffle):
    Natural:   apple → truck → sand → river → lamp  (all edges valid)
    Shuffled:  river → apple → lamp → sand → truck  (most pairs NOT edges)
    (Destroys all sequential graph structure)

Hypothesis:
    Block permutation: Mild effect (local structure preserved)
    Full shuffle: Strong effect (no sequential structure) - should show >>3x ratio

Measurements:
    1. Cluster separation Φ across context lengths [10,20,50,100,200]
    2. MDS trajectories showing geometric structure
    3. CSS scores identifying influential tokens
    4. Per-token perplexity comparison

Usage:
    python run_block_permutation_experiment.py --model gpt2 --n-samples 50
    python run_block_permutation_experiment.py --model gpt2 --permutation-mode full_shuffle
    python run_block_permutation_experiment.py --model meta-llama/Llama-3.1-8B
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
import numpy as np
try:
    import matplotlib.pyplot as plt
    from sklearn.manifold import MDS
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/sklearn not available, plotting disabled")

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable, desc=""):
        return iterable

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data import HierarchicalGraph, HierarchicalGraphConfig
from src.models import HookedLLM
from src.metrics import ClusterSeparation


# =============================================================================
# Block Identification and Permutation
# =============================================================================

def identify_blocks(node_sequence, graph):
    """
    Identify maximal consecutive sequences (blocks) from the same cluster.

    Args:
        node_sequence: List of node indices from the walk
        graph: HierarchicalGraph instance

    Returns:
        List of blocks, where each block is (start_idx, end_idx, cluster_id)
    """
    if not node_sequence:
        return []

    blocks = []
    current_cluster = graph.get_cluster(node_sequence[0])
    block_start = 0

    for i in range(1, len(node_sequence)):
        node_cluster = graph.get_cluster(node_sequence[i])

        if node_cluster != current_cluster:
            # End of current block
            blocks.append((block_start, i, current_cluster))
            block_start = i
            current_cluster = node_cluster

    # Add final block
    blocks.append((block_start, len(node_sequence), current_cluster))

    return blocks


def permute_blocks(node_sequence, blocks, seed=None):
    """
    Permute the order of blocks while keeping internal block structure.

    Args:
        node_sequence: Original node sequence
        blocks: List of (start, end, cluster_id) tuples
        seed: Random seed for reproducibility

    Returns:
        Permuted node sequence
    """
    rng = np.random.default_rng(seed)

    # Extract block sequences
    block_sequences = []
    for start, end, cluster_id in blocks:
        block_sequences.append(node_sequence[start:end])

    # Permute block order
    permuted_indices = rng.permutation(len(block_sequences))

    # Reconstruct sequence
    permuted_sequence = []
    for idx in permuted_indices:
        permuted_sequence.extend(block_sequences[idx])

    return permuted_sequence


def shuffle_all_tokens(node_sequence, seed=None):
    """
    Completely randomize token order (destroys ALL graph structure).

    Unlike block permutation which preserves local edges within blocks,
    full shuffle destroys ALL sequential structure. Consecutive tokens
    in the shuffled sequence are unlikely to be graph neighbors.

    Args:
        node_sequence: Original node sequence (list of node indices)
        seed: Random seed for reproducibility

    Returns:
        Shuffled node sequence (same tokens, random order)
    """
    rng = np.random.default_rng(seed)
    shuffled = list(node_sequence)
    rng.shuffle(shuffled)
    return shuffled


def sequence_to_prompt(node_sequence, graph):
    """Convert node sequence to space-separated token string."""
    tokens = [graph.node_to_token[node] for node in node_sequence]
    return " ".join(tokens)


def align_cluster_labels_with_tokens(model, prompt, node_sequence, graph):
    """
    Align cluster labels with actual tokenized sequence.

    The tokenizer may split words into multiple subword tokens.
    This function creates a cluster label for each actual token.

    Args:
        model: HookedLLM instance
        prompt: Space-separated token string
        node_sequence: List of node indices
        graph: HierarchicalGraph instance

    Returns:
        List of cluster labels aligned with tokenized sequence
    """
    # Get actual tokens from model
    tokens = model.tokenizer.encode(prompt, return_tensors="pt")[0]
    num_tokens = len(tokens)

    # Get word tokens
    word_tokens = [graph.node_to_token[node] for node in node_sequence]
    cluster_ids = [graph.get_cluster(node) for node in node_sequence]

    # Build mapping: for each tokenized position, find which word it belongs to
    # Decode each token and match to words
    aligned_labels = []

    # Re-tokenize each word separately to understand the mapping
    word_start_positions = []
    current_pos = 0

    for word in word_tokens:
        # Tokenize this word (with space prefix for proper BPE)
        word_toks = model.tokenizer.encode(" " + word, add_special_tokens=False)
        word_len = len(word_toks)
        word_start_positions.append((current_pos, current_pos + word_len))
        current_pos += word_len

    # Handle possible BOS token
    # Check if first token is BOS or if there's offset
    decoded_tokens = [model.tokenizer.decode([t]) for t in tokens]

    # Simple approach: distribute cluster labels proportionally
    # For each token position, find which word it most likely belongs to
    labels_per_token = []

    # Method: tokenize the full prompt and use word boundaries
    word_idx = 0
    accumulated_len = 0

    # For GPT-2 style tokenizers, words are typically preceded by space (Ġ)
    # Let's use a simpler approach: tokenize and track
    for tok_idx, tok_id in enumerate(tokens):
        # Find which word this token belongs to based on position
        # Calculate approximate position
        if word_idx < len(word_tokens):
            labels_per_token.append(cluster_ids[word_idx])

            # Check if we should move to next word
            # Decode accumulated tokens and see if we've passed the word
            prefix_tokens = tokens[:tok_idx + 1]
            prefix_text = model.tokenizer.decode(prefix_tokens)

            # Count spaces to estimate word position
            space_count = prefix_text.count(' ')
            if space_count > word_idx and word_idx < len(word_tokens) - 1:
                word_idx = min(space_count, len(word_tokens) - 1)
        else:
            # Past all words, use last cluster
            labels_per_token.append(cluster_ids[-1])

    return labels_per_token


# =============================================================================
# Representation Extraction
# =============================================================================

def extract_representations(model, prompt, layer_idx=-5):
    """
    Extract layer representations and compute per-token losses.

    Args:
        model: HookedLLM instance
        prompt: Space-separated token string
        layer_idx: Which layer to extract (negative indexing from end)

    Returns:
        representations: (seq_len, hidden_dim) tensor
        losses: (seq_len,) tensor of per-token losses
    """
    # Convert negative layer index to positive (hooked model does this internally)
    actual_layer = layer_idx if layer_idx >= 0 else model.num_layers + layer_idx

    output, cache = model.forward_with_cache(prompt, layers=[layer_idx])
    representations = cache.get_residual_stream(actual_layer)

    if representations is None:
        raise ValueError(f"Failed to extract representations at layer {layer_idx} (actual: {actual_layer})")

    # Remove batch dimension
    representations = representations.squeeze(0)

    # Compute per-token losses
    losses = model.compute_per_token_loss(prompt)

    return representations, losses


def compute_cluster_separation(representations, cluster_labels):
    """
    Compute cluster separation metric.

    Args:
        representations: (seq_len, hidden_dim)
        cluster_labels: (seq_len,) cluster assignments

    Returns:
        Separation score (float)
    """
    metric = ClusterSeparation()
    return metric.compute(representations, cluster_labels)


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment(
    model_name="gpt2",
    context_lengths=[10, 20, 50, 100, 200],
    n_samples=50,
    layer_idx=-5,
    graph_config=None,
    output_dir="results/block_permutation",
    seed=42,
    permutation_mode="block"
):
    """
    Run permutation experiment across multiple context lengths.

    Args:
        model_name: HuggingFace model identifier
        context_lengths: List of context lengths to test
        n_samples: Number of random walks per condition per length
        layer_idx: Which layer to analyze
        graph_config: HierarchicalGraphConfig (uses default if None)
        output_dir: Where to save results
        seed: Random seed
        permutation_mode: Type of permutation ('block' or 'full_shuffle')
            - 'block': Shuffle block order, preserve local edges
            - 'full_shuffle': Completely randomize all token positions

    Returns:
        results: Dict with natural and permuted condition results
    """
    mode_names = {
        'block': 'BLOCK PERMUTATION',
        'full_shuffle': 'FULL SHUFFLE (INVALID WALK)'
    }
    experiment_name = mode_names.get(permutation_mode, 'PERMUTATION')

    print("=" * 80)
    print(f"{experiment_name} EXPERIMENT")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Permutation mode: {permutation_mode}")
    print(f"Context lengths: {context_lengths}")
    print(f"Samples per condition: {n_samples}")
    print(f"Layer: {layer_idx}")
    print(f"Random seed: {seed}")
    print()

    # Setup
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize graph
    if graph_config is None:
        graph_config = HierarchicalGraphConfig(
            num_superclusters=3,
            nodes_per_cluster=5,
            p_intra_cluster=0.8,
            p_inter_cluster=0.1,
            seed=seed
        )

    graph = HierarchicalGraph(graph_config)
    print(f"Graph: {graph.num_nodes} nodes, {graph_config.num_superclusters} clusters")
    print()

    # Load model
    print("Loading model...")
    model = HookedLLM.from_pretrained(model_name)
    print(f"Model loaded: {model.model.config.num_hidden_layers} layers")
    print()

    # Storage for results
    results = {
        'natural': defaultdict(list),    # cluster_separation values
        'permuted': defaultdict(list),
        'natural_perplexity': defaultdict(list),
        'permuted_perplexity': defaultdict(list),
        'context_lengths': context_lengths,
        'metadata': {
            'model': model_name,
            'layer': layer_idx,
            'n_samples': n_samples,
            'seed': seed,
            'permutation_mode': permutation_mode,
            'timestamp': datetime.now().isoformat()
        }
    }

    # Run experiment for each context length
    for N in context_lengths:
        print(f"Context Length N={N}")
        print("-" * 40)

        natural_separations = []
        permuted_separations = []
        natural_perplexities = []
        permuted_perplexities = []

        for sample_idx in tqdm(range(n_samples), desc=f"N={N}"):
            # Generate natural random walk
            prompt_natural, node_seq_natural = graph.generate_random_walk(
                length=N,
                return_nodes=True
            )

            # Create permuted version based on mode
            if permutation_mode == 'block':
                # Block permutation: shuffle block order, preserve local edges
                blocks = identify_blocks(node_seq_natural, graph)
                node_seq_permuted = permute_blocks(
                    node_seq_natural,
                    blocks,
                    seed=seed + sample_idx  # Vary seed per sample
                )
            else:  # full_shuffle
                # Full shuffle: completely randomize all token positions
                node_seq_permuted = shuffle_all_tokens(
                    node_seq_natural,
                    seed=seed + sample_idx
                )
            prompt_permuted = sequence_to_prompt(node_seq_permuted, graph)

            # Extract representations - NATURAL
            try:
                reps_natural, losses_natural = extract_representations(
                    model, prompt_natural, layer_idx
                )

                # Get cluster labels aligned with tokenized sequence
                cluster_labels_natural = align_cluster_labels_with_tokens(
                    model, prompt_natural, node_seq_natural, graph
                )

                # Ensure labels match representation length
                if len(cluster_labels_natural) != reps_natural.shape[0]:
                    # Truncate or pad to match
                    if len(cluster_labels_natural) > reps_natural.shape[0]:
                        cluster_labels_natural = cluster_labels_natural[:reps_natural.shape[0]]
                    else:
                        cluster_labels_natural = cluster_labels_natural + [cluster_labels_natural[-1]] * (reps_natural.shape[0] - len(cluster_labels_natural))

                sep_natural = compute_cluster_separation(
                    reps_natural,
                    torch.tensor(cluster_labels_natural)
                )
                natural_separations.append(sep_natural)

                # Mean perplexity
                perplexity_natural = torch.exp(losses_natural.mean()).item()
                natural_perplexities.append(perplexity_natural)

            except Exception as e:
                print(f"Warning: Natural sample {sample_idx} failed: {e}")
                continue

            # Extract representations - PERMUTED
            try:
                reps_permuted, losses_permuted = extract_representations(
                    model, prompt_permuted, layer_idx
                )

                # Get cluster labels aligned with tokenized sequence
                cluster_labels_permuted = align_cluster_labels_with_tokens(
                    model, prompt_permuted, node_seq_permuted, graph
                )

                # Ensure labels match representation length
                if len(cluster_labels_permuted) != reps_permuted.shape[0]:
                    if len(cluster_labels_permuted) > reps_permuted.shape[0]:
                        cluster_labels_permuted = cluster_labels_permuted[:reps_permuted.shape[0]]
                    else:
                        cluster_labels_permuted = cluster_labels_permuted + [cluster_labels_permuted[-1]] * (reps_permuted.shape[0] - len(cluster_labels_permuted))

                sep_permuted = compute_cluster_separation(
                    reps_permuted,
                    torch.tensor(cluster_labels_permuted)
                )
                permuted_separations.append(sep_permuted)

                # Mean perplexity
                perplexity_permuted = torch.exp(losses_permuted.mean()).item()
                permuted_perplexities.append(perplexity_permuted)

            except Exception as e:
                print(f"Warning: Permuted sample {sample_idx} failed: {e}")
                continue

        # Store results
        results['natural'][N] = natural_separations
        results['permuted'][N] = permuted_separations
        results['natural_perplexity'][N] = natural_perplexities
        results['permuted_perplexity'][N] = permuted_perplexities

        # Print summary
        mean_nat = np.mean(natural_separations) if natural_separations else 0
        mean_perm = np.mean(permuted_separations) if permuted_separations else 0
        std_nat = np.std(natural_separations) if natural_separations else 0
        std_perm = np.std(permuted_separations) if permuted_separations else 0

        print(f"  Natural:  Φ = {mean_nat:.3f} ± {std_nat:.3f}")
        print(f"  Permuted: Φ = {mean_perm:.3f} ± {std_perm:.3f}")
        print(f"  Ratio:    {mean_nat / mean_perm:.2f}x" if mean_perm > 0 else "  Ratio: N/A")
        print()

    # Save results
    output_file = output_path / f"{permutation_mode}_permutation_{model_name.replace('/', '_')}.json"

    # Convert defaultdicts to regular dicts for JSON serialization
    results_serializable = {
        'natural': dict(results['natural']),
        'permuted': dict(results['permuted']),
        'natural_perplexity': dict(results['natural_perplexity']),
        'permuted_perplexity': dict(results['permuted_perplexity']),
        'context_lengths': results['context_lengths'],
        'metadata': results['metadata']
    }

    with open(output_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)

    print(f"Results saved to {output_file}")

    return results


# =============================================================================
# Visualization
# =============================================================================

def plot_results(results, output_dir="results/block_permutation"):
    """
    Generate comparison plots for natural vs permuted conditions.

    Creates:
        1. Cluster separation trajectory plot
        2. Perplexity comparison plot
    """
    if not PLOTTING_AVAILABLE:
        print("Plotting libraries not available, skipping visualization")
        return

    output_path = Path(output_dir)
    context_lengths = results['context_lengths']

    # Get permutation mode from metadata (default to 'block' for backward compatibility)
    permutation_mode = results.get('metadata', {}).get('permutation_mode', 'block')

    # Set labels and titles based on mode
    if permutation_mode == 'full_shuffle':
        permuted_label = 'Full Shuffle'
        title_suffix = 'Full Shuffle'
    else:
        permuted_label = 'Block-Permuted'
        title_suffix = 'Block Permutation'

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Cluster Separation
    ax = axes[0]

    natural_means = [np.mean(results['natural'][N]) for N in context_lengths]
    natural_stds = [np.std(results['natural'][N]) for N in context_lengths]
    permuted_means = [np.mean(results['permuted'][N]) for N in context_lengths]
    permuted_stds = [np.std(results['permuted'][N]) for N in context_lengths]

    ax.errorbar(context_lengths, natural_means, yerr=natural_stds,
                marker='o', linewidth=2, capsize=5, label='Natural Walk',
                color='#2E86AB')
    ax.errorbar(context_lengths, permuted_means, yerr=permuted_stds,
                marker='s', linewidth=2, capsize=5, label=permuted_label,
                color='#A23B72')

    ax.set_xlabel('Context Length (N)', fontsize=12)
    ax.set_ylabel('Cluster Separation (Φ)', fontsize=12)
    ax.set_title(f'Effect of {title_suffix} on Structure Learning', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # Plot 2: Perplexity
    ax = axes[1]

    nat_ppl_means = [np.mean(results['natural_perplexity'][N]) for N in context_lengths]
    nat_ppl_stds = [np.std(results['natural_perplexity'][N]) for N in context_lengths]
    perm_ppl_means = [np.mean(results['permuted_perplexity'][N]) for N in context_lengths]
    perm_ppl_stds = [np.std(results['permuted_perplexity'][N]) for N in context_lengths]

    ax.errorbar(context_lengths, nat_ppl_means, yerr=nat_ppl_stds,
                marker='o', linewidth=2, capsize=5, label='Natural Walk',
                color='#2E86AB')
    ax.errorbar(context_lengths, perm_ppl_means, yerr=perm_ppl_stds,
                marker='s', linewidth=2, capsize=5, label=permuted_label,
                color='#A23B72')

    ax.set_xlabel('Context Length (N)', fontsize=12)
    ax.set_ylabel('Perplexity', fontsize=12)
    ax.set_title('Prediction Difficulty Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    # Save with mode-specific filename
    output_file = output_path / f"{permutation_mode}_permutation_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")

    plt.close()


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Block permutation experiment for ICL structure learning"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt2',
        help='HuggingFace model name (default: gpt2)'
    )
    parser.add_argument(
        '--context-lengths',
        type=str,
        default='10,20,50,100,200',
        help='Comma-separated context lengths (default: 10,20,50,100,200)'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=50,
        help='Number of samples per condition per length (default: 50)'
    )
    parser.add_argument(
        '--layer',
        type=int,
        default=-5,
        help='Layer to analyze, negative indexing from end (default: -5)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/block_permutation',
        help='Output directory (default: results/block_permutation)'
    )
    parser.add_argument(
        '--permutation-mode',
        type=str,
        choices=['block', 'full_shuffle'],
        default='block',
        help='Type of permutation: block (shuffle block order) or full_shuffle (randomize all tokens) (default: block)'
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Parse context lengths
    context_lengths = [int(x) for x in args.context_lengths.split(',')]

    # Run experiment
    results = run_experiment(
        model_name=args.model,
        context_lengths=context_lengths,
        n_samples=args.n_samples,
        layer_idx=args.layer,
        output_dir=args.output_dir,
        seed=args.seed,
        permutation_mode=args.permutation_mode
    )

    # Generate plots
    if PLOTTING_AVAILABLE:
        plot_results(results, output_dir=args.output_dir)
    else:
        print("\nPlotting skipped (libraries not available)")
        print("Results saved to JSON - you can plot separately")

    print("\nExperiment complete!")
