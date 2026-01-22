#!/usr/bin/env python3
"""
Inline block permutation experiment - runs directly without subprocess issues.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.data import HierarchicalGraph, HierarchicalGraphConfig
from src.models import HookedLLM
from src.metrics import ClusterSeparation


def identify_blocks(node_sequence, graph):
    """Identify maximal consecutive sequences from same cluster."""
    if not node_sequence:
        return []

    blocks = []
    current_cluster = graph.get_cluster(node_sequence[0])
    block_start = 0

    for i in range(1, len(node_sequence)):
        node_cluster = graph.get_cluster(node_sequence[i])
        if node_cluster != current_cluster:
            blocks.append((block_start, i, current_cluster))
            block_start = i
            current_cluster = node_cluster

    blocks.append((block_start, len(node_sequence), current_cluster))
    return blocks


def permute_blocks(node_sequence, blocks, seed=None):
    """Permute block order while keeping internal structure."""
    rng = np.random.default_rng(seed)
    block_sequences = []
    for start, end, cluster_id in blocks:
        block_sequences.append(node_sequence[start:end])

    permuted_indices = rng.permutation(len(block_sequences))
    permuted_sequence = []
    for idx in permuted_indices:
        permuted_sequence.extend(block_sequences[idx])

    return permuted_sequence


def sequence_to_prompt(node_sequence, graph):
    """Convert node sequence to prompt string."""
    tokens = [graph.node_to_token[node] for node in node_sequence]
    return " ".join(tokens)


def extract_and_measure(model, prompt, cluster_labels, layer_idx=-5):
    """Extract representations and compute metrics."""
    try:
        output, cache = model.forward_with_cache(prompt, layers=[layer_idx])
        representations = cache.get_residual_stream(layer_idx)

        if representations is None:
            return None, None

        representations = representations.squeeze(0)
        losses = model.compute_per_token_loss(prompt)

        metric = ClusterSeparation()
        separation = metric.compute(representations, torch.tensor(cluster_labels))
        perplexity = torch.exp(losses.mean()).item()

        return separation, perplexity
    except Exception as e:
        print(f"Error: {e}")
        return None, None


# Configuration
MODEL_NAME = "gpt2"
CONTEXT_LENGTHS = [10, 20, 50, 100]
N_SAMPLES = 30
LAYER_IDX = -5
SEED = 42

print("=" * 80)
print("BLOCK PERMUTATION EXPERIMENT - INLINE VERSION")
print("=" * 80)
print(f"Model: {MODEL_NAME}")
print(f"Context lengths: {CONTEXT_LENGTHS}")
print(f"Samples: {N_SAMPLES}")
print(f"Layer: {LAYER_IDX}")
print()

# Setup
output_dir = Path("results/block_permutation")
output_dir.mkdir(parents=True, exist_ok=True)

# Graph
graph_config = HierarchicalGraphConfig(
    num_superclusters=3,
    nodes_per_cluster=5,
    p_intra_cluster=0.8,
    p_inter_cluster=0.1,
    seed=SEED
)
graph = HierarchicalGraph(graph_config)
print(f"Graph: {graph.num_nodes} nodes, 3 clusters\n")

# Load model
print("Loading model...")
model = HookedLLM.from_pretrained(MODEL_NAME)
print(f"Model loaded\n")

# Results storage
results = {
    'natural': defaultdict(list),
    'permuted': defaultdict(list),
    'natural_perplexity': defaultdict(list),
    'permuted_perplexity': defaultdict(list),
    'context_lengths': CONTEXT_LENGTHS,
    'metadata': {
        'model': MODEL_NAME,
        'layer': LAYER_IDX,
        'n_samples': N_SAMPLES,
        'seed': SEED,
        'timestamp': datetime.now().isoformat()
    }
}

# Run experiment
for N in CONTEXT_LENGTHS:
    print(f"Context Length N={N}")
    print("-" * 40)

    natural_seps = []
    permuted_seps = []
    natural_ppls = []
    permuted_ppls = []

    for sample_idx in range(N_SAMPLES):
        if sample_idx % 10 == 0:
            print(f"  Sample {sample_idx}/{N_SAMPLES}...")

        # Generate natural walk
        prompt_natural, node_seq_natural = graph.generate_random_walk(
            length=N, return_nodes=True
        )

        # Get blocks and permute
        blocks = identify_blocks(node_seq_natural, graph)
        node_seq_permuted = permute_blocks(
            node_seq_natural, blocks, seed=SEED + sample_idx
        )
        prompt_permuted = sequence_to_prompt(node_seq_permuted, graph)

        # Cluster labels
        cluster_labels_nat = [graph.get_cluster(n) for n in node_seq_natural]
        cluster_labels_perm = [graph.get_cluster(n) for n in node_seq_permuted]

        # Natural
        sep_nat, ppl_nat = extract_and_measure(
            model, prompt_natural, cluster_labels_nat, LAYER_IDX
        )
        if sep_nat is not None:
            natural_seps.append(sep_nat)
            natural_ppls.append(ppl_nat)

        # Permuted
        sep_perm, ppl_perm = extract_and_measure(
            model, prompt_permuted, cluster_labels_perm, LAYER_IDX
        )
        if sep_perm is not None:
            permuted_seps.append(sep_perm)
            permuted_ppls.append(ppl_perm)

    # Store
    results['natural'][N] = natural_seps
    results['permuted'][N] = permuted_seps
    results['natural_perplexity'][N] = natural_ppls
    results['permuted_perplexity'][N] = permuted_ppls

    # Summary
    mean_nat = np.mean(natural_seps) if natural_seps else 0
    mean_perm = np.mean(permuted_seps) if permuted_seps else 0
    std_nat = np.std(natural_seps) if natural_seps else 0
    std_perm = np.std(permuted_seps) if permuted_seps else 0
    ratio = mean_nat / mean_perm if mean_perm > 0 else 0

    print(f"  Natural:  Φ = {mean_nat:.3f} ± {std_nat:.3f}")
    print(f"  Permuted: Φ = {mean_perm:.3f} ± {std_perm:.3f}")
    print(f"  Ratio:    {ratio:.2f}x")
    print()

# Save results
output_file = output_dir / f"block_permutation_{MODEL_NAME.replace('/', '_')}.json"

results_serializable = {
    'natural': {str(k): v for k, v in results['natural'].items()},
    'permuted': {str(k): v for k, v in results['permuted'].items()},
    'natural_perplexity': {str(k): v for k, v in results['natural_perplexity'].items()},
    'permuted_perplexity': {str(k): v for k, v in results['permuted_perplexity'].items()},
    'context_lengths': results['context_lengths'],
    'metadata': results['metadata']
}

with open(output_file, 'w') as f:
    json.dump(results_serializable, f, indent=2)

print(f"\nResults saved to {output_file}")
print("\nEXPERIMENT COMPLETE!")
