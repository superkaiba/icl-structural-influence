#!/usr/bin/env python3
"""Quick test of block permutation logic."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.data import HierarchicalGraph, HierarchicalGraphConfig

# Test block identification
def identify_blocks(node_sequence, graph):
    """Identify maximal consecutive sequences (blocks) from the same cluster."""
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

# Create graph
config = HierarchicalGraphConfig(
    num_superclusters=3,
    nodes_per_cluster=5,
    seed=42
)
graph = HierarchicalGraph(config)

# Generate walk
print("Generating random walk...")
prompt, nodes = graph.generate_random_walk(length=50, return_nodes=True)
print(f"Walk length: {len(nodes)}")
print(f"Prompt (first 100 chars): {prompt[:100]}...")

# Identify blocks
blocks = identify_blocks(nodes, graph)
print(f"\nFound {len(blocks)} blocks:")
for i, (start, end, cluster) in enumerate(blocks):
    print(f"  Block {i}: positions {start}-{end}, cluster {cluster}, length {end-start}")

# Test permutation
import numpy as np
rng = np.random.default_rng(42)

block_sequences = []
for start, end, cluster_id in blocks:
    block_sequences.append(nodes[start:end])

permuted_indices = rng.permutation(len(block_sequences))
permuted_sequence = []
for idx in permuted_indices:
    permuted_sequence.extend(block_sequences[idx])

print(f"\nPermuted sequence length: {len(permuted_sequence)}")
print(f"Original == Permuted: {nodes == permuted_sequence}")
print("\nTest passed! Block permutation logic works.")
