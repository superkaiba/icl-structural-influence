#!/usr/bin/env python3
"""
Investigate why cosine similarities are all so high (~0.9-0.97).
Is this the curse of dimensionality?
"""

import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))
from models import HookedLLM
from run_hierarchy_and_semantic_experiments import SemanticConflictGraph, HierarchicalGraph3Level


def analyze_representation_geometry(model, tokenizer, graph, context_lengths, n_samples=50, layer_idx=-5):
    """Analyze the geometry of token representations."""

    print("=" * 70)
    print("REPRESENTATION GEOMETRY ANALYSIS")
    print("=" * 70)

    # Collect representations
    token_reps = {}
    for ctx_len in context_lengths:
        print(f"  Collecting N={ctx_len}...", end=" ", flush=True)
        token_representations = defaultdict(list)

        for _ in range(n_samples):
            prompt, node_sequence = graph.generate_random_walk(length=ctx_len, return_nodes=True)
            tokens = tokenizer.encode(prompt, add_special_tokens=False)
            input_ids = torch.tensor([tokens]).to(model.device)

            with torch.no_grad():
                outputs = model(input_ids, output_hidden_states=True)
                hidden_states = outputs.hidden_states[layer_idx][0]

            token_texts = prompt.split()
            for pos, (node, token_text) in enumerate(zip(node_sequence, token_texts)):
                if pos < hidden_states.shape[0]:
                    rep = hidden_states[pos].cpu().float().numpy()
                    token_representations[token_text].append(rep)

        for token_text, reps in token_representations.items():
            if reps:
                token_reps[(ctx_len, token_text)] = np.mean(reps, axis=0)

        print(f"done")

    # Analyze geometry at each context length
    print("\n" + "=" * 70)
    print("GEOMETRY STATISTICS")
    print("=" * 70)

    for ctx_len in context_lengths:
        tokens = [t for (c, t) in token_reps.keys() if c == ctx_len]
        if len(tokens) < 2:
            continue

        X = np.array([token_reps[(ctx_len, t)] for t in tokens])

        # Compute all pairwise cosine similarities
        cos_sim = cosine_similarity(X)
        # Get upper triangle (excluding diagonal)
        upper_tri = cos_sim[np.triu_indices_from(cos_sim, k=1)]

        # Compute all pairwise L2 distances
        l2_dist = euclidean_distances(X)
        upper_tri_l2 = l2_dist[np.triu_indices_from(l2_dist, k=1)]

        # Compute norms
        norms = np.linalg.norm(X, axis=1)

        # Compute mean and variance of representations
        mean_rep = np.mean(X, axis=0)
        mean_norm = np.linalg.norm(mean_rep)

        # Centered representations
        X_centered = X - mean_rep
        centered_norms = np.linalg.norm(X_centered, axis=1)

        # Cosine similarity of centered representations
        cos_sim_centered = cosine_similarity(X_centered)
        upper_tri_centered = cos_sim_centered[np.triu_indices_from(cos_sim_centered, k=1)]

        print(f"\nN={ctx_len}:")
        print(f"  Dimensionality: {X.shape[1]}")
        print(f"  Num tokens: {len(tokens)}")
        print(f"  ")
        print(f"  RAW COSINE SIMILARITY:")
        print(f"    Mean: {np.mean(upper_tri):.4f}")
        print(f"    Std:  {np.std(upper_tri):.4f}")
        print(f"    Min:  {np.min(upper_tri):.4f}")
        print(f"    Max:  {np.max(upper_tri):.4f}")
        print(f"    Range: {np.max(upper_tri) - np.min(upper_tri):.4f}")
        print(f"  ")
        print(f"  L2 DISTANCE:")
        print(f"    Mean: {np.mean(upper_tri_l2):.2f}")
        print(f"    Std:  {np.std(upper_tri_l2):.2f}")
        print(f"    Min:  {np.min(upper_tri_l2):.2f}")
        print(f"    Max:  {np.max(upper_tri_l2):.2f}")
        print(f"    Range: {np.max(upper_tri_l2) - np.min(upper_tri_l2):.2f}")
        print(f"  ")
        print(f"  REPRESENTATION NORMS:")
        print(f"    Mean norm: {np.mean(norms):.2f}")
        print(f"    Std norm:  {np.std(norms):.2f}")
        print(f"    Mean of mean rep: {mean_norm:.2f}")
        print(f"  ")
        print(f"  CENTERED COSINE SIMILARITY:")
        print(f"    Mean: {np.mean(upper_tri_centered):.4f}")
        print(f"    Std:  {np.std(upper_tri_centered):.4f}")
        print(f"    Min:  {np.min(upper_tri_centered):.4f}")
        print(f"    Max:  {np.max(upper_tri_centered):.4f}")
        print(f"    Range: {np.max(upper_tri_centered) - np.min(upper_tri_centered):.4f}")

    # Compare what fraction of variance is "shared" vs "distinctive"
    print("\n" + "=" * 70)
    print("VARIANCE DECOMPOSITION")
    print("=" * 70)

    for ctx_len in context_lengths:
        tokens = [t for (c, t) in token_reps.keys() if c == ctx_len]
        if len(tokens) < 2:
            continue

        X = np.array([token_reps[(ctx_len, t)] for t in tokens])

        # Total variance
        total_var = np.var(X)

        # Between-token variance (variance of means)
        mean_rep = np.mean(X, axis=0)
        between_var = np.mean((X - mean_rep) ** 2)

        # Variance explained by mean direction
        mean_direction = mean_rep / np.linalg.norm(mean_rep)
        projections = X @ mean_direction
        var_along_mean = np.var(projections)

        print(f"\nN={ctx_len}:")
        print(f"  Total variance: {total_var:.4f}")
        print(f"  Variance along mean direction: {var_along_mean:.4f}")
        print(f"  Ratio (var_along_mean / total): {var_along_mean / total_var:.4f}")

    return token_reps


def compare_random_baseline(dim=4096, n_vectors=16):
    """What's the expected cosine similarity for random vectors?"""
    print("\n" + "=" * 70)
    print("RANDOM BASELINE COMPARISON")
    print("=" * 70)

    # Random unit vectors
    random_vecs = np.random.randn(n_vectors, dim)
    random_vecs = random_vecs / np.linalg.norm(random_vecs, axis=1, keepdims=True)

    cos_sim = cosine_similarity(random_vecs)
    upper_tri = cos_sim[np.triu_indices_from(cos_sim, k=1)]

    print(f"\nRandom unit vectors in {dim}D:")
    print(f"  Mean cosine similarity: {np.mean(upper_tri):.4f}")
    print(f"  Std: {np.std(upper_tri):.4f}")
    print(f"  Expected (theory): ~0 for orthogonal")

    # Random vectors with shared component (like a "context" direction)
    shared_component = np.random.randn(dim)
    shared_component = shared_component / np.linalg.norm(shared_component)

    for shared_weight in [0.5, 0.9, 0.95, 0.99]:
        vecs_with_shared = shared_weight * shared_component + (1 - shared_weight) * np.random.randn(n_vectors, dim)
        vecs_with_shared = vecs_with_shared / np.linalg.norm(vecs_with_shared, axis=1, keepdims=True)

        cos_sim = cosine_similarity(vecs_with_shared)
        upper_tri = cos_sim[np.triu_indices_from(cos_sim, k=1)]

        print(f"\nWith {shared_weight*100:.0f}% shared component:")
        print(f"  Mean cosine similarity: {np.mean(upper_tri):.4f}")
        print(f"  Std: {np.std(upper_tri):.4f}")


def main():
    print("=" * 70)
    print("INVESTIGATING HIGH COSINE SIMILARITIES")
    print("Is this the curse of dimensionality?")
    print("=" * 70)

    # First, check random baseline
    compare_random_baseline()

    # Load model
    print("\n\nLoading model...")
    hooked_model = HookedLLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = hooked_model.model
    tokenizer = hooked_model.tokenizer
    model.eval()

    # Test with semantic conflict graph
    print("\n" + "=" * 70)
    print("SEMANTIC CONFLICT GRAPH")
    print("=" * 70)
    graph = SemanticConflictGraph(seed=42)
    token_reps = analyze_representation_geometry(
        model, tokenizer, graph,
        context_lengths=[5, 20, 50, 100],
        n_samples=50, layer_idx=-5
    )

    # Also test with hierarchy graph for comparison
    print("\n" + "=" * 70)
    print("HIERARCHY GRAPH")
    print("=" * 70)
    graph_hier = HierarchicalGraph3Level(seed=42)
    token_reps_hier = analyze_representation_geometry(
        model, tokenizer, graph_hier,
        context_lengths=[5, 20, 50, 100],
        n_samples=50, layer_idx=-5
    )

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
The high cosine similarities (~0.9+) are likely because:

1. LLM representations have a strong "shared component" - a common direction
   that most token representations point towards (the "mean direction")

2. The actual token-specific information is in the RESIDUAL after removing
   this shared component

3. For better discrimination, use:
   - CENTERED cosine similarity (subtract mean before computing)
   - L2 distance (not affected by shared direction)
   - Or normalize by baseline similarity
    """)


if __name__ == "__main__":
    main()
