#!/usr/bin/env python3
"""
Compare different influence metrics on the same data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cosine
from collections import defaultdict

from run_hierarchy_and_semantic_experiments import SemanticConflictGraph


def load_model(model_name="meta-llama/Llama-3.1-8B"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    return model, tokenizer


def get_all_token_representations(model, tokenizer, context_tokens, layer_idx):
    """Get representations for all tokens at specified layer."""
    prompt = " ".join(context_tokens)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    representations = {}

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            representations['hidden'] = output[0].detach()
        else:
            representations['hidden'] = output.detach()

    if layer_idx == 0:
        handle = model.model.embed_tokens.register_forward_hook(hook_fn)
    else:
        handle = model.model.layers[layer_idx - 1].register_forward_hook(hook_fn)

    with torch.no_grad():
        model(**inputs)

    handle.remove()

    hidden = representations['hidden'][0]

    # Map tokens to representations
    token_reps = {}
    current_pos = 0

    for token in context_tokens:
        word_tokens = tokenizer.encode(" " + token, add_special_tokens=False)
        n_subtokens = len(word_tokens)

        if current_pos + n_subtokens <= hidden.shape[0]:
            rep = hidden[current_pos:current_pos + n_subtokens].mean(dim=0)
            if token not in token_reps:
                token_reps[token] = []
            token_reps[token].append(rep)
            current_pos += n_subtokens

    # Average repeated tokens
    for token in token_reps:
        token_reps[token] = torch.stack(token_reps[token]).mean(dim=0)

    return token_reps


def compute_metrics(token_reps, graph):
    """Compute multiple metrics from token representations."""

    metrics = {}

    # Get pairs
    semantic_pairs = [(t1, t2) for t1, t2, _ in graph.get_semantic_pairs()]

    within_cluster_pairs = []
    cross_cluster_pairs = []
    for cid, tokens in graph.graph_clusters.items():
        for i, t1 in enumerate(tokens):
            for t2 in tokens[i+1:]:
                within_cluster_pairs.append((t1, t2))

    for t1, t2, _ in graph.get_semantic_pairs():
        cross_cluster_pairs.append((t1, t2))

    # 1. Current ratio metric
    def mean_dist(pairs):
        dists = []
        for t1, t2 in pairs:
            if t1 in token_reps and t2 in token_reps:
                dists.append(torch.norm(token_reps[t1] - token_reps[t2]).item())
        return np.mean(dists) if dists else np.nan

    cross_dist = mean_dist(cross_cluster_pairs)
    within_dist = mean_dist(within_cluster_pairs)
    metrics['ratio'] = cross_dist / within_dist if within_dist > 0 else np.nan

    # 2. Separate distances
    metrics['cross_cluster_dist'] = cross_dist
    metrics['within_cluster_dist'] = within_dist

    # 3. Specific semantic pair distances (e.g., cat-dog)
    for t1, t2, group in graph.get_semantic_pairs()[:3]:
        if t1 in token_reps and t2 in token_reps:
            metrics[f'{t1}_{t2}_L2'] = torch.norm(token_reps[t1] - token_reps[t2]).item()
            # Cosine distance
            v1 = token_reps[t1].cpu().numpy()
            v2 = token_reps[t2].cpu().numpy()
            metrics[f'{t1}_{t2}_cosine'] = cosine(v1, v2)

    # 4. Silhouette score
    tokens_with_reps = [t for t in graph.vocabulary if t in token_reps]
    if len(tokens_with_reps) >= 4:
        X = np.array([token_reps[t].cpu().numpy() for t in tokens_with_reps])
        labels = [graph.token_to_graph_cluster[t] for t in tokens_with_reps]
        if len(set(labels)) > 1:
            metrics['silhouette'] = silhouette_score(X, labels)
        else:
            metrics['silhouette'] = np.nan
    else:
        metrics['silhouette'] = np.nan

    return metrics


def compute_loo_influence_multi(model, tokenizer, context_tokens, graph, layer_idx, pos):
    """Compute multiple influence metrics for removing position pos."""

    # Full context metrics
    full_reps = get_all_token_representations(model, tokenizer, context_tokens, layer_idx)
    full_metrics = compute_metrics(full_reps, graph)

    # LOO context metrics
    loo_tokens = context_tokens[:pos] + context_tokens[pos+1:]
    if len(loo_tokens) < 2:
        return {k: 0.0 for k in full_metrics}

    loo_reps = get_all_token_representations(model, tokenizer, loo_tokens, layer_idx)
    loo_metrics = compute_metrics(loo_reps, graph)

    # Compute influences (full - loo for most, but reversed for some)
    influences = {}
    for key in full_metrics:
        if key in ['silhouette', 'ratio']:
            # Higher is better, so influence = full - loo
            influences[f'inf_{key}'] = full_metrics[key] - loo_metrics[key]
        elif 'dist' in key or 'L2' in key:
            # For distances, positive influence means removing decreases distance
            influences[f'inf_{key}'] = full_metrics[key] - loo_metrics[key]
        elif 'cosine' in key:
            # Cosine distance: higher = more different
            influences[f'inf_{key}'] = full_metrics[key] - loo_metrics[key]

    return influences


def main():
    model, tokenizer = load_model()
    graph = SemanticConflictGraph(seed=42, use_semantic_tokens=True)

    # Test on a few context lengths
    layer_idx = 16

    print("\n" + "="*70)
    print("COMPARING INFLUENCE METRICS")
    print("="*70)

    results = defaultdict(list)

    for N in [6, 8, 10]:
        print(f"\n--- Context length N={N} ---")

        # Generate context
        context, _ = graph.generate_random_walk(length=N, return_nodes=True)
        context_tokens = context.split()
        print(f"Context: {context}")

        # Identify bridge/anchor
        bridge_pos = []
        anchor_pos = []
        prev_cluster = None
        for i, tok in enumerate(context_tokens):
            cluster = graph.token_to_graph_cluster[tok]
            if prev_cluster is not None and cluster != prev_cluster:
                bridge_pos.append(i)
            else:
                anchor_pos.append(i)
            prev_cluster = cluster

        print(f"Bridge positions: {bridge_pos}")
        print(f"Anchor positions: {anchor_pos}")

        # Compute influences for a sample position
        if bridge_pos:
            pos = bridge_pos[0]
            pos_type = "BRIDGE"
        else:
            pos = anchor_pos[0] if anchor_pos else 0
            pos_type = "ANCHOR"

        print(f"\nInfluence of position {pos} ({context_tokens[pos]}, {pos_type}):")

        influences = compute_loo_influence_multi(
            model, tokenizer, context_tokens, graph, layer_idx, pos
        )

        for key, val in sorted(influences.items()):
            print(f"  {key:30s}: {val:+.4f}")
            results[key].append(val)

    print("\n" + "="*70)
    print("METRIC CORRELATIONS")
    print("="*70)

    # Simple correlation check
    keys = list(results.keys())
    if len(keys) >= 2 and len(results[keys[0]]) >= 3:
        print("\nCorrelation between inf_ratio and other metrics:")
        ratio_vals = np.array(results.get('inf_ratio', []))
        for key in keys:
            if key != 'inf_ratio' and len(results[key]) == len(ratio_vals):
                vals = np.array(results[key])
                if not (np.isnan(ratio_vals).any() or np.isnan(vals).any()):
                    corr = np.corrcoef(ratio_vals, vals)[0, 1]
                    print(f"  {key:30s}: r={corr:+.3f}")


if __name__ == "__main__":
    main()
