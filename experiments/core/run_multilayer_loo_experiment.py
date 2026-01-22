#!/usr/bin/env python3
"""
Multi-Layer LOO Influence Experiment.

Runs Leave-One-Out influence analysis across ALL 32 layers and multiple context lengths
for both semantic and unrelated token conditions.

Uses HookedLLM.forward_with_cache() to efficiently extract all layers in one forward pass.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.models.hooked_model import HookedLLM
from run_hierarchy_and_semantic_experiments import SemanticConflictGraph


# =============================================================================
# Token Extraction from Cache
# =============================================================================

def extract_token_reps_from_cache(cache, layer_idx, context_tokens, tokenizer):
    """Extract token representations from cached residuals for a specific layer."""
    residual = cache.get_residual_stream(layer_idx)

    if residual is None:
        return {}

    # Remove batch dimension
    hidden = residual.squeeze(0).cpu()

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


def get_all_layer_representations(model, context_tokens, layers_to_extract):
    """Extract representations from specified layers in one forward pass."""
    prompt = " ".join(context_tokens)

    logits, cache = model.forward_with_cache(prompt, layers=layers_to_extract)

    # Extract per-layer token representations
    layer_reps = {}
    for layer_idx in layers_to_extract:
        layer_reps[layer_idx] = extract_token_reps_from_cache(
            cache, layer_idx, context_tokens, model.tokenizer
        )

    # Compute per-token loss from logits
    if logits is not None:
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        inputs = model.tokenize(prompt)
        input_ids = inputs['input_ids'][0]

        if logits.shape[0] > 1:
            shift_logits = logits[0, :-1].float()
            shift_labels = input_ids[1:].to(shift_logits.device)
            token_losses = loss_fn(shift_logits, shift_labels).cpu().numpy()
        else:
            token_losses = np.array([])
    else:
        token_losses = np.array([])

    # Clear cache to free memory
    del cache
    del logits
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return layer_reps, token_losses


# =============================================================================
# Metric Computation (reuse from compare_loo_metrics.py)
# =============================================================================

def compute_ratio_metric(token_reps, graph):
    """Compute original ratio metric."""
    available_tokens = set(token_reps.keys())

    # Cross-cluster pairs (semantic pairs in different clusters)
    cross_pairs = [(t1, t2) for t1, t2, _ in graph.get_semantic_pairs()
                   if t1 in available_tokens and t2 in available_tokens]

    # Within-cluster pairs
    within_pairs = []
    for cid, tokens in graph.graph_clusters.items():
        for i, t1 in enumerate(tokens):
            for t2 in tokens[i+1:]:
                if t1 in available_tokens and t2 in available_tokens:
                    within_pairs.append((t1, t2))

    def mean_dist(pairs):
        if not pairs:
            return np.nan
        dists = []
        for t1, t2 in pairs:
            dists.append(torch.norm(token_reps[t1] - token_reps[t2]).item())
        return np.mean(dists) if dists else np.nan

    cross_dist = mean_dist(cross_pairs)
    within_dist = mean_dist(within_pairs)

    if np.isnan(cross_dist) or np.isnan(within_dist):
        ratio = np.nan
    else:
        # Use relative epsilon to bound ratio (max ~1000)
        eps = 1e-3
        min_within = max(eps * cross_dist, 1e-10)
        within_dist_bounded = max(within_dist, min_within)
        ratio = cross_dist / within_dist_bounded

    return {
        'ratio': ratio,
        'cross_dist': cross_dist,
        'within_dist': within_dist
    }


def compute_dirichlet_energy(token_reps, graph):
    """Compute Dirichlet Energy: E(X) = sum A_{ij} ||x_i - x_j||^2."""
    available_tokens = set(token_reps.keys())

    energy = 0.0
    edge_count = 0

    # Within-cluster pairs (adjacent in graph)
    for cid, tokens in graph.graph_clusters.items():
        for i, t1 in enumerate(tokens):
            for t2 in tokens[i+1:]:
                if t1 in available_tokens and t2 in available_tokens:
                    diff = token_reps[t1] - token_reps[t2]
                    energy += torch.sum(diff ** 2).item()
                    edge_count += 1

    if edge_count > 0:
        return energy / edge_count
    else:
        return np.nan


def compute_css_components(token_reps, token_losses, context_tokens, graph):
    """Compute CSS components: loss and phi (between-cluster variance)."""
    cluster_reps = defaultdict(list)
    for token in context_tokens:
        if token in token_reps:
            cid = graph.token_to_graph_cluster.get(token)
            if cid is not None:
                cluster_reps[cid].append(token_reps[token])

    cluster_means = {}
    for cid, reps in cluster_reps.items():
        if reps:
            cluster_means[cid] = torch.stack(reps).mean(dim=0)

    # Phi = between-cluster variance
    if len(cluster_means) >= 2:
        global_mean = torch.stack(list(cluster_means.values())).mean(dim=0)
        phi = sum(torch.sum((m - global_mean)**2).item() for m in cluster_means.values())
        phi /= len(cluster_means)
    else:
        phi = 0.0

    mean_loss = np.mean(token_losses) if len(token_losses) > 0 else 0.0

    return mean_loss, phi


# =============================================================================
# LOO Influence Computation
# =============================================================================

def compute_multilayer_loo_influences(model, context_tokens, graph, layers_to_test, pos):
    """Compute LOO influence for specified metrics across specified layers."""

    # Full context - one forward pass extracts specified layers
    full_layer_reps, full_losses = get_all_layer_representations(
        model, context_tokens, layers_to_test
    )

    # LOO context - remove position
    loo_tokens = context_tokens[:pos] + context_tokens[pos+1:]
    if len(loo_tokens) < 2:
        return None

    loo_layer_reps, loo_losses = get_all_layer_representations(
        model, loo_tokens, layers_to_test
    )

    # Compute metrics for each layer
    results = {}
    for layer_idx in layers_to_test:
        full_reps = full_layer_reps.get(layer_idx, {})
        loo_reps = loo_layer_reps.get(layer_idx, {})

        if not full_reps or not loo_reps:
            results[layer_idx] = {
                'ratio_influence': np.nan,
                'energy_influence': np.nan,
                'cross_dist_influence': np.nan,
                'within_dist_influence': np.nan,
            }
            continue

        full_ratio = compute_ratio_metric(full_reps, graph)
        loo_ratio = compute_ratio_metric(loo_reps, graph)

        full_energy = compute_dirichlet_energy(full_reps, graph)
        loo_energy = compute_dirichlet_energy(loo_reps, graph)

        results[layer_idx] = {
            'ratio_influence': full_ratio['ratio'] - loo_ratio['ratio'],
            'energy_influence': loo_energy - full_energy,  # Positive = removing hurts
            'cross_dist_influence': full_ratio['cross_dist'] - loo_ratio['cross_dist'],
            'within_dist_influence': full_ratio['within_dist'] - loo_ratio['within_dist'],
            # CSS components stored for later aggregation
            'full_loss': np.mean(full_losses) if len(full_losses) > 0 else np.nan,
            'full_phi': compute_css_components(full_reps, full_losses, context_tokens, graph)[1],
        }

    return results


def identify_token_types(context_tokens, graph):
    """Identify bridge and anchor positions."""
    bridge_positions = []
    anchor_positions = []

    prev_cluster = None
    for i, token in enumerate(context_tokens):
        cluster = graph.token_to_graph_cluster.get(token)
        if prev_cluster is not None and cluster != prev_cluster:
            bridge_positions.append(i)
        else:
            anchor_positions.append(i)
        prev_cluster = cluster

    return bridge_positions, anchor_positions


# =============================================================================
# Main Experiment
# =============================================================================

def run_multilayer_experiment(
    context_lengths=[6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100],
    n_trials=5,
    layers_to_test=None,
    use_semantic_tokens=True,
    output_dir="results/loo_multilayer",
    model_name="Qwen/Qwen2.5-7B"
):
    """Run multi-layer LOO experiment."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    condition = "semantic" if use_semantic_tokens else "unrelated"
    print(f"\n{'='*70}")
    print(f"Running multi-layer LOO experiment: {condition.upper()} tokens")
    print(f"{'='*70}")

    # Load model
    print(f"Loading model: {model_name}...")
    model = HookedLLM.from_pretrained(
        model_name,
        dtype=torch.float16
    )

    # Determine layers to test based on model
    if layers_to_test is None:
        layers_to_test = list(range(model.num_layers))
    print(f"Model loaded: {model.num_layers} layers total")
    print(f"Testing layers: {layers_to_test}")

    # Initialize results storage
    # Structure: results[layer][N] = {'bridge': {...}, 'anchor': {...}}
    results = {
        'condition': condition,
        'context_lengths': context_lengths,
        'layers_tested': layers_to_test,
        'n_trials': n_trials,
        'by_layer_N': {}
    }

    # Initialize nested structure - only for layers we're testing
    for layer in layers_to_test:
        results['by_layer_N'][layer] = {}
        for N in context_lengths:
            results['by_layer_N'][layer][N] = {
                'bridge': defaultdict(list),
                'anchor': defaultdict(list),
                'css_bridge_data': {'losses': [], 'phis': []},
                'css_anchor_data': {'losses': [], 'phis': []},
            }

    # Check for existing checkpoints and skip completed context lengths
    completed_Ns = set()
    for N in context_lengths:
        checkpoint_file = output_path / f"checkpoint_{condition}_N{N}.json"
        if checkpoint_file.exists():
            completed_Ns.add(N)
            print(f"Checkpoint exists for N={N}, will skip")

    remaining_Ns = [N for N in context_lengths if N not in completed_Ns]
    print(f"\nCompleted: {sorted(completed_Ns)}")
    print(f"Remaining: {remaining_Ns}")

    # Run experiment for remaining context lengths
    for N in remaining_Ns:
        print(f"\n--- Context length N={N} ---")

        for trial in tqdm(range(n_trials), desc=f"N={N}"):
            # Generate random walk
            graph = SemanticConflictGraph(
                seed=42 + trial * 100 + N,
                use_semantic_tokens=use_semantic_tokens
            )
            context, _ = graph.generate_random_walk(length=N, return_nodes=True)
            context_tokens = context.split()

            # Identify token types
            bridge_pos, anchor_pos = identify_token_types(context_tokens, graph)

            # Sample positions to test
            if N > 20:
                test_positions = list(range(0, N, 3))[:10]
            else:
                test_positions = list(range(N))

            for pos in test_positions:
                try:
                    layer_influences = compute_multilayer_loo_influences(
                        model, context_tokens, graph, layers_to_test, pos
                    )
                except Exception as e:
                    warnings.warn(f"Error at N={N}, pos={pos}: {e}")
                    continue

                if layer_influences is None:
                    continue

                # Store results by layer - only for layers we're testing
                for layer in layers_to_test:
                    layer_data = layer_influences.get(layer, {})

                    if pos in bridge_pos:
                        token_type = 'bridge'
                        css_key = 'css_bridge_data'
                    else:
                        token_type = 'anchor'
                        css_key = 'css_anchor_data'

                    storage = results['by_layer_N'][layer][N][token_type]

                    for metric in ['ratio_influence', 'energy_influence',
                                   'cross_dist_influence', 'within_dist_influence']:
                        if metric in layer_data:
                            storage[metric].append(layer_data[metric])

                    # Store CSS components
                    if 'full_loss' in layer_data and 'full_phi' in layer_data:
                        css_storage = results['by_layer_N'][layer][N][css_key]
                        css_storage['losses'].append(layer_data['full_loss'])
                        css_storage['phis'].append(layer_data['full_phi'])

            # Clear memory after each trial
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Save intermediate checkpoint after each context length
        checkpoint_file = output_path / f"checkpoint_{condition}_N{N}.json"
        print(f"Saving checkpoint: {checkpoint_file}")

        # Aggregate results for this N across all layers
        checkpoint_data = {
            'condition': condition,
            'context_length': N,
            'layers_tested': layers_to_test,
            'n_trials': n_trials,
            'by_layer': {}
        }

        for layer in layers_to_test:
            data = results['by_layer_N'][layer][N]
            agg = {'bridge': {}, 'anchor': {}}

            for token_type in ['bridge', 'anchor']:
                for metric in ['ratio_influence', 'energy_influence',
                               'cross_dist_influence', 'within_dist_influence']:
                    vals = data[token_type][metric]
                    if vals:
                        agg[token_type][metric] = {
                            'mean': float(np.nanmean(vals)),
                            'std': float(np.nanstd(vals)),
                            'n': len(vals)
                        }

            checkpoint_data['by_layer'][layer] = agg

        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

        # Clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Aggregate results from all checkpoint files
    print("\nAggregating results from checkpoint files...")
    aggregated = {
        'condition': condition,
        'context_lengths': context_lengths,
        'layers_tested': layers_to_test,
        'n_trials': n_trials,
        'by_layer_N': {}
    }

    for layer in layers_to_test:
        aggregated['by_layer_N'][layer] = {}

    # Load from checkpoint files
    for N in context_lengths:
        checkpoint_file = output_path / f"checkpoint_{condition}_N{N}.json"
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)

            # Extract data for each layer from checkpoint
            for layer_str, layer_data in checkpoint_data.get('by_layer', {}).items():
                layer = int(layer_str)
                if layer in layers_to_test:
                    aggregated['by_layer_N'][layer][N] = layer_data
        else:
            # Use data from this run (for any N that was just computed)
            for layer in layers_to_test:
                data = results['by_layer_N'][layer][N]

                agg = {'bridge': {}, 'anchor': {}}

                for token_type in ['bridge', 'anchor']:
                    for metric in ['ratio_influence', 'energy_influence',
                                   'cross_dist_influence', 'within_dist_influence']:
                        vals = data[token_type][metric]
                        if vals:
                            agg[token_type][metric] = {
                                'mean': float(np.nanmean(vals)),
                                'std': float(np.nanstd(vals)),
                                'n': len(vals)
                            }

                # Compute CSS for each token type
                for token_type, css_key in [('bridge', 'css_bridge_data'),
                                            ('anchor', 'css_anchor_data')]:
                    css_data = data[css_key]
                    if len(css_data['losses']) > 1:
                        cov_matrix = np.cov(css_data['losses'], css_data['phis'])
                        if cov_matrix.shape == (2, 2):
                            agg[f'css_{token_type}'] = float(-cov_matrix[0, 1])
                        else:
                            agg[f'css_{token_type}'] = np.nan
                    else:
                        agg[f'css_{token_type}'] = np.nan

                aggregated['by_layer_N'][layer][N] = agg

    # Save results
    results_file = output_path / f"results_{condition}.json"

    # Convert to serializable
    def to_serializable(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj) if not np.isnan(obj) else None
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        return obj

    with open(results_file, 'w') as f:
        json.dump(to_serializable(aggregated), f, indent=2)

    print(f"\nSaved results to {results_file}")

    return aggregated


def main():
    """Run experiments for both conditions."""
    parser = argparse.ArgumentParser(description="Multi-Layer LOO Influence Experiment")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B",
                        help="Model to use (default: Qwen/Qwen2.5-7B)")
    parser.add_argument("--context-lengths", type=str, default="6,10,20,50,100",
                        help="Comma-separated context lengths (default: 6,10,20,50,100)")
    parser.add_argument("--n-trials", type=int, default=5,
                        help="Number of trials per condition (default: 5)")
    parser.add_argument("--output-dir", type=str, default="results/loo_multilayer",
                        help="Output directory (default: results/loo_multilayer)")
    parser.add_argument("--semantic-only", action="store_true",
                        help="Run only semantic tokens condition")
    parser.add_argument("--unrelated-only", action="store_true",
                        help="Run only unrelated tokens condition")
    args = parser.parse_args()

    context_lengths = [int(x) for x in args.context_lengths.split(",")]
    output_dir = args.output_dir

    run_semantic = not args.unrelated_only
    run_unrelated = not args.semantic_only

    # Run semantic condition
    if run_semantic:
        print("\n" + "="*80)
        print("SEMANTIC TOKENS")
        print("="*80)
        semantic_results = run_multilayer_experiment(
            context_lengths=context_lengths,
            n_trials=args.n_trials,
            use_semantic_tokens=True,
            output_dir=output_dir,
            model_name=args.model
        )

    # Run unrelated condition
    if run_unrelated:
        print("\n" + "="*80)
        print("UNRELATED TOKENS")
        print("="*80)
        unrelated_results = run_multilayer_experiment(
            context_lengths=context_lengths,
            n_trials=args.n_trials,
            use_semantic_tokens=False,
            output_dir=output_dir,
            model_name=args.model
        )

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
