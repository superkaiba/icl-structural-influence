#!/usr/bin/env python3
"""
Incongruous Token Injection Experiment.

Investigates how a single random/incongruous token affects representations
at different positions (early vs later) in the context.

Research Questions:
- Does early noise get "washed out" by later context?
- Does late noise disrupt already-learned structure?
- Which layers are most sensitive to positional injection?

Incongruous Token Types:
1. wrong_cluster: Token from a different graph cluster
2. random_vocab: Random common word (in-distribution noise)
3. semantic_outlier: Clearly unrelated concept (e.g., "quantum")
4. out_of_vocab: Pseudo-OOV string (e.g., "ZZINCONGRUOUSZZ")

Usage:
    # Quick test
    python run_incongruous_injection_experiment.py \
        --model gpt2 \
        --context-lengths 20,50 \
        --n-samples 10

    # Full experiment
    python run_incongruous_injection_experiment.py \
        --model Qwen/Qwen2.5-7B \
        --context-lengths 20,50,100,200 \
        --n-samples 50
"""

import argparse
import json
import os
import random
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Optional

import torch
import numpy as np
from tqdm import tqdm

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent))
from src.data.hierarchical_graph import HierarchicalGraph, HierarchicalGraphConfig
from src.models.hooked_model import HookedLLM
from src.metrics.structural_influence import ClusterSeparation


# =============================================================================
# Incongruous Token Generators
# =============================================================================

# Common words for random vocab injection (semantically diverse)
COMMON_WORDS = [
    "chair", "window", "paper", "music", "coffee", "bottle", "pencil", "kitchen",
    "shadow", "garden", "silver", "orange", "camera", "library", "bicycle", "thunder",
    "puzzle", "blanket", "mirror", "diamond", "whisper", "basket", "dragon", "velvet",
]

# Semantic outliers (scientific/technical terms out of place in a graph walk)
SEMANTIC_OUTLIERS = [
    "quantum", "electron", "photon", "galaxy", "algorithm", "entropy", "molecule",
    "neuron", "chromosome", "isotope", "paradox", "hypothesis", "theorem", "axiom",
]

# Out-of-vocabulary pseudo-token
OOV_TOKEN = "ZZINCONGRUOUSZZ"


def get_wrong_cluster_token(graph: HierarchicalGraph, current_cluster: int) -> str:
    """Get a token from a cluster different from the current one."""
    all_clusters = list(range(graph.config.num_superclusters))
    other_clusters = [c for c in all_clusters if c != current_cluster]

    if not other_clusters:
        # Only one cluster - fall back to random vocab
        return random.choice(COMMON_WORDS)

    target_cluster = random.choice(other_clusters)

    # Find tokens in target cluster
    tokens_in_cluster = [
        graph.node_to_token[n]
        for n in range(graph.num_nodes)
        if graph.get_cluster(n) == target_cluster
    ]

    return random.choice(tokens_in_cluster)


def get_incongruous_token(
    token_type: str,
    graph: HierarchicalGraph,
    current_cluster: int
) -> str:
    """Get an incongruous token of the specified type."""
    if token_type == "wrong_cluster":
        return get_wrong_cluster_token(graph, current_cluster)
    elif token_type == "random_vocab":
        return random.choice(COMMON_WORDS)
    elif token_type == "semantic_outlier":
        return random.choice(SEMANTIC_OUTLIERS)
    elif token_type == "out_of_vocab":
        return OOV_TOKEN
    else:
        raise ValueError(f"Unknown token type: {token_type}")


# =============================================================================
# Token Injection Utilities
# =============================================================================

def inject_token(context_tokens: list[str], token: str, position: int) -> list[str]:
    """
    Insert token at position, shifting subsequent tokens right.

    Args:
        context_tokens: Original list of tokens
        token: Token to inject
        position: Position to insert at (0-indexed)

    Returns:
        New list with token inserted
    """
    return context_tokens[:position] + [token] + context_tokens[position:]


def compute_cluster_separation(
    model: HookedLLM,
    prompt: str,
    cluster_labels: list[int],
    layers: list[int],
    exclude_position: Optional[int] = None,
) -> dict[int, float]:
    """
    Compute cluster separation metric at each specified layer.

    Args:
        model: HookedLLM instance
        prompt: Space-separated token string
        cluster_labels: Cluster assignment for each token
        layers: Layer indices to analyze
        exclude_position: Position to exclude from metric calculation (for injected token)

    Returns:
        Dict mapping layer -> cluster separation value
    """
    metric = ClusterSeparation()

    _, cache = model.forward_with_cache(prompt, layers=layers)

    results = {}
    for layer in layers:
        residual = cache.get_residual_stream(layer)
        if residual is None:
            results[layer] = 0.0
            continue

        # Remove batch dimension and handle sequence alignment
        reps = residual.squeeze(0)

        # Truncate if representations are longer than labels (due to tokenization)
        min_len = min(reps.shape[0], len(cluster_labels))
        reps = reps[:min_len]
        labels_list = cluster_labels[:min_len]

        # Exclude injected position if specified
        if exclude_position is not None and exclude_position < min_len:
            mask = [i for i in range(min_len) if i != exclude_position]
            reps = reps[mask]
            labels_list = [labels_list[i] for i in mask]

        labels = torch.tensor(labels_list)

        if len(torch.unique(labels)) < 2:
            results[layer] = 0.0
            continue

        try:
            phi = metric.compute(reps, cluster_labels=labels)
            results[layer] = phi
        except Exception:
            results[layer] = 0.0

    return results


def compute_per_token_losses(model: HookedLLM, prompt: str) -> torch.Tensor:
    """Compute per-token next-token prediction loss."""
    return model.compute_per_token_loss(prompt).squeeze(0)


# =============================================================================
# Main Experiment
# =============================================================================

def run_injection_experiment(
    model_name: str = "gpt2",
    context_lengths: list[int] = [20, 50, 100, 200],
    n_samples: int = 50,
    layers_to_test: Optional[list[int]] = None,
    position_step: int = 5,
    seed: int = 42,
    output_dir: str = "results/incongruous_injection",
    device: str = "auto",
    dtype: str = "bfloat16",
):
    """
    Run the incongruous token injection experiment.

    For each context length and sample:
    1. Generate a natural random walk (baseline)
    2. For each token type and position:
       - Inject incongruous token
       - Measure cluster separation delta
       - Measure loss spike
    3. Aggregate results by position, token type, layer

    Args:
        model_name: HuggingFace model identifier
        context_lengths: List of context lengths to test
        n_samples: Number of random walks per context length
        layers_to_test: Layer indices to analyze (auto-selected if None)
        position_step: Test every Nth position
        seed: Random seed
        output_dir: Directory for results
        device: Device for model
        dtype: Data type for model
    """
    # Setup
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    token_types = ["wrong_cluster", "random_vocab", "semantic_outlier", "out_of_vocab"]

    # Initialize graph
    print("Initializing hierarchical graph...")
    graph_config = HierarchicalGraphConfig(
        num_superclusters=3,
        nodes_per_cluster=5,
        walk_length=max(context_lengths) + 10,  # Extra for injection
        seed=seed,
    )
    graph = HierarchicalGraph(graph_config)
    print(f"  Graph: {graph.config.num_superclusters} clusters × "
          f"{graph.config.nodes_per_cluster} nodes = {graph.num_nodes} total")

    # Load model
    print(f"\nLoading model: {model_name}")
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16

    model = HookedLLM.from_pretrained(
        model_name,
        device=device,
        dtype=torch_dtype,
    )

    n_layers = model.num_layers
    print(f"  Layers: {n_layers}")
    print(f"  Hidden size: {model.hidden_size}")
    print(f"  Device: {model.device}")

    # Select layers to test if not specified
    if layers_to_test is None:
        # Sample 8 layers across the depth
        if n_layers >= 28:
            layers_to_test = [0, 4, 8, 12, 16, 20, 24, n_layers - 1]
        elif n_layers >= 12:
            layers_to_test = [0, 2, 4, 6, 8, 10, n_layers - 1]
        else:
            layers_to_test = list(range(n_layers))

    layers_to_test = [l for l in layers_to_test if l < n_layers]
    print(f"  Testing layers: {layers_to_test}")

    # Initialize results structure
    results = {
        "metadata": {
            "model": model_name,
            "context_lengths": context_lengths,
            "n_samples": n_samples,
            "layers": layers_to_test,
            "position_step": position_step,
            "token_types": token_types,
            "seed": seed,
            "timestamp": datetime.now().isoformat(),
        },
        "by_token_type": {tt: {"by_position": {}, "by_context_length": {}} for tt in token_types},
        "baselines": {},
        "raw_results": [],
    }

    # Main experiment loop
    total_iterations = len(context_lengths) * n_samples
    pbar = tqdm(total=total_iterations, desc="Running injection experiment")

    for N in context_lengths:
        print(f"\n{'='*60}")
        print(f"Context length N={N}")
        print(f"{'='*60}")

        # Generate positions: fine granularity early (1-5), then increasing gaps
        # Pattern: 1,2,3,4,5, then 10,15,20,30,50,75,100,150,200...
        positions = []
        # Early positions with fine granularity (skip 0, start at 1)
        for p in [1, 2, 3, 4, 5]:
            if p < N:
                positions.append(p)
        # Then increasing gaps: 10, 15, 20, 30, 50, 75, 100, 150, 200...
        sparse_positions = [10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500]
        for p in sparse_positions:
            if p < N and p not in positions:
                positions.append(p)
        positions = sorted(positions)
        print(f"  Positions to test: {positions}")

        # Storage for this context length
        baseline_phis = {layer: [] for layer in layers_to_test}
        injection_results = {
            tt: {pos: {layer: [] for layer in layers_to_test} for pos in positions}
            for tt in token_types
        }
        loss_results = {
            tt: {pos: {"at_injection": [], "after_injection": []} for pos in positions}
            for tt in token_types
        }

        for trial in range(n_samples):
            # Generate natural walk
            prompt, nodes = graph.generate_random_walk(length=N, return_nodes=True)
            context_tokens = prompt.split()
            clusters = [graph.get_cluster(n) for n in nodes]

            # Compute baseline metrics
            baseline_phi = compute_cluster_separation(model, prompt, clusters, layers_to_test)
            baseline_losses = compute_per_token_losses(model, prompt)

            for layer in layers_to_test:
                baseline_phis[layer].append(baseline_phi[layer])

            # Test each token type and position
            for token_type in token_types:
                for pos in positions:
                    # Get cluster at injection position
                    current_cluster = clusters[pos] if pos < len(clusters) else 0

                    # Generate incongruous token
                    incon_token = get_incongruous_token(token_type, graph, current_cluster)

                    # Inject token
                    injected_tokens = inject_token(context_tokens, incon_token, pos)
                    injected_prompt = " ".join(injected_tokens)

                    # Cluster labels for injected context:
                    # Insert a placeholder at the injection position, but exclude it from the metric
                    # This ensures cluster labels align with token positions
                    injected_clusters = clusters[:pos] + [0] + clusters[pos:]  # placeholder

                    # Compute metrics after injection, excluding the injected position
                    injected_phi = compute_cluster_separation(
                        model, injected_prompt, injected_clusters, layers_to_test,
                        exclude_position=pos
                    )

                    for layer in layers_to_test:
                        phi_delta = baseline_phi[layer] - injected_phi[layer]
                        injection_results[token_type][pos][layer].append(phi_delta)

                    # Compute loss at and after injection
                    try:
                        injected_losses = compute_per_token_losses(model, injected_prompt)

                        # Loss at injection position (model's surprise at incongruous token)
                        if pos < len(injected_losses):
                            loss_at = injected_losses[pos].item()
                        else:
                            loss_at = 0.0

                        # Average loss after injection
                        if pos + 1 < len(injected_losses):
                            loss_after = injected_losses[pos + 1:].mean().item()
                        else:
                            loss_after = 0.0

                        loss_results[token_type][pos]["at_injection"].append(loss_at)
                        loss_results[token_type][pos]["after_injection"].append(loss_after)

                    except Exception:
                        loss_results[token_type][pos]["at_injection"].append(0.0)
                        loss_results[token_type][pos]["after_injection"].append(0.0)

            # Store raw result for this trial
            results["raw_results"].append({
                "context_length": N,
                "trial": trial,
                "baseline_phi": {str(k): v for k, v in baseline_phi.items()},
            })

            pbar.update(1)

        # Aggregate results for this context length
        results["baselines"][N] = {
            str(layer): {
                "mean": float(np.mean(baseline_phis[layer])),
                "std": float(np.std(baseline_phis[layer])),
            }
            for layer in layers_to_test
        }

        for token_type in token_types:
            results["by_token_type"][token_type]["by_context_length"][N] = {}

            for pos in positions:
                pos_key = str(pos)

                # Phi delta by layer - store both mean and median due to outliers
                phi_deltas = {}
                for layer in layers_to_test:
                    vals = injection_results[token_type][pos][layer]
                    if vals:
                        vals_arr = np.array(vals)
                        abs_vals = np.abs(vals_arr)
                        phi_deltas[str(layer)] = {
                            "mean": float(np.mean(vals_arr)),
                            "median": float(np.median(vals_arr)),
                            "median_abs": float(np.median(abs_vals)),
                            "std": float(np.std(vals_arr)),
                            "iqr": float(np.percentile(vals_arr, 75) - np.percentile(vals_arr, 25)),
                            "iqr_abs": float(np.percentile(abs_vals, 75) - np.percentile(abs_vals, 25)),
                            "n": len(vals_arr),
                        }
                    else:
                        phi_deltas[str(layer)] = {
                            "mean": 0.0, "median": 0.0, "median_abs": 0.0,
                            "std": 0.0, "iqr": 0.0, "iqr_abs": 0.0, "n": 0
                        }

                # Loss metrics
                loss_at_vals = loss_results[token_type][pos]["at_injection"]
                loss_after_vals = loss_results[token_type][pos]["after_injection"]

                pos_results = {
                    "phi_delta_by_layer": phi_deltas,
                    "loss_at_injection": {
                        "mean": float(np.mean(loss_at_vals)) if loss_at_vals else 0.0,
                        "std": float(np.std(loss_at_vals)) if loss_at_vals else 0.0,
                    },
                    "loss_after_injection": {
                        "mean": float(np.mean(loss_after_vals)) if loss_after_vals else 0.0,
                        "std": float(np.std(loss_after_vals)) if loss_after_vals else 0.0,
                    },
                }

                # Store by position within token type
                if pos_key not in results["by_token_type"][token_type]["by_position"]:
                    results["by_token_type"][token_type]["by_position"][pos_key] = {}
                results["by_token_type"][token_type]["by_position"][pos_key][N] = pos_results

                # Store by context length
                results["by_token_type"][token_type]["by_context_length"][N][pos_key] = pos_results

    pbar.close()

    # Save results
    results_file = output_path / f"injection_results_{model_name.replace('/', '_')}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Generate visualizations
    create_visualizations(results, output_path, model_name)

    return results


# =============================================================================
# Visualizations
# =============================================================================

def create_visualizations(results: dict, output_dir: Path, model_name: str):
    """Generate visualization plots from experiment results."""
    import matplotlib.pyplot as plt

    print("\nGenerating visualizations...")

    metadata = results["metadata"]
    token_types = metadata["token_types"]
    context_lengths = metadata["context_lengths"]
    layers = metadata["layers"]

    # Darker color palette
    dark_colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']  # blue, red, green, purple

    # Figure 1: Position effect by token type - separate subplot per context length
    # This ensures each N has its own y-scale so SE shading is visible
    n_ctx = len(context_lengths)
    fig, axes = plt.subplots(len(token_types), n_ctx, figsize=(4 * n_ctx, 4 * len(token_types)))

    n_samples = metadata.get("n_samples", 50)
    mid_layer = str(layers[len(layers) // 2])

    for row_idx, token_type in enumerate(token_types):
        pos_data = results["by_token_type"][token_type]["by_position"]
        positions = sorted([int(p) for p in pos_data.keys()])

        for col_idx, N in enumerate(context_lengths):
            ax = axes[row_idx, col_idx] if len(token_types) > 1 else axes[col_idx]
            N_str = str(N)

            phi_deltas = []
            phi_stds = []
            valid_positions = []

            for pos in positions:
                pos_key = str(pos)
                pos_N_data = pos_data.get(pos_key, {}).get(N_str) or pos_data.get(pos_key, {}).get(N)
                if pos_N_data:
                    delta = pos_N_data.get("phi_delta_by_layer", {}).get(mid_layer, {})
                    if delta:
                        # Use median of absolute values to show effect magnitude
                        phi_deltas.append(delta.get("median_abs", abs(delta.get("median", 0))))
                        # Use IQR of absolute values / 2 for error bars
                        phi_stds.append(delta.get("iqr_abs", delta.get("iqr", 0)) / 2)
                        valid_positions.append(pos)

            if valid_positions:
                color = dark_colors[row_idx % len(dark_colors)]
                phi_deltas = np.array(phi_deltas)
                phi_iqr_half = np.array(phi_stds)  # Already IQR/2
                valid_positions = np.array(valid_positions)

                # Use IQR/2 directly as error (robust measure)
                phi_ses = phi_iqr_half

                # Plot line
                ax.plot(valid_positions, phi_deltas, 'o-',
                       color=color, linewidth=2, markersize=5)
                # Plot standard error as shaded region
                ax.fill_between(valid_positions,
                               phi_deltas - phi_ses,
                               phi_deltas + phi_ses,
                               alpha=0.3, color=color)

            ax.axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=1)
            ax.grid(True, alpha=0.3)

            # Labels
            if row_idx == len(token_types) - 1:
                ax.set_xlabel('Position', fontsize=10)
            if col_idx == 0:
                ax.set_ylabel('|Φ Delta|', fontsize=10)
            if row_idx == 0:
                ax.set_title(f'N={N}', fontsize=11, fontweight='bold')

            # Add token type label on the right
            if col_idx == n_ctx - 1:
                ax.annotate(token_type.replace('_', '\n'), xy=(1.05, 0.5),
                           xycoords='axes fraction', fontsize=9,
                           ha='left', va='center')

    plt.suptitle(f'Position Effect by Token Type (median |Φ delta| ± IQR/2)\n{model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    fig_path = output_dir / "position_effect_by_token_type.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")

    # Figure 2: Heatmap - Position vs Context Length (one per token type)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for idx, token_type in enumerate(token_types):
        ax = axes[idx // 2, idx % 2]

        pos_data = results["by_token_type"][token_type]["by_position"]
        positions = sorted([int(p) for p in pos_data.keys()])

        mid_layer = str(layers[len(layers) // 2])

        # Build heatmap matrix
        heatmap = np.zeros((len(context_lengths), len(positions)))
        for i, N in enumerate(context_lengths):
            for j, pos in enumerate(positions):
                pos_key = str(pos)
                if pos_key in pos_data and N in pos_data[pos_key]:
                    delta = pos_data[pos_key][N]["phi_delta_by_layer"].get(mid_layer, {})
                    heatmap[i, j] = delta.get("mean", 0.0)

        im = ax.imshow(heatmap, aspect='auto', cmap='RdBu_r', origin='lower')
        ax.set_xticks(range(len(positions)))
        ax.set_xticklabels(positions, rotation=45, fontsize=8)
        ax.set_yticks(range(len(context_lengths)))
        ax.set_yticklabels(context_lengths)
        ax.set_xlabel('Injection Position')
        ax.set_ylabel('Context Length')
        ax.set_title(f'{token_type.replace("_", " ").title()}')
        plt.colorbar(im, ax=ax, label='Φ Delta')

    plt.suptitle(f'Injection Effect Heatmap (Position × Context Length)\n{model_name}', fontsize=14)
    plt.tight_layout()

    fig_path = output_dir / "heatmap_position_vs_context.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")

    # Figure 3: Layer sensitivity comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Early vs Late injection by layer
    ax = axes[0]
    N = max(context_lengths)
    N_str = str(N)

    # Find earliest and latest positions that have data
    avail_positions = [int(p) for p in results["by_token_type"]["wrong_cluster"]["by_position"].keys()
                       if N_str in results["by_token_type"]["wrong_cluster"]["by_position"].get(p, {})]
    if avail_positions:
        early_pos = str(min(avail_positions))
        late_pos = str(max(avail_positions))
    else:
        early_pos = "1"
        late_pos = "50"

    layer_colors = {'wrong_cluster': ('#1f77b4', '#aec7e8'),
                    'semantic_outlier': ('#d62728', '#ffbb78')}

    for token_type in ["wrong_cluster", "semantic_outlier"]:
        pos_data = results["by_token_type"][token_type]["by_position"]
        color_early, color_late = layer_colors.get(token_type, ('#333333', '#999999'))

        early_deltas = []
        early_stds = []
        late_deltas = []
        late_stds = []

        for layer in layers:
            layer_key = str(layer)
            early_data = pos_data.get(early_pos, {}).get(N_str, {}).get("phi_delta_by_layer", {}).get(layer_key, {})
            late_data = pos_data.get(late_pos, {}).get(N_str, {}).get("phi_delta_by_layer", {}).get(layer_key, {})

            early_deltas.append(early_data.get("mean", 0.0))
            early_stds.append(early_data.get("std", 0.0))
            late_deltas.append(late_data.get("mean", 0.0))
            late_stds.append(late_data.get("std", 0.0))

        early_deltas = np.array(early_deltas)
        early_stds = np.array(early_stds)
        late_deltas = np.array(late_deltas)
        late_stds = np.array(late_stds)
        layers_arr = np.array(layers)

        # Convert std to standard error
        n_samples = metadata.get("n_samples", 50)
        early_ses = early_stds / np.sqrt(n_samples)
        late_ses = late_stds / np.sqrt(n_samples)

        ax.plot(layers_arr, early_deltas, 'o-', label=f'{token_type} (pos {early_pos})',
               color=color_early, linewidth=2, markersize=6)
        ax.fill_between(layers_arr, early_deltas - early_ses, early_deltas + early_ses,
                       alpha=0.2, color=color_early)
        ax.plot(layers_arr, late_deltas, 's--', label=f'{token_type} (pos {late_pos})',
               color=color_late, linewidth=2, markersize=6)
        ax.fill_between(layers_arr, late_deltas - late_ses, late_deltas + late_ses,
                       alpha=0.2, color=color_late)

    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('Φ Delta', fontsize=11)
    ax.set_title(f'Early vs Late Injection by Layer (N={N})', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=1)
    ax.grid(True, alpha=0.3)

    # Right: Loss at injection by token type with std
    ax = axes[1]
    for color_idx, token_type in enumerate(token_types):
        pos_data = results["by_token_type"][token_type]["by_position"]
        positions = sorted([int(p) for p in pos_data.keys()])

        losses = []
        loss_stds = []
        valid_positions = []
        for pos in positions:
            pos_key = str(pos)
            if pos_key in pos_data and N_str in pos_data[pos_key]:
                loss = pos_data[pos_key][N_str].get("loss_at_injection", {})
                if loss:
                    losses.append(loss["mean"])
                    loss_stds.append(loss.get("std", 0))
                    valid_positions.append(pos)

        if valid_positions:
            color = dark_colors[color_idx % len(dark_colors)]
            losses = np.array(losses)
            loss_stds = np.array(loss_stds)
            valid_positions = np.array(valid_positions)

            # Convert std to standard error
            n_samples = metadata.get("n_samples", 50)
            loss_ses = loss_stds / np.sqrt(n_samples)

            ax.plot(valid_positions, losses, 'o-', label=token_type.replace('_', ' '),
                   color=color, linewidth=2, markersize=6)
            ax.fill_between(valid_positions, losses - loss_ses, losses + loss_ses,
                           alpha=0.2, color=color)

    ax.set_xlabel('Injection Position', fontsize=11)
    ax.set_ylabel('Loss at Injection', fontsize=11)
    ax.set_title(f'Model Surprise at Incongruous Token (N={N})', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Layer Sensitivity Analysis (±1 SE shaded)\n{model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    fig_path = output_dir / "layer_sensitivity_analysis.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")

    # Figure 4: Summary comparison across token types
    fig, ax = plt.subplots(figsize=(10, 6))

    # Average phi delta across all positions for each token type
    bar_width = 0.2
    x = np.arange(len(context_lengths))

    n_samples = metadata.get("n_samples", 50)
    for i, token_type in enumerate(token_types):
        avg_deltas = []
        se_deltas = []
        for N in context_lengths:
            N_str = str(N)
            ctx_data = results["by_token_type"][token_type]["by_context_length"].get(N_str, {})
            if not ctx_data:
                ctx_data = results["by_token_type"][token_type]["by_context_length"].get(N, {})
            deltas = []
            mid_layer = str(layers[len(layers) // 2])
            for pos_key, pos_data in ctx_data.items():
                delta = pos_data.get("phi_delta_by_layer", {}).get(mid_layer, {})
                if delta:
                    deltas.append(delta["mean"])
            avg_deltas.append(np.mean(deltas) if deltas else 0.0)
            # Standard error = std / sqrt(n)
            se_deltas.append(np.std(deltas) / np.sqrt(len(deltas)) if deltas else 0.0)

        color = dark_colors[i % len(dark_colors)]
        ax.bar(x + i * bar_width, avg_deltas, bar_width,
               label=token_type.replace('_', ' ').title(),
               color=color, yerr=se_deltas, capsize=3, error_kw={'linewidth': 1.5})

    ax.set_xlabel('Context Length', fontsize=11)
    ax.set_ylabel('Average Φ Delta', fontsize=11)
    ax.set_title(f'Injection Impact by Token Type (±1 SE error bars)\n{model_name}', fontsize=12, fontweight='bold')
    ax.set_xticks(x + bar_width * 1.5)
    ax.set_xticklabels(context_lengths)
    ax.legend(fontsize=10)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig_path = output_dir / "token_type_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Incongruous Token Injection Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--model", type=str, default="gpt2",
        help="HuggingFace model name (default: gpt2)"
    )
    parser.add_argument(
        "--context-lengths", type=str, default="20,50,100,200",
        help="Comma-separated context lengths (default: 20,50,100,200)"
    )
    parser.add_argument(
        "--n-samples", type=int, default=50,
        help="Number of samples per context length (default: 50)"
    )
    parser.add_argument(
        "--layers", type=str, default=None,
        help="Comma-separated layer indices (default: auto-select)"
    )
    parser.add_argument(
        "--position-step", type=int, default=5,
        help="Test every Nth position (default: 5)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/incongruous_injection",
        help="Output directory (default: results/incongruous_injection)"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device (default: auto)"
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"],
        help="Model dtype (default: bfloat16)"
    )

    args = parser.parse_args()

    # Parse arguments
    context_lengths = [int(x) for x in args.context_lengths.split(",")]
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None

    # Run experiment
    run_injection_experiment(
        model_name=args.model,
        context_lengths=context_lengths,
        n_samples=args.n_samples,
        layers_to_test=layers,
        position_step=args.position_step,
        seed=args.seed,
        output_dir=args.output_dir,
        device=args.device,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
