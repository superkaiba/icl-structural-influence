#!/usr/bin/env python3
"""
Graph Adherence Experiment: Compare 4 metrics for measuring H1 vs H2 structure.

Metrics:
1. Dirichlet Energy Ratio - smoothness over graph edges
2. Cluster Separation Ratio - cluster quality under each interpretation
3. Linear Probe Accuracy Difference - decodability of cluster labels
4. Neighbor Consistency Ratio - k-NN overlap with graph neighbors

Tests multiple disambiguation points on context length >= 1000.
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.hooked_model import HookedLLM
from src.data.dual_interpretation_graph import (
    DualInterpretationGraph,
    DualInterpretationConfig,
)
from src.metrics.graph_adherence_metrics import (
    compute_graph_adherence_metrics,
    GraphAdherenceMetrics,
)


def run_experiment(
    model_name: str = "Qwen/Qwen2.5-7B",
    context_length: int = 1000,
    disambig_percentages: list[float] = None,
    n_trials: int = 10,
    layers: list[int] = None,
    window_size: int = 50,
    checkpoints: list[int] = None,
    output_dir: str = "results/graph_adherence",
    dtype: str = "bfloat16",
    seed: int = 42,
):
    """Run the graph adherence experiment."""

    if disambig_percentages is None:
        disambig_percentages = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]  # 1.0 = no disambig (fully ambiguous)

    if layers is None:
        layers = [0, 7, 14, 21, 27]

    if checkpoints is None:
        # Dense checkpoints
        checkpoints = list(range(50, context_length + 1, 50))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save config
    config = {
        "model": model_name,
        "context_length": context_length,
        "disambig_percentages": disambig_percentages,
        "n_trials": n_trials,
        "layers": layers,
        "window_size": window_size,
        "checkpoints": checkpoints,
        "seed": seed,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Loading model: {model_name}")
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
    model = HookedLLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    # Create graph
    print("Creating dual interpretation graph...")
    graph_config = DualInterpretationConfig(
        vocab_size=15,
        clusters_per_interpretation=3,
        p_intra_cluster=0.8,
        p_inter_cluster=0.15,
        seed=seed,
    )
    graph = DualInterpretationGraph(graph_config)

    # Get graph structures
    adj_h1 = graph.G1_adj
    adj_h2 = graph.G2_adj

    results = defaultdict(lambda: defaultdict(list))

    for disambig_pct in disambig_percentages:
        condition = f"disambig_{int(disambig_pct * 100)}pct" if disambig_pct < 1.0 else "fully_ambiguous"
        print(f"\n{'='*60}")
        print(f"Condition: {condition}")
        print(f"{'='*60}")

        for trial in range(n_trials):
            print(f"\nTrial {trial + 1}/{n_trials}")
            np.random.seed(seed + trial)
            torch.manual_seed(seed + trial)

            # Generate walk
            if disambig_pct >= 1.0:
                # Fully ambiguous - no disambiguation
                prompt, nodes, metadata = graph.generate_ambiguous_walk(
                    context_length,
                    disambig_position=None,
                    true_hypothesis="H1",
                    return_nodes=True,
                )
            else:
                disambig_pos = int(context_length * disambig_pct)
                prompt, nodes, metadata = graph.generate_ambiguous_walk(
                    context_length,
                    disambig_position=disambig_pos,
                    true_hypothesis="H1",
                    return_nodes=True,
                )

            tokens = prompt.split()

            # Get labels for each token
            labels_h1 = [graph.H1_clusters[n] for n in nodes]
            labels_h2 = [graph.H2_clusters[n] for n in nodes]

            # Process and collect representations
            print("  Extracting representations...")
            with torch.no_grad():
                outputs, cache = model.forward_with_cache(prompt, layers=layers)

            # Get number of model tokens
            n_model_tokens = cache.get_residual_stream(layers[0]).shape[1]

            # Map model tokens back to our tokens (approximately)
            # This is a simplification - assumes 1:1 mapping
            token_to_node = nodes[:n_model_tokens]
            token_labels_h1 = labels_h1[:n_model_tokens]
            token_labels_h2 = labels_h2[:n_model_tokens]

            # Compute metrics at each checkpoint for each layer
            for layer in layers:
                reps = cache.get_residual_stream(layer)[0].cpu().float().numpy()

                for cp in checkpoints:
                    if cp >= len(reps):
                        continue

                    # Get window
                    start = max(0, cp - window_size + 1)
                    window_reps = reps[start:cp + 1]
                    window_labels_h1 = np.array(token_labels_h1[start:cp + 1])
                    window_labels_h2 = np.array(token_labels_h2[start:cp + 1])
                    window_nodes = token_to_node[start:cp + 1]

                    if len(window_reps) < 10:
                        continue

                    # Compute metrics
                    metrics = compute_graph_adherence_metrics(
                        window_reps,
                        adj_h1,
                        adj_h2,
                        window_labels_h1,
                        window_labels_h2,
                        window_nodes,
                    )

                    results[condition][f"layer_{layer}_cp_{cp}"].append(metrics.to_dict())

            # Cleanup
            del cache
            torch.cuda.empty_cache()

        # Save intermediate results
        save_results(results, output_path / "results_intermediate.json")

    # Save final results
    save_results(results, output_path / "results.json")
    print(f"\nResults saved to {output_path}")

    return results


def save_results(results: dict, path: Path):
    """Save results to JSON."""
    # Convert defaultdict to regular dict
    serializable = {}
    for condition, data in results.items():
        serializable[condition] = {}
        for key, trials in data.items():
            serializable[condition][key] = trials

    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


def plot_results(results_path: str, output_dir: str = None):
    """Generate plots from results."""
    import matplotlib.pyplot as plt

    with open(results_path) as f:
        results = json.load(f)

    if output_dir is None:
        output_dir = Path(results_path).parent / "plots"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Parse results to get data by condition, layer, checkpoint
    conditions = list(results.keys())

    # Get all layers and checkpoints
    sample_key = list(results[conditions[0]].keys())[0]
    # Parse layer_X_cp_Y format
    all_layers = set()
    all_checkpoints = set()
    for key in results[conditions[0]].keys():
        parts = key.split("_")
        layer = int(parts[1])
        cp = int(parts[3])
        all_layers.add(layer)
        all_checkpoints.add(cp)

    layers = sorted(all_layers)
    checkpoints = sorted(all_checkpoints)

    # Helper function to get disambiguation percentage from condition name
    def get_disambig_pct(cond):
        if "fully" in cond:
            return 100
        return int(cond.split("_")[1].replace("pct", ""))

    # Metrics to plot
    metrics = [
        ("dirichlet_ratio", "Dirichlet Energy Ratio", "H1/(H1+H2) - Higher = more H2-adherent"),
        ("separation_ratio", "Cluster Separation Ratio", "H1/(H1+H2) - Higher = more H1-adherent"),
        ("probe_diff", "Linear Probe Accuracy Diff", "H1 - H2 - Positive = H1 more decodable"),
        ("neighbor_ratio", "Neighbor Consistency Ratio", "H1/(H1+H2) - Higher = more H1-adherent"),
    ]

    # Use final layer for main comparison
    final_layer = max(layers)

    # Plot 1: All 4 metrics over context for each condition (final layer)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors = plt.cm.viridis(np.linspace(0, 1, len(conditions)))

    for idx, (metric_key, metric_name, metric_desc) in enumerate(metrics):
        ax = axes[idx]

        for cond_idx, condition in enumerate(sorted(conditions)):
            means = []
            stds = []

            for cp in checkpoints:
                key = f"layer_{final_layer}_cp_{cp}"
                if key in results[condition]:
                    trials = results[condition][key]
                    values = [t[metric_key] for t in trials]
                    means.append(np.mean(values))
                    stds.append(np.std(values))
                else:
                    means.append(np.nan)
                    stds.append(np.nan)

            means = np.array(means)
            stds = np.array(stds)

            label = condition.replace("_", " ").replace("pct", "%")
            ax.plot(checkpoints, means, label=label, color=colors[cond_idx], linewidth=2)
            ax.fill_between(checkpoints, means - stds, means + stds, alpha=0.2, color=colors[cond_idx])

            # Add vertical line at disambiguation point
            disambig_pct = get_disambig_pct(condition)
            if disambig_pct < 100:
                disambig_pos = disambig_pct * max(checkpoints) / 100
                ax.axvline(x=disambig_pos, color=colors[cond_idx], linestyle='--', alpha=0.5, linewidth=1)

        ax.set_xlabel("Context Position")
        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name}\n({metric_desc})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Graph Adherence Metrics Over Context (Layer {final_layer})", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path / "all_metrics_by_condition.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 2: Compare metrics at final checkpoint for different disambiguations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    sorted_conditions = sorted(conditions, key=get_disambig_pct)
    disambig_pcts = [get_disambig_pct(c) for c in sorted_conditions]

    final_cp = max(checkpoints)

    for idx, (metric_key, metric_name, metric_desc) in enumerate(metrics):
        ax = axes[idx]

        for layer in layers:
            means = []
            stds = []

            for condition in sorted_conditions:
                key = f"layer_{layer}_cp_{final_cp}"
                if key in results[condition]:
                    trials = results[condition][key]
                    values = [t[metric_key] for t in trials]
                    means.append(np.mean(values))
                    stds.append(np.std(values))
                else:
                    means.append(np.nan)
                    stds.append(np.nan)

            ax.errorbar(disambig_pcts, means, yerr=stds, label=f"Layer {layer}", marker='o', capsize=3)

        ax.set_xlabel("Disambiguation Point (%)")
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Graph Adherence vs Disambiguation Point (Context Position {final_cp})", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path / "metrics_vs_disambiguation.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 3: Layer comparison for each metric
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Pick a specific condition (e.g., 25% disambiguation)
    target_condition = "disambig_25pct" if "disambig_25pct" in conditions else sorted_conditions[1]

    for idx, (metric_key, metric_name, metric_desc) in enumerate(metrics):
        ax = axes[idx]

        for layer in layers:
            means = []
            stds = []

            for cp in checkpoints:
                key = f"layer_{layer}_cp_{cp}"
                if key in results[target_condition]:
                    trials = results[target_condition][key]
                    values = [t[metric_key] for t in trials]
                    means.append(np.mean(values))
                    stds.append(np.std(values))
                else:
                    means.append(np.nan)
                    stds.append(np.nan)

            means = np.array(means)
            ax.plot(checkpoints, means, label=f"Layer {layer}", linewidth=2)

        ax.set_xlabel("Context Position")
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Mark disambiguation point
        disambig_pos = get_disambig_pct(target_condition) * max(checkpoints) / 100
        ax.axvline(x=disambig_pos, color='red', linestyle='--', alpha=0.5, label='Disambig')

    plt.suptitle(f"Layer Comparison - {target_condition.replace('_', ' ')}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path / "layer_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 4: Normalized comparison of all metrics (to see which is most sensitive)
    fig, ax = plt.subplots(figsize=(12, 6))

    # For 25% disambiguation condition, final layer
    target_condition = "disambig_25pct" if "disambig_25pct" in conditions else sorted_conditions[1]

    for metric_key, metric_name, _ in metrics:
        means = []
        for cp in checkpoints:
            key = f"layer_{final_layer}_cp_{cp}"
            if key in results[target_condition]:
                trials = results[target_condition][key]
                values = [t[metric_key] for t in trials]
                means.append(np.mean(values))
            else:
                means.append(np.nan)

        # Normalize to [0, 1] for comparison
        means = np.array(means)
        if not np.all(np.isnan(means)):
            means_norm = (means - np.nanmin(means)) / (np.nanmax(means) - np.nanmin(means) + 1e-8)
            ax.plot(checkpoints, means_norm, label=metric_name, linewidth=2)

    ax.set_xlabel("Context Position")
    ax.set_ylabel("Normalized Metric Value")
    ax.set_title(f"Metric Comparison (Normalized) - {target_condition.replace('_', ' ')}, Layer {final_layer}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mark disambiguation point
    disambig_pos = get_disambig_pct(target_condition) * max(checkpoints) / 100
    ax.axvline(x=disambig_pos, color='red', linestyle='--', alpha=0.5, label='Disambig')

    plt.tight_layout()
    plt.savefig(output_path / "metric_comparison_normalized.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 5: Per-layer plots for each metric (separate figure per metric)
    for metric_key, metric_name, metric_desc in metrics:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()

        for layer_idx, layer in enumerate(layers):
            if layer_idx >= len(axes):
                break
            ax = axes[layer_idx]

            for cond_idx, condition in enumerate(sorted(conditions, key=get_disambig_pct)):
                means = []
                stds = []

                for cp in checkpoints:
                    key = f"layer_{layer}_cp_{cp}"
                    if key in results[condition]:
                        trials = results[condition][key]
                        values = [t[metric_key] for t in trials]
                        means.append(np.mean(values))
                        stds.append(np.std(values))
                    else:
                        means.append(np.nan)
                        stds.append(np.nan)

                means = np.array(means)
                stds = np.array(stds)

                label = condition.replace("_", " ").replace("pct", "%")
                ax.plot(checkpoints, means, label=label, color=colors[cond_idx], linewidth=2)
                ax.fill_between(checkpoints, means - stds, means + stds, alpha=0.15, color=colors[cond_idx])

                # Add vertical line at disambiguation point
                disambig_pct = get_disambig_pct(condition)
                if disambig_pct < 100:
                    disambig_pos = disambig_pct * max(checkpoints) / 100
                    ax.axvline(x=disambig_pos, color=colors[cond_idx], linestyle='--', alpha=0.5, linewidth=1)

            ax.set_xlabel("Context Position")
            ax.set_ylabel(metric_name)
            ax.set_title(f"Layer {layer}")
            ax.grid(True, alpha=0.3)

        # Add legend to last subplot
        axes[-1].legend(loc='center', fontsize=9)
        axes[-1].axis('off')

        plt.suptitle(f"{metric_name} by Layer\n({metric_desc})", fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path / f"{metric_key}_by_layer.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Plots saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph Adherence Experiment")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--context-length", type=int, default=1000)
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="results/graph_adherence")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot-only", type=str, default=None, help="Path to results.json to plot")

    args = parser.parse_args()

    if args.plot_only:
        plot_results(args.plot_only)
    else:
        results = run_experiment(
            model_name=args.model,
            context_length=args.context_length,
            n_trials=args.n_trials,
            output_dir=args.output_dir,
            dtype=args.dtype,
            seed=args.seed,
        )
        plot_results(f"{args.output_dir}/results.json", args.output_dir)
