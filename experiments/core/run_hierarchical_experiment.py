#!/usr/bin/env python3
"""
Hierarchical Graph Tracing Experiment: Testing Stagewise ICL Learning

This experiment tests three key hypotheses:
1. Influence in ICL is Non-Monotonic - sensitivity fluctuates with context length
2. Geometry is Learned Hierarchically - super-clusters before sub-nodes
3. Structural Influence as Diagnostic - identifies "anchor" tokens

We run across multiple SOTA models and varying context lengths to generate
the "Money Plot": Context Length vs Influence showing two waves of peaks.

Models tested (largest that fit on 3x A40):
- Llama 4 Scout (MoE 109B total, 17B active)
- Qwen 3 14B
- Gemma 3 12B
- Mistral Nemo 12B
"""

import argparse
import json
import os
import gc
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Local imports
from src.data import HierarchicalGraph, HierarchicalGraphConfig
from src.models import HookedLLM
from src.metrics import (
    DirichletEnergy,
    ClusterSeparation,
    RepresentationCoherence,
    ContextSensitivityScore,
)


# =============================================================================
# Configuration
# =============================================================================

MODELS_TO_TEST = [
    # Qwen 3 14B (April 2025) - LARGEST accessible
    {
        "name": "Qwen3-14B",
        "hf_id": "Qwen/Qwen3-14B",
        "dtype": "bfloat16",
        "mem_gb": 28,
    },
    # Gemma 3 12B (March 2025) - multimodal text decoder
    {
        "name": "Gemma-3-12B",
        "hf_id": "google/gemma-3-12b-it",
        "dtype": "bfloat16",
        "mem_gb": 24,
    },
    # Mistral Nemo 12B (July 2024) - already completed in prior run
    {
        "name": "Mistral-Nemo-12B",
        "hf_id": "mistralai/Mistral-Nemo-Instruct-2407",
        "dtype": "bfloat16",
        "mem_gb": 24,
    },
    # LLaMA 3.1 8B for comparison (gated but we have token)
    {
        "name": "LLaMA-3.1-8B",
        "hf_id": "meta-llama/Llama-3.1-8B",
        "dtype": "bfloat16",
        "mem_gb": 16,
    },
]

# Context lengths to test (key points to detect phase transitions)
CONTEXT_LENGTHS = [10, 20, 30, 50, 75, 100, 150, 200]

# Number of random walks per context length
N_CONTEXTS_PER_LENGTH = 25

# Graph configuration
GRAPH_CONFIG = {
    "num_superclusters": 3,
    "nodes_per_cluster": 5,
    "p_intra_cluster": 0.8,
    "p_inter_cluster": 0.1,
}


@dataclass
class ExperimentConfig:
    """Configuration for the full experiment."""
    models: list
    context_lengths: list
    n_contexts: int
    graph_config: dict
    output_dir: str
    seed: int = 42
    layers_to_analyze: Optional[list] = None  # None = auto-select


def setup_dtype(dtype_str: str) -> torch.dtype:
    """Convert string to torch dtype."""
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_str]


def clear_gpu_memory():
    """Clear GPU memory between models."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def convert_numpy(obj):
    """Convert numpy arrays to JSON-serializable types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


# =============================================================================
# Core Experiment Functions
# =============================================================================

def compute_structural_metrics_at_context_length(
    model: HookedLLM,
    graph: HierarchicalGraph,
    context_length: int,
    n_contexts: int,
    layers: list,
) -> dict:
    """
    Compute structural metrics for multiple contexts at a given length.

    Returns per-position sensitivity scores and aggregate metrics.
    """
    all_losses = []
    all_representations = {layer: [] for layer in layers}
    all_clusters = []

    # Structural metrics
    metrics = {
        "dirichlet": DirichletEnergy(),
        "cluster_sep": ClusterSeparation(),
        "coherence": RepresentationCoherence(),
    }

    # Per-context metric values
    per_context_metrics = {name: [] for name in metrics.keys()}

    for _ in range(n_contexts):
        # Generate random walk of specified length
        prompt, nodes = graph.generate_random_walk(
            length=context_length,
            return_nodes=True
        )
        clusters = torch.tensor([graph.get_cluster(n) for n in nodes])

        try:
            # Get losses
            token_losses = model.compute_per_token_loss(prompt)
            loss_len = token_losses.squeeze(0).shape[0]

            # Get representations
            _, cache = model.forward_with_cache(prompt, layers=layers)

            all_losses.append(token_losses.squeeze(0).cpu())

            for layer in layers:
                residual = cache.get_residual_stream(layer)
                if residual is not None:
                    # Align representations with losses (next-token prediction)
                    reps = residual.squeeze(0)[:-1].cpu()
                    rep_len = reps.shape[0]

                    # Create cluster labels that match representation length
                    # Repeat cluster labels to match tokenized length
                    n_nodes = len(clusters)
                    if rep_len != n_nodes - 1:
                        # Tokenization created different length - interpolate clusters
                        cluster_indices = np.linspace(1, n_nodes - 1, rep_len).astype(int)
                        cluster_indices = np.clip(cluster_indices, 0, n_nodes - 1)
                        aligned_clusters = clusters[cluster_indices]
                    else:
                        aligned_clusters = clusters[1:]

                    all_representations[layer].append(reps)

                    # Store aligned clusters only once (same for all layers)
                    if layer == layers[0]:
                        all_clusters.append(aligned_clusters)

                    # Compute metrics for this context at this layer
                    if layer == layers[-1]:  # Use final layer for per-context metrics
                        for name, metric in metrics.items():
                            try:
                                val = metric.compute(reps, cluster_labels=aligned_clusters)
                                per_context_metrics[name].append(val)
                            except Exception as metric_e:
                                per_context_metrics[name].append(np.nan)

        except Exception as e:
            print(f"    Warning: Error processing context: {e}")
            continue

    if len(all_losses) < 2:
        return None

    # Compute CSS for each layer
    results = {
        "context_length": context_length,
        "n_contexts_processed": len(all_losses),
        "per_context_metrics": {k: np.array(v) for k, v in per_context_metrics.items()},
        "layers": {},
    }

    for layer in layers:
        if len(all_representations[layer]) < 2:
            continue

        layer_results = {}

        for metric_name, metric in metrics.items():
            css = ContextSensitivityScore(metric)

            try:
                sensitivity_result = css.compute_batch(
                    all_losses,
                    all_representations[layer],
                    all_clusters,
                )

                layer_results[metric_name] = {
                    "phi_mean": float(sensitivity_result['phi_mean']),
                    "phi_std": float(sensitivity_result['phi_std']),
                    "position_sensitivities": sensitivity_result['position_sensitivities'].tolist(),
                    "max_sensitivity": float(np.max(np.abs(sensitivity_result['position_sensitivities']))),
                    "mean_abs_sensitivity": float(np.mean(np.abs(sensitivity_result['position_sensitivities']))),
                }

                # Hierarchical decomposition for cluster separation
                if metric_name == "cluster_sep":
                    decomp = css.compute_hierarchical_decomposition(
                        all_losses,
                        all_representations[layer],
                        all_clusters,
                    )
                    layer_results["hierarchical"] = {
                        "within_sensitivities": decomp['within_cluster_sensitivities'].tolist(),
                        "between_sensitivities": decomp['between_cluster_sensitivities'].tolist(),
                        "within_phi_mean": float(decomp['within_phi_values'].mean()),
                        "between_phi_mean": float(decomp['between_phi_values'].mean()),
                    }

            except Exception as e:
                layer_results[metric_name] = {"error": str(e)}

        results["layers"][f"layer_{layer}"] = layer_results

    return results


def run_single_model_experiment(
    model_config: dict,
    graph: HierarchicalGraph,
    context_lengths: list,
    n_contexts: int,
    output_dir: Path,
) -> dict:
    """
    Run the full experiment for a single model.
    """
    model_name = model_config["name"]
    print(f"\n{'='*70}")
    print(f"Running experiment for: {model_name}")
    print(f"{'='*70}")

    # Load model
    print(f"\nLoading {model_config['hf_id']}...")
    try:
        model = HookedLLM.from_pretrained(
            model_config["hf_id"],
            device="auto",
            dtype=setup_dtype(model_config["dtype"]),
        )
        print(f"  Loaded successfully. Layers: {model.num_layers}, Hidden: {model.hidden_size}")
    except Exception as e:
        print(f"  ERROR loading model: {e}")
        return {"model": model_name, "error": str(e)}

    # Select layers to analyze (evenly spaced)
    n_layers = model.num_layers
    layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    layers = sorted(set(layers))
    print(f"  Analyzing layers: {layers}")

    # Results container
    results = {
        "model": model_name,
        "hf_id": model_config["hf_id"],
        "n_layers": n_layers,
        "hidden_size": model.hidden_size,
        "layers_analyzed": layers,
        "context_length_results": {},
    }

    # Run for each context length
    for ctx_len in tqdm(context_lengths, desc=f"Context lengths for {model_name}"):
        print(f"\n  Context length: {ctx_len}")

        ctx_results = compute_structural_metrics_at_context_length(
            model=model,
            graph=graph,
            context_length=ctx_len,
            n_contexts=n_contexts,
            layers=layers,
        )

        if ctx_results is not None:
            results["context_length_results"][str(ctx_len)] = ctx_results

            # Print summary
            final_layer = f"layer_{layers[-1]}"
            if final_layer in ctx_results.get("layers", {}):
                layer_data = ctx_results["layers"][final_layer]
                if "cluster_sep" in layer_data and "phi_mean" in layer_data["cluster_sep"]:
                    print(f"    Cluster Sep Φ: {layer_data['cluster_sep']['phi_mean']:.4f}")
                    print(f"    Max CSS: {layer_data['cluster_sep']['max_sensitivity']:.4f}")

    # Cleanup
    del model
    clear_gpu_memory()

    # Save model results
    model_results_path = output_dir / f"{model_name.replace('/', '_')}_results.json"
    with open(model_results_path, "w") as f:
        json.dump(convert_numpy(results), f, indent=2)
    print(f"\n  Results saved to: {model_results_path}")

    return results


# =============================================================================
# Analysis and Visualization
# =============================================================================

def generate_money_plot(all_results: dict, output_dir: Path):
    """
    Generate the "Money Plot": Context Length vs Structural Metrics

    Shows:
    1. How cluster separation (Φ) evolves with context
    2. How sensitivity changes - looking for two waves
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    # Plot 1: Cluster Separation vs Context Length
    ax1 = axes[0, 0]
    for (model_name, results), color in zip(all_results.items(), colors):
        if "error" in results:
            continue

        ctx_lengths = []
        phi_means = []
        phi_stds = []

        for ctx_len, ctx_data in results.get("context_length_results", {}).items():
            layers = ctx_data.get("layers", {})
            # Use the deepest layer
            layer_keys = sorted(layers.keys(), key=lambda x: int(x.split("_")[1]))
            if layer_keys:
                final_layer = layers[layer_keys[-1]]
                if "cluster_sep" in final_layer and "phi_mean" in final_layer["cluster_sep"]:
                    ctx_lengths.append(int(ctx_len))
                    phi_means.append(final_layer["cluster_sep"]["phi_mean"])
                    phi_stds.append(final_layer["cluster_sep"]["phi_std"])

        if ctx_lengths:
            ctx_lengths, phi_means, phi_stds = zip(*sorted(zip(ctx_lengths, phi_means, phi_stds)))
            ax1.plot(ctx_lengths, phi_means, 'o-', label=model_name, color=color, linewidth=2)
            ax1.fill_between(ctx_lengths,
                            np.array(phi_means) - np.array(phi_stds),
                            np.array(phi_means) + np.array(phi_stds),
                            alpha=0.2, color=color)

    ax1.set_xlabel("Context Length (tokens)")
    ax1.set_ylabel("Cluster Separation (Φ)")
    ax1.set_title("Structural Emergence: Cluster Separation vs Context")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Max Sensitivity vs Context Length
    ax2 = axes[0, 1]
    for (model_name, results), color in zip(all_results.items(), colors):
        if "error" in results:
            continue

        ctx_lengths = []
        max_sens = []

        for ctx_len, ctx_data in results.get("context_length_results", {}).items():
            layers = ctx_data.get("layers", {})
            layer_keys = sorted(layers.keys(), key=lambda x: int(x.split("_")[1]))
            if layer_keys:
                final_layer = layers[layer_keys[-1]]
                if "cluster_sep" in final_layer and "max_sensitivity" in final_layer["cluster_sep"]:
                    ctx_lengths.append(int(ctx_len))
                    max_sens.append(final_layer["cluster_sep"]["max_sensitivity"])

        if ctx_lengths:
            ctx_lengths, max_sens = zip(*sorted(zip(ctx_lengths, max_sens)))
            ax2.plot(ctx_lengths, max_sens, 'o-', label=model_name, color=color, linewidth=2)

    ax2.set_xlabel("Context Length (tokens)")
    ax2.set_ylabel("Max Sensitivity (|CSS|)")
    ax2.set_title("Peak Sensitivity vs Context Length")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Within vs Between Cluster Sensitivity (Hierarchical Decomposition)
    ax3 = axes[1, 0]
    for (model_name, results), color in zip(all_results.items(), colors):
        if "error" in results:
            continue

        ctx_lengths = []
        within_max = []
        between_max = []

        for ctx_len, ctx_data in results.get("context_length_results", {}).items():
            layers = ctx_data.get("layers", {})
            layer_keys = sorted(layers.keys(), key=lambda x: int(x.split("_")[1]))
            if layer_keys:
                final_layer = layers[layer_keys[-1]]
                if "hierarchical" in final_layer:
                    h = final_layer["hierarchical"]
                    ctx_lengths.append(int(ctx_len))
                    within_max.append(np.max(np.abs(h["within_sensitivities"])))
                    between_max.append(np.max(np.abs(h["between_sensitivities"])))

        if ctx_lengths:
            ctx_lengths, within_max, between_max = zip(*sorted(zip(ctx_lengths, within_max, between_max)))
            ax3.plot(ctx_lengths, between_max, 'o-', label=f"{model_name} (between)", color=color, linewidth=2)
            ax3.plot(ctx_lengths, within_max, 's--', label=f"{model_name} (within)", color=color, alpha=0.6, linewidth=1.5)

    ax3.set_xlabel("Context Length (tokens)")
    ax3.set_ylabel("Max Sensitivity")
    ax3.set_title("Hierarchical Decomposition: Within vs Between Clusters")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Layer-wise Structural Emergence (for one model as example)
    ax4 = axes[1, 1]
    # Pick first successful model
    for model_name, results in all_results.items():
        if "error" not in results and results.get("context_length_results"):
            # Get data for longest context
            ctx_data = list(results["context_length_results"].values())[-1]
            layers_data = ctx_data.get("layers", {})

            layer_nums = []
            phi_vals = []

            for layer_key, layer_data in sorted(layers_data.items(), key=lambda x: int(x[0].split("_")[1])):
                if "cluster_sep" in layer_data and "phi_mean" in layer_data["cluster_sep"]:
                    layer_nums.append(int(layer_key.split("_")[1]))
                    phi_vals.append(layer_data["cluster_sep"]["phi_mean"])

            if layer_nums:
                ax4.plot(layer_nums, phi_vals, 'o-', label=model_name, linewidth=2)
            break

    ax4.set_xlabel("Layer")
    ax4.set_ylabel("Cluster Separation (Φ)")
    ax4.set_title("Layer-wise Structural Emergence (Longest Context)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = output_dir / "money_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "money_plot.pdf", bbox_inches='tight')
    print(f"\nMoney plot saved to: {plot_path}")

    plt.close()


def generate_position_sensitivity_heatmap(all_results: dict, output_dir: Path):
    """
    Generate heatmap showing position-wise sensitivity across context lengths.
    """
    for model_name, results in all_results.items():
        if "error" in results:
            continue

        ctx_results = results.get("context_length_results", {})
        if not ctx_results:
            continue

        # Collect position sensitivities
        ctx_lengths = sorted([int(k) for k in ctx_results.keys()])

        # Use final layer
        layers = results.get("layers_analyzed", [])
        if not layers:
            continue
        final_layer_key = f"layer_{layers[-1]}"

        # Build matrix (pad shorter sequences)
        max_len = max(ctx_lengths)
        sensitivity_matrix = np.full((len(ctx_lengths), max_len), np.nan)

        for i, ctx_len in enumerate(ctx_lengths):
            ctx_data = ctx_results.get(str(ctx_len), {})
            layer_data = ctx_data.get("layers", {}).get(final_layer_key, {})
            cluster_sep = layer_data.get("cluster_sep", {})
            sensitivities = cluster_sep.get("position_sensitivities", [])

            if sensitivities:
                sensitivity_matrix[i, :len(sensitivities)] = sensitivities

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(12, 6))

        im = ax.imshow(sensitivity_matrix, aspect='auto', cmap='RdBu_r',
                       interpolation='nearest', vmin=-np.nanmax(np.abs(sensitivity_matrix)),
                       vmax=np.nanmax(np.abs(sensitivity_matrix)))

        ax.set_xlabel("Token Position")
        ax.set_ylabel("Context Length")
        ax.set_yticks(range(len(ctx_lengths)))
        ax.set_yticklabels(ctx_lengths)
        ax.set_title(f"Position Sensitivity Heatmap: {model_name}")

        plt.colorbar(im, ax=ax, label="CSS (Cluster Separation)")

        plt.tight_layout()
        plt.savefig(output_dir / f"{model_name.replace('/', '_')}_heatmap.png", dpi=150)
        plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run hierarchical ICL experiment")
    parser.add_argument("--models", type=str, default="all",
                       help="Comma-separated model names or 'all'")
    parser.add_argument("--context-lengths", type=str, default=None,
                       help="Comma-separated context lengths (default: predefined)")
    parser.add_argument("--n-contexts", type=int, default=N_CONTEXTS_PER_LENGTH,
                       help="Number of contexts per length")
    parser.add_argument("--output-dir", type=str, default="results/hierarchical_experiment",
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select models
    if args.models == "all":
        models = MODELS_TO_TEST
    else:
        model_names = [m.strip() for m in args.models.split(",")]
        models = [m for m in MODELS_TO_TEST if m["name"] in model_names]

    # Context lengths
    if args.context_lengths:
        context_lengths = [int(x) for x in args.context_lengths.split(",")]
    else:
        context_lengths = CONTEXT_LENGTHS

    print("=" * 70)
    print("HIERARCHICAL GRAPH TRACING EXPERIMENT")
    print("Testing Stagewise ICL Learning Hypothesis")
    print("=" * 70)
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    print(f"Models: {[m['name'] for m in models]}")
    print(f"Context lengths: {context_lengths}")
    print(f"Contexts per length: {args.n_contexts}")
    print(f"Output: {output_dir}")

    # Create hierarchical graph
    print("\n" + "-" * 70)
    print("Creating Hierarchical Graph")
    print("-" * 70)

    graph_config = HierarchicalGraphConfig(
        num_superclusters=GRAPH_CONFIG["num_superclusters"],
        nodes_per_cluster=GRAPH_CONFIG["nodes_per_cluster"],
        p_intra_cluster=GRAPH_CONFIG["p_intra_cluster"],
        p_inter_cluster=GRAPH_CONFIG["p_inter_cluster"],
        walk_length=max(context_lengths),  # Max length needed
        seed=args.seed,
    )

    graph = HierarchicalGraph(graph_config)

    stats = graph.get_graph_statistics()
    print("\nGraph Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Save graph visualization
    try:
        graph.visualize(save_path=str(output_dir / "graph_structure.png"))
    except:
        pass

    # Run experiments for each model
    all_results = {}

    for model_config in models:
        results = run_single_model_experiment(
            model_config=model_config,
            graph=graph,
            context_lengths=context_lengths,
            n_contexts=args.n_contexts,
            output_dir=output_dir,
        )
        all_results[model_config["name"]] = results

    # Save combined results
    combined_path = output_dir / "all_results.json"
    with open(combined_path, "w") as f:
        json.dump(convert_numpy(all_results), f, indent=2)
    print(f"\nCombined results saved to: {combined_path}")

    # Generate visualizations
    print("\n" + "-" * 70)
    print("Generating Visualizations")
    print("-" * 70)

    generate_money_plot(all_results, output_dir)
    generate_position_sensitivity_heatmap(all_results, output_dir)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"\nResults directory: {output_dir}")
    print("Key outputs:")
    print(f"  - {output_dir}/money_plot.png")
    print(f"  - {output_dir}/all_results.json")
    print(f"  - Per-model results and heatmaps")

    return all_results


if __name__ == "__main__":
    main()
