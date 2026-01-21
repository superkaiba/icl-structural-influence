#!/usr/bin/env python3
"""
Main experiment runner for Hierarchical Graph Tracing with Context Sensitivity.

This script orchestrates the full experiment pipeline:
1. Generate hierarchical graph and random walks
2. Load LLaMA-3-8B with hook infrastructure
3. Extract representations and compute per-token losses
4. Calculate structural metrics (Φ) and Context Sensitivity Scores
5. Analyze hierarchical learning patterns

Note: We use Context Sensitivity Scores (CSS) instead of Bayesian Influence
Functions (BIF) because true BIF requires weight-space sampling via SGLD,
which is infeasible for large models. CSS measures correlational sensitivity
across different input contexts with frozen weights.

Usage:
    python run_experiment.py --model meta-llama/Meta-Llama-3-8B --n-contexts 100

    # Quick test with smaller model
    python run_experiment.py --model gpt2 --n-contexts 10 --layers 0,5,11
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import torch
import numpy as np

from src.data import HierarchicalGraph, HierarchicalGraphConfig
from src.models import HookedLLM
from src.metrics import (
    DirichletEnergy,
    ClusterSeparation,
    RepresentationCoherence,
    ContextSensitivityScore,
    ContextSensitivityExperiment,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Context Sensitivity experiment on hierarchical graph tracing"
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
        help="HuggingFace model identifier"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, cpu)"
    )

    # Graph arguments
    parser.add_argument(
        "--num-superclusters",
        type=int,
        default=3,
        help="Number of super-clusters in hierarchical graph"
    )
    parser.add_argument(
        "--nodes-per-cluster",
        type=int,
        default=5,
        help="Number of nodes per super-cluster"
    )
    parser.add_argument(
        "--walk-length",
        type=int,
        default=50,
        help="Length of random walks"
    )

    # Experiment arguments
    parser.add_argument(
        "--n-contexts",
        type=int,
        default=100,
        help="Number of random walk contexts for CSS estimation"
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layer indices to analyze (default: all)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for saving results"
    )
    parser.add_argument(
        "--save-activations",
        action="store_true",
        help="Save raw activations (warning: large files)"
    )

    return parser.parse_args()


def setup_dtype(dtype_str: str) -> torch.dtype:
    """Convert string to torch dtype."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map[dtype_str]


def run_experiment(args):
    """Main experiment function."""
    print("=" * 70)
    print("Context Sensitivity Experiment: Hierarchical Graph Tracing")
    print("=" * 70)
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================================================
    # Step 1: Create Hierarchical Graph
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Step 1: Creating Hierarchical Graph")
    print("-" * 70)

    graph_config = HierarchicalGraphConfig(
        num_superclusters=args.num_superclusters,
        nodes_per_cluster=args.nodes_per_cluster,
        walk_length=args.walk_length,
        seed=args.seed,
    )

    graph = HierarchicalGraph(graph_config)

    stats = graph.get_graph_statistics()
    print("\nGraph Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Save graph visualization
    try:
        fig, ax = graph.visualize(save_path=str(output_dir / "graph_structure.png"))
        print(f"\nGraph visualization saved to {output_dir / 'graph_structure.png'}")
        import matplotlib.pyplot as plt
        plt.close(fig)
    except ImportError:
        print("\nNote: matplotlib not available for visualization")

    # Generate sample walks for inspection
    print("\nSample Random Walks:")
    for i in range(3):
        prompt, nodes = graph.generate_random_walk(length=15, return_nodes=True)
        clusters = [graph.get_cluster(n) for n in nodes]
        print(f"  Walk {i+1}: {prompt[:60]}...")
        print(f"          Clusters: {clusters}")

    # ==========================================================================
    # Step 2: Load Model
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Step 2: Loading Model with Hook Infrastructure")
    print("-" * 70)

    dtype = setup_dtype(args.dtype)

    print(f"\nLoading {args.model}...")
    model = HookedLLM.from_pretrained(
        args.model,
        device=args.device,
        dtype=dtype,
    )

    print(f"  Model type: {model._model_type}")
    print(f"  Num layers: {model.num_layers}")
    print(f"  Hidden size: {model.hidden_size}")
    print(f"  Device: {model.device}")

    # Parse layers to analyze
    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    else:
        # Default: sample across depth
        n_layers = model.num_layers
        layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
        layers = sorted(set(layers))  # Remove duplicates

    print(f"\nLayers to analyze: {layers}")

    # ==========================================================================
    # Step 3: Run Context Sensitivity Analysis
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Step 3: Running Context Sensitivity Analysis")
    print("-" * 70)

    experiment = ContextSensitivityExperiment(
        model=model,
        graph=graph,
        layers=layers,
    )

    print(f"\nRunning batch analysis with {args.n_contexts} contexts...")

    # Collect data
    all_losses = []
    all_representations = {layer: [] for layer in layers}
    all_clusters = []
    all_prompts = []

    for ctx_idx in range(args.n_contexts):
        if (ctx_idx + 1) % 10 == 0:
            print(f"  Processing context {ctx_idx + 1}/{args.n_contexts}")

        prompt, nodes = graph.generate_random_walk(return_nodes=True)
        clusters = torch.tensor([graph.get_cluster(n) for n in nodes])

        # Get losses
        token_losses = model.compute_per_token_loss(prompt)

        # Get representations
        _, cache = model.forward_with_cache(prompt, layers=layers)

        all_losses.append(token_losses.squeeze(0).cpu())
        all_clusters.append(clusters[1:])  # Align with loss (next-token prediction)
        all_prompts.append(prompt)

        for layer in layers:
            residual = cache.get_residual_stream(layer)
            if residual is not None:
                # Exclude last token to align with losses
                all_representations[layer].append(residual.squeeze(0)[:-1].cpu())

    # ==========================================================================
    # Step 4: Compute Context Sensitivity Scores
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Step 4: Computing Context Sensitivity Scores")
    print("-" * 70)

    results = {
        "metadata": {
            "model": args.model,
            "n_contexts": args.n_contexts,
            "walk_length": args.walk_length,
            "num_superclusters": args.num_superclusters,
            "nodes_per_cluster": args.nodes_per_cluster,
            "layers": layers,
            "seed": args.seed,
            "timestamp": datetime.now().isoformat(),
        },
        "graph_stats": stats,
        "layer_results": {},
    }

    metrics = [
        DirichletEnergy(),
        ClusterSeparation(),
        RepresentationCoherence(),
    ]

    for layer in layers:
        print(f"\nLayer {layer}:")
        layer_results = {}

        for metric in metrics:
            css = ContextSensitivityScore(metric)

            # Compute batch sensitivity
            try:
                sensitivity_result = css.compute_batch(
                    all_losses,
                    all_representations[layer],
                    all_clusters,
                )

                print(f"  {metric.name}:")
                print(f"    Φ mean: {sensitivity_result['phi_mean']:.4f}")
                print(f"    Φ std: {sensitivity_result['phi_std']:.4f}")

                # Find most sensitive positions
                sensitivities = sensitivity_result['position_sensitivities']
                top_k = 5
                top_positions = np.argsort(np.abs(sensitivities))[-top_k:][::-1]
                print(f"    Top sensitive positions: {top_positions.tolist()}")

                layer_results[metric.name] = {
                    "phi_mean": float(sensitivity_result['phi_mean']),
                    "phi_std": float(sensitivity_result['phi_std']),
                    "position_sensitivities": sensitivities.tolist(),
                }

            except Exception as e:
                print(f"  {metric.name}: Error - {e}")
                layer_results[metric.name] = {"error": str(e)}

        # Hierarchical decomposition
        print(f"\n  Hierarchical Decomposition:")
        try:
            css_cluster = ContextSensitivityScore(ClusterSeparation())
            decomp = css_cluster.compute_hierarchical_decomposition(
                all_losses,
                all_representations[layer],
                all_clusters,
            )

            within_mean = float(decomp['within_phi_values'].mean())
            between_mean = float(decomp['between_phi_values'].mean())

            print(f"    Within-cluster Φ mean: {within_mean:.4f}")
            print(f"    Between-cluster Φ mean: {between_mean:.4f}")

            # Compare sensitivity patterns
            within_sens = decomp['within_cluster_sensitivities']
            between_sens = decomp['between_cluster_sensitivities']

            correlation = np.corrcoef(within_sens, between_sens)[0, 1]
            print(f"    Correlation (within vs between): {correlation:.4f}")

            layer_results["hierarchical_decomposition"] = {
                "within_phi_mean": within_mean,
                "between_phi_mean": between_mean,
                "within_sensitivities": within_sens.tolist(),
                "between_sensitivities": between_sens.tolist(),
                "correlation": float(correlation) if not np.isnan(correlation) else None,
            }

        except Exception as e:
            print(f"    Error: {e}")
            layer_results["hierarchical_decomposition"] = {"error": str(e)}

        results["layer_results"][f"layer_{layer}"] = layer_results

    # ==========================================================================
    # Step 5: Save Results
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Step 5: Saving Results")
    print("-" * 70)

    results_path = output_dir / "experiment_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Save sample prompts for inspection
    prompts_path = output_dir / "sample_prompts.json"
    with open(prompts_path, "w") as f:
        json.dump(all_prompts[:10], f, indent=2)
    print(f"Sample prompts saved to {prompts_path}")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("Experiment Complete!")
    print("=" * 70)

    print("\nKey Findings:")
    print("-" * 40)

    # Analyze layer-wise trends
    for metric in metrics:
        print(f"\n{metric.name}:")
        phi_means = []
        for layer in layers:
            layer_key = f"layer_{layer}"
            if layer_key in results["layer_results"]:
                metric_result = results["layer_results"][layer_key].get(metric.name, {})
                if "phi_mean" in metric_result:
                    phi_means.append((layer, metric_result["phi_mean"]))

        if phi_means:
            for layer, phi in phi_means:
                print(f"  Layer {layer:2d}: Φ = {phi:.4f}")

    print(f"\nOutput directory: {output_dir}")

    return results


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
