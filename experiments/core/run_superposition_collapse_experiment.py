#!/usr/bin/env python3
"""
Hypothesis Superposition & Collapse Experiment

This experiment tests whether LLMs maintain two competing hypotheses in
superposition until a disambiguating token forces representation collapse
to one interpretation.

Core idea:
- Ambiguous tokens consistent with BOTH H1 and H2 interpretations
- Representations should sit "between" the two interpretation centroids
- Disambiguating token reveals the true interpretation
- Representations should "snap" to the correct interpretation centroid

Usage:
    # Quick test with small model
    python run_superposition_collapse_experiment.py --model gpt2 --n-trials 10

    # Full experiment with larger model
    python run_superposition_collapse_experiment.py --model meta-llama/Llama-3.1-8B \
        --n-trials 100 --context-length 100

    # Test specific layers
    python run_superposition_collapse_experiment.py --model gpt2 --layers 0,5,11

References:
    - Park et al. (2024) arXiv:2501.00070 "ICLR: In-Context Learning of Representations"
"""

import argparse
import json
import os
import gc
from datetime import datetime
from pathlib import Path
from typing import Optional
import warnings
from dataclasses import asdict

import torch
import numpy as np
from tqdm import tqdm

# Local imports
from src.data.dual_interpretation_graph import (
    DualInterpretationGraph,
    DualInterpretationConfig,
)
from src.models import HookedLLM
from src.metrics.superposition_metrics import (
    compute_hypothesis_centroid,
    analyze_position_trajectory,
    analyze_collapse,
    aggregate_trial_results,
    SuperpositionResult,
    CollapseAnalysis,
)


def setup_dtype(dtype_str: str) -> torch.dtype:
    """Convert string to torch dtype."""
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }[dtype_str]


def clear_gpu_memory():
    """Clear GPU memory between operations."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def convert_numpy(obj):
    """Convert numpy arrays and types to JSON-serializable formats."""
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
    elif isinstance(obj, SuperpositionResult):
        d = {
            "position": obj.position,
            "dist_H1": obj.dist_H1,
            "dist_H2": obj.dist_H2,
            "superposition_score": obj.superposition_score,
            "hypothesis_ratio": obj.hypothesis_ratio,
            "velocity": obj.velocity,
            "H1_cluster": obj.H1_cluster,
            "H2_cluster": obj.H2_cluster,
        }
        return convert_numpy(d)
    elif isinstance(obj, CollapseAnalysis):
        return convert_numpy(asdict(obj))
    return obj


def run_single_trial(
    model: HookedLLM,
    graph: DualInterpretationGraph,
    context_length: int,
    disambig_position: Optional[int],
    true_hypothesis: str,
    layer: int,
) -> dict:
    """
    Run a single trial of the superposition experiment.

    Returns:
        Dict with trial results including position metrics and collapse analysis
    """
    # Generate ambiguous walk
    prompt, nodes, metadata = graph.generate_ambiguous_walk(
        length=context_length,
        disambig_position=disambig_position,
        true_hypothesis=true_hypothesis,
        return_nodes=True,
    )

    # Get representations
    try:
        _, cache = model.forward_with_cache(prompt, layers=[layer])
        residual = cache.get_residual_stream(layer)
        if residual is None:
            return {"error": "No residual stream cached"}

        representations = residual.squeeze(0).cpu()

        # Get cluster labels
        H1_labels = torch.tensor(metadata["H1_clusters"])
        H2_labels = torch.tensor(metadata["H2_clusters"])

        # Compute centroids for each hypothesis
        # Use all representations to compute centroids
        H1_centroid = compute_hypothesis_centroid(representations, H1_labels)
        H2_centroid = compute_hypothesis_centroid(representations, H2_labels)

        # Analyze trajectory
        position_results = analyze_position_trajectory(
            representations,
            H1_labels,
            H2_labels,
            H1_centroid,
            H2_centroid,
        )

        # Analyze collapse (if disambiguation occurred)
        collapse_analysis = None
        if disambig_position is not None and metadata.get("disambig_achieved", False):
            collapse_analysis = analyze_collapse(
                position_results,
                disambig_position,
                true_hypothesis,
            )

        return {
            "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            "nodes": nodes,
            "metadata": metadata,
            "position_results": position_results,
            "collapse_analysis": collapse_analysis,
            "H1_consistency": graph.compute_interpretation_consistency(nodes, "H1"),
            "H2_consistency": graph.compute_interpretation_consistency(nodes, "H2"),
            "H1_centroid_norm": float(torch.norm(H1_centroid).item()),
            "H2_centroid_norm": float(torch.norm(H2_centroid).item()),
            "centroid_distance": float(torch.norm(H2_centroid - H1_centroid).item()),
        }

    except Exception as e:
        return {"error": str(e)}


def run_condition(
    model: HookedLLM,
    graph: DualInterpretationGraph,
    context_length: int,
    disambig_position: Optional[int],
    n_trials: int,
    layer: int,
    condition_name: str,
) -> dict:
    """
    Run all trials for a single experimental condition.
    """
    print(f"\n  Running condition: {condition_name}")
    print(f"    Context length: {context_length}, Disambig at: {disambig_position}")
    print(f"    Layer: {layer}, N trials: {n_trials}")

    trial_results = []

    for trial_idx in tqdm(range(n_trials), desc=f"    {condition_name}", leave=False):
        # Alternate true hypothesis
        true_hyp = "H1" if trial_idx % 2 == 0 else "H2"

        result = run_single_trial(
            model, graph, context_length, disambig_position, true_hyp, layer
        )

        if "error" not in result:
            trial_results.append(result)
        else:
            warnings.warn(f"Trial {trial_idx} failed: {result['error']}")

    # Aggregate results
    aggregated = aggregate_trial_results(trial_results)

    return {
        "condition_name": condition_name,
        "context_length": context_length,
        "disambig_position": disambig_position,
        "layer": layer,
        "n_trials_completed": len(trial_results),
        "aggregated": aggregated,
        "trial_results": trial_results,
    }


def run_experiment(args):
    """Main experiment runner."""

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 70)
    print("HYPOTHESIS SUPERPOSITION & COLLAPSE EXPERIMENT")
    print("=" * 70)
    print(f"\nTimestamp: {timestamp}")
    print(f"Output directory: {output_dir}")

    # Create dual interpretation graph
    print("\n" + "-" * 70)
    print("Creating Dual Interpretation Graph")
    print("-" * 70)

    graph_config = DualInterpretationConfig(
        vocab_size=args.vocab_size,
        clusters_per_interpretation=args.n_clusters,
        p_intra_cluster=args.p_intra,
        p_inter_cluster=args.p_inter,
        seed=args.seed,
    )

    graph = DualInterpretationGraph(graph_config)

    stats = graph.get_graph_statistics()
    print("\nGraph Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Save graph visualization
    try:
        graph.visualize(
            save_path=str(output_dir / "graph_structure.png"),
            hypothesis="both"
        )
        print(f"\n  Graph visualization saved")
    except Exception as e:
        warnings.warn(f"Could not visualize graph: {e}")

    # Load model
    print("\n" + "-" * 70)
    print(f"Loading Model: {args.model}")
    print("-" * 70)

    try:
        model = HookedLLM.from_pretrained(
            args.model,
            device="auto",
            dtype=setup_dtype(args.dtype),
        )
        print(f"  Loaded successfully")
        print(f"  Layers: {model.num_layers}, Hidden size: {model.hidden_size}")
    except Exception as e:
        print(f"  ERROR loading model: {e}")
        return

    # Determine layers to analyze
    if args.layers:
        layers = [int(x) for x in args.layers.split(',')]
    else:
        # Sample layers across depth
        n_layers = model.num_layers
        layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
        layers = list(set(layers))  # Remove duplicates
        layers.sort()

    print(f"  Layers to analyze: {layers}")

    # Define experimental conditions
    context_length = args.context_length
    disambig_positions = [
        int(context_length * 0.25),   # Early reveal
        int(context_length * 0.50),   # Mid reveal
        int(context_length * 0.75),   # Late reveal
    ]

    conditions = [
        ("early_reveal", disambig_positions[0]),
        ("mid_reveal", disambig_positions[1]),
        ("late_reveal", disambig_positions[2]),
        ("no_reveal", None),  # Control - no disambiguation
    ]

    print("\n" + "-" * 70)
    print("Experimental Conditions")
    print("-" * 70)
    print(f"  Context length: {context_length}")
    print(f"  Conditions: {[c[0] for c in conditions]}")
    print(f"  Disambig positions: {[c[1] for c in conditions]}")
    print(f"  Trials per condition: {args.n_trials}")

    # Run experiment
    print("\n" + "-" * 70)
    print("Running Experiment")
    print("-" * 70)

    all_results = {
        "config": {
            "model": args.model,
            "context_length": context_length,
            "n_trials": args.n_trials,
            "layers": layers,
            "vocab_size": args.vocab_size,
            "n_clusters": args.n_clusters,
            "seed": args.seed,
            "timestamp": timestamp,
        },
        "graph_stats": stats,
        "conditions": {},
    }

    for layer in layers:
        print(f"\nLayer {layer}:")
        all_results["conditions"][f"layer_{layer}"] = {}

        for condition_name, disambig_pos in conditions:
            condition_result = run_condition(
                model=model,
                graph=graph,
                context_length=context_length,
                disambig_position=disambig_pos,
                n_trials=args.n_trials,
                layer=layer,
                condition_name=condition_name,
            )

            # Print summary
            agg = condition_result["aggregated"]
            print(f"\n    {condition_name}:")
            print(f"      Collapse distance: {agg['collapse_distance_mean']:.4f} +/- {agg['collapse_distance_std']:.4f}")
            print(f"      Velocity spike: {agg['velocity_spike_mean']:.2f}x")
            print(f"      Collapse accuracy: {agg['collapse_accuracy']:.2%}")
            print(f"      Pre-superposition: {agg['pre_superposition_mean']:.4f}")
            print(f"      Post-superposition: {agg['post_superposition_mean']:.4f}")

            # Store results (without full trial data for space)
            all_results["conditions"][f"layer_{layer}"][condition_name] = {
                "config": {
                    "disambig_position": disambig_pos,
                    "n_trials": condition_result["n_trials_completed"],
                },
                "aggregated": agg,
            }

            # Save detailed trial results to separate file
            if args.save_trials:
                trial_path = output_dir / f"trials_layer{layer}_{condition_name}.json"
                with open(trial_path, 'w') as f:
                    # Exclude full representations for space
                    trial_data = []
                    for tr in condition_result["trial_results"]:
                        trial_entry = {
                            k: convert_numpy(v)
                            for k, v in tr.items()
                            if k != "position_results"  # These are large
                        }
                        # Include position metrics summary
                        if "position_results" in tr:
                            trial_entry["position_summary"] = [
                                convert_numpy(r) for r in tr["position_results"]
                            ]
                        trial_data.append(trial_entry)
                    json.dump(trial_data, f, indent=2)

    # Save main results
    results_path = output_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(convert_numpy(all_results), f, indent=2)
    print(f"\n  Results saved to: {results_path}")

    # Cleanup
    del model
    clear_gpu_memory()

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"\nResults directory: {output_dir}")
    print("Key outputs:")
    print(f"  - {output_dir}/results.json")
    print(f"  - {output_dir}/graph_structure.png")
    if args.save_trials:
        print(f"  - {output_dir}/trials_layer*_*.json")

    # Print summary table
    print("\n" + "-" * 70)
    print("SUMMARY: Collapse Distance by Layer and Condition")
    print("-" * 70)
    print(f"{'Layer':<10} {'Early':<12} {'Mid':<12} {'Late':<12} {'Control':<12}")
    print("-" * 70)

    for layer in layers:
        row = f"Layer {layer:<4}"
        for cond in ["early_reveal", "mid_reveal", "late_reveal", "no_reveal"]:
            if f"layer_{layer}" in all_results["conditions"]:
                agg = all_results["conditions"][f"layer_{layer}"].get(cond, {}).get("aggregated", {})
                val = agg.get("collapse_distance_mean", 0)
                row += f" {val:<12.4f}"
            else:
                row += f" {'N/A':<12}"
        print(row)


def main():
    parser = argparse.ArgumentParser(
        description="Run hypothesis superposition and collapse experiment"
    )

    # Model configuration
    parser.add_argument("--model", type=str, default="gpt2",
                       help="HuggingFace model name")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                       choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--layers", type=str, default=None,
                       help="Comma-separated layer indices (default: sample across depth)")

    # Graph configuration
    parser.add_argument("--vocab-size", type=int, default=15,
                       help="Number of tokens in vocabulary")
    parser.add_argument("--n-clusters", type=int, default=3,
                       help="Number of clusters per interpretation")
    parser.add_argument("--p-intra", type=float, default=0.8,
                       help="Intra-cluster edge probability")
    parser.add_argument("--p-inter", type=float, default=0.15,
                       help="Inter-cluster edge probability")

    # Experiment configuration
    parser.add_argument("--context-length", type=int, default=100,
                       help="Total context length")
    parser.add_argument("--n-trials", type=int, default=100,
                       help="Number of trials per condition")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    # Output configuration
    parser.add_argument("--output-dir", type=str, default="results/superposition_collapse",
                       help="Output directory")
    parser.add_argument("--save-trials", action="store_true",
                       help="Save detailed trial results")

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Run experiment
    run_experiment(args)


if __name__ == "__main__":
    main()
