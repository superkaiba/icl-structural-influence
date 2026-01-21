#!/usr/bin/env python3
"""
Deep Hierarchy Experiment: N-Level Stagewise Learning Analysis

This experiment extends the hierarchical graph experiments to support 3-4 levels
of hierarchy depth, with per-level CSS decomposition and trajectory tracking.

Key Research Questions:
1. Do models learn hierarchical structure in a stagewise manner?
2. Which hierarchy levels emerge first (coarse-to-fine vs fine-to-coarse)?
3. How does CSS vary across hierarchy levels?

Usage:
    # Quick test with small model
    python run_deep_hierarchy_experiment.py --model gpt2 --context-lengths 10,20,30 --n-contexts 5

    # Full experiment
    python run_deep_hierarchy_experiment.py --model meta-llama/Llama-3.1-8B \
        --context-lengths 10,20,30,50,75,100,150,200 --n-contexts 25

    # Custom hierarchy structure
    python run_deep_hierarchy_experiment.py --branching 3,2,3 --model gpt2
"""

import argparse
import json
import os
import gc
from datetime import datetime
from pathlib import Path
from typing import Optional
import warnings

import torch
import numpy as np
from tqdm import tqdm

# Local imports
from src.data import DeepHierarchicalGraph, DeepHierarchyConfig
from src.models import HookedLLM
from src.metrics import (
    LevelSpecificClusterSeparation,
    ContextSensitivityScore,
    compute_levelwise_phi_trajectory,
)
from src.visualization import (
    plot_multilevel_phi_trajectory,
    create_representation_evolution_gif,
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
    return obj


def collect_representations_at_context_length(
    model: HookedLLM,
    graph: DeepHierarchicalGraph,
    context_length: int,
    n_contexts: int,
    layer: int,
) -> tuple[list, dict, dict]:
    """
    Collect representations and labels at a specific context length.

    Returns:
        tuple of (all_losses, all_representations, level_labels)
        where level_labels is dict[level][context_idx] -> tensor of labels
    """
    all_losses = []
    all_representations = []
    level_labels = {level: [] for level in range(1, graph.num_levels)}

    print(f"  Collecting {n_contexts} samples at context length {context_length}...")

    for _ in tqdm(range(n_contexts), desc=f"  N={context_length}", leave=False):
        try:
            # Generate random walk
            prompt, nodes = graph.generate_random_walk(
                length=context_length,
                return_nodes=True
            )

            # Get losses and representations
            token_losses = model.compute_per_token_loss(prompt)
            _, cache = model.forward_with_cache(prompt, layers=[layer])

            residual = cache.get_residual_stream(layer)
            if residual is None:
                continue

            # Align representations with losses (next-token prediction)
            reps = residual.squeeze(0)[:-1].cpu()
            losses = token_losses.squeeze(0).cpu()

            # Get level labels for each token
            # Skip first token since we're doing next-token prediction
            for level in range(1, graph.num_levels):
                labels = torch.tensor([
                    graph.get_cluster_at_level(nodes[i], level)
                    for i in range(1, len(nodes))  # Skip first token
                ])
                level_labels[level].append(labels[:len(losses)])  # Align with losses

            all_losses.append(losses)
            all_representations.append(reps[:len(losses)])

        except Exception as e:
            warnings.warn(f"Error processing context: {e}")
            continue

    return all_losses, all_representations, level_labels


def run_experiment(args):
    """Main experiment runner."""

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 70)
    print("DEEP HIERARCHY EXPERIMENT")
    print("N-Level Stagewise Learning Analysis")
    print("=" * 70)
    print(f"\nTimestamp: {timestamp}")
    print(f"Output directory: {output_dir}")

    # Parse branching factors
    branching_factors = [int(x) for x in args.branching.split(',')]
    print(f"Branching factors: {branching_factors}")
    print(f"Total nodes: {np.prod(branching_factors)}")
    print(f"Hierarchy levels: {len(branching_factors) + 1} (including root)")

    # Create deep hierarchical graph
    print("\n" + "-" * 70)
    print("Creating Deep Hierarchical Graph")
    print("-" * 70)

    graph_config = DeepHierarchyConfig(
        branching_factors=branching_factors,
        p_same_level=args.p_same_level,
        p_decay_per_level=args.p_decay,
        walk_length=max(args.context_lengths),
        bridge_penalty_per_level=args.bridge_penalty,
        seed=args.seed,
    )

    graph = DeepHierarchicalGraph(graph_config)

    stats = graph.get_graph_statistics()
    print("\nGraph Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Visualize graph
    try:
        graph.visualize(
            save_path=str(output_dir / "graph_structure.png"),
            color_level=1
        )
        print(f"  Graph visualization saved")
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
        print(f"✓ Loaded successfully")
        print(f"  Layers: {model.num_layers}, Hidden size: {model.hidden_size}")
    except Exception as e:
        print(f"✗ ERROR loading model: {e}")
        return

    # Select layer to analyze
    layer = args.layer if args.layer is not None else model.num_layers - 1
    print(f"  Analyzing layer: {layer}")

    # Collect representations across context lengths
    print("\n" + "-" * 70)
    print("Collecting Representations Across Context Lengths")
    print("-" * 70)

    representations_by_ctx = {}
    level_labels_by_ctx = {}
    token_reps_by_ctx = {}  # For GIF animation

    for ctx_len in args.context_lengths:
        print(f"\nContext length: {ctx_len}")

        all_losses, all_reps, level_labels = collect_representations_at_context_length(
            model, graph, ctx_len, args.n_contexts, layer
        )

        if len(all_losses) < 2:
            warnings.warn(f"Not enough samples collected for context length {ctx_len}, skipping")
            continue

        representations_by_ctx[ctx_len] = all_reps
        level_labels_by_ctx[ctx_len] = level_labels

        # For GIF: collect average representation per token
        token_reps_by_ctx[ctx_len] = {}
        for i, token in enumerate(graph.node_to_token.values()):
            # Average representations across contexts where this token appears
            token_reps = []
            for reps in all_reps:
                if i < len(reps):
                    # Convert to float32 to handle BFloat16
                    token_reps.append(reps[i].float().numpy())
            if token_reps:
                token_reps_by_ctx[ctx_len][token] = np.mean(token_reps, axis=0)

        print(f"  Collected {len(all_losses)} samples")

    # Compute multilevel CSS decomposition (for longest context)
    print("\n" + "-" * 70)
    print("Computing Multilevel CSS Decomposition")
    print("-" * 70)

    longest_ctx = max(args.context_lengths)
    if longest_ctx in representations_by_ctx:
        # Re-collect losses and ensure alignment for CSS computation
        all_losses_longest = []
        all_reps_longest = []
        level_labels_longest = {level: [] for level in range(1, graph.num_levels)}

        print(f"  Re-collecting {args.n_contexts} aligned samples for CSS at context length {longest_ctx}...")
        for _ in range(args.n_contexts):
            try:
                prompt, nodes = graph.generate_random_walk(length=longest_ctx, return_nodes=True)
                losses = model.compute_per_token_loss(prompt)
                _, cache = model.forward_with_cache(prompt, layers=[layer])

                residual = cache.get_residual_stream(layer)
                if residual is None:
                    continue

                reps = residual.squeeze(0)[:-1].cpu()
                loss_tensor = losses.squeeze(0).cpu()

                # Ensure same length
                min_len = min(len(reps), len(loss_tensor))
                reps = reps[:min_len]
                loss_tensor = loss_tensor[:min_len]

                # Get aligned level labels
                # Labels correspond to nodes[1:] since we're doing next-token prediction
                for level in range(1, graph.num_levels):
                    # Create labels matching the representation length exactly
                    labels = []
                    for i in range(min_len):
                        # Node index is i+1 because we skip the first token
                        node_idx = min(i + 1, len(nodes) - 1)
                        labels.append(graph.get_cluster_at_level(nodes[node_idx], level))
                    level_labels_longest[level].append(torch.tensor(labels))

                all_losses_longest.append(loss_tensor)
                all_reps_longest.append(reps)
            except Exception as e:
                warnings.warn(f"Error in CSS sample collection: {e}")
                continue

        if len(all_losses_longest) >= 2:
            # Compute CSS at each level
            css = ContextSensitivityScore(LevelSpecificClusterSeparation(1))  # Dummy metric
            css_results = css.compute_multilevel_decomposition(
                all_losses_longest,
                all_reps_longest,
                level_labels_longest,
                num_levels=graph.num_levels - 1,
            )
        else:
            css_results = None
            warnings.warn("Not enough samples for CSS computation")

        if css_results:
            print(f"\nCSS Results (Context Length = {longest_ctx}):")
            for level in range(1, graph.num_levels):
                if f'phi_mean_level_{level}' in css_results:
                    phi_mean = css_results[f'phi_mean_level_{level}']
                    print(f"  Level {level}: Φ_mean = {phi_mean:.4f}")

            # Save CSS results
            css_path = output_dir / f"css_decomposition_ctx{longest_ctx}.json"
            with open(css_path, 'w') as f:
                json.dump(convert_numpy(css_results), f, indent=2)
            print(f"\n✓ CSS results saved to: {css_path}")

    # Compute Phi trajectories
    print("\n" + "-" * 70)
    print("Computing Phi Trajectories")
    print("-" * 70)

    trajectory = compute_levelwise_phi_trajectory(
        representations_by_ctx,
        level_labels_by_ctx,
        args.context_lengths,
        num_levels=graph.num_levels - 1,
    )

    print("\nPhi Trajectory Results:")
    for level in range(1, graph.num_levels):
        phi_traj = trajectory.get(f'phi_trajectory_level_{level}', [])
        if phi_traj and not all(np.isnan(phi_traj)):
            phi_initial = phi_traj[0] if not np.isnan(phi_traj[0]) else np.nan
            phi_final = phi_traj[-1] if not np.isnan(phi_traj[-1]) else np.nan
            print(f"  Level {level}: Φ_initial = {phi_initial:.4f}, Φ_final = {phi_final:.4f}")

    # Save trajectory data
    trajectory_path = output_dir / "phi_trajectories.json"
    with open(trajectory_path, 'w') as f:
        json.dump(convert_numpy(trajectory), f, indent=2)
    print(f"\n✓ Trajectory data saved to: {trajectory_path}")

    # Generate visualizations
    print("\n" + "-" * 70)
    print("Generating Visualizations")
    print("-" * 70)

    # Multi-line Phi plot
    print("  Creating multi-level Phi trajectory plot...")
    plot_multilevel_phi_trajectory(
        trajectory,
        output_path=output_dir / "phi_evolution.png",
        title=f"Hierarchical Structure Emergence: {args.model}",
    )

    # GIF animation
    if not args.skip_gif and token_reps_by_ctx:
        print("  Creating representation evolution GIF...")
        try:
            create_representation_evolution_gif(
                token_reps_by_ctx,
                graph,
                args.context_lengths,
                output_path=output_dir / "representation_evolution.gif",
                method=args.gif_method,
                fps=args.gif_fps,
                tight_bounds=True,
                color_by_level=None if args.hierarchical_colors else 1,
                hierarchical_colors=args.hierarchical_colors,
                colormap=args.gif_colormap,
            )
        except Exception as e:
            warnings.warn(f"Could not create GIF: {e}")

    # Cleanup
    del model
    clear_gpu_memory()

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"\nResults directory: {output_dir}")
    print("Key outputs:")
    print(f"  - {output_dir}/phi_evolution.png")
    print(f"  - {output_dir}/phi_trajectories.json")
    if not args.skip_gif:
        print(f"  - {output_dir}/representation_evolution.gif")


def main():
    parser = argparse.ArgumentParser(
        description="Run deep hierarchy experiment with N-level structure"
    )

    # Graph configuration
    parser.add_argument(
        "--branching",
        type=str,
        default="2,2,4",
        help="Branching factors (comma-separated), e.g., '2,2,4' = 16 nodes",
    )
    parser.add_argument("--p-same-level", type=float, default=0.9,
                       help="Edge probability within same leaf cluster")
    parser.add_argument("--p-decay", type=float, default=0.3,
                       help="Decay factor per hierarchy level")
    parser.add_argument("--bridge-penalty", type=float, default=0.5,
                       help="Penalty for crossing hierarchy levels in random walk")

    # Model configuration
    parser.add_argument("--model", type=str, default="gpt2",
                       help="HuggingFace model name")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                       choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--layer", type=int, default=None,
                       help="Layer to analyze (default: final layer)")

    # Experiment configuration
    parser.add_argument(
        "--context-lengths",
        type=str,
        default="10,20,30,50,75,100",
        help="Context lengths to test (comma-separated)",
    )
    parser.add_argument("--n-contexts", type=int, default=25,
                       help="Number of contexts per length")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    # Visualization configuration
    parser.add_argument("--skip-gif", action="store_true",
                       help="Skip GIF generation")
    parser.add_argument("--gif-method", type=str, default="mds",
                       choices=["mds", "pca", "umap", "tsne"],
                       help="Dimensionality reduction method for GIF")
    parser.add_argument("--gif-fps", type=int, default=2,
                       help="Frames per second for GIF")
    parser.add_argument("--hierarchical-colors", action="store_true",
                       help="Use hierarchical color encoding in GIF (colors reflect tree similarity)")
    parser.add_argument("--gif-colormap", type=str, default="twilight",
                       choices=["twilight", "hsv", "viridis", "plasma", "twilight_shifted"],
                       help="Colormap for hierarchical coloring (default: twilight)")

    # Output configuration
    parser.add_argument("--output-dir", type=str, default="results/deep_hierarchy",
                       help="Output directory")

    args = parser.parse_args()

    # Parse context lengths
    args.context_lengths = sorted([int(x) for x in args.context_lengths.split(',')])

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Run experiment
    run_experiment(args)


if __name__ == "__main__":
    main()
