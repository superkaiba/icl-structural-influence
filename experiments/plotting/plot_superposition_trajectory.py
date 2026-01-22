#!/usr/bin/env python3
"""
Visualization Scripts for Hypothesis Superposition & Collapse Experiment

This module provides visualization functions for:
1. 2D trajectory plots showing representation movement through context
2. Distance-to-centroids over position plots
3. Superposition score evolution
4. Layer-wise comparison grids
5. Summary comparison across conditions

Usage:
    # Plot from experiment results
    python plot_superposition_trajectory.py --results-dir results/superposition_collapse

    # Generate from a single trial's data
    python plot_superposition_trajectory.py --trial-file results/superposition_collapse/trials_layer5_mid_reveal.json
"""

import argparse
import json
from pathlib import Path
from typing import Optional
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA

# Try to import UMAP for alternative projections
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


def plot_representation_trajectory(
    position_metrics: list[dict],
    H1_centroid: np.ndarray,
    H2_centroid: np.ndarray,
    disambig_pos: Optional[int] = None,
    true_hypothesis: str = "H1",
    method: str = "pca",
    output_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """
    Create 2D trajectory plot showing sudden shift at disambiguation.

    Projects representations to 2D and plots the trajectory through
    context positions, highlighting the disambiguation point.

    Args:
        position_metrics: List of dicts with 'representation' key (and position info)
        H1_centroid: Centroid for hypothesis H1
        H2_centroid: Centroid for hypothesis H2
        disambig_pos: Position of disambiguating token
        true_hypothesis: The true interpretation ("H1" or "H2")
        method: Projection method ("pca" or "umap")
        output_path: If provided, save figure to this path
        title: Plot title
    """
    # Extract representations
    all_reps = np.stack([m['representation'] for m in position_metrics if 'representation' in m])

    if len(all_reps) == 0:
        warnings.warn("No representations found in position_metrics")
        return None

    # Stack with centroids for joint projection
    combined = np.vstack([all_reps, H1_centroid.reshape(1, -1), H2_centroid.reshape(1, -1)])

    # Project to 2D
    if method == "umap" and HAS_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=42)
        projected = reducer.fit_transform(combined)
    else:
        pca = PCA(n_components=2, random_state=42)
        projected = pca.fit_transform(combined)

    traj_2d = projected[:-2]  # Token representations
    H1_2d = projected[-2]      # H1 centroid
    H2_2d = projected[-1]      # H2 centroid

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot trajectory with color gradient (position)
    positions = np.arange(len(traj_2d))
    scatter = ax.scatter(
        traj_2d[:, 0], traj_2d[:, 1],
        c=positions, cmap='viridis', s=50, alpha=0.8,
        edgecolors='white', linewidths=0.5,
        zorder=2
    )

    # Connect points with lines
    ax.plot(traj_2d[:, 0], traj_2d[:, 1], 'k-', alpha=0.3, linewidth=1, zorder=1)

    # Mark centroids
    ax.scatter(*H1_2d, marker='*', s=300, c='red', edgecolors='darkred',
               linewidths=2, label='H1 centroid', zorder=3)
    ax.scatter(*H2_2d, marker='*', s=300, c='blue', edgecolors='darkblue',
               linewidths=2, label='H2 centroid', zorder=3)

    # Mark disambiguation point
    if disambig_pos is not None and disambig_pos < len(traj_2d):
        ax.scatter(
            traj_2d[disambig_pos, 0], traj_2d[disambig_pos, 1],
            marker='X', s=200, c='orange', edgecolors='darkorange',
            linewidths=2, label=f'Disambig (pos {disambig_pos})',
            zorder=4
        )

        # Add arrow showing direction of movement at disambiguation
        if disambig_pos > 0 and disambig_pos < len(traj_2d) - 1:
            dx = traj_2d[disambig_pos + 1, 0] - traj_2d[disambig_pos - 1, 0]
            dy = traj_2d[disambig_pos + 1, 1] - traj_2d[disambig_pos - 1, 1]
            ax.annotate(
                '', xy=(traj_2d[disambig_pos, 0] + dx*0.3, traj_2d[disambig_pos, 1] + dy*0.3),
                xytext=(traj_2d[disambig_pos, 0], traj_2d[disambig_pos, 1]),
                arrowprops=dict(arrowstyle='->', color='orange', lw=2),
                zorder=5
            )

    # Mark start and end
    ax.scatter(traj_2d[0, 0], traj_2d[0, 1], marker='o', s=150, c='green',
               edgecolors='darkgreen', linewidths=2, label='Start', zorder=4)
    ax.scatter(traj_2d[-1, 0], traj_2d[-1, 1], marker='s', s=150, c='purple',
               edgecolors='darkpurple', linewidths=2, label='End', zorder=4)

    # Add midpoint for reference
    midpoint = (H1_2d + H2_2d) / 2
    ax.scatter(*midpoint, marker='+', s=100, c='gray', linewidths=2,
               label='Midpoint', zorder=3)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Position in context')

    # Labels and title
    ax.set_xlabel(f'{method.upper()} Component 1')
    ax.set_ylabel(f'{method.upper()} Component 2')

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Representation Trajectory (True: {true_hypothesis})')

    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved trajectory plot to: {output_path}")

    return fig


def plot_distance_over_position(
    position_metrics: list[dict],
    disambig_pos: Optional[int] = None,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """
    Plot distance to each centroid over context position.

    Look for sudden shift at disambiguation point where distances diverge.

    Args:
        position_metrics: List of dicts with 'dist_H1', 'dist_H2' keys
        disambig_pos: Position of disambiguating token
        output_path: If provided, save figure to this path
        title: Plot title
    """
    positions = [m.get('position', i) for i, m in enumerate(position_metrics)]
    dist_H1 = [m['dist_H1'] for m in position_metrics]
    dist_H2 = [m['dist_H2'] for m in position_metrics]

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(positions, dist_H1, 'r-', linewidth=2, label='Distance to H1', alpha=0.8)
    ax.plot(positions, dist_H2, 'b-', linewidth=2, label='Distance to H2', alpha=0.8)

    # Mark disambiguation
    if disambig_pos is not None:
        ax.axvline(disambig_pos, color='orange', linestyle='--', linewidth=2,
                   label=f'Disambiguation (pos {disambig_pos})')

        # Shade regions
        ax.axvspan(0, disambig_pos, alpha=0.1, color='gray', label='Ambiguous region')
        ax.axvspan(disambig_pos, len(positions), alpha=0.1, color='green', label='Post-disambig')

    ax.set_xlabel('Position in Context')
    ax.set_ylabel('Distance to Centroid')

    if title:
        ax.set_title(title)
    else:
        ax.set_title('Centroid Distances Over Context (Look for Sudden Shift)')

    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved distance plot to: {output_path}")

    return fig


def plot_superposition_score_over_position(
    position_metrics: list[dict],
    disambig_pos: Optional[int] = None,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """
    Plot superposition score over context position.

    Low score = in superposition (between both hypotheses)
    High score = committed to one hypothesis

    Args:
        position_metrics: List of dicts with 'superposition_score' key
        disambig_pos: Position of disambiguating token
        output_path: If provided, save figure to this path
        title: Plot title
    """
    positions = [m.get('position', i) for i, m in enumerate(position_metrics)]
    superposition = [m['superposition_score'] for m in position_metrics]

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(positions, superposition, 'purple', linewidth=2, alpha=0.8)
    ax.fill_between(positions, 0, superposition, alpha=0.3, color='purple')

    # Mark disambiguation
    if disambig_pos is not None:
        ax.axvline(disambig_pos, color='orange', linestyle='--', linewidth=2,
                   label=f'Disambiguation (pos {disambig_pos})')

    ax.set_xlabel('Position in Context')
    ax.set_ylabel('Superposition Score (lower = more superposition)')

    if title:
        ax.set_title(title)
    else:
        ax.set_title('Superposition Score Over Context')

    if disambig_pos is not None:
        ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved superposition plot to: {output_path}")

    return fig


def plot_velocity_over_position(
    position_metrics: list[dict],
    disambig_pos: Optional[int] = None,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """
    Plot representation velocity (rate of change) over position.

    Look for velocity spike at disambiguation point.

    Args:
        position_metrics: List of dicts with 'velocity' key
        disambig_pos: Position of disambiguating token
        output_path: If provided, save figure to this path
        title: Plot title
    """
    positions = [m.get('position', i) for i, m in enumerate(position_metrics)]
    velocities = [m['velocity'] for m in position_metrics]

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(positions, velocities, color='teal', alpha=0.7, edgecolor='darkcyan')

    # Mark disambiguation
    if disambig_pos is not None and disambig_pos < len(velocities):
        ax.bar(disambig_pos, velocities[disambig_pos], color='orange',
               edgecolor='darkorange', linewidth=2, label=f'Disambig (pos {disambig_pos})')

    ax.set_xlabel('Position in Context')
    ax.set_ylabel('Velocity (||rep[t] - rep[t-1]||)')

    if title:
        ax.set_title(title)
    else:
        ax.set_title('Representation Velocity Over Context (Look for Spike at Disambiguation)')

    if disambig_pos is not None:
        ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved velocity plot to: {output_path}")

    return fig


def plot_layerwise_comparison(
    results: dict,
    metric: str = "collapse_distance_mean",
    output_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """
    Create comparison grid showing metric across layers and conditions.

    Args:
        results: Experiment results dict with conditions nested by layer
        metric: Which metric to plot from aggregated results
        output_path: If provided, save figure to this path
        title: Plot title
    """
    conditions_data = results.get("conditions", {})

    # Get layers and conditions
    layers = sorted([int(k.split("_")[1]) for k in conditions_data.keys()])
    if not layers:
        warnings.warn("No layer data found in results")
        return None

    first_layer_key = f"layer_{layers[0]}"
    condition_names = list(conditions_data.get(first_layer_key, {}).keys())

    # Build data matrix
    data = np.zeros((len(layers), len(condition_names)))

    for i, layer in enumerate(layers):
        layer_key = f"layer_{layer}"
        for j, cond in enumerate(condition_names):
            agg = conditions_data.get(layer_key, {}).get(cond, {}).get("aggregated", {})
            data[i, j] = agg.get(metric, 0)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(data, cmap='YlOrRd', aspect='auto')

    # Labels
    ax.set_xticks(range(len(condition_names)))
    ax.set_xticklabels([c.replace('_', '\n') for c in condition_names], fontsize=10)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f'Layer {l}' for l in layers])

    # Add values
    for i in range(len(layers)):
        for j in range(len(condition_names)):
            text = ax.text(j, i, f'{data[i, j]:.3f}',
                          ha='center', va='center', fontsize=9)

    plt.colorbar(im, ax=ax, label=metric.replace('_', ' ').title())

    ax.set_xlabel('Condition')
    ax.set_ylabel('Layer')

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'{metric.replace("_", " ").title()} by Layer and Condition')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved layerwise comparison to: {output_path}")

    return fig


def plot_condition_comparison(
    results: dict,
    layer: int,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """
    Create bar chart comparing key metrics across conditions for a single layer.

    Args:
        results: Experiment results dict
        layer: Which layer to plot
        output_path: If provided, save figure to this path
        title: Plot title
    """
    layer_key = f"layer_{layer}"
    layer_data = results.get("conditions", {}).get(layer_key, {})

    if not layer_data:
        warnings.warn(f"No data found for layer {layer}")
        return None

    conditions = list(layer_data.keys())
    metrics = ["collapse_distance_mean", "velocity_spike_mean", "collapse_accuracy"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, metric in zip(axes, metrics):
        values = [layer_data[c]["aggregated"].get(metric, 0) for c in conditions]
        bars = ax.bar(conditions, values, color='steelblue', alpha=0.8, edgecolor='navy')

        ax.set_xlabel('Condition')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_xticklabels([c.replace('_', '\n') for c in conditions], fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    if title:
        fig.suptitle(title, fontsize=12)
    else:
        fig.suptitle(f'Condition Comparison (Layer {layer})', fontsize=12)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved condition comparison to: {output_path}")

    return fig


def create_summary_figure(
    results: dict,
    output_path: Optional[str] = None,
):
    """
    Create a comprehensive summary figure with multiple panels.

    Args:
        results: Experiment results dict
        output_path: If provided, save figure to this path
    """
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)

    conditions_data = results.get("conditions", {})
    layers = sorted([int(k.split("_")[1]) for k in conditions_data.keys()])

    if not layers:
        warnings.warn("No layer data found")
        return None

    first_layer = layers[0]
    mid_layer = layers[len(layers) // 2]
    last_layer = layers[-1]

    # Panel 1: Collapse distance heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    condition_names = list(conditions_data.get(f"layer_{first_layer}", {}).keys())

    data = np.zeros((len(layers), len(condition_names)))
    for i, layer in enumerate(layers):
        for j, cond in enumerate(condition_names):
            agg = conditions_data.get(f"layer_{layer}", {}).get(cond, {}).get("aggregated", {})
            data[i, j] = agg.get("collapse_distance_mean", 0)

    im = ax1.imshow(data, cmap='YlOrRd', aspect='auto')
    ax1.set_xticks(range(len(condition_names)))
    ax1.set_xticklabels([c.replace('_', '\n') for c in condition_names], fontsize=8)
    ax1.set_yticks(range(len(layers)))
    ax1.set_yticklabels([f'L{l}' for l in layers], fontsize=9)
    plt.colorbar(im, ax=ax1, label='Collapse Distance')
    ax1.set_title('Collapse Distance by Layer and Condition')

    # Panel 2: Velocity spike heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    data2 = np.zeros((len(layers), len(condition_names)))
    for i, layer in enumerate(layers):
        for j, cond in enumerate(condition_names):
            agg = conditions_data.get(f"layer_{layer}", {}).get(cond, {}).get("aggregated", {})
            data2[i, j] = agg.get("velocity_spike_mean", 0)

    im2 = ax2.imshow(data2, cmap='Blues', aspect='auto')
    ax2.set_xticks(range(len(condition_names)))
    ax2.set_xticklabels([c.replace('_', '\n') for c in condition_names], fontsize=8)
    ax2.set_yticks(range(len(layers)))
    ax2.set_yticklabels([f'L{l}' for l in layers], fontsize=9)
    plt.colorbar(im2, ax=ax2, label='Velocity Spike (x)')
    ax2.set_title('Velocity Spike by Layer and Condition')

    # Panel 3: Collapse accuracy by condition (middle layer)
    ax3 = fig.add_subplot(gs[1, 0])
    mid_data = conditions_data.get(f"layer_{mid_layer}", {})
    conds = list(mid_data.keys())
    accuracies = [mid_data[c]["aggregated"].get("collapse_accuracy", 0) for c in conds]
    bars = ax3.bar(conds, accuracies, color='seagreen', alpha=0.8)
    ax3.set_ylabel('Collapse Accuracy')
    ax3.set_title(f'Collapse Accuracy (Layer {mid_layer})')
    ax3.set_xticklabels([c.replace('_', '\n') for c in conds], fontsize=8)
    ax3.set_ylim(0, 1)
    ax3.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax3.legend()

    # Panel 4: Superposition change
    ax4 = fig.add_subplot(gs[1, 1])
    sup_changes = [mid_data[c]["aggregated"].get("superposition_change", 0) for c in conds]
    colors = ['green' if v < 0 else 'red' for v in sup_changes]
    bars = ax4.bar(conds, sup_changes, color=colors, alpha=0.7)
    ax4.set_ylabel('Superposition Change (post - pre)')
    ax4.set_title(f'Superposition Change (Layer {mid_layer})')
    ax4.set_xticklabels([c.replace('_', '\n') for c in conds], fontsize=8)
    ax4.axhline(0, color='gray', linestyle='-', alpha=0.5)

    # Panel 5: Collapse distance trend across layers (mid_reveal condition)
    ax5 = fig.add_subplot(gs[2, 0])
    mid_reveal_by_layer = [
        conditions_data.get(f"layer_{l}", {}).get("mid_reveal", {}).get("aggregated", {}).get("collapse_distance_mean", 0)
        for l in layers
    ]
    ax5.plot(layers, mid_reveal_by_layer, 'o-', color='coral', linewidth=2, markersize=8)
    ax5.set_xlabel('Layer')
    ax5.set_ylabel('Collapse Distance')
    ax5.set_title('Collapse Distance vs Layer (Mid-Reveal Condition)')
    ax5.grid(True, alpha=0.3)

    # Panel 6: Pre vs Post superposition
    ax6 = fig.add_subplot(gs[2, 1])
    pre_sup = [mid_data[c]["aggregated"].get("pre_superposition_mean", 0) for c in conds]
    post_sup = [mid_data[c]["aggregated"].get("post_superposition_mean", 0) for c in conds]

    x = np.arange(len(conds))
    width = 0.35
    ax6.bar(x - width/2, pre_sup, width, label='Pre-disambig', color='skyblue', alpha=0.8)
    ax6.bar(x + width/2, post_sup, width, label='Post-disambig', color='salmon', alpha=0.8)
    ax6.set_ylabel('Superposition Score')
    ax6.set_title(f'Pre vs Post Superposition (Layer {mid_layer})')
    ax6.set_xticks(x)
    ax6.set_xticklabels([c.replace('_', '\n') for c in conds], fontsize=8)
    ax6.legend()

    # Main title
    model_name = results.get("config", {}).get("model", "Unknown")
    fig.suptitle(f'Hypothesis Superposition & Collapse Experiment: {model_name}', fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved summary figure to: {output_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot visualizations for superposition collapse experiment"
    )

    parser.add_argument("--results-dir", type=str, default="results/superposition_collapse",
                       help="Directory with experiment results")
    parser.add_argument("--trial-file", type=str, default=None,
                       help="Specific trial file to visualize")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for plots (default: same as results)")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GENERATING SUPERPOSITION EXPERIMENT VISUALIZATIONS")
    print("=" * 70)

    # Load main results
    results_file = results_dir / "results.json"
    if results_file.exists():
        print(f"\nLoading results from: {results_file}")
        with open(results_file) as f:
            results = json.load(f)

        # Create summary figure
        print("\nCreating summary figure...")
        create_summary_figure(results, output_path=output_dir / "summary.png")

        # Create layerwise comparisons
        print("\nCreating layerwise comparison plots...")
        for metric in ["collapse_distance_mean", "velocity_spike_mean", "collapse_accuracy"]:
            plot_layerwise_comparison(
                results, metric=metric,
                output_path=output_dir / f"layerwise_{metric}.png"
            )

        # Create condition comparison for each layer
        layers = sorted([int(k.split("_")[1]) for k in results.get("conditions", {}).keys()])
        for layer in layers:
            plot_condition_comparison(
                results, layer=layer,
                output_path=output_dir / f"condition_comparison_layer{layer}.png"
            )

    else:
        print(f"  Results file not found: {results_file}")

    # If trial file specified, create trajectory plots
    if args.trial_file:
        trial_path = Path(args.trial_file)
        if trial_path.exists():
            print(f"\nLoading trial data from: {trial_path}")
            with open(trial_path) as f:
                trials = json.load(f)

            # Plot first few trials
            for i, trial in enumerate(trials[:3]):
                if "position_summary" in trial:
                    print(f"\n  Creating plots for trial {i}...")

                    # Get metadata
                    disambig_pos = trial.get("metadata", {}).get("disambig_position")
                    true_hyp = trial.get("metadata", {}).get("true_hypothesis", "H1")

                    # Distance plot
                    plot_distance_over_position(
                        trial["position_summary"],
                        disambig_pos=disambig_pos,
                        output_path=output_dir / f"distance_trial{i}.png",
                        title=f"Trial {i} (True: {true_hyp})"
                    )

                    # Superposition plot
                    plot_superposition_score_over_position(
                        trial["position_summary"],
                        disambig_pos=disambig_pos,
                        output_path=output_dir / f"superposition_trial{i}.png",
                        title=f"Trial {i} (True: {true_hyp})"
                    )

                    # Velocity plot
                    plot_velocity_over_position(
                        trial["position_summary"],
                        disambig_pos=disambig_pos,
                        output_path=output_dir / f"velocity_trial{i}.png",
                        title=f"Trial {i} (True: {true_hyp})"
                    )
        else:
            print(f"  Trial file not found: {trial_path}")

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
