#!/usr/bin/env python3
"""
Create 2D heatmap visualization of LOO influence across layers and context lengths.

Supports visualization of both mean and standard deviation.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, LogNorm
from matplotlib.patches import Rectangle
from pathlib import Path


def load_results():
    """Load LOO results for both token conditions."""
    results_base = Path("results/loo_experiments")

    with open(results_base / "semantic_tokens/loo_results.json") as f:
        semantic = json.load(f)

    with open(results_base / "unrelated_tokens/loo_results.json") as f:
        unrelated = json.load(f)

    return semantic, unrelated


def build_influence_matrix(results, token_type="bridge", stat="mean"):
    """
    Build 2D matrix of influence values.

    Args:
        results: LOO results dict
        token_type: "bridge" or "anchor"
        stat: "mean" or "std"

    Returns:
        matrix: (n_contexts, n_layers) array
        context_lengths: list of N values
        layers: list of layer indices
    """
    exp_a = results["exp_a"]

    # Get sorted context lengths
    context_lengths = sorted([int(k) for k in exp_a["influence_by_N"].keys()])

    # Dynamically detect all available layers from the data
    all_layers = set()
    for n_str, data in exp_a["influence_by_N"].items():
        for key in ["bridge_mean", "anchor_mean"]:
            if key in data and data[key]:
                all_layers.update(int(l) for l in data[key].keys())
    layers = sorted(all_layers)

    # Build matrix
    matrix = np.full((len(context_lengths), len(layers)), np.nan)

    key = f"{token_type}_{stat}"

    for i, n in enumerate(context_lengths):
        data = exp_a["influence_by_N"][str(n)]
        if key in data and data[key]:
            for j, layer in enumerate(layers):
                val = data[key].get(str(layer))
                if val is not None:
                    matrix[i, j] = val

    return matrix, context_lengths, layers


def create_heatmap_with_variance():
    """Create heatmap with both mean and std visualization."""
    semantic, unrelated = load_results()

    # Check if std data is available
    sample_data = semantic["exp_a"]["influence_by_N"]
    first_key = list(sample_data.keys())[0]
    has_std = "bridge_std" in sample_data[first_key]

    if not has_std:
        print("No std data available - falling back to mean-only heatmap")
        return create_heatmap()

    # Build all matrices (mean and std)
    bridge_sem_mean, context_lengths, layers = build_influence_matrix(semantic, "bridge", "mean")
    anchor_sem_mean, _, _ = build_influence_matrix(semantic, "anchor", "mean")
    bridge_unrel_mean, _, _ = build_influence_matrix(unrelated, "bridge", "mean")
    anchor_unrel_mean, _, _ = build_influence_matrix(unrelated, "anchor", "mean")

    bridge_sem_std, _, _ = build_influence_matrix(semantic, "bridge", "std")
    anchor_sem_std, _, _ = build_influence_matrix(semantic, "anchor", "std")
    bridge_unrel_std, _, _ = build_influence_matrix(unrelated, "bridge", "std")
    anchor_unrel_std, _, _ = build_influence_matrix(unrelated, "anchor", "std")

    n_layers = len(layers)
    n_contexts = len(context_lengths)

    print(f"Data dimensions: {n_contexts} context lengths × {n_layers} layers")

    # Find range for mean (symmetric log scale)
    all_means = np.concatenate([
        bridge_sem_mean.flatten(), anchor_sem_mean.flatten(),
        bridge_unrel_mean.flatten(), anchor_unrel_mean.flatten()
    ])
    all_means = all_means[~np.isnan(all_means)]
    vmax_mean = np.nanmax(np.abs(all_means))
    linthresh = 0.1

    print(f"Mean range: [{np.nanmin(all_means):.2f}, {np.nanmax(all_means):.2f}]")

    # Find range for std (log scale, always positive)
    all_stds = np.concatenate([
        bridge_sem_std.flatten(), anchor_sem_std.flatten(),
        bridge_unrel_std.flatten(), anchor_unrel_std.flatten()
    ])
    all_stds = all_stds[~np.isnan(all_stds)]
    all_stds = all_stds[all_stds > 0]  # Remove zeros for log scale
    if len(all_stds) > 0:
        vmin_std = max(0.01, np.percentile(all_stds, 5))
        vmax_std = np.percentile(all_stds, 95)
    else:
        vmin_std, vmax_std = 0.01, 10

    print(f"Std range: [{np.nanmin(all_stds):.2f}, {np.nanmax(all_stds):.2f}]")

    # Create figure: 4 rows (Bridge Mean, Bridge Std, Anchor Mean, Anchor Std) x 2 cols (Semantic, Unrelated)
    fig_width = max(12, 5 + n_layers * 0.25)
    fig_height = max(16, 6 + n_contexts * 0.5)
    fig, axes = plt.subplots(4, 2, figsize=(fig_width, fig_height))

    # Data organization: [row][col]
    mean_data = [
        [bridge_sem_mean, bridge_unrel_mean],  # Bridge means
        [anchor_sem_mean, anchor_unrel_mean],  # Anchor means
    ]
    std_data = [
        [bridge_sem_std, bridge_unrel_std],    # Bridge stds
        [anchor_sem_std, anchor_unrel_std],    # Anchor stds
    ]

    row_titles = ["Bridge Mean", "Bridge Std Dev", "Anchor Mean", "Anchor Std Dev"]
    col_titles = ["Semantic", "Unrelated"]

    # Normalizations
    norm_mean = SymLogNorm(linthresh=linthresh, linscale=1, vmin=-vmax_mean, vmax=vmax_mean, base=10)
    norm_std = LogNorm(vmin=vmin_std, vmax=vmax_std)

    # Plot
    ims_mean = []
    ims_std = []

    for col in range(2):  # Semantic, Unrelated
        for row_idx, (data_list, norm, cmap, im_list) in enumerate([
            (mean_data, norm_mean, 'RdBu_r', ims_mean),
            (std_data, norm_std, 'YlOrRd', ims_std)
        ]):
            for token_idx in range(2):  # Bridge, Anchor
                ax_row = token_idx * 2 + row_idx  # 0,1,2,3
                ax = axes[ax_row, col]
                data = data_list[token_idx][col]

                # Handle zeros/negatives for log scale
                if row_idx == 1:  # std data
                    data = np.clip(data, vmin_std, None)

                X, Y = np.meshgrid(np.arange(n_layers + 1), np.arange(n_contexts + 1))

                im = ax.pcolormesh(
                    X, Y, data,
                    cmap=cmap,
                    norm=norm,
                    edgecolors='black',
                    linewidth=0.3
                )
                im_list.append(im)

                # X-axis ticks
                if n_layers > 10:
                    step = max(1, n_layers // 8)
                    x_tick_indices = list(range(0, n_layers, step))
                    if (n_layers - 1) not in x_tick_indices:
                        x_tick_indices.append(n_layers - 1)
                else:
                    x_tick_indices = list(range(n_layers))

                ax.set_xticks([idx + 0.5 for idx in x_tick_indices])
                ax.set_xticklabels([layers[idx] for idx in x_tick_indices], fontsize=7)

                # Y-axis ticks
                if n_contexts > 10:
                    step_y = max(1, n_contexts // 8)
                    y_tick_indices = list(range(0, n_contexts, step_y))
                else:
                    y_tick_indices = list(range(n_contexts))

                ax.set_yticks([idx + 0.5 for idx in y_tick_indices])
                ax.set_yticklabels([context_lengths[idx] for idx in y_tick_indices], fontsize=7)

                ax.set_xlim(0, n_layers)
                ax.set_ylim(0, n_contexts)

                # Labels
                if ax_row == 3:
                    ax.set_xlabel("Layer", fontsize=9)
                if col == 0:
                    ax.set_ylabel("Context Length (N)", fontsize=9)

                # Title for top row only
                if ax_row == 0:
                    ax.set_title(col_titles[col], fontsize=11, fontweight='bold')

    # Row labels on left side
    for ax_row, title in enumerate(row_titles):
        axes[ax_row, 0].annotate(
            title, xy=(-0.15, 0.5), xycoords='axes fraction',
            fontsize=10, fontweight='bold', rotation=90,
            ha='center', va='center'
        )

    # Colorbars
    fig.subplots_adjust(right=0.85, hspace=0.3, wspace=0.15)

    # Mean colorbar
    cbar_ax_mean = fig.add_axes([0.87, 0.55, 0.02, 0.35])
    cbar_mean = fig.colorbar(ims_mean[0], cax=cbar_ax_mean)
    cbar_mean.set_label("Mean Influence\n(symlog scale)", fontsize=9)

    # Std colorbar
    cbar_ax_std = fig.add_axes([0.87, 0.1, 0.02, 0.35])
    cbar_std = fig.colorbar(ims_std[0], cax=cbar_ax_std)
    cbar_std.set_label("Std Dev\n(log scale)", fontsize=9)

    # Overall title
    fig.suptitle(
        "Leave-One-Out Influence: Mean and Standard Deviation\n"
        "Red/Blue = Mean (positive/negative), Yellow/Red = Std Dev (low/high)",
        fontsize=12, fontweight='bold', y=0.98
    )

    # Save
    output_path = Path("results/loo_experiments/loo_influence_heatmap_with_std.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Saved: {output_path}")

    return output_path


def create_heatmap():
    """Create 2x2 heatmap grid with black borders around cells (mean only)."""
    semantic, unrelated = load_results()

    # Build all 4 matrices
    bridge_semantic, context_lengths, layers = build_influence_matrix(semantic, "bridge", "mean")
    anchor_semantic, _, _ = build_influence_matrix(semantic, "anchor", "mean")
    bridge_unrelated, _, _ = build_influence_matrix(unrelated, "bridge", "mean")
    anchor_unrelated, _, _ = build_influence_matrix(unrelated, "anchor", "mean")

    n_layers = len(layers)
    n_contexts = len(context_lengths)

    print(f"Data dimensions: {n_contexts} context lengths × {n_layers} layers")

    # Find global min/max for consistent colormap
    all_data = np.concatenate([
        bridge_semantic.flatten(),
        anchor_semantic.flatten(),
        bridge_unrelated.flatten(),
        anchor_unrelated.flatten()
    ])
    all_data = all_data[~np.isnan(all_data)]

    vmax = np.nanmax(np.abs(all_data))
    linthresh = 0.1

    print(f"Data range: [{np.nanmin(all_data):.2f}, {np.nanmax(all_data):.2f}]")
    print(f"Using SymLogNorm with linthresh={linthresh}, vmax={vmax:.2f}")

    # Create figure
    fig_width = max(14, 6 + n_layers * 0.3)
    fig_height = max(10, 4 + n_contexts * 0.3)
    fig, axes = plt.subplots(2, 2, figsize=(fig_width, fig_height))

    data_grid = [
        [bridge_semantic, bridge_unrelated],
        [anchor_semantic, anchor_unrelated]
    ]

    row_titles = ["Bridge Tokens", "Anchor Tokens"]
    col_titles = ["Semantic Tokens", "Unrelated Tokens"]

    norm = SymLogNorm(linthresh=linthresh, linscale=1, vmin=-vmax, vmax=vmax, base=10)

    ims = []
    for i in range(2):
        for j in range(2):
            ax = axes[i, j]
            data = data_grid[i][j]

            X, Y = np.meshgrid(np.arange(n_layers + 1), np.arange(n_contexts + 1))

            im = ax.pcolormesh(
                X, Y, data,
                cmap='RdBu_r',
                norm=norm,
                edgecolors='black',
                linewidth=0.5
            )
            ims.append(im)

            if n_layers > 10:
                step = max(1, n_layers // 8)
                x_tick_indices = list(range(0, n_layers, step))
                if (n_layers - 1) not in x_tick_indices:
                    x_tick_indices.append(n_layers - 1)
            else:
                x_tick_indices = list(range(n_layers))

            ax.set_xticks([idx + 0.5 for idx in x_tick_indices])
            ax.set_xticklabels([layers[idx] for idx in x_tick_indices], fontsize=8)
            ax.set_xlabel("Layer", fontsize=10)

            if n_contexts > 10:
                y_tick_indices = [0, 2, 4, 6, 8, 10, 12, 14]
                y_tick_indices = [idx for idx in y_tick_indices if idx < n_contexts]
            else:
                y_tick_indices = list(range(n_contexts))

            ax.set_yticks([idx + 0.5 for idx in y_tick_indices])
            ax.set_yticklabels([context_lengths[idx] for idx in y_tick_indices], fontsize=8)
            ax.set_ylabel("Context Length (N)", fontsize=10)

            ax.set_title(f"{row_titles[i]} - {col_titles[j]}", fontsize=11, fontweight='bold')
            ax.set_xlim(0, n_layers)
            ax.set_ylim(0, n_contexts)

    fig.subplots_adjust(right=0.88, top=0.90)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(ims[0], cax=cbar_ax)
    cbar.set_label("LOO Influence (symmetric log scale)", fontsize=11)

    fig.suptitle(
        "Leave-One-Out Influence: Layer × Context Length\n"
        "Red = positive (removing hurts), Blue = negative (removing helps)",
        fontsize=13, fontweight='bold', y=0.95
    )

    output_path = Path("results/loo_experiments/loo_influence_heatmap.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Saved: {output_path}")

    return output_path


if __name__ == "__main__":
    # Try to create heatmap with variance, fall back to mean-only
    create_heatmap_with_variance()
