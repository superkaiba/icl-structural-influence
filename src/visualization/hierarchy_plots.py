"""
Visualization functions for multi-level hierarchical experiments.

Provides plotting functions for:
- Multi-line Phi trajectory plots (one line per hierarchy level)
- Per-level CSS comparison plots
- Layer-wise structure emergence

These visualizations help reveal whether models learn hierarchical
structure in a stagewise manner (coarse-to-fine or fine-to-coarse).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pathlib import Path
from typing import Optional, Union


# Color palette for hierarchy levels (colorblind-friendly)
LEVEL_COLORS = [
    '#e41a1c',  # Red - Level 1 (coarsest)
    '#377eb8',  # Blue - Level 2
    '#4daf4a',  # Green - Level 3
    '#984ea3',  # Purple - Level 4 (finest)
    '#ff7f00',  # Orange - Level 5 (if needed)
]

LEVEL_NAMES = {
    1: 'Super-cluster',
    2: 'Mid-cluster',
    3: 'Sub-cluster',
    4: 'Leaf',
}


def plot_multilevel_phi_trajectory(
    trajectory_data: dict,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Hierarchical Structure Emergence",
    figsize: tuple = (12, 8),
    show_std: bool = True,
    normalize: bool = False,
) -> Figure:
    """
    Plot Phi trajectory for each hierarchy level on the same axes.

    Creates a publication-quality "Money Plot" showing:
    - X-axis: Context length (N)
    - Y-axis: Cluster separation (Phi)
    - Multiple lines: One per hierarchy level
    - Shaded regions: Standard deviation (optional)

    The plot reveals the temporal ordering of hierarchical learning:
    - If coarse levels rise first → coarse-to-fine learning
    - If fine levels rise first → fine-to-coarse learning

    Args:
        trajectory_data: Dict from compute_levelwise_phi_trajectory() with keys:
            - 'context_lengths': List of context lengths
            - 'phi_trajectory_level_{i}': Phi means for level i
            - 'phi_std_level_{i}': Phi stds for level i
        output_path: Path to save figure (optional)
        title: Plot title
        figsize: Figure size (width, height)
        show_std: Whether to show shaded standard deviation region
        normalize: If True, normalize each level's trajectory to [0, 1]

    Returns:
        matplotlib Figure object
    """
    context_lengths = trajectory_data['context_lengths']

    # Find which levels are present
    levels = []
    for key in trajectory_data.keys():
        if key.startswith('phi_trajectory_level_'):
            level = int(key.split('_')[-1])
            levels.append(level)
    levels = sorted(levels)

    if not levels:
        raise ValueError("No level data found in trajectory_data")

    fig, ax = plt.subplots(figsize=figsize)

    for level in levels:
        phi = np.array(trajectory_data[f'phi_trajectory_level_{level}'])
        phi_std = np.array(trajectory_data.get(f'phi_std_level_{level}', np.zeros_like(phi)))

        # Skip if all NaN
        if np.all(np.isnan(phi)):
            continue

        if normalize and not np.all(np.isnan(phi)):
            valid_mask = ~np.isnan(phi)
            phi_min = np.nanmin(phi)
            phi_max = np.nanmax(phi)
            if phi_max - phi_min > 1e-8:
                phi = (phi - phi_min) / (phi_max - phi_min)
                phi_std = phi_std / (phi_max - phi_min)

        color = LEVEL_COLORS[(level - 1) % len(LEVEL_COLORS)]
        level_name = LEVEL_NAMES.get(level, f'Level {level}')

        ax.plot(
            context_lengths, phi,
            'o-',
            color=color,
            linewidth=2.5,
            markersize=8,
            label=f'{level_name} (Level {level})'
        )

        if show_std:
            ax.fill_between(
                context_lengths,
                phi - phi_std,
                phi + phi_std,
                color=color,
                alpha=0.2
            )

    ax.set_xlabel("Context Length (N)", fontsize=12)
    ylabel = "Normalized Cluster Separation" if normalize else "Cluster Separation (Φ)"
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    # Use log scale for x-axis if range is large
    if max(context_lengths) / min(context_lengths) > 10:
        ax.set_xscale('log')

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        # Also save as PDF for vector graphics
        plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')

    return fig


def plot_multilevel_css_comparison(
    css_data: dict,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Per-Level Context Sensitivity",
    figsize: tuple = (14, 5),
) -> Figure:
    """
    Plot CSS (Context Sensitivity Score) for each hierarchy level side by side.

    Creates a multi-panel figure showing position-wise sensitivity at each level,
    helping identify which token positions are most sensitive to structure at
    different granularities.

    Args:
        css_data: Dict from compute_multilevel_decomposition() with keys:
            - 'css_level_{i}': Per-position CSS array for level i
        output_path: Path to save figure (optional)
        title: Overall figure title
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    # Find which levels are present
    levels = []
    for key in css_data.keys():
        if key.startswith('css_level_'):
            level = int(key.split('_')[-1])
            levels.append(level)
    levels = sorted(levels)

    if not levels:
        raise ValueError("No CSS level data found")

    n_levels = len(levels)
    fig, axes = plt.subplots(1, n_levels, figsize=figsize, sharey=True)

    if n_levels == 1:
        axes = [axes]

    for i, level in enumerate(levels):
        css = css_data[f'css_level_{level}']
        positions = np.arange(len(css))

        color = LEVEL_COLORS[(level - 1) % len(LEVEL_COLORS)]
        level_name = LEVEL_NAMES.get(level, f'Level {level}')

        axes[i].bar(positions, css, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        axes[i].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[i].set_xlabel("Token Position", fontsize=10)
        axes[i].set_title(f'{level_name}\n(Level {level})', fontsize=11)
        axes[i].grid(True, alpha=0.3, axis='y')

    axes[0].set_ylabel("CSS (Context Sensitivity)", fontsize=10)
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')

    return fig


def plot_phi_heatmap(
    phi_by_level_and_context: dict,
    context_lengths: list,
    levels: list,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Structure Emergence Heatmap",
    figsize: tuple = (10, 6),
) -> Figure:
    """
    Create a heatmap showing Phi values (level × context length).

    This 2D visualization makes it easy to see which hierarchy levels
    have high structural separation at each context length.

    Args:
        phi_by_level_and_context: Dict[level][context_length] -> Phi value
        context_lengths: List of context lengths (x-axis)
        levels: List of hierarchy levels (y-axis)
        output_path: Path to save figure
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    # Build matrix
    matrix = np.zeros((len(levels), len(context_lengths)))

    for i, level in enumerate(levels):
        for j, ctx_len in enumerate(context_lengths):
            if level in phi_by_level_and_context:
                matrix[i, j] = phi_by_level_and_context[level].get(ctx_len, np.nan)
            else:
                matrix[i, j] = np.nan

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        matrix,
        aspect='auto',
        cmap='viridis',
        interpolation='nearest'
    )

    ax.set_xticks(range(len(context_lengths)))
    ax.set_xticklabels(context_lengths)
    ax.set_yticks(range(len(levels)))
    ax.set_yticklabels([LEVEL_NAMES.get(l, f'Level {l}') for l in levels])

    ax.set_xlabel("Context Length (N)", fontsize=12)
    ax.set_ylabel("Hierarchy Level", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Cluster Separation (Φ)", fontsize=10)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')

    return fig


def plot_learning_order_analysis(
    trajectory_data: dict,
    output_path: Optional[Union[str, Path]] = None,
    threshold_fraction: float = 0.5,
    figsize: tuple = (10, 6),
) -> Figure:
    """
    Analyze and visualize the order in which hierarchy levels are learned.

    Computes the context length at which each level reaches a threshold
    fraction of its maximum Phi value, then plots this as a bar chart.

    Args:
        trajectory_data: Dict from compute_levelwise_phi_trajectory()
        output_path: Path to save figure
        threshold_fraction: Fraction of max Phi to use as threshold (default 0.5)
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    context_lengths = trajectory_data['context_lengths']

    # Find which levels are present
    levels = []
    for key in trajectory_data.keys():
        if key.startswith('phi_trajectory_level_'):
            level = int(key.split('_')[-1])
            levels.append(level)
    levels = sorted(levels)

    # For each level, find context length to reach threshold
    threshold_contexts = {}

    for level in levels:
        phi = np.array(trajectory_data[f'phi_trajectory_level_{level}'])

        if np.all(np.isnan(phi)):
            continue

        max_phi = np.nanmax(phi)
        threshold = threshold_fraction * max_phi

        # Find first context length where phi >= threshold
        for i, ctx_len in enumerate(context_lengths):
            if not np.isnan(phi[i]) and phi[i] >= threshold:
                threshold_contexts[level] = ctx_len
                break
        else:
            # Never reached threshold
            threshold_contexts[level] = max(context_lengths)

    fig, ax = plt.subplots(figsize=figsize)

    x_positions = range(len(threshold_contexts))
    bars = ax.bar(
        x_positions,
        [threshold_contexts[l] for l in sorted(threshold_contexts.keys())],
        color=[LEVEL_COLORS[(l-1) % len(LEVEL_COLORS)] for l in sorted(threshold_contexts.keys())],
        edgecolor='black',
        linewidth=1
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([
        LEVEL_NAMES.get(l, f'Level {l}')
        for l in sorted(threshold_contexts.keys())
    ])

    ax.set_xlabel("Hierarchy Level", fontsize=12)
    ax.set_ylabel(f"Context Length to Reach {threshold_fraction:.0%} Max Φ", fontsize=12)
    ax.set_title("Learning Order Analysis: When Each Level Emerges", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')

    return fig
