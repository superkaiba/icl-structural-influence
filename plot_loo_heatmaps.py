#!/usr/bin/env python3
"""
Plot 2D heatmaps of LOO influence across layers and context lengths.

Creates heatmaps for each metric (Ratio, Energy, CSS) showing:
- X-axis: Layer (0-31)
- Y-axis: Context length (log scale)
- Color: Influence magnitude (diverging colormap)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_results(results_dir="results/loo_multilayer"):
    """Load results for both conditions."""
    results_path = Path(results_dir)

    results = {}
    for condition in ['semantic', 'unrelated']:
        filepath = results_path / f"results_{condition}.json"
        if filepath.exists():
            with open(filepath) as f:
                results[condition] = json.load(f)
            print(f"Loaded {condition} results")
        else:
            print(f"Warning: {filepath} not found")

    return results


def build_heatmap_matrices(results):
    """Build 2D matrices for heatmap visualization."""
    matrices = {}

    for condition, data in results.items():
        context_lengths = data['context_lengths']
        layers_tested = data.get('layers_tested', list(range(data.get('num_layers', 32))))

        matrices[condition] = {
            'context_lengths': context_lengths,
            'layers_tested': layers_tested,
        }

        # Initialize matrices for each metric and token type
        metrics = ['ratio_influence', 'energy_influence', 'cross_dist_influence',
                   'within_dist_influence']
        token_types = ['bridge', 'anchor']

        for metric in metrics:
            for token_type in token_types:
                key = f"{metric}_{token_type}"
                matrix = np.full((len(context_lengths), len(layers_tested)), np.nan)

                for i, N in enumerate(context_lengths):
                    for j, layer in enumerate(layers_tested):
                        layer_data = data['by_layer_N'].get(str(layer), {}).get(str(N), {})
                        token_data = layer_data.get(token_type, {})
                        metric_data = token_data.get(metric, {})
                        if isinstance(metric_data, dict) and 'mean' in metric_data:
                            matrix[i, j] = metric_data['mean']

                matrices[condition][key] = matrix

        # CSS matrices
        for token_type in ['bridge', 'anchor']:
            key = f"css_{token_type}"
            matrix = np.full((len(context_lengths), len(layers_tested)), np.nan)

            for i, N in enumerate(context_lengths):
                for j, layer in enumerate(layers_tested):
                    layer_data = data['by_layer_N'].get(str(layer), {}).get(str(N), {})
                    css_val = layer_data.get(f'css_{token_type}')
                    if css_val is not None:
                        matrix[i, j] = css_val

            matrices[condition][key] = matrix

    return matrices


def plot_single_metric_heatmap(matrices, metric_name, output_path, title_prefix=""):
    """Create a 2x2 heatmap for a single metric across conditions and token types."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    conditions = ['semantic', 'unrelated']
    token_types = ['bridge', 'anchor']

    # Determine color scale
    all_values = []
    for condition in conditions:
        if condition not in matrices:
            continue
        for token_type in token_types:
            key = f"{metric_name}_{token_type}"
            if key in matrices[condition]:
                vals = matrices[condition][key]
                all_values.extend(vals[~np.isnan(vals)].flatten())

    if not all_values:
        print(f"No data for {metric_name}")
        return

    # Use symmetric scale centered at 0
    vmax = np.nanpercentile(np.abs(all_values), 95)
    vmin = -vmax

    for col, condition in enumerate(conditions):
        if condition not in matrices:
            continue

        context_lengths = matrices[condition]['context_lengths']
        layers_tested = matrices[condition]['layers_tested']

        for row, token_type in enumerate(token_types):
            ax = axes[row, col]
            key = f"{metric_name}_{token_type}"

            if key not in matrices[condition]:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax.transAxes)
                continue

            data = matrices[condition][key]

            # Create heatmap
            im = ax.pcolormesh(
                np.arange(len(layers_tested) + 1),
                context_lengths + [context_lengths[-1] * 1.5],  # Extend for pcolormesh
                data,
                cmap='RdBu_r',
                vmin=vmin,
                vmax=vmax,
                shading='flat'
            )

            ax.set_yscale('log')
            ax.set_xlabel('Layer')
            ax.set_ylabel('Context Length (N)')
            ax.set_title(f'{condition.title()} - {token_type.title()}')

            # Add colorbar
            plt.colorbar(im, ax=ax, label='Influence')

            # Set x-ticks to show actual layer numbers
            ax.set_xticks(np.arange(len(layers_tested)) + 0.5)
            ax.set_xticklabels([str(layer) for layer in layers_tested])

            # Set y-ticks to show actual context lengths
            ax.set_yticks(context_lengths)
            ax.set_yticklabels([str(n) for n in context_lengths])

    plt.suptitle(f'{title_prefix}{metric_name.replace("_", " ").title()}\n'
                 f'LOO Influence across Layers and Context Lengths',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_css_heatmap(matrices, output_path):
    """Create a 2x2 heatmap for CSS (bridge vs anchor, semantic vs unrelated)."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    conditions = ['semantic', 'unrelated']
    token_types = ['bridge', 'anchor']

    # Determine color scale
    all_values = []
    for condition in conditions:
        if condition not in matrices:
            continue
        for token_type in token_types:
            key = f"css_{token_type}"
            if key in matrices[condition]:
                vals = matrices[condition][key]
                all_values.extend(vals[~np.isnan(vals)].flatten())

    if not all_values:
        print("No CSS data")
        return

    vmax = np.nanpercentile(np.abs(all_values), 95)
    vmin = -vmax

    for col, condition in enumerate(conditions):
        if condition not in matrices:
            continue

        context_lengths = matrices[condition]['context_lengths']
        num_layers = matrices[condition]['num_layers']

        for row, token_type in enumerate(token_types):
            ax = axes[row, col]
            key = f"css_{token_type}"

            if key not in matrices[condition]:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax.transAxes)
                continue

            data = matrices[condition][key]

            im = ax.pcolormesh(
                np.arange(num_layers + 1),
                context_lengths + [context_lengths[-1] * 1.5],
                data,
                cmap='RdBu_r',
                vmin=vmin,
                vmax=vmax,
                shading='flat'
            )

            ax.set_yscale('log')
            ax.set_xlabel('Layer')
            ax.set_ylabel('Context Length (N)')
            ax.set_title(f'{condition.title()} - {token_type.title()}')

            plt.colorbar(im, ax=ax, label='CSS = -Cov(loss, phi)')

            ax.set_yticks(context_lengths[::2])
            ax.set_yticklabels([str(n) for n in context_lengths[::2]])

    plt.suptitle('CSS (Covariance Sample Significance)\n'
                 'Positive = structure helps prediction',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_all_metrics_combined(matrices, output_path):
    """Create a mega-figure with all metrics."""
    metrics = [
        ('ratio_influence', 'Ratio Influence'),
        ('energy_influence', 'Dirichlet Energy Influence'),
        ('css', 'CSS'),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(24, 16))

    # Columns: Semantic-Bridge, Semantic-Anchor, Unrelated-Bridge, Unrelated-Anchor
    col_configs = [
        ('semantic', 'bridge'),
        ('semantic', 'anchor'),
        ('unrelated', 'bridge'),
        ('unrelated', 'anchor'),
    ]

    for row, (metric_key, metric_label) in enumerate(metrics):
        # Determine color scale for this metric
        all_values = []
        for condition, token_type in col_configs:
            if condition not in matrices:
                continue
            key = f"{metric_key}_{token_type}" if metric_key != 'css' else f"css_{token_type}"
            if key in matrices[condition]:
                vals = matrices[condition][key]
                all_values.extend(vals[~np.isnan(vals)].flatten())

        if not all_values:
            continue

        vmax = np.nanpercentile(np.abs(all_values), 95)
        vmin = -vmax

        for col, (condition, token_type) in enumerate(col_configs):
            ax = axes[row, col]

            if condition not in matrices:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax.transAxes)
                ax.set_title(f'{condition.title()} - {token_type.title()}')
                continue

            context_lengths = matrices[condition]['context_lengths']
            layers_tested = matrices[condition]['layers_tested']

            key = f"{metric_key}_{token_type}" if metric_key != 'css' else f"css_{token_type}"
            if key not in matrices[condition]:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax.transAxes)
                ax.set_title(f'{condition.title()} - {token_type.title()}')
                continue

            data = matrices[condition][key]

            im = ax.pcolormesh(
                np.arange(len(layers_tested) + 1),
                context_lengths + [context_lengths[-1] * 1.5],
                data,
                cmap='RdBu_r',
                vmin=vmin,
                vmax=vmax,
                shading='flat'
            )

            ax.set_yscale('log')

            if row == 2:
                ax.set_xlabel('Layer')
            if col == 0:
                ax.set_ylabel(f'{metric_label}\nContext Length')

            ax.set_title(f'{condition.title()} - {token_type.title()}')

            # Set x-ticks to show actual layer numbers
            ax.set_xticks(np.arange(len(layers_tested)) + 0.5)
            ax.set_xticklabels([str(layer) for layer in layers_tested], fontsize=8)

            # Set y-ticks
            ax.set_yticks(context_lengths)
            ax.set_yticklabels([str(n) for n in context_lengths])

            # Add colorbar only on rightmost column
            if col == 3:
                plt.colorbar(im, ax=ax, label='Influence')

    plt.suptitle('Multi-Layer LOO Influence Heatmaps\n'
                 'All Metrics across Layers (0-31) and Context Lengths (6-500)',
                 fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Generate all heatmap visualizations."""
    results_dir = Path("results/loo_multilayer")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    results = load_results(results_dir)
    if not results:
        print("No results found. Run run_multilayer_loo_experiment.py first.")
        return

    # Build matrices
    matrices = build_heatmap_matrices(results)

    # Generate individual metric heatmaps
    for metric in ['ratio_influence', 'energy_influence',
                   'cross_dist_influence', 'within_dist_influence']:
        plot_single_metric_heatmap(
            matrices, metric,
            results_dir / f"heatmap_{metric}.png"
        )

    # CSS heatmap
    plot_css_heatmap(matrices, results_dir / "heatmap_css.png")

    # Combined mega-figure
    plot_all_metrics_combined(matrices, results_dir / "heatmap_all_metrics.png")

    print("\n" + "="*60)
    print("All heatmaps generated!")
    print("="*60)


if __name__ == "__main__":
    main()
