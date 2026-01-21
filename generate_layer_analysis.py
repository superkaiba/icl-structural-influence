#!/usr/bin/env python3
"""
Layer-wise Analysis for Hierarchical Learning.

This script creates visualizations showing how representation geometry
evolves across transformer layers at different context lengths.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_layer_data(results_path):
    """Load layer-wise data from results."""
    with open(results_path) as f:
        data = json.load(f)
    return data


def create_layer_heatmap(model_data, output_path: Path):
    """Create heatmap of structural metrics across layers and context lengths."""

    model_name = model_data.get("model", "Unknown")
    context_results = model_data.get("context_length_results", {})
    layers_analyzed = model_data.get("layers_analyzed", [])

    # Build matrices for heatmap
    context_lengths = sorted([int(k) for k in context_results.keys()])

    # Extract dirichlet energy per layer per context length
    dirichlet_matrix = np.zeros((len(layers_analyzed), len(context_lengths)))
    cluster_sep_matrix = np.zeros((len(layers_analyzed), len(context_lengths)))
    coherence_matrix = np.zeros((len(layers_analyzed), len(context_lengths)))

    for ctx_idx, ctx_len in enumerate(context_lengths):
        ctx_data = context_results.get(str(ctx_len), {})
        layer_data = ctx_data.get("layers", {})

        for layer_idx, layer_num in enumerate(layers_analyzed):
            layer_key = f"layer_{layer_num}"
            if layer_key in layer_data:
                metrics = layer_data[layer_key].get("metrics", {})
                dirichlet_matrix[layer_idx, ctx_idx] = metrics.get("dirichlet_mean", 0)
                cluster_sep_matrix[layer_idx, ctx_idx] = metrics.get("cluster_sep_mean", 0)
                coherence_matrix[layer_idx, ctx_idx] = metrics.get("coherence_mean", 0)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Normalize for visualization
    def safe_normalize(matrix):
        min_val = matrix.min()
        max_val = matrix.max()
        if max_val - min_val > 0:
            return (matrix - min_val) / (max_val - min_val)
        return matrix

    # Plot 1: Dirichlet Energy
    ax1 = axes[0]
    im1 = ax1.imshow(safe_normalize(dirichlet_matrix), aspect='auto', cmap='viridis', origin='lower')
    ax1.set_xlabel("Context Length")
    ax1.set_ylabel("Layer")
    ax1.set_title(f"{model_name}: Dirichlet Energy (Structural Complexity)")
    ax1.set_xticks(range(len(context_lengths)))
    ax1.set_xticklabels(context_lengths, rotation=45)
    ax1.set_yticks(range(len(layers_analyzed)))
    ax1.set_yticklabels(layers_analyzed)
    plt.colorbar(im1, ax=ax1, label="Normalized Φ")

    # Plot 2: Cluster Separation
    ax2 = axes[1]
    im2 = ax2.imshow(safe_normalize(cluster_sep_matrix), aspect='auto', cmap='RdYlBu_r', origin='lower')
    ax2.set_xlabel("Context Length")
    ax2.set_ylabel("Layer")
    ax2.set_title(f"{model_name}: Cluster Separation (Global Structure)")
    ax2.set_xticks(range(len(context_lengths)))
    ax2.set_xticklabels(context_lengths, rotation=45)
    ax2.set_yticks(range(len(layers_analyzed)))
    ax2.set_yticklabels(layers_analyzed)
    plt.colorbar(im2, ax=ax2, label="Normalized Separation")

    # Plot 3: Coherence
    ax3 = axes[2]
    im3 = ax3.imshow(coherence_matrix, aspect='auto', cmap='coolwarm', origin='lower', vmin=0, vmax=1)
    ax3.set_xlabel("Context Length")
    ax3.set_ylabel("Layer")
    ax3.set_title(f"{model_name}: Representation Coherence (Local Structure)")
    ax3.set_xticks(range(len(context_lengths)))
    ax3.set_xticklabels(context_lengths, rotation=45)
    ax3.set_yticks(range(len(layers_analyzed)))
    ax3.set_yticklabels(layers_analyzed)
    plt.colorbar(im3, ax=ax3, label="Coherence")

    plt.tight_layout()

    # Save
    safe_name = model_name.replace("/", "_").replace(" ", "_")
    plt.savefig(output_path / f"{safe_name}_layer_heatmap.png", dpi=300, bbox_inches='tight')
    print(f"Layer heatmap saved for {model_name}")

    return fig, dirichlet_matrix, cluster_sep_matrix, coherence_matrix


def create_layer_trajectory_plot(all_model_data, output_path: Path):
    """Create plot showing how structural metrics evolve layer-by-layer."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {
        'Qwen2.5-7B': '#E63946',
        'LLaMA-3.1-8B': '#457B9D',
    }

    # Select specific context lengths to compare
    target_ctx_lengths = [10, 50, 100, 200]

    for model_name, model_data in all_model_data.items():
        context_results = model_data.get("context_length_results", {})
        layers_analyzed = model_data.get("layers_analyzed", [])
        color = colors.get(model_name, '#333333')

        for ctx_idx, target_ctx in enumerate(target_ctx_lengths):
            ax = axes[ctx_idx // 2, ctx_idx % 2]

            # Find closest context length
            available_ctx = sorted([int(k) for k in context_results.keys()])
            closest_ctx = min(available_ctx, key=lambda x: abs(x - target_ctx))

            ctx_data = context_results.get(str(closest_ctx), {})
            layer_data = ctx_data.get("layers", {})

            dirichlet_values = []
            for layer_num in layers_analyzed:
                layer_key = f"layer_{layer_num}"
                if layer_key in layer_data:
                    metrics = layer_data[layer_key].get("metrics", {})
                    dirichlet_values.append(metrics.get("dirichlet_mean", 0))
                else:
                    dirichlet_values.append(0)

            # Normalize by first layer for comparison
            dirichlet_values = np.array(dirichlet_values)
            if dirichlet_values[0] != 0:
                dirichlet_values = dirichlet_values / dirichlet_values[0]

            # Normalize layer indices to [0, 1] for comparison
            normalized_layers = np.array(layers_analyzed) / max(layers_analyzed)

            ax.plot(normalized_layers, dirichlet_values,
                   marker='o', linewidth=2, markersize=6,
                   color=color, label=model_name, alpha=0.8)

            ax.set_xlabel("Normalized Layer Depth")
            ax.set_ylabel("Normalized Dirichlet Energy")
            ax.set_title(f"N = {closest_ctx} (Context Length)")
            ax.legend()
            ax.grid(True, alpha=0.3)

    fig.suptitle("Layer-wise Structure Evolution at Different Context Lengths",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    plt.savefig(output_path / "layer_trajectory_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_path / "layer_trajectory_comparison.pdf", bbox_inches='tight')
    print(f"Layer trajectory comparison saved")

    return fig


def create_phase_transition_analysis(all_model_data, output_path: Path):
    """Detailed phase transition analysis."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {
        'Qwen2.5-7B': '#E63946',
        'LLaMA-3.1-8B': '#457B9D',
    }

    # Plot 1: Rate of change across context lengths (for middle layers)
    ax1 = axes[0]
    for model_name, model_data in all_model_data.items():
        context_results = model_data.get("context_length_results", {})
        layers_analyzed = model_data.get("layers_analyzed", [])
        color = colors.get(model_name, '#333333')

        # Use middle layer
        middle_layer = layers_analyzed[len(layers_analyzed) // 2]

        context_lengths = []
        dirichlet_values = []

        for ctx_key in sorted(context_results.keys(), key=int):
            ctx_data = context_results[ctx_key]
            layer_data = ctx_data.get("layers", {})
            layer_key = f"layer_{middle_layer}"

            if layer_key in layer_data:
                metrics = layer_data[layer_key].get("metrics", {})
                context_lengths.append(int(ctx_key))
                dirichlet_values.append(metrics.get("dirichlet_mean", 0))

        context_lengths = np.array(context_lengths)
        dirichlet_values = np.array(dirichlet_values)

        if len(context_lengths) > 1:
            # Compute derivative
            log_ctx = np.log(context_lengths)
            log_phi = np.log(dirichlet_values + 1e-10)
            derivative = np.diff(log_phi) / np.diff(log_ctx)
            midpoints = np.exp((log_ctx[:-1] + log_ctx[1:]) / 2)

            ax1.plot(midpoints, derivative,
                    marker='o', linewidth=2, markersize=6,
                    color=color, label=f"{model_name} (Layer {middle_layer})", alpha=0.8)

    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax1.axhline(y=1, color='gray', linestyle=':', alpha=0.3, label="Linear scaling")
    ax1.set_xlabel("Context Length (N)")
    ax1.set_ylabel("d(log Φ)/d(log N)")
    ax1.set_title("Rate of Structural Change (Middle Layer)")
    ax1.set_xscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Annotate interpretation
    ax1.annotate("Super-linear\ngrowth", xy=(0.7, 0.9), xycoords='axes fraction',
                fontsize=9, color='gray', ha='center')
    ax1.annotate("Sub-linear\ngrowth", xy=(0.7, 0.1), xycoords='axes fraction',
                fontsize=9, color='gray', ha='center')

    # Plot 2: Variance in structural metrics (indicator of phase transitions)
    ax2 = axes[1]
    for model_name, model_data in all_model_data.items():
        context_results = model_data.get("context_length_results", {})
        color = colors.get(model_name, '#333333')

        context_lengths = []
        variances = []

        for ctx_key in sorted(context_results.keys(), key=int):
            ctx_data = context_results[ctx_key]
            per_context = ctx_data.get("per_context_metrics", {})
            dirichlet = per_context.get("dirichlet", [])

            if dirichlet:
                context_lengths.append(int(ctx_key))
                # Coefficient of variation (normalized variance)
                cv = np.std(dirichlet) / (np.mean(dirichlet) + 1e-10)
                variances.append(cv)

        ax2.plot(context_lengths, variances,
                marker='s', linewidth=2, markersize=6,
                color=color, label=model_name, alpha=0.8)

    ax2.set_xlabel("Context Length (N)")
    ax2.set_ylabel("Coefficient of Variation (σ/μ)")
    ax2.set_title("Structural Variability Across Contexts")
    ax2.set_xscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Annotate
    ax2.annotate("High variability may indicate\nphase transition boundary",
                xy=(0.5, 0.95), xycoords='axes fraction',
                fontsize=9, color='gray', ha='center', style='italic')

    plt.tight_layout()
    plt.savefig(output_path / "phase_transition_analysis.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_path / "phase_transition_analysis.pdf", bbox_inches='tight')
    print("Phase transition analysis saved")

    return fig


def main():
    """Generate layer-wise analysis plots."""

    results_base = Path("/workspace/research/projects/in_context_representation_influence/results")
    output_dir = results_base / "combined_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all model results
    all_results = {}

    for subdir in ["hierarchical_qwen", "hierarchical_llama"]:
        subdir_path = results_base / subdir
        if subdir_path.exists():
            for json_file in subdir_path.glob("*_results.json"):
                if not json_file.name.startswith("all_"):
                    with open(json_file) as f:
                        data = json.load(f)
                        model_name = data.get("model", json_file.stem.replace("_results", ""))
                        all_results[model_name] = data
                        print(f"Loaded: {model_name}")

    if not all_results:
        print("No results found!")
        return

    # Generate layer heatmaps for each model
    print("\nGenerating layer heatmaps...")
    for model_name, model_data in all_results.items():
        create_layer_heatmap(model_data, output_dir)

    # Generate comparison plots
    print("\nGenerating layer trajectory comparison...")
    create_layer_trajectory_plot(all_results, output_dir)

    print("\nGenerating phase transition analysis...")
    create_phase_transition_analysis(all_results, output_dir)

    print(f"\nAll analysis plots saved to: {output_dir}")

    plt.close('all')


if __name__ == "__main__":
    main()
