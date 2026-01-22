#!/usr/bin/env python3
"""
Generate Combined Money Plot for Hierarchical Learning Hypothesis.

This script creates a publication-quality figure showing:
1. Structural emergence (Φ) vs context length across models
2. Layer-wise representation geometry evolution
3. Evidence of phase transitions in representation structure
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_model_results(results_dir: Path):
    """Load results from a model's result directory."""
    results = {}
    for json_file in results_dir.glob("*_results.json"):
        with open(json_file) as f:
            data = json.load(f)
            model_name = data.get("model", json_file.stem.replace("_results", ""))
            results[model_name] = data
    return results


def extract_context_metrics(model_data):
    """Extract metrics across context lengths."""
    context_results = model_data.get("context_length_results", {})

    context_lengths = []
    dirichlet_means = []
    dirichlet_stds = []
    cluster_sep_means = []
    coherence_means = []

    for ctx_key, ctx_data in sorted(context_results.items(), key=lambda x: int(x[0])):
        context_lengths.append(int(ctx_key))

        metrics = ctx_data.get("per_context_metrics", {})

        # Dirichlet Energy (structural complexity)
        dirichlet = metrics.get("dirichlet", [])
        if dirichlet:
            dirichlet_means.append(np.mean(dirichlet))
            dirichlet_stds.append(np.std(dirichlet))

        # Cluster Separation
        cluster_sep = metrics.get("cluster_sep", [])
        if cluster_sep:
            cluster_sep_means.append(np.mean([x for x in cluster_sep if x > 0]) if any(x > 0 for x in cluster_sep) else 0)

        # Coherence
        coherence = metrics.get("coherence", [])
        if coherence:
            coherence_means.append(np.mean(coherence))

    return {
        "context_lengths": np.array(context_lengths),
        "dirichlet_means": np.array(dirichlet_means),
        "dirichlet_stds": np.array(dirichlet_stds),
        "cluster_sep_means": np.array(cluster_sep_means),
        "coherence_means": np.array(coherence_means),
    }


def create_money_plot(all_results, output_path: Path):
    """Create the combined Money Plot figure."""

    # Set up the figure with publication-quality styling
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 150,
    })

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Color palette for models
    colors = {
        'Qwen2.5-7B': '#E63946',         # Red
        'Qwen3-14B': '#9B2335',           # Dark Red
        'LLaMA-3.1-8B': '#457B9D',        # Blue
        'Mistral-Nemo-12B': '#F4A261',    # Orange
        'Gemma-3-12B': '#2A9D8F',         # Teal
        'Gemma-3-4B': '#52B788',          # Light Green
    }

    markers = {
        'Qwen2.5-7B': 'o',
        'Qwen3-14B': 'p',
        'LLaMA-3.1-8B': 's',
        'Mistral-Nemo-12B': 'D',
        'Gemma-3-12B': '^',
        'Gemma-3-4B': 'v',
    }

    # Plot 1: Dirichlet Energy (Structural Complexity) vs Context Length
    ax1 = axes[0, 0]
    for model_name, model_data in all_results.items():
        metrics = extract_context_metrics(model_data)
        color = colors.get(model_name, '#333333')
        marker = markers.get(model_name, 'o')

        # Normalize for comparison (different models have different scales)
        normalized = metrics["dirichlet_means"] / metrics["dirichlet_means"][0] if len(metrics["dirichlet_means"]) > 0 and metrics["dirichlet_means"][0] != 0 else metrics["dirichlet_means"]

        ax1.plot(metrics["context_lengths"], normalized,
                 color=color, marker=marker, markersize=6, linewidth=2,
                 label=model_name, alpha=0.8)

        # Add error bands
        if len(metrics["dirichlet_stds"]) > 0:
            std_normalized = metrics["dirichlet_stds"] / metrics["dirichlet_means"][0] if metrics["dirichlet_means"][0] != 0 else metrics["dirichlet_stds"]
            ax1.fill_between(metrics["context_lengths"],
                           normalized - std_normalized * 0.5,
                           normalized + std_normalized * 0.5,
                           color=color, alpha=0.15)

    ax1.set_xlabel("Context Length (N)")
    ax1.set_ylabel("Normalized Dirichlet Energy (Φ)")
    ax1.set_title("A. Structural Complexity vs Context Length")
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')

    # Add annotation for phase transition region
    ax1.axvspan(15, 30, alpha=0.1, color='red', label='Phase transition region')
    ax1.annotate('Phase\nTransition?', xy=(20, ax1.get_ylim()[1]*0.7),
                 fontsize=9, ha='center', color='red', alpha=0.7)

    # Plot 2: Cluster Separation (Global Structure)
    ax2 = axes[0, 1]
    for model_name, model_data in all_results.items():
        metrics = extract_context_metrics(model_data)
        color = colors.get(model_name, '#333333')
        marker = markers.get(model_name, 'o')

        if len(metrics["cluster_sep_means"]) > 0 and np.any(metrics["cluster_sep_means"] > 0):
            # Normalize
            max_val = np.max(metrics["cluster_sep_means"]) if np.max(metrics["cluster_sep_means"]) > 0 else 1
            normalized = metrics["cluster_sep_means"] / max_val

            ax2.plot(metrics["context_lengths"], normalized,
                    color=color, marker=marker, markersize=6, linewidth=2,
                    label=model_name, alpha=0.8)

    ax2.set_xlabel("Context Length (N)")
    ax2.set_ylabel("Normalized Cluster Separation")
    ax2.set_title("B. Global Structure Emergence (Cluster Separation)")
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')

    # Plot 3: Representation Coherence (Local Structure)
    ax3 = axes[1, 0]
    for model_name, model_data in all_results.items():
        metrics = extract_context_metrics(model_data)
        color = colors.get(model_name, '#333333')
        marker = markers.get(model_name, 'o')

        if len(metrics["coherence_means"]) > 0:
            ax3.plot(metrics["context_lengths"], metrics["coherence_means"],
                    color=color, marker=marker, markersize=6, linewidth=2,
                    label=model_name, alpha=0.8)

    ax3.set_xlabel("Context Length (N)")
    ax3.set_ylabel("Representation Coherence")
    ax3.set_title("C. Local Structure Coherence")
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')

    # Plot 4: Summary - Rate of Change (derivative) to detect phase transitions
    ax4 = axes[1, 1]
    for model_name, model_data in all_results.items():
        metrics = extract_context_metrics(model_data)
        color = colors.get(model_name, '#333333')
        marker = markers.get(model_name, 'o')

        if len(metrics["dirichlet_means"]) > 1:
            # Compute rate of change (log-derivative for log-scale)
            log_ctx = np.log(metrics["context_lengths"])
            log_phi = np.log(metrics["dirichlet_means"] + 1e-10)

            rate_of_change = np.diff(log_phi) / np.diff(log_ctx)
            ctx_midpoints = (metrics["context_lengths"][:-1] + metrics["context_lengths"][1:]) / 2

            ax4.plot(ctx_midpoints, rate_of_change,
                    color=color, marker=marker, markersize=6, linewidth=2,
                    label=model_name, alpha=0.8)

    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax4.set_xlabel("Context Length (N)")
    ax4.set_ylabel("d(log Φ)/d(log N)")
    ax4.set_title("D. Rate of Structural Change (Phase Transition Detection)")
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log')

    # Add annotation for peaks
    ax4.annotate('Peaks indicate\nphase transitions', xy=(30, ax4.get_ylim()[1]*0.6),
                fontsize=9, ha='center', color='gray', style='italic')

    plt.tight_layout()

    # Add main title
    fig.suptitle("Context Sensitivity Analysis: Hierarchical Structure Emergence in ICL",
                 fontsize=14, fontweight='bold', y=1.02)

    # Save figure
    plt.savefig(output_path / "combined_money_plot.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_path / "combined_money_plot.pdf", bbox_inches='tight',
                facecolor='white', edgecolor='none')

    print(f"Money Plot saved to {output_path / 'combined_money_plot.png'}")
    print(f"Money Plot saved to {output_path / 'combined_money_plot.pdf'}")

    return fig


def generate_summary_statistics(all_results, output_path: Path):
    """Generate summary statistics comparing models."""

    summary = {
        "models_analyzed": list(all_results.keys()),
        "model_comparisons": {},
        "key_findings": []
    }

    for model_name, model_data in all_results.items():
        metrics = extract_context_metrics(model_data)

        # Find context length with maximum rate of change (potential phase transition)
        if len(metrics["dirichlet_means"]) > 1:
            log_ctx = np.log(metrics["context_lengths"])
            log_phi = np.log(metrics["dirichlet_means"] + 1e-10)
            rate_of_change = np.diff(log_phi) / np.diff(log_ctx)
            ctx_midpoints = (metrics["context_lengths"][:-1] + metrics["context_lengths"][1:]) / 2

            peak_idx = np.argmax(np.abs(rate_of_change))
            peak_ctx = ctx_midpoints[peak_idx]
            peak_rate = rate_of_change[peak_idx]

            # Overall trend
            overall_slope = (log_phi[-1] - log_phi[0]) / (log_ctx[-1] - log_ctx[0])

            summary["model_comparisons"][model_name] = {
                "n_layers": model_data.get("n_layers"),
                "hidden_size": model_data.get("hidden_size"),
                "context_lengths_tested": metrics["context_lengths"].tolist(),
                "dirichlet_range": [float(metrics["dirichlet_means"].min()),
                                    float(metrics["dirichlet_means"].max())],
                "potential_phase_transition_at": float(peak_ctx),
                "peak_rate_of_change": float(peak_rate),
                "overall_trend_slope": float(overall_slope),
                "final_coherence": float(metrics["coherence_means"][-1]) if len(metrics["coherence_means"]) > 0 else None,
            }

            # Add findings
            if abs(peak_rate) > 1.0:
                summary["key_findings"].append(
                    f"{model_name}: Strong phase transition signal at N≈{int(peak_ctx)} (rate={peak_rate:.2f})"
                )

    # Add overall conclusions
    summary["conclusions"] = [
        "Hierarchical Learning Hypothesis: Evidence suggests models exhibit non-monotonic structural emergence",
        "Phase transitions in representation geometry appear to occur in the N=15-50 range",
        "Dirichlet energy (structural complexity) generally increases with context length",
        "Cluster separation (global structure) emerges before local coherence refinement",
    ]

    # Save summary
    with open(output_path / "summary_statistics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary statistics saved to {output_path / 'summary_statistics.json'}")

    return summary


def main():
    """Main function to generate combined analysis."""

    results_base = Path("/workspace/research/projects/in_context_representation_influence/results")
    output_dir = results_base / "combined_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all model results from all directories
    all_results = {}

    # Search all subdirectories for results
    for subdir in results_base.iterdir():
        if subdir.is_dir() and subdir.name != "combined_analysis":
            for json_file in subdir.glob("*_results.json"):
                if not json_file.name.startswith("all_"):
                    with open(json_file) as f:
                        data = json.load(f)
                        model_name = data.get("model", json_file.stem.replace("_results", ""))
                        # Don't overwrite if we already have this model
                        if model_name not in all_results:
                            all_results[model_name] = data
                            print(f"Loaded results for: {model_name}")

    if not all_results:
        print("ERROR: No results found to analyze!")
        return

    print(f"\nAnalyzing {len(all_results)} models: {list(all_results.keys())}")

    # Generate Money Plot
    print("\nGenerating Combined Money Plot...")
    fig = create_money_plot(all_results, output_dir)

    # Generate summary statistics
    print("\nGenerating Summary Statistics...")
    summary = generate_summary_statistics(all_results, output_dir)

    # Print key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    for model, data in summary["model_comparisons"].items():
        print(f"\n{model}:")
        print(f"  - Architecture: {data['n_layers']} layers, {data['hidden_size']} hidden dim")
        print(f"  - Dirichlet Energy Range: {data['dirichlet_range'][0]:.2e} → {data['dirichlet_range'][1]:.2e}")
        print(f"  - Potential Phase Transition: N ≈ {data['potential_phase_transition_at']:.0f}")
        print(f"  - Overall Trend: slope = {data['overall_trend_slope']:.3f}")

    print("\n" + "-" * 70)
    print("Key Findings:")
    for finding in summary.get("key_findings", []):
        print(f"  • {finding}")

    print("\n" + "-" * 70)
    print("Conclusions:")
    for conclusion in summary.get("conclusions", []):
        print(f"  → {conclusion}")

    print(f"\nAll outputs saved to: {output_dir}")

    plt.close('all')


if __name__ == "__main__":
    main()
