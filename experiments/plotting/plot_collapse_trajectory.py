#!/usr/bin/env python3
"""
Plot collapse metric trajectories over context length.

Usage:
    PYTHONPATH=. python experiments/plotting/plot_collapse_trajectory.py
    PYTHONPATH=. python experiments/plotting/plot_collapse_trajectory.py --results-dir results/collapse_trajectory_test
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_results(results_dir: str) -> tuple[dict, dict]:
    with open(Path(results_dir) / "trajectory_results.json") as f:
        data = json.load(f)
    return data["config"], data["trajectories"]


def extract_metric_by_layer(trajectories: list[list[dict]], layer: int,
                             metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (positions, mean_values, std_values) across trials for a given layer and metric."""
    # Get positions from first trial
    positions = np.array([t["position"] for t in trajectories[0]])

    all_values = []
    for trial in trajectories:
        values = []
        for t in trial:
            lm = t["layer_metrics"].get(str(layer), {})
            values.append(lm.get(metric, np.nan))
        all_values.append(values)

    all_values = np.array(all_values)
    mean_vals = np.nanmean(all_values, axis=0)
    std_vals = np.nanstd(all_values, axis=0)

    return positions, mean_vals, std_vals


COLORS = {
    "structured_walk": "#e74c3c",
    "natural_books": "#2ecc71",
    "repeated_token": "#9b59b6",
}

LABELS = {
    "structured_walk": "Structured walk",
    "natural_books": "Natural books",
    "repeated_token": "Repeated token",
}


def plot_trajectory_by_type(config: dict, trajectories: dict, output_dir: Path,
                             metric: str = "cos_sim", ylabel: str = "Cosine Similarity"):
    """One subplot per layer, lines per context type."""
    layers = config["layers"]
    n_layers = len(layers)

    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 5), sharey=True)
    if n_layers == 1:
        axes = [axes]

    for idx, layer in enumerate(layers):
        ax = axes[idx]

        for ctx_type, trials in trajectories.items():
            color = COLORS.get(ctx_type, "#333333")
            label = LABELS.get(ctx_type, ctx_type)

            positions, mean_vals, std_vals = extract_metric_by_layer(
                trials, layer, metric
            )

            ax.plot(positions, mean_vals, color=color, linewidth=2, label=label)
            if len(trials) > 1:
                ax.fill_between(positions,
                                mean_vals - std_vals,
                                mean_vals + std_vals,
                                color=color, alpha=0.15)

        ax.set_title(f"Layer {layer}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Token Position", fontsize=10)
        if idx == 0:
            ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)

    axes[-1].legend(fontsize=9, loc='best')

    fig.suptitle(f"Collapse Trajectory: {ylabel} Over Context",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    fname = f"trajectory_{metric}.png"
    fig.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_trajectory_by_layer(config: dict, trajectories: dict, output_dir: Path,
                              metric: str = "cos_sim", ylabel: str = "Cosine Similarity"):
    """One subplot per context type, lines per layer."""
    layers = config["layers"]
    ctx_types = list(trajectories.keys())

    fig, axes = plt.subplots(1, len(ctx_types), figsize=(5 * len(ctx_types), 5), sharey=True)
    if len(ctx_types) == 1:
        axes = [axes]

    layer_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(layers)))

    for idx, ctx_type in enumerate(ctx_types):
        ax = axes[idx]
        trials = trajectories[ctx_type]

        for li, layer in enumerate(layers):
            positions, mean_vals, std_vals = extract_metric_by_layer(
                trials, layer, metric
            )

            ax.plot(positions, mean_vals, color=layer_colors[li], linewidth=2,
                    label=f"L{layer}")
            if len(trials) > 1:
                ax.fill_between(positions,
                                mean_vals - std_vals,
                                mean_vals + std_vals,
                                color=layer_colors[li], alpha=0.12)

        title = LABELS.get(ctx_type, ctx_type)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("Token Position", fontsize=10)
        if idx == 0:
            ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best')
        ax.tick_params(labelsize=9)

    fig.suptitle(f"Layer-wise {ylabel} Trajectory",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    fname = f"trajectory_{metric}_by_layer.png"
    fig.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_combined_last_layer(config: dict, trajectories: dict, output_dir: Path):
    """Single 2-panel plot: cos_sim and eff_dim at the last layer."""
    last_layer = config["layers"][-1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    metrics_info = [
        ("cos_sim", "Cosine Similarity", axes[0]),
        ("eff_dim", "Effective Dimension", axes[1]),
    ]

    for metric, ylabel, ax in metrics_info:
        for ctx_type, trials in trajectories.items():
            color = COLORS.get(ctx_type, "#333333")
            label = LABELS.get(ctx_type, ctx_type)

            positions, mean_vals, std_vals = extract_metric_by_layer(
                trials, last_layer, metric
            )

            ax.plot(positions, mean_vals, color=color, linewidth=2.5, label=label)
            if len(trials) > 1:
                ax.fill_between(positions,
                                mean_vals - std_vals,
                                mean_vals + std_vals,
                                color=color, alpha=0.15)

        ax.set_xlabel("Token Position", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"Layer {last_layer}: {ylabel}", fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    fig.suptitle("Collapse Trajectory Over Context Length",
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "trajectory_combined.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: trajectory_combined.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/collapse_trajectory")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = str(Path(args.results_dir) / "plots")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {args.results_dir}...")
    config, trajectories = load_results(args.results_dir)
    print(f"Context types: {list(trajectories.keys())}")
    print(f"Layers: {config['layers']}")

    print(f"\nGenerating plots in {output_dir}")
    print("-" * 50)

    # Main plots
    plot_combined_last_layer(config, trajectories, output_dir)

    # Per-metric, by context type (one subplot per layer)
    plot_trajectory_by_type(config, trajectories, output_dir,
                            metric="cos_sim", ylabel="Cosine Similarity")
    plot_trajectory_by_type(config, trajectories, output_dir,
                            metric="eff_dim", ylabel="Effective Dimension")

    # Per-metric, by layer (one subplot per context type)
    plot_trajectory_by_layer(config, trajectories, output_dir,
                              metric="cos_sim", ylabel="Cosine Similarity")
    plot_trajectory_by_layer(config, trajectories, output_dir,
                              metric="eff_dim", ylabel="Effective Dimension")

    print("\nAll trajectory plots generated.")


if __name__ == "__main__":
    main()
