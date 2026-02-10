#!/usr/bin/env python3
"""
Plotting Script for Collapse Reversal Experiment.

Generates visualizations for the collapse reversal experiment results:
1. Collapse trajectory with injection marker
2. Delta at injection (bar chart)
3. Before vs After comparison

Usage:
    python plot_collapse_reversal.py --results-dir results/collapse_reversal
    python plot_collapse_reversal.py --results-dir results/collapse_reversal --output-dir results/collapse_reversal/plots
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import defaultdict

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    "control_h1_continuous": "#1f77b4",      # Blue
    "inject_h2_same_tokens": "#ff7f0e",      # Orange
    "inject_different_graph": "#2ca02c",     # Green
    "inject_natural_books": "#d62728",       # Red
    "inject_natural_wikipedia": "#9467bd",   # Purple
}

CONDITION_LABELS = {
    "control_h1_continuous": "Control (H1 continuous)",
    "inject_h2_same_tokens": "H2 injection (same tokens)",
    "inject_different_graph": "Different graph injection",
    "inject_natural_books": "Natural language (books)",
    "inject_natural_wikipedia": "Natural language (Wikipedia)",
}


def load_trial_results(results_dir: Path) -> dict:
    """Load all trial results from raw directory."""
    raw_dir = results_dir / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw results directory not found: {raw_dir}")

    trial_results = defaultdict(list)
    for json_file in sorted(raw_dir.glob("*.json")):
        with open(json_file) as f:
            result = json.load(f)

        condition = result.get("condition")
        if condition and not result.get("error"):
            trial_results[condition].append(result)

    return dict(trial_results)


def load_config(results_dir: Path) -> dict:
    """Load experiment configuration."""
    config_path = results_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}


def aggregate_metrics_by_checkpoint(
    trial_results: list[dict],
    layers: list[int],
) -> dict:
    """
    Aggregate metrics across trials for each checkpoint.

    Returns:
        Dict mapping phase -> checkpoint -> layer -> {metric_mean, metric_std}
    """
    aggregated = {"phase1": defaultdict(lambda: defaultdict(dict)),
                  "phase2": defaultdict(lambda: defaultdict(dict))}

    for phase in ["phase1", "phase2"]:
        # Collect all checkpoints across trials
        all_checkpoints = set()
        for trial in trial_results:
            phase_results = trial.get("results", {}).get(phase, {})
            all_checkpoints.update(int(cp) for cp in phase_results.keys())

        for checkpoint in sorted(all_checkpoints):
            cp_str = str(checkpoint)

            for layer in layers:
                layer_str = str(layer)

                # Collect metric values across trials
                metric_values = defaultdict(list)

                for trial in trial_results:
                    phase_results = trial.get("results", {}).get(phase, {})
                    if cp_str in phase_results:
                        cp_data = phase_results[cp_str]
                        if layer_str in cp_data and cp_data[layer_str]:
                            for metric in ["avg_cos_sim", "avg_l2_dist", "spread", "effective_dim"]:
                                val = cp_data[layer_str].get(metric)
                                if val is not None:
                                    metric_values[metric].append(val)

                # Compute mean and std
                layer_agg = {}
                for metric, values in metric_values.items():
                    if values:
                        layer_agg[f"{metric}_mean"] = np.mean(values)
                        layer_agg[f"{metric}_std"] = np.std(values)

                if layer_agg:
                    aggregated[phase][checkpoint][layer] = layer_agg

    return aggregated


def plot_collapse_trajectory(
    trial_results: dict,
    config: dict,
    output_path: Optional[Path] = None,
    metric: str = "avg_cos_sim",
    layer: Optional[int] = None,
):
    """
    Plot collapse trajectory with injection marker for all conditions.

    Args:
        trial_results: Dict mapping condition -> list of trial results
        config: Experiment configuration
        output_path: Path to save figure
        metric: Which metric to plot
        layer: Which layer to plot (default: middle layer)
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    injection_point = config.get("injection_point", 5000)
    layers = config.get("layers", [0])

    if layer is None:
        layer = layers[len(layers) // 2]

    metric_labels = {
        "avg_cos_sim": "Average Cosine Similarity",
        "avg_l2_dist": "Average L2 Distance",
        "spread": "Spread (Total Variance)",
        "effective_dim": "Effective Dimension",
    }

    for condition, trials in trial_results.items():
        if not trials:
            continue

        color = COLORS.get(condition, "#333333")
        label = CONDITION_LABELS.get(condition, condition)

        # Aggregate metrics
        agg = aggregate_metrics_by_checkpoint(trials, [layer])

        # Combine phase1 and phase2 data
        checkpoints = []
        means = []
        stds = []

        for phase in ["phase1", "phase2"]:
            for cp in sorted(agg[phase].keys()):
                if layer in agg[phase][cp]:
                    mean_key = f"{metric}_mean"
                    std_key = f"{metric}_std"
                    if mean_key in agg[phase][cp][layer]:
                        checkpoints.append(cp)
                        means.append(agg[phase][cp][layer][mean_key])
                        stds.append(agg[phase][cp][layer].get(std_key, 0))

        if checkpoints:
            checkpoints = np.array(checkpoints)
            means = np.array(means)
            stds = np.array(stds)

            # Plot with error band
            ax.plot(checkpoints, means, color=color, label=label, linewidth=2)
            ax.fill_between(checkpoints, means - stds, means + stds,
                           color=color, alpha=0.2)

    # Add injection marker
    ax.axvline(x=injection_point, color='black', linestyle='--', linewidth=2,
               label=f'Injection point ({injection_point})')
    ax.axvspan(0, injection_point, alpha=0.05, color='blue', label='Phase 1 (collapse)')
    ax.axvspan(injection_point, config.get("total_length", injection_point * 2),
               alpha=0.05, color='green', label='Phase 2 (reversal test)')

    ax.set_xlabel("Context Length (tokens)", fontsize=12)
    ax.set_ylabel(metric_labels.get(metric, metric), fontsize=12)
    ax.set_title(f"Collapse Reversal: {metric_labels.get(metric, metric)} Over Context\n"
                 f"(Layer {layer})", fontsize=14)

    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_delta_at_injection(
    trial_results: dict,
    config: dict,
    output_path: Optional[Path] = None,
    metric: str = "effective_dim",
):
    """
    Plot bar chart of metric change (delta) at injection point for each condition.

    Args:
        trial_results: Dict mapping condition -> list of trial results
        config: Experiment configuration
        output_path: Path to save figure
        metric: Which metric to plot delta for
    """
    layers = config.get("layers", [0])
    conditions = list(trial_results.keys())

    fig, axes = plt.subplots(1, len(layers), figsize=(4 * len(layers), 6), sharey=True)
    if len(layers) == 1:
        axes = [axes]

    metric_labels = {
        "avg_cos_sim": "Cos Sim Change",
        "avg_l2_dist": "L2 Dist Change",
        "spread": "Spread Change",
        "effective_dim": "Eff. Dim. Change",
    }

    for ax_idx, layer in enumerate(layers):
        ax = axes[ax_idx]

        deltas = []
        delta_stds = []
        cond_labels = []
        cond_colors = []

        for condition in conditions:
            trials = trial_results.get(condition, [])
            if not trials:
                continue

            # Collect delta values across trials
            trial_deltas = []
            for trial in trials:
                transition = trial.get("results", {}).get("transition", {})
                delta = transition.get("delta", {})
                if delta and str(layer) in delta:
                    layer_delta = delta[str(layer)]
                    if metric in layer_delta:
                        trial_deltas.append(layer_delta[metric])

            if trial_deltas:
                deltas.append(np.mean(trial_deltas))
                delta_stds.append(np.std(trial_deltas))
                cond_labels.append(CONDITION_LABELS.get(condition, condition).replace(" ", "\n"))
                cond_colors.append(COLORS.get(condition, "#333333"))

        if deltas:
            x = np.arange(len(deltas))
            bars = ax.bar(x, deltas, yerr=delta_stds, capsize=5,
                         color=cond_colors, alpha=0.7, edgecolor='black')

            # Add zero line
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

            # Color bars based on positive/negative
            for bar, delta in zip(bars, deltas):
                if delta > 0:
                    bar.set_edgecolor('darkgreen')
                else:
                    bar.set_edgecolor('darkred')

            ax.set_xticks(x)
            ax.set_xticklabels(cond_labels, fontsize=8)
            ax.set_title(f"Layer {layer}", fontsize=12)

            if ax_idx == 0:
                ax.set_ylabel(metric_labels.get(metric, f"{metric} Change"), fontsize=11)

    fig.suptitle(f"Change in {metric_labels.get(metric, metric)} at Injection Point",
                fontsize=14, y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_before_after_comparison(
    trial_results: dict,
    config: dict,
    output_path: Optional[Path] = None,
):
    """
    Plot side-by-side comparison of final Phase 1 metrics vs final Phase 2 metrics.

    Args:
        trial_results: Dict mapping condition -> list of trial results
        config: Experiment configuration
        output_path: Path to save figure
    """
    layers = config.get("layers", [0])
    metrics = ["avg_cos_sim", "effective_dim", "spread"]
    conditions = list(trial_results.keys())

    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)))
    if len(metrics) == 1:
        axes = [axes]

    metric_labels = {
        "avg_cos_sim": "Average Cosine Similarity",
        "effective_dim": "Effective Dimension",
        "spread": "Spread (Total Variance)",
    }

    # Use middle layer
    layer = layers[len(layers) // 2]

    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]

        before_vals = []
        after_vals = []
        before_stds = []
        after_stds = []
        cond_labels = []
        cond_colors = []

        for condition in conditions:
            trials = trial_results.get(condition, [])
            if not trials:
                continue

            # Collect last Phase 1 and last Phase 2 values
            trial_before = []
            trial_after = []

            for trial in trials:
                transition = trial.get("results", {}).get("transition", {})

                last_before = transition.get("last_before", {})
                if last_before and str(layer) in last_before:
                    val = last_before[str(layer)].get(metric)
                    if val is not None:
                        trial_before.append(val)

                # Get last checkpoint in phase2
                phase2_results = trial.get("results", {}).get("phase2", {})
                if phase2_results:
                    last_cp = max(int(cp) for cp in phase2_results.keys())
                    last_after = phase2_results.get(str(last_cp), {})
                    if str(layer) in last_after:
                        val = last_after[str(layer)].get(metric)
                        if val is not None:
                            trial_after.append(val)

            if trial_before and trial_after:
                before_vals.append(np.mean(trial_before))
                after_vals.append(np.mean(trial_after))
                before_stds.append(np.std(trial_before))
                after_stds.append(np.std(trial_after))
                cond_labels.append(CONDITION_LABELS.get(condition, condition))
                cond_colors.append(COLORS.get(condition, "#333333"))

        if before_vals:
            x = np.arange(len(before_vals))
            width = 0.35

            bars1 = ax.bar(x - width/2, before_vals, width, yerr=before_stds,
                          label='End of Phase 1 (before injection)',
                          color=[c for c in cond_colors], alpha=0.6, capsize=3)
            bars2 = ax.bar(x + width/2, after_vals, width, yerr=after_stds,
                          label='End of Phase 2 (after injection)',
                          color=[c for c in cond_colors], alpha=1.0, capsize=3,
                          hatch='//')

            ax.set_xticks(x)
            ax.set_xticklabels(cond_labels, fontsize=9, rotation=15, ha='right')
            ax.set_ylabel(metric_labels.get(metric, metric), fontsize=11)
            ax.set_title(f"{metric_labels.get(metric, metric)} (Layer {layer})", fontsize=12)
            ax.legend(fontsize=9)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_all_layers_trajectory(
    trial_results: dict,
    config: dict,
    output_path: Optional[Path] = None,
    metric: str = "avg_cos_sim",
    condition: str = "inject_h2_same_tokens",
):
    """
    Plot collapse trajectory across all layers for a single condition.

    Args:
        trial_results: Dict mapping condition -> list of trial results
        config: Experiment configuration
        output_path: Path to save figure
        metric: Which metric to plot
        condition: Which condition to plot
    """
    trials = trial_results.get(condition, [])
    if not trials:
        print(f"No trials found for condition: {condition}")
        return None

    layers = config.get("layers", [0])
    injection_point = config.get("injection_point", 5000)

    fig, ax = plt.subplots(figsize=(14, 8))

    # Color gradient from early to late layers
    cmap = plt.cm.viridis
    layer_colors = [cmap(i / len(layers)) for i in range(len(layers))]

    metric_labels = {
        "avg_cos_sim": "Average Cosine Similarity",
        "avg_l2_dist": "Average L2 Distance",
        "spread": "Spread (Total Variance)",
        "effective_dim": "Effective Dimension",
    }

    for layer_idx, layer in enumerate(layers):
        color = layer_colors[layer_idx]

        # Aggregate metrics
        agg = aggregate_metrics_by_checkpoint(trials, [layer])

        # Combine phase1 and phase2 data
        checkpoints = []
        means = []
        stds = []

        for phase in ["phase1", "phase2"]:
            for cp in sorted(agg[phase].keys()):
                if layer in agg[phase][cp]:
                    mean_key = f"{metric}_mean"
                    std_key = f"{metric}_std"
                    if mean_key in agg[phase][cp][layer]:
                        checkpoints.append(cp)
                        means.append(agg[phase][cp][layer][mean_key])
                        stds.append(agg[phase][cp][layer].get(std_key, 0))

        if checkpoints:
            checkpoints = np.array(checkpoints)
            means = np.array(means)
            stds = np.array(stds)

            ax.plot(checkpoints, means, color=color, label=f"Layer {layer}",
                   linewidth=2, alpha=0.8)
            ax.fill_between(checkpoints, means - stds, means + stds,
                           color=color, alpha=0.15)

    # Add injection marker
    ax.axvline(x=injection_point, color='red', linestyle='--', linewidth=2,
               label=f'Injection point')

    ax.set_xlabel("Context Length (tokens)", fontsize=12)
    ax.set_ylabel(metric_labels.get(metric, metric), fontsize=12)
    ax.set_title(f"{CONDITION_LABELS.get(condition, condition)}: "
                 f"{metric_labels.get(metric, metric)} by Layer", fontsize=14)

    ax.legend(loc='best', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot collapse reversal experiment results")

    parser.add_argument("--results-dir", type=str, default="results/collapse_reversal",
                       help="Directory containing experiment results")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for plots (default: results_dir/plots)")
    parser.add_argument("--metric", type=str, default="avg_cos_sim",
                       choices=["avg_cos_sim", "avg_l2_dist", "spread", "effective_dim"],
                       help="Primary metric to plot")
    parser.add_argument("--layer", type=int, default=None,
                       help="Specific layer to plot (default: middle layer)")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading results from: {results_dir}")
    trial_results = load_trial_results(results_dir)
    config = load_config(results_dir)

    print(f"Found {len(trial_results)} conditions:")
    for cond, trials in trial_results.items():
        print(f"  {cond}: {len(trials)} trials")

    # Generate plots
    print("\nGenerating plots...")

    # 1. Main trajectory plot (all conditions)
    for metric in ["avg_cos_sim", "effective_dim"]:
        plot_collapse_trajectory(
            trial_results, config,
            output_path=output_dir / f"trajectory_{metric}.png",
            metric=metric,
            layer=args.layer,
        )

    # 2. Delta at injection (bar charts)
    for metric in ["avg_cos_sim", "effective_dim", "spread"]:
        plot_delta_at_injection(
            trial_results, config,
            output_path=output_dir / f"delta_{metric}.png",
            metric=metric,
        )

    # 3. Before vs After comparison
    plot_before_after_comparison(
        trial_results, config,
        output_path=output_dir / "before_after_comparison.png",
    )

    # 4. All layers trajectory for key conditions
    for condition in ["inject_h2_same_tokens", "inject_different_graph", "control_h1_continuous"]:
        if condition in trial_results:
            for metric in ["avg_cos_sim", "effective_dim"]:
                plot_all_layers_trajectory(
                    trial_results, config,
                    output_path=output_dir / f"layers_{condition}_{metric}.png",
                    metric=metric,
                    condition=condition,
                )

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
