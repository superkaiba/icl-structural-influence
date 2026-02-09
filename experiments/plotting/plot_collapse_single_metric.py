#!/usr/bin/env python3
"""
Plot each collapse metric separately with ALL conditions.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def load_results(results_dir: Path):
    """Load all trial results."""
    results = {}
    raw_dir = results_dir / "raw"

    for f in raw_dir.glob("*.json"):
        with open(f) as fp:
            data = json.load(fp)

        condition = data.get("condition", f.stem.rsplit("_trial_", 1)[0])
        if condition not in results:
            results[condition] = []
        results[condition].append(data)

    return results


def get_sort_key(condition: str) -> tuple:
    """Sort key for conditions."""
    if "no_ambig" in condition:
        return (0, 0)
    if "full_ambig" in condition:
        return (0, 1)
    if "natural" in condition:
        return (1, condition)
    if "vocab" in condition:
        num = int(condition.split("_")[1])
        return (2, num)
    if "disambig" in condition:
        # Parse percentage
        parts = condition.split("_")
        for i, p in enumerate(parts):
            if p == "disambig" and i + 1 < len(parts):
                pct_str = parts[i + 1]
                if i + 2 < len(parts) and "pct" in parts[i + 2]:
                    pct_str = pct_str + "." + parts[i + 2].replace("pct", "")
                return (3, float(pct_str.replace("pct", "")))
    return (4, condition)


def aggregate_by_checkpoint(trials: list, metric: str, layer: int) -> tuple:
    """Aggregate metric values across trials for each checkpoint."""
    checkpoint_values = {}

    for trial in trials:
        if trial.get("error") not in [None, False, ""]:
            continue

        trial_results = trial.get("results", {})
        if not trial_results:
            continue

        layer_key = str(layer)
        for cp_str, layer_data in trial_results.items():
            try:
                cp = int(cp_str)
            except ValueError:
                continue

            if layer_key not in layer_data:
                continue

            metrics_data = layer_data[layer_key]
            if metric in metrics_data:
                val = metrics_data[metric]
                if val is not None:
                    if cp not in checkpoint_values:
                        checkpoint_values[cp] = []
                    checkpoint_values[cp].append(val)

    checkpoints = sorted(checkpoint_values.keys())
    means = [np.mean(checkpoint_values[cp]) if checkpoint_values[cp] else np.nan for cp in checkpoints]
    stds = [np.std(checkpoint_values[cp]) if checkpoint_values[cp] else np.nan for cp in checkpoints]

    return checkpoints, np.array(means), np.array(stds)


def plot_single_metric(results: dict, metric: str, metric_name: str, output_path: Path, layer: int = 27):
    """Plot a single metric with all conditions."""

    # Sort conditions
    conditions = sorted(results.keys(), key=get_sort_key)

    # Create color map - group by type
    n_conditions = len(conditions)

    fig, ax = plt.subplots(figsize=(14, 8))

    # Different colormaps for different condition types
    for cond in conditions:
        cps, means, stds = aggregate_by_checkpoint(results[cond], metric, layer)
        if len(cps) == 0:
            continue

        # Determine color based on condition type
        if "no_ambig" in cond:
            color = "darkgreen"
            linewidth = 3
            alpha = 1.0
        elif "full_ambig" in cond:
            color = "darkred"
            linewidth = 3
            alpha = 1.0
        elif "natural" in cond:
            if "books" in cond:
                color = "purple"
            elif "conversation" in cond:
                color = "orange"
            else:
                color = "brown"
            linewidth = 2.5
            alpha = 0.9
        elif "vocab" in cond:
            vocab_colors = {"vocab_15": "cyan", "vocab_50": "blue", "vocab_200": "navy", "vocab_1000": "black"}
            color = vocab_colors.get(cond, "gray")
            linewidth = 2.5
            alpha = 0.9
        elif "disambig" in cond:
            # Get percentage for color
            parts = cond.split("_")
            pct = 50
            for i, p in enumerate(parts):
                if p == "disambig" and i + 1 < len(parts):
                    pct_str = parts[i + 1]
                    if i + 2 < len(parts) and "pct" in parts[i + 2]:
                        pct_str = pct_str + "." + parts[i + 2].replace("pct", "")
                    pct = float(pct_str.replace("pct", ""))
            color = plt.cm.viridis(pct / 100)
            linewidth = 1.5
            alpha = 0.7
        else:
            color = "gray"
            linewidth = 1
            alpha = 0.5

        label = cond.replace("_", " ").replace("structured ", "")
        ax.plot(cps, means, label=label, color=color, linewidth=linewidth, alpha=alpha)

    ax.set_xlabel("Context Length (tokens)", fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f"{metric_name} Over Context (Layer {layer})", fontsize=14)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    # Legend outside
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results/collapse_10k_experiment")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--layer", type=int, default=27)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {results_dir}")
    results = load_results(results_dir)
    print(f"Loaded {len(results)} conditions")

    metrics = [
        ("avg_cos_sim", "Average Cosine Similarity (Anisotropy)"),
        ("spread", "Spread (Total Variance)"),
        ("effective_dim", "Effective Dimension (Participation Ratio)"),
        ("intrinsic_dim", "Intrinsic Dimension (Two-NN)"),
    ]

    print(f"\nGenerating plots for layer {args.layer}...")
    for metric_key, metric_name in metrics:
        plot_single_metric(
            results,
            metric_key,
            metric_name,
            output_dir / f"{metric_key}_all_conditions.png",
            layer=args.layer
        )

    print(f"\nAll plots saved to {output_dir}")
