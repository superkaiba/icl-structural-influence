#!/usr/bin/env python3
"""
Plot all collapse metrics with disambiguation conditions.
"""

import json
import sys
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


def get_disambig_pct(condition: str) -> float:
    """Extract disambiguation percentage from condition name."""
    if "no_ambig" in condition:
        return 0.0
    if "full_ambig" in condition:
        return 100.0
    if "disambig_" in condition:
        # Parse structured_disambig_X_Ypct format
        parts = condition.split("_")
        for i, p in enumerate(parts):
            if p == "disambig" and i + 1 < len(parts):
                pct_str = parts[i + 1]
                if i + 2 < len(parts) and "pct" in parts[i + 2]:
                    pct_str = pct_str + "." + parts[i + 2].replace("pct", "")
                return float(pct_str.replace("pct", ""))
    return -1


def aggregate_by_checkpoint(trials: list, metric: str, layer: int) -> tuple:
    """Aggregate metric values across trials for each checkpoint."""
    checkpoint_values = {}

    for trial in trials:
        # Skip trials with actual error messages (not just None)
        if trial.get("error") not in [None, False, ""]:
            continue

        trial_results = trial.get("results", {})
        if not trial_results:
            continue

        # Structure is results[checkpoint][layer] = metrics
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


def plot_all_metrics_by_condition(results: dict, output_dir: Path, layer: int = 27):
    """Plot all 4 collapse metrics for key conditions."""
    metrics = [
        ("avg_cos_sim", "Avg Cosine Similarity", "Higher = more collapsed"),
        ("spread", "Spread (Total Variance)", "Lower = more collapsed"),
        ("effective_dim", "Effective Dimension", "Lower = more collapsed"),
        ("intrinsic_dim", "Intrinsic Dimension (Two-NN)", "Lower = more collapsed"),
    ]

    # Key conditions to highlight
    key_conditions = [
        "structured_no_ambig",
        "structured_full_ambig",
        "natural_books",
        "natural_conversation",
        "vocab_15",
        "vocab_1000",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, len(key_conditions)))

    for idx, (metric_key, metric_name, metric_desc) in enumerate(metrics):
        ax = axes[idx]

        for cond_idx, condition in enumerate(key_conditions):
            if condition not in results:
                continue

            cps, means, stds = aggregate_by_checkpoint(results[condition], metric_key, layer)
            if len(cps) == 0:
                continue

            label = condition.replace("_", " ").replace("structured ", "").replace("natural ", "")
            ax.plot(cps, means, label=label, color=colors[cond_idx], linewidth=2)
            ax.fill_between(cps, means - stds, means + stds, alpha=0.2, color=colors[cond_idx])

        ax.set_xlabel("Context Length (tokens)")
        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name}\n({metric_desc})")
        ax.set_xscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Collapse Metrics Over Context (Layer {layer})", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "all_collapse_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'all_collapse_metrics.png'}")


def plot_disambiguation_comparison(results: dict, output_dir: Path, layer: int = 27):
    """Plot collapse metrics for different disambiguation points."""
    metrics = [
        ("avg_cos_sim", "Avg Cosine Similarity"),
        ("spread", "Spread"),
        ("effective_dim", "Effective Dimension"),
        ("intrinsic_dim", "Intrinsic Dimension"),
    ]

    # Get all disambiguation conditions
    disambig_conditions = []
    for cond in results.keys():
        if "disambig_" in cond and "structured" in cond:
            pct = get_disambig_pct(cond)
            if pct >= 0:
                disambig_conditions.append((pct, cond))

    # Sort by percentage
    disambig_conditions.sort(key=lambda x: x[0])

    # Select subset for clarity (every few)
    selected = []
    target_pcts = [0.5, 2, 5, 10, 25, 50, 75, 90, 99]
    for target in target_pcts:
        closest = min(disambig_conditions, key=lambda x: abs(x[0] - target))
        if closest not in selected:
            selected.append(closest)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors = plt.cm.viridis(np.linspace(0, 1, len(selected)))

    for idx, (metric_key, metric_name) in enumerate(metrics):
        ax = axes[idx]

        for cond_idx, (pct, condition) in enumerate(selected):
            if condition not in results:
                continue

            cps, means, stds = aggregate_by_checkpoint(results[condition], metric_key, layer)
            if len(cps) == 0:
                continue

            label = f"{pct}%"
            ax.plot(cps, means, label=label, color=colors[cond_idx], linewidth=2)
            ax.fill_between(cps, means - stds, means + stds, alpha=0.15, color=colors[cond_idx])

            # Add vertical line at disambiguation point
            disambig_pos = int(10000 * pct / 100)
            if disambig_pos > 0 and disambig_pos < max(cps):
                ax.axvline(x=disambig_pos, color=colors[cond_idx], linestyle='--', alpha=0.5, linewidth=1)

        ax.set_xlabel("Context Length (tokens)")
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.set_xscale("log")
        ax.legend(fontsize=8, title="Disambig @")
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Collapse Metrics by Disambiguation Point (Layer {layer})", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "collapse_by_disambiguation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'collapse_by_disambiguation.png'}")


def plot_disambiguation_by_layer(results: dict, output_dir: Path, metric: str = "avg_cos_sim"):
    """Plot a single metric across layers for different disambiguation points."""
    layers = [0, 7, 14, 21, 27]

    # Get disambiguation conditions
    disambig_conditions = []
    for cond in results.keys():
        if "disambig_" in cond and "structured" in cond:
            pct = get_disambig_pct(cond)
            if pct >= 0:
                disambig_conditions.append((pct, cond))

    disambig_conditions.sort(key=lambda x: x[0])

    # Select subset
    target_pcts = [0.5, 5, 25, 50, 90]
    selected = []
    for target in target_pcts:
        closest = min(disambig_conditions, key=lambda x: abs(x[0] - target))
        if closest not in selected:
            selected.append(closest)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    colors = plt.cm.viridis(np.linspace(0, 1, len(selected)))

    for layer_idx, layer in enumerate(layers):
        ax = axes[layer_idx]

        for cond_idx, (pct, condition) in enumerate(selected):
            if condition not in results:
                continue

            cps, means, stds = aggregate_by_checkpoint(results[condition], metric, layer)
            if len(cps) == 0:
                continue

            label = f"{pct}%"
            ax.plot(cps, means, label=label, color=colors[cond_idx], linewidth=2)
            ax.fill_between(cps, means - stds, means + stds, alpha=0.15, color=colors[cond_idx])

            # Vertical line at disambiguation
            disambig_pos = int(10000 * pct / 100)
            if disambig_pos > 0 and disambig_pos < max(cps):
                ax.axvline(x=disambig_pos, color=colors[cond_idx], linestyle='--', alpha=0.5, linewidth=1)

        ax.set_xlabel("Context Length")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"Layer {layer}")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)

    # Legend in last subplot
    axes[-1].legend(handles=axes[0].get_legend_handles_labels()[0],
                    labels=axes[0].get_legend_handles_labels()[1],
                    loc='center', fontsize=10, title="Disambig @")
    axes[-1].axis('off')

    metric_title = metric.replace("_", " ").title()
    plt.suptitle(f"{metric_title} by Layer and Disambiguation Point", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f"{metric}_by_layer_disambig.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / f'{metric}_by_layer_disambig.png'}")


def plot_final_collapse_vs_disambiguation(results: dict, output_dir: Path, layer: int = 27):
    """Plot final collapse value vs disambiguation percentage."""
    metrics = [
        ("avg_cos_sim", "Avg Cosine Similarity"),
        ("spread", "Spread"),
        ("effective_dim", "Effective Dimension"),
        ("intrinsic_dim", "Intrinsic Dimension"),
    ]

    # Get all disambiguation conditions
    disambig_data = []
    for cond in results.keys():
        if "disambig_" in cond and "structured" in cond:
            pct = get_disambig_pct(cond)
            if pct >= 0:
                disambig_data.append((pct, cond))

    disambig_data.sort(key=lambda x: x[0])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (metric_key, metric_name) in enumerate(metrics):
        ax = axes[idx]

        pcts = []
        final_means = []
        final_stds = []

        for pct, condition in disambig_data:
            if condition not in results:
                continue

            cps, means, stds = aggregate_by_checkpoint(results[condition], metric_key, layer)
            if len(cps) == 0 or np.all(np.isnan(means)):
                continue

            # Get final (max checkpoint) value
            final_idx = -1
            while np.isnan(means[final_idx]) and abs(final_idx) < len(means):
                final_idx -= 1

            pcts.append(pct)
            final_means.append(means[final_idx])
            final_stds.append(stds[final_idx])

        ax.errorbar(pcts, final_means, yerr=final_stds, marker='o', capsize=3, linewidth=2)
        ax.set_xlabel("Disambiguation Point (%)")
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Final Collapse Value vs Disambiguation Point (Layer {layer})", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "final_collapse_vs_disambiguation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'final_collapse_vs_disambiguation.png'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results/collapse_10k_experiment")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {results_dir}")
    results = load_results(results_dir)
    print(f"Loaded {len(results)} conditions")

    print("\nGenerating plots...")
    plot_all_metrics_by_condition(results, output_dir)
    plot_disambiguation_comparison(results, output_dir)
    plot_disambiguation_by_layer(results, output_dir, metric="avg_cos_sim")
    plot_disambiguation_by_layer(results, output_dir, metric="effective_dim")
    plot_final_collapse_vs_disambiguation(results, output_dir)

    print(f"\nAll plots saved to {output_dir}")
