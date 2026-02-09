#!/usr/bin/env python3
"""
Plot collapse metrics with all layers as subplots in the same figure.
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


def get_disambig_pct(condition: str) -> float:
    """Extract disambiguation percentage from condition name."""
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


def plot_all_layers_structured(results: dict, metric: str, metric_name: str, output_path: Path, layers: list):
    """Plot all structured conditions (no ambig, full ambig, disambiguation) with all layers as subplots."""

    # Get all disambiguation conditions
    disambig_conditions = []
    for cond in results.keys():
        if "disambig_" in cond and "structured" in cond:
            pct = get_disambig_pct(cond)
            if pct >= 0:
                disambig_conditions.append((pct, cond))
    disambig_conditions.sort(key=lambda x: x[0])

    n_layers = len(layers)
    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 5), sharey=True)

    # Colors for disambiguation conditions (viridis gradient)
    disambig_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(disambig_conditions)))

    for layer_idx, layer in enumerate(layers):
        ax = axes[layer_idx]

        # Plot no ambiguity (green, thick)
        if "structured_no_ambig" in results:
            cps, means, stds = aggregate_by_checkpoint(results["structured_no_ambig"], metric, layer)
            if len(cps) > 0:
                label = "No Ambig" if layer_idx == n_layers - 1 else None
                ax.plot(cps, means, color="green", linewidth=3, label=label)
                ax.fill_between(cps, means - stds, means + stds, alpha=0.2, color="green")

        # Plot full ambiguity (red, thick)
        if "structured_full_ambig" in results:
            cps, means, stds = aggregate_by_checkpoint(results["structured_full_ambig"], metric, layer)
            if len(cps) > 0:
                label = "Full Ambig" if layer_idx == n_layers - 1 else None
                ax.plot(cps, means, color="red", linewidth=3, label=label)
                ax.fill_between(cps, means - stds, means + stds, alpha=0.2, color="red")

        # Plot disambiguation conditions
        for idx, (pct, condition) in enumerate(disambig_conditions):
            cps, means, stds = aggregate_by_checkpoint(results[condition], metric, layer)
            if len(cps) == 0:
                continue

            label = f"Disambig {pct}%" if layer_idx == n_layers - 1 else None
            ax.plot(cps, means, color=disambig_colors[idx], linewidth=1.2, label=label)
            ax.fill_between(cps, means - stds, means + stds, alpha=0.08, color=disambig_colors[idx])

            # Vertical line at disambiguation point
            disambig_pos = int(10000 * pct / 100)
            if 0 < disambig_pos < max(cps):
                ax.axvline(x=disambig_pos, color=disambig_colors[idx], linestyle='--', alpha=0.3, linewidth=0.8)

        ax.set_xlabel("Context Length")
        ax.set_title(f"Layer {layer}")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(metric_name)
    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=6, title="Condition")

    plt.suptitle(f"{metric_name}: All Structured Conditions", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")




def plot_all_layers_vocab_size(results: dict, metric: str, metric_name: str, output_path: Path, layers: list):
    """Plot vocabulary size comparison with all layers as subplots."""

    conditions = {
        "vocab_15": ("Vocab 15", "cyan"),
        "vocab_50": ("Vocab 50", "blue"),
        "vocab_200": ("Vocab 200", "darkblue"),
        "vocab_1000": ("Vocab 1000", "black"),
    }

    n_layers = len(layers)
    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 5), sharey=True)

    for layer_idx, layer in enumerate(layers):
        ax = axes[layer_idx]

        for condition, (label, color) in conditions.items():
            if condition not in results:
                continue

            cps, means, stds = aggregate_by_checkpoint(results[condition], metric, layer)
            if len(cps) == 0:
                continue

            lbl = label if layer_idx == n_layers - 1 else None
            ax.plot(cps, means, label=lbl, color=color, linewidth=2.5)
            ax.fill_between(cps, means - stds, means + stds, alpha=0.2, color=color)

        ax.set_xlabel("Context Length")
        ax.set_title(f"Layer {layer}")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(metric_name)
    axes[-1].legend(loc='best', fontsize=10)

    plt.suptitle(f"{metric_name} by Vocabulary Size", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_all_layers_data_type(results: dict, metric: str, metric_name: str, output_path: Path, layers: list):
    """Plot structured vs natural language with all layers as subplots."""

    conditions = {
        "structured_no_ambig": ("Structured (No Ambig)", "green"),
        "structured_full_ambig": ("Structured (Full Ambig)", "red"),
        "natural_books": ("Natural: Books", "purple"),
        "natural_conversation": ("Natural: Conversation", "orange"),
    }

    n_layers = len(layers)
    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 5), sharey=True)

    for layer_idx, layer in enumerate(layers):
        ax = axes[layer_idx]

        for condition, (label, color) in conditions.items():
            if condition not in results:
                continue

            cps, means, stds = aggregate_by_checkpoint(results[condition], metric, layer)
            if len(cps) == 0:
                continue

            lbl = label if layer_idx == n_layers - 1 else None
            ax.plot(cps, means, label=lbl, color=color, linewidth=2.5)
            ax.fill_between(cps, means - stds, means + stds, alpha=0.2, color=color)

        ax.set_xlabel("Context Length")
        ax.set_title(f"Layer {layer}")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(metric_name)
    axes[-1].legend(loc='best', fontsize=9)

    plt.suptitle(f"{metric_name}: Structured vs Natural Language", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


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

    layers = [0, 7, 14, 21, 27]

    metrics = [
        ("avg_cos_sim", "Average Cosine Similarity"),
        ("spread", "Spread (Total Variance)"),
        ("effective_dim", "Effective Dimension"),
        ("intrinsic_dim", "Intrinsic Dimension"),
    ]

    print(f"\nGenerating all-layers plots...")

    for metric_key, metric_name in metrics:
        print(f"\n=== {metric_name} ===")

        plot_all_layers_structured(
            results, metric_key, metric_name,
            output_dir / f"{metric_key}_all_layers_structured.png",
            layers
        )

        plot_all_layers_vocab_size(
            results, metric_key, metric_name,
            output_dir / f"{metric_key}_all_layers_vocab_size.png",
            layers
        )

        plot_all_layers_data_type(
            results, metric_key, metric_name,
            output_dir / f"{metric_key}_all_layers_data_type.png",
            layers
        )

    print(f"\nAll plots saved to {output_dir}")
