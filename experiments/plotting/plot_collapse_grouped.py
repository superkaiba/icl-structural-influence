#!/usr/bin/env python3
"""
Plot collapse metrics with conditions grouped for comparison:
1. Disambiguation points (0.5% to 99%)
2. No ambiguity vs Full ambiguity
3. Vocabulary sizes (15, 50, 200, 1000)
4. Data types (structured vs natural language)
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


def plot_disambiguation_comparison(results: dict, metric: str, metric_name: str, output_path: Path, layer: int = 27):
    """Plot disambiguation conditions comparison."""

    # Get all disambiguation conditions
    disambig_conditions = []
    for cond in results.keys():
        if "disambig_" in cond and "structured" in cond:
            pct = get_disambig_pct(cond)
            if pct >= 0:
                disambig_conditions.append((pct, cond))

    disambig_conditions.sort(key=lambda x: x[0])

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = plt.cm.viridis(np.linspace(0, 1, len(disambig_conditions)))

    for idx, (pct, condition) in enumerate(disambig_conditions):
        cps, means, stds = aggregate_by_checkpoint(results[condition], metric, layer)
        if len(cps) == 0:
            continue

        label = f"{pct}%"
        ax.plot(cps, means, label=label, color=colors[idx], linewidth=1.5)
        ax.fill_between(cps, means - stds, means + stds, alpha=0.1, color=colors[idx])

        # Vertical line at disambiguation point
        disambig_pos = int(10000 * pct / 100)
        if 0 < disambig_pos < max(cps):
            ax.axvline(x=disambig_pos, color=colors[idx], linestyle='--', alpha=0.4, linewidth=1)

    ax.set_xlabel("Context Length (tokens)", fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f"{metric_name} by Disambiguation Point (Layer {layer})", fontsize=14)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, title="Disambig @")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_ambiguity_comparison(results: dict, metric: str, metric_name: str, output_path: Path, layer: int = 27):
    """Plot no ambiguity vs full ambiguity comparison."""

    conditions = {
        "structured_no_ambig": ("No Ambiguity (H1 only)", "green"),
        "structured_full_ambig": ("Full Ambiguity (H1 & H2 valid)", "red"),
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for condition, (label, color) in conditions.items():
        if condition not in results:
            continue

        cps, means, stds = aggregate_by_checkpoint(results[condition], metric, layer)
        if len(cps) == 0:
            continue

        ax.plot(cps, means, label=label, color=color, linewidth=2.5)
        ax.fill_between(cps, means - stds, means + stds, alpha=0.2, color=color)

    ax.set_xlabel("Context Length (tokens)", fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f"{metric_name}: No Ambiguity vs Full Ambiguity (Layer {layer})", fontsize=14)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_vocab_size_comparison(results: dict, metric: str, metric_name: str, output_path: Path, layer: int = 27):
    """Plot vocabulary size comparison."""

    conditions = {
        "vocab_15": ("Vocab 15", "cyan"),
        "vocab_50": ("Vocab 50", "blue"),
        "vocab_200": ("Vocab 200", "darkblue"),
        "vocab_1000": ("Vocab 1000", "black"),
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for condition, (label, color) in conditions.items():
        if condition not in results:
            continue

        cps, means, stds = aggregate_by_checkpoint(results[condition], metric, layer)
        if len(cps) == 0:
            continue

        ax.plot(cps, means, label=label, color=color, linewidth=2.5)
        ax.fill_between(cps, means - stds, means + stds, alpha=0.2, color=color)

    ax.set_xlabel("Context Length (tokens)", fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f"{metric_name} by Vocabulary Size (Layer {layer})", fontsize=14)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_data_type_comparison(results: dict, metric: str, metric_name: str, output_path: Path, layer: int = 27):
    """Plot structured vs natural language comparison."""

    conditions = {
        "structured_no_ambig": ("Structured (No Ambig)", "green"),
        "structured_full_ambig": ("Structured (Full Ambig)", "red"),
        "natural_books": ("Natural: Books", "purple"),
        "natural_conversation": ("Natural: Conversation (OpenAssistant)", "orange"),
        "natural_wildchat": ("Natural: Long Conversations (WildChat)", "magenta"),
        "natural_wikipedia": ("Natural: Wikipedia", "brown"),
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for condition, (label, color) in conditions.items():
        if condition not in results:
            continue

        cps, means, stds = aggregate_by_checkpoint(results[condition], metric, layer)
        if len(cps) == 0:
            continue

        ax.plot(cps, means, label=label, color=color, linewidth=2.5)
        ax.fill_between(cps, means - stds, means + stds, alpha=0.2, color=color)

    ax.set_xlabel("Context Length (tokens)", fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f"{metric_name}: Structured vs Natural Language (Layer {layer})", fontsize=14)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

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
        ("avg_cos_sim", "Average Cosine Similarity"),
        ("spread", "Spread (Total Variance)"),
        ("effective_dim", "Effective Dimension"),
        ("intrinsic_dim", "Intrinsic Dimension"),
    ]

    print(f"\nGenerating grouped plots for layer {args.layer}...")

    for metric_key, metric_name in metrics:
        print(f"\n=== {metric_name} ===")

        # 1. Disambiguation comparison
        plot_disambiguation_comparison(
            results, metric_key, metric_name,
            output_dir / f"{metric_key}_disambiguation.png",
            layer=args.layer
        )

        # 2. Ambiguity comparison
        plot_ambiguity_comparison(
            results, metric_key, metric_name,
            output_dir / f"{metric_key}_ambiguity.png",
            layer=args.layer
        )

        # 3. Vocabulary size comparison
        plot_vocab_size_comparison(
            results, metric_key, metric_name,
            output_dir / f"{metric_key}_vocab_size.png",
            layer=args.layer
        )

        # 4. Data type comparison
        plot_data_type_comparison(
            results, metric_key, metric_name,
            output_dir / f"{metric_key}_data_type.png",
            layer=args.layer
        )

    print(f"\nAll plots saved to {output_dir}")
