#!/usr/bin/env python3
"""
Cross-model comparison plots for the Safety Collapse Sweep.

Loads judged results from multiple sub-experiments and generates comparison plots.

Usage:
    PYTHONPATH=. python experiments/plotting/plot_safety_sweep_comparison.py
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


BASE_DIR = Path("results/safety_collapse_sweep")

# ── Color palettes ────────────────────────────────────────────────────────

SIZE_COLORS = {
    "0.5B": "#fee08b",
    "3B": "#fdae61",
    "7B": "#f46d43",
    "14B": "#d73027",
    "72B": "#a50026",
}

ARCH_COLORS = {
    "Qwen 7B": "#e74c3c",
    "Llama 8B": "#3498db",
    "Qwen 72B": "#c0392b",
    "Llama 70B": "#2980b9",
}

VOCAB_COLORS = {
    15: "#d73027",
    50: "#fc8d59",
    200: "#fee08b",
    1000: "#91cf60",
}


# ── Data Loading ──────────────────────────────────────────────────────────

def load_judged_results(results_dir: Path) -> tuple[dict, list[dict]]:
    """Load judge results and raw judged data from a results directory."""
    judge_path = results_dir / "judge" / "judge_results.json"
    judged_path = results_dir / "judge" / "all_results_judged.json"

    judge = None
    raw = []

    if judge_path.exists():
        with open(judge_path) as f:
            judge = json.load(f)
    if judged_path.exists():
        with open(judged_path) as f:
            raw = json.load(f)

    return judge, raw


def load_sweep_data(sweep_name: str) -> dict[str, tuple[dict, list[dict]]]:
    """Load all sub-experiments for a sweep dimension."""
    sweep_dir = BASE_DIR / sweep_name
    if not sweep_dir.exists():
        return {}

    data = {}
    for subdir in sorted(sweep_dir.iterdir()):
        if subdir.is_dir():
            judge, raw = load_judged_results(subdir)
            if judge and raw:
                data[subdir.name] = (judge, raw)

    return data


def load_existing_7b_data() -> tuple[dict, list[dict]]:
    """Load existing Qwen2.5-7B data from the original + 50K experiments."""
    raw_all = []

    for results_dir in [Path("results/safety_collapse"), Path("results/safety_collapse_50k")]:
        judged_path = results_dir / "judge" / "all_results_judged.json"
        if judged_path.exists():
            with open(judged_path) as f:
                raw_all.extend(json.load(f))

    return raw_all


def compute_safety_stats(raw: list[dict], condition: str = "structured_walk_raw"):
    """Compute refusal/compliance/incoherent fractions for baseline-refused prompts."""
    baseline_refused = [
        r for r in raw
        if r.get("baseline_refused") and r.get("condition") == condition
    ]

    lengths = sorted(set(r["context_length"] for r in baseline_refused))
    stats = {}

    for l in lengths:
        trials = [r for r in baseline_refused if r["context_length"] == l]
        n = len(trials)
        if n == 0:
            continue

        n_refused = sum(1 for r in trials if r.get("judge_classification") in ("full_refusal", "partial_refusal"))
        n_compliance = sum(1 for r in trials if r.get("judge_classification") == "compliance")
        n_incoherent = sum(1 for r in trials if r.get("judge_classification") in ("incoherent", "parse_error"))

        stats[l] = {
            "refusal": n_refused / n,
            "compliance": n_compliance / n,
            "incoherent": n_incoherent / n,
            "n": n,
        }

    return stats


# ── Plot 1: Model Size Scaling ────────────────────────────────────────────

def plot_model_size_scaling(output_dir: Path):
    """Refusal rate vs context length, one line per model size."""
    data = load_sweep_data("model_size")
    existing_7b = load_existing_7b_data()

    fig, ax = plt.subplots(figsize=(12, 7))

    # Add existing 7B data
    if existing_7b:
        stats = compute_safety_stats(existing_7b)
        if stats:
            lengths = sorted(stats.keys())
            rates = [stats[l]["refusal"] for l in lengths]
            ax.plot(lengths, rates, color=SIZE_COLORS["7B"], marker="o",
                    linewidth=2.5, markersize=7, label="7B (Qwen2.5)")

    # Add sweep data
    size_order = {"qwen25_0.5b": "0.5B", "qwen25_3b": "3B",
                  "qwen25_14b": "14B", "qwen25_72b": "72B"}

    for short_name, size_label in size_order.items():
        if short_name not in data:
            continue
        _, raw = data[short_name]
        stats = compute_safety_stats(raw)
        if not stats:
            continue

        color = SIZE_COLORS.get(size_label, "#666")
        lengths = sorted(stats.keys())
        rates = [stats[l]["refusal"] for l in lengths]
        ax.plot(lengths, rates, color=color, marker="o",
                linewidth=2.5, markersize=7, label=f"{size_label} (Qwen2.5)")

    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Context Length (tokens)", fontsize=13)
    ax.set_ylabel("Refusal Rate (judge, baseline-refused)", fontsize=13)
    ax.set_title("Safety Resilience by Model Size (structured_walk_raw)", fontsize=15)
    ax.legend(fontsize=11)
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("symlog", linthresh=1000)

    plt.tight_layout()
    fig.savefig(output_dir / "model_size_scaling.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: model_size_scaling.png")


# ── Plot 2: Phase Transition Heatmap ──────────────────────────────────────

def plot_phase_transition_heatmap(output_dir: Path):
    """X=context length, Y=model size, color=compliance rate."""
    data = load_sweep_data("model_size")
    existing_7b = load_existing_7b_data()

    size_order = ["0.5B", "3B", "7B", "14B", "72B"]
    all_stats = {}

    # Existing 7B
    if existing_7b:
        all_stats["7B"] = compute_safety_stats(existing_7b)

    size_map = {"qwen25_0.5b": "0.5B", "qwen25_3b": "3B",
                "qwen25_14b": "14B", "qwen25_72b": "72B"}
    for short_name, size_label in size_map.items():
        if short_name in data:
            _, raw = data[short_name]
            all_stats[size_label] = compute_safety_stats(raw)

    available_sizes = [s for s in size_order if s in all_stats]
    if len(available_sizes) < 2:
        print("  SKIP: phase_transition_heatmap (need >= 2 model sizes)")
        return

    all_lengths = sorted(set(
        l for stats in all_stats.values() for l in stats.keys()
    ))

    matrix = np.full((len(available_sizes), len(all_lengths)), np.nan)
    for i, size in enumerate(available_sizes):
        for j, length in enumerate(all_lengths):
            if length in all_stats[size]:
                matrix[i, j] = all_stats[size][length]["compliance"]

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(all_lengths)))
    ax.set_xticklabels([f"{l//1000}K" if l >= 1000 else str(l) for l in all_lengths],
                        fontsize=9)
    ax.set_yticks(range(len(available_sizes)))
    ax.set_yticklabels(available_sizes, fontsize=11)

    for i in range(len(available_sizes)):
        for j in range(len(all_lengths)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if val > 0.4 else "black"
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                        fontsize=8, color=color)

    ax.set_xlabel("Context Length", fontsize=12)
    ax.set_ylabel("Model Size", fontsize=12)
    ax.set_title("Genuine Compliance Rate (Danger Zone) by Model Size", fontsize=14)
    plt.colorbar(im, ax=ax, label="Compliance Rate")

    plt.tight_layout()
    fig.savefig(output_dir / "phase_transition_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: phase_transition_heatmap.png")


# ── Plot 3: 3-Color Stacked Bars by Model Size ───────────────────────────

def plot_stacked_by_model_size(output_dir: Path):
    """Multi-panel stacked bars: refusal/compliance/incoherent per model."""
    data = load_sweep_data("model_size")
    existing_7b = load_existing_7b_data()

    all_stats = {}
    if existing_7b:
        all_stats["7B"] = compute_safety_stats(existing_7b)

    size_map = {"qwen25_0.5b": "0.5B", "qwen25_3b": "3B",
                "qwen25_14b": "14B", "qwen25_72b": "72B"}
    for short_name, size_label in size_map.items():
        if short_name in data:
            _, raw = data[short_name]
            all_stats[size_label] = compute_safety_stats(raw)

    size_order = ["0.5B", "3B", "7B", "14B", "72B"]
    available = [s for s in size_order if s in all_stats]
    if not available:
        print("  SKIP: stacked_by_model_size (no data)")
        return

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 7), sharey=True)
    if len(available) == 1:
        axes = [axes]

    for ax, size in zip(axes, available):
        stats = all_stats[size]
        lengths = sorted(stats.keys())

        ref = np.array([stats[l]["refusal"] for l in lengths])
        comp = np.array([stats[l]["compliance"] for l in lengths])
        inc = np.array([stats[l]["incoherent"] for l in lengths])

        x = np.arange(len(lengths))
        w = 0.65

        ax.bar(x, ref, w, label="Refusal (safe)", color="#2ecc71")
        ax.bar(x, comp, w, bottom=ref, label="Compliance (unsafe)", color="#e74c3c")
        ax.bar(x, inc, w, bottom=ref + comp, label="Incoherent", color="#95a5a6")

        ax.set_xticks(x)
        ax.set_xticklabels([f"{l//1000}K" if l >= 1000 else str(l) for l in lengths],
                            fontsize=8, rotation=45)
        ax.set_xlabel("Context Length", fontsize=11)
        ax.set_title(f"Qwen2.5-{size}", fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.2, axis="y")

    axes[0].set_ylabel("Proportion of Baseline-Refused", fontsize=12)
    axes[len(available) // 2].legend(fontsize=10, loc="upper center",
                                       bbox_to_anchor=(0.5, -0.15), ncol=3)

    fig.suptitle("Safety Outcome by Model Size (structured_walk_raw, LLM Judge)",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "stacked_by_model_size.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: stacked_by_model_size.png")


# ── Plot 4: Architecture Comparison ───────────────────────────────────────

def plot_architecture_comparison(output_dir: Path):
    """Qwen vs Llama at matched sizes."""
    arch_data = load_sweep_data("architecture")
    size_data = load_sweep_data("model_size")
    existing_7b = load_existing_7b_data()

    pairs = {}

    # 7-8B pair
    if existing_7b:
        pairs["Qwen 7B"] = compute_safety_stats(existing_7b)
    if "llama31_8b" in arch_data:
        _, raw = arch_data["llama31_8b"]
        pairs["Llama 8B"] = compute_safety_stats(raw)

    # 70-72B pair
    if "qwen25_72b" in size_data:
        _, raw = size_data["qwen25_72b"]
        pairs["Qwen 72B"] = compute_safety_stats(raw)
    if "llama33_70b" in arch_data:
        _, raw = arch_data["llama33_70b"]
        pairs["Llama 70B"] = compute_safety_stats(raw)

    if len(pairs) < 2:
        print("  SKIP: architecture_comparison (need >= 2 models)")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    for label, stats in pairs.items():
        color = ARCH_COLORS.get(label, "#666")
        ls = "-" if "Qwen" in label else "--"
        marker = "o" if "7" in label or "8" in label else "s"

        lengths = sorted(stats.keys())
        rates = [stats[l]["refusal"] for l in lengths]
        ax.plot(lengths, rates, color=color, linestyle=ls, marker=marker,
                linewidth=2.5, markersize=7, label=label)

    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Context Length (tokens)", fontsize=13)
    ax.set_ylabel("Refusal Rate (judge, baseline-refused)", fontsize=13)
    ax.set_title("Architecture Comparison: Qwen vs Llama", fontsize=15)
    ax.legend(fontsize=11)
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("symlog", linthresh=1000)

    plt.tight_layout()
    fig.savefig(output_dir / "architecture_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: architecture_comparison.png")


# ── Plot 5: Vocab Size Effect ─────────────────────────────────────────────

def plot_vocab_size_effect(output_dir: Path):
    """Refusal rate vs context length, one line per vocab size."""
    vocab_data = load_sweep_data("vocab_size")
    existing_7b = load_existing_7b_data()

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Left: refusal rate
    ax = axes[0]

    # Existing 7B = vocab 15
    if existing_7b:
        stats = compute_safety_stats(existing_7b)
        if stats:
            lengths = sorted(stats.keys())
            rates = [stats[l]["refusal"] for l in lengths]
            ax.plot(lengths, rates, color=VOCAB_COLORS[15], marker="o",
                    linewidth=2.5, markersize=7, label="vocab=15")

    vocab_map = {"vocab50": 50, "vocab200": 200, "vocab1000": 1000}
    for short_name, vocab in vocab_map.items():
        if short_name not in vocab_data:
            continue
        _, raw = vocab_data[short_name]
        stats = compute_safety_stats(raw)
        if not stats:
            continue

        lengths = sorted(stats.keys())
        rates = [stats[l]["refusal"] for l in lengths]
        ax.plot(lengths, rates, color=VOCAB_COLORS.get(vocab, "#666"),
                marker="o", linewidth=2.5, markersize=7, label=f"vocab={vocab}")

    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Context Length (tokens)", fontsize=13)
    ax.set_ylabel("Refusal Rate (judge, baseline-refused)", fontsize=13)
    ax.set_title("Safety vs Vocab Size", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("symlog", linthresh=1000)

    # Right: collapse cos_sim
    ax = axes[1]

    if existing_7b:
        sw_data = [r for r in existing_7b if r.get("condition") == "structured_walk_raw"]
        lengths_set = sorted(set(r["context_length"] for r in sw_data))
        cos_sims = []
        for l in lengths_set:
            trials = [r for r in sw_data if r["context_length"] == l]
            sims = []
            for r in trials:
                cm = r.get("collapse_metrics", {})
                if cm:
                    last_key = sorted(cm.keys(), key=int)[-1]
                    if cm[last_key] and "avg_cos_sim" in cm[last_key]:
                        sims.append(cm[last_key]["avg_cos_sim"])
            if sims:
                cos_sims.append((l, np.mean(sims)))

        if cos_sims:
            vl, vc = zip(*cos_sims)
            ax.plot(vl, vc, color=VOCAB_COLORS[15], marker="o",
                    linewidth=2.5, markersize=7, label="vocab=15")

    for short_name, vocab in vocab_map.items():
        if short_name not in vocab_data:
            continue
        _, raw = vocab_data[short_name]
        sw_data = [r for r in raw if r.get("condition") == "structured_walk_raw"]
        lengths_set = sorted(set(r["context_length"] for r in sw_data))

        cos_sims = []
        for l in lengths_set:
            trials = [r for r in sw_data if r["context_length"] == l]
            sims = []
            for r in trials:
                cm = r.get("collapse_metrics", {})
                if cm:
                    last_key = sorted(cm.keys(), key=int)[-1]
                    if cm[last_key] and "avg_cos_sim" in cm[last_key]:
                        sims.append(cm[last_key]["avg_cos_sim"])
            if sims:
                cos_sims.append((l, np.mean(sims)))

        if cos_sims:
            vl, vc = zip(*cos_sims)
            ax.plot(vl, vc, color=VOCAB_COLORS.get(vocab, "#666"),
                    marker="o", linewidth=2.5, markersize=7, label=f"vocab={vocab}")

    ax.set_xlabel("Context Length (tokens)", fontsize=13)
    ax.set_ylabel("Cosine Similarity (collapse)", fontsize=13)
    ax.set_title("Collapse vs Vocab Size", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("symlog", linthresh=1000)

    fig.suptitle("Effect of Vocabulary Size on Safety and Collapse (Qwen2.5-7B)",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "vocab_size_effect.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: vocab_size_effect.png")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot safety sweep comparison results")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else BASE_DIR / "comparison_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating comparison plots in {output_dir}")
    print("-" * 50)

    plot_model_size_scaling(output_dir)
    plot_phase_transition_heatmap(output_dir)
    plot_stacked_by_model_size(output_dir)
    plot_architecture_comparison(output_dir)
    plot_vocab_size_effect(output_dir)

    print("\nAll comparison plots generated.")


if __name__ == "__main__":
    main()
