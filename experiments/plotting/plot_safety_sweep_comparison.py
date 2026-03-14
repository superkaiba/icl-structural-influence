#!/usr/bin/env python3
"""
Cross-model comparison plots for the Safety Collapse Sweep.

Generates 3-band stacked bar plots (refusal/incoherent/compliance) and a
compliance-rate heatmap.

Usage:
    PYTHONPATH=. python experiments/plotting/plot_safety_sweep_comparison.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


BASE_DIR = Path("results/safety_collapse_sweep")

# ── Colors ───────────────────────────────────────────────────────────────
COLOR_REFUSAL = "#2ecc71"
COLOR_COMPLIANCE = "#e74c3c"
COLOR_INCOHERENT = "#95a5a6"

LEGEND_ELEMENTS = [
    Patch(facecolor=COLOR_REFUSAL, label="Refusal (safe)"),
    Patch(facecolor=COLOR_INCOHERENT, label="Incoherent"),
    Patch(facecolor=COLOR_COMPLIANCE, label="Compliance (unsafe)"),
]


# ── Data Loading ─────────────────────────────────────────────────────────

def load_judged(path: Path) -> list[dict]:
    """Load all_results_judged.json from a directory."""
    judged_path = path / "judge" / "all_results_judged.json"
    if judged_path.exists():
        with open(judged_path) as f:
            return json.load(f)
    return []


def compute_safety_stats(
    raw: list[dict],
    condition: str = "structured_walk_raw",
) -> dict[int, dict]:
    """Compute refusal/compliance/incoherent fractions for baseline-refused prompts.

    Returns {context_length: {"refusal": float, "compliance": float,
                               "incoherent": float, "n": int}}
    """
    baseline_refused = [
        r for r in raw
        if r.get("baseline_refused")
        and r.get("condition") == condition
    ]

    lengths = sorted(set(r["context_length"] for r in baseline_refused))
    stats = {}

    for length in lengths:
        trials = [r for r in baseline_refused if r["context_length"] == length]
        n = len(trials)
        if n == 0:
            continue

        n_refused = sum(
            1 for r in trials
            if r.get("judge_classification") in ("full_refusal", "partial_refusal")
        )
        n_compliance = sum(
            1 for r in trials if r.get("judge_classification") == "compliance"
        )
        n_incoherent = sum(
            1 for r in trials
            if r.get("judge_classification") in ("incoherent", "parse_error")
        )

        stats[length] = {
            "refusal": n_refused / n,
            "compliance": n_compliance / n,
            "incoherent": n_incoherent / n,
            "n": n,
        }

    return stats


def _format_length(length: int) -> str:
    """Pretty-print a context length."""
    if length >= 1000:
        return f"{length // 1000}K"
    return str(length)


# ── Stacked bar helper ───────────────────────────────────────────────────

def _draw_stacked_bars(ax, stats: dict[int, dict], title: str):
    """Draw a stacked bar chart on the given axes."""
    lengths = sorted(stats.keys())
    if not lengths:
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="gray")
        return

    ref = np.array([stats[l]["refusal"] for l in lengths])
    inc = np.array([stats[l]["incoherent"] for l in lengths])
    comp = np.array([stats[l]["compliance"] for l in lengths])

    x = np.arange(len(lengths))
    w = 0.65

    # Stack order: refusal (bottom), incoherent (middle), compliance (top)
    ax.bar(x, ref, w, color=COLOR_REFUSAL)
    ax.bar(x, inc, w, bottom=ref, color=COLOR_INCOHERENT)
    ax.bar(x, comp, w, bottom=ref + inc, color=COLOR_COMPLIANCE)

    ax.set_xticks(x)
    ax.set_xticklabels([_format_length(l) for l in lengths], fontsize=8, rotation=45)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.2, axis="y")


# ── Data loaders for each sweep ──────────────────────────────────────────

def _load_7b_data() -> list[dict]:
    """Load existing Qwen2.5-7B data (original experiment)."""
    path = Path("results/safety_collapse")
    return load_judged(path)


def _load_model_size_data() -> dict[str, list[dict]]:
    """Return {label: raw_data} for model size comparison."""
    result = {}

    # 0.5B, 3B, 14B from sweep
    size_map = {
        "0.5B": BASE_DIR / "model_size" / "qwen25_0.5b",
        "3B": BASE_DIR / "model_size" / "qwen25_3b",
        "14B": BASE_DIR / "model_size" / "qwen25_14b",
    }
    for label, path in size_map.items():
        raw = load_judged(path)
        if raw:
            result[label] = raw

    # 7B from original experiment
    raw_7b = _load_7b_data()
    if raw_7b:
        result["7B"] = raw_7b

    return result


def _load_architecture_data() -> dict[str, list[dict]]:
    """Return {label: raw_data} for architecture comparison."""
    result = {}

    raw_7b = _load_7b_data()
    if raw_7b:
        result["Qwen2.5-7B"] = raw_7b

    llama_path = BASE_DIR / "architecture" / "llama31_8b"
    raw_llama = load_judged(llama_path)
    if raw_llama:
        result["Llama-3.1-8B"] = raw_llama

    return result


def _load_vocab_data() -> dict[str, list[dict]]:
    """Return {label: raw_data} for vocab size comparison.

    Merges base and _100k runs for vocab 50/200/1000.
    vocab 15 comes from the original 7B experiment.
    """
    result = {}

    # vocab 15 = original 7B experiment
    raw_7b = _load_7b_data()
    if raw_7b:
        result["vocab=15"] = raw_7b

    for vocab in [50, 200, 1000]:
        merged = []
        base_path = BASE_DIR / "vocab_size" / f"vocab{vocab}"
        ext_path = BASE_DIR / "vocab_size" / f"vocab{vocab}_100k"
        merged.extend(load_judged(base_path))
        merged.extend(load_judged(ext_path))
        if merged:
            result[f"vocab={vocab}"] = merged

    return result


# ── Plot 1: Model Size Stacked ───────────────────────────────────────────

def plot_model_size_stacked(output_dir: Path):
    """One row per model size, stacked bars at each context length."""
    data = _load_model_size_data()
    size_order = ["0.5B", "3B", "7B", "14B"]
    available = [s for s in size_order if s in data]

    if not available:
        print("  SKIP: model_size_stacked (no data)")
        return

    n_rows = len(available)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3.5 * n_rows), squeeze=False)

    for i, size in enumerate(available):
        ax = axes[i, 0]
        stats = compute_safety_stats(data[size])
        family = "Qwen2.5"
        _draw_stacked_bars(ax, stats, f"{family}-{size}")
        if i < n_rows - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Context Length", fontsize=11)
        ax.set_ylabel("Proportion", fontsize=10)

    # Shared legend at bottom
    fig.legend(
        handles=LEGEND_ELEMENTS, loc="lower center",
        ncol=3, fontsize=11, frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "Safety Outcome by Model Size (structured_walk_raw, LLM Judge)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(output_dir / "model_size_stacked.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: model_size_stacked.png")


# ── Plot 2: Architecture Stacked ─────────────────────────────────────────

def plot_architecture_stacked(output_dir: Path):
    """Two rows: Qwen2.5-7B vs Llama-3.1-8B, stacked bars."""
    data = _load_architecture_data()
    arch_order = ["Qwen2.5-7B", "Llama-3.1-8B"]
    available = [a for a in arch_order if a in data]

    if len(available) < 2:
        print(f"  SKIP: architecture_stacked (need 2 models, have {len(available)})")
        return

    n_rows = len(available)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3.5 * n_rows), squeeze=False)

    for i, arch in enumerate(available):
        ax = axes[i, 0]
        stats = compute_safety_stats(data[arch])
        _draw_stacked_bars(ax, stats, arch)
        if i < n_rows - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Context Length", fontsize=11)
        ax.set_ylabel("Proportion", fontsize=10)

    fig.legend(
        handles=LEGEND_ELEMENTS, loc="lower center",
        ncol=3, fontsize=11, frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "Architecture Comparison: Safety Outcome (structured_walk_raw, LLM Judge)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(output_dir / "architecture_stacked.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: architecture_stacked.png")


# ── Plot 3: Vocab Size Stacked ───────────────────────────────────────────

def plot_vocab_size_stacked(output_dir: Path):
    """One row per vocab size, stacked bars."""
    data = _load_vocab_data()
    vocab_order = ["vocab=15", "vocab=50", "vocab=200", "vocab=1000"]
    available = [v for v in vocab_order if v in data]

    if not available:
        print("  SKIP: vocab_size_stacked (no data)")
        return

    n_rows = len(available)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3.5 * n_rows), squeeze=False)

    for i, vocab_label in enumerate(available):
        ax = axes[i, 0]
        stats = compute_safety_stats(data[vocab_label])
        _draw_stacked_bars(ax, stats, f"Qwen2.5-7B, {vocab_label}")
        if i < n_rows - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Context Length", fontsize=11)
        ax.set_ylabel("Proportion", fontsize=10)

    fig.legend(
        handles=LEGEND_ELEMENTS, loc="lower center",
        ncol=3, fontsize=11, frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "Vocab Size Effect on Safety (structured_walk_raw, LLM Judge)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(output_dir / "vocab_size_stacked.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: vocab_size_stacked.png")


# ── Plot 4: Phase Transition Heatmap ─────────────────────────────────────

def plot_phase_transition_heatmap(output_dir: Path):
    """X=context length, Y=model size. Color = genuine compliance rate."""
    data = _load_model_size_data()
    size_order = ["0.5B", "3B", "7B", "14B"]
    available_sizes = [s for s in size_order if s in data]

    if len(available_sizes) < 2:
        print("  SKIP: phase_transition_heatmap (need >= 2 model sizes)")
        return

    all_stats = {}
    for size in available_sizes:
        all_stats[size] = compute_safety_stats(data[size])

    # Collect all context lengths across all models
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
    ax.set_xticklabels(
        [_format_length(l) for l in all_lengths], fontsize=9,
    )
    ax.set_yticks(range(len(available_sizes)))
    ax.set_yticklabels(available_sizes, fontsize=11)

    # Annotate cells
    for i in range(len(available_sizes)):
        for j in range(len(all_lengths)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if val > 0.4 else "black"
                ax.text(
                    j, i, f"{val:.0%}", ha="center", va="center",
                    fontsize=8, color=color,
                )

    ax.set_xlabel("Context Length", fontsize=12)
    ax.set_ylabel("Model Size", fontsize=12)
    ax.set_title(
        "Genuine Compliance Rate (Danger Zone) by Model Size\n"
        "(structured_walk_raw, baseline-refused prompts only)",
        fontsize=13,
    )
    plt.colorbar(im, ax=ax, label="Compliance Rate")

    plt.tight_layout()
    fig.savefig(
        output_dir / "phase_transition_heatmap.png", dpi=150, bbox_inches="tight",
    )
    plt.close(fig)
    print("  Saved: phase_transition_heatmap.png")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot safety sweep comparison results (stacked bars)",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = (
        Path(args.output_dir) if args.output_dir
        else BASE_DIR / "comparison_plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating comparison plots in {output_dir}")
    print("-" * 50)

    plot_model_size_stacked(output_dir)
    plot_architecture_stacked(output_dir)
    plot_vocab_size_stacked(output_dir)
    plot_phase_transition_heatmap(output_dir)

    print("\nAll comparison plots generated.")


if __name__ == "__main__":
    main()
