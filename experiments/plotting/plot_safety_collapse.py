#!/usr/bin/env python3
"""
Plotting for Safety Collapse Experiment.

Generates 5 plot types:
1. Refusal rate vs context length (line per condition)
2. Refusal score (log-prob) vs context length
3. Collapse-safety scatter (cos_sim vs refusal_score)
4. Category heatmap (degradation rate by category x condition)
5. Benign control check (compliance on D/E prompts)

Usage:
    PYTHONPATH=. python experiments/plotting/plot_safety_collapse.py \
        --results-dir results/safety_collapse
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ── Style Constants ───────────────────────────────────────────────────────

# Colors for base context types
BASE_TYPE_COLORS = {
    "no_context": "#333333",
    "structured_walk": "#e74c3c",
    "natural_books": "#2ecc71",
    "template_small_vocab": "#f39c12",
    "repeated_token": "#9b59b6",
}

# Wrapping mode line styles
WRAP_STYLES = {
    "raw": "-",
    "chat": "--",
}

# Condition colors (type + wrapping)
def get_condition_style(condition: str) -> tuple[str, str, str]:
    """Return (color, linestyle, label) for a condition like 'structured_walk_raw'."""
    if condition == "no_context":
        return BASE_TYPE_COLORS["no_context"], "-", "No context (baseline)"

    parts = condition.rsplit("_", 1)
    if len(parts) == 2 and parts[1] in ("raw", "chat"):
        base_type, wrap_mode = parts
        color = BASE_TYPE_COLORS.get(base_type, "#666666")
        ls = WRAP_STYLES.get(wrap_mode, "-")
        label = f"{base_type.replace('_', ' ').title()} ({wrap_mode})"
        return color, ls, label
    else:
        return "#666666", "-", condition


CATEGORY_LABELS = {
    "A_direct_harmful": "A: Direct Harmful",
    "B_indirect_harmful": "B: Indirect Harmful",
    "C_borderline": "C: Borderline",
    "D_benign_control": "D: Benign Control",
    "E_safety_knowledge": "E: Safety Knowledge",
}

HARMFUL_CATEGORIES = ["A_direct_harmful", "B_indirect_harmful", "C_borderline"]
BENIGN_CATEGORIES = ["D_benign_control", "E_safety_knowledge"]


# ── Data Loading ──────────────────────────────────────────────────────────

def load_results(results_dir: Path) -> tuple[dict, list[dict]]:
    """Load aggregated results and raw results."""
    with open(results_dir / "results.json") as f:
        results = json.load(f)

    all_results_path = results_dir / "all_results.json"
    raw_results = []
    if all_results_path.exists():
        with open(all_results_path) as f:
            raw_results = json.load(f)

    return results, raw_results


# ── Plot 1: Refusal Rate vs Context Length ────────────────────────────────

def plot_refusal_vs_context_length(results: dict, output_dir: Path):
    """Refusal rate (among baseline-refused prompts) vs context length."""
    fig, ax = plt.subplots(figsize=(11, 7))
    agg = results["aggregated"]

    conditions = sorted(agg.keys())
    for condition in conditions:
        color, ls, label = get_condition_style(condition)
        lengths = sorted(int(k) for k in agg[condition].keys())
        refusal_rates = []

        for l in lengths:
            entry = agg[condition][str(l)]
            rate = entry.get("refusal_rate_among_baseline_refused")
            refusal_rates.append(rate if rate is not None else np.nan)

        valid = [(l, r) for l, r in zip(lengths, refusal_rates) if not np.isnan(r)]
        if not valid:
            continue
        vl, vr = zip(*valid)

        ax.plot(vl, vr, color=color, linestyle=ls, marker="o",
                linewidth=2, markersize=7, label=label)

    # Baseline reference
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5,
               label="Baseline (100% refusal)")

    ax.set_xlabel("Context Length (tokens)", fontsize=12)
    ax.set_ylabel("Refusal Rate (among baseline-refused)", fontsize=12)
    ax.set_title("Safety Refusal Rate vs Context Length", fontsize=14)
    ax.legend(fontsize=9, loc="lower left")
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("symlog", linthresh=100)

    plt.tight_layout()
    fig.savefig(output_dir / "refusal_vs_context_length.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  Saved: refusal_vs_context_length.png")


# ── Plot 2: Refusal Score vs Context Length ───────────────────────────────

def plot_refusal_score_vs_context_length(results: dict, output_dir: Path):
    """Mean refusal score (log-prob difference) vs context length."""
    fig, ax = plt.subplots(figsize=(11, 7))
    agg = results["aggregated"]

    conditions = sorted(agg.keys())
    for condition in conditions:
        color, ls, label = get_condition_style(condition)
        lengths = sorted(int(k) for k in agg[condition].keys())
        scores = []
        score_stds = []

        for l in lengths:
            entry = agg[condition][str(l)]
            scores.append(entry.get("mean_refusal_score", np.nan))
            score_stds.append(entry.get("std_refusal_score", 0))

        valid = [(l, s, e) for l, s, e in zip(lengths, scores, score_stds)
                 if not np.isnan(s)]
        if not valid:
            continue
        vl, vs, ve = zip(*valid)

        ax.errorbar(vl, vs, yerr=ve, color=color, linestyle=ls, marker="o",
                    linewidth=2, markersize=6, label=label, capsize=3, alpha=0.9)

    # Zero line: neutral
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5,
               label="Neutral (refusal = compliance)")

    ax.set_xlabel("Context Length (tokens)", fontsize=12)
    ax.set_ylabel("Refusal Score (log P(refuse) - log P(comply))", fontsize=12)
    ax.set_title("Refusal Confidence vs Context Length", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("symlog", linthresh=100)

    plt.tight_layout()
    fig.savefig(output_dir / "refusal_score_vs_context_length.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  Saved: refusal_score_vs_context_length.png")


# ── Plot 3: Collapse-Safety Scatter ───────────────────────────────────────

def plot_collapse_safety_scatter(raw_results: list[dict], results: dict,
                                  output_dir: Path):
    """Scatter of collapse cos_sim vs refusal_score, colored by condition."""
    fig, ax = plt.subplots(figsize=(10, 7))

    config = results.get("config", {})
    layers = config.get("layers", [])
    last_layer = str(layers[-1]) if layers else None

    if not last_layer:
        print("  SKIP: collapse_safety_scatter (no layers in config)")
        return

    conditions = sorted(set(r["condition"] for r in raw_results))
    for condition in conditions:
        color, _, label = get_condition_style(condition)
        trials = [r for r in raw_results if r["condition"] == condition]

        cos_sims = []
        refusal_scores = []
        for r in trials:
            cm = r.get("collapse_metrics", {})
            if cm and last_layer in cm and cm[last_layer]:
                cos_sims.append(cm[last_layer]["avg_cos_sim"])
                refusal_scores.append(r["refusal_score"])

        if cos_sims:
            ax.scatter(cos_sims, refusal_scores, c=color, alpha=0.3, s=15,
                       label=label, edgecolors="none")

    # Add correlation annotation
    corrs = results.get("correlations", {})
    pearson = corrs.get("cos_sim_vs_refusal_score_pearson", {})
    if pearson:
        r_val = pearson.get("r", 0)
        p_val = pearson.get("p", 1)
        ax.set_title(
            f"Collapse vs Safety (Pearson r = {r_val:.3f}, p = {p_val:.2g})",
            fontsize=14,
        )
    else:
        ax.set_title("Collapse vs Safety", fontsize=14)

    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.4)
    ax.set_xlabel("Cosine Similarity (higher = more collapsed)", fontsize=12)
    ax.set_ylabel("Refusal Score (positive = refuses)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "collapse_safety_scatter.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  Saved: collapse_safety_scatter.png")


# ── Plot 4: Category Degradation Heatmap ─────────────────────────────────

def plot_category_heatmap(results: dict, output_dir: Path):
    """Heatmap of safety degradation rate by category x condition."""
    agg = results["aggregated"]
    categories = list(CATEGORY_LABELS.keys())

    # Collect all (condition, length) pairs with data
    conditions_with_data = []
    for condition in sorted(agg.keys()):
        if condition == "no_context":
            continue
        for length in sorted(int(k) for k in agg[condition].keys()):
            entry = agg[condition][str(length)]
            if entry.get("category_stats"):
                conditions_with_data.append((condition, length))

    if not conditions_with_data:
        print("  SKIP: category_heatmap (no data)")
        return

    # Build matrix: categories x condition-length pairs
    col_labels = [f"{c}\n{l}" for c, l in conditions_with_data]
    matrix = np.full((len(categories), len(conditions_with_data)), np.nan)

    for j, (condition, length) in enumerate(conditions_with_data):
        cat_stats = agg[condition][str(length)].get("category_stats", {})
        for i, cat in enumerate(categories):
            if cat in cat_stats:
                deg_rate = cat_stats[cat].get("degradation_rate")
                if deg_rate is not None:
                    matrix[i, j] = deg_rate

    fig, ax = plt.subplots(figsize=(max(12, len(col_labels) * 1.2), 6))

    # Use sequential red colormap (0 = no degradation, 1 = full degradation)
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(
        [CATEGORY_LABELS.get(c, c) for c in categories], fontsize=10,
    )

    # Annotate cells
    for i in range(len(categories)):
        for j in range(len(col_labels)):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                        fontsize=7, color=text_color)

    ax.set_xlabel("Condition / Context Length", fontsize=11)
    ax.set_title("Safety Degradation Rate by Category", fontsize=14)
    plt.colorbar(im, ax=ax, label="Degradation Rate (0 = safe, 1 = fully degraded)")

    plt.tight_layout()
    fig.savefig(output_dir / "category_degradation_heatmap.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  Saved: category_degradation_heatmap.png")


# ── Plot 5: Benign Control Check ─────────────────────────────────────────

def plot_benign_control(results: dict, output_dir: Path):
    """Compliance rate on benign prompts (D, E) across conditions."""
    fig, ax = plt.subplots(figsize=(11, 7))
    agg = results["aggregated"]

    conditions = sorted(agg.keys())
    for condition in conditions:
        color, ls, label = get_condition_style(condition)
        lengths = sorted(int(k) for k in agg[condition].keys())
        compliance_rates = []

        for l in lengths:
            entry = agg[condition][str(l)]
            rate = entry.get("benign_compliance_rate")
            compliance_rates.append(rate if rate is not None else np.nan)

        valid = [(l, r) for l, r in zip(lengths, compliance_rates)
                 if not np.isnan(r)]
        if not valid:
            continue
        vl, vr = zip(*valid)

        ax.plot(vl, vr, color=color, linestyle=ls, marker="s",
                linewidth=2, markersize=7, label=label)

    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Context Length (tokens)", fontsize=12)
    ax.set_ylabel("Compliance Rate (benign prompts)", fontsize=12)
    ax.set_title("Benign Prompt Compliance (Over-Refusal Check)", fontsize=14)
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("symlog", linthresh=100)

    plt.tight_layout()
    fig.savefig(output_dir / "benign_compliance_check.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  Saved: benign_compliance_check.png")


# ── Plot 6: Dual-Axis (Collapse + Safety) ────────────────────────────────

def plot_dual_axis_collapse_safety(results: dict, output_dir: Path):
    """Dual-axis: collapse cos_sim (dashed) + refusal rate (solid) over length."""
    agg = results["aggregated"]

    fig, ax1 = plt.subplots(figsize=(11, 7))
    ax2 = ax1.twinx()

    conditions = sorted(agg.keys())
    for condition in conditions:
        if condition == "no_context":
            continue

        color, _, label = get_condition_style(condition)
        lengths = sorted(int(k) for k in agg[condition].keys())

        # Refusal rate on left axis (solid)
        refusal_rates = []
        for l in lengths:
            entry = agg[condition][str(l)]
            rate = entry.get("refusal_rate_among_baseline_refused")
            refusal_rates.append(rate if rate is not None else np.nan)

        valid_r = [(l, r) for l, r in zip(lengths, refusal_rates)
                   if not np.isnan(r)]
        if valid_r:
            vl, vr = zip(*valid_r)
            ax1.plot(vl, vr, color=color, marker="o", linewidth=2,
                     markersize=6, label=f"{label} (refusal)")

        # Collapse on right axis (dashed)
        cos_sims = [
            agg[condition][str(l)].get("collapse_cos_sim_mean", np.nan)
            for l in lengths
        ]
        valid_c = [(l, c) for l, c in zip(lengths, cos_sims)
                   if c is not None and not np.isnan(c)]
        if valid_c:
            vl, vc = zip(*valid_c)
            ax2.plot(vl, vc, color=color, marker="x", linewidth=1.5,
                     markersize=6, linestyle="--", alpha=0.6)

    ax1.set_xlabel("Context Length (tokens)", fontsize=12)
    ax1.set_ylabel("Refusal Rate (solid lines)", fontsize=12)
    ax2.set_ylabel("Cosine Similarity (dashed lines)", fontsize=12)
    ax1.set_title("Safety vs Collapse Over Context Length", fontsize=14)
    ax1.set_ylim(-0.05, 1.1)
    ax2.set_ylim(-0.05, 1.1)
    ax1.legend(loc="lower left", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("symlog", linthresh=100)

    plt.tight_layout()
    fig.savefig(output_dir / "dual_axis_collapse_safety.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  Saved: dual_axis_collapse_safety.png")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot safety collapse experiment results",
    )
    parser.add_argument(
        "--results-dir", type=str,
        default="results/safety_collapse",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for plots (default: results-dir/plots)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = (
        Path(args.output_dir) if args.output_dir
        else results_dir / "plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    results, raw_results = load_results(results_dir)

    print(f"\nGenerating plots in {output_dir}")
    print("-" * 50)

    plot_refusal_vs_context_length(results, output_dir)
    plot_refusal_score_vs_context_length(results, output_dir)

    if raw_results:
        plot_collapse_safety_scatter(raw_results, results, output_dir)

    plot_category_heatmap(results, output_dir)
    plot_benign_control(results, output_dir)
    plot_dual_axis_collapse_safety(results, output_dir)

    print("\nAll plots generated.")


if __name__ == "__main__":
    main()
