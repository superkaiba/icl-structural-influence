#!/usr/bin/env python3
"""
Plots for LLM-Judge Safety Collapse Results.

Generates 6 plots:
1. Judge refusal rate vs context length (primary result)
2. Judge degradation rate vs context length
3. Classification breakdown stacked bars
4. Category degradation heatmap
5. Collapse-safety scatter (cos_sim vs judge degradation)
6. Dual-axis: collapse + judge refusal over context length

Usage:
    PYTHONPATH=. python experiments/plotting/plot_safety_collapse_judge.py \
        --results-dir results/safety_collapse
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── Style ─────────────────────────────────────────────────────────────────

CONDITION_COLORS = {
    "no_context": "#333333",
    "structured_walk_raw": "#e74c3c",
    "structured_walk_chat": "#c0392b",
    "natural_books_raw": "#2ecc71",
    "natural_books_chat": "#27ae60",
    "template_small_vocab_raw": "#f39c12",
    "template_small_vocab_chat": "#e67e22",
    "repeated_token_raw": "#9b59b6",
    "repeated_token_chat": "#8e44ad",
}

CONDITION_STYLES = {
    "no_context": ("-", "s"),
    "structured_walk_raw": ("-", "o"),
    "structured_walk_chat": ("--", "o"),
    "natural_books_raw": ("-", "^"),
    "natural_books_chat": ("--", "^"),
    "template_small_vocab_raw": ("-", "D"),
    "template_small_vocab_chat": ("--", "D"),
    "repeated_token_raw": ("-", "v"),
    "repeated_token_chat": ("--", "v"),
}

CONDITION_LABELS = {
    "no_context": "No context (baseline)",
    "structured_walk_raw": "Structured walk (raw)",
    "structured_walk_chat": "Structured walk (chat)",
    "natural_books_raw": "Natural books (raw)",
    "natural_books_chat": "Natural books (chat)",
    "template_small_vocab_raw": "Template vocab (raw)",
    "template_small_vocab_chat": "Template vocab (chat)",
    "repeated_token_raw": "Repeated token (raw)",
    "repeated_token_chat": "Repeated token (chat)",
}

CLASSIFICATION_COLORS = {
    "full_refusal": "#2ecc71",
    "partial_refusal": "#f1c40f",
    "compliance": "#e74c3c",
    "incoherent": "#95a5a6",
    "parse_error": "#bdc3c7",
}

CATEGORY_LABELS = {
    "A_direct_harmful": "A: Direct Harmful",
    "B_indirect_harmful": "B: Indirect Harmful",
    "C_borderline": "C: Borderline",
    "D_benign_control": "D: Benign Control",
    "E_safety_knowledge": "E: Safety Knowledge",
}


def load_data(results_dir: Path):
    judge_path = results_dir / "judge" / "judge_results.json"
    with open(judge_path) as f:
        judge = json.load(f)

    judged_path = results_dir / "judge" / "all_results_judged.json"
    raw = []
    if judged_path.exists():
        with open(judged_path) as f:
            raw = json.load(f)

    return judge, raw


# ── Plot 1: Judge Refusal Rate vs Context Length ──────────────────────────

def plot_refusal_rate(judge: dict, output_dir: Path):
    fig, ax = plt.subplots(figsize=(11, 7))
    agg = judge["aggregated"]

    for condition in sorted(agg.keys()):
        color = CONDITION_COLORS.get(condition, "#666")
        ls, marker = CONDITION_STYLES.get(condition, ("-", "o"))
        label = CONDITION_LABELS.get(condition, condition)

        lengths = sorted(int(k) for k in agg[condition].keys())
        rates = []
        for l in lengths:
            r = agg[condition][str(l)].get("judge_refusal_rate_among_baseline")
            rates.append(r if r is not None else np.nan)

        valid = [(l, r) for l, r in zip(lengths, rates) if not np.isnan(r)]
        if not valid:
            continue
        vl, vr = zip(*valid)
        ax.plot(vl, vr, color=color, linestyle=ls, marker=marker,
                linewidth=2, markersize=7, label=label)

    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Context Length (tokens)", fontsize=13)
    ax.set_ylabel("Refusal Rate (judge, among baseline-refused)", fontsize=13)
    ax.set_title("Safety Refusal Rate vs Context Length (LLM Judge)", fontsize=15)
    ax.legend(fontsize=9, loc="lower left")
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("symlog", linthresh=100)

    plt.tight_layout()
    fig.savefig(output_dir / "judge_refusal_vs_length.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: judge_refusal_vs_length.png")


# ── Plot 2: Degradation Rate vs Context Length ────────────────────────────

def plot_degradation_rate(judge: dict, output_dir: Path):
    fig, ax = plt.subplots(figsize=(11, 7))
    agg = judge["aggregated"]

    for condition in sorted(agg.keys()):
        if condition == "no_context":
            continue
        color = CONDITION_COLORS.get(condition, "#666")
        ls, marker = CONDITION_STYLES.get(condition, ("-", "o"))
        label = CONDITION_LABELS.get(condition, condition)

        lengths = sorted(int(k) for k in agg[condition].keys())
        rates = []
        for l in lengths:
            r = agg[condition][str(l)].get("judge_degradation_rate")
            rates.append(r if r is not None else np.nan)

        valid = [(l, r) for l, r in zip(lengths, rates) if not np.isnan(r)]
        if not valid:
            continue
        vl, vr = zip(*valid)
        ax.plot(vl, vr, color=color, linestyle=ls, marker=marker,
                linewidth=2, markersize=7, label=label)

    ax.axhline(y=0.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Context Length (tokens)", fontsize=13)
    ax.set_ylabel("Safety Degradation Rate (judge)", fontsize=13)
    ax.set_title("Safety Degradation vs Context Length (LLM Judge)", fontsize=15)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("symlog", linthresh=100)

    plt.tight_layout()
    fig.savefig(output_dir / "judge_degradation_vs_length.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: judge_degradation_vs_length.png")


# ── Plot 3: Classification Breakdown Stacked Bars ─────────────────────────

def plot_classification_breakdown(judge: dict, output_dir: Path):
    agg = judge["aggregated"]

    # Collect all (condition, length) pairs
    entries = []
    for condition in sorted(agg.keys()):
        for length in sorted(int(k) for k in agg[condition].keys()):
            bd = agg[condition][str(length)].get("classification_breakdown", {})
            if bd:
                entries.append((condition, length, bd))

    if not entries:
        return

    fig, ax = plt.subplots(figsize=(max(14, len(entries) * 0.7), 7))

    x = np.arange(len(entries))
    bar_width = 0.7
    classifications = ["full_refusal", "partial_refusal", "compliance", "incoherent"]
    cls_labels = ["Full Refusal", "Partial Refusal", "Compliance", "Incoherent"]

    bottoms = np.zeros(len(entries))
    for cls, cls_label in zip(classifications, cls_labels):
        values = []
        for _, _, bd in entries:
            total = sum(bd.values())
            values.append(bd.get(cls, 0) / total if total > 0 else 0)
        values = np.array(values)
        ax.bar(x, values, bar_width, bottom=bottoms,
               color=CLASSIFICATION_COLORS.get(cls, "#ccc"), label=cls_label)
        bottoms += values

    # Labels
    tick_labels = []
    for cond, length, _ in entries:
        short = cond.replace("structured_walk", "sw").replace("natural_books", "nb").replace("_raw", "/r").replace("_chat", "/c").replace("no_context", "base")
        tick_labels.append(f"{short}\n{length}")

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=6, rotation=45, ha="right")
    ax.set_ylabel("Proportion", fontsize=12)
    ax.set_title("Classification Breakdown by Condition (LLM Judge)", fontsize=14)
    ax.legend(fontsize=10, loc="upper right")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    fig.savefig(output_dir / "judge_classification_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: judge_classification_breakdown.png")


# ── Plot 4: Category Degradation Heatmap ──────────────────────────────────

def plot_category_heatmap(judge: dict, output_dir: Path):
    agg = judge["aggregated"]
    categories = ["A_direct_harmful", "B_indirect_harmful", "C_borderline",
                   "D_benign_control", "E_safety_knowledge"]

    # Collect condition-length pairs (skip no_context)
    col_entries = []
    for condition in sorted(agg.keys()):
        if condition == "no_context":
            continue
        for length in sorted(int(k) for k in agg[condition].keys()):
            cat_stats = agg[condition][str(length)].get("category_stats", {})
            if cat_stats:
                col_entries.append((condition, length))

    if not col_entries:
        return

    matrix = np.full((len(categories), len(col_entries)), np.nan)
    for j, (cond, length) in enumerate(col_entries):
        cat_stats = agg[cond][str(length)].get("category_stats", {})
        for i, cat in enumerate(categories):
            if cat in cat_stats:
                dr = cat_stats[cat].get("judge_degradation_rate")
                if dr is not None:
                    matrix[i, j] = dr

    col_labels = []
    for cond, length in col_entries:
        short = cond.replace("structured_walk", "sw").replace("natural_books", "nb").replace("_raw", "/r").replace("_chat", "/c")
        col_labels.append(f"{short}\n{length}")

    fig, ax = plt.subplots(figsize=(max(14, len(col_labels) * 0.8), 5))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=6, rotation=45, ha="right")
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels([CATEGORY_LABELS.get(c, c) for c in categories], fontsize=10)

    for i in range(len(categories)):
        for j in range(len(col_labels)):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                        fontsize=6, color=text_color)

    ax.set_title("Safety Degradation by Category (LLM Judge)", fontsize=14)
    plt.colorbar(im, ax=ax, label="Degradation Rate")
    plt.tight_layout()
    fig.savefig(output_dir / "judge_category_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: judge_category_heatmap.png")


# ── Plot 5: Collapse-Safety Scatter ───────────────────────────────────────

def plot_collapse_scatter(raw: list[dict], output_dir: Path):
    if not raw:
        print("  SKIP: collapse scatter (no raw data)")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    conditions = sorted(set(r["condition"] for r in raw))
    for condition in conditions:
        color = CONDITION_COLORS.get(condition, "#666")
        label = CONDITION_LABELS.get(condition, condition)
        trials = [r for r in raw if r["condition"] == condition]

        cos_sims = []
        degraded = []
        for r in trials:
            cm = r.get("collapse_metrics", {})
            if not cm:
                continue
            layer_keys = sorted(cm.keys(), key=lambda x: int(x))
            if not layer_keys:
                continue
            last_cm = cm[layer_keys[-1]]
            if last_cm and "avg_cos_sim" in last_cm:
                cos_sims.append(last_cm["avg_cos_sim"])
                degraded.append(int(r.get("judge_safety_degraded", False)))

        if not cos_sims:
            continue

        # Jitter degraded for visibility
        jittered_y = [d + np.random.normal(0, 0.03) for d in degraded]
        ax.scatter(cos_sims, jittered_y, c=color, alpha=0.2, s=12,
                   label=label, edgecolors="none")

    ax.set_xlabel("Cosine Similarity (higher = more collapsed)", fontsize=12)
    ax.set_ylabel("Safety Degraded (jittered)", fontsize=12)
    ax.set_title("Collapse vs Safety Degradation (LLM Judge)", fontsize=14)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Safe", "Degraded"])
    ax.legend(fontsize=8, loc="center right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "judge_collapse_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: judge_collapse_scatter.png")


# ── Plot 6: Dual-Axis Collapse + Safety ───────────────────────────────────

def plot_dual_axis(judge: dict, output_dir: Path):
    agg = judge["aggregated"]

    fig, ax1 = plt.subplots(figsize=(11, 7))
    ax2 = ax1.twinx()

    for condition in sorted(agg.keys()):
        if condition == "no_context":
            continue
        color = CONDITION_COLORS.get(condition, "#666")
        ls, marker = CONDITION_STYLES.get(condition, ("-", "o"))
        label = CONDITION_LABELS.get(condition, condition)

        lengths = sorted(int(k) for k in agg[condition].keys())

        # Refusal rate (left, solid)
        rates = []
        for l in lengths:
            r = agg[condition][str(l)].get("judge_refusal_rate_among_baseline")
            rates.append(r if r is not None else np.nan)

        valid = [(l, r) for l, r in zip(lengths, rates) if not np.isnan(r)]
        if valid:
            vl, vr = zip(*valid)
            ax1.plot(vl, vr, color=color, linestyle=ls, marker=marker,
                     linewidth=2, markersize=6, label=f"{label}")

        # Collapse (right, dashed, no label)
        cos_sims = []
        for l in lengths:
            c = agg[condition][str(l)].get("collapse_cos_sim_mean")
            cos_sims.append(c if c is not None else np.nan)

        valid_c = [(l, c) for l, c in zip(lengths, cos_sims) if not np.isnan(c)]
        if valid_c:
            vl, vc = zip(*valid_c)
            ax2.plot(vl, vc, color=color, marker="x", linewidth=1,
                     markersize=5, linestyle=":", alpha=0.5)

    ax1.set_xlabel("Context Length (tokens)", fontsize=13)
    ax1.set_ylabel("Refusal Rate — judge (solid lines)", fontsize=12)
    ax2.set_ylabel("Cosine Similarity (dotted lines)", fontsize=12)
    ax1.set_title("Safety vs Collapse Over Context Length", fontsize=15)
    ax1.set_ylim(-0.05, 1.1)
    ax2.set_ylim(-0.05, 1.1)
    ax1.legend(loc="lower left", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("symlog", linthresh=100)

    plt.tight_layout()
    fig.savefig(output_dir / "judge_dual_axis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: judge_dual_axis.png")


# ── Plot 7: Keyword vs Judge comparison ───────────────────────────────────

def plot_keyword_vs_judge(raw: list[dict], output_dir: Path):
    """Compare keyword-based and judge-based refusal classifications."""
    if not raw:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: confusion matrix
    ax = axes[0]
    # keyword_refused x judge_classification
    matrix = np.zeros((2, 4))  # rows: keyword refused/complied, cols: full/partial/compliance/incoherent
    cls_order = ["full_refusal", "partial_refusal", "compliance", "incoherent"]

    for r in raw:
        kw_row = 0 if r.get("refused", False) else 1
        judge_cls = r.get("judge_classification", "compliance")
        if judge_cls in cls_order:
            col = cls_order.index(judge_cls)
            matrix[kw_row, col] += 1

    # Normalize by row
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix_norm = matrix / row_sums

    im = ax.imshow(matrix_norm, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Full\nRefusal", "Partial\nRefusal", "Compliance", "Incoherent"], fontsize=9)
    ax.set_yticks(range(2))
    ax.set_yticklabels(["Keyword: Refused", "Keyword: Complied"], fontsize=10)

    for i in range(2):
        for j in range(4):
            count = int(matrix[i, j])
            pct = matrix_norm[i, j]
            text_color = "white" if pct > 0.5 else "black"
            ax.text(j, i, f"{count}\n({pct:.0%})", ha="center", va="center",
                    fontsize=9, color=text_color)

    ax.set_title("Keyword vs Judge Classification", fontsize=13)
    ax.set_xlabel("Judge Classification", fontsize=11)
    plt.colorbar(im, ax=ax, label="Row proportion")

    # Right: agreement over context length for harmful prompts only
    ax = axes[1]
    harmful = [r for r in raw if r.get("expected_behavior") == "refuse"]
    conditions_to_plot = ["structured_walk_raw", "structured_walk_chat",
                          "natural_books_raw"]

    for condition in conditions_to_plot:
        color = CONDITION_COLORS.get(condition, "#666")
        ls, marker = CONDITION_STYLES.get(condition, ("-", "o"))
        label = CONDITION_LABELS.get(condition, condition)

        cond_trials = [r for r in harmful if r["condition"] == condition]
        lengths = sorted(set(r["context_length"] for r in cond_trials))

        agreement = []
        for l in lengths:
            lt = [r for r in cond_trials if r["context_length"] == l]
            if not lt:
                continue
            agree = sum(
                1 for r in lt
                if r.get("refused", False) == r.get("judge_refused", False)
            )
            agreement.append((l, agree / len(lt)))

        if agreement:
            al, ar = zip(*agreement)
            ax.plot(al, ar, color=color, linestyle=ls, marker=marker,
                    linewidth=2, markersize=6, label=label)

    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Context Length (tokens)", fontsize=11)
    ax.set_ylabel("Agreement Rate (keyword vs judge)", fontsize=11)
    ax.set_title("Keyword-Judge Agreement on Harmful Prompts", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("symlog", linthresh=100)

    plt.tight_layout()
    fig.savefig(output_dir / "judge_vs_keyword_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: judge_vs_keyword_comparison.png")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot LLM judge safety collapse results")
    parser.add_argument("--results-dir", type=str, default="results/safety_collapse")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "judge" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    judge, raw = load_data(results_dir)
    print(f"  {judge['n_evaluations']} evaluations, {len(raw)} raw records")

    print(f"\nGenerating plots in {output_dir}")
    print("-" * 50)

    plot_refusal_rate(judge, output_dir)
    plot_degradation_rate(judge, output_dir)
    plot_classification_breakdown(judge, output_dir)
    plot_category_heatmap(judge, output_dir)
    plot_dual_axis(judge, output_dir)

    if raw:
        plot_collapse_scatter(raw, output_dir)
        plot_keyword_vs_judge(raw, output_dir)

    print("\nAll plots generated.")


if __name__ == "__main__":
    main()
