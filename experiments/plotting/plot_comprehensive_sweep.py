#!/usr/bin/env python3
"""
Comprehensive safety collapse sweep plots.

Generates multi-panel figures showing the effect of:
1. Model size (Qwen2.5 family: 0.5B → 72B)
2. Model generation (Qwen2.5 vs Qwen3 vs Qwen3.5 at ~8B scale)
3. Persona injection (with vs without, across models)
4. Architecture (Qwen vs Llama at ~70B scale)
5. 3-way outcome breakdown (refusal / compliance / incoherent)

Usage:
    PYTHONPATH=. python experiments/plotting/plot_comprehensive_sweep.py
"""

import json
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BASE_DIR = Path("results/safety_collapse_sweep")
ORIG_7B = Path("results/safety_collapse")
OUTPUT_DIR = BASE_DIR / "comparison_plots"

# Colors
C_REF = "#2ecc71"   # green = refusal (safe)
C_INC = "#95a5a6"   # gray = incoherent
C_COMP = "#e74c3c"  # red = compliance (unsafe)

LEGEND = [
    Patch(facecolor=C_REF, label="Refusal (safe)"),
    Patch(facecolor=C_INC, label="Incoherent"),
    Patch(facecolor=C_COMP, label="Compliance (unsafe)"),
]


def load_judged(path: Path) -> list[dict]:
    p = path / "judge" / "all_results_judged.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return []


def three_way_split(data: list[dict], condition="structured_walk_raw", baseline_only=True):
    """Compute refusal/compliance/incoherent fractions by context length.

    Returns {length: {"refusal": float, "compliance": float, "incoherent": float, "n": int}}
    """
    filtered = [r for r in data if r.get("condition") == condition]
    if baseline_only:
        filtered = [r for r in filtered if r.get("baseline_refused")]

    lengths = sorted(set(r["context_length"] for r in filtered))
    stats = {}
    for l in lengths:
        trials = [r for r in filtered if r["context_length"] == l]
        n = len(trials)
        if n == 0:
            continue
        c = Counter(r.get("judge_classification", "unknown") for r in trials)
        ref = (c.get("full_refusal", 0) + c.get("partial_refusal", 0)) / n
        comp = c.get("compliance", 0) / n
        inc = (c.get("incoherent", 0) + c.get("parse_error", 0)) / n
        stats[l] = {"refusal": ref, "compliance": comp, "incoherent": inc, "n": n}
    return stats


def fmt_len(l):
    return f"{l//1000}K" if l >= 1000 else str(l)


def draw_stacked(ax, stats, title, show_xlabel=True):
    lengths = sorted(stats.keys())
    if not lengths:
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        return

    ref = np.array([stats[l]["refusal"] for l in lengths])
    inc = np.array([stats[l]["incoherent"] for l in lengths])
    comp = np.array([stats[l]["compliance"] for l in lengths])

    x = np.arange(len(lengths))
    w = 0.7
    ax.bar(x, ref, w, color=C_REF)
    ax.bar(x, inc, w, bottom=ref, color=C_INC)
    ax.bar(x, comp, w, bottom=ref + inc, color=C_COMP)

    ax.set_xticks(x)
    ax.set_xticklabels([fmt_len(l) for l in lengths], fontsize=7, rotation=45)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.15, axis="y")
    if show_xlabel:
        ax.set_xlabel("Context Length", fontsize=9)


# ── Plot 1: Model Size (Qwen2.5 family) ─────────────────────────────

def plot_model_size():
    """Qwen2.5 0.5B → 72B, structured_walk, no persona."""
    models = {
        "0.5B": BASE_DIR / "model_size" / "qwen25_0.5b",
        "3B": BASE_DIR / "model_size" / "qwen25_3b",
        "7B": ORIG_7B,
        "14B": BASE_DIR / "model_size" / "qwen25_14b",
        "72B": BASE_DIR / "model_size" / "qwen25_72b_short",  # has short contexts
    }

    # Merge 72B data from multiple runs
    data_72b = []
    for sub in ["qwen25_72b", "qwen25_72b_short", "qwen25_72b_finegrained"]:
        data_72b.extend(load_judged(BASE_DIR / "model_size" / sub))

    available = {}
    for label, path in models.items():
        if label == "72B":
            available[label] = data_72b
        else:
            d = load_judged(path)
            if d:
                available[label] = d

    size_order = [s for s in ["0.5B", "3B", "7B", "14B", "72B"] if s in available]
    n = len(size_order)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 1, figsize=(14, 3.2 * n), squeeze=False)
    for i, size in enumerate(size_order):
        stats = three_way_split(available[size])
        draw_stacked(axes[i, 0], stats, f"Qwen2.5-{size}-Instruct", show_xlabel=(i == n - 1))
        axes[i, 0].set_ylabel("Proportion", fontsize=9)

    fig.legend(handles=LEGEND, loc="lower center", ncol=3, fontsize=10,
               frameon=True, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Safety Collapse by Model Size (Qwen2.5, structured_walk, baseline-refused only)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(OUTPUT_DIR / "01_model_size.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 01_model_size.png")


# ── Plot 2: Model Generation (~8B scale) ────────────────────────────

def plot_model_generation():
    """Qwen2.5-7B vs Qwen3-8B vs Qwen3.5-9B at similar scale."""
    models = {
        "Qwen2.5-7B": ORIG_7B,
        "Qwen3-8B": BASE_DIR / "model_gen" / "qwen3_8b",
        "Qwen3.5-9B": BASE_DIR / "model_gen" / "qwen35_9b",
    }

    available = {k: load_judged(v) for k, v in models.items() if load_judged(v)}
    order = [m for m in models if m in available]
    n = len(order)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 1, figsize=(14, 3.2 * n), squeeze=False)
    for i, model in enumerate(order):
        stats = three_way_split(available[model])
        # For models with weak baseline safety, show all prompts
        if not stats:
            stats = three_way_split(available[model], baseline_only=False)
        draw_stacked(axes[i, 0], stats, model, show_xlabel=(i == n - 1))
        axes[i, 0].set_ylabel("Proportion", fontsize=9)

    fig.legend(handles=LEGEND, loc="lower center", ncol=3, fontsize=10,
               frameon=True, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Safety Collapse by Model Generation (~8B scale, structured_walk)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(OUTPUT_DIR / "02_model_generation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 02_model_generation.png")


# ── Plot 3: Persona Injection Effect ────────────────────────────────

def plot_persona_effect():
    """Side-by-side: no persona vs persona for each model that has both."""
    pairs = [
        ("Qwen2.5-0.5B", BASE_DIR / "model_size" / "qwen25_0.5b",
                          BASE_DIR / "model_size" / "qwen25_0.5b_persona"),
        ("Qwen2.5-72B",  BASE_DIR / "model_size" / "qwen25_72b",
                          BASE_DIR / "model_size" / "qwen25_72b_persona"),
        ("Qwen3-8B",     BASE_DIR / "model_gen" / "qwen3_8b",
                          BASE_DIR / "model_gen" / "qwen3_8b_persona"),
        ("Qwen3.5-9B",   BASE_DIR / "model_gen" / "qwen35_9b",
                          BASE_DIR / "model_gen" / "qwen35_9b_persona"),
        ("Qwen3.5-0.8B", BASE_DIR / "model_gen" / "qwen35_0.8b",
                          BASE_DIR / "model_gen" / "qwen35_0.8b_persona"),
        ("Qwen3.5-2B",   BASE_DIR / "model_gen" / "qwen35_2b",
                          BASE_DIR / "model_gen" / "qwen35_2b_persona"),
    ]

    valid = []
    for label, base_path, persona_path in pairs:
        base_data = load_judged(base_path)
        persona_data = load_judged(persona_path)
        if base_data and persona_data:
            valid.append((label, base_data, persona_data))

    if not valid:
        print("  SKIP: persona effect (no paired data)")
        return

    n = len(valid)
    fig, axes = plt.subplots(n, 2, figsize=(16, 3.0 * n), squeeze=False)

    for i, (label, base_data, persona_data) in enumerate(valid):
        base_stats = three_way_split(base_data)
        persona_stats = three_way_split(persona_data)

        # If no baseline-refused stats, try all prompts
        if not base_stats:
            base_stats = three_way_split(base_data, baseline_only=False)
        if not persona_stats:
            persona_stats = three_way_split(persona_data, baseline_only=False)

        draw_stacked(axes[i, 0], base_stats, f"{label} (no injection)", show_xlabel=(i == n - 1))
        draw_stacked(axes[i, 1], persona_stats, f"{label} + persona", show_xlabel=(i == n - 1))
        axes[i, 0].set_ylabel("Proportion", fontsize=9)

    fig.legend(handles=LEGEND, loc="lower center", ncol=3, fontsize=10,
               frameon=True, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Effect of Persona Injection on Safety Collapse (structured_walk)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(OUTPUT_DIR / "03_persona_effect.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 03_persona_effect.png")


# ── Plot 4: Architecture Comparison (70B scale) ─────────────────────

def plot_architecture_70b():
    """Qwen2.5-72B vs Llama-3.3-70B."""
    models = {
        "Qwen2.5-72B": None,  # merged below
        "Llama-3.3-70B": None,  # merged below
    }

    qwen72b = []
    for sub in ["qwen25_72b", "qwen25_72b_short"]:
        qwen72b.extend(load_judged(BASE_DIR / "model_size" / sub))

    llama70b = []
    for sub in ["llama33_70b", "llama33_70b_short"]:
        llama70b.extend(load_judged(BASE_DIR / "architecture" / sub))

    if not qwen72b or not llama70b:
        print("  SKIP: architecture 70B (missing data)")
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), squeeze=False)

    draw_stacked(axes[0, 0], three_way_split(qwen72b), "Qwen2.5-72B-Instruct", show_xlabel=False)
    draw_stacked(axes[1, 0], three_way_split(llama70b), "Llama-3.3-70B-Instruct")
    axes[0, 0].set_ylabel("Proportion", fontsize=9)
    axes[1, 0].set_ylabel("Proportion", fontsize=9)

    fig.legend(handles=LEGEND, loc="lower center", ncol=3, fontsize=10,
               frameon=True, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Architecture Comparison at 70B Scale (structured_walk, baseline-refused)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(OUTPUT_DIR / "04_architecture_70b.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 04_architecture_70b.png")


# ── Plot 5: Compliance Heatmap ───────────────────────────────────────

def plot_compliance_heatmap():
    """Heatmap: model × context length → genuine compliance rate."""
    models = {
        "Qwen2.5-0.5B": load_judged(BASE_DIR / "model_size" / "qwen25_0.5b"),
        "Qwen2.5-3B": load_judged(BASE_DIR / "model_size" / "qwen25_3b"),
        "Qwen2.5-7B": load_judged(ORIG_7B),
        "Qwen2.5-14B": load_judged(BASE_DIR / "model_size" / "qwen25_14b"),
        "Qwen2.5-72B": load_judged(BASE_DIR / "model_size" / "qwen25_72b") +
                        load_judged(BASE_DIR / "model_size" / "qwen25_72b_short"),
        "Qwen3-8B": load_judged(BASE_DIR / "model_gen" / "qwen3_8b"),
        "Qwen3.5-9B": load_judged(BASE_DIR / "model_gen" / "qwen35_9b"),
        "Llama-3.3-70B": load_judged(BASE_DIR / "architecture" / "llama33_70b") +
                          load_judged(BASE_DIR / "architecture" / "llama33_70b_short"),
    }

    available = {k: v for k, v in models.items() if v}
    if len(available) < 2:
        print("  SKIP: compliance heatmap (need >= 2 models)")
        return

    model_order = [m for m in models if m in available]
    all_stats = {m: three_way_split(available[m]) for m in model_order}
    all_lengths = sorted(set(l for s in all_stats.values() for l in s))

    matrix = np.full((len(model_order), len(all_lengths)), np.nan)
    for i, m in enumerate(model_order):
        for j, l in enumerate(all_lengths):
            if l in all_stats[m]:
                matrix[i, j] = all_stats[m][l]["compliance"]

    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=0.6)

    ax.set_xticks(range(len(all_lengths)))
    ax.set_xticklabels([fmt_len(l) for l in all_lengths], fontsize=8)
    ax.set_yticks(range(len(model_order)))
    ax.set_yticklabels(model_order, fontsize=10)

    for i in range(len(model_order)):
        for j in range(len(all_lengths)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if val > 0.3 else "black"
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                        fontsize=7, color=color)

    ax.set_xlabel("Context Length", fontsize=11)
    ax.set_ylabel("Model", fontsize=11)
    ax.set_title("Genuine Compliance Rate (Danger Zone)\n"
                 "structured_walk, baseline-refused prompts only",
                 fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Compliance Rate", shrink=0.8)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_compliance_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 05_compliance_heatmap.png")


# ── Plot 6: Qwen3.5 Scale Sweep ─────────────────────────────────────

def plot_qwen35_scale():
    """Qwen3.5 at different scales: 0.8B, 2B, 4B, 9B, 27B."""
    models = {
        "0.8B": BASE_DIR / "model_gen" / "qwen35_0.8b",
        "2B": BASE_DIR / "model_gen" / "qwen35_2b",
        "4B": BASE_DIR / "model_gen" / "qwen35_4b",
        "9B": BASE_DIR / "model_gen" / "qwen35_9b",
        "27B": BASE_DIR / "model_gen" / "qwen35_27b",
    }

    available = {}
    for label, path in models.items():
        d = load_judged(path)
        if d:
            available[label] = d

    order = [s for s in models if s in available]
    n = len(order)
    if n == 0:
        print("  SKIP: qwen3.5 scale (no data)")
        return

    fig, axes = plt.subplots(n, 1, figsize=(14, 3.0 * n), squeeze=False)
    for i, size in enumerate(order):
        # These models have weak safety — show all prompts if no baseline-refused
        stats = three_way_split(available[size])
        if not stats:
            stats = three_way_split(available[size], baseline_only=False)
        draw_stacked(axes[i, 0], stats, f"Qwen3.5-{size}", show_xlabel=(i == n - 1))
        axes[i, 0].set_ylabel("Proportion", fontsize=9)

    fig.legend(handles=LEGEND, loc="lower center", ncol=3, fontsize=10,
               frameon=True, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Safety Collapse by Scale (Qwen3.5 family, structured_walk)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(OUTPUT_DIR / "06_qwen35_scale.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 06_qwen35_scale.png")


# ── Plot 7: Collapse Metric vs Safety (scatter) ─────────────────────

def plot_collapse_vs_safety():
    """Scatter: cos_sim vs compliance rate across all models."""
    all_experiments = {
        "Qwen2.5-0.5B": load_judged(BASE_DIR / "model_size" / "qwen25_0.5b"),
        "Qwen2.5-3B": load_judged(BASE_DIR / "model_size" / "qwen25_3b"),
        "Qwen2.5-7B": load_judged(ORIG_7B),
        "Qwen2.5-14B": load_judged(BASE_DIR / "model_size" / "qwen25_14b"),
        "Qwen2.5-72B": load_judged(BASE_DIR / "model_size" / "qwen25_72b"),
        "Qwen3-8B": load_judged(BASE_DIR / "model_gen" / "qwen3_8b"),
        "Qwen3.5-9B": load_judged(BASE_DIR / "model_gen" / "qwen35_9b"),
        "Llama-3.3-70B": load_judged(BASE_DIR / "architecture" / "llama33_70b"),
    }

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_experiments)))

    for (label, data), color in zip(all_experiments.items(), colors):
        if not data:
            continue
        stats = three_way_split(data)
        if not stats:
            continue

        # Get collapse cos_sim from aggregated results
        # Use per-condition stats from the judge data
        for length, s in stats.items():
            # Find collapse metric for this length
            trials = [r for r in data if r.get("condition") == "structured_walk_raw"
                      and r["context_length"] == length and r.get("baseline_refused")]
            if not trials:
                continue

            cos_sims = []
            for r in trials:
                cm = r.get("collapse_metrics", {})
                if cm:
                    # Get last layer
                    last_key = sorted(cm.keys())[-1] if cm else None
                    if last_key and cm[last_key]:
                        cos_sims.append(cm[last_key].get("avg_cos_sim", 0))

            if cos_sims:
                avg_cos = np.mean(cos_sims)
                ax.scatter(avg_cos, s["compliance"], c=[color], s=60, alpha=0.7,
                          label=label if length == min(stats.keys()) else "")

    ax.set_xlabel("Collapse (cos_sim)", fontsize=12)
    ax.set_ylabel("Genuine Compliance Rate", fontsize=12)
    ax.set_title("Collapse Metric vs Safety Degradation\n(each point = one model × context length)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(-0.02, 0.65)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "07_collapse_vs_compliance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 07_collapse_vs_compliance.png")


# ── Plot 8: Vocab Size Effect ────────────────────────────────────────

def plot_vocab_size():
    """Vocab 15 vs 50 vs 200 vs 1000."""
    models = {
        "vocab=15": load_judged(ORIG_7B),
        "vocab=50": load_judged(BASE_DIR / "vocab_size" / "vocab50") +
                    load_judged(BASE_DIR / "vocab_size" / "vocab50_100k"),
        "vocab=200": load_judged(BASE_DIR / "vocab_size" / "vocab200") +
                     load_judged(BASE_DIR / "vocab_size" / "vocab200_100k"),
        "vocab=1000": load_judged(BASE_DIR / "vocab_size" / "vocab1000") +
                      load_judged(BASE_DIR / "vocab_size" / "vocab1000_100k"),
    }

    available = {k: v for k, v in models.items() if v}
    order = [v for v in models if v in available]
    n = len(order)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 1, figsize=(14, 3.0 * n), squeeze=False)
    for i, label in enumerate(order):
        stats = three_way_split(available[label])
        draw_stacked(axes[i, 0], stats, f"Qwen2.5-7B, {label}", show_xlabel=(i == n - 1))
        axes[i, 0].set_ylabel("Proportion", fontsize=9)

    fig.legend(handles=LEGEND, loc="lower center", ncol=3, fontsize=10,
               frameon=True, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Vocab Size Effect on Safety (Qwen2.5-7B, structured_walk)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(OUTPUT_DIR / "08_vocab_size.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 08_vocab_size.png")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating comprehensive plots in {OUTPUT_DIR}")
    print("-" * 50)

    plot_model_size()
    plot_model_generation()
    plot_persona_effect()
    plot_architecture_70b()
    plot_compliance_heatmap()
    plot_qwen35_scale()
    plot_collapse_vs_safety()
    plot_vocab_size()

    print("\nAll plots generated.")


if __name__ == "__main__":
    main()
