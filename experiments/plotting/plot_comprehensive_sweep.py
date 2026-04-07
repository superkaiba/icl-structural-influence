#!/usr/bin/env python3
"""
Comprehensive safety collapse sweep v2 plots.

Star-shaped design:
  1. Context length (default model, full 18-point range)
  2. Model size (Qwen3.5 family: 0.8B → 27B)
  3. Context type (14 types on default model)
  4. Architecture (Qwen3.5-9B, Llama-3.3-70B, OLMo-3-7B)

Reads from results/safety_collapse_sweep_v2/ and also merges v1 data
from results/safety_collapse_sweep/ when available.

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
from matplotlib.ticker import FuncFormatter

# ── Paths ─────────────────────────────────────────────────────────────

V2_DIR = Path("results/safety_collapse_sweep_v2")
V1_DIR = Path("results/safety_collapse_sweep")
ORIG_7B = Path("results/safety_collapse")
OUTPUT_DIR = V2_DIR / "plots"

# ── Colors / Legend ───────────────────────────────────────────────────

C_REF = "#2ecc71"   # green = refusal (safe)
C_INC = "#95a5a6"   # gray = incoherent
C_COMP = "#e74c3c"  # red = compliance (unsafe)

LEGEND = [
    Patch(facecolor=C_REF, label="Refusal (safe)"),
    Patch(facecolor=C_INC, label="Incoherent"),
    Patch(facecolor=C_COMP, label="Compliance (unsafe)"),
]


# ── Helpers ───────────────────────────────────────────────────────────

def load_judged(path: Path) -> list[dict]:
    """Load LLM-judged results from a directory, falling back to unjudged."""
    p = path / "judge" / "all_results_judged.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    # Fallback to unjudged all_results.json (keyword classifier)
    p2 = path / "all_results.json"
    if p2.exists():
        with open(p2) as f:
            return json.load(f)
    return []


def load_judged_multi(*paths: Path) -> list[dict]:
    """Load and merge judged results from multiple directories."""
    data = []
    for p in paths:
        data.extend(load_judged(p))
    return data


def three_way_split(data: list[dict], condition=None, baseline_only=True):
    """Compute refusal/compliance/incoherent fractions by context length.

    Args:
        data: list of judged result dicts
        condition: filter to this condition string (or None for auto-detect structured_walk)
        baseline_only: if True, only consider prompts that were refused at baseline

    Returns:
        {length: {"refusal": float, "compliance": float, "incoherent": float, "n": int}}
    """
    if condition is None:
        # Auto-detect: prefer structured_walk_15_raw, fall back to structured_walk_raw
        conditions = set(r.get("condition", "") for r in data)
        for c in ["structured_walk_15_raw", "structured_walk_raw"]:
            if c in conditions:
                condition = c
                break
        if condition is None:
            # Use whatever condition has most data
            if conditions:
                condition = max(conditions, key=lambda c: sum(
                    1 for r in data if r.get("condition") == c))
            else:
                return {}

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
    if l >= 1000:
        return f"{l//1000}K"
    return str(l)


def log_x_formatter(x, pos):
    if x >= 1000:
        return f"{int(x)//1000}K"
    return str(int(x))


def draw_stacked(ax, stats, title, show_xlabel=True, use_log_x=False):
    """Draw a stacked bar chart of refusal/incoherent/compliance."""
    lengths = sorted(stats.keys())
    if not lengths:
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        return

    ref = np.array([stats[l]["refusal"] for l in lengths])
    inc = np.array([stats[l]["incoherent"] for l in lengths])
    comp = np.array([stats[l]["compliance"] for l in lengths])

    if use_log_x and all(l > 0 for l in lengths):
        x_pos = np.log10(np.array(lengths, dtype=float))
        w = np.min(np.diff(x_pos)) * 0.7 if len(x_pos) > 1 else 0.5
    else:
        x_pos = np.arange(len(lengths), dtype=float)
        w = 0.7

    ax.bar(x_pos, ref, w, color=C_REF)
    ax.bar(x_pos, inc, w, bottom=ref, color=C_INC)
    ax.bar(x_pos, comp, w, bottom=ref + inc, color=C_COMP)

    if use_log_x and all(l > 0 for l in lengths):
        ax.set_xticks(x_pos)
        ax.set_xticklabels([fmt_len(l) for l in lengths], fontsize=7, rotation=45)
    else:
        ax.set_xticks(x_pos)
        ax.set_xticklabels([fmt_len(l) for l in lengths], fontsize=7, rotation=45)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.15, axis="y")
    if show_xlabel:
        ax.set_xlabel("Context Length", fontsize=9)


# ── Plot 1: Model Size (Qwen3.5 family) ─────────────────────────────

def plot_model_size():
    """Qwen3.5 0.8B → 27B, structured_walk_15, no injection."""
    models = {
        "0.8B": V2_DIR / "model_size" / "qwen35_0.8b",
        "2B": V2_DIR / "model_size" / "qwen35_2b",
        "4B": V2_DIR / "model_size" / "qwen35_4b",
        "9B": V2_DIR / "model_size" / "qwen35_9b",
        "27B": V2_DIR / "model_size" / "qwen35_27b",
    }

    # Fallback paths: 9B is in context_length/, older data in v1
    fallbacks = {
        "9B": [V2_DIR / "context_length" / "qwen35_9b"],
        "0.8B": [V1_DIR / "model_gen" / "qwen35_0.8b"],
        "2B": [V1_DIR / "model_gen" / "qwen35_2b"],
        "4B": [V1_DIR / "model_gen" / "qwen35_4b"],
        "27B": [V1_DIR / "model_gen" / "qwen35_27b"],
    }

    available = {}
    for label in models:
        d = load_judged(models[label])
        if not d:
            for fb in fallbacks.get(label, []):
                d = load_judged(fb)
                if d:
                    break
        if d:
            available[label] = d

    size_order = [s for s in ["0.8B", "2B", "4B", "9B", "27B"] if s in available]
    n = len(size_order)
    if n == 0:
        print("  SKIP: model_size (no data)")
        return

    fig, axes = plt.subplots(n, 1, figsize=(14, 3.2 * n), squeeze=False)
    for i, size in enumerate(size_order):
        stats = three_way_split(available[size])
        if not stats:
            stats = three_way_split(available[size], baseline_only=False)
        draw_stacked(axes[i, 0], stats, f"Qwen3.5-{size}",
                     show_xlabel=(i == n - 1), use_log_x=True)
        axes[i, 0].set_ylabel("Proportion", fontsize=9)

    fig.legend(handles=LEGEND, loc="lower center", ncol=3, fontsize=10,
               frameon=True, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Safety Collapse by Model Size (Qwen3.5, structured_walk, baseline-refused only)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(OUTPUT_DIR / "01_model_size.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 01_model_size.png")


# ── Plot 2: Context Type Comparison ──────────────────────────────────

def plot_context_type():
    """Compare all context types on default model."""
    # Group: structured walks vs random tokens
    structured_types = [
        ("structured_walk_15", "Walk vocab=15"),
        ("structured_walk_50", "Walk vocab=50"),
        ("structured_walk_200", "Walk vocab=200"),
        ("structured_walk_1000", "Walk vocab=1000"),
    ]
    random_types = [
        ("random_tokens_2", "Random vocab=2"),
        ("random_tokens_3", "Random vocab=3"),
        ("random_tokens_5", "Random vocab=5"),
        ("random_tokens_8", "Random vocab=8"),
        ("random_tokens_10", "Random vocab=10"),
        ("random_tokens_12", "Random vocab=12"),
        ("random_tokens_15", "Random vocab=15"),
        ("random_tokens_50", "Random vocab=50"),
        ("random_tokens_200", "Random vocab=200"),
        ("random_tokens_1000", "Random vocab=1000"),
    ]
    other_types = [
        ("repeated_token", "Repeated token"),
        ("least_probable_tokens", "Least probable"),
        ("lorem_ipsum", "Lorem ipsum"),
        ("natural_books", "Natural books"),
    ]

    all_types = structured_types + random_types + other_types

    available = {}
    for ctx_type, label in all_types:
        d = load_judged(V2_DIR / "context_type" / ctx_type)
        if d:
            # For context_type experiments, condition = "{ctx_type}_raw"
            available[label] = (d, f"{ctx_type}_raw")

    if not available:
        print("  SKIP: context_type (no data)")
        return

    order = [label for _, label in all_types if label in available]
    n = len(order)

    fig, axes = plt.subplots(n, 1, figsize=(14, 2.8 * n), squeeze=False)
    for i, label in enumerate(order):
        data, condition = available[label]
        stats = three_way_split(data, condition=condition)
        if not stats:
            stats = three_way_split(data, condition=condition, baseline_only=False)
        draw_stacked(axes[i, 0], stats, label,
                     show_xlabel=(i == n - 1), use_log_x=True)
        axes[i, 0].set_ylabel("Proportion", fontsize=9)

    fig.legend(handles=LEGEND, loc="lower center", ncol=3, fontsize=10,
               frameon=True, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Safety Collapse by Context Type (Qwen3.5-9B, baseline-refused only)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(OUTPUT_DIR / "02_context_type.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 02_context_type.png")


# ── Plot 3: Structured Walk vs Random Tokens (paired comparison) ─────

def plot_structured_vs_random():
    """Direct comparison: structured walk vs random tokens at same vocab size."""
    vocab_sizes = [15, 50, 200, 1000]

    available_pairs = []
    for v in vocab_sizes:
        walk_data = load_judged(V2_DIR / "context_type" / f"structured_walk_{v}")
        rand_data = load_judged(V2_DIR / "context_type" / f"random_tokens_{v}")
        if walk_data and rand_data:
            available_pairs.append((v, walk_data, rand_data))

    if not available_pairs:
        print("  SKIP: structured_vs_random (no paired data)")
        return

    n = len(available_pairs)
    fig, axes = plt.subplots(n, 2, figsize=(16, 3.0 * n), squeeze=False)

    for i, (vocab, walk_data, rand_data) in enumerate(available_pairs):
        walk_stats = three_way_split(walk_data, condition=f"structured_walk_{vocab}_raw")
        rand_stats = three_way_split(rand_data, condition=f"random_tokens_{vocab}_raw")
        if not walk_stats:
            walk_stats = three_way_split(walk_data, condition=f"structured_walk_{vocab}_raw",
                                         baseline_only=False)
        if not rand_stats:
            rand_stats = three_way_split(rand_data, condition=f"random_tokens_{vocab}_raw",
                                         baseline_only=False)

        draw_stacked(axes[i, 0], walk_stats, f"Structured Walk (vocab={vocab})",
                     show_xlabel=(i == n - 1), use_log_x=True)
        draw_stacked(axes[i, 1], rand_stats, f"Random Tokens (vocab={vocab})",
                     show_xlabel=(i == n - 1), use_log_x=True)
        axes[i, 0].set_ylabel("Proportion", fontsize=9)

    fig.legend(handles=LEGEND, loc="lower center", ncol=3, fontsize=10,
               frameon=True, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Structure vs Randomness at Same Vocabulary Size (Qwen3.5-9B)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(OUTPUT_DIR / "03_structured_vs_random.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 03_structured_vs_random.png")


# ── Plot 4: Architecture Comparison ──────────────────────────────────

def plot_architecture():
    """Qwen3.5-9B vs Llama-3.3-70B vs OLMo-3-7B."""
    models = {
        "Qwen3.5-9B": V2_DIR / "architecture" / "qwen35_9b",
        "Llama-3.3-70B": V2_DIR / "architecture" / "llama33_70b",
        "OLMo-3-7B": V2_DIR / "architecture" / "olmo3_7b",
    }

    # v1 fallbacks
    v1_fallbacks = {
        "Qwen3.5-9B": V1_DIR / "model_gen" / "qwen35_9b",
        "Llama-3.3-70B": V1_DIR / "architecture" / "llama33_70b",
    }

    available = {}
    for label, path in models.items():
        d = load_judged(path)
        if not d and label in v1_fallbacks:
            d = load_judged(v1_fallbacks[label])
        if d:
            available[label] = d

    order = [m for m in models if m in available]
    n = len(order)
    if n == 0:
        print("  SKIP: architecture (no data)")
        return

    fig, axes = plt.subplots(n, 1, figsize=(14, 3.2 * n), squeeze=False)
    for i, model in enumerate(order):
        stats = three_way_split(available[model])
        if not stats:
            stats = three_way_split(available[model], baseline_only=False)
        draw_stacked(axes[i, 0], stats, model,
                     show_xlabel=(i == n - 1), use_log_x=True)
        axes[i, 0].set_ylabel("Proportion", fontsize=9)

    fig.legend(handles=LEGEND, loc="lower center", ncol=3, fontsize=10,
               frameon=True, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Architecture Comparison (structured_walk, baseline-refused only)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(OUTPUT_DIR / "04_architecture.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 04_architecture.png")


# ── Plot 5: Thinking Mode Effect ────────────────────────────────────

def plot_thinking_mode():
    """Compare structured_walk_15 with vs without thinking mode."""
    no_thinking = load_judged(V2_DIR / "context_type" / "structured_walk_15")
    with_thinking = load_judged(V2_DIR / "context_type" / "structured_walk_15_thinking")

    if not no_thinking or not with_thinking:
        print("  SKIP: thinking_mode (missing data)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 4), squeeze=False)

    stats_no = three_way_split(no_thinking, condition="structured_walk_15_raw")
    stats_yes = three_way_split(with_thinking, condition="structured_walk_15_thinking_raw")
    if not stats_no:
        stats_no = three_way_split(no_thinking, condition="structured_walk_15_raw",
                                   baseline_only=False)
    if not stats_yes:
        stats_yes = three_way_split(with_thinking, condition="structured_walk_15_thinking_raw",
                                    baseline_only=False)

    draw_stacked(axes[0, 0], stats_no, "Thinking OFF (default)", use_log_x=True)
    draw_stacked(axes[0, 1], stats_yes, "Thinking ON", use_log_x=True)
    axes[0, 0].set_ylabel("Proportion", fontsize=9)

    fig.legend(handles=LEGEND, loc="lower center", ncol=3, fontsize=10,
               frameon=True, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Thinking Mode Effect on Safety (Qwen3.5-9B, structured_walk_15)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    fig.savefig(OUTPUT_DIR / "05_thinking_mode.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 05_thinking_mode.png")


# ── Plot 6: Compliance Heatmap ───────────────────────────────────────

def plot_compliance_heatmap():
    """Heatmap: model × context length → genuine compliance rate."""
    model_data = {}

    # v2 model_size
    for short, label in [("qwen35_0.8b", "Qwen3.5-0.8B"),
                         ("qwen35_2b", "Qwen3.5-2B"),
                         ("qwen35_4b", "Qwen3.5-4B"),
                         ("qwen35_9b", "Qwen3.5-9B"),
                         ("qwen35_27b", "Qwen3.5-27B")]:
        d = load_judged(V2_DIR / "model_size" / short)
        if d:
            model_data[label] = d

    # v2 architecture
    for short, label in [("llama33_70b", "Llama-3.3-70B"),
                         ("olmo3_7b", "OLMo-3-7B")]:
        d = load_judged(V2_DIR / "architecture" / short)
        if d:
            model_data[label] = d

    if len(model_data) < 2:
        print("  SKIP: compliance heatmap (need >= 2 models)")
        return

    model_order = list(model_data.keys())
    all_stats = {m: three_way_split(model_data[m]) for m in model_order}
    all_lengths = sorted(set(l for s in all_stats.values() for l in s))

    if not all_lengths:
        print("  SKIP: compliance heatmap (no length data)")
        return

    matrix = np.full((len(model_order), len(all_lengths)), np.nan)
    for i, m in enumerate(model_order):
        for j, l in enumerate(all_lengths):
            if l in all_stats[m]:
                matrix[i, j] = all_stats[m][l]["compliance"]

    fig, ax = plt.subplots(figsize=(16, max(4, len(model_order) * 0.8 + 2)))
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
    fig.savefig(OUTPUT_DIR / "06_compliance_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 06_compliance_heatmap.png")


# ── Plot 7: Collapse Metric vs Safety (scatter) ─────────────────────

def plot_collapse_vs_safety():
    """Scatter: cos_sim vs compliance rate across all models."""
    all_experiments = {}

    # Collect from model_size
    for short, label in [("qwen35_0.8b", "Qwen3.5-0.8B"),
                         ("qwen35_2b", "Qwen3.5-2B"),
                         ("qwen35_4b", "Qwen3.5-4B"),
                         ("qwen35_9b", "Qwen3.5-9B"),
                         ("qwen35_27b", "Qwen3.5-27B")]:
        d = load_judged(V2_DIR / "model_size" / short)
        if d:
            all_experiments[label] = d

    # Architecture
    for short, label in [("llama33_70b", "Llama-3.3-70B"),
                         ("olmo3_7b", "OLMo-3-7B")]:
        d = load_judged(V2_DIR / "architecture" / short)
        if d:
            all_experiments[label] = d

    if not all_experiments:
        print("  SKIP: collapse_vs_safety (no data)")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_experiments)))

    for (label, data), color in zip(all_experiments.items(), colors):
        stats = three_way_split(data)
        if not stats:
            continue

        for length, s in stats.items():
            trials = [r for r in data
                      if r.get("condition", "").startswith("structured_walk")
                      and r["context_length"] == length and r.get("baseline_refused")]
            if not trials:
                continue

            cos_sims = []
            for r in trials:
                cm = r.get("collapse_metrics", {})
                if cm:
                    last_key = sorted(cm.keys())[-1] if cm else None
                    if last_key and cm[last_key]:
                        cos_sims.append(cm[last_key].get("avg_cos_sim", 0))

            if cos_sims:
                avg_cos = np.mean(cos_sims)
                ax.scatter(avg_cos, s["compliance"], c=[color], s=60, alpha=0.7,
                           label=label if length == min(stats.keys()) else "")

    ax.set_xlabel("Collapse (cos_sim)", fontsize=12)
    ax.set_ylabel("Genuine Compliance Rate", fontsize=12)
    ax.set_title("Collapse Metric vs Safety Degradation\n"
                 "(each point = one model x context length)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(-0.02, 0.65)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "07_collapse_vs_compliance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 07_collapse_vs_compliance.png")


# ── Plot 8: Context Length (full 18-point) ───────────────────────────

def plot_context_length():
    """Full 18-point context length sweep on default model."""
    # Try v2 first, then v1
    data = load_judged(V2_DIR / "context_length" / "qwen35_9b")
    if not data:
        data = load_judged(V2_DIR / "model_size" / "qwen35_9b")
    if not data:
        data = load_judged(V1_DIR / "model_gen" / "qwen35_9b")
    if not data:
        print("  SKIP: context_length (no data)")
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), squeeze=False)

    # Structured walk
    sw_stats = three_way_split(data)
    if not sw_stats:
        sw_stats = three_way_split(data, baseline_only=False)
    draw_stacked(axes[0, 0], sw_stats, "Qwen3.5-9B: structured_walk_15",
                 show_xlabel=False, use_log_x=True)
    axes[0, 0].set_ylabel("Proportion", fontsize=9)

    # Natural books
    nb_stats = three_way_split(data, condition="natural_books_raw")
    if not nb_stats:
        nb_stats = three_way_split(data, condition="natural_books_raw", baseline_only=False)
    draw_stacked(axes[1, 0], nb_stats, "Qwen3.5-9B: natural_books",
                 use_log_x=True)
    axes[1, 0].set_ylabel("Proportion", fontsize=9)

    fig.legend(handles=LEGEND, loc="lower center", ncol=3, fontsize=10,
               frameon=True, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Full Context Length Sweep (Qwen3.5-9B, 18-point dense log scale)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(OUTPUT_DIR / "08_context_length.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 08_context_length.png")


# ── Plot 9: Vocab Size Sweep (T-030) ─────────────────────────────────

def plot_vocab_size_sweep():
    """Sweep compliance rate vs vocabulary size (T-030).

    Shows how token diversity (1 to 1000) affects safety degradation.
    Includes repeated_token (vocab=1) through random_tokens_1000.
    """
    vocab_map = [
        (1, "repeated_token", "Repeated (1)"),
        (2, "random_tokens_2", "Random (2)"),
        (3, "random_tokens_3", "Random (3)"),
        (5, "random_tokens_5", "Random (5)"),
        (8, "random_tokens_8", "Random (8)"),
        (10, "random_tokens_10", "Random (10)"),
        (12, "random_tokens_12", "Random (12)"),
        (15, "random_tokens_15", "Random (15)"),
        (50, "random_tokens_50", "Random (50)"),
        (200, "random_tokens_200", "Random (200)"),
        (1000, "random_tokens_1000", "Random (1000)"),
    ]

    # Load available data
    available = []
    for vocab, ctx_type, label in vocab_map:
        d = load_judged(V2_DIR / "context_type" / ctx_type)
        if not d:
            # Try unjudged
            p = V2_DIR / "context_type" / ctx_type / "all_results.json"
            if p.exists():
                with open(p) as f:
                    d = json.load(f)
        if d:
            condition = f"{ctx_type}_raw"
            available.append((vocab, label, d, condition))

    if len(available) < 3:
        print("  SKIP: vocab_size_sweep (< 3 vocab sizes available)")
        return

    # Target context lengths to show
    target_lengths = [100, 500, 2000, 10000, 50000]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(target_lengths)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: compliance rate vs vocab size at each context length
    for li, tgt_len in enumerate(target_lengths):
        vocabs, compliances = [], []
        for vocab, label, data, condition in available:
            stats = three_way_split(data, condition=condition)
            # Find closest available length
            if tgt_len in stats:
                compliances.append(stats[tgt_len]["compliance"])
                vocabs.append(vocab)
        if vocabs:
            ax1.plot(vocabs, compliances, 'o-', color=colors[li],
                     label=f"{fmt_len(tgt_len)} tokens", markersize=5)

    ax1.set_xscale("log")
    ax1.set_xlabel("Vocabulary Size", fontsize=11)
    ax1.set_ylabel("Compliance Rate (baseline-refused)", fontsize=11)
    ax1.set_title("Safety Compliance vs Token Diversity", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(alpha=0.3)

    # Right: effective dimension vs vocab size at each context length
    for li, tgt_len in enumerate(target_lengths):
        vocabs, effdims = [], []
        for vocab, label, data, condition in available:
            # Get collapse metrics for this condition+length
            for r in data:
                if (r.get("condition") == condition and
                    r.get("context_length") == tgt_len):
                    cm = r.get("collapse_metrics", {})
                    # Find the last layer
                    layers = sorted(cm.keys(), key=lambda x: int(x) if x.isdigit() else 0)
                    if layers:
                        last_layer = layers[-1]
                        if cm[last_layer] and "effective_dim" in cm[last_layer]:
                            effdims.append(cm[last_layer]["effective_dim"])
                            vocabs.append(vocab)
                    break  # Only need one sample per condition
        if vocabs:
            ax2.plot(vocabs, effdims, 'o-', color=colors[li],
                     label=f"{fmt_len(tgt_len)} tokens", markersize=5)

    ax2.set_xscale("log")
    ax2.set_xlabel("Vocabulary Size", fontsize=11)
    ax2.set_ylabel("Effective Dimension (last layer)", fontsize=11)
    ax2.set_title("Representation Diversity vs Token Diversity", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.suptitle("Vocabulary Size Sweep: random_tokens_N (Qwen3.5-9B)", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUTPUT_DIR / "09_vocab_size_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 09_vocab_size_sweep.png")


# ── Plot 10: Structure Amount Sweep (T-033) ──────────────────────────

def plot_structure_amount_sweep():
    """Sweep compliance rate vs structure amount (p_intra variation, T-033).

    Shows how graph structure strength affects safety degradation,
    from fully random (p=0) to strongly clustered (p=0.95).
    """
    p_values = [0, 15, 30, 50, 65, 80, 95]
    p_labels = [f"p={p/100:.2f}" for p in p_values]

    available = []
    for p in p_values:
        if p == 80:
            # Default p_intra=0.80 is the standard structured_walk_15
            ctx_type = "structured_walk_15"
        else:
            ctx_type = f"structured_walk_15_p{p}"
        d = load_judged(V2_DIR / "context_type" / ctx_type)
        if not d:
            p2 = V2_DIR / "context_type" / ctx_type / "all_results.json"
            if p2.exists():
                with open(p2) as f:
                    d = json.load(f)
        if d:
            condition = f"{ctx_type}_raw"
            available.append((p / 100.0, f"p_intra={p/100:.2f}", d, condition))

    if len(available) < 3:
        print("  SKIP: structure_amount_sweep (< 3 p_intra values available)")
        return

    target_lengths = [100, 500, 2000, 10000, 50000]
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(target_lengths)))

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for li, tgt_len in enumerate(target_lengths):
        ps, compliances = [], []
        for p_val, label, data, condition in available:
            stats = three_way_split(data, condition=condition)
            if tgt_len in stats:
                compliances.append(stats[tgt_len]["compliance"])
                ps.append(p_val)
        if ps:
            ax.plot(ps, compliances, 'o-', color=colors[li],
                    label=f"{fmt_len(tgt_len)} tokens", markersize=6)

    ax.set_xlabel("p_intra (structure strength)", fontsize=11)
    ax.set_ylabel("Compliance Rate (baseline-refused)", fontsize=11)
    ax.set_title("Safety Compliance vs Graph Structure Strength\n"
                 "(structured_walk_15 with varying p_intra, Qwen3.5-9B)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.05, 1.05)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "10_structure_amount_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 10_structure_amount_sweep.png")


# ── Plot 11: Least Probable Tokens Comparison (T-031) ────────────────

def plot_least_probable():
    """Compare least_probable_tokens against random_tokens and structured_walk."""
    comparisons = [
        ("least_probable_tokens", "Least probable (adversarial)"),
        ("random_tokens_15", "Random vocab=15"),
        ("structured_walk_15", "Structured walk vocab=15"),
        ("repeated_token", "Repeated token"),
    ]

    available = {}
    for ctx_type, label in comparisons:
        d = load_judged(V2_DIR / "context_type" / ctx_type)
        if not d:
            p = V2_DIR / "context_type" / ctx_type / "all_results.json"
            if p.exists():
                with open(p) as f:
                    d = json.load(f)
        if d:
            available[label] = (d, f"{ctx_type}_raw")

    if "Least probable (adversarial)" not in available:
        print("  SKIP: least_probable (no least_probable_tokens data)")
        return

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)

    for i, (label, (data, condition)) in enumerate(available.items()):
        stats = three_way_split(data, condition=condition)
        if not stats:
            stats = three_way_split(data, condition=condition, baseline_only=False)
        draw_stacked(axes[0, i], stats, label, use_log_x=True)
        if i == 0:
            axes[0, i].set_ylabel("Proportion", fontsize=10)

    fig.legend(handles=LEGEND, loc="lower center", ncol=3, fontsize=10,
               frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Adversarial Context: Least Probable Tokens vs Baselines (Qwen3.5-9B)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(OUTPUT_DIR / "11_least_probable.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 11_least_probable.png")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating comprehensive v2 plots in {OUTPUT_DIR}")
    print("-" * 50)

    plot_model_size()
    plot_context_type()
    plot_structured_vs_random()
    plot_architecture()
    plot_thinking_mode()
    plot_compliance_heatmap()
    plot_collapse_vs_safety()
    plot_context_length()
    plot_vocab_size_sweep()
    plot_structure_amount_sweep()
    plot_least_probable()

    print("\nAll plots generated.")


if __name__ == "__main__":
    main()
