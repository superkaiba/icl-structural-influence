#!/usr/bin/env python3
"""
Publication-quality plots for 5 new context rot experiments.

Generates:
  1. Context Type Comparison (2x2 grid)
  2. Jailbreak Amplification (side-by-side)
  3. Persona Injection (grouped bar)
  4. What Causes Collapse? (scatter: cos_sim vs degradation)
  5. Collapse Metric by Context Type (cos_sim vs context length)

Usage:
    PYTHONPATH=. python experiments/plotting/plot_new_context_experiments.py
"""

import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
NEW_CTX_DIR = PROJECT_ROOT / "results" / "new_context_types"
ORIGINAL_DIR = PROJECT_ROOT / "results" / "safety_collapse" / "raw"
GRANULARITY_DIR = (
    PROJECT_ROOT / "results" / "safety_collapse_sweep"
    / "context_granularity" / "qwen25_7b_fillin" / "raw"
)
OUTPUT_DIR = NEW_CTX_DIR / "plots"

# ── Style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Colorblind-friendly palette (Wong 2011 + extensions)
COLORS = {
    "structured_walk": "#D55E00",       # vermilion
    "repeated_token": "#CC79A7",        # reddish purple
    "natural_books": "#009E73",         # bluish green
    "shuffled_books": "#56B4E9",        # sky blue
    "code_python": "#0072B2",           # blue
    "code_json": "#E69F00",             # orange
    "template_small_vocab": "#F0E442",  # yellow (not used in these plots)
}

MARKERS = {
    "structured_walk": "o",
    "repeated_token": "D",
    "natural_books": "s",
    "shuffled_books": "^",
    "code_python": "v",
    "code_json": "P",
}

LABELS = {
    "structured_walk": "Structured walk",
    "repeated_token": "Repeated token",
    "natural_books": "Natural books",
    "shuffled_books": "Shuffled books",
    "code_python": "Python code",
    "code_json": "JSON",
}


# ── Data loading ─────────────────────────────────────────────────────────

def load_raw_trials(raw_dir: Path) -> list[dict]:
    """Load all trial JSON files from a raw/ directory."""
    all_items = []
    if not raw_dir.exists():
        return all_items
    for fname in sorted(raw_dir.iterdir()):
        if fname.suffix != ".json":
            continue
        with open(fname) as f:
            data = json.load(f)
        if isinstance(data, list):
            all_items.extend(data)
        elif isinstance(data, dict):
            all_items.append(data)
    return all_items


def compute_degradation_rate(items: list[dict]) -> float | None:
    """Fraction of baseline-refused prompts that became safety_degraded."""
    baseline_refused = [it for it in items if it.get("baseline_refused")]
    if not baseline_refused:
        return None
    degraded = sum(1 for it in baseline_refused if it.get("safety_degraded"))
    return degraded / len(baseline_refused)


def compute_cos_sim_last_layer(items: list[dict]) -> float | None:
    """Mean cos_sim at the last layer (27) across items."""
    vals = []
    for it in items:
        cm = it.get("collapse_metrics", {})
        # Try layer "27" (last for Qwen2.5-7B)
        layer_data = cm.get("27") or cm.get(27)
        if layer_data and layer_data.get("avg_cos_sim") is not None:
            vals.append(layer_data["avg_cos_sim"])
    if not vals:
        return None
    return float(np.mean(vals))


def group_by_condition_and_length(items: list[dict]) -> dict:
    """Group items into {base_context_type: {context_length: [items]}}."""
    grouped = defaultdict(lambda: defaultdict(list))
    for it in items:
        ctx_type = it.get("base_context_type", "unknown")
        ctx_len = it.get("context_length", 0)
        grouped[ctx_type][ctx_len].append(it)
    return grouped


def build_degradation_curve(grouped: dict, ctx_type: str) -> tuple[list, list, list]:
    """Return (lengths, rates, stderrs) for a context type."""
    if ctx_type not in grouped:
        return [], [], []
    lengths = sorted(grouped[ctx_type].keys())
    rates, stderrs = [], []
    for l in lengths:
        items = grouped[ctx_type][l]
        # Group by trial to compute per-trial degradation, then mean+stderr
        trials = defaultdict(list)
        for it in items:
            trials[it.get("trial_idx", 0)].append(it)
        trial_rates = []
        for t_items in trials.values():
            r = compute_degradation_rate(t_items)
            if r is not None:
                trial_rates.append(r)
        if trial_rates:
            rates.append(np.mean(trial_rates))
            stderrs.append(np.std(trial_rates) / max(np.sqrt(len(trial_rates)), 1))
        else:
            rates.append(np.nan)
            stderrs.append(0)
    return lengths, rates, stderrs


def build_cos_sim_curve(grouped: dict, ctx_type: str) -> tuple[list, list, list]:
    """Return (lengths, cos_sims, stderrs) for a context type."""
    if ctx_type not in grouped:
        return [], [], []
    lengths = sorted(grouped[ctx_type].keys())
    means, stderrs = [], []
    for l in lengths:
        items = grouped[ctx_type][l]
        trials = defaultdict(list)
        for it in items:
            trials[it.get("trial_idx", 0)].append(it)
        trial_vals = []
        for t_items in trials.values():
            v = compute_cos_sim_last_layer(t_items)
            if v is not None:
                trial_vals.append(v)
        if trial_vals:
            means.append(np.mean(trial_vals))
            stderrs.append(np.std(trial_vals) / max(np.sqrt(len(trial_vals)), 1))
        else:
            means.append(np.nan)
            stderrs.append(0)
    return lengths, means, stderrs


# ── Load all data ────────────────────────────────────────────────────────

def load_all_data():
    """Load data from all experiment directories and return grouped dicts."""
    data = {}

    # 1. Original safety collapse (structured_walk, natural_books without injection)
    original = load_raw_trials(ORIGINAL_DIR)
    # Filter to raw wrapping only
    original = [it for it in original if it.get("wrapping_mode", "raw") == "raw"]
    data["original"] = group_by_condition_and_length(original)

    # 2. Repeated token
    repeated = load_raw_trials(NEW_CTX_DIR / "repeated_token" / "raw")
    data["repeated_token"] = group_by_condition_and_length(repeated)

    # 3. Shuffled books (includes natural_books control)
    shuffled = load_raw_trials(NEW_CTX_DIR / "shuffled_books" / "raw")
    data["shuffled_books"] = group_by_condition_and_length(shuffled)

    # 4. Jailbreak stacking
    jailbreak = load_raw_trials(NEW_CTX_DIR / "jailbreak_stacking" / "raw")
    data["jailbreak"] = group_by_condition_and_length(jailbreak)

    # 5. Persona injection (combine both dirs)
    persona = load_raw_trials(NEW_CTX_DIR / "persona_injection" / "raw")
    persona += load_raw_trials(NEW_CTX_DIR / "persona_injection_extended" / "raw")
    data["persona"] = group_by_condition_and_length(persona)

    # 6. Code language (includes natural_books control)
    code = load_raw_trials(NEW_CTX_DIR / "code_language" / "raw")
    data["code"] = group_by_condition_and_length(code)

    # 7. Context granularity fill-in (extra structured_walk lengths)
    if GRANULARITY_DIR.exists():
        gran = load_raw_trials(GRANULARITY_DIR)
        gran = [it for it in gran if it.get("wrapping_mode", "raw") == "raw"]
        data["granularity"] = group_by_condition_and_length(gran)

    return data


def _plot_line(ax, lengths, rates, stderrs, ctx_type, **kwargs):
    """Plot a line with error band for a context type."""
    color = COLORS.get(ctx_type, "#999999")
    marker = MARKERS.get(ctx_type, "o")
    label = LABELS.get(ctx_type, ctx_type)
    label = kwargs.pop("label", label)
    lw = kwargs.pop("lw", 2)
    ms = kwargs.pop("ms", 6)

    valid = [(l, r, s) for l, r, s in zip(lengths, rates, stderrs)
             if not np.isnan(r)]
    if not valid:
        return
    vl, vr, vs = zip(*valid)
    vl, vr, vs = np.array(vl), np.array(vr), np.array(vs)

    ax.plot(vl, vr, color=color, marker=marker, linewidth=lw, markersize=ms,
            label=label, zorder=3, **kwargs)
    if np.any(vs > 0):
        ax.fill_between(vl, vr - vs, vr + vs, color=color, alpha=0.15, zorder=2)


# ── Plot 1: Context Type Comparison (2x2) ───────────────────────────────

def plot_context_type_comparison(data: dict):
    """2x2 grid showing degradation rate vs context length for all types."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)

    # Build curves for all context types
    # We need a unified set of curves. Merge original + new data.
    # For structured_walk: use original (no injection)
    # For natural_books: use original (no injection)
    # For repeated_token: from repeated_token experiment
    # For shuffled_books: from shuffled_books experiment
    # For code_python, code_json: from code_language experiment

    curves = {}

    # structured_walk from original
    curves["structured_walk"] = build_degradation_curve(data["original"], "structured_walk")

    # natural_books from original
    curves["natural_books"] = build_degradation_curve(data["original"], "natural_books")

    # repeated_token
    curves["repeated_token"] = build_degradation_curve(data["repeated_token"], "repeated_token")

    # shuffled_books
    curves["shuffled_books"] = build_degradation_curve(data["shuffled_books"], "shuffled_books")

    # code
    curves["code_python"] = build_degradation_curve(data["code"], "code_python")
    curves["code_json"] = build_degradation_curve(data["code"], "code_json")

    # Panel A: Collapse-inducing
    ax = axes[0, 0]
    ax.set_title("A. Collapse-Inducing Contexts", fontweight="bold")
    for ct in ["structured_walk", "repeated_token"]:
        if curves[ct][0]:
            _plot_line(ax, *curves[ct], ct)
    ax.axhline(0, color="gray", ls=":", alpha=0.4)
    ax.legend(loc="upper left")
    ax.set_ylabel("Degradation Rate")
    ax.set_xlabel("Context Length (tokens)")

    # Panel B: Natural & shuffled
    ax = axes[0, 1]
    ax.set_title("B. Natural & Shuffled Text", fontweight="bold")
    for ct in ["natural_books", "shuffled_books"]:
        if curves[ct][0]:
            _plot_line(ax, *curves[ct], ct)
    ax.axhline(0, color="gray", ls=":", alpha=0.4)
    ax.legend(loc="upper left")
    ax.set_xlabel("Context Length (tokens)")

    # Panel C: Code
    ax = axes[1, 0]
    ax.set_title("C. Code Contexts", fontweight="bold")
    for ct in ["code_python", "code_json"]:
        if curves[ct][0]:
            _plot_line(ax, *curves[ct], ct)
    ax.axhline(0, color="gray", ls=":", alpha=0.4)
    ax.legend(loc="upper left")
    ax.set_ylabel("Degradation Rate")
    ax.set_xlabel("Context Length (tokens)")

    # Panel D: All overlaid
    ax = axes[1, 1]
    ax.set_title("D. All Context Types", fontweight="bold")
    for ct in ["structured_walk", "repeated_token", "natural_books",
               "shuffled_books", "code_python", "code_json"]:
        if curves[ct][0]:
            _plot_line(ax, *curves[ct], ct)
    ax.axhline(0, color="gray", ls=":", alpha=0.4)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlabel("Context Length (tokens)")

    for ax in axes.flat:
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.2)
        # Format x-axis as K
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}K" if x >= 1000 else f"{x:.0f}")
        )

    fig.suptitle("Safety Degradation by Context Type (Qwen2.5-7B-Instruct)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "context_type_comparison.png")
    plt.close(fig)
    print("  Saved: context_type_comparison.png")


# ── Plot 2: Jailbreak Amplification ─────────────────────────────────────

def plot_jailbreak_amplification(data: dict):
    """Side-by-side: degradation WITH vs WITHOUT jailbreak for two context types."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

    for ax_idx, ctx_type in enumerate(["structured_walk", "natural_books"]):
        ax = axes[ax_idx]
        color = COLORS[ctx_type]

        # Without jailbreak (from original)
        lengths_no, rates_no, stds_no = build_degradation_curve(
            data["original"], ctx_type
        )
        # With jailbreak
        lengths_jb, rates_jb, stds_jb = build_degradation_curve(
            data["jailbreak"], ctx_type
        )

        if lengths_no:
            vl = np.array(lengths_no)
            vr = np.array(rates_no)
            vs = np.array(stds_no)
            ax.plot(vl, vr, color=color, marker="o", linewidth=2, markersize=7,
                    label=f"{LABELS[ctx_type]} only", alpha=0.8, ls="--")
            if np.any(vs > 0):
                ax.fill_between(vl, vr - vs, vr + vs, color=color, alpha=0.1)

        if lengths_jb:
            vl = np.array(lengths_jb)
            vr = np.array(rates_jb)
            vs = np.array(stds_jb)
            ax.plot(vl, vr, color=color, marker="s", linewidth=2.5, markersize=7,
                    label=f"{LABELS[ctx_type]} + jailbreak", alpha=1.0, ls="-")
            if np.any(vs > 0):
                ax.fill_between(vl, vr - vs, vr + vs, color=color, alpha=0.15)

        ax.axhline(0, color="gray", ls=":", alpha=0.4)
        ax.set_title(LABELS[ctx_type], fontweight="bold", fontsize=12)
        ax.set_xlabel("Context Length (tokens)")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.set_ylim(-0.05, 1.05)
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}K" if x >= 1000 else f"{x:.0f}")
        )

    axes[0].set_ylabel("Degradation Rate")

    fig.suptitle("Collapse Amplifies Simple Jailbreaks",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "jailbreak_amplification.png")
    plt.close(fig)
    print("  Saved: jailbreak_amplification.png")


# ── Plot 3: Persona Injection ───────────────────────────────────────────

def plot_persona_injection(data: dict):
    """Grouped bar: degradation WITH vs WITHOUT persona injection at 20K-50K."""
    # Persona data
    lengths_p, rates_p, stds_p = build_degradation_curve(
        data["persona"], "structured_walk"
    )

    # Without persona: combine original + granularity data for matching lengths
    # Merge original and granularity structured_walk data
    merged = defaultdict(list)
    for src_key in ["original", "granularity"]:
        if src_key in data and "structured_walk" in data[src_key]:
            for l, items in data[src_key]["structured_walk"].items():
                merged[l].extend(items)
    merged_grouped = {"structured_walk": merged}
    lengths_no, rates_no, stds_no = build_degradation_curve(
        merged_grouped, "structured_walk"
    )

    # Keep only lengths present in persona data
    persona_lengths = set(lengths_p)
    idx_no = [i for i, l in enumerate(lengths_no) if l in persona_lengths]
    lengths_no_f = [lengths_no[i] for i in idx_no]
    rates_no_f = [rates_no[i] for i in idx_no]
    stds_no_f = [stds_no[i] for i in idx_no]

    # Ensure we have data for all persona lengths
    # Build lookup for no-persona rates
    no_persona_map = {l: (r, s) for l, r, s in zip(lengths_no_f, rates_no_f, stds_no_f)}
    persona_map = {l: (r, s) for l, r, s in zip(lengths_p, rates_p, stds_p)}

    all_lengths = sorted(persona_lengths)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(all_lengths))
    width = 0.35

    # No persona bars
    no_rates = [no_persona_map.get(l, (np.nan, 0))[0] for l in all_lengths]
    no_stds = [no_persona_map.get(l, (0, 0))[1] for l in all_lengths]
    p_rates = [persona_map.get(l, (np.nan, 0))[0] for l in all_lengths]
    p_stds = [persona_map.get(l, (0, 0))[1] for l in all_lengths]

    bars1 = ax.bar(x - width/2, no_rates, width,
                   yerr=no_stds, capsize=5,
                   color=COLORS["structured_walk"], alpha=0.6,
                   label="Structured walk only", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width/2, p_rates, width,
                   yerr=p_stds, capsize=5,
                   color="#7B2D8E", alpha=0.85,
                   label="Structured walk + persona injection",
                   edgecolor="white", linewidth=0.5)

    # Add value labels on bars
    for bar_group in [bars1, bars2]:
        for bar in bar_group:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                        f"{h:.0%}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("Degradation Rate")
    ax.set_title("Persona Injection After Collapse (Structured Walk)",
                 fontweight="bold", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l/1000:.0f}K" for l in all_lengths])
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.2)
    ax.axhline(0, color="gray", ls=":", alpha=0.4)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "persona_injection.png")
    plt.close(fig)
    print("  Saved: persona_injection.png")


# ── Plot 4: What Causes Collapse? (cos_sim vs degradation) ──────────────

def plot_collapse_vs_degradation(data: dict):
    """Scatter: cos_sim (last layer) vs degradation rate, colored by type."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Collect all (ctx_type, length) -> (cos_sim, degradation) points
    points = []  # (ctx_type, length, cos_sim, degradation)

    # Sources for each context type
    sources = {
        "structured_walk": [("original", "structured_walk")],
        "natural_books": [("original", "natural_books")],
        "repeated_token": [("repeated_token", "repeated_token")],
        "shuffled_books": [("shuffled_books", "shuffled_books")],
        "code_python": [("code", "code_python")],
        "code_json": [("code", "code_json")],
    }

    for ctx_type, source_list in sources.items():
        for src_key, src_type in source_list:
            if src_key not in data or src_type not in data[src_key]:
                continue
            for length, items in data[src_key][src_type].items():
                cs = compute_cos_sim_last_layer(items)
                dr = compute_degradation_rate(items)
                if cs is not None and dr is not None:
                    points.append((ctx_type, length, cs, dr))

    # Plot
    for ctx_type in ["structured_walk", "repeated_token", "natural_books",
                     "shuffled_books", "code_python", "code_json"]:
        pts = [(l, cs, dr) for ct, l, cs, dr in points if ct == ctx_type]
        if not pts:
            continue
        lengths, cos_sims, deg_rates = zip(*pts)
        color = COLORS.get(ctx_type, "#999999")
        marker = MARKERS.get(ctx_type, "o")
        label = LABELS.get(ctx_type, ctx_type)
        ax.scatter(cos_sims, deg_rates, c=color, marker=marker, s=80,
                   label=label, edgecolors="white", linewidths=0.5, zorder=3)

    # Annotate a few key points
    for ctx_type, length, cs, dr in points:
        # Annotate extremes
        if (ctx_type == "repeated_token" and length == max(l for ct, l, _, _ in points if ct == "repeated_token")) or \
           (ctx_type == "structured_walk" and dr == max(d for ct, _, _, d in points if ct == "structured_walk")) or \
           (ctx_type == "natural_books" and length == max(l for ct, l, _, _ in points if ct == "natural_books")):
            ax.annotate(f"{ctx_type.replace('_', ' ')}\n{length/1000:.0f}K",
                        (cs, dr), fontsize=7, alpha=0.7,
                        xytext=(8, 8), textcoords="offset points",
                        arrowprops=dict(arrowstyle="-", alpha=0.3, lw=0.5))

    ax.set_xlabel("Cosine Similarity (Layer 27, Last Layer)")
    ax.set_ylabel("Degradation Rate")
    ax.set_title("Geometric Collapse Predicts Safety Degradation",
                 fontweight="bold", fontsize=13)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(left=None, right=1.02)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "collapse_vs_degradation.png")
    plt.close(fig)
    print("  Saved: collapse_vs_degradation.png")


# ── Plot 5: Collapse Metric by Context Type ─────────────────────────────

def plot_cos_sim_by_context(data: dict):
    """cos_sim (last layer) vs context length, one line per type."""
    fig, ax = plt.subplots(figsize=(11, 6))

    # Build curves
    curves = {}
    curves["structured_walk"] = build_cos_sim_curve(data["original"], "structured_walk")
    curves["natural_books"] = build_cos_sim_curve(data["original"], "natural_books")
    curves["repeated_token"] = build_cos_sim_curve(data["repeated_token"], "repeated_token")
    curves["shuffled_books"] = build_cos_sim_curve(data["shuffled_books"], "shuffled_books")
    curves["code_python"] = build_cos_sim_curve(data["code"], "code_python")
    curves["code_json"] = build_cos_sim_curve(data["code"], "code_json")

    for ct in ["structured_walk", "repeated_token", "natural_books",
               "shuffled_books", "code_python", "code_json"]:
        lengths, means, stderrs = curves[ct]
        if not lengths:
            continue
        color = COLORS.get(ct, "#999999")
        marker = MARKERS.get(ct, "o")
        label = LABELS.get(ct, ct)
        vl = np.array(lengths)
        vm = np.array(means)
        vs = np.array(stderrs)
        valid = ~np.isnan(vm)
        ax.plot(vl[valid], vm[valid], color=color, marker=marker,
                linewidth=2, markersize=7, label=label, zorder=3)
        if np.any(vs[valid] > 0):
            ax.fill_between(vl[valid], vm[valid] - vs[valid], vm[valid] + vs[valid],
                            color=color, alpha=0.15, zorder=2)

    # Reference lines
    ax.axhline(0.95, color="gray", ls="--", alpha=0.3, lw=1)
    ax.text(500, 0.96, "Near-degenerate", fontsize=8, color="gray", alpha=0.6)

    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("Avg Cosine Similarity (Layer 27)")
    ax.set_title("Representation Collapse by Context Type",
                 fontweight="bold", fontsize=13)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}K" if x >= 1000 else f"{x:.0f}")
    )

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "cos_sim_by_context.png")
    plt.close(fig)
    print("  Saved: cos_sim_by_context.png")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading data from {NEW_CTX_DIR} and {ORIGINAL_DIR} ...")
    data = load_all_data()

    # Print summary of loaded data
    for key, grouped in data.items():
        for ctx_type, by_len in grouped.items():
            total = sum(len(v) for v in by_len.values())
            lengths = sorted(by_len.keys())
            print(f"  [{key}] {ctx_type}: {total} items, "
                  f"lengths={[int(l) for l in lengths]}")

    print(f"\nGenerating plots -> {OUTPUT_DIR}/")

    print("\n1. Context Type Comparison (2x2 grid)")
    plot_context_type_comparison(data)

    print("\n2. Jailbreak Amplification")
    plot_jailbreak_amplification(data)

    print("\n3. Persona Injection")
    plot_persona_injection(data)

    print("\n4. Collapse vs Degradation (scatter)")
    plot_collapse_vs_degradation(data)

    print("\n5. Collapse Metric by Context Type")
    plot_cos_sim_by_context(data)

    print(f"\nAll plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
