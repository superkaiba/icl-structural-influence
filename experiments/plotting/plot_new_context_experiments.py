#!/usr/bin/env python3
"""
Publication-quality plots for context rot experiments with 3-band stacked bars.

Uses LLM judge classifications (refusal / compliance / incoherent) instead of
plain degradation rate.

Generates:
  1. Context Type Comparison (2x2 grid of stacked bars)
  2. Jailbreak Amplification (side-by-side stacked bars)
  3. Persona Injection (grouped stacked bars)
  4. Collapse vs Compliance (scatter: cos_sim vs genuine compliance rate)
  5. Collapse Metric by Context Type (cos_sim line plot — unchanged)

Usage:
    PYTHONPATH=. python experiments/plotting/plot_new_context_experiments.py
"""

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
NEW_CTX_DIR = PROJECT_ROOT / "results" / "new_context_types"
ORIGINAL_DIR = PROJECT_ROOT / "results" / "safety_collapse"
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

# 3-band colors
GREEN = "#2ecc71"   # refusal (safe)
RED = "#e74c3c"     # compliance (unsafe)
GRAY = "#95a5a6"    # incoherent

# Per-context-type colors (used in scatter / line plots)
COLORS = {
    "structured_walk": "#D55E00",
    "repeated_token": "#CC79A7",
    "natural_books": "#009E73",
    "shuffled_books": "#56B4E9",
    "code_python": "#0072B2",
    "code_json": "#E69F00",
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

def load_judged_data(judge_path: Path) -> list[dict]:
    """Load all_results_judged.json from a judge directory."""
    if not judge_path.exists():
        return []
    with open(judge_path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return []


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


def classify_item(item: dict) -> str:
    """Map judge_classification to one of: refusal, compliance, incoherent.

    If judge data exists:
      full_refusal, partial_refusal -> refusal
      compliance -> compliance
      incoherent, parse_error -> incoherent

    Fallback (no judge): refused -> refusal, not refused + coherent -> compliance,
    not refused + not coherent -> incoherent.
    """
    jc = item.get("judge_classification")
    if jc:
        if jc in ("full_refusal", "partial_refusal"):
            return "refusal"
        elif jc == "compliance":
            return "compliance"
        else:
            return "incoherent"
    # Fallback heuristic
    if item.get("refused"):
        return "refusal"
    if item.get("coherent_response"):
        return "compliance"
    return "incoherent"


def compute_band_proportions(items: list[dict]) -> dict:
    """From baseline-refused items, compute proportions of each classification.

    Returns {"refusal": float, "compliance": float, "incoherent": float, "n": int}.
    """
    bl_refused = [it for it in items if it.get("baseline_refused")]
    n = len(bl_refused)
    if n == 0:
        return {"refusal": 0.0, "compliance": 0.0, "incoherent": 0.0, "n": 0}
    counts = Counter(classify_item(it) for it in bl_refused)
    return {
        "refusal": counts.get("refusal", 0) / n,
        "compliance": counts.get("compliance", 0) / n,
        "incoherent": counts.get("incoherent", 0) / n,
        "n": n,
    }


def compute_cos_sim_last_layer(items: list[dict]) -> float | None:
    """Mean cos_sim at last layer (27) across items."""
    vals = []
    for it in items:
        cm = it.get("collapse_metrics", {})
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


# ── Load all data ────────────────────────────────────────────────────────

def load_all_data():
    """Load judged data from all experiment directories."""
    data = {}

    # 1. Original safety collapse (structured_walk, natural_books, no injection)
    original_items = load_judged_data(
        ORIGINAL_DIR / "judge" / "all_results_judged.json"
    )
    # Filter to raw wrapping only
    original_items = [it for it in original_items if it.get("wrapping_mode", "raw") == "raw"]
    data["original"] = group_by_condition_and_length(original_items)

    # 2. Repeated token
    rt_items = load_judged_data(
        NEW_CTX_DIR / "repeated_token" / "judge" / "all_results_judged.json"
    )
    if not rt_items:
        rt_items = load_raw_trials(NEW_CTX_DIR / "repeated_token" / "raw")
    data["repeated_token"] = group_by_condition_and_length(rt_items)

    # 3. Shuffled books
    sb_items = load_judged_data(
        NEW_CTX_DIR / "shuffled_books" / "judge" / "all_results_judged.json"
    )
    if not sb_items:
        sb_items = load_raw_trials(NEW_CTX_DIR / "shuffled_books" / "raw")
    data["shuffled_books"] = group_by_condition_and_length(sb_items)

    # 4. Jailbreak stacking
    jb_items = load_judged_data(
        NEW_CTX_DIR / "jailbreak_stacking" / "judge" / "all_results_judged.json"
    )
    if not jb_items:
        jb_items = load_raw_trials(NEW_CTX_DIR / "jailbreak_stacking" / "raw")
    data["jailbreak"] = group_by_condition_and_length(jb_items)

    # 5. Persona injection (combine both dirs)
    p_items = load_judged_data(
        NEW_CTX_DIR / "persona_injection" / "judge" / "all_results_judged.json"
    )
    p_ext = load_judged_data(
        NEW_CTX_DIR / "persona_injection_extended" / "judge" / "all_results_judged.json"
    )
    if not p_items:
        p_items = load_raw_trials(NEW_CTX_DIR / "persona_injection" / "raw")
    if not p_ext:
        p_ext = load_raw_trials(NEW_CTX_DIR / "persona_injection_extended" / "raw")
    data["persona"] = group_by_condition_and_length(p_items + p_ext)

    # 6. Code language
    code_items = load_judged_data(
        NEW_CTX_DIR / "code_language" / "judge" / "all_results_judged.json"
    )
    if not code_items:
        code_items = load_raw_trials(NEW_CTX_DIR / "code_language" / "raw")
    data["code"] = group_by_condition_and_length(code_items)

    # 7. Context granularity fill-in (extra structured_walk lengths, no judge)
    if GRANULARITY_DIR.exists():
        gran = load_raw_trials(GRANULARITY_DIR)
        gran = [it for it in gran if it.get("wrapping_mode", "raw") == "raw"]
        data["granularity"] = group_by_condition_and_length(gran)

    return data


def build_band_data(grouped: dict, ctx_type: str) -> tuple[list, list, list, list, list]:
    """Return (lengths, refusal_fracs, compliance_fracs, incoherent_fracs, n_counts)
    for a context type."""
    if ctx_type not in grouped:
        return [], [], [], [], []
    lengths = sorted(grouped[ctx_type].keys())
    ref_fracs, comp_fracs, inc_fracs, ns = [], [], [], []
    for l in lengths:
        items = grouped[ctx_type][l]
        bp = compute_band_proportions(items)
        ref_fracs.append(bp["refusal"])
        comp_fracs.append(bp["compliance"])
        inc_fracs.append(bp["incoherent"])
        ns.append(bp["n"])
    return lengths, ref_fracs, comp_fracs, inc_fracs, ns


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


# ── Stacked bar helpers ──────────────────────────────────────────────────

def draw_stacked_bar(ax, x_pos, width, refusal, compliance, incoherent,
                     edgecolor="white", linewidth=0.5, hatch=None):
    """Draw a single 3-band stacked bar at position x_pos."""
    ax.bar(x_pos, refusal, width, color=GREEN, edgecolor=edgecolor,
           linewidth=linewidth, hatch=hatch, zorder=3)
    ax.bar(x_pos, incoherent, width, bottom=refusal, color=GRAY,
           edgecolor=edgecolor, linewidth=linewidth, hatch=hatch, zorder=3)
    ax.bar(x_pos, compliance, width, bottom=refusal + incoherent, color=RED,
           edgecolor=edgecolor, linewidth=linewidth, hatch=hatch, zorder=3)


def band_legend_handles():
    """Return legend handles for the 3-band stacked bars."""
    return [
        Patch(facecolor=GREEN, edgecolor="white", label="Refusal (safe)"),
        Patch(facecolor=GRAY, edgecolor="white", label="Incoherent"),
        Patch(facecolor=RED, edgecolor="white", label="Compliance (unsafe)"),
    ]


# ── Plot 1: Context Type Comparison (2x2) ───────────────────────────────

def plot_context_type_comparison(data: dict):
    """One row per context type — each row is a simple stacked bar chart."""
    # Build band data for all context types
    bands = {}
    bands["structured_walk"] = build_band_data(data["original"], "structured_walk")
    bands["repeated_token"] = build_band_data(data["repeated_token"], "repeated_token")
    bands["natural_books"] = build_band_data(data["original"], "natural_books")
    bands["shuffled_books"] = build_band_data(data["shuffled_books"], "shuffled_books")
    bands["code_python"] = build_band_data(data["code"], "code_python")
    bands["code_json"] = build_band_data(data["code"], "code_json")

    ctx_types = ["structured_walk", "repeated_token", "natural_books",
                 "shuffled_books", "code_python", "code_json"]
    n_rows = len(ctx_types)

    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 2.5 * n_rows), sharey=True)

    for row_idx, ct in enumerate(ctx_types):
        ax = axes[row_idx]
        lengths, ref_f, comp_f, inc_f, ns = bands[ct]

        if not lengths:
            ax.set_title(LABELS[ct], fontweight="bold", loc="left")
            continue

        x_positions = np.arange(len(lengths))
        bar_width = 0.6

        for i in range(len(lengths)):
            draw_stacked_bar(ax, x_positions[i], bar_width,
                             ref_f[i], comp_f[i], inc_f[i])

        ax.set_title(LABELS[ct], fontweight="bold", loc="left", fontsize=12,
                     color=COLORS.get(ct, "black"))
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"{l/1000:.0f}K" if l >= 1000 else str(l)
                            for l in lengths])
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.2)
        if row_idx == n_rows - 1:
            ax.set_xlabel("Context Length (tokens)")
        ax.set_ylabel("Proportion")

    # Global legend for band colors
    handles = band_legend_handles()
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=11,
               bbox_to_anchor=(0.5, -0.02), frameon=True)

    fig.suptitle("Safety Response Classification by Context Type (Qwen2.5-7B-Instruct)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout(rect=[0, 0.03, 1, 0.99])
    fig.savefig(OUTPUT_DIR / "context_type_comparison.png")
    plt.close(fig)
    print("  Saved: context_type_comparison.png")


# ── Plot 2: Jailbreak Amplification ─────────────────────────────────────

def plot_jailbreak_amplification(data: dict):
    """Side-by-side panels: WITH vs WITHOUT jailbreak, stacked bars."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax_idx, ctx_type in enumerate(["structured_walk", "natural_books"]):
        ax = axes[ax_idx]

        # Without jailbreak (from original)
        lengths_no, ref_no, comp_no, inc_no, _ = build_band_data(
            data["original"], ctx_type
        )
        # With jailbreak
        lengths_jb, ref_jb, comp_jb, inc_jb, _ = build_band_data(
            data["jailbreak"], ctx_type
        )

        # Combine all lengths present in both
        all_lengths = sorted(set(lengths_no) | set(lengths_jb))
        # Filter to lengths where at least one source has data
        # Build lookup maps
        no_map = {l: (r, c, i) for l, r, c, i in
                  zip(lengths_no, ref_no, comp_no, inc_no)}
        jb_map = {l: (r, c, i) for l, r, c, i in
                  zip(lengths_jb, ref_jb, comp_jb, inc_jb)}

        # Only show lengths present in jailbreak data (the focus)
        show_lengths = sorted(set(lengths_jb))
        if not show_lengths:
            show_lengths = all_lengths

        n_lengths = len(show_lengths)
        x_positions = np.arange(n_lengths)
        bar_width = 0.35

        for g_idx, gl in enumerate(show_lengths):
            # No jailbreak bar (solid)
            if gl in no_map:
                r, c, i = no_map[gl]
                draw_stacked_bar(ax, x_positions[g_idx] - bar_width / 2 - 0.02,
                                 bar_width, r, c, i)
            # With jailbreak bar (hatched)
            if gl in jb_map:
                r, c, i = jb_map[gl]
                draw_stacked_bar(ax, x_positions[g_idx] + bar_width / 2 + 0.02,
                                 bar_width, r, c, i, hatch="//")

        ax.set_title(LABELS[ctx_type], fontweight="bold", fontsize=12)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"{l/1000:.0f}K" if l >= 1000 else str(l)
                            for l in show_lengths])
        ax.set_xlabel("Context Length (tokens)")
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.2)

        # Condition legend using hatching
        cond_handles = [
            Patch(facecolor="white", edgecolor="black", linewidth=0.8,
                  label="No jailbreak"),
            Patch(facecolor="white", edgecolor="black", linewidth=0.8,
                  hatch="//", label="+ jailbreak"),
        ]
        ax.legend(handles=cond_handles, loc="upper left", fontsize=9)

    axes[0].set_ylabel("Proportion")

    # Band legend
    handles = band_legend_handles()
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.02), frameon=True)

    fig.suptitle("Jailbreak Amplification of Context Rot",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(OUTPUT_DIR / "jailbreak_amplification.png")
    plt.close(fig)
    print("  Saved: jailbreak_amplification.png")


# ── Plot 3: Persona Injection ───────────────────────────────────────────

def plot_persona_injection(data: dict):
    """Grouped stacked bars: WITH vs WITHOUT persona injection at 20K-50K."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Persona data (structured_walk)
    lengths_p, ref_p, comp_p, inc_p, _ = build_band_data(
        data["persona"], "structured_walk"
    )
    p_map = {l: (r, c, i) for l, r, c, i in
             zip(lengths_p, ref_p, comp_p, inc_p)}

    # No persona: from original structured_walk + granularity
    merged = defaultdict(list)
    for src_key in ["original", "granularity"]:
        if src_key in data and "structured_walk" in data[src_key]:
            for l, items in data[src_key]["structured_walk"].items():
                merged[l].extend(items)
    merged_grouped = {"structured_walk": merged}
    lengths_no, ref_no, comp_no, inc_no, _ = build_band_data(
        merged_grouped, "structured_walk"
    )
    no_map = {l: (r, c, i) for l, r, c, i in
              zip(lengths_no, ref_no, comp_no, inc_no)}

    # Target lengths: union of persona lengths
    all_lengths = sorted(set(lengths_p))

    n_lengths = len(all_lengths)
    x_positions = np.arange(n_lengths)
    bar_width = 0.35

    for g_idx, gl in enumerate(all_lengths):
        # No persona bar (solid)
        if gl in no_map:
            r, c, i = no_map[gl]
            draw_stacked_bar(ax, x_positions[g_idx] - bar_width / 2 - 0.02,
                             bar_width, r, c, i)
        # With persona bar (hatched)
        if gl in p_map:
            r, c, i = p_map[gl]
            draw_stacked_bar(ax, x_positions[g_idx] + bar_width / 2 + 0.02,
                             bar_width, r, c, i, hatch="//")

    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("Proportion")
    ax.set_title("Persona Injection After Structured Walk Context",
                 fontweight="bold", fontsize=13)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{l/1000:.0f}K" for l in all_lengths])
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.2)

    # Condition legend using hatching
    cond_handles = [
        Patch(facecolor="white", edgecolor="black", linewidth=0.8,
              label="Structured walk only"),
        Patch(facecolor="white", edgecolor="black", linewidth=0.8,
              hatch="//", label="+ persona injection"),
    ]
    ax.legend(handles=cond_handles, loc="upper right", fontsize=10)

    # Band legend
    handles = band_legend_handles()
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.02), frameon=True)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(OUTPUT_DIR / "persona_injection.png")
    plt.close(fig)
    print("  Saved: persona_injection.png")


# ── Plot 4: Collapse vs Compliance (scatter) ────────────────────────────

def plot_collapse_vs_degradation(data: dict):
    """Scatter: cos_sim (last layer) vs genuine compliance rate (red fraction)."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Sources for each context type
    sources = {
        "structured_walk": [("original", "structured_walk")],
        "natural_books": [("original", "natural_books")],
        "repeated_token": [("repeated_token", "repeated_token")],
        "shuffled_books": [("shuffled_books", "shuffled_books")],
        "code_python": [("code", "code_python")],
        "code_json": [("code", "code_json")],
    }

    points = []  # (ctx_type, length, cos_sim, compliance_rate)

    for ctx_type, source_list in sources.items():
        for src_key, src_type in source_list:
            if src_key not in data or src_type not in data[src_key]:
                continue
            for length, items in data[src_key][src_type].items():
                cs = compute_cos_sim_last_layer(items)
                bp = compute_band_proportions(items)
                if cs is not None and bp["n"] > 0:
                    points.append((ctx_type, length, cs, bp["compliance"]))

    # Plot by context type
    for ctx_type in ["structured_walk", "repeated_token", "natural_books",
                     "shuffled_books", "code_python", "code_json"]:
        pts = [(l, cs, cr) for ct, l, cs, cr in points if ct == ctx_type]
        if not pts:
            continue
        lengths, cos_sims, comp_rates = zip(*pts)
        color = COLORS.get(ctx_type, "#999999")
        marker = MARKERS.get(ctx_type, "o")
        label = LABELS.get(ctx_type, ctx_type)
        ax.scatter(cos_sims, comp_rates, c=color, marker=marker, s=80,
                   label=label, edgecolors="white", linewidths=0.5, zorder=3)

    # Annotate extremes
    for ctx_type, length, cs, cr in points:
        annotate = False
        pts_of_type = [(l, c, r) for ct, l, c, r in points if ct == ctx_type]
        max_len = max(l for l, _, _ in pts_of_type)
        max_cr = max(r for _, _, r in pts_of_type)
        if length == max_len and cr > 0.01:
            annotate = True
        if cr == max_cr and cr > 0.05:
            annotate = True
        if annotate:
            ax.annotate(f"{ctx_type.replace('_', ' ')}\n{length/1000:.0f}K",
                        (cs, cr), fontsize=7, alpha=0.7,
                        xytext=(8, 8), textcoords="offset points",
                        arrowprops=dict(arrowstyle="-", alpha=0.3, lw=0.5))

    ax.set_xlabel("Cosine Similarity (Layer 27, Last Layer)")
    ax.set_ylabel("Genuine Compliance Rate (Unsafe)")
    ax.set_title("Geometric Collapse vs Actual Safety Risk",
                 fontweight="bold", fontsize=13)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(left=None, right=1.02)
    ax.set_ylim(bottom=-0.02)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "collapse_vs_degradation.png")
    plt.close(fig)
    print("  Saved: collapse_vs_degradation.png")


# ── Plot 5: Collapse Metric by Context Type (unchanged) ─────────────────

def plot_cos_sim_by_context(data: dict):
    """cos_sim (last layer) vs context length, one line per type."""
    fig, ax = plt.subplots(figsize=(11, 6))

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
            # Count baseline_refused
            bl = sum(1 for items in by_len.values()
                     for it in items if it.get("baseline_refused"))
            print(f"  [{key}] {ctx_type}: {total} items ({bl} baseline_refused), "
                  f"lengths={[int(l) for l in lengths]}")

    print(f"\nGenerating plots -> {OUTPUT_DIR}/")

    print("\n1. Context Type Comparison (2x2 grid)")
    plot_context_type_comparison(data)

    print("\n2. Jailbreak Amplification")
    plot_jailbreak_amplification(data)

    print("\n3. Persona Injection")
    plot_persona_injection(data)

    print("\n4. Collapse vs Compliance (scatter)")
    plot_collapse_vs_degradation(data)

    print("\n5. Collapse Metric by Context Type")
    plot_cos_sim_by_context(data)

    print(f"\nAll plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
