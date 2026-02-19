#!/usr/bin/env python3
"""
Plot context variation experiments compared to baseline conditions.

Combines:
- Original probing experiment (structured_walk, natural_books, repeated_token, 0-20K)
- Extended length (natural_books at 20K-128K)
- Misspellings v2 (10%, 25%, 50% at 20K-128K)
- Topic changes v2 (single_topic, multi_topic_300, multi_topic_1000 at 20K-128K)

Usage:
    PYTHONPATH=. python experiments/plotting/plot_context_variation_comparison.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ── Data Loading ──────────────────────────────────────────────────────────

def load_aggregated(path: str) -> dict:
    with open(path) as f:
        return json.load(f)["aggregated"]


def extract_series(aggregated: dict, condition: str) -> tuple[list, list]:
    """Extract (lengths, accuracies) for a condition, sorted by length."""
    data = aggregated.get(condition, {})
    lengths = sorted(int(k) for k in data.keys())
    accuracies = [data[str(l)]["accuracy"] * 100 for l in lengths]
    return lengths, accuracies


def extract_series_metric(aggregated: dict, condition: str, metric: str) -> tuple[list, list]:
    """Extract (lengths, metric_values) for a condition."""
    data = aggregated.get(condition, {})
    lengths = sorted(int(k) for k in data.keys())
    values = [data[str(l)].get(metric) for l in lengths]
    return lengths, values


def load_all_data():
    """Load and merge all experiment results."""
    base = Path("results")

    # Original probing experiment (0-20K)
    orig = load_aggregated(base / "probing_collapse_performance" / "results.json")

    # Extended length (20K-128K)  — merge both runs
    ext1 = load_aggregated(base / "context_variation" / "extended_length" / "results.json")
    ext2 = load_aggregated(base / "context_variation_maxctx" / "extended_length" / "results.json")

    # Misspellings v2 (20K-128K)
    misspell = load_aggregated(base / "context_variation_v2_fixed" / "misspellings" / "results.json")

    # Topic changes v2 (20K-128K)
    topics = load_aggregated(base / "context_variation_v2_fixed" / "topic_changes" / "results.json")

    # Build combined natural_books series (0-128K)
    nat_lengths = []
    nat_acc = []
    # From original: lengths 500-20000
    for l_str, data in sorted(orig.get("natural_books", {}).items(), key=lambda x: int(x[0])):
        nat_lengths.append(int(l_str))
        nat_acc.append(data["accuracy"] * 100)
    # From extended: 30K+ (skip 20K duplicate)
    for src in [ext1, ext2]:
        for l_str, data in sorted(src.get("natural_books_extended", {}).items(), key=lambda x: int(x[0])):
            l = int(l_str)
            if l not in nat_lengths:
                nat_lengths.append(l)
                nat_acc.append(data["accuracy"] * 100)
    # Sort
    order = np.argsort(nat_lengths)
    nat_lengths = [nat_lengths[i] for i in order]
    nat_acc = [nat_acc[i] for i in order]

    return {
        "orig": orig,
        "misspell": misspell,
        "topics": topics,
        "natural_books": (nat_lengths, nat_acc),
    }


# ── Plot 1: All conditions accuracy comparison ───────────────────────────

def plot_accuracy_all_conditions(data: dict, output_dir: Path):
    """Main comparison plot: accuracy vs context length for all conditions."""
    fig, ax = plt.subplots(figsize=(14, 8))

    orig = data["orig"]
    misspell = data["misspell"]
    topics = data["topics"]

    # Baselines from original experiment
    for cond, color, ls, label in [
        ("structured_walk", "#e74c3c", "-", "Structured walk"),
        ("repeated_token", "#9b59b6", "-", "Repeated token"),
    ]:
        lengths, acc = extract_series(orig, cond)
        ax.plot(lengths, acc, color=color, linewidth=2.5, linestyle=ls,
                marker="o", markersize=5, label=label, zorder=5)

    # Natural books (full range 0-128K)
    nat_l, nat_a = data["natural_books"]
    ax.plot(nat_l, nat_a, color="#2ecc71", linewidth=2.5,
            marker="o", markersize=5, label="Natural books (clean)", zorder=5)

    # Misspellings
    misspell_colors = {"misspell_10pct": "#f39c12", "misspell_25pct": "#e67e22", "misspell_50pct": "#d35400"}
    misspell_labels = {"misspell_10pct": "10% misspelling", "misspell_25pct": "25% misspelling", "misspell_50pct": "50% misspelling"}
    for cond in ["misspell_10pct", "misspell_25pct", "misspell_50pct"]:
        lengths, acc = extract_series(misspell, cond)
        ax.plot(lengths, acc, color=misspell_colors[cond], linewidth=2, linestyle="--",
                marker="s", markersize=5, label=misspell_labels[cond], zorder=4)

    # Topic changes
    topic_colors = {"multi_topic_300": "#3498db", "multi_topic_1000": "#2980b9"}
    topic_labels = {"multi_topic_300": "Multi-topic (switch/300)", "multi_topic_1000": "Multi-topic (switch/1000)"}
    for cond in ["multi_topic_300", "multi_topic_1000"]:
        lengths, acc = extract_series(topics, cond)
        ax.plot(lengths, acc, color=topic_colors[cond], linewidth=2, linestyle="-.",
                marker="^", markersize=6, label=topic_labels[cond], zorder=4)

    ax.set_xlabel("Context Length (tokens)", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title("Knowledge Retrieval Accuracy vs Context Length:\nAll Conditions", fontsize=15, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(400, 150000)
    ax.set_ylim(-2, 105)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K" if x >= 1000 else str(int(x))))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="lower left", ncol=2)
    ax.tick_params(labelsize=11)

    plt.tight_layout()
    fig.savefig(output_dir / "accuracy_all_conditions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: accuracy_all_conditions.png")


# ── Plot 2: Misspellings focus ───────────────────────────────────────────

def plot_misspellings_focus(data: dict, output_dir: Path):
    """Misspellings vs clean natural language, zoomed to 20K-128K."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    misspell = data["misspell"]
    nat_l, nat_a = data["natural_books"]

    # Filter natural books to 20K+
    nat_long = [(l, a) for l, a in zip(nat_l, nat_a) if l >= 20000]
    nl, na = zip(*nat_long) if nat_long else ([], [])

    colors = {
        "clean": "#2ecc71",
        "misspell_10pct": "#f39c12",
        "misspell_25pct": "#e67e22",
        "misspell_50pct": "#d35400",
    }
    labels = {
        "clean": "Clean (0%)",
        "misspell_10pct": "10% misspelling",
        "misspell_25pct": "25% misspelling",
        "misspell_50pct": "50% misspelling",
    }

    # Left: Accuracy
    ax1.plot(nl, na, color=colors["clean"], linewidth=2.5, marker="o", markersize=6, label=labels["clean"])
    for cond in ["misspell_10pct", "misspell_25pct", "misspell_50pct"]:
        lengths, acc = extract_series(misspell, cond)
        ax1.plot(lengths, acc, color=colors[cond], linewidth=2, marker="s", markersize=5,
                 linestyle="--", label=labels[cond])

    ax1.set_xlabel("Context Length (tokens)", fontsize=12)
    ax1.set_ylabel("Accuracy (%)", fontsize=12)
    ax1.set_title("Accuracy: Misspellings vs Clean", fontsize=13, fontweight="bold")
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    ax1.set_ylim(70, 102)

    # Right: Cosine Similarity
    nat_cos = []
    for src_name in ["natural_books"]:
        orig_data = data["orig"].get(src_name, {})
        for l_str in sorted(orig_data.keys(), key=int):
            l = int(l_str)
            if l >= 20000:
                nat_cos.append((l, orig_data[l_str].get("collapse_cos_sim_mean")))

    # Add from extended
    for src in [
        load_aggregated(Path("results/context_variation/extended_length/results.json")),
        load_aggregated(Path("results/context_variation_maxctx/extended_length/results.json")),
    ]:
        for l_str, d in sorted(src.get("natural_books_extended", {}).items(), key=lambda x: int(x[0])):
            l = int(l_str)
            if l >= 20000 and not any(x[0] == l for x in nat_cos):
                nat_cos.append((l, d.get("collapse_cos_sim_mean")))

    nat_cos.sort()
    if nat_cos:
        nc_l, nc_v = zip(*nat_cos)
        ax2.plot(nc_l, nc_v, color=colors["clean"], linewidth=2.5, marker="o", markersize=6, label=labels["clean"])

    for cond in ["misspell_10pct", "misspell_25pct", "misspell_50pct"]:
        lengths, vals = extract_series_metric(misspell, cond, "collapse_cos_sim_mean")
        ax2.plot(lengths, vals, color=colors[cond], linewidth=2, marker="s", markersize=5,
                 linestyle="--", label=labels[cond])

    ax2.set_xlabel("Context Length (tokens)", fontsize=12)
    ax2.set_ylabel("Cosine Similarity (L27)", fontsize=12)
    ax2.set_title("Collapse: Misspellings vs Clean", fontsize=13, fontweight="bold")
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / "misspellings_focus.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: misspellings_focus.png")


# ── Plot 3: Topic changes focus ──────────────────────────────────────────

def plot_topic_changes_focus(data: dict, output_dir: Path):
    """Topic changes vs single topic, zoomed to 20K-128K."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    topics = data["topics"]

    colors = {
        "single_topic": "#2ecc71",
        "multi_topic_300": "#3498db",
        "multi_topic_1000": "#2980b9",
    }
    labels = {
        "single_topic": "Single topic (one book)",
        "multi_topic_300": "Multi-topic (switch every 300)",
        "multi_topic_1000": "Multi-topic (switch every 1000)",
    }

    # Left: Accuracy
    for cond in ["single_topic", "multi_topic_300", "multi_topic_1000"]:
        lengths, acc = extract_series(topics, cond)
        marker = "o" if cond == "single_topic" else "^"
        ls = "-" if cond == "single_topic" else "-."
        ax1.plot(lengths, acc, color=colors[cond], linewidth=2.5, marker=marker,
                 markersize=6, linestyle=ls, label=labels[cond])

    ax1.set_xlabel("Context Length (tokens)", fontsize=12)
    ax1.set_ylabel("Accuracy (%)", fontsize=12)
    ax1.set_title("Accuracy: Topic Switching", fontsize=13, fontweight="bold")
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_ylim(78, 102)

    # Right: Cosine Similarity
    for cond in ["single_topic", "multi_topic_300", "multi_topic_1000"]:
        lengths, vals = extract_series_metric(topics, cond, "collapse_cos_sim_mean")
        marker = "o" if cond == "single_topic" else "^"
        ls = "-" if cond == "single_topic" else "-."
        ax2.plot(lengths, vals, color=colors[cond], linewidth=2.5, marker=marker,
                 markersize=6, linestyle=ls, label=labels[cond])

    ax2.set_xlabel("Context Length (tokens)", fontsize=12)
    ax2.set_ylabel("Cosine Similarity (L27)", fontsize=12)
    ax2.set_title("Collapse: Topic Switching", fontsize=13, fontweight="bold")
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(output_dir / "topic_changes_focus.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: topic_changes_focus.png")


# ── Plot 4: 128K comparison bar chart ─────────────────────────────────────

def plot_128k_comparison(data: dict, output_dir: Path):
    """Bar chart comparing all conditions at 128K tokens."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Gather 128K accuracies
    conditions = []
    accuracies = []
    colors_list = []

    # From original (extrapolate structured_walk and repeated_token at 20K as their "max")
    orig = data["orig"]
    for cond, label, color in [
        ("structured_walk", "Structured walk\n(20K)", "#e74c3c"),
        ("repeated_token", "Repeated token\n(20K)", "#9b59b6"),
    ]:
        d = orig.get(cond, {}).get("20000")
        if d:
            conditions.append(label)
            accuracies.append(d["accuracy"] * 100)
            colors_list.append(color)

    # Natural books at 128K
    nat_l, nat_a = data["natural_books"]
    for l, a in zip(nat_l, nat_a):
        if l == 128000:
            conditions.append("Natural books\n(clean, 128K)")
            accuracies.append(a)
            colors_list.append("#2ecc71")

    # Misspellings at 128K
    misspell = data["misspell"]
    for cond, label, color in [
        ("misspell_10pct", "10% misspelling\n(128K)", "#f39c12"),
        ("misspell_25pct", "25% misspelling\n(128K)", "#e67e22"),
        ("misspell_50pct", "50% misspelling\n(128K)", "#d35400"),
    ]:
        d = misspell.get(cond, {}).get("128000")
        if d:
            conditions.append(label)
            accuracies.append(d["accuracy"] * 100)
            colors_list.append(color)

    # Topic changes at 128K
    topics = data["topics"]
    for cond, label, color in [
        ("multi_topic_300", "Multi-topic\n(300, 128K)", "#3498db"),
        ("multi_topic_1000", "Multi-topic\n(1000, 128K)", "#2980b9"),
    ]:
        d = topics.get(cond, {}).get("128000")
        if d:
            conditions.append(label)
            accuracies.append(d["accuracy"] * 100)
            colors_list.append(color)

    bars = ax.bar(range(len(conditions)), accuracies, color=colors_list, edgecolor="white", linewidth=0.5)

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Knowledge Retrieval at Maximum Context Length", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.axhline(y=100, color="gray", linestyle=":", alpha=0.5)
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    fig.savefig(output_dir / "128k_comparison_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 128k_comparison_bar.png")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    output_dir = Path("results/context_variation_v2_fixed/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading all experiment data...")
    data = load_all_data()
    print("Generating plots...")

    plot_accuracy_all_conditions(data, output_dir)
    plot_misspellings_focus(data, output_dir)
    plot_topic_changes_focus(data, output_dir)
    plot_128k_comparison(data, output_dir)

    print("\nAll plots generated.")


if __name__ == "__main__":
    main()
