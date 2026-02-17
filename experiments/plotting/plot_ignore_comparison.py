#!/usr/bin/env python3
"""
Comparison plots: Original vs Ignore-Instruction vs Chat-Template conditions.

Loads results from all three probing experiments to show how different context
formatting strategies affect knowledge retrieval under representational collapse.

Usage:
    PYTHONPATH=. python experiments/plotting/plot_ignore_comparison.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ── Data Loading ──────────────────────────────────────────────────────────

def load_agg(results_dir: str) -> dict:
    with open(Path(results_dir) / "results.json") as f:
        return json.load(f)["aggregated"]


def get_accuracy_series(agg: dict, ctx_type: str) -> tuple[list, list]:
    """Return (lengths, accuracies) for a given context type."""
    if ctx_type not in agg:
        return [], []
    lengths = sorted(int(k) for k in agg[ctx_type].keys())
    accs = [agg[ctx_type][str(l)]["accuracy"] for l in lengths]
    return lengths, accs


def get_logprob_series(agg: dict, ctx_type: str) -> tuple[list, list]:
    if ctx_type not in agg:
        return [], []
    lengths = sorted(int(k) for k in agg[ctx_type].keys())
    lps = [agg[ctx_type][str(l)]["mean_log_prob"] for l in lengths]
    return lengths, lps


# ── Plots ─────────────────────────────────────────────────────────────────

def plot_accuracy_comparison(orig: dict, ignore: dict, output_dir: Path):
    """Side-by-side accuracy: original vs ignore for both context types."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    pairs = [
        ("structured_walk", "structured_walk_ignore", "Structured Walk", "#e74c3c"),
        ("repeated_token", "repeated_token_ignore", "Repeated Token", "#9b59b6"),
    ]

    for idx, (orig_type, ign_type, title, color) in enumerate(pairs):
        ax = axes[idx]

        # Original
        lengths, accs = get_accuracy_series(orig, orig_type)
        if lengths:
            ax.plot(lengths, accs, color=color, marker='o', linewidth=2.5,
                    markersize=8, label="Original", linestyle='-')

        # Ignore
        lengths_ig, accs_ig = get_accuracy_series(ignore, ign_type)
        if lengths_ig:
            ax.plot(lengths_ig, accs_ig, color=color, marker='s', linewidth=2.5,
                    markersize=8, label="+ Ignore instruction", linestyle='--',
                    alpha=0.85)

        # Natural books reference
        lengths_nb, accs_nb = get_accuracy_series(orig, "natural_books")
        if lengths_nb:
            ax.plot(lengths_nb, accs_nb, color='#2ecc71', marker='^', linewidth=1.5,
                    markersize=6, label="Natural books (reference)", linestyle=':',
                    alpha=0.6)

        # Baseline
        if "no_context" in orig and "0" in orig["no_context"]:
            ax.axhline(y=orig["no_context"]["0"]["accuracy"], color='gray',
                        linestyle='--', alpha=0.4, label="Baseline (100%)")

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Context Length (tokens)", fontsize=12)
        if idx == 0:
            ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='lower left')
        ax.set_xscale('symlog', linthresh=100)
        ticks = sorted(set(lengths + lengths_ig))
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks], fontsize=9)

    fig.suptitle("Does an Ignore Instruction Mitigate Collapse?",
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "accuracy_ignore_comparison.png", dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print("  Saved: accuracy_ignore_comparison.png")


def plot_accuracy_delta(orig: dict, ignore: dict, output_dir: Path):
    """Bar chart: accuracy improvement from ignore instruction at each length."""
    fig, ax = plt.subplots(figsize=(10, 6))

    pairs = [
        ("structured_walk", "structured_walk_ignore", "Structured Walk", "#e74c3c"),
        ("repeated_token", "repeated_token_ignore", "Repeated Token", "#9b59b6"),
    ]

    bar_width = 0.35
    all_lengths = sorted(set(
        [int(k) for k in ignore.get("structured_walk_ignore", {}).keys()] +
        [int(k) for k in ignore.get("repeated_token_ignore", {}).keys()]
    ))
    x = np.arange(len(all_lengths))

    for i, (orig_type, ign_type, label, color) in enumerate(pairs):
        deltas = []
        for l in all_lengths:
            sl = str(l)
            orig_acc = orig.get(orig_type, {}).get(sl, {}).get("accuracy", None)
            ign_acc = ignore.get(ign_type, {}).get(sl, {}).get("accuracy", None)
            if orig_acc is not None and ign_acc is not None:
                deltas.append((ign_acc - orig_acc) * 100)  # percentage points
            else:
                deltas.append(0)

        bars = ax.bar(x + i * bar_width, deltas, bar_width, label=label,
                      color=color, alpha=0.8, edgecolor='white', linewidth=0.5)

        # Add value labels
        for bar, delta in zip(bars, deltas):
            va = 'bottom' if delta >= 0 else 'top'
            offset = 1 if delta >= 0 else -1
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                    f"{delta:+.1f}", ha='center', va=va, fontsize=9,
                    fontweight='bold')

    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels([f"{l:,}" for l in all_lengths], fontsize=11)
    ax.set_xlabel("Context Length (tokens)", fontsize=12)
    ax.set_ylabel("Accuracy Change (percentage points)", fontsize=12)
    ax.set_title("Effect of Ignore Instruction on Accuracy", fontsize=14,
                 fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(output_dir / "accuracy_delta_ignore.png", dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print("  Saved: accuracy_delta_ignore.png")


def plot_logprob_comparison(orig: dict, ignore: dict, output_dir: Path):
    """Log-prob comparison: original vs ignore for both context types."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    pairs = [
        ("structured_walk", "structured_walk_ignore", "Structured Walk", "#e74c3c"),
        ("repeated_token", "repeated_token_ignore", "Repeated Token", "#9b59b6"),
    ]

    for idx, (orig_type, ign_type, title, color) in enumerate(pairs):
        ax = axes[idx]

        lengths, lps = get_logprob_series(orig, orig_type)
        if lengths:
            ax.plot(lengths, lps, color=color, marker='o', linewidth=2.5,
                    markersize=8, label="Original", linestyle='-')

        lengths_ig, lps_ig = get_logprob_series(ignore, ign_type)
        if lengths_ig:
            ax.plot(lengths_ig, lps_ig, color=color, marker='s', linewidth=2.5,
                    markersize=8, label="+ Ignore instruction", linestyle='--',
                    alpha=0.85)

        # Baseline
        if "no_context" in orig and "0" in orig["no_context"]:
            baseline_lp = orig["no_context"]["0"]["mean_log_prob"]
            ax.axhline(y=baseline_lp, color='gray', linestyle='--', alpha=0.4,
                        label=f"Baseline ({baseline_lp:.2f})")

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Context Length (tokens)", fontsize=12)
        if idx == 0:
            ax.set_ylabel("Mean Log-Probability", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_xscale('symlog', linthresh=100)
        ticks = sorted(set(lengths + lengths_ig))
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks], fontsize=9)

    fig.suptitle("Log-Probability: Original vs Ignore Instruction",
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "logprob_ignore_comparison.png", dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print("  Saved: logprob_ignore_comparison.png")


def plot_combined_all_conditions(orig: dict, ignore: dict, output_dir: Path,
                                 chat: dict | None = None):
    """Single plot with all conditions from all experiments."""
    fig, ax = plt.subplots(figsize=(14, 8))

    conditions = [
        # (ctx_type, agg_dict, label, color, linestyle, marker)
        ("natural_books", orig, "Natural books", "#2ecc71", '-', 'o'),
        # Raw injection variants
        ("structured_walk", orig, "Struct. walk (raw)", "#e74c3c", '-', 'o'),
        ("structured_walk_ignore", ignore, "Struct. walk (raw) + ignore", "#e74c3c", '--', 's'),
        ("repeated_token", orig, "Repeated token (raw)", "#9b59b6", '-', 'o'),
        ("repeated_token_ignore", ignore, "Repeated token (raw) + ignore", "#9b59b6", '--', 's'),
    ]

    if chat is not None:
        conditions.extend([
            # Chat template variants
            ("structured_walk_chat", chat, "Struct. walk (chat)", "#e67e22", '-', 'D'),
            ("structured_walk_chat_ignore", chat, "Struct. walk (chat) + ignore", "#e67e22", '--', 'v'),
            ("repeated_token_chat", chat, "Repeated token (chat)", "#3498db", '-', 'D'),
            ("repeated_token_chat_ignore", chat, "Repeated token (chat) + ignore", "#3498db", '--', 'v'),
        ])

    for ctx_type, agg, label, color, ls, marker in conditions:
        lengths, accs = get_accuracy_series(agg, ctx_type)
        if lengths:
            ax.plot(lengths, accs, color=color, marker=marker, linewidth=2,
                    markersize=7, label=label, linestyle=ls,
                    alpha=0.9 if ls == '-' else 0.7)

    # Baseline
    if "no_context" in orig and "0" in orig["no_context"]:
        ax.axhline(y=orig["no_context"]["0"]["accuracy"], color='gray',
                    linestyle=':', alpha=0.4, label="Baseline (100%)")

    ax.set_xlabel("Context Length (tokens)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Knowledge Retrieval Under Collapse: All Conditions", fontsize=14,
                 fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='lower left', ncol=2)
    ax.set_xscale('symlog', linthresh=100)
    all_lengths = [500, 1000, 2000, 5000, 10000, 20000]
    ax.set_xticks(all_lengths)
    ax.set_xticklabels([f"{l:,}" for l in all_lengths], fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / "all_conditions_accuracy.png", dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print("  Saved: all_conditions_accuracy.png")


def plot_chat_comparison(orig: dict, chat: dict, output_dir: Path):
    """Side-by-side: raw injection vs chat template for both context types."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    pairs = [
        ("structured_walk", "structured_walk_chat", "structured_walk_chat_ignore",
         "Structured Walk", "#e74c3c", "#e67e22"),
        ("repeated_token", "repeated_token_chat", "repeated_token_chat_ignore",
         "Repeated Token", "#9b59b6", "#3498db"),
    ]

    for idx, (raw_type, chat_type, chat_ign_type, title, raw_color, chat_color) in enumerate(pairs):
        ax = axes[idx]

        # Raw injection
        lengths, accs = get_accuracy_series(orig, raw_type)
        if lengths:
            ax.plot(lengths, accs, color=raw_color, marker='o', linewidth=2.5,
                    markersize=8, label="Raw injection", linestyle='-')

        # Chat template
        lengths_c, accs_c = get_accuracy_series(chat, chat_type)
        if lengths_c:
            ax.plot(lengths_c, accs_c, color=chat_color, marker='D', linewidth=2.5,
                    markersize=8, label="Chat template", linestyle='-')

        # Chat + ignore
        lengths_ci, accs_ci = get_accuracy_series(chat, chat_ign_type)
        if lengths_ci:
            ax.plot(lengths_ci, accs_ci, color=chat_color, marker='v', linewidth=2.5,
                    markersize=8, label="Chat + ignore", linestyle='--', alpha=0.85)

        # Natural books reference
        lengths_nb, accs_nb = get_accuracy_series(orig, "natural_books")
        if lengths_nb:
            ax.plot(lengths_nb, accs_nb, color='#2ecc71', marker='^', linewidth=1.5,
                    markersize=6, label="Natural books (ref)", linestyle=':',
                    alpha=0.6)

        # Baseline
        if "no_context" in orig and "0" in orig["no_context"]:
            ax.axhline(y=orig["no_context"]["0"]["accuracy"], color='gray',
                        linestyle='--', alpha=0.4, label="Baseline (100%)")

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Context Length (tokens)", fontsize=12)
        if idx == 0:
            ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='lower left')
        ax.set_xscale('symlog', linthresh=100)
        all_ticks = sorted(set(lengths + lengths_c + lengths_ci))
        ax.set_xticks(all_ticks)
        ax.set_xticklabels([f"{t:,}" for t in all_ticks], fontsize=8, rotation=30)

    fig.suptitle("Raw Injection vs Chat Template: Effect on Knowledge Retrieval",
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "accuracy_chat_comparison.png", dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print("  Saved: accuracy_chat_comparison.png")


def plot_chat_delta(orig: dict, chat: dict, output_dir: Path):
    """Bar chart: accuracy change from chat template wrapping vs raw injection."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Compare: raw vs chat, raw vs chat+ignore
    groups = [
        ("structured_walk", "structured_walk_chat", "Struct. walk: chat vs raw", "#e67e22"),
        ("structured_walk", "structured_walk_chat_ignore", "Struct. walk: chat+ignore vs raw", "#d35400"),
        ("repeated_token", "repeated_token_chat", "Repeated: chat vs raw", "#3498db"),
        ("repeated_token", "repeated_token_chat_ignore", "Repeated: chat+ignore vs raw", "#2980b9"),
    ]

    # Use context lengths from the chat experiment
    all_lengths = sorted(set(
        int(k) for ct in ["structured_walk_chat", "repeated_token_chat"]
        if ct in chat
        for k in chat[ct].keys()
    ))
    x = np.arange(len(all_lengths))
    n_groups = len(groups)
    bar_width = 0.8 / n_groups

    for i, (raw_type, chat_type, label, color) in enumerate(groups):
        deltas = []
        for l in all_lengths:
            sl = str(l)
            raw_acc = orig.get(raw_type, {}).get(sl, {}).get("accuracy", None)
            chat_acc = chat.get(chat_type, {}).get(sl, {}).get("accuracy", None)
            if raw_acc is not None and chat_acc is not None:
                deltas.append((chat_acc - raw_acc) * 100)
            else:
                deltas.append(0)

        bars = ax.bar(x + i * bar_width, deltas, bar_width, label=label,
                      color=color, alpha=0.8, edgecolor='white', linewidth=0.5)

        for bar, delta in zip(bars, deltas):
            va = 'bottom' if delta >= 0 else 'top'
            offset = 1.5 if delta >= 0 else -1.5
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                    f"{delta:+.1f}", ha='center', va=va, fontsize=7,
                    fontweight='bold')

    ax.set_xticks(x + bar_width * (n_groups - 1) / 2)
    ax.set_xticklabels([f"{l:,}" for l in all_lengths], fontsize=11)
    ax.set_xlabel("Context Length (tokens)", fontsize=12)
    ax.set_ylabel("Accuracy Change (percentage points)", fontsize=12)
    ax.set_title("Effect of Chat Template on Accuracy (vs Raw Injection)", fontsize=14,
                 fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(output_dir / "accuracy_delta_chat.png", dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print("  Saved: accuracy_delta_chat.png")


def plot_structured_walk_all_variants(orig: dict, ignore: dict, chat: dict,
                                       output_dir: Path):
    """Focused comparison of all structured_walk variants."""
    fig, ax = plt.subplots(figsize=(12, 7))

    variants = [
        ("natural_books", orig, "Natural books (control)", "#2ecc71", '-', 'o', 2.5),
        ("structured_walk", orig, "Raw injection", "#e74c3c", '-', 'o', 2.5),
        ("structured_walk_ignore", ignore, "Raw + ignore instruction", "#e74c3c", '--', 's', 2),
        ("structured_walk_chat", chat, "Chat template", "#e67e22", '-', 'D', 2.5),
        ("structured_walk_chat_ignore", chat, "Chat + ignore instruction", "#e67e22", '--', 'v', 2),
    ]

    for ctx_type, agg, label, color, ls, marker, lw in variants:
        lengths, accs = get_accuracy_series(agg, ctx_type)
        if lengths:
            ax.plot(lengths, accs, color=color, marker=marker, linewidth=lw,
                    markersize=8, label=label, linestyle=ls,
                    alpha=0.9 if ls == '-' else 0.75)

    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.4, label="Baseline (100%)")

    ax.set_xlabel("Context Length (tokens)", fontsize=13)
    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title("Structured Walk: All Mitigation Strategies", fontsize=15,
                 fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='lower left')
    ax.set_xscale('symlog', linthresh=100)
    all_lengths = sorted(set([500, 1000, 2000, 5000, 10000, 20000]))
    ax.set_xticks(all_lengths)
    ax.set_xticklabels([f"{l:,}" for l in all_lengths], fontsize=10)

    plt.tight_layout()
    fig.savefig(output_dir / "structured_walk_all_variants.png", dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print("  Saved: structured_walk_all_variants.png")


def plot_repeated_token_all_variants(orig: dict, ignore: dict, chat: dict,
                                      output_dir: Path):
    """Focused comparison of all repeated_token variants."""
    fig, ax = plt.subplots(figsize=(12, 7))

    variants = [
        ("natural_books", orig, "Natural books (control)", "#2ecc71", '-', 'o', 2.5),
        ("repeated_token", orig, "Raw injection", "#9b59b6", '-', 'o', 2.5),
        ("repeated_token_ignore", ignore, "Raw + ignore instruction", "#9b59b6", '--', 's', 2),
        ("repeated_token_chat", chat, "Chat template", "#3498db", '-', 'D', 2.5),
        ("repeated_token_chat_ignore", chat, "Chat + ignore instruction", "#3498db", '--', 'v', 2),
    ]

    for ctx_type, agg, label, color, ls, marker, lw in variants:
        lengths, accs = get_accuracy_series(agg, ctx_type)
        if lengths:
            ax.plot(lengths, accs, color=color, marker=marker, linewidth=lw,
                    markersize=8, label=label, linestyle=ls,
                    alpha=0.9 if ls == '-' else 0.75)

    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.4, label="Baseline (100%)")

    ax.set_xlabel("Context Length (tokens)", fontsize=13)
    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title("Repeated Token: All Mitigation Strategies", fontsize=15,
                 fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='lower left')
    ax.set_xscale('symlog', linthresh=100)
    all_lengths = sorted(set([500, 1000, 2000, 5000, 10000, 20000]))
    ax.set_xticks(all_lengths)
    ax.set_xticklabels([f"{l:,}" for l in all_lengths], fontsize=10)

    plt.tight_layout()
    fig.savefig(output_dir / "repeated_token_all_variants.png", dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print("  Saved: repeated_token_all_variants.png")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    orig_dir = "results/probing_collapse_performance"
    ignore_dir = "results/probing_collapse_ignore"
    chat_dir = "results/probing_collapse_chat"
    output_dir = Path("results/probing_collapse_performance/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading original results...")
    orig = load_agg(orig_dir)
    print("Loading ignore-instruction results...")
    ignore = load_agg(ignore_dir)

    chat = None
    chat_path = Path(chat_dir) / "results.json"
    if chat_path.exists():
        print("Loading chat-template results...")
        chat = load_agg(chat_dir)

    print(f"\nGenerating comparison plots in {output_dir}")
    print("-" * 50)

    # Original comparison plots
    plot_accuracy_comparison(orig, ignore, output_dir)
    plot_accuracy_delta(orig, ignore, output_dir)
    plot_logprob_comparison(orig, ignore, output_dir)
    plot_combined_all_conditions(orig, ignore, output_dir, chat=chat)

    # Chat-specific plots (only if chat results exist)
    if chat is not None:
        plot_chat_comparison(orig, chat, output_dir)
        plot_chat_delta(orig, chat, output_dir)
        plot_structured_walk_all_variants(orig, ignore, chat, output_dir)
        plot_repeated_token_all_variants(orig, ignore, chat, output_dir)

    print("\nAll comparison plots generated.")


if __name__ == "__main__":
    main()
