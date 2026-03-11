#!/usr/bin/env python3
"""
Combined disambiguation plot: velocity and preference trajectories with
vertical lines marking each disambiguation point.

Usage:
    python plot_disambig_combined.py --results-dir results/natural_disambig_pilot
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

LAYER_COLORS = {0: "#bdbdbd", 7: "#74c476", 14: "#fd8d3c", 21: "#6baed6", 27: "#08519c"}
DISAMBIG_COLORS = {"disambig_25pct": "#e6550d", "disambig_50pct": "#d62728", "disambig_75pct": "#8b0000"}
DISAMBIG_LABELS = {"disambig_25pct": "25%", "disambig_50pct": "50%", "disambig_75pct": "75%"}


def load_arm(results_dir: Path, arm: str):
    with open(results_dir / f"{arm}_model" / "results.json") as f:
        results = json.load(f)
    with open(results_dir / f"{arm}_model" / "config.json") as f:
        config = json.load(f)
    return results, config


def resample_trajectories(trials, layer_str, metric, n_grid=200):
    """Resample variable-length trajectories onto a common [0, 1] grid."""
    grid = np.linspace(0, 1, n_grid)
    resampled = []
    for t in trials:
        if t.get("error") is not None:
            continue
        traj = t["layers"][layer_str][metric]
        n = len(traj)
        if n < 3:
            continue
        x = np.linspace(0, 1, n)
        interp = np.interp(grid, x, traj)
        resampled.append(interp)
    if not resampled:
        return grid, np.full(n_grid, np.nan), np.full(n_grid, np.nan)
    arr = np.array(resampled)
    return grid, np.mean(arr, axis=0), np.std(arr, axis=0) / np.sqrt(len(arr))


def get_mean_disambig_frac(trials):
    """Get mean fractional position of the disambig token."""
    fracs = []
    for t in trials:
        pos = t.get("disambig_token_pos")
        total = t.get("total_tokens")
        if pos is not None and total is not None and total > 0:
            fracs.append(pos / total)
    return np.mean(fracs) if fracs else None


def plot_combined(results, config, arm_label, output_path):
    layers = config["layers"]
    disambig_conds = ["disambig_25pct", "disambig_50pct", "disambig_75pct"]
    n_grid = 200

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    ax_vel, ax_pref = axes

    # Use disambig_50pct as the "main" condition for trajectories (most data at each position).
    # Also overlay no_disambig as baseline.
    plot_conds = ["no_disambig", "disambig_50pct"]

    for cond in plot_conds:
        if cond not in results:
            continue
        trials = results[cond]["trials"]
        linestyle = "--" if cond == "no_disambig" else "-"
        alpha_base = 0.5 if cond == "no_disambig" else 1.0

        for layer in layers:
            ls = str(layer)
            color = LAYER_COLORS.get(layer, "gray")

            # Velocity
            grid, mean_v, se_v = resample_trajectories(trials, ls, "velocities", n_grid)
            label = f"L{layer}" if cond != "no_disambig" else (f"L{layer} (no disambig)" if layer == layers[-1] else None)
            ax_vel.plot(grid, mean_v, color=color, linewidth=2 if layer in [21, 27] else 1,
                        linestyle=linestyle, alpha=alpha_base, label=label)
            if cond != "no_disambig":
                ax_vel.fill_between(grid, mean_v - se_v, mean_v + se_v, color=color, alpha=0.08)

            # Preference
            grid, mean_p, se_p = resample_trajectories(trials, ls, "h1_preference", n_grid)
            ax_pref.plot(grid, mean_p, color=color, linewidth=2 if layer in [21, 27] else 1,
                         linestyle=linestyle, alpha=alpha_base, label=label)
            if cond != "no_disambig":
                ax_pref.fill_between(grid, mean_p - se_p, mean_p + se_p, color=color, alpha=0.08)

    # Vertical lines for each disambiguation point
    for cond in disambig_conds:
        if cond not in results:
            continue
        frac = get_mean_disambig_frac(results[cond]["trials"])
        if frac is None:
            continue
        color = DISAMBIG_COLORS[cond]
        label = f"disambig @ {DISAMBIG_LABELS[cond]}"
        ax_vel.axvline(x=frac, color=color, linestyle="--", linewidth=2, alpha=0.8, label=label)
        ax_pref.axvline(x=frac, color=color, linestyle="--", linewidth=2, alpha=0.8, label=label)

    # Formatting
    ax_vel.set_ylabel("Velocity (L2 norm of residual stream change)", fontsize=12)
    ax_vel.set_title(f"Velocity Trajectory — {arm_label}", fontsize=14, fontweight="bold")
    ax_vel.legend(fontsize=9, ncol=2, loc="upper right")
    ax_vel.grid(True, alpha=0.2)

    ax_pref.set_ylabel("H1 Preference (cosine similarity)", fontsize=12)
    ax_pref.set_xlabel("Normalized token position", fontsize=12)
    ax_pref.set_title(f"H1 Preference Trajectory — {arm_label}", fontsize=14, fontweight="bold")
    ax_pref.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4)
    ax_pref.legend(fontsize=9, ncol=2, loc="upper right")
    ax_pref.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_base_vs_instruct_combined(results_dir, output_path):
    """Side-by-side base vs instruct with vertical disambig lines."""
    base_results, base_config = load_arm(results_dir, "base")
    inst_results, inst_config = load_arm(results_dir, "instruct")

    layers = base_config["layers"]
    disambig_conds = ["disambig_25pct", "disambig_50pct", "disambig_75pct"]
    n_grid = 200

    fig, axes = plt.subplots(2, 2, figsize=(20, 10), sharex=True)

    for col, (results, config, arm_label) in enumerate([
        (base_results, base_config, "Base (Qwen2.5-7B)"),
        (inst_results, inst_config, "Instruct (Qwen2.5-7B-Instruct)"),
    ]):
        ax_vel = axes[0, col]
        ax_pref = axes[1, col]

        # Plot disambig_50pct trajectories (solid) + no_disambig (dashed)
        for cond in ["no_disambig", "disambig_50pct"]:
            if cond not in results:
                continue
            trials = results[cond]["trials"]
            ls_style = "--" if cond == "no_disambig" else "-"
            alpha_base = 0.4 if cond == "no_disambig" else 1.0

            for layer in layers:
                ls = str(layer)
                color = LAYER_COLORS.get(layer, "gray")
                lw = 2.5 if layer == 27 else (1.8 if layer == 21 else 1)

                grid, mean_v, se_v = resample_trajectories(trials, ls, "velocities", n_grid)
                label = f"L{layer}" if cond != "no_disambig" else None
                ax_vel.plot(grid, mean_v, color=color, linewidth=lw,
                            linestyle=ls_style, alpha=alpha_base, label=label)
                if cond != "no_disambig" and layer in [21, 27]:
                    ax_vel.fill_between(grid, mean_v - se_v, mean_v + se_v, color=color, alpha=0.1)

                grid, mean_p, se_p = resample_trajectories(trials, ls, "h1_preference", n_grid)
                ax_pref.plot(grid, mean_p, color=color, linewidth=lw,
                             linestyle=ls_style, alpha=alpha_base, label=label)
                if cond != "no_disambig" and layer in [21, 27]:
                    ax_pref.fill_between(grid, mean_p - se_p, mean_p + se_p, color=color, alpha=0.1)

        # Vertical disambig lines
        for cond in disambig_conds:
            if cond not in results:
                continue
            frac = get_mean_disambig_frac(results[cond]["trials"])
            if frac is None:
                continue
            color = DISAMBIG_COLORS[cond]
            label = f"disambig {DISAMBIG_LABELS[cond]}"
            ax_vel.axvline(x=frac, color=color, linestyle="--", linewidth=2, alpha=0.8, label=label)
            ax_pref.axvline(x=frac, color=color, linestyle="--", linewidth=2, alpha=0.8, label=label)

        ax_vel.set_title(f"Velocity — {arm_label}", fontsize=13, fontweight="bold")
        ax_vel.set_ylabel("Velocity" if col == 0 else "")
        ax_vel.legend(fontsize=8, ncol=2, loc="upper right")
        ax_vel.grid(True, alpha=0.2)

        ax_pref.set_title(f"H1 Preference — {arm_label}", fontsize=13, fontweight="bold")
        ax_pref.set_ylabel("H1 Preference" if col == 0 else "")
        ax_pref.set_xlabel("Normalized token position", fontsize=11)
        ax_pref.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4)
        ax_pref.legend(fontsize=8, ncol=2, loc="upper right")
        ax_pref.grid(True, alpha=0.2)

    plt.suptitle("Natural Language Disambiguation: Base vs Instruct", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results/natural_disambig_pilot")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = results_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Individual arm plots
    for arm, label in [("instruct", "Instruct (Qwen2.5-7B-Instruct)"),
                        ("base", "Base (Qwen2.5-7B)")]:
        results, config = load_arm(results_dir, arm)
        plot_combined(results, config, label, output_dir / f"{arm}_combined.png")

    # Side-by-side comparison
    plot_base_vs_instruct_combined(results_dir, output_dir / "base_vs_instruct_combined.png")

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
