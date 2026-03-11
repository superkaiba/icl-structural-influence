#!/usr/bin/env python3
"""
Combined disambiguation plot: grid of conditions, each with its own
velocity and preference panel, vertical line at disambig point.

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


def load_arm(results_dir: Path, arm: str):
    with open(results_dir / f"{arm}_model" / "results.json") as f:
        results = json.load(f)
    with open(results_dir / f"{arm}_model" / "config.json") as f:
        config = json.load(f)
    return results, config


def average_trajectories_raw(trials, layer_str, metric):
    """Average trajectories in raw token space."""
    trajs = []
    for t in trials:
        if t.get("error") is not None:
            continue
        traj = t["layers"][layer_str][metric]
        if len(traj) < 3:
            continue
        trajs.append(np.array(traj, dtype=float))
    if not trajs:
        return np.array([]), np.array([]), np.array([])
    max_len = max(len(t) for t in trajs)
    padded = np.full((len(trajs), max_len), np.nan)
    for i, t in enumerate(trajs):
        padded[i, :len(t)] = t
    positions = np.arange(max_len)
    mean = np.nanmean(padded, axis=0)
    se = np.nanstd(padded, axis=0) / np.sqrt(np.sum(~np.isnan(padded), axis=0).clip(1))
    return positions, mean, se


def get_mean_disambig_pos(trials):
    positions = [t["disambig_token_pos"] for t in trials
                 if t.get("disambig_token_pos") is not None and t.get("error") is None]
    return np.mean(positions) if positions else None


def plot_grid(results, config, arm_label, output_path):
    """Grid: columns = conditions, rows = [velocity, preference]."""
    layers = config["layers"]
    conds = ["no_disambig", "disambig_25pct", "disambig_50pct", "disambig_75pct"]
    conds = [c for c in conds if c in results]
    ncols = len(conds)

    fig, axes = plt.subplots(2, ncols, figsize=(5 * ncols, 8), squeeze=False)

    # Shared x-axis limit across all panels
    max_tok = 0
    for cond in conds:
        for t in results[cond]["trials"]:
            if t.get("error") is None:
                max_tok = max(max_tok, t.get("total_tokens", 0))

    for col, cond in enumerate(conds):
        trials = results[cond]["trials"]
        ax_vel = axes[0, col]
        ax_pref = axes[1, col]

        for layer in layers:
            ls = str(layer)
            color = LAYER_COLORS.get(layer, "gray")
            lw = 2.5 if layer == 27 else (1.5 if layer == 21 else 0.8)

            pos, mean_v, se_v = average_trajectories_raw(trials, ls, "velocities")
            if len(pos) == 0:
                continue
            label = f"L{layer}" if col == 0 else None
            ax_vel.plot(pos, mean_v, color=color, linewidth=lw, label=label)
            if layer in [21, 27]:
                ax_vel.fill_between(pos, np.maximum(mean_v - se_v, 1e-1), mean_v + se_v,
                                    color=color, alpha=0.08)

            pos, mean_p, se_p = average_trajectories_raw(trials, ls, "h1_preference")
            if len(pos) == 0:
                continue
            ax_pref.plot(pos, mean_p, color=color, linewidth=lw, label=label)
            if layer in [21, 27]:
                ax_pref.fill_between(pos, mean_p - se_p, mean_p + se_p, color=color, alpha=0.08)

        # Vertical disambig line
        mean_pos = get_mean_disambig_pos(trials)
        if mean_pos is not None:
            ax_vel.axvline(x=mean_pos, color="red", linestyle="--", linewidth=2, alpha=0.8)
            ax_pref.axvline(x=mean_pos, color="red", linestyle="--", linewidth=2, alpha=0.8)

        # Formatting
        cond_label = cond.replace("disambig_", "").replace("pct", "%").replace("no_disambig", "no disambig")
        ax_vel.set_title(cond_label, fontsize=13, fontweight="bold")
        ax_vel.set_yscale("log")
        ax_vel.set_xlim(0, max_tok)
        ax_vel.grid(True, alpha=0.2)
        if col == 0:
            ax_vel.set_ylabel("Velocity (log)", fontsize=11)
            ax_vel.legend(fontsize=8)

        ax_pref.set_xlim(0, max_tok)
        ax_pref.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4)
        ax_pref.grid(True, alpha=0.2)
        ax_pref.set_xlabel("Token position", fontsize=10)
        if col == 0:
            ax_pref.set_ylabel("H1 Preference", fontsize=11)

    plt.suptitle(f"Natural Disambiguation — {arm_label}", fontsize=15, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_base_vs_instruct_grid(results_dir, output_path):
    """2-row grid: base on top, instruct on bottom. Columns = conditions."""
    base_results, base_config = load_arm(results_dir, "base")
    inst_results, inst_config = load_arm(results_dir, "instruct")

    layers = base_config["layers"]
    conds = ["no_disambig", "disambig_25pct", "disambig_50pct", "disambig_75pct"]
    conds = [c for c in conds if c in base_results and c in inst_results]
    ncols = len(conds)

    fig, axes = plt.subplots(2, ncols, figsize=(5 * ncols, 8), squeeze=False)

    max_tok = 0
    for r in [base_results, inst_results]:
        for cond in conds:
            for t in r[cond]["trials"]:
                if t.get("error") is None:
                    max_tok = max(max_tok, t.get("total_tokens", 0))

    for row, (results, config, arm_label) in enumerate([
        (base_results, base_config, "Base (Qwen2.5-7B)"),
        (inst_results, inst_config, "Instruct (Qwen2.5-7B-Instruct)"),
    ]):
        for col, cond in enumerate(conds):
            ax = axes[row, col]
            trials = results[cond]["trials"]

            for layer in layers:
                ls = str(layer)
                color = LAYER_COLORS.get(layer, "gray")
                lw = 2.5 if layer == 27 else (1.5 if layer == 21 else 0.8)

                pos, mean_p, se_p = average_trajectories_raw(trials, ls, "h1_preference")
                if len(pos) == 0:
                    continue
                label = f"L{layer}" if (row == 0 and col == 0) else None
                ax.plot(pos, mean_p, color=color, linewidth=lw, label=label)
                if layer in [21, 27]:
                    ax.fill_between(pos, mean_p - se_p, mean_p + se_p, color=color, alpha=0.08)

            mean_pos = get_mean_disambig_pos(trials)
            if mean_pos is not None:
                ax.axvline(x=mean_pos, color="red", linestyle="--", linewidth=2, alpha=0.8)

            ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4)
            ax.set_xlim(0, max_tok)
            ax.grid(True, alpha=0.2)

            if row == 0:
                cond_label = cond.replace("disambig_", "").replace("pct", "%").replace("no_disambig", "no disambig")
                ax.set_title(cond_label, fontsize=13, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{arm_label}\nH1 Preference", fontsize=10)
                if row == 0:
                    ax.legend(fontsize=8)
            if row == 1:
                ax.set_xlabel("Token position", fontsize=10)

    plt.suptitle("H1 Preference: Base vs Instruct", fontsize=15, fontweight="bold")
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

    for arm, label in [("instruct", "Instruct (Qwen2.5-7B-Instruct)"),
                        ("base", "Base (Qwen2.5-7B)")]:
        results, config = load_arm(results_dir, arm)
        plot_grid(results, config, label, output_dir / f"{arm}_grid.png")

    plot_base_vs_instruct_grid(results_dir, output_dir / "base_vs_instruct_grid.png")

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
