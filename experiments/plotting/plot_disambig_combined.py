#!/usr/bin/env python3
"""
Combined disambiguation plot: velocity and preference trajectories with
vertical lines marking each disambiguation point. Raw token positions,
all conditions overlaid (shorter ones end early).

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
COND_LINESTYLES = {"disambig_25pct": "-", "disambig_50pct": "-", "disambig_75pct": "-", "no_disambig": "--"}


def load_arm(results_dir: Path, arm: str):
    with open(results_dir / f"{arm}_model" / "results.json") as f:
        results = json.load(f)
    with open(results_dir / f"{arm}_model" / "config.json") as f:
        config = json.load(f)
    return results, config


def average_trajectories_raw(trials, layer_str, metric):
    """Average trajectories in raw token space. Returns (positions, mean, se)."""
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

    # Pad shorter sequences with NaN to the max length
    max_len = max(len(t) for t in trajs)
    padded = np.full((len(trajs), max_len), np.nan)
    for i, t in enumerate(trajs):
        padded[i, :len(t)] = t

    positions = np.arange(max_len)
    mean = np.nanmean(padded, axis=0)
    se = np.nanstd(padded, axis=0) / np.sqrt(np.sum(~np.isnan(padded), axis=0).clip(1))
    return positions, mean, se


def get_mean_disambig_pos(trials):
    """Get mean raw token position of the disambig token."""
    positions = [t["disambig_token_pos"] for t in trials
                 if t.get("disambig_token_pos") is not None and t.get("error") is None]
    return np.mean(positions) if positions else None


def plot_combined(results, config, arm_label, output_path):
    layers = config["layers"]
    disambig_conds = ["disambig_25pct", "disambig_50pct", "disambig_75pct"]
    all_conds = ["no_disambig"] + disambig_conds

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    ax_vel, ax_pref = axes

    for cond in all_conds:
        if cond not in results:
            continue
        trials = results[cond]["trials"]
        linestyle = COND_LINESTYLES.get(cond, "-")
        is_baseline = cond == "no_disambig"
        alpha_base = 0.35 if is_baseline else 0.85

        for layer in layers:
            ls = str(layer)
            color = LAYER_COLORS.get(layer, "gray")
            lw = 2.5 if layer == 27 else (1.5 if layer == 21 else 0.8)

            # Velocity
            pos, mean_v, se_v = average_trajectories_raw(trials, ls, "velocities")
            if len(pos) == 0:
                continue
            # Only label once per layer (from the first disambig condition)
            vel_label = None
            if cond == "disambig_25pct":
                vel_label = f"L{layer}"
            elif is_baseline and layer == 27:
                vel_label = "no disambig (L27)"
            ax_vel.plot(pos, mean_v, color=color, linewidth=lw,
                        linestyle=linestyle, alpha=alpha_base, label=vel_label)
            if not is_baseline and layer in [21, 27]:
                ax_vel.fill_between(pos, mean_v - se_v, mean_v + se_v, color=color, alpha=0.06)

            # Preference
            pos, mean_p, se_p = average_trajectories_raw(trials, ls, "h1_preference")
            if len(pos) == 0:
                continue
            pref_label = None
            if cond == "disambig_25pct":
                pref_label = f"L{layer}"
            elif is_baseline and layer == 27:
                pref_label = "no disambig (L27)"
            ax_pref.plot(pos, mean_p, color=color, linewidth=lw,
                         linestyle=linestyle, alpha=alpha_base, label=pref_label)
            if not is_baseline and layer in [21, 27]:
                ax_pref.fill_between(pos, mean_p - se_p, mean_p + se_p, color=color, alpha=0.06)

    # Vertical lines at mean disambig positions
    for cond in disambig_conds:
        if cond not in results:
            continue
        mean_pos = get_mean_disambig_pos(results[cond]["trials"])
        if mean_pos is None:
            continue
        color = DISAMBIG_COLORS[cond]
        label = f"disambig @ {DISAMBIG_LABELS[cond]}"
        ax_vel.axvline(x=mean_pos, color=color, linestyle="--", linewidth=2, alpha=0.8, label=label)
        ax_pref.axvline(x=mean_pos, color=color, linestyle="--", linewidth=2, alpha=0.8, label=label)

    ax_vel.set_yscale("log")
    ax_vel.set_ylabel("Velocity (L2 norm of residual stream change)", fontsize=12)
    ax_vel.set_title(f"Velocity Trajectory — {arm_label}", fontsize=14, fontweight="bold")
    ax_vel.legend(fontsize=9, ncol=2, loc="upper right")
    ax_vel.grid(True, alpha=0.2)

    ax_pref.set_ylabel("H1 Preference (cosine similarity)", fontsize=12)
    ax_pref.set_xlabel("Token position", fontsize=12)
    ax_pref.set_title(f"H1 Preference Trajectory — {arm_label}", fontsize=14, fontweight="bold")
    ax_pref.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4)
    ax_pref.legend(fontsize=9, ncol=2, loc="upper right")
    ax_pref.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_base_vs_instruct_combined(results_dir, output_path):
    """Side-by-side base vs instruct, raw token positions, all conditions overlaid."""
    base_results, base_config = load_arm(results_dir, "base")
    inst_results, inst_config = load_arm(results_dir, "instruct")

    layers = base_config["layers"]
    disambig_conds = ["disambig_25pct", "disambig_50pct", "disambig_75pct"]
    all_conds = ["no_disambig"] + disambig_conds

    fig, axes = plt.subplots(2, 2, figsize=(20, 10), sharex=True)

    for col, (results, config, arm_label) in enumerate([
        (base_results, base_config, "Base (Qwen2.5-7B)"),
        (inst_results, inst_config, "Instruct (Qwen2.5-7B-Instruct)"),
    ]):
        ax_vel = axes[0, col]
        ax_pref = axes[1, col]

        for cond in all_conds:
            if cond not in results:
                continue
            trials = results[cond]["trials"]
            linestyle = COND_LINESTYLES.get(cond, "-")
            is_baseline = cond == "no_disambig"
            alpha_base = 0.35 if is_baseline else 0.85

            for layer in layers:
                ls = str(layer)
                color = LAYER_COLORS.get(layer, "gray")
                lw = 2.5 if layer == 27 else (1.5 if layer == 21 else 0.8)

                pos, mean_v, se_v = average_trajectories_raw(trials, ls, "velocities")
                if len(pos) == 0:
                    continue
                label = f"L{layer}" if cond == "disambig_25pct" else None
                ax_vel.plot(pos, mean_v, color=color, linewidth=lw,
                            linestyle=linestyle, alpha=alpha_base, label=label)
                if not is_baseline and layer in [21, 27]:
                    ax_vel.fill_between(pos, mean_v - se_v, mean_v + se_v, color=color, alpha=0.06)

                pos, mean_p, se_p = average_trajectories_raw(trials, ls, "h1_preference")
                if len(pos) == 0:
                    continue
                ax_pref.plot(pos, mean_p, color=color, linewidth=lw,
                             linestyle=linestyle, alpha=alpha_base, label=label)
                if not is_baseline and layer in [21, 27]:
                    ax_pref.fill_between(pos, mean_p - se_p, mean_p + se_p, color=color, alpha=0.06)

        # Vertical disambig lines
        for cond in disambig_conds:
            if cond not in results:
                continue
            mean_pos = get_mean_disambig_pos(results[cond]["trials"])
            if mean_pos is None:
                continue
            color = DISAMBIG_COLORS[cond]
            label = f"disambig {DISAMBIG_LABELS[cond]}"
            ax_vel.axvline(x=mean_pos, color=color, linestyle="--", linewidth=2, alpha=0.8, label=label)
            ax_pref.axvline(x=mean_pos, color=color, linestyle="--", linewidth=2, alpha=0.8, label=label)

        ax_vel.set_yscale("log")
        ax_vel.set_title(f"Velocity — {arm_label}", fontsize=13, fontweight="bold")
        ax_vel.set_ylabel("Velocity" if col == 0 else "")
        ax_vel.legend(fontsize=8, ncol=2, loc="upper right")
        ax_vel.grid(True, alpha=0.2)

        ax_pref.set_title(f"H1 Preference — {arm_label}", fontsize=13, fontweight="bold")
        ax_pref.set_ylabel("H1 Preference" if col == 0 else "")
        ax_pref.set_xlabel("Token position", fontsize=11)
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

    for arm, label in [("instruct", "Instruct (Qwen2.5-7B-Instruct)"),
                        ("base", "Base (Qwen2.5-7B)")]:
        results, config = load_arm(results_dir, arm)
        plot_combined(results, config, label, output_dir / f"{arm}_combined.png")

    plot_base_vs_instruct_combined(results_dir, output_dir / "base_vs_instruct_combined.png")

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
