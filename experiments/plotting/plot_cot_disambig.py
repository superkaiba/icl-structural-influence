#!/usr/bin/env python3
"""
Plotting for CoT Disambiguation Experiment.

Usage:
    python plot_cot_disambig.py --results-dir results/cot_disambig_pilot
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_results(results_dir: Path) -> tuple[dict, dict]:
    with open(results_dir / "results.json") as f:
        results = json.load(f)
    with open(results_dir / "config.json") as f:
        config = json.load(f)
    return results, config


def compute_trial_stats(trials: list[dict], layer: str) -> dict:
    """Compute per-layer stats from trial data (aggregates may be zeroed)."""
    pref_changes = []
    commit_vels = []
    spike_count = 0
    valid = 0

    for t in trials:
        if t.get("error") is not None:
            continue
        ld = t.get("layers", {}).get(layer)
        if ld is None:
            continue
        valid += 1

        if ld.get("spike_position") is not None:
            spike_count += 1
        if ld.get("pref_change") is not None:
            pref_changes.append(ld["pref_change"])
        if ld.get("commit_velocity") is not None:
            commit_vels.append(ld["commit_velocity"])

    return {
        "pref_change_mean": np.mean(pref_changes) if pref_changes else 0,
        "pref_change_std": np.std(pref_changes) if pref_changes else 0,
        "commit_vel_mean": np.mean(commit_vels) if commit_vels else 0,
        "commit_vel_std": np.std(commit_vels) if commit_vels else 0,
        "spike_detection_rate": spike_count / valid if valid > 0 else 0,
        "n_valid": valid,
    }


def plot_condition_comparison(results: dict, config: dict, output_dir: Path):
    """Bar chart comparing preference change and commit velocity across conditions."""
    layers = config["layers"]
    conditions = [c for c in results if len(results[c].get("trials", [])) > 0]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Preference change
    ax = axes[0]
    x = np.arange(len(conditions))
    width = 0.15
    for i, layer in enumerate(layers):
        vals = []
        errs = []
        for cond in conditions:
            stats = compute_trial_stats(results[cond]["trials"], str(layer))
            vals.append(stats["pref_change_mean"])
            errs.append(stats["pref_change_std"])
        ax.bar(x + i * width, vals, width, label=f"Layer {layer}", yerr=errs, capsize=2)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xticks(x + width * (len(layers) - 1) / 2)
    ax.set_xticklabels(conditions, rotation=20, ha="right")
    ax.set_ylabel("Preference Change (post - pre commit)")
    ax.set_title("H1 Preference Change by Condition")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3, axis="y")

    # Commit velocity
    ax = axes[1]
    for i, layer in enumerate(layers):
        vals = []
        for cond in conditions:
            stats = compute_trial_stats(results[cond]["trials"], str(layer))
            vals.append(stats["commit_vel_mean"])
        ax.bar(x + i * width, vals, width, label=f"Layer {layer}")
    ax.set_xticks(x + width * (len(layers) - 1) / 2)
    ax.set_xticklabels(conditions, rotation=20, ha="right")
    ax.set_ylabel("Velocity at Commitment Point")
    ax.set_title("Commitment Velocity by Condition")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "condition_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: condition_comparison.png")


def plot_velocity_aligned_on_commitment(results: dict, config: dict, output_dir: Path):
    """Plot velocity trajectories aligned on commitment point for CoT conditions."""
    layers = config["layers"]
    cot_conditions = [c for c in ["cot_ambig", "cot_disambig", "external_cot"] if c in results]
    colors = {"cot_ambig": "#1f77b4", "cot_disambig": "#ff7f0e", "external_cot": "#2ca02c"}

    fig, axes = plt.subplots(len(layers), 1, figsize=(12, 3 * len(layers)), squeeze=False)

    for ax_idx, layer in enumerate(layers):
        ax = axes[ax_idx, 0]
        ls = str(layer)

        for cond in cot_conditions:
            trials = results[cond]["trials"]
            all_vels_aligned = []

            for trial in trials:
                if trial.get("error") is not None:
                    continue
                if ls not in trial.get("layers", {}):
                    continue

                vels = trial["layers"][ls]["velocities"]
                commit_pos = trial["commitment"].get("lexical_position") or trial["commitment"].get("answer_position")

                if commit_pos is None or len(vels) == 0:
                    continue

                # Align: commitment point = 0, relative to start of generation
                prompt_len = trial.get("prompt_len", 0)
                commit_in_gen = commit_pos - prompt_len
                if commit_in_gen < 0 or commit_in_gen >= len(vels):
                    continue

                aligned = [(i - commit_in_gen, v) for i, v in enumerate(vels)]
                all_vels_aligned.append(aligned)

            if not all_vels_aligned:
                continue

            # Resample onto common grid
            grid = np.arange(-100, 50)
            resampled = []
            for aligned in all_vels_aligned:
                pos_to_vel = {p: v for p, v in aligned}
                row = [pos_to_vel.get(g, np.nan) for g in grid]
                resampled.append(row)
            resampled = np.array(resampled)
            mean_v = np.nanmean(resampled, axis=0)
            std_v = np.nanstd(resampled, axis=0)

            color = colors.get(cond, "gray")
            ax.plot(grid, mean_v, label=cond, color=color, linewidth=2)
            ax.fill_between(grid, mean_v - std_v, mean_v + std_v, color=color, alpha=0.15)

        ax.axvline(x=0, color="red", linestyle="--", alpha=0.5, label="commitment")
        ax.set_title(f"Layer {layer}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Velocity")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Token offset from commitment point")
    plt.suptitle("Velocity Aligned on Commitment Point (CoT conditions)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "velocity_aligned_on_commitment.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: velocity_aligned_on_commitment.png")


def plot_preference_trajectory(results: dict, config: dict, output_dir: Path):
    """Plot preference trajectories aligned on commitment for CoT conditions."""
    layers = config["layers"]
    cot_conditions = [c for c in ["cot_ambig", "cot_disambig", "external_cot"] if c in results]
    colors = {"cot_ambig": "#1f77b4", "cot_disambig": "#ff7f0e", "external_cot": "#2ca02c"}

    fig, axes = plt.subplots(len(layers), 1, figsize=(12, 3 * len(layers)), squeeze=False)

    for ax_idx, layer in enumerate(layers):
        ax = axes[ax_idx, 0]
        ls = str(layer)

        for cond in cot_conditions:
            trials = results[cond]["trials"]
            all_prefs_aligned = []

            for trial in trials:
                if trial.get("error") is not None or ls not in trial.get("layers", {}):
                    continue
                prefs = trial["layers"][ls]["h1_preference"]
                commit_pos = trial["commitment"].get("lexical_position") or trial["commitment"].get("answer_position")
                if commit_pos is None or len(prefs) == 0:
                    continue

                prompt_len = trial.get("prompt_len", 0)
                commit_in_gen = commit_pos - prompt_len
                if commit_in_gen < 0 or commit_in_gen >= len(prefs):
                    continue

                aligned = [(i - commit_in_gen, p) for i, p in enumerate(prefs)]
                all_prefs_aligned.append(aligned)

            if not all_prefs_aligned:
                continue

            grid = np.arange(-100, 50)
            resampled = []
            for aligned in all_prefs_aligned:
                pos_to_pref = {p: v for p, v in aligned}
                row = [pos_to_pref.get(g, np.nan) for g in grid]
                resampled.append(row)
            resampled = np.array(resampled)
            mean_p = np.nanmean(resampled, axis=0)

            color = colors.get(cond, "gray")
            ax.plot(grid, mean_p, label=cond, color=color, linewidth=2)

        ax.axvline(x=0, color="red", linestyle="--", alpha=0.5)
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
        ax.set_title(f"Layer {layer}", fontsize=12, fontweight="bold")
        ax.set_ylabel("H1 Preference")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Token offset from commitment point")
    plt.suptitle("H1 Preference Aligned on Commitment Point", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "preference_aligned_on_commitment.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: preference_aligned_on_commitment.png")


def plot_commitment_timing(results: dict, config: dict, output_dir: Path):
    """Analyze timing: does the velocity spike precede textual commitment?"""
    cot_conditions = [c for c in ["cot_ambig", "cot_disambig"] if c in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: histogram of spike-commit offsets
    ax = axes[0]
    for cond in cot_conditions:
        offsets = [t["spike_commit_offset"] for t in results[cond]["trials"]
                   if t.get("spike_commit_offset") is not None]
        if offsets:
            ax.hist(offsets, bins=20, alpha=0.5, label=cond)
    ax.axvline(x=0, color="red", linestyle="--", label="simultaneous")
    ax.set_xlabel("Spike - Commit offset (tokens)")
    ax.set_ylabel("Count")
    ax.set_title("Velocity Spike vs Textual Commitment Timing")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: per-category spike timing
    ax = axes[1]
    if "cot_ambig" in results:
        trials = results["cot_ambig"]["trials"]
        categories = {}
        for t in trials:
            if t.get("error") is not None:
                continue
            cat = t.get("category", "?")
            if cat not in categories:
                categories[cat] = []
            offset = t.get("spike_commit_offset")
            if offset is not None:
                categories[cat].append(offset)

        labels = []
        offsets_list = []
        for cat, offs in sorted(categories.items()):
            if offs:
                labels.append(f"Cat {cat}")
                offsets_list.append(offs)

        if offsets_list:
            ax.boxplot(offsets_list, labels=labels)
            ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
            ax.set_ylabel("Spike - Commit offset")
            ax.set_title("Timing by Category (cot_ambig)")
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "commitment_timing.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: commitment_timing.png")


def plot_cot_vs_external(results: dict, config: dict, output_dir: Path):
    """Compare CoT (self-generated) vs external disambiguation."""
    layers = config["layers"]
    conditions_to_compare = ["cot_ambig", "cot_disambig", "external_cot", "external_disambig"]
    available = [c for c in conditions_to_compare if c in results and len(results[c].get("trials", [])) > 0]

    if len(available) < 2:
        print("Skipping CoT vs external (need at least 2 conditions)")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(layers))
    width = 0.2
    bar_colors = {"cot_ambig": "#1f77b4", "cot_disambig": "#ff7f0e",
                  "external_cot": "#2ca02c", "external_disambig": "#d62728"}

    for i, cond in enumerate(available):
        vals = []
        for l in layers:
            stats = compute_trial_stats(results[cond]["trials"], str(l))
            vals.append(stats["commit_vel_mean"])
        ax.bar(x + i * width, vals, width, label=cond, color=bar_colors.get(cond, "gray"))

    ax.set_xticks(x + width * (len(available) - 1) / 2)
    ax.set_xticklabels([f"Layer {l}" for l in layers])
    ax.set_ylabel("Velocity at Commitment Point")
    ax.set_title("Commitment Velocity: Self-Generated vs External")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "cot_vs_external.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: cot_vs_external.png")


def plot_spike_heatmap(results: dict, config: dict, output_dir: Path):
    """Spike detection rate heatmap across layers and conditions."""
    layers = config["layers"]
    conditions = [c for c in results if len(results[c].get("trials", [])) > 0]

    data = np.zeros((len(layers), len(conditions)))
    for j, cond in enumerate(conditions):
        for i, layer in enumerate(layers):
            stats = compute_trial_stats(results[cond]["trials"], str(layer))
            data[i, j] = stats["spike_detection_rate"]

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=20, ha="right")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"Layer {l}" for l in layers])
    ax.set_title("Velocity Spike Detection Rate")

    for i in range(len(layers)):
        for j in range(len(conditions)):
            ax.text(j, i, f"{data[i, j]:.0%}", ha="center", va="center",
                    color="black" if data[i, j] < 0.5 else "white", fontsize=10)

    plt.colorbar(im, ax=ax, label="Spike Detection Rate")
    plt.tight_layout()
    plt.savefig(output_dir / "spike_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: spike_heatmap.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results/cot_disambig_pilot")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = results_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    results, config = load_results(results_dir)
    print(f"Loaded results from {results_dir}")
    print(f"Conditions: {list(results.keys())}")

    plot_condition_comparison(results, config, output_dir)
    plot_velocity_aligned_on_commitment(results, config, output_dir)
    plot_preference_trajectory(results, config, output_dir)
    plot_commitment_timing(results, config, output_dir)
    plot_cot_vs_external(results, config, output_dir)
    plot_spike_heatmap(results, config, output_dir)

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
