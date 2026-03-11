#!/usr/bin/env python3
"""
Plotting for Natural Language Disambiguation Experiment.

Generates:
1. Velocity trajectory aligned on disambiguation position
2. H1 preference trajectory by layer
3. Per-category breakdown
4. Velocity spike heatmap (layers × conditions)
5. Base vs instruct comparison (if both arms available)

Usage:
    python plot_natural_disambig.py --results-dir results/natural_disambig_pilot
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_results(results_dir: Path, arm: str) -> dict | None:
    """Load results for a given arm."""
    path = results_dir / f"{arm}_model" / "results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_config(results_dir: Path, arm: str) -> dict | None:
    """Load config for a given arm."""
    path = results_dir / f"{arm}_model" / "config.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def normalize_trajectories(trials: list[dict], layer: str, metric: str,
                           align_on_disambig: bool = True) -> tuple:
    """
    Normalize trajectories to [0, 1] position space and optionally
    align on disambiguation position.

    Returns:
        positions: array of normalized positions
        mean_traj: mean trajectory
        std_traj: std trajectory
    """
    all_trajs = []

    for trial in trials:
        if trial.get("error") is not None:
            continue
        traj = trial["layers"][layer][metric]
        n = len(traj)
        if n == 0:
            continue

        disambig_pos = trial.get("disambig_token_pos")
        total = trial.get("total_tokens", n)

        if align_on_disambig and disambig_pos is not None and disambig_pos > 0:
            # Normalize so disambig_pos = 0.0
            positions = [(i - disambig_pos) / total for i in range(n)]
        else:
            positions = [i / max(total, 1) for i in range(n)]

        all_trajs.append((positions, traj))

    if not all_trajs:
        return np.array([]), np.array([]), np.array([])

    # Resample all trajectories onto a common grid
    if align_on_disambig:
        grid = np.linspace(-0.6, 0.5, 100)
    else:
        grid = np.linspace(0, 1, 100)

    resampled = []
    for positions, traj in all_trajs:
        if len(positions) < 2:
            continue
        interp = np.interp(grid, positions, traj, left=np.nan, right=np.nan)
        resampled.append(interp)

    if not resampled:
        return grid, np.full_like(grid, np.nan), np.full_like(grid, np.nan)

    resampled = np.array(resampled)
    mean_traj = np.nanmean(resampled, axis=0)
    std_traj = np.nanstd(resampled, axis=0)
    return grid, mean_traj, std_traj


def plot_velocity_aligned(results: dict, config: dict, output_dir: Path):
    """Plot velocity trajectories aligned on disambiguation position."""
    layers = config["layers"]
    disambig_conditions = [c for c in results if c.startswith("disambig")]
    control_conditions = [c for c in results if not c.startswith("disambig") and c != "unambiguous_H1" and c != "unambiguous_H2"]

    colors = {"disambig_25pct": "#1f77b4", "disambig_50pct": "#ff7f0e",
              "disambig_75pct": "#2ca02c", "no_disambig": "#d62728"}

    fig, axes = plt.subplots(len(layers), 1, figsize=(12, 3 * len(layers)), squeeze=False)

    for ax_idx, layer in enumerate(layers):
        ax = axes[ax_idx, 0]
        layer_str = str(layer)

        for cond in disambig_conditions + control_conditions:
            if cond not in results:
                continue
            trials = results[cond]["trials"]
            align = cond.startswith("disambig")
            grid, mean_v, std_v = normalize_trajectories(
                trials, layer_str, "velocities", align_on_disambig=align
            )
            if len(grid) == 0:
                continue

            color = colors.get(cond, "gray")
            ax.plot(grid, mean_v, label=cond, color=color, linewidth=2)
            ax.fill_between(grid, mean_v - std_v, mean_v + std_v,
                           color=color, alpha=0.15)

        if any(c.startswith("disambig") for c in results):
            ax.axvline(x=0, color="red", linestyle="--", alpha=0.5, label="disambig point")

        ax.set_title(f"Layer {layer}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Velocity")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Normalized Position (0 = disambig point)")
    plt.suptitle("Velocity Aligned on Disambiguation Point", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "velocity_aligned_on_disambig.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: velocity_aligned_on_disambig.png")


def plot_preference_trajectory(results: dict, config: dict, output_dir: Path):
    """Plot H1 preference trajectory by layer."""
    layers = config["layers"]
    conditions = [c for c in results if c in [
        "disambig_25pct", "disambig_50pct", "disambig_75pct", "no_disambig"
    ]]

    colors = {"disambig_25pct": "#1f77b4", "disambig_50pct": "#ff7f0e",
              "disambig_75pct": "#2ca02c", "no_disambig": "#d62728"}

    fig, axes = plt.subplots(len(layers), 1, figsize=(12, 3 * len(layers)), squeeze=False)

    for ax_idx, layer in enumerate(layers):
        ax = axes[ax_idx, 0]
        layer_str = str(layer)

        for cond in conditions:
            if cond not in results:
                continue
            trials = results[cond]["trials"]
            align = cond.startswith("disambig")
            grid, mean_p, std_p = normalize_trajectories(
                trials, layer_str, "h1_preference", align_on_disambig=align
            )
            if len(grid) == 0:
                continue

            color = colors.get(cond, "gray")
            ax.plot(grid, mean_p, label=cond, color=color, linewidth=2)
            ax.fill_between(grid, mean_p - std_p, mean_p + std_p,
                           color=color, alpha=0.15)

        if any(c.startswith("disambig") for c in conditions):
            ax.axvline(x=0, color="red", linestyle="--", alpha=0.5)
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)

        ax.set_title(f"Layer {layer}", fontsize=12, fontweight="bold")
        ax.set_ylabel("H1 Preference (cosine)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Normalized Position (0 = disambig point)")
    plt.suptitle("H1 Preference Trajectory by Layer", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "preference_trajectory.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: preference_trajectory.png")


def plot_summary_bars(results: dict, config: dict, output_dir: Path):
    """Plot summary bar charts: preference change and disambig velocity by condition × layer."""
    layers = config["layers"]
    conditions = [c for c in ["disambig_25pct", "disambig_50pct", "disambig_75pct", "no_disambig"]
                  if c in results]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Left: preference change
    ax = axes[0]
    x = np.arange(len(conditions))
    width = 0.15
    for i, layer in enumerate(layers):
        vals = []
        for cond in conditions:
            agg = results[cond]["aggregate"]
            if agg.get("n_trials", 0) > 0:
                vals.append(agg["layers"][str(layer)]["pref_change_mean"])
            else:
                vals.append(0)
        ax.bar(x + i * width, vals, width, label=f"Layer {layer}")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xticks(x + width * (len(layers) - 1) / 2)
    ax.set_xticklabels(conditions, rotation=15)
    ax.set_ylabel("Preference Change (post - pre)")
    ax.set_title("H1 Preference Change After Disambiguation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Right: disambig velocity
    ax = axes[1]
    for i, layer in enumerate(layers):
        vals = []
        for cond in conditions:
            agg = results[cond]["aggregate"]
            if agg.get("n_trials", 0) > 0:
                vals.append(agg["layers"][str(layer)]["disambig_vel_mean"])
            else:
                vals.append(0)
        ax.bar(x + i * width, vals, width, label=f"Layer {layer}")
    ax.set_xticks(x + width * (len(layers) - 1) / 2)
    ax.set_xticklabels(conditions, rotation=15)
    ax.set_ylabel("Velocity at Disambig Token")
    ax.set_title("Disambiguation Velocity by Condition")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "summary_bars.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: summary_bars.png")


def plot_per_category(results: dict, config: dict, output_dir: Path):
    """Plot per-category-pair breakdown for the primary condition."""
    cond = "disambig_50pct"
    if cond not in results:
        return

    trials = results[cond]["trials"]
    successful = [t for t in trials if t.get("error") is None]

    # Group by category pair
    pairs = {}
    for t in successful:
        pair = t["category_pair"]
        if pair not in pairs:
            pairs[pair] = []
        pairs[pair].append(t)

    layers = config["layers"]
    pair_names = sorted(pairs.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Preference change by category
    ax = axes[0]
    x = np.arange(len(pair_names))
    width = 0.15
    for i, layer in enumerate(layers):
        vals = []
        for pname in pair_names:
            pref_changes = []
            for t in pairs[pname]:
                lr = t["layers"][str(layer)]
                pref_changes.append(lr["pref_change"])
            vals.append(np.mean(pref_changes) if pref_changes else 0)
        ax.bar(x + i * width, vals, width, label=f"Layer {layer}")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xticks(x + width * (len(layers) - 1) / 2)
    ax.set_xticklabels(pair_names, rotation=15)
    ax.set_ylabel("Preference Change")
    ax.set_title(f"Preference Change by Category ({cond})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Disambig velocity by category
    ax = axes[1]
    for i, layer in enumerate(layers):
        vals = []
        for pname in pair_names:
            vels = []
            for t in pairs[pname]:
                lr = t["layers"][str(layer)]
                vels.append(lr["disambig_velocity"])
            vals.append(np.mean(vels) if vels else 0)
        ax.bar(x + i * width, vals, width, label=f"Layer {layer}")
    ax.set_xticks(x + width * (len(layers) - 1) / 2)
    ax.set_xticklabels(pair_names, rotation=15)
    ax.set_ylabel("Velocity at Disambig Token")
    ax.set_title(f"Disambiguation Velocity by Category ({cond})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "per_category_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: per_category_breakdown.png")


def plot_spike_heatmap(results: dict, config: dict, output_dir: Path):
    """Plot velocity spike detection rate heatmap across layers and conditions."""
    layers = config["layers"]
    conditions = [c for c in ["disambig_25pct", "disambig_50pct", "disambig_75pct", "no_disambig"]
                  if c in results]

    data = np.zeros((len(layers), len(conditions)))
    for j, cond in enumerate(conditions):
        agg = results[cond]["aggregate"]
        if agg.get("n_trials", 0) == 0:
            continue
        for i, layer in enumerate(layers):
            data[i, j] = agg["layers"][str(layer)]["spike_detection_rate"]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=15)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"Layer {l}" for l in layers])
    ax.set_title("Velocity Spike Detection Rate")

    # Annotate cells
    for i in range(len(layers)):
        for j in range(len(conditions)):
            ax.text(j, i, f"{data[i, j]:.0%}", ha="center", va="center",
                    color="black" if data[i, j] < 0.5 else "white", fontsize=10)

    plt.colorbar(im, ax=ax, label="Spike Detection Rate")
    plt.tight_layout()
    plt.savefig(output_dir / "velocity_spike_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: velocity_spike_heatmap.png")


def plot_base_vs_instruct(results_dir: Path, output_dir: Path):
    """Compare base and instruct model results side by side."""
    base_results = load_results(results_dir, "base")
    instruct_results = load_results(results_dir, "instruct")

    if base_results is None or instruct_results is None:
        print("Skipping base vs instruct comparison (need both arms)")
        return

    base_config = load_config(results_dir, "base")
    instruct_config = load_config(results_dir, "instruct")

    cond = "disambig_50pct"
    if cond not in base_results or cond not in instruct_results:
        return

    layers = base_config["layers"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Preference change comparison
    ax = axes[0]
    x = np.arange(len(layers))
    width = 0.35

    base_vals = []
    instruct_vals = []
    for layer in layers:
        base_agg = base_results[cond]["aggregate"]
        inst_agg = instruct_results[cond]["aggregate"]
        base_vals.append(base_agg["layers"][str(layer)]["pref_change_mean"]
                        if base_agg.get("n_trials", 0) > 0 else 0)
        instruct_vals.append(inst_agg["layers"][str(layer)]["pref_change_mean"]
                            if inst_agg.get("n_trials", 0) > 0 else 0)

    ax.bar(x - width/2, base_vals, width, label="Base model", color="#1f77b4")
    ax.bar(x + width/2, instruct_vals, width, label="Instruct model", color="#ff7f0e")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.set_ylabel("Preference Change")
    ax.set_title(f"Preference Change: Base vs Instruct ({cond})")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Disambig velocity comparison
    ax = axes[1]
    base_vals = []
    instruct_vals = []
    for layer in layers:
        base_agg = base_results[cond]["aggregate"]
        inst_agg = instruct_results[cond]["aggregate"]
        base_vals.append(base_agg["layers"][str(layer)]["disambig_vel_mean"]
                        if base_agg.get("n_trials", 0) > 0 else 0)
        instruct_vals.append(inst_agg["layers"][str(layer)]["disambig_vel_mean"]
                            if inst_agg.get("n_trials", 0) > 0 else 0)

    ax.bar(x - width/2, base_vals, width, label="Base model", color="#1f77b4")
    ax.bar(x + width/2, instruct_vals, width, label="Instruct model", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.set_ylabel("Velocity at Disambig Token")
    ax.set_title(f"Disambiguation Velocity: Base vs Instruct ({cond})")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "base_vs_instruct.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: base_vs_instruct.png")


def main():
    parser = argparse.ArgumentParser(description="Plot Natural Language Disambiguation Results")
    parser.add_argument("--results-dir", type=str, default="results/natural_disambig_pilot",
                        help="Path to results directory")
    parser.add_argument("--arm", type=str, default="base", choices=["base", "instruct"],
                        help="Which arm to plot (default: base)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = results_dir / "plots" / args.arm
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(results_dir, args.arm)
    config = load_config(results_dir, args.arm)

    if results is None:
        print(f"No results found for {args.arm} arm at {results_dir}")
        return

    print(f"Loaded {args.arm} results from {results_dir}")
    print(f"Conditions: {list(results.keys())}")
    print(f"Layers: {config['layers']}")
    print(f"Output: {output_dir}")

    # Generate all plots
    plot_velocity_aligned(results, config, output_dir)
    plot_preference_trajectory(results, config, output_dir)
    plot_summary_bars(results, config, output_dir)
    plot_per_category(results, config, output_dir)
    plot_spike_heatmap(results, config, output_dir)
    # Base vs instruct goes to shared plots dir
    shared_output = results_dir / "plots"
    shared_output.mkdir(parents=True, exist_ok=True)
    plot_base_vs_instruct(results_dir, shared_output)

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
