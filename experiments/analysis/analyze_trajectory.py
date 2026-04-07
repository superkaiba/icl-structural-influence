"""T-034: Analyze representation geometry trajectories over context length.

Extracts collapse_trajectory data from experiment results and plots how
representation metrics (cos_sim, effective_dim, spread) evolve as context grows.

Key questions:
- At what context length do representations start collapsing?
- Does vocab size affect the collapse onset point?
- How does structure (p_intra) affect the trajectory shape?
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path("results/safety_collapse_sweep_v2/context_type")
PLOTS_DIR = Path("results/safety_collapse_sweep_v2/plots")


def load_trajectories(experiment_dir: Path) -> list[dict]:
    """Load all trajectory data from an experiment's all_results.json."""
    f = experiment_dir / "all_results.json"
    if not f.exists():
        return []
    with open(f) as fh:
        data = json.load(fh)
    # Group trajectories by context_length and trial
    trajectories = []
    for r in data:
        traj = r.get("collapse_trajectory", [])
        if traj:
            trajectories.append({
                "context_length": r["context_length"],
                "condition": r.get("condition", ""),
                "trial": r.get("trial_label", ""),
                "trajectory": traj,
            })
    return trajectories


def extract_metric_curves(trajectories: list[dict], metric: str = "avg_cos_sim",
                          layer: str = None) -> dict:
    """Extract metric curves grouped by context_length.

    Returns: {context_length: {"positions": [...], "values": [...]}} averaged across trials.
    """
    by_length = defaultdict(lambda: defaultdict(list))

    for t in trajectories:
        ctx_len = t["context_length"]
        for point in t["trajectory"]:
            pos = point["position"]
            layer_metrics = point.get("layer_metrics", {})
            if not layer_metrics:
                continue
            # Use specified layer or last available
            if layer is not None:
                lm = layer_metrics.get(str(layer), {})
            else:
                # Use the last (deepest) layer
                keys = sorted(layer_metrics.keys(), key=lambda x: int(x))
                lm = layer_metrics.get(keys[-1], {}) if keys else {}
            val = lm.get(metric)
            if val is not None:
                by_length[ctx_len][pos].append(val)

    # Average across trials
    result = {}
    for ctx_len in sorted(by_length):
        positions = sorted(by_length[ctx_len].keys())
        values = [np.mean(by_length[ctx_len][p]) for p in positions]
        stds = [np.std(by_length[ctx_len][p]) for p in positions]
        result[ctx_len] = {
            "positions": positions,
            "values": values,
            "stds": stds,
        }
    return result


def plot_trajectory_by_vocab(metric: str = "avg_cos_sim", ylabel: str = "Avg Cosine Similarity"):
    """Plot trajectory curves for different vocab sizes."""
    vocab_experiments = [
        (1, "repeated_token", "Repeated (1)"),
        (2, "random_tokens_2", "Random (2)"),
        (5, "random_tokens_5", "Random (5)"),
        (15, "random_tokens_15", "Random (15)"),
        (50, "random_tokens_50", "Random (50)"),
        (200, "random_tokens_200", "Random (200)"),
        (1000, "random_tokens_1000", "Random (1000)"),
    ]

    # Target long context lengths for trajectory visibility
    target_lengths = [10000, 50000, 150000, 262144]
    fig, axes = plt.subplots(1, len(target_lengths), figsize=(5 * len(target_lengths), 5),
                              squeeze=False)

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(vocab_experiments)))

    for li, tgt_len in enumerate(target_lengths):
        ax = axes[0, li]
        for ci, (vocab, ctx_type, label) in enumerate(vocab_experiments):
            trajs = load_trajectories(BASE_DIR / ctx_type)
            if not trajs:
                continue
            curves = extract_metric_curves(trajs, metric=metric)
            if tgt_len in curves:
                c = curves[tgt_len]
                ax.plot(c["positions"], c["values"], color=colors[ci],
                        label=label, linewidth=1.5, alpha=0.8)
                # Light shading for ±1 std
                if any(s > 0 for s in c["stds"]):
                    lo = [v - s for v, s in zip(c["values"], c["stds"])]
                    hi = [v + s for v, s in zip(c["values"], c["stds"])]
                    ax.fill_between(c["positions"], lo, hi, color=colors[ci], alpha=0.1)

        ax.set_xlabel("Tokens Processed", fontsize=10)
        if li == 0:
            ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"Context = {tgt_len // 1000}K tokens", fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3)
        if li == len(target_lengths) - 1:
            ax.legend(fontsize=8, loc="best")

    plt.suptitle(f"Representation Trajectory: {ylabel} by Vocab Size (last layer, Qwen3.5-9B)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_trajectory_by_structure(metric: str = "avg_cos_sim", ylabel: str = "Avg Cosine Similarity"):
    """Plot trajectory curves for different structure amounts (p_intra)."""
    p_experiments = [
        (0.0, "structured_walk_15_p0", "p=0.00"),
        (0.15, "structured_walk_15_p15", "p=0.15"),
        (0.30, "structured_walk_15_p30", "p=0.30"),
        (0.50, "structured_walk_15_p50", "p=0.50"),
        (0.65, "structured_walk_15_p65", "p=0.65"),
        (0.80, "structured_walk_15", "p=0.80 (default)"),
        (0.95, "structured_walk_15_p95", "p=0.95"),
    ]

    target_lengths = [10000, 50000, 150000, 262144]
    fig, axes = plt.subplots(1, len(target_lengths), figsize=(5 * len(target_lengths), 5),
                              squeeze=False)

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(p_experiments)))

    for li, tgt_len in enumerate(target_lengths):
        ax = axes[0, li]
        for ci, (p_val, ctx_type, label) in enumerate(p_experiments):
            trajs = load_trajectories(BASE_DIR / ctx_type)
            if not trajs:
                continue
            curves = extract_metric_curves(trajs, metric=metric)
            if tgt_len in curves:
                c = curves[tgt_len]
                ax.plot(c["positions"], c["values"], color=colors[ci],
                        label=label, linewidth=1.5, alpha=0.8)

        ax.set_xlabel("Tokens Processed", fontsize=10)
        if li == 0:
            ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"Context = {tgt_len // 1000}K tokens", fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3)
        if li == len(target_lengths) - 1:
            ax.legend(fontsize=8, loc="best")

    plt.suptitle(f"Representation Trajectory: {ylabel} by Structure Strength (last layer, Qwen3.5-9B)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_layer_comparison():
    """Compare trajectory across layers for a single experiment."""
    # Use random_tokens_12 as reference (has trajectory data from T-030)
    trajs = load_trajectories(BASE_DIR / "random_tokens_12")
    if not trajs:
        print("  SKIP: layer comparison (no trajectory data)")
        return None

    # Find available layers from the first trajectory
    sample_point = trajs[0]["trajectory"][0]
    layers = sorted(sample_point["layer_metrics"].keys(), key=int)
    if len(layers) < 3:
        print(f"  SKIP: only {len(layers)} layers available")
        return None

    # Pick representative layers: early, middle, late
    layer_subset = [layers[0], layers[len(layers)//2], layers[-1]]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ["#2ecc71", "#3498db", "#e74c3c"]

    for ai, metric_info in enumerate([
        ("avg_cos_sim", "Avg Cosine Similarity"),
        ("effective_dim", "Effective Dimension"),
        ("spread", "Spread (Variance)"),
    ]):
        metric, ylabel = metric_info
        ax = axes[ai]
        for li, layer in enumerate(layer_subset):
            curves = extract_metric_curves(trajs, metric=metric, layer=layer)
            # Use 262K context
            tgt = 262144
            if tgt in curves:
                c = curves[tgt]
                ax.plot(c["positions"], c["values"], color=colors[li],
                        label=f"Layer {layer}", linewidth=1.5)

        ax.set_xlabel("Tokens Processed", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle("Collapse Trajectory Across Layers (random_tokens_12, 262K context, Qwen3.5-9B)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print("T-034: Analyzing representation trajectories")
    print("-" * 50)

    # Plot 12: Cos sim trajectory by vocab size
    fig = plot_trajectory_by_vocab("avg_cos_sim", "Avg Cosine Similarity")
    fig.savefig(PLOTS_DIR / "12_trajectory_cossim_vocab.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 12_trajectory_cossim_vocab.png")

    # Plot 13: Effective dim trajectory by vocab size
    fig = plot_trajectory_by_vocab("effective_dim", "Effective Dimension")
    fig.savefig(PLOTS_DIR / "13_trajectory_effdim_vocab.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 13_trajectory_effdim_vocab.png")

    # Plot 14: Cos sim trajectory by structure amount
    fig = plot_trajectory_by_structure("avg_cos_sim", "Avg Cosine Similarity")
    fig.savefig(PLOTS_DIR / "14_trajectory_cossim_structure.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 14_trajectory_cossim_structure.png")

    # Plot 15: Layer comparison
    fig = plot_layer_comparison()
    if fig:
        fig.savefig(PLOTS_DIR / "15_trajectory_layer_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Saved: 15_trajectory_layer_comparison.png")

    print("\nTrajectory analysis complete.")


if __name__ == "__main__":
    main()
