#!/usr/bin/env python3
"""
Log LOO metric comparison results to Weights & Biases.
"""

import json
import wandb
from pathlib import Path
import numpy as np


def main():
    results_dir = Path("results/loo_metric_comparison")

    # Load results
    with open(results_dir / "comparison_results.json") as f:
        results = json.load(f)

    # Initialize W&B
    run = wandb.init(
        project="icl-structural-influence",
        name="loo-metric-comparison",
        config={
            "experiment": "loo-metric-comparison",
            "context_lengths": results["context_lengths"],
            "n_trials": results["n_trials"],
            "layer_idx": results["layer_idx"],
            "metrics_compared": ["ratio", "dirichlet_energy", "css"],
        },
        tags=["loo", "metric-comparison", "park-et-al", "lee-et-al", "dirichlet-energy", "css"],
    )

    # Create comparison table
    table = wandb.Table(columns=[
        "Context Length",
        "Bridge Ratio Inf", "Bridge Ratio Std",
        "Anchor Ratio Inf", "Anchor Ratio Std",
        "Bridge Energy Inf", "Bridge Energy Std",
        "Anchor Energy Inf", "Anchor Energy Std",
        "CSS (All)", "CSS (Bridge)", "CSS (Anchor)",
        "Bridge Cross Dist Inf", "Anchor Cross Dist Inf",
        "Bridge Within Dist Inf", "Anchor Within Dist Inf",
    ])

    for N in results["context_lengths"]:
        N_str = str(N)
        data = results["by_N"].get(N_str, {})

        bridge = data.get("bridge", {})
        anchor = data.get("anchor", {})

        def get_metric(d, metric, stat):
            if metric in d and d[metric]:
                return d[metric].get(stat)
            return None

        table.add_data(
            N,
            get_metric(bridge, "ratio_influence", "mean"),
            get_metric(bridge, "ratio_influence", "std"),
            get_metric(anchor, "ratio_influence", "mean"),
            get_metric(anchor, "ratio_influence", "std"),
            get_metric(bridge, "energy_influence", "mean"),
            get_metric(bridge, "energy_influence", "std"),
            get_metric(anchor, "energy_influence", "mean"),
            get_metric(anchor, "energy_influence", "std"),
            data.get("css"),
            data.get("css_bridge"),
            data.get("css_anchor"),
            get_metric(bridge, "cross_dist_influence", "mean"),
            get_metric(anchor, "cross_dist_influence", "mean"),
            get_metric(bridge, "within_dist_influence", "mean"),
            get_metric(anchor, "within_dist_influence", "mean"),
        )

    wandb.log({"metric_comparison_table": table})

    # Log figures
    figures = [
        ("metric_comparison.png", "LOO Metric Comparison (Ratio vs Energy vs CSS)"),
        ("metric_comparison_detailed.png", "Detailed LOO Metric Comparison"),
    ]

    for filename, caption in figures:
        fig_path = results_dir / filename
        if fig_path.exists():
            wandb.log({filename.replace(".png", ""): wandb.Image(str(fig_path), caption=caption)})
            print(f"Logged: {filename}")

    # Log line plots for each metric
    context_lengths = results["context_lengths"]

    # Extract data for plots
    ratio_bridge = []
    ratio_anchor = []
    energy_bridge = []
    energy_anchor = []
    css_values = []
    css_bridge_values = []
    css_anchor_values = []

    for N in context_lengths:
        N_str = str(N)
        data = results["by_N"].get(N_str, {})
        bridge = data.get("bridge", {})
        anchor = data.get("anchor", {})

        ratio_bridge.append(bridge.get("ratio_influence", {}).get("mean"))
        ratio_anchor.append(anchor.get("ratio_influence", {}).get("mean"))
        energy_bridge.append(bridge.get("energy_influence", {}).get("mean"))
        energy_anchor.append(anchor.get("energy_influence", {}).get("mean"))
        css_values.append(data.get("css"))
        css_bridge_values.append(data.get("css_bridge"))
        css_anchor_values.append(data.get("css_anchor"))

    # Log as line data
    for i, N in enumerate(context_lengths):
        log_data = {"context_length": N}

        if ratio_bridge[i] is not None:
            log_data["ratio_influence/bridge"] = ratio_bridge[i]
        if ratio_anchor[i] is not None:
            log_data["ratio_influence/anchor"] = ratio_anchor[i]
        if energy_bridge[i] is not None:
            log_data["energy_influence/bridge"] = energy_bridge[i]
        if energy_anchor[i] is not None:
            log_data["energy_influence/anchor"] = energy_anchor[i]
        if css_values[i] is not None:
            log_data["css/all"] = css_values[i]
        if css_bridge_values[i] is not None:
            log_data["css/bridge"] = css_bridge_values[i]
        if css_anchor_values[i] is not None:
            log_data["css/anchor"] = css_anchor_values[i]

        wandb.log(log_data)

    # Add summary notes
    wandb.run.notes = """
# LOO Influence Metric Comparison

## Metrics Compared

### 1. Original Ratio Metric
- Formula: `Influence = Ratio(full) - Ratio(without_pos)`
- Where: `Ratio = cross_cluster_dist / within_cluster_dist`
- Interpretation: Positive = removing hurts structure

### 2. Dirichlet Energy (Park et al. 2024)
- Paper: "ICLR: In-Context Learning of Representations"
- Formula: `E(X) = sum_{i,j} A_{i,j} ||x_i - x_j||^2`
- Influence: `E(without_pos) - E(full)`
- Interpretation: Positive = removing increases energy (hurts structure)

### 3. CSS - Covariance Sample Significance (Lee et al.)
- Formula: `CSS = -Cov(loss, phi)`
- Where: loss = prediction loss, phi = between-cluster variance
- Interpretation: Positive = structure helps prediction
- **Now computed separately for bridge vs anchor positions**

## Key Findings

1. **CSS Bridge vs Anchor shows striking difference at short contexts**:
   - N=7: Bridge CSS = +61, Anchor CSS = -901
   - N=10: Bridge CSS = +514, Anchor CSS = -40
   - N=15: Bridge CSS = +2374, Anchor CSS = +898
   - Bridge tokens show STRONGER structure-prediction coupling

2. **Anchor CSS is NEGATIVE at very short contexts (N=7)**
   - Structure hurts prediction at anchor positions early on
   - Model may use pretraining priors that conflict with graph

3. **Both CSS values converge to ~0 at long contexts (N>100)**
   - With 12 tokens × 40+ repetitions, individual effects vanish

4. **Dirichlet Energy requires long contexts** (N>=75) to have valid data

5. **Phase transition around N=50-75** where CSS bridge/anchor converge

6. **Metric correlation r=0.411** - weak agreement between ratio and energy
"""

    wandb.finish()

    print(f"\n{'='*70}")
    print("W&B LOGGING COMPLETE")
    print(f"{'='*70}")
    print(f"\nView results at: {run.url}")


if __name__ == "__main__":
    main()
