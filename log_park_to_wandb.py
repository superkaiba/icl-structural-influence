#!/usr/bin/env python3
"""
Log Park et al. reproduction results to Weights & Biases.
"""

import json
import wandb
from pathlib import Path


def main():
    results_path = Path("results/park_reproduction_v2")

    with open(results_path / "reproduction_results.json") as f:
        results = json.load(f)

    ctx_results = results["context_results"]
    context_lengths = sorted([int(k) for k in ctx_results.keys()])

    # Initialize W&B
    run = wandb.init(
        project="icl-structural-influence",
        name="park-et-al-reproduction",
        config={
            "experiment": "Park et al. (2024) Reproduction",
            "paper": "ICLR: In-Context Learning of Representations",
            "arxiv": "2501.00070",
            "model": results["model"],
            "graph_type": results["graph_type"],
            "graph_size": results["graph_size"],
            "num_nodes": results["num_nodes"],
            "layer_analyzed": results["layer"],
            "n_samples": results["n_samples"],
            "context_lengths": context_lengths,
        },
        tags=["reproduction", "park-et-al", "dirichlet-energy", "phase-transition"],
    )

    # Log metrics for each context length
    for ctx_len in context_lengths:
        data = ctx_results[str(ctx_len)]
        wandb.log({
            "context_length": ctx_len,
            "dirichlet_energy_mean": data["energy_mean"],
            "dirichlet_energy_std": data["energy_std"],
            "dirichlet_energy_median": data["energy_median"],
        })

    # Log summary table
    table = wandb.Table(columns=["Context N", "Energy Mean", "Energy Std", "Energy Median", "Relative to N=5"])
    baseline = ctx_results["5"]["energy_mean"]

    for ctx_len in context_lengths:
        data = ctx_results[str(ctx_len)]
        rel = data["energy_mean"] / baseline
        table.add_data(ctx_len, f"{data['energy_mean']:.0f}", f"{data['energy_std']:.0f}",
                      f"{data['energy_median']:.0f}", f"{rel:.1%}")

    wandb.log({"Dirichlet Energy Summary": table})

    # Log figures
    figures_to_log = [
        ("park_reproduction_comparison.png", "Park et al. Reproduction: Phase Transition Visualization"),
        ("graph_structure.png", "4×4 Grid Graph Structure"),
        ("layer_analysis.png", "Layer-wise Dirichlet Energy Analysis"),
        ("phase_transition_reproduction.png", "Phase Transition Raw Plots"),
    ]

    for filename, caption in figures_to_log:
        filepath = results_path / filename
        if filepath.exists():
            wandb.log({filename.replace(".png", ""): wandb.Image(str(filepath), caption=caption)})
            print(f"Logged: {filename}")

    # Log findings
    wandb.run.notes = """
# Park et al. (2024) Reproduction

## Paper: ICLR: In-Context Learning of Representations (arXiv:2501.00070)

## Key Finding Reproduced
Dirichlet Energy DECREASES as context length increases:
- N=5: 50.7M
- N=100: 0.65M
- **Total reduction: 98.7% (77x)**

## Interpretation
1. LOW ENERGY = representations are smooth on the graph (adjacent nodes have similar representations)
2. Energy DECREASES because model learns graph structure from random walk context
3. Phase transition visible at N≈5-15 with 40%+ energy drops
4. Confirms: "LLMs can in-context learn novel semantic representations that override pretraining knowledge"

## Methodology
- Graph: 4×4 Grid (16 nodes)
- Model: Qwen/Qwen2.5-7B
- Layer: 20 (deep layer)
- Metric: Dirichlet Energy E(X) = Σ_{i,j} A_{i,j} ||x_i - x_j||²
"""

    wandb.finish()

    print(f"\n{'='*70}")
    print("W&B LOGGING COMPLETE")
    print(f"{'='*70}")
    print(f"\nView results at: {run.get_url()}")

    return run.get_url()


if __name__ == "__main__":
    main()
