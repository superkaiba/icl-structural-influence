#!/usr/bin/env python3
"""Log MDS branching results to W&B."""

import json
import wandb
from pathlib import Path


def main():
    # Load results
    results_path = Path("results/stagewise_learning/mds_branching_results.json")
    with open(results_path) as f:
        results = json.load(f)

    # Initialize W&B
    run = wandb.init(
        project="icl-structural-influence",
        name="mds-branching-lee-et-al",
        tags=["lee-et-al", "mds", "branching", "figure-4b"],
        config={
            "model": results["model"],
            "context_lengths": results["context_lengths"],
            "branching_ctx": results["branching_ctx"],
        }
    )

    # Log figure
    fig_path = Path("results/stagewise_learning/mds_branching.png")
    if fig_path.exists():
        wandb.log({"mds_branching_figure": wandb.Image(str(fig_path))})

    # Log distances
    distances = results["distances"]
    dist_table = wandb.Table(
        columns=["context_length", "inter_cluster_distance"],
        data=[[int(k), v] for k, v in sorted(distances.items(), key=lambda x: int(x[0]))]
    )
    wandb.log({"inter_cluster_distance_table": dist_table})

    # Log tokens per cluster
    tokens = results["tokens_per_cluster"]
    wandb.summary["tokens_cluster_A"] = tokens["A"]
    wandb.summary["tokens_cluster_B"] = tokens["B"]
    wandb.summary["tokens_cluster_C"] = tokens["C"]
    wandb.summary["branching_ctx"] = results["branching_ctx"]

    print(f"\nMDS results logged to: {run.url}")
    wandb.finish()


if __name__ == "__main__":
    main()
