#!/usr/bin/env python3
"""
Log stagewise learning results to Weights & Biases.
"""

import json
import wandb
from pathlib import Path

def main():
    # Load results
    results_path = Path("results/stagewise_learning/stagewise_results.json")
    with open(results_path) as f:
        results = json.load(f)

    # Initialize W&B
    run = wandb.init(
        project="icl-structural-influence",
        name="stagewise-learning-lee-et-al",
        tags=["lee-et-al", "stagewise", "hierarchical", "mds"],
        config={
            "model": results["model"],
            "context_lengths": results["context_lengths"],
            "n_samples": results["n_samples"],
            "layer_idx": results["layer_idx"],
        }
    )

    # Log summary metrics
    wandb.summary["sign_flips"] = results["sign_flips"]
    wandb.summary["branching_ctx"] = results["branching_ctx"]
    wandb.summary["same_cluster_css_final"] = results["same_cluster_css"]["100"]
    wandb.summary["diff_cluster_css_final"] = results["diff_cluster_css"]["100"]

    # Log CSS trajectories as tables
    ctx_lengths = results["context_lengths"]

    # Same vs Different cluster CSS table
    css_table = wandb.Table(columns=["context_length", "same_cluster_css", "diff_cluster_css", "ratio"])
    for ctx in ctx_lengths:
        same = results["same_cluster_css"][str(ctx)]
        diff = results["diff_cluster_css"][str(ctx)]
        ratio = same / diff if diff != 0 else float('inf')
        css_table.add_data(ctx, same, diff, ratio)

    wandb.log({"css_comparison_table": css_table})

    # Log influence matrix as table
    influence_table = wandb.Table(columns=["cluster_pair", "context_length", "css"])
    for pair_key, ctx_vals in results["influence_matrix"].items():
        for ctx, css in ctx_vals.items():
            influence_table.add_data(pair_key, int(ctx), css)

    wandb.log({"influence_matrix_table": influence_table})

    # Log the main figure
    fig_path = Path("results/stagewise_learning/stagewise_learning.png")
    if fig_path.exists():
        wandb.log({"stagewise_learning_figure": wandb.Image(str(fig_path))})

    # Log line plots
    same_css_data = [[ctx, results["same_cluster_css"][str(ctx)]] for ctx in ctx_lengths]
    diff_css_data = [[ctx, results["diff_cluster_css"][str(ctx)]] for ctx in ctx_lengths]

    wandb.log({
        "same_cluster_css_plot": wandb.plot.line_series(
            xs=[ctx_lengths, ctx_lengths],
            ys=[[results["same_cluster_css"][str(ctx)] for ctx in ctx_lengths],
                [results["diff_cluster_css"][str(ctx)] for ctx in ctx_lengths]],
            keys=["Same Cluster", "Different Cluster"],
            title="Cluster-Pair Influence (CSS)",
            xname="Context Length"
        )
    })

    # Log distances
    distances = results["distances"]
    dist_data = [[int(ctx), dist] for ctx, dist in distances.items()]
    wandb.log({
        "inter_cluster_distance": wandb.Table(
            columns=["context_length", "distance"],
            data=dist_data
        )
    })

    print(f"\nResults logged to W&B run: {run.url}")

    wandb.finish()


if __name__ == "__main__":
    main()


def log_mds_branching():
    """Log MDS branching results."""
    import wandb

    # Load results
    results_path = Path("results/stagewise_learning/mds_branching_results.json")
    if not results_path.exists():
        print("MDS results not found, skipping...")
        return

    with open(results_path) as f:
        results = json.load(f)

    # Log to existing run or create new one
    run = wandb.init(
        project="icl-structural-influence",
        name="mds-branching-lee-et-al",
        tags=["lee-et-al", "mds", "branching"],
        config=results
    )

    # Log figure
    fig_path = Path("results/stagewise_learning/mds_branching.png")
    if fig_path.exists():
        wandb.log({"mds_branching_figure": wandb.Image(str(fig_path))})

    # Log distances
    distances = results["distances"]
    dist_table = wandb.Table(
        columns=["context_length", "inter_cluster_distance"],
        data=[[int(k), v] for k, v in distances.items()]
    )
    wandb.log({"inter_cluster_distance_table": dist_table})

    wandb.summary["mds_branching_ctx"] = results["branching_ctx"]

    print(f"\nMDS results logged to W&B run: {run.url}")
    wandb.finish()


if __name__ == "__main__":
    main()
    log_mds_branching()
