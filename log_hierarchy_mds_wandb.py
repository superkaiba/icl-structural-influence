#!/usr/bin/env python3
"""Log hierarchy MDS results to W&B."""

import json
import wandb
from pathlib import Path


def main():
    output_dir = Path("results/hierarchy_mds_plots")

    # Fine-grained context lengths up to 10,000
    context_lengths = (
        [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500, 750] +
        list(range(1000, 10001, 100))  # 1000, 1100, ..., 10000
    )

    # Initialize W&B
    run = wandb.init(
        project="icl-structural-influence",
        name="hierarchy-mds-16tokens-N10000-with-metrics",
        tags=["lee-et-al", "mds", "hierarchy", "3-level", "stagewise-learning", "long-context",
              "deterministic", "perplexity", "influence-metrics", "fine-grained"],
        config={
            "model": "meta-llama/Llama-3.1-8B",
            "num_tokens": 16,
            "tokens_per_mid_cluster": 4,
            "num_mid_clusters": 4,
            "num_super_clusters": 2,
            "context_lengths": context_lengths,
            "num_context_lengths": len(context_lengths),
            "n_samples": 50,
            "layer_idx": -5,
            "dim_reduction": "MDS",
            "edge_type": "deterministic",
            "metrics_tracked": ["perplexity", "influence_ratios", "hierarchy_distances"],
        }
    )

    # Log all figures
    figures = [
        ("hierarchy_trajectories_super.png", "Token Trajectories by Super-Cluster"),
        ("hierarchy_trajectories_mid.png", "Token Trajectories by Mid-Cluster"),
        ("hierarchy_centroids_super.png", "Super-Cluster Centroid Trajectories"),
        ("hierarchy_centroids_mid.png", "Mid-Cluster Centroid Trajectories"),
        ("hierarchy_lee_et_al_style.png", "Combined Lee et al. Style Figure"),
        ("hierarchy_zoomed_late.png", "Zoomed Late Context (N=30-100) with Labels"),
        ("hierarchy_centroids_labeled.png", "Centroid Trajectories with Context Labels"),
        ("hierarchy_final_n100.png", "Final Representations at N=100 with Token Names"),
        ("hierarchy_midcluster_evolution.png", "Mid-Cluster Evolution with Lines and Sized Points"),
        ("hierarchy_midcluster_arrows.png", "Mid-Cluster Evolution with Direction Arrows"),
        ("hierarchy_supercluster_arrows.png", "Super-Cluster Evolution with Direction Arrows"),
        ("hierarchy_evolution_combined.png", "Combined Super + Mid Cluster Evolution"),
    ]

    for filename, description in figures:
        fig_path = output_dir / filename
        if fig_path.exists():
            wandb.log({filename.replace(".png", ""): wandb.Image(str(fig_path), caption=description)})
            print(f"Logged: {filename}")

    # Log hierarchy structure as table
    hierarchy_table = wandb.Table(
        columns=["token", "super_cluster", "mid_cluster"],
        data=[
            ["crystal", "Super_A", "Mid_A1"],
            ["marble", "Super_A", "Mid_A1"],
            ["diamond", "Super_A", "Mid_A1"],
            ["granite", "Super_A", "Mid_A1"],
            ["lantern", "Super_A", "Mid_A2"],
            ["castle", "Super_A", "Mid_A2"],
            ["beacon", "Super_A", "Mid_A2"],
            ["fortress", "Super_A", "Mid_A2"],
            ["cloud", "Super_B", "Mid_B1"],
            ["canvas", "Super_B", "Mid_B1"],
            ["mist", "Super_B", "Mid_B1"],
            ["fabric", "Super_B", "Mid_B1"],
            ["pillar", "Super_B", "Mid_B2"],
            ["tunnel", "Super_B", "Mid_B2"],
            ["column", "Super_B", "Mid_B2"],
            ["passage", "Super_B", "Mid_B2"],
        ]
    )
    wandb.log({"hierarchy_structure": hierarchy_table})

    # Log summary
    wandb.summary["total_mds_points"] = 158  # some tokens missing at N=1
    wandb.summary["max_context_length"] = 10000
    wandb.summary["edge_type"] = "deterministic"
    wandb.summary["edges_within_mid"] = 24  # fully connected within each mid-cluster
    wandb.summary["edges_sibling_mid"] = 4   # 2 edges between M_A1-M_A2, 2 between M_B1-M_B2
    wandb.summary["edges_cross_super"] = 1   # single edge for connectivity
    wandb.summary["hierarchy_levels"] = 3
    wandb.summary["experiment_type"] = "stagewise_learning_replication"

    # Log interactive HTML plots
    interactive_plots = [
        "interactive_midcluster_evolution.html",
        "interactive_supercluster_evolution.html",
        "interactive_all_tokens.html",
        "interactive_combined.html",
        "interactive_tokens_and_centroids.html",
        "interactive_slider.html",
        # Enhanced plots with perplexity and influence metrics
        "interactive_enhanced_slider.html",
        "perplexity_by_cluster.html",
        "influence_ratios.html",
        "clustering_quality.html",
    ]

    for filename in interactive_plots:
        html_path = output_dir / filename
        if html_path.exists():
            wandb.log({filename.replace(".html", ""): wandb.Html(open(html_path).read())})
            print(f"Logged interactive: {filename}")

    # Log enhanced metrics from JSON
    metrics_path = output_dir / "enhanced_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

        # Log influence metrics as line plots
        if "influence_metrics" in metrics:
            inf = metrics["influence_metrics"]
            ctx_lens = inf["context_lengths"]

            # Create W&B tables for metrics
            influence_table = wandb.Table(
                columns=["context_length", "ratio_super_to_mid", "ratio_super_to_sibling",
                         "within_mid", "within_super", "across_super"]
            )
            for i, ctx in enumerate(ctx_lens):
                influence_table.add_data(
                    ctx,
                    inf["ratio_super_to_mid"][i],
                    inf["ratio_super_to_sibling"][i],
                    inf["within_mid"][i],
                    inf["within_super"][i],
                    inf["across_super"][i]
                )
            wandb.log({"influence_metrics_table": influence_table})

        # Log perplexity by cluster
        if "mid_cluster_perplexity" in metrics:
            ppl = metrics["mid_cluster_perplexity"]
            ctx_lens = metrics["context_lengths"]

            ppl_table = wandb.Table(
                columns=["context_length", "M_A1_ppl", "M_A2_ppl", "M_B1_ppl", "M_B2_ppl"]
            )
            for i, ctx in enumerate(ctx_lens):
                ppl_table.add_data(
                    ctx,
                    ppl["M_A1"][i] if i < len(ppl["M_A1"]) else 0,
                    ppl["M_A2"][i] if i < len(ppl["M_A2"]) else 0,
                    ppl["M_B1"][i] if i < len(ppl["M_B1"]) else 0,
                    ppl["M_B2"][i] if i < len(ppl["M_B2"]) else 0,
                )
            wandb.log({"perplexity_by_cluster_table": ppl_table})

        # Log clustering quality metrics
        if "clustering_metrics" in metrics:
            clust = metrics["clustering_metrics"]
            ctx_lens = clust["context_lengths"]

            clustering_table = wandb.Table(
                columns=["context_length", "silhouette_mid", "silhouette_super",
                         "davies_bouldin", "calinski_harabasz"]
            )
            for i, ctx in enumerate(ctx_lens):
                clustering_table.add_data(
                    ctx,
                    clust["silhouette_mid"][i],
                    clust["silhouette_super"][i],
                    clust["davies_bouldin"][i],
                    clust["calinski_harabasz"][i]
                )
            wandb.log({"clustering_metrics_table": clustering_table})

            # Log best clustering metrics to summary
            best_sil_idx = clust["silhouette_mid"].index(max(clust["silhouette_mid"]))
            wandb.summary["best_silhouette_mid"] = clust["silhouette_mid"][best_sil_idx]
            wandb.summary["best_silhouette_mid_N"] = ctx_lens[best_sil_idx]

        print("Logged enhanced metrics tables")

    print(f"\nHierarchy MDS results logged to: {run.url}")
    wandb.finish()


if __name__ == "__main__":
    main()
