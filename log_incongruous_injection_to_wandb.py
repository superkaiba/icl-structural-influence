#!/usr/bin/env python3
"""
Log incongruous token injection experiment results to Weights & Biases.

Usage:
    python log_incongruous_injection_to_wandb.py
    python log_incongruous_injection_to_wandb.py --results-file results/incongruous_injection/injection_results_gpt2.json
"""

import argparse
import json
from pathlib import Path

import wandb
import numpy as np


def log_injection_results(results_file: str):
    """Log injection experiment results to W&B."""

    # Load results
    results_path = Path(results_file)
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return

    with open(results_path) as f:
        results = json.load(f)

    metadata = results["metadata"]
    model_name = metadata["model"].replace("/", "_")

    # Initialize W&B
    run = wandb.init(
        project="icl-structural-influence",
        name=f"incongruous-injection-{model_name}",
        tags=["injection", "position-analysis", "incongruous-token", metadata["model"]],
        config={
            "model": metadata["model"],
            "context_lengths": metadata["context_lengths"],
            "n_samples": metadata["n_samples"],
            "layers": metadata["layers"],
            "position_step": metadata["position_step"],
            "token_types": metadata["token_types"],
            "seed": metadata["seed"],
        }
    )

    token_types = metadata["token_types"]
    context_lengths = metadata["context_lengths"]
    layers = metadata["layers"]
    mid_layer = str(layers[len(layers) // 2])

    # Log summary metrics
    wandb.summary["num_token_types"] = len(token_types)
    wandb.summary["num_context_lengths"] = len(context_lengths)
    wandb.summary["num_layers_tested"] = len(layers)

    # Table 1: Position effect summary (aggregated across positions)
    # Use median_abs for robustness to outliers
    position_table = wandb.Table(columns=[
        "token_type", "context_length", "position", "phi_delta_median_abs", "phi_delta_iqr_abs",
        "loss_at_injection_mean", "loss_after_injection_mean"
    ])

    for token_type in token_types:
        for N in context_lengths:
            ctx_data = results["by_token_type"][token_type]["by_context_length"].get(str(N), {})
            if not ctx_data:
                ctx_data = results["by_token_type"][token_type]["by_context_length"].get(N, {})

            for pos_key, pos_data in ctx_data.items():
                phi_delta = pos_data.get("phi_delta_by_layer", {}).get(mid_layer, {})
                loss_at = pos_data.get("loss_at_injection", {})
                loss_after = pos_data.get("loss_after_injection", {})

                position_table.add_data(
                    token_type,
                    N,
                    int(pos_key),
                    phi_delta.get("median_abs", abs(phi_delta.get("median", 0.0))),
                    phi_delta.get("iqr_abs", phi_delta.get("iqr", 0.0)),
                    loss_at.get("mean", 0.0),
                    loss_after.get("mean", 0.0),
                )

    wandb.log({"position_effect_table": position_table})

    # Table 2: Layer sensitivity (phi delta by layer for each token type)
    # Use median_abs for robustness to outliers
    layer_table = wandb.Table(columns=[
        "token_type", "context_length", "position_type", "layer", "phi_delta_median_abs"
    ])

    for token_type in token_types:
        pos_data = results["by_token_type"][token_type]["by_position"]
        positions = sorted([int(p) for p in pos_data.keys()])

        if not positions:
            continue

        early_pos = str(positions[0])
        late_pos = str(positions[-1])

        for N in context_lengths:
            for layer in layers:
                layer_key = str(layer)

                # Early injection
                early_data = pos_data.get(early_pos, {}).get(N, {})
                if not early_data:
                    early_data = pos_data.get(early_pos, {}).get(str(N), {})
                early_delta = early_data.get("phi_delta_by_layer", {}).get(layer_key, {})
                layer_table.add_data(
                    token_type, N, "early", layer,
                    early_delta.get("median_abs", abs(early_delta.get("median", 0.0)))
                )

                # Late injection
                late_data = pos_data.get(late_pos, {}).get(N, {})
                if not late_data:
                    late_data = pos_data.get(late_pos, {}).get(str(N), {})
                late_delta = late_data.get("phi_delta_by_layer", {}).get(layer_key, {})
                layer_table.add_data(
                    token_type, N, "late", layer,
                    late_delta.get("median_abs", abs(late_delta.get("median", 0.0)))
                )

    wandb.log({"layer_sensitivity_table": layer_table})

    # Table 3: Token type comparison (average effect per token type and context length)
    # Use median of median_abs values for robustness
    comparison_table = wandb.Table(columns=[
        "token_type", "context_length", "median_phi_delta_abs", "avg_loss_at_injection"
    ])

    for token_type in token_types:
        for N in context_lengths:
            ctx_data = results["by_token_type"][token_type]["by_context_length"].get(str(N), {})
            if not ctx_data:
                ctx_data = results["by_token_type"][token_type]["by_context_length"].get(N, {})

            phi_deltas_abs = []
            losses = []
            for pos_key, pos_data in ctx_data.items():
                delta = pos_data.get("phi_delta_by_layer", {}).get(mid_layer, {})
                loss = pos_data.get("loss_at_injection", {})
                if delta:
                    phi_deltas_abs.append(delta.get("median_abs", abs(delta.get("median", 0.0))))
                if loss:
                    losses.append(loss.get("mean", 0.0))

            median_delta_abs = np.median(phi_deltas_abs) if phi_deltas_abs else 0.0
            avg_loss = np.mean(losses) if losses else 0.0

            comparison_table.add_data(token_type, N, median_delta_abs, avg_loss)

    wandb.log({"token_type_comparison_table": comparison_table})

    # Line plots: Position effect by token type (using median_abs)
    for token_type in token_types:
        pos_data = results["by_token_type"][token_type]["by_position"]
        positions = sorted([int(p) for p in pos_data.keys()])

        if not positions:
            continue

        for N in context_lengths:
            phi_deltas = []
            valid_positions = []
            for pos in positions:
                pos_key = str(pos)
                n_data = pos_data.get(pos_key, {}).get(N, {})
                if not n_data:
                    n_data = pos_data.get(pos_key, {}).get(str(N), {})
                delta = n_data.get("phi_delta_by_layer", {}).get(mid_layer, {})
                if delta:
                    phi_deltas.append(delta.get("median_abs", abs(delta.get("median", 0.0))))
                    valid_positions.append(pos)

            if valid_positions:
                data = [[pos, delta] for pos, delta in zip(valid_positions, phi_deltas)]
                table = wandb.Table(data=data, columns=["position", "phi_delta_median_abs"])
                wandb.log({
                    f"position_effect_{token_type}_N{N}": wandb.plot.line(
                        table, "position", "phi_delta_median_abs",
                        title=f"{token_type} Position Effect |Φ Δ| (N={N})"
                    )
                })

    # Log visualizations if they exist
    fig_dir = results_path.parent

    fig_files = [
        "position_effect_by_token_type.png",
        "heatmap_position_vs_context.png",
        "layer_sensitivity_analysis.png",
        "token_type_comparison.png",
    ]

    for fig_file in fig_files:
        fig_path = fig_dir / fig_file
        if fig_path.exists():
            wandb.log({fig_file.replace(".png", ""): wandb.Image(str(fig_path))})
            print(f"  Logged figure: {fig_file}")

    # Compute and log key findings (using median_abs for robustness)
    print("\nKey Findings:")

    # Finding 1: Which token type has the largest median effect?
    median_effects = {}
    for token_type in token_types:
        all_deltas_abs = []
        for N in context_lengths:
            ctx_data = results["by_token_type"][token_type]["by_context_length"].get(str(N), {})
            if not ctx_data:
                ctx_data = results["by_token_type"][token_type]["by_context_length"].get(N, {})
            for pos_data in ctx_data.values():
                delta = pos_data.get("phi_delta_by_layer", {}).get(mid_layer, {})
                if delta:
                    all_deltas_abs.append(delta.get("median_abs", abs(delta.get("median", 0.0))))
        median_effects[token_type] = np.median(all_deltas_abs) if all_deltas_abs else 0.0

    max_effect_type = max(median_effects, key=median_effects.get)
    wandb.summary["most_disruptive_token_type"] = max_effect_type
    wandb.summary["most_disruptive_effect_magnitude"] = median_effects[max_effect_type]
    print(f"  Most disruptive token type: {max_effect_type} (median |Φ delta| = {median_effects[max_effect_type]:.4f})")

    # Finding 2: Early vs Late injection comparison (using median_abs)
    for token_type in ["wrong_cluster", "semantic_outlier"]:
        if token_type not in token_types:
            continue

        pos_data = results["by_token_type"][token_type]["by_position"]
        positions = sorted([int(p) for p in pos_data.keys()])

        if len(positions) < 2:
            continue

        early_pos = str(positions[0])
        late_pos = str(positions[-1])

        early_deltas_abs = []
        late_deltas_abs = []

        for N in context_lengths:
            early_data = pos_data.get(early_pos, {}).get(N, {})
            if not early_data:
                early_data = pos_data.get(early_pos, {}).get(str(N), {})
            late_data = pos_data.get(late_pos, {}).get(N, {})
            if not late_data:
                late_data = pos_data.get(late_pos, {}).get(str(N), {})

            early_delta = early_data.get("phi_delta_by_layer", {}).get(mid_layer, {})
            late_delta = late_data.get("phi_delta_by_layer", {}).get(mid_layer, {})

            if early_delta:
                early_deltas_abs.append(early_delta.get("median_abs", abs(early_delta.get("median", 0.0))))
            if late_delta:
                late_deltas_abs.append(late_delta.get("median_abs", abs(late_delta.get("median", 0.0))))

        if early_deltas_abs and late_deltas_abs:
            early_median = np.median(early_deltas_abs)
            late_median = np.median(late_deltas_abs)
            wandb.summary[f"{token_type}_early_median_abs"] = early_median
            wandb.summary[f"{token_type}_late_median_abs"] = late_median
            wandb.summary[f"{token_type}_early_vs_late_ratio"] = early_median / late_median if late_median != 0 else float('inf')
            print(f"  {token_type}: Early median |Φ Δ| = {early_median:.4f}, Late median |Φ Δ| = {late_median:.4f}")

    print(f"\nResults logged to W&B run: {run.url}")
    wandb.finish()


def main():
    parser = argparse.ArgumentParser(
        description="Log incongruous injection results to W&B"
    )
    parser.add_argument(
        "--results-file", type=str,
        default="results/incongruous_injection/injection_results_gpt2.json",
        help="Path to results JSON file"
    )
    args = parser.parse_args()

    log_injection_results(args.results_file)


if __name__ == "__main__":
    main()
