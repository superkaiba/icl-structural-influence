#!/usr/bin/env python3
"""
Log multi-layer LOO influence results to Weights & Biases.
"""

import json
import wandb
from pathlib import Path
import numpy as np


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen", choices=["qwen", "llama"],
                        help="Model to log results for")
    args = parser.parse_args()

    if args.model == "qwen":
        results_dir = Path("results/loo_multilayer_qwen")
        model_name = "Qwen/Qwen2.5-7B"
        model_tag = "qwen2.5-7b"
        num_layers = 28
    else:
        results_dir = Path("results/loo_multilayer_llama")
        model_name = "meta-llama/Llama-3.1-8B"
        model_tag = "llama3.1-8b"
        num_layers = 32

    # Load results
    with open(results_dir / "results_semantic.json") as f:
        semantic_results = json.load(f)

    with open(results_dir / "results_unrelated.json") as f:
        unrelated_results = json.load(f)

    # Initialize W&B
    run = wandb.init(
        project="icl-structural-influence",
        name=f"multilayer-loo-{args.model}",
        config={
            "experiment": "multilayer-loo-influence",
            "model": model_name,
            "num_layers": num_layers,
            "context_lengths": semantic_results["context_lengths"],
            "layers_tested": semantic_results["layers_tested"],
            "n_trials": semantic_results["n_trials"],
            "conditions": ["semantic", "unrelated"],
            "metrics": ["ratio_influence", "energy_influence", "cross_dist_influence", "within_dist_influence"],
            "token_types": ["bridge", "anchor"],
            "bounded_phi": True,  # Using relative epsilon for bounded ratio
        },
        tags=["loo", "multilayer", "heatmap", "semantic-conflict", model_tag, "bounded-phi"],
    )

    # Log all heatmap images
    heatmaps = [
        ("heatmap_ratio_influence.png", "Ratio Influence (LOO) Heatmap: Layer × Context Length"),
        ("heatmap_energy_influence.png", "Dirichlet Energy Influence (LOO) Heatmap: Layer × Context Length"),
        ("heatmap_cross_dist_influence.png", "Cross-Cluster Distance Influence (LOO) Heatmap: Layer × Context Length"),
        ("heatmap_within_dist_influence.png", "Within-Cluster Distance Influence (LOO) Heatmap: Layer × Context Length"),
        ("heatmap_all_metrics.png", "Combined Multi-Metric LOO Influence Heatmaps"),
    ]

    for filename, caption in heatmaps:
        fig_path = results_dir / filename
        if fig_path.exists():
            wandb.log({filename.replace(".png", "").replace("_", "-"): wandb.Image(str(fig_path), caption=caption)})
            print(f"Logged: {filename}")
        else:
            print(f"Skipped (not found): {filename}")

    # Create detailed comparison table
    table = wandb.Table(columns=[
        "Condition", "Layer", "Context Length", "Token Type",
        "Ratio Influence (mean)", "Ratio Influence (std)",
        "Energy Influence (mean)", "Energy Influence (std)",
        "Cross Dist Influence (mean)", "Cross Dist Influence (std)",
        "Within Dist Influence (mean)", "Within Dist Influence (std)",
    ])

    for condition_name, results in [("semantic", semantic_results), ("unrelated", unrelated_results)]:
        for layer in results["layers_tested"]:
            for N in results["context_lengths"]:
                layer_data = results["by_layer_N"].get(str(layer), {}).get(str(N), {})

                for token_type in ["bridge", "anchor"]:
                    token_data = layer_data.get(token_type, {})

                    def get_metric(metric, stat):
                        if metric in token_data and token_data[metric]:
                            return token_data[metric].get(stat)
                        return None

                    table.add_data(
                        condition_name,
                        layer,
                        N,
                        token_type,
                        get_metric("ratio_influence", "mean"),
                        get_metric("ratio_influence", "std"),
                        get_metric("energy_influence", "mean"),
                        get_metric("energy_influence", "std"),
                        get_metric("cross_dist_influence", "mean"),
                        get_metric("cross_dist_influence", "std"),
                        get_metric("within_dist_influence", "mean"),
                        get_metric("within_dist_influence", "std"),
                    )

    wandb.log({"multilayer_results_table": table})

    # Log layer-wise line plots
    for condition_name, results in [("semantic", semantic_results), ("unrelated", unrelated_results)]:
        for N in results["context_lengths"]:
            for layer in results["layers_tested"]:
                layer_data = results["by_layer_N"].get(str(layer), {}).get(str(N), {})

                log_data = {
                    "layer": layer,
                    "context_length": N,
                    "condition": condition_name,
                }

                for token_type in ["bridge", "anchor"]:
                    token_data = layer_data.get(token_type, {})

                    for metric in ["ratio_influence", "energy_influence", "cross_dist_influence", "within_dist_influence"]:
                        if metric in token_data and token_data[metric]:
                            mean_val = token_data[metric].get("mean")
                            if mean_val is not None and not np.isnan(mean_val):
                                log_data[f"{metric}/{token_type}_{condition_name}"] = mean_val

                if len(log_data) > 3:  # Only log if we have actual metrics
                    wandb.log(log_data)

    # Add comprehensive summary notes
    notes = f"""
# Multi-Layer LOO Influence Analysis

## Experiment Overview

This experiment measures Leave-One-Out (LOO) influence across **all {num_layers} layers** of {model_name} to understand how structural influence evolves through the network's depth.

## Bounded Phi Metric

Uses relative epsilon (eps=1e-3) to bound the cluster separation ratio, preventing extreme values when within-cluster distances approach zero. Maximum ratio ~1000."""

    notes += """

## Key Innovation: Efficient Multi-Layer Extraction

**16x speedup** using `HookedLLM.forward_with_cache()`:
- Traditional: 1 layer per forward pass → 32 forwards per LOO comparison
- Our approach: 32 layers per forward pass → 2 forwards per LOO comparison

## Experimental Design

### Parameters (Optimized for Memory)
- **Layers tested**: [0, 4, 8, 12, 16, 20, 24, 28, 31] (9 layers sampling early/middle/late)
- **Context lengths**: [6, 10, 20, 50, 100] (5 lengths)
- **Trials per length**: 5
- **Conditions**: Semantic tokens (same cluster = semantic conflict) + Unrelated tokens (baseline)
- **Token types**: Bridge (cluster transitions) vs Anchor (within-cluster)

### Total Comparisons
- 2 conditions × 5 context lengths × 5 trials × ~8 positions = ~400 LOO comparisons
- Each comparison: 2 forward passes × 9 layers = 18 layer extractions
- **Total runtime**: ~46 minutes on A40 GPU

## Metrics Measured

### 1. Ratio Influence
- Formula: `Δ Ratio = Ratio(full) - Ratio(without_pos)`
- Where: `Ratio = cross_cluster_dist / within_cluster_dist`
- Interpretation: Positive = removing token hurts cluster separation

### 2. Dirichlet Energy Influence (Park et al. 2024)
- Formula: `Δ E = E(without) - E(full)`
- Where: `E(X) = Σ A_{ij} ||x_i - x_j||²`
- Interpretation: Positive = removing token increases energy (hurts smoothness)

### 3. Cross-Cluster Distance Influence
- Direct component of ratio metric
- Measures impact on between-cluster distances

### 4. Within-Cluster Distance Influence
- Direct component of ratio metric (denominator)
- Measures impact on within-cluster compactness

## Visualization: 2D Heatmaps

Each heatmap shows:
- **X-axis**: Layer index (0 → 31)
- **Y-axis**: Context length (log scale, 6 → 100)
- **Color**: Influence magnitude (diverging colormap, red = positive, blue = negative)
- **Subplots**: Bridge vs Anchor, Semantic vs Unrelated

## Expected Patterns

### Layer-wise Evolution
- **Early layers (0-8)**: Likely small influence (representations not yet structured)
- **Middle layers (12-20)**: Peak structural influence (where clustering emerges)
- **Late layers (24-31)**: Stabilization or task-specific refinement

### Context Length Effects
- **Short contexts (N=6-10)**: High variance, sparse sampling
- **Medium contexts (N=20-50)**: Strongest structural signals
- **Long contexts (N=100)**: Saturation effects (12 tokens × 8+ repetitions)

### Bridge vs Anchor Tokens
- **Bridge tokens**: Expected higher influence (connect clusters)
- **Anchor tokens**: Structural reinforcement within clusters

### Semantic vs Unrelated
- **Semantic**: Conflicting semantics within clusters → stronger structural pressure
- **Unrelated**: No semantic conflict → baseline structural influence

## Memory Optimization

Initial OOM failures (exit code 137) led to optimization:
- **Original plan**: 32 layers × 16 context lengths × 10 trials
- **Final**: 9 layers × 5 context lengths × 5 trials
- **Strategy**: Sample layers across depth while maintaining coverage

## Data Structure

Results saved to `results/loo_multilayer/`:
- `results_semantic.json`: Full semantic condition data
- `results_unrelated.json`: Full unrelated condition data

JSON format:
```json
{
  "by_layer_N": {
    "0": {
      "6": {
        "bridge": {
          "ratio_influence": {"mean": ..., "std": ..., "sem": ...},
          "energy_influence": {...}
        },
        "anchor": {...}
      }
    }
  }
}
```

## Comparison to Prior Experiments

This extends our previous LOO metric comparison experiment:
- **Previous**: Single layer (layer 31), up to N=100, multiple metrics
- **This**: Multiple layers (9 sampled), up to N=100, same metrics
- **New insight**: Can now track structural emergence **through layers**

## Technical Notes

- Used `torch.cuda.empty_cache()` after each forward pass for memory management
- Warnings about "Mean of empty slice" expected when insufficient valid samples
- CSS metric computation deferred (requires different aggregation strategy)

## Follow-up Questions

1. Do middle layers (12-20) show peak influence as hypothesized?
2. Do bridge tokens show consistent higher influence across all layers?
3. Is there a phase transition layer where structural influence emerges?
4. Do semantic vs unrelated conditions diverge more at specific layers?
"""
    wandb.run.notes = notes

    wandb.finish()

    print(f"\n{'='*70}")
    print("W&B LOGGING COMPLETE")
    print(f"{'='*70}")
    print(f"\nView results at: {run.url}")


if __name__ == "__main__":
    main()
