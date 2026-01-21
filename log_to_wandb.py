#!/usr/bin/env python3
"""
Log experiment results and figures to Weights & Biases.
"""

import json
import wandb
from pathlib import Path


def main():
    results_base = Path("/workspace/research/projects/in_context_representation_influence/results")

    # Load all model results
    all_results = {}
    for subdir in results_base.iterdir():
        if subdir.is_dir() and subdir.name != "combined_analysis":
            for json_file in subdir.glob("*_results.json"):
                if not json_file.name.startswith("all_"):
                    with open(json_file) as f:
                        data = json.load(f)
                        model_name = data.get("model", json_file.stem.replace("_results", ""))
                        if model_name not in all_results:
                            all_results[model_name] = data

    # Load summary statistics
    summary_path = results_base / "combined_analysis" / "summary_statistics.json"
    with open(summary_path) as f:
        summary = json.load(f)

    # Initialize wandb run
    run = wandb.init(
        project="icl-structural-influence",
        name="hierarchical-graph-tracing-experiment",
        config={
            "models": list(all_results.keys()),
            "context_lengths": [10, 20, 30, 50, 75, 100, 150, 200],
            "n_contexts_per_length": 25,
            "graph_config": {
                "num_superclusters": 3,
                "nodes_per_cluster": 5,
            },
            "metrics": ["dirichlet_energy", "cluster_separation", "representation_coherence"],
        },
        tags=["icl", "phase-transitions", "hierarchical-learning", "mechanistic-interpretability"],
    )

    # Log summary statistics as metrics
    for model_name, model_data in summary["model_comparisons"].items():
        wandb.log({
            f"{model_name}/n_layers": model_data["n_layers"],
            f"{model_name}/hidden_size": model_data["hidden_size"],
            f"{model_name}/dirichlet_min": model_data["dirichlet_range"][0],
            f"{model_name}/dirichlet_max": model_data["dirichlet_range"][1],
            f"{model_name}/phase_transition_ctx": model_data["potential_phase_transition_at"],
            f"{model_name}/peak_rate_of_change": model_data["peak_rate_of_change"],
            f"{model_name}/overall_trend_slope": model_data["overall_trend_slope"],
        })

    # Log per-context-length metrics for each model
    for model_name, model_data in all_results.items():
        ctx_results = model_data.get("context_length_results", {})
        layers_analyzed = model_data.get("layers_analyzed", [])

        for ctx_len, ctx_data in sorted(ctx_results.items(), key=lambda x: int(x[0])):
            layers = ctx_data.get("layers", {})
            final_layer_key = f"layer_{layers_analyzed[-1]}" if layers_analyzed else None

            if final_layer_key and final_layer_key in layers:
                layer_data = layers[final_layer_key]

                if "cluster_sep" in layer_data:
                    css = layer_data["cluster_sep"]
                    wandb.log({
                        f"{model_name}/context_length": int(ctx_len),
                        f"{model_name}/phi_mean": css.get("phi_mean", 0),
                        f"{model_name}/phi_std": css.get("phi_std", 0),
                        f"{model_name}/max_sensitivity": css.get("max_sensitivity", 0),
                        f"{model_name}/mean_abs_sensitivity": css.get("mean_abs_sensitivity", 0),
                    })

    # Log figures
    combined_analysis = results_base / "combined_analysis"

    figures_to_log = [
        ("combined_money_plot.png", "Combined Money Plot - Structural Emergence vs Context Length"),
        ("phase_transition_analysis.png", "Phase Transition Analysis - Rate of Structural Change"),
        ("layer_trajectory_comparison.png", "Layer Trajectory Comparison"),
        ("Qwen2.5-7B_layer_heatmap.png", "Qwen2.5-7B Layer Heatmap"),
        ("LLaMA-3.1-8B_layer_heatmap.png", "LLaMA-3.1-8B Layer Heatmap"),
    ]

    for filename, caption in figures_to_log:
        filepath = combined_analysis / filename
        if filepath.exists():
            wandb.log({filename.replace(".png", ""): wandb.Image(str(filepath), caption=caption)})
            print(f"Logged: {filename}")

    # Log per-model figures from other directories
    for subdir in results_base.iterdir():
        if subdir.is_dir() and subdir.name != "combined_analysis":
            for png_file in subdir.glob("*.png"):
                if "heatmap" in png_file.name or "money_plot" in png_file.name:
                    caption = f"{subdir.name}/{png_file.name}"
                    wandb.log({f"{subdir.name}_{png_file.stem}": wandb.Image(str(png_file), caption=caption)})
                    print(f"Logged: {caption}")

    # Log the phase transition discovery table as a wandb.Table
    phase_transition_table = wandb.Table(columns=["Model", "N=10 Φ", "N=20 Φ", "N=30 Φ", "N=10 CSS", "N=20 CSS", "N=30 CSS", "Spike Factor (Φ)", "Spike Factor (CSS)"])

    for model_name, model_data in all_results.items():
        ctx_results = model_data.get("context_length_results", {})
        layers_analyzed = model_data.get("layers_analyzed", [])
        final_layer_key = f"layer_{layers_analyzed[-1]}" if layers_analyzed else None

        phi_values = {}
        css_values = {}

        for ctx_len in ["10", "20", "30"]:
            if ctx_len in ctx_results:
                layers = ctx_results[ctx_len].get("layers", {})
                if final_layer_key and final_layer_key in layers:
                    cluster_sep = layers[final_layer_key].get("cluster_sep", {})
                    phi_values[ctx_len] = cluster_sep.get("phi_mean", 0)
                    css_values[ctx_len] = cluster_sep.get("max_sensitivity", 0)

        if all(k in phi_values for k in ["10", "20", "30"]):
            phi_spike = phi_values["20"] / phi_values["10"] if phi_values["10"] > 0 else 0
            css_spike = css_values["20"] / css_values["10"] if css_values["10"] > 0 else 0

            phase_transition_table.add_data(
                model_name,
                f"{phi_values['10']:.1f}",
                f"{phi_values['20']:.1f}",
                f"{phi_values['30']:.1f}",
                f"{css_values['10']:.1f}",
                f"{css_values['20']:.1f}",
                f"{css_values['30']:.1f}",
                f"{phi_spike:.1f}x",
                f"{css_spike:.1f}x",
            )

    wandb.log({"Phase Transition Discovery": phase_transition_table})

    # Log key findings as notes
    wandb.run.notes = """
# Hierarchical ICL Structural Influence Experiment

## Key Discovery: Phase Transitions at N≈20

Qwen models show dramatic phase transitions in representation geometry at context length 20:
- **Qwen3-14B**: 24x spike in structural metric (Φ), 33x spike in context sensitivity (CSS)
- **Qwen2.5-7B**: 8x spike in Φ, 11x spike in CSS

LLaMA and Mistral models show gradual structural emergence without sharp transitions.

## Metrics Explanation

### Structural Metric (Φ) - Dirichlet Energy
Measures the "smoothness" of representations on the hierarchical graph structure.
Higher values indicate more variation/structure emerging in the representation space.

### Context Sensitivity Score (CSS)
Measures how much the structural metric changes across different input contexts.
CSS(token_i, Φ) = -Cov_{contexts}(L(token_i), Φ)

High CSS indicates tokens that are "anchors" - positions where structure formation is sensitive to context.

## Conclusions
1. ICL involves phase transitions in representation geometry
2. Structural emergence is non-monotonic - spikes then stabilizes
3. Model architecture affects the sharpness of phase transitions
"""

    # Finish the run
    wandb.finish()

    print(f"\n{'='*70}")
    print("W&B LOGGING COMPLETE")
    print(f"{'='*70}")
    print(f"\nView your results at: {run.get_url()}")

    return run.get_url()


if __name__ == "__main__":
    main()
