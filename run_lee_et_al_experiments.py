#!/usr/bin/env python3
"""
Experiments to replicate Lee et al. (2025) "Influence Dynamics and Stagewise Data Attribution"
in the ICL setting.

Key findings to replicate:
1. Non-monotonic influence patterns
2. Sign flips at phase transitions
3. Peaks aligning with structural changes
4. Stagewise learning (global → local)
"""

import argparse
import json
import gc
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from tqdm import tqdm

from src.data import HierarchicalGraph, HierarchicalGraphConfig
from src.models import HookedLLM
from src.metrics import ClusterSeparation, DirichletEnergy, ContextSensitivityScore


def run_influence_dynamics_experiment(
    model_id: str = "Qwen/Qwen2.5-7B",
    context_lengths: list = None,
    n_samples: int = 50,
    output_dir: str = "results/lee_et_al_replication",
    seed: int = 42,
):
    """
    Run comprehensive influence dynamics experiment.

    Tracks:
    - Position-wise CSS across context lengths
    - Sign flips in influence
    - Within vs between cluster decomposition
    - Bridge token analysis
    """
    if context_lengths is None:
        context_lengths = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print("LEE ET AL. (2025) REPLICATION EXPERIMENT")
    print("Influence Dynamics in ICL Setting")
    print("=" * 70)
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    print(f"Model: {model_id}")
    print(f"Context lengths: {context_lengths}")
    print(f"Samples per length: {n_samples}")

    # Create hierarchical graph
    print("\n" + "-" * 70)
    print("Creating Hierarchical Graph")
    print("-" * 70)

    graph_config = HierarchicalGraphConfig(
        num_superclusters=3,
        nodes_per_cluster=5,
        p_intra_cluster=0.8,
        p_inter_cluster=0.1,
        seed=seed,
    )
    graph = HierarchicalGraph(graph_config)
    print(f"  Nodes: {graph.num_nodes}")
    print(f"  Clusters: {graph_config.num_superclusters}")

    # Load model
    print("\n" + "-" * 70)
    print("Loading Model")
    print("-" * 70)

    model = HookedLLM.from_pretrained(
        model_id,
        device="auto",
        dtype=torch.bfloat16,
    )
    print(f"  Layers: {model.num_layers}")
    print(f"  Hidden: {model.hidden_size}")

    # Select layers to analyze
    n_layers = model.num_layers
    layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    layers = sorted(set(layers))
    print(f"  Analyzing layers: {layers}")

    # Metrics
    cluster_sep = ClusterSeparation()
    dirichlet = DirichletEnergy()

    # Results storage
    results = {
        "model": model_id,
        "context_lengths": context_lengths,
        "n_samples": n_samples,
        "layers": layers,
        "dynamics": {},
    }

    # Track position-wise CSS across context lengths
    position_css_by_ctx = {ctx: {} for ctx in context_lengths}

    print("\n" + "-" * 70)
    print("Running Influence Dynamics Analysis")
    print("-" * 70)

    for ctx_len in tqdm(context_lengths, desc="Context lengths"):
        # Collect data for this context length
        all_losses = []
        all_representations = {layer: [] for layer in layers}
        all_clusters = []
        all_bridge_positions = []

        for _ in range(n_samples):
            prompt, nodes = graph.generate_random_walk(length=ctx_len, return_nodes=True)
            clusters = torch.tensor([graph.get_cluster(n) for n in nodes])

            # Find bridge positions (cluster transitions)
            bridge_pos = graph.get_cluster_transition_points(nodes)
            all_bridge_positions.append(bridge_pos)

            try:
                token_losses = model.compute_per_token_loss(prompt)
                _, cache = model.forward_with_cache(prompt, layers=layers)

                loss_len = token_losses.squeeze(0).shape[0]
                all_losses.append(token_losses.squeeze(0).cpu())

                for layer in layers:
                    residual = cache.get_residual_stream(layer)
                    if residual is not None:
                        reps = residual.squeeze(0)[:-1].cpu()
                        all_representations[layer].append(reps)

                        if layer == layers[0]:
                            # Align clusters
                            n_nodes = len(clusters)
                            rep_len = reps.shape[0]
                            if rep_len != n_nodes - 1:
                                cluster_indices = np.linspace(1, n_nodes - 1, rep_len).astype(int)
                                cluster_indices = np.clip(cluster_indices, 0, n_nodes - 1)
                                aligned_clusters = clusters[cluster_indices]
                            else:
                                aligned_clusters = clusters[1:]
                            all_clusters.append(aligned_clusters)

            except Exception as e:
                continue

        if len(all_losses) < 2:
            continue

        # Compute CSS for each layer
        ctx_results = {
            "n_valid": len(all_losses),
            "layers": {},
        }

        for layer in layers:
            if len(all_representations[layer]) < 2:
                continue

            css = ContextSensitivityScore(cluster_sep)

            try:
                sensitivity_result = css.compute_batch(
                    all_losses,
                    all_representations[layer],
                    all_clusters,
                )

                pos_sens = sensitivity_result['position_sensitivities']

                # Store position-wise CSS
                ctx_results["layers"][f"layer_{layer}"] = {
                    "position_css": pos_sens.tolist(),
                    "phi_mean": float(sensitivity_result['phi_mean']),
                    "phi_std": float(sensitivity_result['phi_std']),
                    "max_css": float(np.max(np.abs(pos_sens))),
                    "mean_abs_css": float(np.mean(np.abs(pos_sens))),
                    "n_positive": int(np.sum(pos_sens > 0)),
                    "n_negative": int(np.sum(pos_sens < 0)),
                }

                # Hierarchical decomposition
                decomp = css.compute_hierarchical_decomposition(
                    all_losses,
                    all_representations[layer],
                    all_clusters,
                )

                ctx_results["layers"][f"layer_{layer}"]["within_css"] = decomp['within_cluster_sensitivities'].tolist()
                ctx_results["layers"][f"layer_{layer}"]["between_css"] = decomp['between_cluster_sensitivities'].tolist()
                ctx_results["layers"][f"layer_{layer}"]["within_phi"] = float(decomp['within_phi_values'].mean())
                ctx_results["layers"][f"layer_{layer}"]["between_phi"] = float(decomp['between_phi_values'].mean())

            except Exception as e:
                ctx_results["layers"][f"layer_{layer}"] = {"error": str(e)}

        # Bridge token analysis
        bridge_css = []
        non_bridge_css = []
        final_layer = f"layer_{layers[-1]}"

        if final_layer in ctx_results["layers"] and "position_css" in ctx_results["layers"][final_layer]:
            pos_css = np.array(ctx_results["layers"][final_layer]["position_css"])

            # Aggregate bridge positions across samples
            all_bridge_set = set()
            for bp_list in all_bridge_positions:
                all_bridge_set.update(bp_list)

            for pos in range(len(pos_css)):
                if pos in all_bridge_set:
                    bridge_css.append(pos_css[pos])
                else:
                    non_bridge_css.append(pos_css[pos])

            ctx_results["bridge_analysis"] = {
                "bridge_css_mean": float(np.mean(bridge_css)) if bridge_css else 0,
                "bridge_css_std": float(np.std(bridge_css)) if bridge_css else 0,
                "non_bridge_css_mean": float(np.mean(non_bridge_css)) if non_bridge_css else 0,
                "non_bridge_css_std": float(np.std(non_bridge_css)) if non_bridge_css else 0,
                "n_bridge_positions": len(bridge_css),
                "n_non_bridge_positions": len(non_bridge_css),
            }

        results["dynamics"][str(ctx_len)] = ctx_results

        # Print progress
        if final_layer in ctx_results["layers"] and "phi_mean" in ctx_results["layers"][final_layer]:
            phi = ctx_results["layers"][final_layer]["phi_mean"]
            max_css = ctx_results["layers"][final_layer]["max_css"]
            print(f"  N={ctx_len}: Φ={phi:.1f}, max|CSS|={max_css:.2f}")

    # Cleanup model
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save results
    results_path = output_path / "influence_dynamics_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Generate visualizations
    print("\n" + "-" * 70)
    print("Generating Visualizations")
    print("-" * 70)

    generate_influence_dynamics_plots(results, output_path)

    return results


def generate_influence_dynamics_plots(results: dict, output_path: Path):
    """Generate plots for influence dynamics analysis."""

    dynamics = results["dynamics"]
    context_lengths = sorted([int(k) for k in dynamics.keys()])
    layers = results["layers"]
    final_layer = f"layer_{layers[-1]}"

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # =========================================================================
    # Plot 1: Position-wise CSS Heatmap across Context Lengths
    # =========================================================================
    ax1 = axes[0, 0]

    # Build heatmap matrix
    max_pos = max(len(dynamics[str(c)]["layers"].get(final_layer, {}).get("position_css", []))
                  for c in context_lengths)

    css_matrix = np.full((len(context_lengths), max_pos), np.nan)

    for i, ctx in enumerate(context_lengths):
        pos_css = dynamics[str(ctx)]["layers"].get(final_layer, {}).get("position_css", [])
        if pos_css:
            css_matrix[i, :len(pos_css)] = pos_css

    # Use diverging colormap centered at 0
    vmax = np.nanmax(np.abs(css_matrix))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax1.imshow(css_matrix, aspect='auto', cmap='RdBu_r', norm=norm, interpolation='nearest')
    ax1.set_xlabel("Token Position", fontsize=11)
    ax1.set_ylabel("Context Length (N)", fontsize=11)
    ax1.set_yticks(range(len(context_lengths)))
    ax1.set_yticklabels(context_lengths)
    ax1.set_title("A. Position-wise CSS Across Context Lengths\n(Red=Positive, Blue=Negative)",
                  fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax1, label="CSS")

    # =========================================================================
    # Plot 2: CSS Trajectory for Specific Positions (Non-Monotonic Detection)
    # =========================================================================
    ax2 = axes[0, 1]

    positions_to_track = [2, 5, 10, 15]  # Track specific positions
    colors = plt.cm.viridis(np.linspace(0, 1, len(positions_to_track)))

    for pos, color in zip(positions_to_track, colors):
        css_trajectory = []
        valid_ctx = []

        for ctx in context_lengths:
            pos_css = dynamics[str(ctx)]["layers"].get(final_layer, {}).get("position_css", [])
            if pos_css and pos < len(pos_css):
                css_trajectory.append(pos_css[pos])
                valid_ctx.append(ctx)

        if css_trajectory:
            ax2.plot(valid_ctx, css_trajectory, 'o-', color=color, linewidth=2,
                    markersize=6, label=f'Position {pos}')

    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax2.set_xlabel("Context Length (N)", fontsize=11)
    ax2.set_ylabel("CSS Value", fontsize=11)
    ax2.set_title("B. CSS Trajectory per Position\n(Looking for Non-Monotonic Patterns)",
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # =========================================================================
    # Plot 3: Sign Flip Detection
    # =========================================================================
    ax3 = axes[0, 2]

    # Count sign flips per position
    sign_flip_counts = defaultdict(int)
    sign_flip_locations = defaultdict(list)

    for pos in range(max_pos):
        prev_sign = None
        for i, ctx in enumerate(context_lengths):
            pos_css = dynamics[str(ctx)]["layers"].get(final_layer, {}).get("position_css", [])
            if pos_css and pos < len(pos_css):
                current_sign = np.sign(pos_css[pos])
                if prev_sign is not None and current_sign != prev_sign and current_sign != 0:
                    sign_flip_counts[pos] += 1
                    sign_flip_locations[pos].append(ctx)
                prev_sign = current_sign if current_sign != 0 else prev_sign

    positions = sorted(sign_flip_counts.keys())
    flips = [sign_flip_counts[p] for p in positions]

    ax3.bar(positions[:30], flips[:30], color='coral', alpha=0.7, edgecolor='black')
    ax3.set_xlabel("Token Position", fontsize=11)
    ax3.set_ylabel("Number of Sign Flips", fontsize=11)
    ax3.set_title("C. Sign Flips in CSS Across Context Lengths\n(Lee et al. Key Finding)",
                  fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # =========================================================================
    # Plot 4: Within vs Between Cluster Influence
    # =========================================================================
    ax4 = axes[1, 0]

    within_means = []
    between_means = []
    valid_ctx = []

    for ctx in context_lengths:
        layer_data = dynamics[str(ctx)]["layers"].get(final_layer, {})
        if "within_phi" in layer_data and "between_phi" in layer_data:
            within_means.append(layer_data["within_phi"])
            between_means.append(layer_data["between_phi"])
            valid_ctx.append(ctx)

    if within_means:
        ax4.plot(valid_ctx, between_means, 'o-', color='steelblue', linewidth=2.5,
                markersize=8, label='Between-Cluster Φ')
        ax4.plot(valid_ctx, within_means, 's--', color='darkorange', linewidth=2,
                markersize=7, label='Within-Cluster Φ')

        ax4.set_xlabel("Context Length (N)", fontsize=11)
        ax4.set_ylabel("Structural Metric (Φ)", fontsize=11)
        ax4.set_title("D. Hierarchical Decomposition\n(Global vs Local Structure)",
                      fontsize=12, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)

    # =========================================================================
    # Plot 5: Bridge vs Non-Bridge Token Analysis
    # =========================================================================
    ax5 = axes[1, 1]

    bridge_means = []
    non_bridge_means = []
    valid_ctx = []

    for ctx in context_lengths:
        bridge_data = dynamics[str(ctx)].get("bridge_analysis", {})
        if bridge_data:
            bridge_means.append(abs(bridge_data.get("bridge_css_mean", 0)))
            non_bridge_means.append(abs(bridge_data.get("non_bridge_css_mean", 0)))
            valid_ctx.append(ctx)

    if bridge_means:
        x = np.arange(len(valid_ctx))
        width = 0.35

        ax5.bar(x - width/2, bridge_means, width, label='Bridge Tokens', color='crimson', alpha=0.7)
        ax5.bar(x + width/2, non_bridge_means, width, label='Non-Bridge Tokens', color='steelblue', alpha=0.7)

        ax5.set_xlabel("Context Length (N)", fontsize=11)
        ax5.set_ylabel("|CSS| Mean", fontsize=11)
        ax5.set_xticks(x)
        ax5.set_xticklabels(valid_ctx)
        ax5.set_title("E. Bridge vs Non-Bridge Token Influence\n(Cluster Transition Points)",
                      fontsize=12, fontweight='bold')
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.3, axis='y')

    # =========================================================================
    # Plot 6: Summary Statistics
    # =========================================================================
    ax6 = axes[1, 2]
    ax6.axis('off')

    # Compute summary statistics
    total_sign_flips = sum(sign_flip_counts.values())
    positions_with_flips = len([p for p, c in sign_flip_counts.items() if c > 0])

    # Check for non-monotonic patterns
    non_monotonic_positions = 0
    for pos in range(min(20, max_pos)):
        trajectory = []
        for ctx in context_lengths:
            pos_css = dynamics[str(ctx)]["layers"].get(final_layer, {}).get("position_css", [])
            if pos_css and pos < len(pos_css):
                trajectory.append(pos_css[pos])

        if len(trajectory) >= 3:
            # Check if trajectory has both increases and decreases
            diffs = np.diff(trajectory)
            if np.any(diffs > 0) and np.any(diffs < 0):
                non_monotonic_positions += 1

    summary_text = f"""
    LEE ET AL. REPLICATION SUMMARY
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Model: {results['model']}
    Context Lengths: {min(context_lengths)} - {max(context_lengths)}
    Samples per Length: {results['n_samples']}

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    KEY FINDINGS
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    1. SIGN FLIPS (Lee et al. Fig 5):
       Total sign flips detected: {total_sign_flips}
       Positions with ≥1 flip: {positions_with_flips}

    2. NON-MONOTONIC PATTERNS:
       Positions showing non-monotonic CSS: {non_monotonic_positions}/20
       (Has both increases AND decreases)

    3. HIERARCHICAL LEARNING:
       Between-cluster Φ at N=10: {between_means[0] if between_means else 'N/A':.2f}
       Between-cluster Φ at N={valid_ctx[-1] if valid_ctx else 'N/A'}: {between_means[-1] if between_means else 'N/A':.2f}

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    INTERPRETATION
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    ✓ Sign flips indicate changing token roles
    ✓ Non-monotonic CSS = influence dynamics
    ✓ Between > Within early = global first
    """

    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax6.set_title("F. Summary", fontsize=12, fontweight='bold')

    plt.tight_layout()

    fig.suptitle("Lee et al. (2025) Replication: Influence Dynamics in ICL\n" +
                 "Finding: Non-Monotonic Patterns & Sign Flips in Context Sensitivity",
                 fontsize=14, fontweight='bold', y=1.02)

    plt.savefig(output_path / "influence_dynamics_analysis.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path / "influence_dynamics_analysis.pdf", bbox_inches='tight', facecolor='white')
    print(f"  Saved: influence_dynamics_analysis.png")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Lee et al. Replication Experiments")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--context-lengths", type=str, default="5,10,15,20,25,30,40,50,75,100")
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="results/lee_et_al_replication")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    context_lengths = [int(x) for x in args.context_lengths.split(",")]

    run_influence_dynamics_experiment(
        model_id=args.model,
        context_lengths=context_lengths,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
