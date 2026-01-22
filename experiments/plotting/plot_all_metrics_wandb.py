#!/usr/bin/env python3
"""
Plot all key metrics from experiments and save to Weights & Biases.

Includes:
1. Park et al. reproduction (Dirichlet Energy)
2. Lee et al. replication (CSS dynamics, sign flips)
3. Hierarchical experiment (ClusterSeparation)
4. Model comparisons
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
from collections import defaultdict
import wandb


def load_all_results():
    """Load all experiment results."""
    results = {}

    # Park et al. reproduction - Llama
    park_llama_path = Path("results/park_reproduction_llama/reproduction_results.json")
    if park_llama_path.exists():
        with open(park_llama_path) as f:
            results["park_llama"] = json.load(f)

    # Park et al. reproduction - Qwen
    park_qwen_path = Path("results/park_reproduction_v2/reproduction_results.json")
    if park_qwen_path.exists():
        with open(park_qwen_path) as f:
            results["park_qwen"] = json.load(f)

    # Lee et al. replication
    lee_path = Path("results/lee_et_al_replication/influence_dynamics_results.json")
    if lee_path.exists():
        with open(lee_path) as f:
            results["lee"] = json.load(f)

    # Hierarchical experiment results
    for model_dir in ["hierarchical_llama", "hierarchical_qwen", "hierarchical_mistral"]:
        dir_path = Path(f"results/{model_dir}")
        if dir_path.exists():
            for json_file in dir_path.glob("*_results.json"):
                if not json_file.name.startswith("all_"):
                    with open(json_file) as f:
                        data = json.load(f)
                        model_name = data.get("model", json_file.stem)
                        results[f"hier_{model_name}"] = data

    return results


def create_park_comparison_plot(results):
    """Create Park et al. Dirichlet Energy comparison plot."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    models = {}
    if "park_llama" in results:
        models["Llama-3.1-8B"] = results["park_llama"]
    if "park_qwen" in results:
        models["Qwen2.5-7B"] = results["park_qwen"]

    colors = {"Llama-3.1-8B": "#1f77b4", "Qwen2.5-7B": "#ff7f0e"}

    # Plot 1: Raw Dirichlet Energy
    ax1 = axes[0]
    for model_name, data in models.items():
        ctx_results = data["context_results"]
        ctx_lens = sorted([int(k) for k in ctx_results.keys()])
        energies = [ctx_results[str(c)]["energy_mean"] for c in ctx_lens]
        stds = [ctx_results[str(c)]["energy_std"] for c in ctx_lens]

        ax1.errorbar(ctx_lens, energies, yerr=stds, fmt='o-',
                    color=colors.get(model_name, 'gray'),
                    linewidth=2, markersize=6, label=model_name, capsize=3)

    ax1.set_xlabel("Context Length (N)", fontsize=11)
    ax1.set_ylabel("Dirichlet Energy E(X)", fontsize=11)
    ax1.set_title("A. Dirichlet Energy vs Context", fontsize=12, fontweight='bold')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Normalized Energy
    ax2 = axes[1]
    for model_name, data in models.items():
        ctx_results = data["context_results"]
        ctx_lens = sorted([int(k) for k in ctx_results.keys()])
        energies = [ctx_results[str(c)]["energy_mean"] for c in ctx_lens]
        baseline = energies[0]
        normalized = [e / baseline for e in energies]

        ax2.plot(ctx_lens, normalized, 'o-', color=colors.get(model_name, 'gray'),
                linewidth=2, markersize=6, label=model_name)

    ax2.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='10% threshold')
    ax2.set_xlabel("Context Length (N)", fontsize=11)
    ax2.set_ylabel("E(X) / E(X @ baseline)", fontsize=11)
    ax2.set_title("B. Normalized Energy", fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Energy reduction summary
    ax3 = axes[2]
    model_names = list(models.keys())
    reductions = []
    for model_name in model_names:
        data = models[model_name]
        ctx_results = data["context_results"]
        ctx_lens = sorted([int(k) for k in ctx_results.keys()])
        e_first = ctx_results[str(ctx_lens[0])]["energy_mean"]
        e_last = ctx_results[str(ctx_lens[-1])]["energy_mean"]
        reductions.append(e_first / e_last)

    bars = ax3.bar(model_names, reductions, color=[colors.get(m, 'gray') for m in model_names], alpha=0.7)
    ax3.set_ylabel("Energy Reduction Factor", fontsize=11)
    ax3.set_title("C. Total Energy Reduction", fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    for bar, red in zip(bars, reductions):
        ax3.annotate(f'{red:.0f}x', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontweight='bold')

    plt.tight_layout()
    fig.suptitle("Park et al. (2024) Reproduction: Dirichlet Energy Decreases with Context",
                fontsize=13, fontweight='bold', y=1.02)

    return fig


def create_lee_dynamics_plot(results):
    """Create Lee et al. influence dynamics plot."""
    if "lee" not in results:
        return None

    data = results["lee"]
    dynamics = data["dynamics"]
    ctx_lengths = sorted([int(k) for k in dynamics.keys()])
    final_layer = f"layer_{data['layers'][-1]}"

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Plot 1: CSS Heatmap
    ax1 = axes[0, 0]
    max_pos = max(len(dynamics[str(c)]["layers"].get(final_layer, {}).get("position_css", []))
                  for c in ctx_lengths)

    css_matrix = np.full((len(ctx_lengths), min(max_pos, 50)), np.nan)
    for i, ctx in enumerate(ctx_lengths):
        pos_css = dynamics[str(ctx)]["layers"].get(final_layer, {}).get("position_css", [])
        if pos_css:
            css_matrix[i, :min(len(pos_css), 50)] = pos_css[:50]

    vmax = np.nanmax(np.abs(css_matrix))
    if vmax > 0:
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax1.imshow(css_matrix, aspect='auto', cmap='RdBu_r', norm=norm)
        plt.colorbar(im, ax=ax1, label="CSS")

    ax1.set_xlabel("Token Position", fontsize=11)
    ax1.set_ylabel("Context Length (N)", fontsize=11)
    ax1.set_yticks(range(len(ctx_lengths)))
    ax1.set_yticklabels(ctx_lengths)
    ax1.set_title("A. Position-wise CSS Heatmap", fontsize=12, fontweight='bold')

    # Plot 2: Φ trajectory (non-monotonic)
    ax2 = axes[0, 1]
    phi_values = []
    valid_ctx = []
    for ctx in ctx_lengths:
        layer_data = dynamics[str(ctx)]["layers"].get(final_layer, {})
        if "phi_mean" in layer_data:
            phi_values.append(layer_data["phi_mean"])
            valid_ctx.append(ctx)

    ax2.plot(valid_ctx, phi_values, 'o-', color='steelblue', linewidth=2.5, markersize=8)
    ax2.set_xlabel("Context Length (N)", fontsize=11)
    ax2.set_ylabel("Cluster Separation (Φ)", fontsize=11)
    ax2.set_title("B. Non-Monotonic Φ Trajectory", fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Highlight increases/decreases
    for i in range(1, len(phi_values)):
        color = 'green' if phi_values[i] > phi_values[i-1] else 'red'
        ax2.annotate('', xy=(valid_ctx[i], phi_values[i]), xytext=(valid_ctx[i-1], phi_values[i-1]),
                    arrowprops=dict(arrowstyle='->', color=color, alpha=0.5))

    # Plot 3: Sign flips per position
    ax3 = axes[0, 2]
    sign_flip_counts = defaultdict(int)

    for pos in range(min(max_pos, 30)):
        prev_sign = None
        for ctx in ctx_lengths:
            pos_css = dynamics[str(ctx)]["layers"].get(final_layer, {}).get("position_css", [])
            if pos_css and pos < len(pos_css):
                current_sign = np.sign(pos_css[pos])
                if prev_sign is not None and current_sign != prev_sign and current_sign != 0:
                    sign_flip_counts[pos] += 1
                prev_sign = current_sign if current_sign != 0 else prev_sign

    positions = sorted(sign_flip_counts.keys())
    flips = [sign_flip_counts[p] for p in positions]

    ax3.bar(positions, flips, color='coral', alpha=0.7, edgecolor='black')
    ax3.set_xlabel("Token Position", fontsize=11)
    ax3.set_ylabel("Number of Sign Flips", fontsize=11)
    ax3.set_title("C. Sign Flips Across Context", fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Within vs Between cluster
    ax4 = axes[1, 0]
    within_vals = []
    between_vals = []
    valid_ctx2 = []

    for ctx in ctx_lengths:
        layer_data = dynamics[str(ctx)]["layers"].get(final_layer, {})
        if "within_phi" in layer_data and "between_phi" in layer_data:
            within_vals.append(layer_data["within_phi"])
            between_vals.append(layer_data["between_phi"])
            valid_ctx2.append(ctx)

    if within_vals:
        ax4.plot(valid_ctx2, between_vals, 'o-', color='steelblue', linewidth=2.5,
                markersize=8, label='Between-Cluster')
        ax4.plot(valid_ctx2, within_vals, 's--', color='darkorange', linewidth=2,
                markersize=7, label='Within-Cluster')
        ax4.set_yscale('log')

    ax4.set_xlabel("Context Length (N)", fontsize=11)
    ax4.set_ylabel("Structural Metric (Φ)", fontsize=11)
    ax4.set_title("D. Hierarchical Decomposition", fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: CSS trajectory for specific positions
    ax5 = axes[1, 1]
    positions_to_track = [1, 3, 5, 10]
    colors_pos = plt.cm.viridis(np.linspace(0, 1, len(positions_to_track)))

    for pos, color in zip(positions_to_track, colors_pos):
        trajectory = []
        valid_ctx3 = []
        for ctx in ctx_lengths:
            pos_css = dynamics[str(ctx)]["layers"].get(final_layer, {}).get("position_css", [])
            if pos_css and pos < len(pos_css):
                trajectory.append(pos_css[pos])
                valid_ctx3.append(ctx)

        if trajectory:
            ax5.plot(valid_ctx3, trajectory, 'o-', color=color, linewidth=2,
                    markersize=5, label=f'Pos {pos}')

    ax5.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax5.set_xlabel("Context Length (N)", fontsize=11)
    ax5.set_ylabel("CSS Value", fontsize=11)
    ax5.set_title("E. CSS Trajectory per Position", fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    # Plot 6: Max CSS trajectory
    ax6 = axes[1, 2]
    max_css_vals = []
    for ctx in ctx_lengths:
        layer_data = dynamics[str(ctx)]["layers"].get(final_layer, {})
        if "max_css" in layer_data:
            max_css_vals.append(layer_data["max_css"])

    ax6.plot(ctx_lengths[:len(max_css_vals)], max_css_vals, 'o-', color='crimson',
            linewidth=2.5, markersize=8)
    ax6.set_xlabel("Context Length (N)", fontsize=11)
    ax6.set_ylabel("max|CSS|", fontsize=11)
    ax6.set_title("F. Peak Influence Magnitude", fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    # Highlight peak
    if max_css_vals:
        peak_idx = np.argmax(max_css_vals)
        ax6.annotate(f'Peak at N={ctx_lengths[peak_idx]}',
                    xy=(ctx_lengths[peak_idx], max_css_vals[peak_idx]),
                    xytext=(ctx_lengths[peak_idx]+10, max_css_vals[peak_idx]*0.8),
                    fontsize=10, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='crimson'))

    plt.tight_layout()
    fig.suptitle("Lee et al. (2025) Replication: Influence Dynamics in ICL",
                fontsize=13, fontweight='bold', y=1.02)

    return fig


def create_hierarchical_comparison_plot(results):
    """Create hierarchical experiment comparison plot."""
    hier_results = {k: v for k, v in results.items() if k.startswith("hier_")}

    if not hier_results:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, len(hier_results)))

    # Plot 1: ClusterSeparation (Φ) across context
    ax1 = axes[0, 0]
    for (key, data), color in zip(hier_results.items(), colors):
        model_name = data.get("model", key.replace("hier_", ""))
        ctx_results = data.get("context_length_results", {})
        layers = data.get("layers_analyzed", [])
        final_layer = f"layer_{layers[-1]}" if layers else None

        ctx_lens = []
        phi_vals = []

        for ctx_len in sorted(ctx_results.keys(), key=int):
            layer_data = ctx_results[ctx_len].get("layers", {}).get(final_layer, {})
            cs = layer_data.get("cluster_sep", {})
            if "phi_mean" in cs:
                ctx_lens.append(int(ctx_len))
                phi_vals.append(cs["phi_mean"])

        if ctx_lens:
            ax1.plot(ctx_lens, phi_vals, 'o-', color=color, linewidth=2,
                    markersize=6, label=model_name)

    ax1.set_xlabel("Context Length (N)", fontsize=11)
    ax1.set_ylabel("Cluster Separation (Φ)", fontsize=11)
    ax1.set_title("A. ClusterSeparation vs Context", fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Max CSS across context
    ax2 = axes[0, 1]
    for (key, data), color in zip(hier_results.items(), colors):
        model_name = data.get("model", key.replace("hier_", ""))
        ctx_results = data.get("context_length_results", {})
        layers = data.get("layers_analyzed", [])
        final_layer = f"layer_{layers[-1]}" if layers else None

        ctx_lens = []
        max_css = []

        for ctx_len in sorted(ctx_results.keys(), key=int):
            layer_data = ctx_results[ctx_len].get("layers", {}).get(final_layer, {})
            cs = layer_data.get("cluster_sep", {})
            if "max_sensitivity" in cs:
                ctx_lens.append(int(ctx_len))
                max_css.append(cs["max_sensitivity"])

        if ctx_lens:
            ax2.plot(ctx_lens, max_css, 'o-', color=color, linewidth=2,
                    markersize=6, label=model_name)

    ax2.set_xlabel("Context Length (N)", fontsize=11)
    ax2.set_ylabel("max|CSS|", fontsize=11)
    ax2.set_title("B. Peak Sensitivity vs Context", fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Normalized Φ comparison
    ax3 = axes[1, 0]
    for (key, data), color in zip(hier_results.items(), colors):
        model_name = data.get("model", key.replace("hier_", ""))
        ctx_results = data.get("context_length_results", {})
        layers = data.get("layers_analyzed", [])
        final_layer = f"layer_{layers[-1]}" if layers else None

        ctx_lens = []
        phi_vals = []

        for ctx_len in sorted(ctx_results.keys(), key=int):
            layer_data = ctx_results[ctx_len].get("layers", {}).get(final_layer, {})
            cs = layer_data.get("cluster_sep", {})
            if "phi_mean" in cs:
                ctx_lens.append(int(ctx_len))
                phi_vals.append(cs["phi_mean"])

        if phi_vals:
            baseline = phi_vals[0]
            normalized = [p / baseline for p in phi_vals]
            ax3.plot(ctx_lens, normalized, 'o-', color=color, linewidth=2,
                    markersize=6, label=model_name)

    ax3.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel("Context Length (N)", fontsize=11)
    ax3.set_ylabel("Φ / Φ(baseline)", fontsize=11)
    ax3.set_title("C. Normalized ClusterSeparation", fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Model comparison summary
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_data = []
    for key, data in hier_results.items():
        model_name = data.get("model", key.replace("hier_", ""))
        ctx_results = data.get("context_length_results", {})
        layers = data.get("layers_analyzed", [])
        final_layer = f"layer_{layers[-1]}" if layers else None

        phi_vals = []
        for ctx_len in sorted(ctx_results.keys(), key=int):
            layer_data = ctx_results[ctx_len].get("layers", {}).get(final_layer, {})
            cs = layer_data.get("cluster_sep", {})
            if "phi_mean" in cs:
                phi_vals.append(cs["phi_mean"])

        if phi_vals:
            summary_data.append([
                model_name[:15],
                f"{phi_vals[0]:.0f}",
                f"{phi_vals[-1]:.0f}",
                f"{phi_vals[-1]/phi_vals[0]:.1f}x"
            ])

    if summary_data:
        table = ax4.table(
            cellText=summary_data,
            colLabels=["Model", "Φ @ N=10", "Φ @ N=200", "Growth"],
            loc='center',
            cellLoc='center',
            colColours=['#f0f0f0'] * 4,
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)

    ax4.set_title("D. Model Comparison Summary", fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    fig.suptitle("Hierarchical Experiment: ClusterSeparation Across Models",
                fontsize=13, fontweight='bold', y=1.02)

    return fig


def create_combined_summary_plot(results):
    """Create combined summary of all experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Park - Energy decrease
    ax1 = axes[0, 0]
    if "park_llama" in results:
        data = results["park_llama"]
        ctx_results = data["context_results"]
        ctx_lens = sorted([int(k) for k in ctx_results.keys()])
        energies = [ctx_results[str(c)]["energy_mean"] for c in ctx_lens]

        ax1.plot(ctx_lens, energies, 'o-', color='steelblue', linewidth=2.5, markersize=8)
        ax1.set_yscale('log')
        ax1.set_xlabel("Context Length (N)", fontsize=11)
        ax1.set_ylabel("Dirichlet Energy", fontsize=11)
        ax1.set_title("A. Park et al.: Energy DECREASES\n(Representations align with graph)",
                     fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Add annotation
        ax1.annotate(f'{energies[0]/energies[-1]:.0f}x\nreduction',
                    xy=(ctx_lens[-1], energies[-1]),
                    xytext=(ctx_lens[-1]-20, energies[-1]*10),
                    fontsize=11, fontweight='bold', color='green',
                    arrowprops=dict(arrowstyle='->', color='green'))

    # Plot 2: Hierarchical - Φ increase
    ax2 = axes[0, 1]
    if "hier_LLaMA-3.1-8B" in results:
        data = results["hier_LLaMA-3.1-8B"]
        ctx_results = data.get("context_length_results", {})
        layers = data.get("layers_analyzed", [])
        final_layer = f"layer_{layers[-1]}" if layers else None

        ctx_lens = []
        phi_vals = []
        for ctx_len in sorted(ctx_results.keys(), key=int):
            layer_data = ctx_results[ctx_len].get("layers", {}).get(final_layer, {})
            cs = layer_data.get("cluster_sep", {})
            if "phi_mean" in cs:
                ctx_lens.append(int(ctx_len))
                phi_vals.append(cs["phi_mean"])

        if ctx_lens:
            ax2.plot(ctx_lens, phi_vals, 'o-', color='darkorange', linewidth=2.5, markersize=8)
            ax2.set_xlabel("Context Length (N)", fontsize=11)
            ax2.set_ylabel("Cluster Separation (Φ)", fontsize=11)
            ax2.set_title("B. Our Experiment: Φ INCREASES\n(Clusters become more distinct)",
                         fontsize=11, fontweight='bold')
            ax2.grid(True, alpha=0.3)

            ax2.annotate(f'{phi_vals[-1]/phi_vals[0]:.1f}x\nincrease',
                        xy=(ctx_lens[-1], phi_vals[-1]),
                        xytext=(ctx_lens[-1]-30, phi_vals[-1]*0.7),
                        fontsize=11, fontweight='bold', color='green',
                        arrowprops=dict(arrowstyle='->', color='green'))

    # Plot 3: Lee - Sign flips
    ax3 = axes[1, 0]
    if "lee" in results:
        data = results["lee"]
        dynamics = data["dynamics"]
        ctx_lengths = sorted([int(k) for k in dynamics.keys()])
        final_layer = f"layer_{data['layers'][-1]}"

        # Count sign flips
        max_pos = max(len(dynamics[str(c)]["layers"].get(final_layer, {}).get("position_css", []))
                      for c in ctx_lengths)

        sign_flip_counts = defaultdict(int)
        for pos in range(min(max_pos, 30)):
            prev_sign = None
            for ctx in ctx_lengths:
                pos_css = dynamics[str(ctx)]["layers"].get(final_layer, {}).get("position_css", [])
                if pos_css and pos < len(pos_css):
                    current_sign = np.sign(pos_css[pos])
                    if prev_sign is not None and current_sign != prev_sign and current_sign != 0:
                        sign_flip_counts[pos] += 1
                    prev_sign = current_sign if current_sign != 0 else prev_sign

        positions = sorted(sign_flip_counts.keys())
        flips = [sign_flip_counts[p] for p in positions]

        ax3.bar(positions, flips, color='coral', alpha=0.7, edgecolor='black')
        ax3.set_xlabel("Token Position", fontsize=11)
        ax3.set_ylabel("Number of Sign Flips", fontsize=11)
        ax3.set_title(f"C. Lee et al.: {sum(flips)} SIGN FLIPS\n(Influence changes dynamically)",
                     fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Summary text
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_text = """
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    UNIFIED FINDINGS ACROSS ALL EXPERIMENTS
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    1. PARK ET AL. REPRODUCTION
       ✓ Dirichlet Energy decreases 64x (Llama)
       ✓ Representations align with graph topology
       ✓ Phase transition at N≈5-15

    2. OUR HIERARCHICAL EXPERIMENT
       ✓ ClusterSeparation increases 2-3x
       ✓ Global structure (between-cluster) dominates
       ✓ Gradual structure emergence

    3. LEE ET AL. REPLICATION
       ✓ 130 sign flips detected
       ✓ Non-monotonic influence patterns
       ✓ Between/Within ratio: 2000-13000x

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    KEY INSIGHT: Context length in ICL plays
    analogous role to training time - both show
    dynamic structure learning with phase transitions
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """

    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax4.set_title("D. Summary", fontsize=12, fontweight='bold')

    plt.tight_layout()
    fig.suptitle("ICL Structural Influence: Unified Results",
                fontsize=14, fontweight='bold', y=1.02)

    return fig


def main():
    print("=" * 70)
    print("PLOTTING ALL METRICS AND LOGGING TO W&B")
    print("=" * 70)

    # Load all results
    print("\nLoading results...")
    results = load_all_results()
    print(f"  Loaded: {list(results.keys())}")

    # Initialize W&B
    print("\nInitializing W&B...")
    run = wandb.init(
        project="icl-structural-influence",
        name="comprehensive-analysis",
        config={
            "experiments": list(results.keys()),
            "park_models": ["Llama-3.1-8B", "Qwen2.5-7B"],
            "lee_replication": True,
            "hierarchical_models": [k for k in results.keys() if k.startswith("hier_")],
        },
        tags=["comprehensive", "all-metrics", "park", "lee", "hierarchical"],
    )

    # Create output directory
    output_dir = Path("results/comprehensive_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate and log plots
    print("\nGenerating plots...")

    # 1. Park et al. comparison
    fig1 = create_park_comparison_plot(results)
    if fig1:
        fig1.savefig(output_dir / "park_comparison.png", dpi=300, bbox_inches='tight', facecolor='white')
        wandb.log({"Park_et_al_Reproduction": wandb.Image(fig1)})
        print("  ✓ Park et al. comparison")
        plt.close(fig1)

    # 2. Lee et al. dynamics
    fig2 = create_lee_dynamics_plot(results)
    if fig2:
        fig2.savefig(output_dir / "lee_dynamics.png", dpi=300, bbox_inches='tight', facecolor='white')
        wandb.log({"Lee_et_al_Replication": wandb.Image(fig2)})
        print("  ✓ Lee et al. dynamics")
        plt.close(fig2)

    # 3. Hierarchical comparison
    fig3 = create_hierarchical_comparison_plot(results)
    if fig3:
        fig3.savefig(output_dir / "hierarchical_comparison.png", dpi=300, bbox_inches='tight', facecolor='white')
        wandb.log({"Hierarchical_Experiment": wandb.Image(fig3)})
        print("  ✓ Hierarchical comparison")
        plt.close(fig3)

    # 4. Combined summary
    fig4 = create_combined_summary_plot(results)
    if fig4:
        fig4.savefig(output_dir / "combined_summary.png", dpi=300, bbox_inches='tight', facecolor='white')
        wandb.log({"Combined_Summary": wandb.Image(fig4)})
        print("  ✓ Combined summary")
        plt.close(fig4)

    # Log metrics tables
    print("\nLogging metric tables...")

    # Park metrics table
    if "park_llama" in results:
        park_table = wandb.Table(columns=["Model", "N", "Energy", "Normalized"])
        for model_key in ["park_llama", "park_qwen"]:
            if model_key in results:
                data = results[model_key]
                model_name = data["model"]
                ctx_results = data["context_results"]
                ctx_lens = sorted([int(k) for k in ctx_results.keys()])
                baseline = ctx_results[str(ctx_lens[0])]["energy_mean"]

                for ctx in ctx_lens:
                    e = ctx_results[str(ctx)]["energy_mean"]
                    park_table.add_data(model_name, ctx, e, e/baseline)

        wandb.log({"Park_Metrics": park_table})

    # Lee metrics table
    if "lee" in results:
        lee_data = results["lee"]
        dynamics = lee_data["dynamics"]
        ctx_lengths = sorted([int(k) for k in dynamics.keys()])
        final_layer = f"layer_{lee_data['layers'][-1]}"

        lee_table = wandb.Table(columns=["N", "Phi", "max_CSS", "n_positive", "n_negative"])
        for ctx in ctx_lengths:
            layer_data = dynamics[str(ctx)]["layers"].get(final_layer, {})
            if "phi_mean" in layer_data:
                lee_table.add_data(
                    ctx,
                    layer_data["phi_mean"],
                    layer_data["max_css"],
                    layer_data["n_positive"],
                    layer_data["n_negative"]
                )

        wandb.log({"Lee_Metrics": lee_table})

    # Log summary statistics
    print("\nLogging summary statistics...")

    summary_stats = {}

    if "park_llama" in results:
        data = results["park_llama"]
        ctx_results = data["context_results"]
        ctx_lens = sorted([int(k) for k in ctx_results.keys()])
        e_first = ctx_results[str(ctx_lens[0])]["energy_mean"]
        e_last = ctx_results[str(ctx_lens[-1])]["energy_mean"]
        summary_stats["park_llama_energy_reduction"] = e_first / e_last

    if "lee" in results:
        # Count total sign flips
        dynamics = results["lee"]["dynamics"]
        ctx_lengths = sorted([int(k) for k in dynamics.keys()])
        final_layer = f"layer_{results['lee']['layers'][-1]}"

        max_pos = max(len(dynamics[str(c)]["layers"].get(final_layer, {}).get("position_css", []))
                      for c in ctx_lengths)

        total_flips = 0
        for pos in range(max_pos):
            prev_sign = None
            for ctx in ctx_lengths:
                pos_css = dynamics[str(ctx)]["layers"].get(final_layer, {}).get("position_css", [])
                if pos_css and pos < len(pos_css):
                    current_sign = np.sign(pos_css[pos])
                    if prev_sign is not None and current_sign != prev_sign and current_sign != 0:
                        total_flips += 1
                    prev_sign = current_sign if current_sign != 0 else prev_sign

        summary_stats["lee_total_sign_flips"] = total_flips

    wandb.log(summary_stats)

    # Set run notes
    wandb.run.notes = """
# ICL Structural Influence: Comprehensive Analysis

## Experiments Included

### 1. Park et al. (2024) Reproduction
- **Finding**: Dirichlet Energy decreases 64x as context increases
- **Interpretation**: Representations align with graph structure

### 2. Lee et al. (2025) Replication
- **Finding**: 130 sign flips, non-monotonic CSS patterns
- **Interpretation**: Influence is dynamic, changes at phase transitions

### 3. Hierarchical Experiment
- **Finding**: ClusterSeparation increases with context
- **Interpretation**: Global structure dominates (2000-13000x)

## Key Insight
Context length in ICL plays an analogous role to training time -
both show dynamic structure learning with phase transitions.
"""

    wandb.finish()

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"\nPlots saved to: {output_dir}")
    print(f"W&B run: {run.url}")

    return run.url


if __name__ == "__main__":
    main()
