#!/usr/bin/env python3
"""
Create a clear visualization of the phase transition at N=20.
Shows raw (non-normalized) values to highlight the dramatic spike.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_model_data(results_base):
    """Load all model results."""
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
    return all_results


def extract_phi_and_css(model_data):
    """Extract Φ (phi_mean) and CSS (max_sensitivity) from layer results."""
    ctx_results = model_data.get("context_length_results", {})
    layers_analyzed = model_data.get("layers_analyzed", [])
    final_layer_key = f"layer_{layers_analyzed[-1]}" if layers_analyzed else None

    context_lengths = []
    phi_values = []
    css_values = []

    for ctx_len, ctx_data in sorted(ctx_results.items(), key=lambda x: int(x[0])):
        layers = ctx_data.get("layers", {})
        if final_layer_key and final_layer_key in layers:
            cluster_sep = layers[final_layer_key].get("cluster_sep", {})
            if "phi_mean" in cluster_sep:
                context_lengths.append(int(ctx_len))
                phi_values.append(cluster_sep["phi_mean"])
                css_values.append(cluster_sep.get("max_sensitivity", 0))

    return np.array(context_lengths), np.array(phi_values), np.array(css_values)


def main():
    results_base = Path("/workspace/research/projects/in_context_representation_influence/results")
    output_dir = results_base / "combined_analysis"

    all_results = load_model_data(results_base)
    print(f"Loaded {len(all_results)} models: {list(all_results.keys())}")

    # Color scheme
    colors = {
        'Qwen3-14B': '#9B2335',
        'Qwen2.5-7B': '#E63946',
        'LLaMA-3.1-8B': '#457B9D',
        'Mistral-Nemo-12B': '#F4A261',
    }

    markers = {
        'Qwen3-14B': 'p',
        'Qwen2.5-7B': 'o',
        'LLaMA-3.1-8B': 's',
        'Mistral-Nemo-12B': 'D',
    }

    # Create the figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # =========================================================================
    # Plot 1: Raw Φ (Cluster Separation) - Shows the spike clearly
    # =========================================================================
    ax1 = axes[0, 0]
    for model_name, model_data in all_results.items():
        ctx_lens, phi_vals, css_vals = extract_phi_and_css(model_data)
        color = colors.get(model_name, '#333333')
        marker = markers.get(model_name, 'o')

        ax1.plot(ctx_lens, phi_vals, marker=marker, color=color,
                 linewidth=2.5, markersize=8, label=model_name, alpha=0.9)

    ax1.set_xlabel("Context Length (N)", fontsize=12)
    ax1.set_ylabel("Cluster Separation Φ (raw)", fontsize=12)
    ax1.set_title("A. Structural Metric (Φ) vs Context Length", fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Highlight N=20
    ax1.axvline(x=20, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax1.annotate('N=20\nPhase Transition', xy=(20, ax1.get_ylim()[1]),
                 xytext=(35, ax1.get_ylim()[1]*0.5),
                 fontsize=10, color='red', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))

    # =========================================================================
    # Plot 2: Raw CSS (Max Sensitivity) - Shows the spike clearly
    # =========================================================================
    ax2 = axes[0, 1]
    for model_name, model_data in all_results.items():
        ctx_lens, phi_vals, css_vals = extract_phi_and_css(model_data)
        color = colors.get(model_name, '#333333')
        marker = markers.get(model_name, 'o')

        ax2.plot(ctx_lens, css_vals, marker=marker, color=color,
                 linewidth=2.5, markersize=8, label=model_name, alpha=0.9)

    ax2.set_xlabel("Context Length (N)", fontsize=12)
    ax2.set_ylabel("Max Context Sensitivity (CSS)", fontsize=12)
    ax2.set_title("B. Context Sensitivity Score vs Context Length", fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    # Highlight N=20
    ax2.axvline(x=20, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax2.annotate('N=20\nPhase Transition', xy=(20, ax2.get_ylim()[1]),
                 xytext=(35, ax2.get_ylim()[1]*0.5),
                 fontsize=10, color='red', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))

    # =========================================================================
    # Plot 3: Spike Factor (ratio to N=10 baseline)
    # =========================================================================
    ax3 = axes[1, 0]

    bar_width = 0.2
    models = list(all_results.keys())
    x = np.arange(len(models))

    phi_spikes = []
    css_spikes = []

    for model_name in models:
        ctx_lens, phi_vals, css_vals = extract_phi_and_css(all_results[model_name])

        # Get values at N=10 and N=20
        idx_10 = np.where(ctx_lens == 10)[0]
        idx_20 = np.where(ctx_lens == 20)[0]

        if len(idx_10) > 0 and len(idx_20) > 0:
            phi_10, phi_20 = phi_vals[idx_10[0]], phi_vals[idx_20[0]]
            css_10, css_20 = css_vals[idx_10[0]], css_vals[idx_20[0]]

            phi_spikes.append(phi_20 / phi_10 if phi_10 > 0 else 0)
            css_spikes.append(css_20 / css_10 if css_10 > 0 else 0)
        else:
            phi_spikes.append(0)
            css_spikes.append(0)

    bars1 = ax3.bar(x - bar_width/2, phi_spikes, bar_width, label='Φ Spike (N=20/N=10)',
                    color=[colors.get(m, '#333') for m in models], alpha=0.7)
    bars2 = ax3.bar(x + bar_width/2, css_spikes, bar_width, label='CSS Spike (N=20/N=10)',
                    color=[colors.get(m, '#333') for m in models], alpha=0.4, hatch='//')

    ax3.set_xlabel("Model", fontsize=12)
    ax3.set_ylabel("Spike Factor (N=20 / N=10)", fontsize=12)
    ax3.set_title("C. Phase Transition Magnitude at N=20", fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, rotation=15, ha='right')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars1, phi_spikes)):
        ax3.annotate(f'{val:.1f}x', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9, fontweight='bold')

    for i, (bar, val) in enumerate(zip(bars2, css_spikes)):
        ax3.annotate(f'{val:.1f}x', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)

    # Horizontal line at 1x (no change)
    ax3.axhline(y=1, color='black', linestyle='--', alpha=0.3)

    # =========================================================================
    # Plot 4: Detailed trajectory for Qwen3-14B (the strongest signal)
    # =========================================================================
    ax4 = axes[1, 1]

    if 'Qwen3-14B' in all_results:
        ctx_lens, phi_vals, css_vals = extract_phi_and_css(all_results['Qwen3-14B'])

        # Twin axis for CSS
        ax4_twin = ax4.twinx()

        line1, = ax4.plot(ctx_lens, phi_vals, 'o-', color='#9B2335',
                          linewidth=3, markersize=10, label='Φ (Cluster Separation)')
        line2, = ax4_twin.plot(ctx_lens, css_vals, 's--', color='#2A9D8F',
                               linewidth=3, markersize=10, label='CSS (Max Sensitivity)')

        ax4.set_xlabel("Context Length (N)", fontsize=12)
        ax4.set_ylabel("Cluster Separation Φ", fontsize=12, color='#9B2335')
        ax4_twin.set_ylabel("Max CSS", fontsize=12, color='#2A9D8F')
        ax4.set_title("D. Qwen3-14B: Phase Transition Detail", fontsize=13, fontweight='bold')

        ax4.tick_params(axis='y', labelcolor='#9B2335')
        ax4_twin.tick_params(axis='y', labelcolor='#2A9D8F')

        # Highlight N=20 spike
        ax4.axvline(x=20, color='red', linestyle='--', alpha=0.7, linewidth=2)

        # Add annotations for the spike
        idx_20 = np.where(ctx_lens == 20)[0][0]
        ax4.annotate(f'Φ = {phi_vals[idx_20]:,.0f}\n(24x spike)',
                    xy=(20, phi_vals[idx_20]),
                    xytext=(50, phi_vals[idx_20]*0.8),
                    fontsize=10, color='#9B2335', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#9B2335'))

        ax4_twin.annotate(f'CSS = {css_vals[idx_20]:,.0f}\n(33x spike)',
                         xy=(20, css_vals[idx_20]),
                         xytext=(60, css_vals[idx_20]*0.6),
                         fontsize=10, color='#2A9D8F', fontweight='bold',
                         arrowprops=dict(arrowstyle='->', color='#2A9D8F'))

        # Legend
        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='upper right', fontsize=10)

        ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Main title
    fig.suptitle("Phase Transition Discovery: Structural Emergence in ICL at N≈20",
                 fontsize=15, fontweight='bold', y=1.02)

    # Save
    output_path = output_dir / "phase_transition_spike.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / "phase_transition_spike.pdf", bbox_inches='tight', facecolor='white')

    print(f"\nPhase transition plot saved to: {output_path}")

    # Also print the data table
    print("\n" + "="*80)
    print("PHASE TRANSITION DATA TABLE")
    print("="*80)
    print(f"{'Model':<20} {'N=10 Φ':>12} {'N=20 Φ':>12} {'N=30 Φ':>12} {'Φ Spike':>10} {'CSS Spike':>10}")
    print("-"*80)

    for model_name, model_data in all_results.items():
        ctx_lens, phi_vals, css_vals = extract_phi_and_css(model_data)

        idx_10 = np.where(ctx_lens == 10)[0]
        idx_20 = np.where(ctx_lens == 20)[0]
        idx_30 = np.where(ctx_lens == 30)[0]

        if len(idx_10) > 0 and len(idx_20) > 0 and len(idx_30) > 0:
            phi_10 = phi_vals[idx_10[0]]
            phi_20 = phi_vals[idx_20[0]]
            phi_30 = phi_vals[idx_30[0]]
            css_10 = css_vals[idx_10[0]]
            css_20 = css_vals[idx_20[0]]

            phi_spike = phi_20 / phi_10 if phi_10 > 0 else 0
            css_spike = css_20 / css_10 if css_10 > 0 else 0

            print(f"{model_name:<20} {phi_10:>12.1f} {phi_20:>12.1f} {phi_30:>12.1f} {phi_spike:>9.1f}x {css_spike:>9.1f}x")

    print("="*80)

    plt.close()
    return output_path


if __name__ == "__main__":
    main()
