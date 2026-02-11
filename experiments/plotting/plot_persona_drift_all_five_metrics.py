"""Plot all 5 collapse metrics for persona drift + WildChat."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    results_path = Path("results/persona_drift_collapse/results_with_wildchat.json")
    output_dir = Path("results/persona_drift_collapse")

    with open(results_path) as f:
        results = json.load(f)

    domains = ["coding", "philosophy", "therapy", "writing", "wildchat"]
    layers = [0, 7, 14, 21, 27]
    metric_names = ["avg_cos_sim", "avg_l2_dist", "spread", "effective_dim", "intrinsic_dim"]

    metric_labels = {
        "avg_cos_sim": "Cosine Similarity\n(higher = collapsed)",
        "avg_l2_dist": "L2 Distance\n(lower = collapsed)",
        "spread": "Spread (Variance)\n(lower = collapsed)",
        "effective_dim": "Effective Dimension\n(lower = collapsed)",
        "intrinsic_dim": "Intrinsic Dimension\n(lower = collapsed)"
    }

    colors = plt.cm.tab10(np.linspace(0, 1, len(domains)))
    styles = ['-', '-', '-', '-', '--']  # Dashed for wildchat

    # 5x5 grid: rows=metrics, cols=layers
    fig, axes = plt.subplots(5, 5, figsize=(20, 20))

    for row, m_name in enumerate(metric_names):
        for col, layer in enumerate(layers):
            ax = axes[row, col]

            for i, domain in enumerate(domains):
                data = results[m_name][domain][str(layer)]
                if not data:
                    continue
                x = sorted([int(k) for k in data.keys()])
                y = [data[str(k)] for k in x]

                lw = 2.5 if domain == "wildchat" else 1.5
                ax.plot(x, y, styles[i], color=colors[i], label=domain,
                       markersize=4, linewidth=lw)

            if row == 0:
                ax.set_title(f"Layer {layer}", fontsize=12)
            if col == 0:
                ax.set_ylabel(metric_labels[m_name], fontsize=10)
            if row == 4:
                ax.set_xlabel("Context Length")
            if col == 4 and row == 0:
                ax.legend(loc='upper right', fontsize=8)

            # Add collapse threshold for cosine similarity
            if m_name == "avg_cos_sim":
                ax.axhline(y=0.9, color='red', linestyle=':', alpha=0.7)
                ax.set_ylim(0, 1)

            ax.grid(True, alpha=0.3)

    plt.suptitle("Persona Drift + WildChat: All 5 Collapse Metrics\n"
                 "(WildChat shown as dashed line - all real conversations show NO collapse)",
                 fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / "all_five_metrics_with_wildchat.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'all_five_metrics_with_wildchat.png'}")

    # Also create Layer 27 bar chart for all 5 metrics
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for idx, m_name in enumerate(metric_names):
        ax = axes[idx]
        final_values = []

        for domain in domains:
            data = results[m_name][domain]["27"]
            if data:
                max_pos = max(int(k) for k in data.keys())
                final_values.append(data[str(max_pos)])
            else:
                final_values.append(0)

        bars = ax.bar(domains, final_values, color=colors[:len(domains)])
        ax.set_title(metric_labels[m_name].replace('\n', ' '), fontsize=10)
        ax.tick_params(axis='x', rotation=45)

        if m_name == "avg_cos_sim":
            ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='Collapse')
            ax.set_ylim(0, 1)

    plt.suptitle("Layer 27 Final Values: All 5 Metrics (Persona Drift + WildChat)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "layer27_all_five_metrics.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'layer27_all_five_metrics.png'}")

if __name__ == "__main__":
    main()
