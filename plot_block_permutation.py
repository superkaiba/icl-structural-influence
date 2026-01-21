#!/usr/bin/env python3
"""
Visualization script for block permutation experiment results.

Reads JSON output from run_block_permutation_experiment.py and generates plots.

Usage:
    python plot_block_permutation.py results/block_permutation/block_permutation_gpt2.json
"""

import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def load_results(filepath):
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_comparison(results, output_dir=None):
    """Generate comparison plots."""
    context_lengths = results['context_lengths']

    # Convert string keys to ints
    natural = {int(k): v for k, v in results['natural'].items()}
    permuted = {int(k): v for k, v in results['permuted'].items()}
    natural_ppl = {int(k): v for k, v in results['natural_perplexity'].items()}
    permuted_ppl = {int(k): v for k, v in results['permuted_perplexity'].items()}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Cluster Separation
    ax = axes[0]

    natural_means = [np.mean(natural[N]) for N in context_lengths]
    natural_stds = [np.std(natural[N]) for N in context_lengths]
    permuted_means = [np.mean(permuted[N]) for N in context_lengths]
    permuted_stds = [np.std(permuted[N]) for N in context_lengths]

    ax.errorbar(context_lengths, natural_means, yerr=natural_stds,
                marker='o', linewidth=2, capsize=5, label='Natural Walk',
                color='#2E86AB')
    ax.errorbar(context_lengths, permuted_means, yerr=permuted_stds,
                marker='s', linewidth=2, capsize=5, label='Block-Permuted',
                color='#A23B72')

    ax.set_xlabel('Context Length (N)', fontsize=12)
    ax.set_ylabel('Cluster Separation (Φ)', fontsize=12)
    ax.set_title('Effect of Block Permutation on Structure Learning',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # Add effect size annotations
    for i, N in enumerate(context_lengths):
        if permuted_means[i] > 0:
            ratio = natural_means[i] / permuted_means[i]
            ax.text(N, max(natural_means[i], permuted_means[i]) * 1.1,
                   f'{ratio:.1f}x', ha='center', fontsize=9, color='gray')

    # Plot 2: Perplexity
    ax = axes[1]

    nat_ppl_means = [np.mean(natural_ppl[N]) for N in context_lengths]
    nat_ppl_stds = [np.std(natural_ppl[N]) for N in context_lengths]
    perm_ppl_means = [np.mean(permuted_ppl[N]) for N in context_lengths]
    perm_ppl_stds = [np.std(permuted_ppl[N]) for N in context_lengths]

    ax.errorbar(context_lengths, nat_ppl_means, yerr=nat_ppl_stds,
                marker='o', linewidth=2, capsize=5, label='Natural Walk',
                color='#2E86AB')
    ax.errorbar(context_lengths, perm_ppl_means, yerr=perm_ppl_stds,
                marker='s', linewidth=2, capsize=5, label='Block-Permuted',
                color='#A23B72')

    ax.set_xlabel('Context Length (N)', fontsize=12)
    ax.set_ylabel('Perplexity', fontsize=12)
    ax.set_title('Prediction Difficulty Comparison',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    # Save
    if output_dir:
        output_path = Path(output_dir)
        output_file = output_path / "block_permutation_comparison.png"
    else:
        output_file = "block_permutation_comparison.png"

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.show()


def print_summary(results):
    """Print statistical summary."""
    context_lengths = results['context_lengths']
    natural = {int(k): v for k, v in results['natural'].items()}
    permuted = {int(k): v for k, v in results['permuted'].items()}

    print("\nSUMMARY: Block Permutation Effect")
    print("=" * 60)
    print(f"{'N':>5} | {'Natural Φ':>12} | {'Permuted Φ':>12} | {'Ratio':>8}")
    print("-" * 60)

    for N in context_lengths:
        nat_mean = np.mean(natural[N])
        nat_std = np.std(natural[N])
        perm_mean = np.mean(permuted[N])
        perm_std = np.std(permuted[N])
        ratio = nat_mean / perm_mean if perm_mean > 0 else float('inf')

        print(f"{N:>5} | {nat_mean:>5.3f} ± {nat_std:>4.3f} | "
              f"{perm_mean:>5.3f} ± {perm_std:>4.3f} | {ratio:>7.2f}x")

    print("=" * 60)
    print(f"Model: {results['metadata']['model']}")
    print(f"Layer: {results['metadata']['layer']}")
    print(f"Samples: {results['metadata']['n_samples']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_block_permutation.py <results_json_file>")
        sys.exit(1)

    results_file = sys.argv[1]
    results = load_results(results_file)

    print_summary(results)
    plot_comparison(results, output_dir=Path(results_file).parent)
