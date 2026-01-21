#!/usr/bin/env python3
"""Log permutation comparison results (natural vs block shuffle vs full shuffle) to W&B."""

import json
from pathlib import Path
import numpy as np
import wandb

output_dir = Path("results/block_permutation")

# Load result files
with open(output_dir / "block_permutation_gpt2.json") as f:
    block_results = json.load(f)

with open(output_dir / "full_shuffle_permutation_gpt2.json") as f:
    shuffle_results = json.load(f)

context_lengths = [10, 20, 50, 100, 200]

# Extract data
natural_means = []
natural_stds = []
block_means = []
block_stds = []
shuffle_means = []
shuffle_stds = []

for N in context_lengths:
    key = str(N) if str(N) in block_results['natural'] else N

    natural_means.append(np.mean(block_results['natural'][key]))
    natural_stds.append(np.std(block_results['natural'][key]))

    block_means.append(np.mean(block_results['permuted'][key]))
    block_stds.append(np.std(block_results['permuted'][key]))

    key_s = str(N) if str(N) in shuffle_results['permuted'] else N
    shuffle_means.append(np.mean(shuffle_results['permuted'][key_s]))
    shuffle_stds.append(np.std(shuffle_results['permuted'][key_s]))

# Compute ratios (natural / permuted)
block_ratios = [n / b if b > 0 else 0 for n, b in zip(natural_means, block_means)]
shuffle_ratios = [n / s if s > 0 else 0 for n, s in zip(natural_means, shuffle_means)]

print("Logging to W&B...")

run = wandb.init(
    project='icl-structural-influence',
    name='permutation-comparison-gpt2',
    tags=['permutation', 'block-shuffle', 'full-shuffle', 'cluster-separation', 'gpt2'],
    mode='offline',  # Use offline mode, sync later with `wandb sync`
    config={
        'model': 'gpt2',
        'n_samples': 50,
        'context_lengths': context_lengths,
        'layer': -5,
        'seed': 42,
        'conditions': ['natural', 'block_shuffle', 'full_shuffle']
    }
)

# Log the comparison plot
wandb.log({
    'three_condition_comparison': wandb.Image(
        str(output_dir / 'three_condition_comparison.png'),
        caption='Cluster separation across three conditions: natural walk, block shuffle, full shuffle'
    )
})

# Log individual experiment plots
wandb.log({
    'block_permutation_plot': wandb.Image(
        str(output_dir / 'block_permutation_comparison.png'),
        caption='Block permutation: natural vs block-shuffled'
    ),
    'full_shuffle_plot': wandb.Image(
        str(output_dir / 'full_shuffle_permutation_comparison.png'),
        caption='Full shuffle: natural vs fully-shuffled'
    )
})

# Create detailed results table
table = wandb.Table(columns=[
    'Context_Length',
    'Natural_Phi', 'Natural_Std',
    'Block_Shuffle_Phi', 'Block_Shuffle_Std', 'Block_Ratio',
    'Full_Shuffle_Phi', 'Full_Shuffle_Std', 'Full_Shuffle_Ratio'
])

for i, N in enumerate(context_lengths):
    table.add_data(
        N,
        round(natural_means[i], 2), round(natural_stds[i], 2),
        round(block_means[i], 2), round(block_stds[i], 2), round(block_ratios[i], 2),
        round(shuffle_means[i], 2), round(shuffle_stds[i], 2), round(shuffle_ratios[i], 2)
    )

wandb.log({'permutation_comparison_table': table})

# Log trajectory data for line plots
for i, N in enumerate(context_lengths):
    wandb.log({
        'context_length': N,
        'natural_phi': natural_means[i],
        'block_shuffle_phi': block_means[i],
        'full_shuffle_phi': shuffle_means[i],
        'natural_vs_block_ratio': block_ratios[i],
        'natural_vs_shuffle_ratio': shuffle_ratios[i],
    })

# Summary statistics
wandb.summary['model'] = 'gpt2'
wandb.summary['n_samples'] = 50
wandb.summary['max_context'] = 200

# Block shuffle summary
wandb.summary['block_shuffle_mean_ratio'] = np.mean(block_ratios)
wandb.summary['block_shuffle_ratio_at_N200'] = block_ratios[-1]

# Full shuffle summary
wandb.summary['full_shuffle_mean_ratio'] = np.mean(shuffle_ratios)
wandb.summary['full_shuffle_ratio_at_N200'] = shuffle_ratios[-1]

# Key finding: full shuffle effect grows with context
wandb.summary['full_shuffle_ratio_N10'] = shuffle_ratios[0]
wandb.summary['full_shuffle_ratio_N200'] = shuffle_ratios[-1]
wandb.summary['full_shuffle_effect_growth'] = shuffle_ratios[-1] / shuffle_ratios[0] if shuffle_ratios[0] > 0 else 0

# Interpretation flags
wandb.summary['block_shuffle_helps'] = np.mean(block_ratios) < 1  # ratio < 1 means block shuffle has higher Phi
wandb.summary['sequential_structure_matters'] = shuffle_ratios[-1] > 1.5  # strong effect at long context

print(f"\nRun URL: {run.url}")
wandb.finish()

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Block shuffle mean ratio: {np.mean(block_ratios):.2f}x")
print(f"Full shuffle mean ratio: {np.mean(shuffle_ratios):.2f}x")
print(f"Full shuffle ratio at N=200: {shuffle_ratios[-1]:.2f}x")
print(f"Sequential structure matters: {shuffle_ratios[-1] > 1.5}")
print("=" * 70)
