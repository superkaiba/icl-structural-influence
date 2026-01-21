#!/usr/bin/env python3
"""Log the variance results to W&B (after the visualization was already created)."""

import json
from pathlib import Path
import wandb

output_dir = Path("results/semantic_override_trajectory")

# Key findings from the run
results = {
    'context_lengths': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000, 7500, 10000],
    'ratio_mean': [0.366, 0.713, 0.856, 0.885, 1.015, 1.016, 1.041, 0.955, 0.915, 0.916, 0.940, 1.093, 1.126, 1.053, 0.990, 1.019, 1.131, 1.000, 1.062, 1.048, 1.052, 1.100, 1.045, 1.175, 1.192, 1.266, 1.339, 1.266, 1.403, 1.510, 1.732, 1.542, 1.672, 1.594, 1.497, 1.572, 1.233, 1.174, 1.097],
    'ratio_std': [0.000, 0.007, 0.190, 0.064, 0.146, 0.099, 0.169, 0.120, 0.087, 0.153, 0.113, 0.117, 0.118, 0.077, 0.093, 0.170, 0.238, 0.130, 0.099, 0.111, 0.122, 0.090, 0.101, 0.025, 0.097, 0.101, 0.038, 0.124, 0.072, 0.119, 0.195, 0.098, 0.148, 0.141, 0.157, 0.155, 0.049, 0.117, 0.028],
}

n_trials = 5
stable_crossover = 4
sig_crossover = 12
peak_n = 500
peak_ratio = 1.732

# Save JSON
with open(output_dir / "variance_results.json", "w") as f:
    json.dump({
        'context_lengths': results['context_lengths'],
        'ratio_mean': results['ratio_mean'],
        'ratio_std': results['ratio_std'],
        'n_trials': n_trials,
        'stable_crossover': stable_crossover,
        'sig_crossover': sig_crossover,
        'peak_n': peak_n,
        'peak_ratio': peak_ratio,
    }, f, indent=2)
print(f"Saved: {output_dir / 'variance_results.json'}")

# Log to W&B
print("\nLogging to W&B...")

run = wandb.init(
    project='icl-structural-influence',
    name='semantic-override-with-variance',
    tags=['semantic-override', 'variance', 'error-bars', 'trajectory'],
    config={
        'n_trials': n_trials,
        'context_lengths': results['context_lengths'],
        'max_context': 10000,
        'n_samples_per_trial': 50,
    }
)

wandb.log({
    'semantic_override_variance_plot': wandb.Html(
        open(output_dir / 'semantic_override_with_variance.html').read()
    )
})

# Log table
table = wandb.Table(columns=[
    'N', 'Ratio_Mean', 'Ratio_Std', 'Ratio_CI95',
    'Winner', 'Significant'
])
for i, n in enumerate(results['context_lengths']):
    winner = 'Graph' if results['ratio_mean'][i] > 1 else 'Semantic'
    significant = 'Yes' if results['ratio_mean'][i] - results['ratio_std'][i] > 1 else 'No'
    table.add_data(
        n,
        results['ratio_mean'][i],
        results['ratio_std'][i],
        2 * results['ratio_std'][i],
        winner,
        significant
    )
wandb.log({'variance_table': table})

# Summary
wandb.summary['n_trials'] = n_trials
wandb.summary['pretrained_ratio_mean'] = results['ratio_mean'][0]
wandb.summary['stable_crossover_N'] = stable_crossover
wandb.summary['significant_crossover_N'] = sig_crossover
wandb.summary['peak_separation_N'] = peak_n
wandb.summary['peak_separation_ratio'] = peak_ratio
wandb.summary['max_variance_N'] = results['context_lengths'][results['ratio_std'].index(max(results['ratio_std']))]
wandb.summary['max_variance_std'] = max(results['ratio_std'])

print(f"\nRun URL: {run.url}")
wandb.finish()

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Pretrained ratio: {results['ratio_mean'][0]:.3f}")
print(f"Stable crossover: N={stable_crossover}")
print(f"Statistically significant crossover: N={sig_crossover}")
print(f"Peak separation: N={peak_n}, ratio={peak_ratio:.3f}")
print("=" * 70)
