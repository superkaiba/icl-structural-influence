# Experiment Guide

This guide explains how to run experiments, interpret results, and create new experiments.

## Getting Started

### Recommended First Experiment

Start with a quick test to verify everything works:

```bash
python experiments/core/run_experiment.py \
    --model gpt2 \
    --n-contexts 10 \
    --layers 0,5,11 \
    --output-dir results/test_run
```

This runs in ~2 minutes on CPU and produces basic CSS results.

### Moving to Full-Scale Experiments

Once verified, run with the full model:

```bash
python experiments/core/run_experiment.py \
    --model meta-llama/Meta-Llama-3-8B \
    --n-contexts 100 \
    --layers 0,8,16,24,31 \
    --dtype bfloat16
```

## Experiment Categories

### Core Experiments (`experiments/core/`)

| Script | Purpose | Runtime |
|--------|---------|---------|
| `run_experiment.py` | Basic CSS computation | 10-30 min |
| `run_hierarchical_experiment.py` | Multi-level hierarchy analysis | 20-60 min |
| `run_deep_hierarchy_experiment.py` | 3-4 level deep hierarchies | 30-90 min |
| `leave_one_out_experiment.py` | Single-layer LOO influence | 15-45 min |
| `run_multilayer_loo_experiment.py` | All 32 layers LOO | 7-8 hours |
| `run_block_permutation_experiment.py` | Positional vs semantic | 30-60 min |
| `run_incongruous_injection_experiment.py` | Out-of-distribution tokens | 20-40 min |
| `run_superposition_collapse_experiment.py` | Superposition analysis | 30-60 min |
| `context_switch_experiment.py` | Context switching dynamics | 15-30 min |

### Reproduction Experiments (`experiments/reproductions/`)

| Script | Paper | Purpose |
|--------|-------|---------|
| `reproduce_park_et_al.py` | Park et al. 2024 | ICL representations |
| `run_lee_et_al_experiments.py` | Lee et al. 2025 | Stagewise influence |

### Analysis Scripts (`experiments/analysis/`)

Use these to analyze and compare experiment results:

```bash
# Compare multiple experiment results
python experiments/analysis/compare_experiments.py \
    --results results/exp1 results/exp2 results/exp3

# Compare LOO metrics across conditions
python experiments/analysis/compare_loo_metrics.py \
    --results-dir results/loo_multilayer
```

## Experiment Workflow

### 1. Run Experiment

```bash
python experiments/core/run_multilayer_loo_experiment.py \
    --model meta-llama/Meta-Llama-3-8B \
    --output-dir results/loo_experiment_jan22
```

### 2. Log to W&B (Optional)

```bash
python experiments/logging/log_multilayer_loo_to_wandb.py \
    --results-dir results/loo_experiment_jan22 \
    --project icl-influence
```

### 3. Generate Visualizations

```bash
python experiments/plotting/plot_loo_heatmap.py \
    --results-dir results/loo_experiment_jan22 \
    --output results/loo_experiment_jan22/heatmap.png
```

### 4. Write Experiment Summary

Create `experiments/2026-01-22_loo-experiment.md` following the template in CLAUDE.md.

## Long-Running Experiments

For experiments taking several hours:

### Using Shell Scripts

```bash
# Start experiment with logging
bash run_full_multilayer_experiment.sh

# In another terminal, monitor progress
bash monitor_full_experiment.sh
```

### Checkpointing

Long experiments automatically save checkpoints:

```bash
# Find checkpoints
ls results/loo_multilayer/checkpoint_*.json

# Resume from checkpoint (if script supports it)
python experiments/core/run_multilayer_loo_experiment.py \
    --resume results/loo_multilayer/checkpoint_layer_16.json
```

### Background Execution

```bash
# Run in background with nohup
nohup python experiments/core/run_multilayer_loo_experiment.py > experiment.log 2>&1 &

# Or use screen/tmux
screen -S experiment
python experiments/core/run_multilayer_loo_experiment.py
# Ctrl+A, D to detach
# screen -r experiment to reattach
```

## Common Parameters

Most experiment scripts share these parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | HuggingFace model ID | `meta-llama/Meta-Llama-3-8B` |
| `--n-contexts` | Number of random contexts | 100 |
| `--layers` | Comma-separated layer indices | All layers |
| `--output-dir` | Results directory | `results/<experiment>` |
| `--dtype` | Model dtype (`float32`, `bfloat16`) | `float32` |
| `--seed` | Random seed | 42 |

## Results Structure

Each experiment creates:

```
results/<experiment_name>/
├── results.json         # Main results data
├── config.json          # Experiment configuration
├── checkpoint_*.json    # Intermediate checkpoints (if applicable)
├── figures/
│   ├── *.png           # Generated plots
│   └── *.html          # Interactive visualizations (if applicable)
└── raw_data/           # Optional: raw activation dumps
```

### Results JSON Format

```json
{
  "model": "meta-llama/Meta-Llama-3-8B",
  "timestamp": "2026-01-22T10:30:00",
  "config": {
    "n_contexts": 100,
    "layers": [0, 8, 16, 24, 31]
  },
  "results": {
    "layer_0": {
      "css_per_token": [...],
      "dirichlet_energy": 0.45,
      "cluster_separation": 2.3
    },
    ...
  }
}
```

## Creating New Experiments

### Template

```python
#!/usr/bin/env python
"""
Brief description of the experiment.

Investigates [what] by [how].
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

from src.data import HierarchicalGraph, HierarchicalGraphConfig
from src.models import HookedLLM
from src.metrics import ClusterSeparation, ContextSensitivityScore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--n-contexts", type=int, default=100)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    config["timestamp"] = datetime.now().isoformat()
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Load model
    model = HookedLLM.from_pretrained(args.model, torch_dtype=args.dtype)

    # Run experiment
    results = {}
    # ... your experiment logic here ...

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
```

### Best Practices

1. **Always save config** with the results for reproducibility
2. **Add checkpointing** for experiments > 1 hour
3. **Use consistent argument names** (see Common Parameters)
4. **Log to W&B** for interactive analysis
5. **Write experiment summary** in `experiments/YYYY-MM-DD_*.md`

## Debugging

### Enable Verbose Output

Most scripts support `--verbose` or setting log level:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Test with Small Scale First

```bash
# Quick test
python experiments/core/run_experiment.py \
    --model gpt2 \
    --n-contexts 5 \
    --layers 0,5,11
```

### Check GPU Memory

```python
import torch
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
```
