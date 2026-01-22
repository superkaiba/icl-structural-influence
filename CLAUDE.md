# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project investigating how in-context examples influence geometric structures in LLM representations. Bridges Park et al. (2024) "In-Context Learning of Representations" with Lee et al. (2025) "Influence Dynamics and Stagewise Data Attribution".

**Core concept**: Context Sensitivity Scores (CSS) approximate Bayesian Influence Functions (BIF) by measuring covariance between per-token losses and structural metrics across different contexts with frozen model weights.

```
CSS(z_i, Φ) = -Cov_contexts(L(z_i), Φ)
```

CSS is a correlational measure, NOT causal influence (true BIF requires SGLD weight-space sampling).

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Main experiment - full pipeline
python experiments/core/run_experiment.py --model meta-llama/Meta-Llama-3-8B --n-contexts 100

# Quick test with smaller model
python experiments/core/run_experiment.py --model gpt2 --n-contexts 10 --layers 0,5,11

# Hierarchical learning analysis
python experiments/core/run_hierarchical_experiment.py

# Deep hierarchy (3-4 levels) with stagewise learning
python experiments/core/run_deep_hierarchy_experiment.py --model gpt2 --context-lengths 10,20,30 --n-contexts 5

# Leave-one-out influence experiments
python experiments/core/leave_one_out_experiment.py --use-semantic-tokens
python experiments/core/run_multilayer_loo_experiment.py  # All 32 layers, ~7-8 hours

# Lee et al. replication experiments
python experiments/reproductions/run_lee_et_al_experiments.py

# Park et al. reproduction
python experiments/reproductions/reproduce_park_et_al.py

# Long-running experiments (with checkpoints)
bash run_full_multilayer_experiment.sh  # Runs experiment with checkpoint saves
bash monitor_full_experiment.sh         # Monitor progress of running experiment
```

## Architecture

### Core Modules (`src/`)

**`src/data/hierarchical_graph.py`** - Stochastic Block Model graph generation
- `HierarchicalGraphConfig`: Configuration dataclass (superclusters, nodes_per_cluster, edge probabilities)
- `HierarchicalGraph`: 3-level hierarchy generator with random walk generation

**`src/models/hooked_model.py`** - Hook infrastructure for activation extraction
- `HookedLLM`: Main wrapper for any HuggingFace causal LM
- `forward_with_cache()`: Captures activations at specified layers
- `compute_per_token_loss()`: Per-token next-token prediction loss

**`src/metrics/structural_influence.py`** - CSS computation engine
- Structural metrics: `DirichletEnergy`, `ClusterSeparation`, `RepresentationCoherence`, `LayerWiseProgress`
- `ContextSensitivityScore`: Core CSS computation (single sample and batch modes)
- `compute_hierarchical_decomposition()`: Within vs between cluster CSS

### Typical Usage Pattern

```python
from src.data import HierarchicalGraph, HierarchicalGraphConfig
from src.models import HookedLLM
from src.metrics import ClusterSeparation, ContextSensitivityScore

# Generate graph structure
config = HierarchicalGraphConfig(num_superclusters=3, nodes_per_cluster=5, walk_length=50)
graph = HierarchicalGraph(config)
prompt, nodes = graph.generate_random_walk(return_nodes=True)

# Load model and extract activations
model = HookedLLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
output, cache = model.forward_with_cache(prompt, layers=[0, 15, 31])  # Efficient multi-layer extraction
losses = model.compute_per_token_loss(prompt)

# Compute CSS
metric = ClusterSeparation()
css = ContextSensitivityScore(metric)
result = css.compute_batch(representations, losses, cluster_labels, n_contexts=100)
```

**Key Pattern**: Use `forward_with_cache()` to extract all layer activations in a single forward pass (more efficient than multiple forwards). The `cache` object provides `get_residual_stream(layer_idx)` for layer-specific representations.

## Key Concepts

- **Dirichlet Energy**: Smoothness of representations over graph structure (lower = smoother)
- **Cluster Separation**: Ratio of inter-cluster to intra-cluster distances
- **Bridge Tokens**: Cluster-crossing positions (structurally important, often show sign flips)
- **Phase Transitions**: Context lengths where structural metrics change rapidly

## Experiment Workflow

**Directory Organization** (under `experiments/`):
- `experiments/core/` - Main research experiment scripts
- `experiments/reproductions/` - Paper reproduction scripts (Park, Lee et al.)
- `experiments/analysis/` - Comparison and analysis scripts
- `experiments/plotting/` - Visualization scripts
- `experiments/logging/` - W&B logging utilities
- `experiments/deprecated/` - Old versions (preserved for reference)

**Naming Patterns**:
- `run_*.py` - Main experiment scripts that generate data
- `log_*_to_wandb.py` - Upload experiment results to Weights & Biases
- `plot_*.py` - Generate visualizations from experiment data
- `compare_*.py` - Cross-experiment comparison scripts

**Checkpoint Pattern**: Long-running experiments (e.g., `run_multilayer_loo_experiment.py`) save checkpoint files after each major step (e.g., after each context length). If interrupted, check `results/*/checkpoint_*.json` for partial results.

**Shell Scripts**: Multi-hour experiments have corresponding `.sh` scripts for execution and monitoring:
- `run_full_*.sh` - Execute full experiment with progress logging
- `monitor_*.sh` - Check progress/status of running experiments

**Results Organization**:
- `results/` - Experiment outputs organized by experiment type (subdirectories)
- `wandb/` - Weights & Biases run histories and artifacts
- `experiments/*.md` - Markdown summaries of completed experiments with embedded figures
- `docs/` - Additional experiment documentation

## Experiment Summaries

After completing an experiment, save a markdown summary to `experiments/` with the following format:

**Filename**: `experiments/YYYY-MM-DD_experiment-name.md`

**Template**:
```markdown
# Experiment: [Descriptive Title]

**Date**: YYYY-MM-DD
**Model**: [model name]
**W&B Run**: [link to wandb run]

## Objective

[Brief description of what the experiment tests]

## Configuration

- Context lengths: [list]
- N trials: [number]
- Layers tested: [list or "all"]
- Token conditions: [semantic/unrelated/both]

## Key Results

[Summary of main findings - 3-5 bullet points]

## Figures

### [Figure Title 1]
![Description](../results/[experiment_dir]/[figure1].png)

[Brief interpretation of the figure]

### [Figure Title 2]
![Description](../results/[experiment_dir]/[figure2].png)

[Brief interpretation of the figure]

## Raw Data

- Results JSON: `results/[experiment_dir]/results.json`
- Checkpoints: `results/[experiment_dir]/checkpoint_*.json`

## Notes

[Any issues encountered, surprising findings, or follow-up ideas]
```

**Key Requirements**:
- Use relative paths (`../results/...`) for images so they render in markdown viewers
- Include W&B link for interactive exploration
- Keep interpretations brief but informative
- Note any anomalies or unexpected results

## Models Tested

Primary: `meta-llama/Meta-Llama-3-8B` (8B), also tested with `Qwen2.5-7B`, `Mistral-7B`. Quick testing: use `gpt2`.

## Hardware Notes

Experiments run on NVIDIA A40 (48GB) GPUs. 8B models require significant VRAM - use `--dtype bfloat16` for efficiency.

## Implementation Guidelines

**Test and Run**: When implementing new code, always:
1. Run syntax checks (`python -m py_compile <file>`) on created files
2. Verify imports work correctly
3. Run any demo functions or basic tests
4. If tests pass, run the actual experiment/script to verify it works end-to-end

**Specification Changes**: If during implementation something comes up that requires deviating from the original specifications or plan:
- **STOP** implementation
- **ASK** for clarification from the user before proceeding
- Do not make assumptions about what changes are acceptable
- Explain what the issue is and what options exist
