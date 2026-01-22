# In-Context Representation Influence

Research on how in-context examples influence geometric structures in LLM representations.

This project bridges [Park et al. (2024) "In-Context Learning of Representations"](https://arxiv.org/abs/2403.04903) with [Lee et al. (2025) "Influence Dynamics and Stagewise Data Attribution"](https://arxiv.org/abs/2501.00345) to study how context shapes internal representations in transformers.

## Quick Start

```bash
# Clone and install
git clone https://github.com/superkaiba/icl-structural-influence.git
cd icl-structural-influence
uv sync

# Run a quick test
uv run python experiments/core/run_experiment.py --model gpt2 --n-contexts 10 --layers 0,5,11

# Run with a larger model
uv run python experiments/core/run_experiment.py --model meta-llama/Meta-Llama-3-8B --n-contexts 100
```

## Core Concept

**Context Sensitivity Scores (CSS)** approximate Bayesian Influence Functions by measuring covariance between per-token losses and structural metrics across different contexts:

```
CSS(z_i, Φ) = -Cov_contexts(L(z_i), Φ)
```

Where:
- `z_i` is a token in the input sequence
- `L(z_i)` is the per-token next-token prediction loss
- `Φ` is a structural metric (e.g., cluster separation, Dirichlet energy)
- The covariance is computed across different context configurations

CSS is a **correlational** measure (true causal BIF requires SGLD weight-space sampling).

## Installation

### Requirements
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- CUDA-capable GPU (8B models need ~40GB VRAM, use `--dtype bfloat16`)

### Setup with uv (recommended)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Install with optional dependencies
uv sync --extra full
```

### Alternative: Setup with pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install from pyproject.toml
pip install -e .

# With optional dependencies
pip install -e ".[full]"
```

## Project Structure

```
.
├── src/                          # Core library code
│   ├── data/                     # Graph generation and data structures
│   │   ├── hierarchical_graph.py # Stochastic Block Model graphs
│   │   └── dual_interpretation_graph.py
│   ├── models/                   # Model wrappers
│   │   └── hooked_model.py       # Activation extraction infrastructure
│   ├── metrics/                  # Structural metrics and CSS computation
│   │   ├── structural_influence.py
│   │   └── superposition_metrics.py
│   └── visualization/            # Plotting and animation utilities
│       ├── hierarchy_plots.py
│       └── animation.py
│
├── experiments/                  # Experiment scripts
│   ├── core/                     # Main research experiments
│   ├── reproductions/            # Paper reproductions
│   ├── analysis/                 # Analysis and comparison scripts
│   ├── plotting/                 # Visualization scripts
│   ├── logging/                  # W&B logging utilities
│   └── deprecated/               # Old versions (preserved for reference)
│
├── scripts/                      # Shell scripts for long-running experiments
├── results/                      # Experiment outputs (JSON, figures)
├── docs/                         # Additional documentation
├── wandb/                        # W&B run histories
├── pyproject.toml                # Project configuration (uv/pip)
└── CLAUDE.md                     # AI assistant instructions
```

## Running Experiments

### Core Experiments

```bash
# Basic experiment - analyze CSS across layers
python experiments/core/run_experiment.py \
    --model meta-llama/Meta-Llama-3-8B \
    --n-contexts 100 \
    --layers 0,8,16,24,31

# Hierarchical structure learning
python experiments/core/run_hierarchical_experiment.py

# Deep hierarchy (3-4 levels)
python experiments/core/run_deep_hierarchy_experiment.py \
    --model gpt2 \
    --context-lengths 10,20,30 \
    --n-contexts 5

# Leave-one-out influence analysis
python experiments/core/leave_one_out_experiment.py --use-semantic-tokens
python experiments/core/run_multilayer_loo_experiment.py  # All 32 layers

# Block permutation experiments
python experiments/core/run_block_permutation_experiment.py
```

### Paper Reproductions

```bash
# Park et al. (2024) reproduction
python experiments/reproductions/reproduce_park_et_al.py

# Lee et al. (2025) replication
python experiments/reproductions/run_lee_et_al_experiments.py
```

### Analysis Scripts

```bash
# Compare experiments
python experiments/analysis/compare_experiments.py

# Dimensionality reduction analysis
python experiments/analysis/run_dim_reduction_comparison.py
```

### Long-Running Experiments

For multi-hour experiments, use the shell scripts:

```bash
# Run with checkpoints
bash scripts/run_full_multilayer_experiment.sh

# Monitor progress
bash scripts/monitor_full_experiment.sh
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Dirichlet Energy** | Smoothness of representations over graph structure (lower = smoother) |
| **Cluster Separation** | Ratio of inter-cluster to intra-cluster distances |
| **Bridge Tokens** | Cluster-crossing positions (structurally important, often show CSS sign flips) |
| **Phase Transitions** | Context lengths where structural metrics change rapidly |

## Usage Example

```python
from src.data import HierarchicalGraph, HierarchicalGraphConfig
from src.models import HookedLLM
from src.metrics import ClusterSeparation, ContextSensitivityScore

# Generate hierarchical graph structure
config = HierarchicalGraphConfig(
    num_superclusters=3,
    nodes_per_cluster=5,
    walk_length=50
)
graph = HierarchicalGraph(config)
prompt, nodes = graph.generate_random_walk(return_nodes=True)

# Load model and extract activations
model = HookedLLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
output, cache = model.forward_with_cache(prompt, layers=[0, 15, 31])
losses = model.compute_per_token_loss(prompt)

# Get representations from specific layer
representations = cache.get_residual_stream(layer_idx=15)

# Compute CSS
metric = ClusterSeparation()
css = ContextSensitivityScore(metric)
result = css.compute_batch(representations, losses, cluster_labels, n_contexts=100)
```

## Models Tested

| Model | Parameters | Notes |
|-------|-----------|-------|
| `meta-llama/Meta-Llama-3-8B` | 8B | Primary model |
| `Qwen/Qwen2.5-7B` | 7B | Alternative |
| `mistralai/Mistral-7B-v0.1` | 7B | Alternative |
| `gpt2` | 124M | Quick testing |

## Results and Experiment Tracking

- **Results**: Saved to `results/<experiment_name>/` with JSON data and figures
- **W&B**: Experiments are logged to Weights & Biases for interactive visualization
- **Experiment summaries**: Markdown reports in `experiments/YYYY-MM-DD_experiment-name.md`

## Hardware Requirements

- **8B models**: NVIDIA A40 (48GB) or equivalent
- **Use `--dtype bfloat16`** for memory efficiency
- **GPT-2 testing**: Any CUDA GPU or CPU

## References

- Park, J., et al. (2024). "In-Context Learning of Representations." arXiv:2403.04903
- Lee, A., et al. (2025). "Influence Dynamics and Stagewise Data Attribution." arXiv:2501.00345

## License

[Add license information]
