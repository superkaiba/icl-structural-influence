# Architecture

This document describes the technical architecture of the In-Context Representation Influence framework.

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EXPERIMENT PIPELINE                                │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   Graph Config   │───▶│  HierarchicalGraph│───▶│   Random Walk    │
│ (SBM parameters) │    │   (generates)     │    │   (prompt text)  │
└──────────────────┘    └──────────────────┘    └──────────────────┘
                                                          │
                                                          ▼
                                               ┌──────────────────┐
                                               │    HookedLLM     │
                                               │  (tokenize +     │
                                               │   forward pass)  │
                                               └──────────────────┘
                                                          │
                              ┌────────────────┬──────────┴──────────┐
                              ▼                ▼                      ▼
                    ┌──────────────┐  ┌──────────────┐      ┌──────────────┐
                    │ Activations  │  │ Per-Token    │      │   Cluster    │
                    │ (per layer)  │  │   Losses     │      │   Labels     │
                    └──────────────┘  └──────────────┘      └──────────────┘
                              │                │                      │
                              └────────────────┴──────────────────────┘
                                               │
                                               ▼
                                    ┌──────────────────┐
                                    │  Structural      │
                                    │  Metrics         │
                                    │  (Φ computation) │
                                    └──────────────────┘
                                               │
                                               ▼
                                    ┌──────────────────┐
                                    │     CSS          │
                                    │  Cov(L, Φ)       │
                                    └──────────────────┘
```

## Module Relationships

### Core Modules

```
src/
├── data/
│   ├── __init__.py              # Exports: HierarchicalGraph, HierarchicalGraphConfig
│   ├── hierarchical_graph.py    # Stochastic Block Model graph generation
│   └── dual_interpretation_graph.py  # Extended graph with semantic/structural views
│
├── models/
│   ├── __init__.py              # Exports: HookedLLM
│   └── hooked_model.py          # Activation extraction with forward hooks
│
├── metrics/
│   ├── __init__.py              # Exports: CSS, DirichletEnergy, ClusterSeparation, etc.
│   ├── structural_influence.py  # Core CSS computation and structural metrics
│   └── superposition_metrics.py # Superposition-specific metrics
│
└── visualization/
    ├── __init__.py              # Exports: visualization utilities
    ├── hierarchy_plots.py       # Plotting functions for hierarchical data
    └── animation.py             # GIF generation for representation dynamics
```

### Dependency Graph

```
                    ┌─────────────────────────┐
                    │     Experiment Script   │
                    │    (e.g., run_*.py)     │
                    └─────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │   src.data   │  │ src.models   │  │ src.metrics  │
    └──────────────┘  └──────────────┘  └──────────────┘
              │               │               │
              │               │               │
              ▼               ▼               ▼
    ┌────────────────────────────────────────────────┐
    │        External Dependencies                    │
    │  torch, transformers, numpy, networkx          │
    └────────────────────────────────────────────────┘
```

## Key Algorithms

### 1. Stochastic Block Model Graph Generation

The `HierarchicalGraph` generates graphs with multi-level cluster structure:

```python
# Configuration
config = HierarchicalGraphConfig(
    num_superclusters=3,      # Top-level groups
    clusters_per_supercluster=4,  # Mid-level groups
    nodes_per_cluster=5,      # Leaf nodes
    p_intra_cluster=0.8,      # Edge prob within cluster
    p_inter_cluster=0.1,      # Edge prob between clusters (same super)
    p_inter_supercluster=0.01 # Edge prob between superclusters
)

# Generates adjacency matrix following block structure
```

**Random Walk Generation**:
- Starts at random node
- At each step, samples neighbor with probability proportional to edge weights
- Produces token sequence representing graph traversal

### 2. Activation Extraction (HookedLLM)

Uses forward hooks to capture intermediate representations:

```python
def forward_with_cache(self, input_text, layers):
    """
    Single forward pass that captures activations at specified layers.

    Returns:
        output: Model output (logits)
        cache: ActivationCache with get_residual_stream(layer_idx) method
    """
    # Register hooks for each requested layer
    # Forward pass stores activations in cache
    # Remove hooks after forward
```

**Key Optimization**: Extract all layers in single forward pass instead of multiple passes.

### 3. Structural Metrics

#### Dirichlet Energy
Measures smoothness of representations over graph structure:

```python
def dirichlet_energy(representations, adjacency):
    """
    E(f) = Σ_ij A_ij ||f(i) - f(j)||²

    Lower energy = smoother representation (similar nodes have similar reps)
    """
    diff = representations[None, :] - representations[:, None]  # pairwise differences
    sq_dist = (diff ** 2).sum(-1)  # squared distances
    return (adjacency * sq_dist).sum()
```

#### Cluster Separation
Ratio of inter-cluster to intra-cluster distances:

```python
def cluster_separation(representations, labels):
    """
    CS = mean(inter-cluster distances) / mean(intra-cluster distances)

    Higher = better cluster separation
    """
    intra = mean_distance_within_clusters(representations, labels)
    inter = mean_distance_between_clusters(representations, labels)
    return inter / intra
```

### 4. Context Sensitivity Score (CSS)

Core computation that approximates influence:

```python
def compute_css(losses, structural_metric, n_contexts):
    """
    CSS(z_i, Φ) = -Cov(L(z_i), Φ)

    For each token position i:
    1. Collect L(z_i) across n_contexts different random walks
    2. Collect Φ (structural metric) for each context
    3. Compute covariance between per-token loss and metric

    Negative covariance: when loss decreases, structure improves
    → Token positively influences structure learning
    """
    # Sample n_contexts different random walks
    # For each walk, compute L(z_i) for all positions
    # Compute structural metric Φ for whole sequence
    # Return -Cov(L, Φ) for each position
```

## Caching and Checkpointing

### Activation Cache

```python
class ActivationCache:
    """Stores activations from forward hooks."""

    def __init__(self):
        self._cache = {}  # layer_idx -> tensor

    def get_residual_stream(self, layer_idx):
        """Return activations at specified layer."""
        return self._cache[layer_idx]
```

### Experiment Checkpoints

Long-running experiments save progress after each major step:

```python
# Example checkpoint structure
checkpoint = {
    "completed_context_lengths": [10, 20, 30],
    "current_context_length": 40,
    "results": {
        10: {"css": [...], "metrics": {...}},
        20: {"css": [...], "metrics": {...}},
        30: {"css": [...], "metrics": {...}},
    },
    "timestamp": "2025-01-21T12:00:00"
}
```

Resume by loading checkpoint and continuing from last completed step.

## Memory Considerations

### GPU Memory for 8B Models

| Operation | Memory Usage |
|-----------|-------------|
| Model weights (bfloat16) | ~16 GB |
| Activations (32 layers, seq_len=512) | ~8 GB |
| Gradients (if computing) | ~16 GB |
| **Total (inference only)** | ~24 GB |

### Optimization Strategies

1. **Use `--dtype bfloat16`** for model weights
2. **Extract only needed layers** (don't cache all 32 if only using 5)
3. **Clear cache between contexts** with `torch.cuda.empty_cache()`
4. **Batch contexts** if memory allows for parallelism
