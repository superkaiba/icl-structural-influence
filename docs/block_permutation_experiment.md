# Block Permutation Experiment

## Research Question

**Does sequential order matter for in-context learning of hierarchical graph structure?**

## Hypothesis

If ICL learns structure by tracking sequential graph traversal, then presenting the same tokens in scrambled order should destroy or significantly degrade the learned structure.

## Experimental Design

### Control: Natural Random Walk
```
[A1 A2 A3] [B1 B2] [A4 A5] [C1 C2 C3]
```
Respects graph structure - stays within clusters with occasional bridges

### Treatment: Block-Permuted Walk
```
[C1 C2 C3] [A1 A2 A3] [B1 B2] [A4 A5]
```
Same tokens, same edges within blocks, but global order shuffled

**Key property:** Both are valid walks (edges within blocks exist in graph), but permutation destroys the natural progression of cluster exploration.

## Predictions

### If Order Matters (Expected)

| Metric | Natural | Permuted | Interpretation |
|--------|---------|----------|----------------|
| Cluster Separation (Φ) | High, increases with N | Low, flat trajectory | Model can't track coherent clusters |
| MDS Structure | Clear clustering | Random cloud | No geometric organization |
| Perplexity | Decreases with N | Remains high or increases | Model struggles with unnatural transitions |
| Effect Size | - | 2-5× reduction in Φ | Large, robust effect |

### If Order Doesn't Matter (Alternative)

Both conditions show similar Φ trajectories → would suggest bag-of-words learning rather than sequential structure tracking.

## Usage

### Quick Test (GPT-2, ~5 minutes)
```bash
python run_block_permutation_experiment.py \
    --model gpt2 \
    --context-lengths 10,20,50,100 \
    --n-samples 20 \
    --seed 42
```

### Full Experiment (LLaMA-3.1-8B, ~2 hours)
```bash
python run_block_permutation_experiment.py \
    --model meta-llama/Llama-3.1-8B \
    --context-lengths 10,20,30,50,75,100,150,200 \
    --n-samples 50 \
    --layer -5 \
    --output-dir results/block_permutation
```

### Generate Plots
```bash
python plot_block_permutation.py results/block_permutation/block_permutation_*.json
```

## Output Files

```
results/block_permutation/
├── block_permutation_gpt2.json           # Raw results
├── block_permutation_comparison.png      # Visualization
└── checkpoint_N=*.json                   # Per-length checkpoints (optional)
```

## Implementation Details

### Block Identification Algorithm
```python
def identify_blocks(node_sequence, graph):
    """
    Find maximal consecutive sequences from same cluster.

    Example:
        Input:  [A1 A2 A3 B1 B2 A4 A5]
        Output: [(0,3,'A'), (3,5,'B'), (5,7,'A')]
    """
```

### Permutation Strategy
- Identify all blocks in natural walk
- Randomly shuffle block order (keep internal structure)
- Result: Valid walk locally, scrambled globally

### Why This Tests Sequential Learning
1. **Token statistics preserved:** Same co-occurrence patterns
2. **Local structure preserved:** Edges within blocks valid
3. **Global structure destroyed:** Cluster transitions scrambled
4. **Bridges lose meaning:** No longer connect established clusters

## Key Metrics

### 1. Cluster Separation (Φ)
```
Φ = mean_{i≠j} ||μ_i - μ_j||² / mean_i Var_i
```
Primary outcome measure. Expected: Natural >> Permuted

### 2. Effect Size
```
Ratio = Φ_natural / Φ_permuted
```
Expected: 2-5× across all context lengths

### 3. Perplexity
```
PPL = exp(mean(losses))
```
Secondary measure. If permutation confuses model, PPL should be higher

## Expected Results Timeline

### N=10 (Early Context)
- **Natural:** Minimal structure (insufficient context)
- **Permuted:** Also minimal
- **Difference:** Small (~10-20%)

### N=50 (Critical Window)
- **Natural:** Structure emerging, Φ ≈ 1.5-2.0
- **Permuted:** Still flat, Φ ≈ 0.5-0.8
- **Difference:** Large (~2-3×)

### N=200 (Full Context)
- **Natural:** Strong structure, Φ ≈ 2.5-3.5
- **Permuted:** Weak/no structure, Φ ≈ 0.8-1.2
- **Difference:** Very large (~3-5×)

## Interpretation Guide

### Strong Effect (Ratio > 3×)
→ Sequential order is critical for ICL structure learning
→ Model tracks coherent graph traversal, not just token statistics

### Moderate Effect (Ratio 1.5-3×)
→ Order matters but isn't sole driver
→ Some structure learned from token co-occurrence

### No Effect (Ratio ~ 1×)
→ Order doesn't matter (would falsify hypothesis)
→ Model uses bag-of-words or purely statistical learning

## Follow-up Experiments

If block permutation shows strong effect:

1. **Reversed Walk:** Test if direction matters
2. **Partial Shuffle:** Permute only some blocks
3. **Within-Block Shuffle:** Keep block order, shuffle tokens within
4. **Controlled Bridges:** Ensure bridge tokens in similar positions

## Connection to Theory

### Supports Park et al. (2024)
Block permutation tests whether ICL builds **sequential** task representations vs statistical associations

### Extends Lee et al. (2025)
If order matters, validates treating context length as temporal analog to training dynamics

### Tests Core ICL Mechanism
- **Sequential hypothesis:** Model tracks state through context
- **Statistical hypothesis:** Model aggregates co-occurrence patterns

## Notes

- All blocks remain valid walks (respects graph edges)
- Permutation is reproducible (seeded)
- Same layer (-5 from end) used for both conditions
- Results saved as JSON for further analysis
- Experiment is fully reversible (can reconstruct natural order)

## Citation

If you use this experiment design:
```
Block Permutation Test for Sequential ICL Structure Learning
Derived from: Park et al. (2024) + Lee et al. (2025)
```
