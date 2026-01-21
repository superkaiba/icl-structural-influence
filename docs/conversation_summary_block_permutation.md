# Conversation Summary: Block Permutation Experiment Design

**Date:** 2026-01-16

## Research Framework Overview

### Core Measurement: Context Sensitivity Scores (CSS)

**Formula:**
```
CSS(z_i, Φ) = -Cov_contexts(L(z_i), Φ)
```

**Components:**
- `z_i` = token at position i in context
- `L(z_i)` = next-token prediction loss
- `Φ` = structural metric (e.g., cluster separation)
- `Cov_contexts` = covariance across different input contexts (frozen weights)

**Key Distinction:** CSS approximates Bayesian Influence Functions (BIF) using context variation instead of weight-space sampling (SGLD), making it tractable for 8B parameter models.

### Data Structure: Hierarchical Stochastic Block Model Graphs

**Graph Properties:**
- 3+ levels: Superclusters → Clusters → Nodes
- Nodes labeled with neutral vocabulary ("apple", "truck", "river")
- Edge probabilities: p(intra-cluster) = 0.8, p(inter-cluster) = 0.1
- Random walks generate sequences staying within clusters with occasional "bridges"

**Token Types:**
- **Anchor tokens:** Remain within clusters
- **Bridge tokens:** Cross cluster boundaries (should have high CSS if hypothesis correct)

### Structural Metrics (Φ)

1. **Dirichlet Energy** - Smoothness over graph structure
2. **Cluster Separation** - Inter-cluster vs intra-cluster distance ratio
3. **Representation Coherence** - Embedding alignment with clusters
4. **Layer-Wise Progress** - Structural emergence across depth

### Dimensionality Reduction: Making Structure Visible

**Problem:** LLaMA-3-8B representations are 4096-dimensional

**Solution:** MDS (Multidimensional Scaling) projection to 2D
- Preserves pairwise distances between high-D representations
- Joint embedding across context lengths [5, 10, 20, 50, 100, 200]
- Token trajectories show geometric evolution

**Visualization Types:**
1. **Token Trajectories** - Individual token paths through space
2. **Centroid Trajectories** - Cluster center-of-mass tracking
3. **Branching Diagrams** - Learning progression via color gradient
4. **Interactive Sliders** - Dynamic exploration with perplexity overlays
5. **Animated GIFs** - Cluster formation over time

**Complementarity:**
- **CSS (quantitative):** Which tokens are influential?
- **MDS (qualitative):** What geometric transformation do they cause?

---

## New Experiment: Block Permutation Test

### Research Question

**Does sequential order matter for in-context learning of hierarchical graph structure?**

If ICL learns by tracking coherent graph traversal, then presenting the same tokens in scrambled order should destroy or significantly degrade learned structure.

### Experimental Design

#### Control: Natural Random Walk
```
[A1 A2 A3] [bridge] [B1 B2] [bridge] [C1 C2 C3]
```
Respects graph structure - stays within clusters with occasional transitions

#### Treatment: Block-Permuted Walk
```
[C1 C2 C3] [B1 B2] [A1 A2 A3]
```
Same tokens, blocks shuffled, internal order preserved

**Critical Property:** Both are valid walks (edges within blocks exist in graph), but permutation destroys natural progression of cluster exploration.

### Why This Design is Clean

1. **Same tokens** → Isolates order as causal variable
2. **Valid walks** → Both respect local graph structure
3. **Global structure destroyed** → Cluster transitions scrambled
4. **Bridge tokens lose meaning** → No longer connect established clusters
5. **Quantifiable effect** → Φ_natural / Φ_permuted gives effect size
6. **Falsifiable** → Clear prediction that could be wrong

### Predictions

#### If Order Matters (Expected)

| Context Length | Natural Φ | Permuted Φ | Ratio | Interpretation |
|----------------|-----------|------------|-------|----------------|
| N=10           | ~0.8      | ~0.7       | 1.1× | Too early to see effect |
| N=50           | ~2.0      | ~0.6       | 3.3× | **Order critical** |
| N=100          | ~2.8      | ~0.8       | 3.5× | **Strong effect** |
| N=200          | ~3.2      | ~0.9       | 3.6× | **Very strong** |

**Expected MDS:**
- Natural: Clear cluster separation increasing with N
- Permuted: Random cloud or minimal structure

**Expected Perplexity:**
- Natural: Decreases with N (model learns structure)
- Permuted: Remains high (model confused by unnatural transitions)

#### If Order Doesn't Matter (Alternative Hypothesis)

Both conditions show similar Φ trajectories → Would suggest bag-of-words learning rather than sequential structure tracking. This would **falsify** the sequential learning hypothesis.

### Implementation Details

**Algorithm: Block Identification**
```python
def identify_blocks(node_sequence, graph):
    """
    Find maximal consecutive sequences from same cluster.

    Input:  [A1 A2 A3 B1 B2 A4 A5]
    Output: [(0,3,'A'), (3,5,'B'), (5,7,'A')]
    """
```

**Algorithm: Block Permutation**
```python
def permute_blocks(node_sequence, blocks, seed):
    """
    1. Extract each block as subsequence
    2. Randomly shuffle block order
    3. Concatenate in new order
    Result: Valid walk locally, scrambled globally
    """
```

**Measurements Per Condition:**
1. Cluster separation Φ at N ∈ [10, 20, 50, 100, 200]
2. Per-token perplexity
3. MDS trajectories (optional, for visualization)
4. CSS scores (optional, for detailed influence analysis)

---

## Files Created

### 1. Main Experiment Script
**File:** `run_block_permutation_experiment.py`

**Features:**
- Generates natural walks and block-permuted variants
- Extracts layer representations (default: layer -5)
- Computes cluster separation for both conditions
- Tracks perplexity across conditions
- Saves results to JSON

**Usage:**
```bash
# Quick test (GPT-2, ~5 min)
python run_block_permutation_experiment.py \
    --model gpt2 \
    --context-lengths 10,20,50,100 \
    --n-samples 20 \
    --seed 42

# Full experiment (LLaMA-3.1-8B, ~2 hours)
python run_block_permutation_experiment.py \
    --model meta-llama/Llama-3.1-8B \
    --context-lengths 10,20,30,50,75,100,150,200 \
    --n-samples 50 \
    --output-dir results/block_permutation
```

### 2. Visualization Script
**File:** `plot_block_permutation.py`

Generates comparison plots from saved JSON:
- Φ trajectory comparison (natural vs permuted)
- Perplexity comparison
- Effect size annotations

**Usage:**
```bash
python plot_block_permutation.py results/block_permutation/block_permutation_gpt2.json
```

### 3. Documentation
**File:** `docs/block_permutation_experiment.md`

Complete protocol including:
- Detailed design rationale
- Prediction tables
- Interpretation guide
- Follow-up experiment ideas

### 4. Test Scripts
**Files:**
- `test_block_permutation.py` - Verifies block identification logic
- `run_block_perm_inline.py` - Standalone version for testing

---

## Expected Results & Interpretation

### Strong Effect (Ratio > 3×)
→ **Sequential order is critical for ICL structure learning**
→ Model tracks coherent graph traversal, not just token statistics
→ Supports hypothesis that ICL builds sequential task representations

### Moderate Effect (Ratio 1.5-3×)
→ Order matters but isn't sole driver
→ Some structure learned from token co-occurrence
→ Partial support for sequential hypothesis

### No Effect (Ratio ~ 1×)
→ Order doesn't matter (would **falsify** hypothesis)
→ Model uses bag-of-words or purely statistical learning
→ Would require rethinking ICL mechanism

---

## Follow-Up Experiments (If Block Permutation Shows Strong Effect)

### 1. Reversed Walk
**Design:** D → C → B → A (backward traversal)
**Tests:** Is graph learning directional?

### 2. Within-Block Shuffle
**Design:** Keep block order, shuffle tokens within blocks
**Tests:** Is intra-block order important?

### 3. Controlled Bridge Positions
**Design:** Ensure bridges appear at same positions in both conditions
**Tests:** Is bridge *timing* critical?

### 4. Partial Permutation
**Design:** Permute only first 50% of blocks
**Tests:** How much disruption is needed to destroy learning?

### 5. Two-Graph Design
**Design:** Compare natural walks on two graphs with different edge structures
**Tests:** Does model learn graph structure vs just token patterns?

---

## Connection to Theory

### Park et al. (2024) - "In-Context Learning of Representations"
Block permutation tests whether ICL builds **sequential** task representations or uses statistical token associations.

**Our contribution:** Identifies which specific tokens drive representation geometry through CSS + visualization.

### Lee et al. (2025) - "Influence Dynamics and Stagewise Data Attribution"
If order matters, validates treating context length as temporal analog to training dynamics.

**Our contribution:** Adapts BIF framework to ICL setting using CSS approximation.

### Core ICL Mechanism Test
- **Sequential hypothesis:** Model tracks state through context → Order should matter
- **Statistical hypothesis:** Model aggregates co-occurrence → Order shouldn't matter

Block permutation experiment provides **causal evidence** distinguishing these hypotheses.

---

## Current Status

### ✅ Completed
1. Comprehensive experimental design
2. Implementation of main experiment script with:
   - Block identification algorithm
   - Block permutation algorithm
   - Representation extraction
   - Cluster separation computation
   - Results serialization
3. Visualization script for plotting results
4. Complete documentation
5. Test scripts validating logic

### 🔄 In Progress
- Running the experiment (encountered environment permission issues)
- Alternative: Can run on local machine or compute cluster

### 📋 Next Steps
1. **Execute experiment** with GPT-2 (fast, ~20 min)
2. **Analyze results** - Check if ratio > 3× at N=100
3. **Generate visualizations** - Compare Φ trajectories
4. **If strong effect found:**
   - Run full experiment with LLaMA-3.1-8B
   - Generate MDS trajectories showing geometric difference
   - Compute CSS to identify most influential tokens
   - Write up findings
5. **If weak/no effect:**
   - Revisit hypothesis
   - Consider alternative designs (reversed walk, two-graph)

---

## Key Insights from Discussion

1. **Adversarial alternation (A→B→A→B) is not a valid walk** on the graph structure (probability ~10^-50), so we pivoted to block permutation which preserves local validity.

2. **Block permutation is the cleanest test** because:
   - Same tokens (controls for vocabulary effects)
   - Valid walks (controls for graph structure)
   - Only variable: sequential order

3. **CSS + MDS provides convergent evidence:**
   - CSS: Quantitative influence scores
   - MDS: Qualitative geometric visualization
   - Together: Complete picture of structure learning

4. **Effect size matters more than p-values:**
   - Looking for 3-5× difference in Φ
   - If found, that's a robust, meaningful effect
   - Small differences could be noise

5. **Falsifiability is critical:**
   - Clear prediction that could be wrong
   - If ratio ~ 1×, sequential hypothesis fails
   - Would require rethinking ICL mechanisms

---

## Implementation Notes

**Graph Configuration:**
```python
HierarchicalGraphConfig(
    num_superclusters=3,
    nodes_per_cluster=5,
    p_intra_cluster=0.8,
    p_inter_cluster=0.1,
    seed=42
)
```

**Model Configuration:**
- Primary: GPT-2 (testing), LLaMA-3.1-8B (main experiments)
- Layer: -5 (5th from end, balances compute and representational richness)
- Context lengths: [10, 20, 50, 100, 200]
- Samples per condition: 30-50

**Output Format:**
```json
{
  "natural": {
    "10": [separation_values...],
    "20": [separation_values...],
    ...
  },
  "permuted": {...},
  "natural_perplexity": {...},
  "permuted_perplexity": {...},
  "context_lengths": [10, 20, 50, 100, 200],
  "metadata": {...}
}
```

---

## Potential Issues & Solutions

### Issue: Tokenization Mismatches
**Problem:** Same text string might tokenize differently
**Solution:** Use pre-tokenized sequences, ensure consistency

### Issue: Small Effect Sizes
**Problem:** Difference might be < 1.5×
**Solution:** Increase sample size, test multiple layers, try longer contexts

### Issue: High Variance
**Problem:** Large standard deviations obscure effect
**Solution:** More samples (50+), ensemble across multiple seeds

### Issue: Model-Specific Behavior
**Problem:** Effect might be strong in GPT-2, weak in LLaMA
**Solution:** Test multiple model families, document differences

---

## Expected Timeline (Once Running)

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| GPT-2 test run | 20 min | Initial results, effect size estimate |
| Analysis & plotting | 30 min | Φ trajectory plots, summary statistics |
| LLaMA-3.1-8B full run | 2 hours | High-quality results |
| MDS visualizations | 1 hour | Token trajectory plots |
| CSS analysis | 1 hour | Influential token identification |
| Write-up | 2-3 hours | Complete experimental report |

**Total:** ~7-8 hours for complete analysis

---

## Summary

We developed a **causal test of sequential structure learning in ICL** using block permutation. The experiment isolates order as the critical variable while controlling for token identity and local graph structure.

**Key prediction:** If ICL learns by tracking coherent graph traversal, permutation should reduce cluster separation by 3-5×. This would provide strong evidence that ICL is doing genuine sequential structure learning rather than bag-of-words association.

The implementation is complete and ready to run. Results will either:
1. **Confirm sequential hypothesis** → Strong evidence for mechanistic understanding of ICL
2. **Falsify sequential hypothesis** → Important negative result requiring theory revision

Either outcome advances our understanding of how LLMs learn from context.
