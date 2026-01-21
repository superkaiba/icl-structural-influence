# Alternative Influence Metrics for LOO Experiment

## Current Metric

```python
Ratio = mean_L2(cross_cluster_pairs) / mean_L2(within_cluster_pairs)
Influence(pos) = Ratio(full_context) - Ratio(context_without_pos)
```

**Limitations**:
1. Ratio conflates numerator and denominator effects
2. Unstable when denominator is small
3. Doesn't directly measure semantic conflict (cat↔dog)
4. Treats all pairs equally (no weighting by semantic similarity)

---

## Category 1: Direct Distance Metrics

### 1A. Separate Cross/Within Cluster Distances

Instead of ratio, measure each component separately:

```python
Influence_cross(pos) = mean_L2_cross(full) - mean_L2_cross(without_pos)
Influence_within(pos) = mean_L2_within(full) - mean_L2_within(without_pos)
```

**Advantage**: Disentangles the two effects
**Interpretation**:
- Positive cross-influence = removing token decreases cross-cluster separation
- Positive within-influence = removing token decreases within-cluster cohesion

### 1B. Semantic Pair Distance (Direct Conflict Measure)

For semantic tokens, directly measure the cat↔dog type pairs:

```python
Semantic_conflict(pos) = L2(cat, dog | full) - L2(cat, dog | without_pos)
```

**Advantage**: Directly measures the semantic conflict we care about
**Interpretation**: Positive = removing token brings cat/dog closer (undoes structure learning)

### 1C. Cosine Similarity Instead of L2

```python
Influence(pos) = cos_sim(cat, dog | full) - cos_sim(cat, dog | without_pos)
```

**Advantage**: Scale-invariant, often more stable than L2

---

## Category 2: Cluster Quality Metrics

### 2A. Silhouette Score

Measures how well-separated clusters are:

```python
silhouette(token) = (b - a) / max(a, b)
# a = mean distance to same-cluster tokens
# b = mean distance to nearest other cluster

Influence(pos) = mean_silhouette(full) - mean_silhouette(without_pos)
```

**Advantage**: Standard clustering metric, interpretable (-1 to 1 scale)

### 2B. Linear Separability (Probing)

Train a linear classifier to predict cluster from representation:

```python
accuracy = train_linear_probe(representations, cluster_labels)
Influence(pos) = accuracy(full) - accuracy(without_pos)
```

**Advantage**: Measures whether clusters are linearly separable
**Note**: Requires held-out test set or cross-validation

### 2C. Davies-Bouldin Index

```python
DB = mean over clusters of (within_scatter_i + within_scatter_j) / between_center_dist_ij
# Lower is better (tighter, more separated clusters)

Influence(pos) = DB(without_pos) - DB(full)  # Note: reversed because lower is better
```

---

## Category 3: Prediction-Based Metrics

### 3A. Next-Token Prediction Probability

If graph structure is learned, transitions should follow graph edges:

```python
# P(next_token | context) should be higher for graph neighbors
graph_prob = mean(P(neighbor | context) for neighbor in graph_neighbors)
Influence(pos) = graph_prob(full) - graph_prob(without_pos)
```

**Advantage**: Uses the model's actual prediction, not just representations

### 3B. Perplexity on Graph-Consistent Continuations

```python
# Generate continuations that follow graph structure
perplexity = model_perplexity(graph_consistent_continuation | context)
Influence(pos) = perplexity(without_pos) - perplexity(full)  # Lower perplexity is better
```

### 3C. Analogy Completion

Test if model learned relational structure:

```python
# cat:computer :: dog:?  (should be television if graph learned)
# Measure: P(television | "cat is to computer as dog is to")
analogy_score = P(correct_analogy_completion | context)
Influence(pos) = analogy_score(full) - analogy_score(without_pos)
```

---

## Category 4: Gradient-Based Influence

### 4A. Influence Functions (Koh & Liang 2017)

Approximate the effect of removing a training point:

```python
Influence(pos) = -∇_θ L(test) · H^(-1) · ∇_θ L(pos)
# H = Hessian of loss
# Requires computing/approximating inverse Hessian
```

**Advantage**: Principled, theoretically grounded
**Disadvantage**: Computationally expensive, approximations needed

### 4B. Gradient Dot Product (Simplified)

```python
# How aligned is the gradient from position i with the overall learning direction?
Influence(pos) = ∇_θ L(test) · ∇_θ L(pos)
```

**Advantage**: Much faster than full influence functions

### 4C. Representation Gradient

```python
# How much does the representation change when we perturb this position?
Influence(pos) = ||∂h_test / ∂x_pos||
```

---

## Category 5: Attention-Based Metrics

### 5A. Attention Weight to Position

```python
# How much attention does the model pay to position i?
Influence(pos) = mean_attention_weight_to_pos(pos)
```

**Advantage**: Directly interpretable, shows what model "looks at"

### 5B. Attention Entropy Change

```python
# Does removing this position change how attention is distributed?
entropy(attn) = -sum(attn * log(attn))
Influence(pos) = entropy(full) - entropy(without_pos)
```

### 5C. Attention Pattern Similarity

```python
# Does the attention pattern match graph structure?
graph_attention_alignment = correlation(attention_weights, adjacency_matrix)
Influence(pos) = alignment(full) - alignment(without_pos)
```

---

## Category 6: Information-Theoretic Metrics

### 6A. Mutual Information: Representation ↔ Cluster

```python
MI(representations; cluster_labels)
Influence(pos) = MI(full) - MI(without_pos)
```

**Advantage**: Captures non-linear relationships
**Disadvantage**: MI estimation is tricky in high dimensions

### 6B. KL Divergence Between Cluster Distributions

```python
# How different are the representation distributions of different clusters?
KL(P_cluster0 || P_cluster1)
Influence(pos) = KL(full) - KL(without_pos)
```

---

## Recommended Metrics for This Experiment

| Priority | Metric | Rationale |
|----------|--------|-----------|
| **High** | Separate cross/within distances (1A) | Disentangles current ratio |
| **High** | Semantic pair distance (1B) | Directly measures cat↔dog conflict |
| **High** | Silhouette score (2A) | Standard, interpretable |
| **Medium** | Linear probe accuracy (2B) | Tests linear separability |
| **Medium** | Next-token graph probability (3A) | Uses actual model predictions |
| **Low** | Gradient-based (4A/4B) | Principled but expensive |
| **Low** | Attention-based (5A) | Interpretable but indirect |

---

## Implementation Sketch

```python
def compute_alternative_influences(model, tokenizer, context_tokens, graph, layer_idx, pos):
    """Compute multiple influence metrics for a single position."""

    full_reps = get_representations(context_tokens, layer_idx)
    loo_reps = get_representations(context_tokens[:pos] + context_tokens[pos+1:], layer_idx)

    influences = {}

    # 1A: Separate distances
    influences['cross_cluster_dist'] = (
        mean_cross_cluster_dist(full_reps) - mean_cross_cluster_dist(loo_reps)
    )
    influences['within_cluster_dist'] = (
        mean_within_cluster_dist(full_reps) - mean_within_cluster_dist(loo_reps)
    )

    # 1B: Semantic pair distance (e.g., cat-dog)
    for t1, t2 in semantic_pairs:
        key = f'{t1}_{t2}_dist'
        influences[key] = L2(full_reps[t1], full_reps[t2]) - L2(loo_reps[t1], loo_reps[t2])

    # 2A: Silhouette score
    influences['silhouette'] = (
        compute_silhouette(full_reps, cluster_labels) -
        compute_silhouette(loo_reps, cluster_labels)
    )

    # 1C: Cosine similarity for semantic pairs
    for t1, t2 in semantic_pairs:
        key = f'{t1}_{t2}_cosine'
        influences[key] = (
            cosine_sim(full_reps[t1], full_reps[t2]) -
            cosine_sim(loo_reps[t1], loo_reps[t2])
        )

    return influences
```
