# Experiment Plan: Replicating Lee et al. (2025) Findings in ICL Setting

## Paper Summary: "Influence Dynamics and Stagewise Data Attribution"

**Core Claim**: Influence is not static - it changes non-monotonically during learning, with sign flips and peaks at phase transitions.

**Key Formula (BIF)**:
```
BIF(z_i, φ) = -Cov(ℓ_i(w), φ(w))
```

**Their Setup**:
- Track influence across **training time** (checkpoints)
- Found non-monotonic patterns, sign flips at ~30k steps
- Validated on Pythia 14M with induction circuit formation

**Our Analog**:
- Track influence across **context length** (N) - ICL's equivalent of "learning"
- Our CSS approximates BIF: `CSS(z_i, Φ) = -Cov_contexts(L(z_i), Φ)`

---

## Experiments to Run

### Experiment 1: Non-Monotonic Influence Detection

**Goal**: Find non-monotonic CSS patterns across context length

**Method**:
1. Compute position-wise CSS at each context length N = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]
2. Track specific positions (e.g., position 5, 10, 15) across all N
3. Look for:
   - Peaks followed by decreases
   - Sign flips (positive → negative or vice versa)
   - Sharp changes at specific N thresholds

**Expected Finding**: CSS at early positions may peak then decrease as model "saturates" on structure

---

### Experiment 2: Sign Flip Analysis

**Goal**: Identify token positions where influence sign flips

**Method**:
1. For each position i, compute CSS(i) at each context length
2. Track sign of CSS across N
3. Identify positions where sign(CSS) changes
4. Correlate with token role (bridge vs. anchor)

**Analysis**:
```python
sign_flips[position] = [N where sign(CSS(N)) != sign(CSS(N-1))]
```

**Expected Finding**: Bridge tokens (crossing clusters) may show sign flips as model learns global→local structure

---

### Experiment 3: Hierarchical Influence Decomposition

**Goal**: Track within-cluster vs. between-cluster influence separately (like Lee et al.'s semantic hierarchy)

**Method**:
1. Decompose CSS into:
   - `CSS_within`: Sensitivity when token stays in same cluster
   - `CSS_between`: Sensitivity when token crosses clusters
2. Track ratio `CSS_between / CSS_within` across N

**Expected Finding**:
- Early N: High between-cluster influence (learning global structure)
- Late N: High within-cluster influence (refining local structure)
- Crossover point = phase transition

---

### Experiment 4: Bridge Token Analysis

**Goal**: Analyze influence dynamics of cluster-crossing tokens specifically

**Method**:
1. Identify "bridge positions" where cluster changes
2. Compute CSS specifically for bridge tokens
3. Compare to non-bridge tokens
4. Track across context length

**Expected Finding**: Bridge tokens show:
- Higher magnitude CSS (structurally important)
- More sign flips (role changes during learning)
- Peaks at phase transitions

---

### Experiment 5: Layer-wise Influence Dynamics

**Goal**: Track when different layers show structural influence

**Method**:
1. Compute CSS at layers [0, L/4, L/2, 3L/4, L-1]
2. For each layer, track CSS across context length
3. Identify which layers show:
   - Earlier structural emergence
   - Stronger phase transitions
   - More sign flips

**Expected Finding**:
- Deep layers: Earlier/stronger structural influence
- Early layers: Retain pretraining semantics longer
- (Matches Park et al. finding)

---

### Experiment 6: Influence Peak Detection

**Goal**: Automatically detect phase transition points via influence peaks

**Method**:
1. Compute total influence magnitude: `|CSS|_total = Σ_i |CSS(i)|`
2. Compute rate of change: `d|CSS|/dN`
3. Find peaks in rate of change
4. Compare to structural metric peaks (Φ, E)

**Expected Finding**: Influence peaks slightly precede structural metric changes (influence → structure)

---

### Experiment 7: Token Class Influence Matrix

**Goal**: Build influence matrix between token classes (like Lee et al.'s delimiter analysis)

**Method**:
1. Define token classes:
   - Cluster A tokens, Cluster B tokens, Cluster C tokens
   - Bridge tokens (A→B, B→C, etc.)
   - Position classes (early, middle, late)
2. Compute mean CSS for each class pair
3. Track how class-class influence changes with N

**Expected Finding**:
- Same-cluster pairs: Positive influence (cooperation)
- Cross-cluster pairs: May flip from positive to negative

---

## Implementation Priority

| Priority | Experiment | Effort | Expected Impact |
|----------|------------|--------|-----------------|
| 1 | Non-Monotonic Detection | Medium | High - Core finding |
| 2 | Sign Flip Analysis | Low | High - Direct Lee analog |
| 3 | Hierarchical Decomposition | Medium | High - Novel contribution |
| 4 | Bridge Token Analysis | Low | Medium - Mechanistic insight |
| 5 | Layer-wise Dynamics | Medium | Medium - Depth analysis |
| 6 | Peak Detection | Low | Medium - Automation |
| 7 | Token Class Matrix | High | High - Rich visualization |

---

## Key Differences from Lee et al.

| Aspect | Lee et al. | Our Approach |
|--------|-----------|--------------|
| Learning axis | Training time | Context length |
| Weight sampling | SGLD on posterior | Fixed weights, vary context |
| Influence type | True BIF | CSS approximation |
| Task | Next-token prediction | Graph tracing |
| Structure | Semantic hierarchy | Graph clusters |

---

## Success Criteria

We successfully replicate Lee et al.'s findings if we observe:

1. ✓ **Non-monotonic CSS** - influence doesn't just increase/decrease
2. ✓ **Sign flips** - at least some positions flip sign
3. ✓ **Peaks at transitions** - CSS peaks correlate with Φ/E changes
4. ✓ **Stagewise pattern** - global structure before local
5. ✓ **Layer differences** - deep layers show earlier emergence

---

## Expected Novel Contributions

Beyond replicating Lee et al., our ICL setting offers:

1. **No training required** - study influence dynamics with frozen weights
2. **Faster iteration** - context length varies in seconds, not hours
3. **Cleaner control** - exact graph structure known
4. **Bridge to Park et al.** - connects influence theory to representation geometry
