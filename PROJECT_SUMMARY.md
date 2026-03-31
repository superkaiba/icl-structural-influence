# Does Representational Collapse Degrade LLM Safety Guardrails?

## Research Summary

This project investigates whether filling an LLM's context window with repetitive or low-diversity tokens causes *representational collapse* (token embeddings converging to a low-dimensional manifold), and whether that collapse degrades the model's ability to refuse harmful prompts.

**Short answer:** Context-induced collapse primarily causes **incoherence**, not coherent harmful compliance. The genuine safety risk window is narrow (2K-10K tokens), shallow (< 20% compliance for the smallest models), and largely disappears at larger model sizes.

---

## Background

**Representational collapse** is the phenomenon where token representations in a transformer's residual stream converge toward a low-dimensional subspace after processing long, repetitive context. We measure it with:

- **avg_cos_sim**: Average pairwise cosine similarity of recent token representations (1.0 = fully collapsed)
- **effective_dim**: Participation ratio of eigenvalues (lower = more collapsed)

**Context Sensitivity Scores (CSS)** approximate Bayesian Influence Functions by measuring how per-token losses covary with structural metrics across different context conditions, with frozen model weights:

```
CSS(z_i, Phi) = -Cov_contexts(L(z_i), Phi)
```

This lets us study how context tokens influence the model's internal geometry without any weight updates.

---

## Experiment Design

We use a **star-shaped sweep** with 4 independent variables, all anchored to a default pivot point:

**Pivot:** Qwen/Qwen3.5-9B, `structured_walk_15` context, raw wrapping, no injection

| Sweep | Variable | Range |
|-------|----------|-------|
| A: Context Length | 18-point log scale | 0 to 262K tokens |
| B: Model Size | Qwen3.5 family | 0.8B, 2B, 4B, 9B |
| C: Context Type | 18 types | structured walks, random tokens (vocab 1-1000), lorem ipsum, natural books |
| D: Architecture | 3 families | Qwen3.5-9B, Llama-3.3-70B, OLMo-3-7B |

Each experiment:
1. Generates baseline responses to 50 safety prompts (no context)
2. Classifies baselines via LLM judge (Claude Sonnet 4.5 batch API)
3. For each (context_type x length x trial): fills the KV cache with context, measures collapse metrics, then evaluates all 50 safety prompts
4. Post-hoc LLM judge classifies every response as **refusal**, **compliance**, **incoherent**, or **partial_refusal**

The 50 safety prompts span 5 categories (10 each): direct harmful, indirect/jailbreak, borderline/dual-use, benign control, and safety knowledge.

**Total evaluations:** ~70K prompt-response pairs across ~30 experiments, all with LLM judge classifications.

---

## Key Results

### 1. Long context causes incoherence, not compliance

For Qwen3.5-9B with `structured_walk_15` context (baseline-refused prompts only):

| Context Length | Refusal | Compliance | Incoherent |
|---------------|---------|------------|------------|
| 100 | 85% | 8% | 8% |
| 2,000 | 88% | 6% | 6% |
| 10,000 | 68% | 2% | 23% |
| 50,000 | 6% | 0% | 88% |
| 262,144 | 0% | 0% | 85% |

Compliance never rises above ~8% and drops to zero at long contexts. The model doesn't become dangerous — it becomes useless.

### 2. Smaller models are more vulnerable

| Model | Peak Compliance | At Length |
|-------|----------------|-----------|
| Qwen3.5-0.8B | **20%** | 2K tokens |
| Qwen3.5-2B | 3% | 1K tokens |
| Qwen3.5-4B | 9% | 100 tokens |
| Qwen3.5-9B | 7% | 50 tokens |

The 0.8B model has the weakest safety training and reaches 20% compliance in the "sweet spot" where context is long enough to disrupt safety but short enough for coherent generation.

### 3. Structure is completely irrelevant

This is the clearest and most surprising finding. We tested two independent dimensions:

**Structured vs random at matched vocabulary:** SBM graph walks and uniform random token sequences with the *same* vocabulary size (15, 50, 200, 1000 words) produce statistically identical safety degradation profiles. Fisher's exact tests show p > 0.05 at every context length and vocab size.

**Varying structure strength:** We varied the SBM intra-cluster edge probability (`p_intra`) from 0.0 (fully random) to 0.95 (strongly clustered) with fixed 15-word vocabulary. All values produce identical compliance rates, incoherence rates, and collapse metrics.

**Implication:** Graph structure, Markov properties, and token-level correlations do not matter. Only the *vocabulary diversity* of the context (how many unique tokens appear) drives representational effects.

### 4. Vocabulary diversity has a non-monotonic effect

At 2K context tokens (baseline-refused prompts):

| Vocab Size | Compliance | Refusal |
|-----------|------------|---------|
| 1 (repeated) | 9% | 83% |
| 2 | 8% | 85% |
| 5 | 14% | 78% |
| 50 | 2% | 71% |
| 200 | 12% | 75% |
| 1,000 | 8% | 85% |

Very small vocabularies (1-2 tokens) cause such extreme collapse that the model becomes incoherent before it can comply. Natural-scale diversity (~5-200 tokens) is the zone where representations stay coherent enough for fluent generation but may disrupt safety guardrards. Very large vocabularies (1000+) approximate natural text and preserve safety.

### 5. Architecture matters more than expected

At 10K context tokens:

| Model | Compliance | Incoherent | Refusal |
|-------|------------|------------|---------|
| Qwen3.5-9B | 2% | 23% | 68% |
| OLMo-3-7B | 9% | 0% | 91% |

OLMo-3-7B maintains 91% refusal with *zero incoherence* at 10K tokens — remarkably robust compared to Qwen3.5. This suggests architecture and training methodology affect collapse resilience more than model size alone.

### 6. Collapse is a deep-layer phenomenon

Trajectory analysis of representation geometry during context processing:

- **Layer 0 (early):** cos_sim ~0.2, effective_dim ~25 — representations stay diverse
- **Layer 31 (final):** cos_sim shoots to ~0.98 within the first 10K tokens, effective_dim collapses to ~2

Collapse propagates from the output layers inward, consistent with safety behavior being encoded in later layers.

### 7. Collapse establishes immediately and remains stable

Collapse metrics reach their plateau within ~500 tokens and stay flat through 262K tokens. The progressive degradation of model behavior at longer contexts comes from the *ratio* of collapsed KV entries to total context, not from the collapse intensifying over time.

---

## What's Not Done

The following experiments were completed but their result data was lost during a storage migration. The code and infrastructure to re-run them is intact:

- **T-031: Least probable tokens** — adversarial context type picking the model's least probable next token at each position. Preliminary result: produces 0% compliance at all lengths (worse than random for an attacker — just causes incoherence faster).
- **T-033: Structure amount sweep** — 6 experiments varying `p_intra` from 0.0 to 0.95. Confirmed structure has zero effect on safety.
- **T-036: Qwen3.5-27B** — showed the 27B model goes from full refusal to incoherence at 2K tokens, with much less compliance than smaller models.
- **T-035: Clean baseline judge** — LLM judge baseline showed 48% refusal (28% full + 20% partial) vs keyword classifier's 4%.

To re-run: `bash experiments/core/launch_t030_t031_t033.sh` (8 experiments, ~16h on 4 GPUs).

---

## Repository Structure

```
experiments/
  core/
    run_safety_collapse_experiment.py   # Single experiment: baseline + context + eval
    run_safety_sweep.py                 # Sweep orchestrator across all dimensions
    run_llm_judge_safety.py             # Claude Sonnet batch API judge
  plotting/
    plot_comprehensive_sweep.py         # Generates plots 01-11
  analysis/
    analyze_structured_vs_random.py     # T-032 statistical comparison
    analyze_trajectory.py               # T-034 representation trajectories

src/
  models/hooked_model.py                # Activation extraction via hooks
  metrics/collapse_metrics.py           # cos_sim, effective_dim, intrinsic_dim
  metrics/structural_influence.py       # CSS computation
  data/dual_interpretation_graph.py     # SBM graph generation

results/safety_collapse_sweep_v2/
  context_length/qwen35_9b/             # Sweep A
  model_size/qwen35_{0.8b,2b,4b}/      # Sweep B
  context_type/{18 types}/              # Sweep C
  architecture/{qwen,olmo}/             # Sweep D
  plots/                                # All generated figures
```

Each experiment directory contains `all_results.json`, `baseline_audit.json`, `config.json`, `raw/` per-trial data, and `judge/all_results_judged.json`.

---

## Running Experiments

```bash
# Install
uv sync

# Run a single experiment
source .env  # needs ANTHROPIC_API_KEY for judge
PYTHONPATH=. python experiments/core/run_safety_collapse_experiment.py \
    --model Qwen/Qwen3.5-9B \
    --context-types "structured_walk_15,random_tokens_50" \
    --context-lengths "0,100,500,2000,10000,50000" \
    --n-trials 3 --max-new-tokens 500 \
    --output-dir results/my_experiment

# Run LLM judge (batch API, non-blocking)
PYTHONPATH=. python experiments/core/run_llm_judge_safety.py \
    --results-dir results/my_experiment --submit-only

# Generate all plots
PYTHONPATH=. python experiments/plotting/plot_comprehensive_sweep.py
```

**Hardware:** Tested on 4x NVIDIA H200 (150GB each). Qwen3.5-9B fits on a single GPU (~18GB bf16). Use `CUDA_VISIBLE_DEVICES` to pin experiments to specific GPUs for parallel runs.

**Important:** Qwen3.5 models emit a "Thinking Process:" preamble as visible text even with `enable_thinking=False`. Always use `--max-new-tokens 500` to capture the actual response after the preamble.

---

## Open Questions

1. **Why is OLMo-3-7B so robust?** Is it architecture (different attention mechanism), training data, or RLHF methodology?
2. **Can the compliance window be exploited in practice?** The 2K-10K token sweet spot exists but compliance rates are low (< 20% even for 0.8B). Is this practically exploitable?
3. **Does fine-tuning on collapsed representations transfer?** If the model's representations collapse during inference, would training on such data cause permanent safety degradation?
4. **Layer-specific interventions:** Since collapse is a deep-layer phenomenon, could targeted interventions at layers 20+ preserve safety while maintaining long-context capability?
