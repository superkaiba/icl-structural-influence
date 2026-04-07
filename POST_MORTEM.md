# Post-Mortem: ICL Structural Influence / Safety Collapse Project

## What the project attempted

This project investigated whether filling an LLM's context window with repetitive or low-diversity tokens causes "representational collapse" (token embeddings converging to a low-dimensional manifold), and whether that collapse degrades the model's ability to refuse harmful prompts. The original theoretical grounding was Context Sensitivity Scores (CSS), approximating Bayesian Influence Functions over frozen-weight context manipulations.

~70K prompt-response pairs were evaluated across ~30 experiments on Qwen3.5 (0.8B-9B), OLMo-3-7B, and Qwen2.5-7B, using Claude Sonnet 4.5 as an LLM judge. Infrastructure includes a star-shaped sweep across context length, model size, context type, and architecture.

---

## Results

### 1. Collapse causes incoherence, not compliance

For Qwen3.5-9B with structured context (baseline-refused prompts only):

| Context Length | Refusal | Compliance | Incoherent |
|---|---|---|---|
| 100 | 85% | 8% | 8% |
| 2,000 | 88% | 6% | 6% |
| 10,000 | 68% | 2% | 23% |
| 50,000 | 6% | 0% | 88% |
| 262,144 | 0% | 0% | 85% |

Compliance never exceeds ~8% for the 9B model. The model doesn't become dangerous; it becomes useless.

### 2. Smaller models are more vulnerable

| Model | Peak Compliance | At Length |
|---|---|---|
| Qwen3.5-0.8B | 20% | 2K tokens |
| Qwen3.5-2B | 3% | 1K tokens |
| Qwen3.5-4B | 9% | 100 tokens |
| Qwen3.5-9B | 7% | 50 tokens |

Even the most vulnerable model (0.8B) only reaches 20% compliance in a narrow window.

### 3. Structure is irrelevant; only vocabulary diversity matters

SBM graph walks and uniform random tokens with matched vocabulary size (15, 50, 200, 1000 words) produce statistically identical safety degradation. Varying intra-cluster edge probability from 0.0 to 0.95 has no effect. Graph structure, Markov properties, and token-level correlations don't matter.

### 4. Vocabulary diversity has a non-monotonic effect

At 2K tokens: very small vocabularies (1-2 tokens) cause incoherence before compliance. Mid-range diversity (~5-200 tokens) is the zone where generation stays coherent but safety may be disrupted. Large vocabularies (1000+) approximate natural text and preserve safety.

### 5. OLMo-3-7B is unusually robust

91% refusal with zero incoherence at 10K tokens, vastly outperforming Qwen3.5-9B (68% refusal, 23% incoherence). Single data point, no explanation.

### 6. Collapse is immediate and stable

Collapse metrics (cos_sim, effective_dim) reach their plateau within ~500 tokens and stay flat through 262K. The progressive behavioral degradation at longer contexts comes from the ratio of collapsed KV entries, not from the collapse geometry intensifying.

### 7. Collapse reversal

Natural language injection reverses collapse (cos_sim drops from 0.97 to 0.40). Same-vocabulary different-structure injection does not. Token identity drives collapse, not latent structure.

### 8. Knowledge retrieval degrades

Accuracy on knowledge questions drops from 97% (500 tokens) to 10% (20K tokens) with structured context. Natural language preserves ~97% accuracy throughout.

---

## Why the project failed

### 1. The core hypothesis was wrong

The project bet that representational collapse would selectively degrade safety while preserving generation quality, creating a "dangerous zone" where models coherently comply with harmful requests. Instead, safety and coherence degrade together. The model can't produce coherent harmful output because it can't produce coherent output at all. This is the expected outcome of feeding garbage into a language model, not a scientific discovery.

### 2. The metrics don't measure the mechanism

cos_sim and effective_dim are computed over a sliding window of the last 50 tokens. At 262K context, this measures the local geometry of tokens 262K-50 to 262K. But the model's attention mechanism sees the entire KV cache. The trajectory experiment proves this disconnect: collapse metrics plateau at 500 tokens, yet safety degrades progressively through 50K. The authors acknowledge the actual mechanism is "KV cache saturation ratio" (the proportion of collapsed entries in the full cache) but never develop a metric for it. The project has no measurement of what it claims is causing the effect.

### 3. The theoretical framework was abandoned

CSS (Context Sensitivity Scores), built on Bayesian Influence Functions, was the original theoretical contribution. It is implemented in `structural_influence.py`. It appears in zero safety experiments. The safety experiments use only cos_sim and effective_dim -- simple summary statistics with no theoretical grounding. The project devolved from "mechanistic theory of how context shapes representations" into "measure cos_sim at different context lengths."

### 4. Critical data is broken or missing

- The most important comparison (structured vs random at vocab=15, the pivot point of the star-shaped design) has a corrupted baseline: `random_tokens_15` has `n_baseline_refused = 0` because every judge call returned `parse_error` due to Qwen3.5's thinking preamble. The Fisher's exact tests at vocab=15 are vacuous.
- T-031, T-033, T-035, T-036 results were lost in a storage migration. T-033 (structure amount sweep) would directly support the structure-irrelevance claim. T-035 (clean baseline) would fix the baseline problem.
- The two headline numbers (61-76% compliance from Qwen2.5 sweep v1, <8% from Qwen3.5 sweep v2) are never reconciled. They differ in model, generation length, and judge methodology simultaneously.

### 5. The null-result claim is underpowered

"Structure doesn't matter" requires adequate statistical power to accept the null. With 50 prompts x 3 trials (~66 observations per cell at ~8% compliance), the minimum detectable effect is ~15-20 percentage points. A real 10% difference between structured and random would be invisible. No power analysis is computed. No confidence intervals appear anywhere. No multiple testing correction is applied to the 32+ Fisher's exact tests in the T-032 analysis.

### 6. Natural language is a confounded control

Structured walks vs. Project Gutenberg text differ in vocabulary size, token-level statistics (uniform vs Zipfian), and semantic coherence. The model was trained on natural text. It handles natural text better because natural text is in-distribution, not necessarily because it avoids "collapse." The distributional mismatch between repetitive tokens and training data is a simpler explanation that doesn't require the collapse framing.

### 7. The project is substantially scooped

- "Context Length Alone Hurts LLM Performance Despite Perfect Retrieval" (Du et al., EMNLP 2025) showed even whitespace padding degrades performance, covering claim 8 more thoroughly.
- "Many-shot Jailbreaking" (Anil et al., NeurIPS 2024) covered long-context safety degradation via semantic demonstrations.
- "Silent Tokens, Loud Effects" (arXiv 2510.01238) demonstrated padding tokens degrade safety and measured representational shifts.
- "When Refusals Fail" (arXiv 2512.02445) analyzed context-length-induced safety failures.

### 8. Too many pivots

The project started as ICL representation theory (velocity, cluster preference, dual-interpretation graphs -- Jan 2026), pivoted to collapse characterization (reversal, attractors, 32K experiments -- Feb 2026), then pivoted to safety (Mar 2026). Each pivot left the previous framework incomplete. The dual-interpretation graph infrastructure is sophisticated engineering for a question (does graph structure matter for safety?) whose answer is boring (no).

### 9. The "attractor dynamics" framing is not rigorous

"Point attractors," "line attractors," and "manifold attractors" are defined by thresholding effective dimension. There is no state space definition, no dynamics function, no Jacobians, no stability analysis. A perturbation test (injecting different content) is not a basin of attraction analysis. This is binning a continuous variable and labeling the bins with dynamical-systems terminology.

### 10. The LLM judge is unvalidated

17% disagreement rate with keyword classification. No human annotation of a random sample. No inter-rater reliability. The judge prompt includes `expected_behavior` and `harm_category`, potentially biasing classifications. The silent `parse_error` corruption of random_tokens_15 demonstrates that judge failures can invalidate entire experiments without any alarm being raised.

---

## What could be salvaged

The structure-irrelevance finding (claim 3) is the most novel result. No prior work systematically varies graph structure vs. vocabulary diversity while measuring safety outcomes. But it requires: fixed baselines at vocab=15, 10x sample size, proper power analysis, confidence intervals, and multiple testing correction. Even then, it's a narrow null result suitable for a short paper or workshop contribution.

The experimental infrastructure (~70K evaluations, sweep framework, LLM judge pipeline, trajectory tracking) is solid engineering that could support a better-designed study.

---

## Lessons

1. Validate your measurement before scaling your experiments. The 50-token window metric was known to be disconnected from the claimed mechanism (KV saturation) by February 2026. The project ran 4 more months of experiments using it anyway.
2. A null result requires more rigor than a positive result, not less. Power analysis should have been done before running experiments, not never.
3. Don't staple a theoretical framework onto empirical work that doesn't use it. CSS is mentioned in the introduction and never appears again.
4. "Garbage in, garbage out" is not a safety finding. The interesting question was whether carefully constructed adversarial context could degrade safety while maintaining coherence. That question was never tested.
