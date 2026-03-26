# TASKS.md

Active task tracker for this research project. Read at session start, update at session end.

**Last updated**: 2026-03-26 by claude

---

## Active Tasks

### T-030: Small vocab random token experiments (vocab 2-12) [DONE 2026-03-26]
- **Status**: done
- **Results**: `results/safety_collapse_sweep_v2/context_type/random_tokens_{2,3,5,8,10,12}/`
- **Summary**: Vocab 2-3 collapse too hard for coherent compliance (3-9% degradation).
  Vocab 8-15 is the danger zone (59-93% degradation). Non-monotonic: extreme collapse
  causes incoherence, not compliance.

### T-031: Least probable token context
- **Status**: planned
- **Priority**: P0
- **Depends on**: none
- **Model**: Qwen/Qwen3.5-9B
- **Description**: At each position, pick the LLM's least probable next token and use that
  as context. Tests whether adversarially improbable sequences cause worse collapse/safety
  degradation than random tokens. Needs a new context type in `generate_context_tokens()`.

### T-032: Analyze structured vs random walk difference
- **Status**: planned
- **Priority**: P1
- **Depends on**: T-030
- **Model**: Qwen/Qwen3.5-9B
- **Description**: Plot 03 shows structured walks and random tokens produce nearly identical
  safety collapse. Develop metrics/analysis to understand why. Compare representation
  geometry (linear dim, nonlinear dim, cos_sim) between the two conditions at matched
  vocab sizes.

### T-033: Vary amount of structure in dataset
- **Status**: planned
- **Priority**: P1
- **Depends on**: T-032
- **Model**: Qwen/Qwen3.5-9B
- **Description**: Parametrically control structure level — interpolate between fully
  structured SBM walk (p_intra=0.8, p_inter=0.15) and fully random. E.g., vary
  p_intra from 0.0 to 1.0 while keeping vocab fixed. Measure how structure amount
  affects collapse metrics and safety.

### T-034: Track representation geometry over context length
- **Status**: planned
- **Priority**: P1
- **Depends on**: none
- **Model**: Qwen/Qwen3.5-9B
- **Description**: Measure linear dimension, nonlinear dimension, and cosine similarity
  of token representations as context grows. Use the existing collapse_metrics
  infrastructure but extract finer-grained trajectories (measure at every 100 tokens,
  not just at the end of context). Shows how the representation space evolves.

### T-035: Get clean baseline refusal/compliance rates
- **Status**: planned
- **Priority**: P1
- **Depends on**: none
- **Model**: Qwen/Qwen3.5-9B
- **Description**: Run baseline audit (no context) with LLM judge classification to
  get the reference refusal/compliance/incoherent rates. Current data has baselines
  embedded in each experiment but worth a clean standalone measurement.
- **Notes**: Baseline data exists in each experiment's `baseline_audit.json` but
  uses keyword classifier first, then LLM judge override. May want judge-only baseline.

### T-036: Rerun Qwen3.5-27B with max_new_tokens=500
- **Status**: planned
- **Priority**: P2
- **Depends on**: T-030 (GPU availability)
- **Model**: Qwen/Qwen3.5-27B
- **GPU**: ~7h (needs both GPUs, device_map=auto)
- **Script**: `experiments/core/run_safety_sweep.py --sweep model_size --include model_size/qwen35_27b`
- **Results**: `results/safety_collapse_sweep_v2/model_size/qwen35_27b/`
- **Description**: The 27B experiment from the first run used max_new_tokens=100
  (truncated thinking preamble). Needs rerun with 500 tokens like the other models.

### T-037: Paper figure generation
- **Status**: planned
- **Priority**: P2
- **Depends on**: T-030, T-031, T-032
- **Description**: Assemble publication-quality figures from sweep v2 results.
  Current plots are good working drafts but need polish for the paper.

---

## Completed (recent)

### T-029: Safety collapse sweep v2 — rerun with max_new_tokens=500 [DONE 2026-03-25]
- **Results**: `results/safety_collapse_sweep_v2/` (16/19 experiments, 27B pending)
- **Summary**: Fixed truncated thinking preamble artifact. Now correctly shows compliance
  at 100-10K tokens transitioning to incoherence at 50K+. All Sweep B (0.8B-4B) and
  Sweep C (12 context types) complete with judge results.

### T-028: Safety collapse sweep v2 — first run [DONE 2026-03-21]
- **Results**: `results/safety_collapse_sweep_v2/`
- **Summary**: 19 experiments, ~53K evaluations across 4 sweep dimensions. Discovered
  max_new_tokens=100 caused false incoherence from Qwen3.5's thinking preamble.

### T-027: Implement sweep v2 code changes [DONE 2026-03-18]
- **Results**: code changes to `run_safety_collapse_experiment.py`, `run_safety_sweep.py`,
  `run_llm_judge_safety.py`, `plot_comprehensive_sweep.py`, `hooked_model.py`
- **Summary**: Added parameterized context types (structured_walk_N, random_tokens_N,
  lorem_ipsum), OLMo 3 support, GPU pinning, async judge submission, experiment
  deduplication, --enable-thinking flag.

### T-026: Sweep v1 comprehensive results [DONE 2026-03-14]
- **Results**: `results/safety_collapse_sweep/`
- **Log**: `experiments/2026-03-14_session-summary.md`
- **Summary**: Qwen2.5 family (0.5B-72B), Llama-3.3-70B, Qwen3-8B, Qwen3.5 family.
  Key finding: 7B most vulnerable (48% compliance at 20K). OLMo-3-7B most robust.

---

## Archive

<!-- T-001 through T-025: Initial collapse experiments, velocity, preference, reversal,
     probing, context evolution, natural repetition. See experiments/experiment_log.md -->
