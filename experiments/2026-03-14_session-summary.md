# Session Summary: Context Rot Experiments (2026-03-11 to 2026-03-14)

## Project
Research project investigating **context rot** — how low-diversity context tokens cause representational collapse in LLMs, which degrades safety guardrails.

## What Was Done

### 1. Safety Collapse Sweep (completed)
Full sweep across model sizes, architectures, and vocab sizes. Results in `results/safety_collapse_sweep/`.

**Completed experiments (all with LLM judge):**
- Qwen2.5: 0.5B, 3B, 7B (original), 14B
- Llama-3.1-8B (no baseline safety — 0/50 refused)
- Qwen3-8B base (only 1/50 refused — weak safety)
- Vocab sizes: 15, 50, 200, 1000 (up to 100K tokens)
- Context granularity fill-in for 7B

**Failed:** Qwen2.5-72B and Llama-3.3-70B (OOM on 2x A40 GPUs). **Qwen3.5-9B** failed because `transformers` is too old — needs `pip install --upgrade transformers` or install from source.

### 2. Five New Context Type Experiments (completed)
Results in `results/new_context_types/`. All judged.

| Experiment | Finding |
|-----------|---------|
| **Repeated token** (" the") | Total collapse by 5K, 100% degradation |
| **Shuffled books** | Safety preserved (grammar irrelevant, vocab diversity protects) |
| **Jailbreak stacking** | Collapse + simple jailbreak = 49% degradation at 15K, 92% at 20K |
| **Persona injection** | 88-100% degradation at 20K+ (near-zero refusal) |
| **Code (Python/JSON)** | No collapse — real code is diverse enough |

### 3. vLLM Hybrid Backend (implemented + tested)
Added `--use-vllm` flag to `run_safety_collapse_experiment.py`:
- Phase 1: HuggingFace processes context + extracts collapse metrics via hooks
- Phase 2: vLLM with prefix caching batch-evaluates all 50 safety prompts
- Eliminates 50 KV cache deep copies per trial
- Also wired into `run_safety_sweep.py` via `--use-vllm` flag
- vLLM 0.17.1 installed, tested with Qwen2.5-0.5B

### 4. Performance Optimizations
In `run_safety_collapse_experiment.py`:
- Removed second `deep_copy_kv_cache()` in `evaluate_safety_prompt()`
- Added optimized `process_context_chunks()` with batched GPU→CPU transfers
- Tensor creation directly on device

### 5. New Features Added
- `--injection` flag: `none` (default), `jailbreak`, `persona` — appends injection text between context and safety prompt
- New context types: `shuffled_books`, `code_python`, `code_json` in `generate_context_tokens()`
- `repeated_token` already existed

### 6. Plots (all with 3-band stacked bars)
All plots use LLM judge classifications: green=refusal, gray=incoherent, red=genuine compliance.

**New context experiments** in `results/new_context_types/plots/`:
- `context_type_comparison.png` — one row per context type
- `jailbreak_amplification.png` — with/without jailbreak (hatched)
- `persona_injection.png` — with/without persona (hatched)
- `collapse_vs_degradation.png` — scatter of cos_sim vs compliance rate
- `cos_sim_by_context.png` — collapse metric by context type

**Sweep comparison** in `results/safety_collapse_sweep/comparison_plots/`:
- `model_size_stacked.png` — 0.5B/3B/7B/14B
- `architecture_stacked.png` — Qwen vs Llama
- `vocab_size_stacked.png` — vocab 15/50/200/1000 up to 100K
- `phase_transition_heatmap.png` — genuine compliance rate by model x length

## What's Left / Next Steps
1. **Upgrade transformers** to support Qwen3.5, then run Qwen3.5-9B
2. **72B/70B models** — need larger GPU or quantized models
3. Future experiments in plan file (`/root/.claude/plans/refactored-sleeping-torvalds.md`)

## Key Files
- Main experiment: `experiments/core/run_safety_collapse_experiment.py`
- Sweep orchestrator: `experiments/core/run_safety_sweep.py`
- LLM judge: `experiments/core/run_llm_judge_safety.py`
- Plot scripts: `experiments/plotting/plot_new_context_experiments.py`, `experiments/plotting/plot_safety_sweep_comparison.py`
- Plan: `/root/.claude/plans/refactored-sleeping-torvalds.md`
- Memory: `/root/.claude/projects/-workspace-research-projects-in-context-representation-influence/memory/MEMORY.md`
- HF cache: `/workspace/.hf_cache` (set via `HF_HOME` in `.bashrc` and `.zshrc`)

## Key Results Summary

### Context Types That Cause Collapse
- Structured random walks (vocab=15): cos_sim > 0.94 at 10K+
- Repeated single token: cos_sim = 1.0 immediately
- Template small vocab (~50 words): moderate collapse

### Context Types That Do NOT Cause Collapse
- Natural book text: cos_sim ~0.30
- Shuffled book text: cos_sim ~0.36 (grammar irrelevant)
- Python code: cos_sim ~0.47
- JSON: cos_sim ~0.53
- Larger vocab walks (200+): cos_sim high but safety preserved

### Safety Impact
- **Danger zone is 10K-20K tokens** of structured walk on Qwen2.5-7B
- At 20K: 48% genuine harmful compliance (LLM judge confirmed)
- At 30K+: model goes 100% incoherent (broken, not jailbroken)
- Natural text preserves safety at all lengths tested (up to 100K)

### Cross-Model Findings
- 7B is most vulnerable (sharp cliff at 10-20K)
- 0.5B has low persistent ~15% compliance at all lengths
- 3B mostly safe (10-16% compliance at long contexts)
- 14B shows sustained 20-29% compliance from 25K-50K
- Llama-3.1-8B has no baseline safety to degrade

### Injection Attacks
- Jailbreak + 15K collapse: 49% degradation (vs ~15% without jailbreak)
- Persona injection + 20K collapse: 88-100% degradation
- Neither jailbreak nor persona has any effect without collapse (natural books)
