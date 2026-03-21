#!/usr/bin/env python3
"""
Safety Collapse Sweep v2: Star-shaped experiment design.

Fix all variables at defaults, sweep one axis at a time:
  A: Context Length (default model + default context type, full 18-point range)
  B: Model Size (Qwen3.5 family, 0.8B to 27B)
  C: Context Type (14 types on default model, 8 representative lengths)
  D: Model Architecture (Qwen3.5-9B, Llama-3.3-70B, OLMo-3-7B)

Orchestrates calls to run_safety_collapse_experiment.py and run_llm_judge_safety.py.
Resumable — skips experiments that already have all_results.json.

Usage:
    # Run a specific sweep dimension
    PYTHONPATH=. python experiments/core/run_safety_sweep.py --sweep context_length
    PYTHONPATH=. python experiments/core/run_safety_sweep.py --sweep model_size
    PYTHONPATH=. python experiments/core/run_safety_sweep.py --sweep context_type
    PYTHONPATH=. python experiments/core/run_safety_sweep.py --sweep architecture

    # Run everything
    PYTHONPATH=. python experiments/core/run_safety_sweep.py --sweep all

    # Dry run (show commands without executing)
    PYTHONPATH=. python experiments/core/run_safety_sweep.py --sweep all --dry-run
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# ── Constants ────────────────────────────────────────────────────────────

BASE_OUTPUT_DIR = "results/safety_collapse_sweep_v2"

# Default pivot point
DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_MODEL_SHORT = "qwen35_9b"
DEFAULT_CONTEXT_TYPES = ["structured_walk_15", "natural_books"]

# Dense log-spaced context lengths (18 points)
ALL_LENGTHS = [
    0, 50, 100, 200, 500, 1000, 2000, 5000,
    10000, 20000, 30000, 50000, 75000, 100000,
    150000, 200000, 250000, 262144,
]

# Per-model maximum context lengths
MODEL_MAX_CTX = {
    "Qwen/Qwen3.5-0.8B": 262144,
    "Qwen/Qwen3.5-2B": 262144,
    "Qwen/Qwen3.5-4B": 262144,
    "Qwen/Qwen3.5-9B": 262144,
    "Qwen/Qwen3.5-27B": 30000,  # KV cache memory constraint on 2xH100
    "meta-llama/Llama-3.3-70B-Instruct": 131072,
    "allenai/OLMo-3-7B-Instruct": 65536,
}

# Models that should NOT use vLLM (flashinfer incompatible)
NO_VLLM_MODELS = {
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-27B",
}

# Models needing special vLLM flags (enforce_eager + TP=2)
LARGE_VLLM_MODELS = {
    "meta-llama/Llama-3.3-70B-Instruct": {
        "vllm_tp": 2,
        "vllm_enforce_eager": True,
        "vllm_max_num_batched_tokens": 4096,
        "vllm_gpu_memory_utilization": 0.95,
    },
}

# Sweep C: Representative context lengths (8 points)
SWEEP_C_LENGTHS = [0, 100, 500, 2000, 10000, 50000, 150000, 262144]

# Sweep C: Context types to test
SWEEP_C_CONTEXT_TYPES = [
    "repeated_token",
    "structured_walk_15",
    "structured_walk_50",
    "structured_walk_200",
    "structured_walk_1000",
    "random_tokens_15",
    "random_tokens_50",
    "random_tokens_200",
    "random_tokens_1000",
    "lorem_ipsum",
    "natural_books",
    "structured_walk_15_thinking",
]


def log_context_lengths(max_ctx: int) -> list[int]:
    """Filter ALL_LENGTHS to those <= max_ctx."""
    return [l for l in ALL_LENGTHS if l <= max_ctx]


def lengths_to_str(lengths: list[int]) -> str:
    """Convert list of lengths to comma-separated string."""
    return ",".join(str(l) for l in lengths)


# ── Sweep Configurations ────────────────────────────────────────────────

def build_sweep_a() -> list[dict]:
    """Sweep A: Context Length (default model, full 18-point range).

    This is the same as the default model entry in Sweep B, so it's
    included automatically. Only generate standalone if needed.
    """
    max_ctx = MODEL_MAX_CTX[DEFAULT_MODEL]
    lengths = log_context_lengths(max_ctx)
    return [{
        "model": DEFAULT_MODEL,
        "short": DEFAULT_MODEL_SHORT,
        "context_types": DEFAULT_CONTEXT_TYPES,
        "lengths": lengths,
        "sweep": "context_length",
        "output_subdir": f"context_length/{DEFAULT_MODEL_SHORT}",
    }]


def build_sweep_b() -> list[dict]:
    """Sweep B: Model Size (Qwen3.5 family, 0.8B to 27B)."""
    models = [
        ("Qwen/Qwen3.5-0.8B", "qwen35_0.8b"),
        ("Qwen/Qwen3.5-2B", "qwen35_2b"),
        ("Qwen/Qwen3.5-4B", "qwen35_4b"),
        ("Qwen/Qwen3.5-9B", "qwen35_9b"),
        ("Qwen/Qwen3.5-27B", "qwen35_27b"),
    ]
    experiments = []
    for model, short in models:
        max_ctx = MODEL_MAX_CTX[model]
        lengths = log_context_lengths(max_ctx)
        experiments.append({
            "model": model,
            "short": short,
            "context_types": DEFAULT_CONTEXT_TYPES,
            "lengths": lengths,
            "sweep": "model_size",
            "output_subdir": f"model_size/{short}",
        })
    return experiments


def build_sweep_c() -> list[dict]:
    """Sweep C: Context Type (default model, 8 representative lengths)."""
    max_ctx = MODEL_MAX_CTX[DEFAULT_MODEL]
    lengths = [l for l in SWEEP_C_LENGTHS if l <= max_ctx]
    experiments = []
    for ctx_type in SWEEP_C_CONTEXT_TYPES:
        # Each context type runs as a separate experiment for resumability
        experiments.append({
            "model": DEFAULT_MODEL,
            "short": ctx_type,
            "context_types": [ctx_type],
            "lengths": lengths,
            "sweep": "context_type",
            "output_subdir": f"context_type/{ctx_type}",
            # Enable thinking mode for the _thinking variant
            "enable_thinking": ctx_type.endswith("_thinking"),
            # Give thinking room for thinking variant
            "max_new_tokens": 500 if ctx_type.endswith("_thinking") else 100,
        })
    return experiments


def build_sweep_d() -> list[dict]:
    """Sweep D: Model Architecture (at ~7-9B scale)."""
    models = [
        ("Qwen/Qwen3.5-9B", "qwen35_9b"),
        ("meta-llama/Llama-3.3-70B-Instruct", "llama33_70b"),
        ("allenai/OLMo-3-7B-Instruct", "olmo3_7b"),
    ]
    experiments = []
    for model, short in models:
        max_ctx = MODEL_MAX_CTX.get(model, 65536)
        lengths = log_context_lengths(max_ctx)
        experiments.append({
            "model": model,
            "short": short,
            "context_types": DEFAULT_CONTEXT_TYPES,
            "lengths": lengths,
            "sweep": "architecture",
            "output_subdir": f"architecture/{short}",
        })
    return experiments


SWEEP_BUILDERS = {
    "context_length": build_sweep_a,
    "model_size": build_sweep_b,
    "context_type": build_sweep_c,
    "architecture": build_sweep_d,
}


# ── Execution ────────────────────────────────────────────────────────────

def get_output_dir(exp: dict) -> Path:
    return Path(BASE_OUTPUT_DIR) / exp["output_subdir"]


def is_experiment_complete(output_dir: Path | str) -> bool:
    return (Path(output_dir) / "all_results.json").exists()


def is_judge_complete(output_dir: Path | str) -> bool:
    return (Path(output_dir) / "judge" / "judge_results.json").exists()


def needs_multi_gpu(model: str) -> bool:
    """Check if a model requires multiple GPUs."""
    return model in LARGE_VLLM_MODELS or model == "Qwen/Qwen3.5-27B"


def build_experiment_command(exp: dict) -> list[str]:
    """Build the command to run a single experiment."""
    output_dir = get_output_dir(exp)
    context_types_str = ",".join(exp["context_types"])
    lengths_str = lengths_to_str(exp["lengths"])
    model = exp["model"]

    cmd = [
        sys.executable, "-u",
        "experiments/core/run_safety_collapse_experiment.py",
        "--model", model,
        "--context-types", context_types_str,
        "--wrapping-modes", "raw",
        "--context-lengths", lengths_str,
        "--n-trials", "3",
        "--output-dir", str(output_dir),
    ]

    # Enable thinking mode if requested
    if exp.get("enable_thinking"):
        cmd.append("--enable-thinking")

    # Override max-new-tokens if specified
    max_new_tokens = exp.get("max_new_tokens", 100)
    cmd.extend(["--max-new-tokens", str(max_new_tokens)])

    # Backend selection
    if model in NO_VLLM_MODELS:
        # HuggingFace only (no vLLM)
        pass
    elif model in LARGE_VLLM_MODELS:
        cfg = LARGE_VLLM_MODELS[model]
        cmd.append("--use-vllm")
        if cfg.get("vllm_tp"):
            cmd.extend(["--vllm-tp", str(cfg["vllm_tp"])])
        if cfg.get("vllm_enforce_eager"):
            cmd.append("--vllm-enforce-eager")
        if cfg.get("vllm_max_num_batched_tokens"):
            cmd.extend(["--vllm-max-num-batched-tokens",
                         str(cfg["vllm_max_num_batched_tokens"])])
        if cfg.get("vllm_gpu_memory_utilization"):
            cmd.extend(["--vllm-gpu-memory-utilization",
                         str(cfg["vllm_gpu_memory_utilization"])])
        # Set max model len based on max context + some headroom
        max_len = max(exp["lengths"]) + 1000
        cmd.extend(["--vllm-max-model-len", str(max_len)])
    else:
        # Default: try vLLM with TP=1
        cmd.append("--use-vllm")

    return cmd


def build_judge_submit_command(output_dir: Path) -> list[str]:
    """Build command to submit judge batch (non-blocking)."""
    return [
        sys.executable, "-u",
        "experiments/core/run_llm_judge_safety.py",
        "--results-dir", str(output_dir),
        "--submit-only",
    ]


def build_judge_collect_command(output_dir: Path) -> list[str]:
    """Build command to collect judge results (blocking)."""
    return [
        sys.executable, "-u",
        "experiments/core/run_llm_judge_safety.py",
        "--results-dir", str(output_dir),
    ]


def run_single_experiment(exp: dict, dry_run: bool, log_file,
                          pending_judges: list, gpu: str | None = None) -> bool:
    """Run a single experiment + submit judge (non-blocking).

    After the GPU experiment finishes, submits the judge batch to the
    Anthropic API but does NOT wait for it. The batch ID is recorded in
    pending_judges for later collection.

    Args:
        gpu: If set, pin to this GPU via CUDA_VISIBLE_DEVICES (e.g. "0", "1").
    """
    output_dir = get_output_dir(exp)
    label = exp["output_subdir"]

    # Check if already done
    if is_experiment_complete(output_dir):
        msg = f"[SKIP] {label}: already complete"
        print(msg)
        log_file.write(msg + "\n")

        # Submit judge if needed (non-blocking)
        if not is_judge_complete(output_dir):
            _submit_judge(label, output_dir, dry_run, log_file, pending_judges)
        return True

    # Build and run experiment
    cmd = build_experiment_command(exp)
    n_lengths = len(exp["lengths"])
    n_types = len(exp["context_types"])
    est_evals = n_lengths * n_types * 3 * 50  # 3 trials, 50 prompts
    gpu_tag = f" [GPU {gpu}]" if gpu else ""
    msg = (f"[RUN]{gpu_tag} {label}: {exp['model']} | "
           f"{n_types} types x {n_lengths} lengths (~{est_evals} evals)")
    print(msg)
    log_file.write(msg + "\n")
    log_file.write(f"  CMD: {' '.join(cmd)}\n")
    log_file.flush()

    if dry_run:
        print(f"  DRY RUN: {' '.join(cmd)}")
        _submit_judge(label, output_dir, dry_run, log_file, pending_judges)
        return True

    # Build subprocess environment with GPU pinning
    env = {**os.environ, "PYTHONPATH": "."}
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    start = time.time()
    result = subprocess.run(
        cmd, env=env,
        capture_output=False,
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        msg = f"[FAIL] {label}: experiment failed (exit {result.returncode}) after {elapsed/3600:.1f}h"
        print(msg)
        log_file.write(msg + "\n")
        return False

    msg = f"[DONE] {label}: experiment complete in {elapsed/3600:.1f}h"
    print(msg)
    log_file.write(msg + "\n")

    # Submit judge batch (non-blocking — GPU can start next experiment immediately)
    _submit_judge(label, output_dir, dry_run, log_file, pending_judges)
    return True


def _submit_judge(label: str, output_dir: Path, dry_run: bool,
                  log_file, pending_judges: list):
    """Submit a judge batch without waiting. Records output_dir for later collection."""
    msg = f"[JUDGE SUBMIT] {label}: submitting batch..."
    print(msg)
    log_file.write(msg + "\n")
    log_file.flush()

    if dry_run:
        print(f"  DRY RUN: submit judge for {label}")
        pending_judges.append((label, output_dir))
        return

    judge_cmd = build_judge_submit_command(output_dir)
    result = subprocess.run(
        judge_cmd, env={**os.environ, "PYTHONPATH": "."},
        capture_output=False,
    )

    if result.returncode != 0:
        msg = f"[WARN] {label}: judge submission failed (exit {result.returncode})"
        print(msg)
        log_file.write(msg + "\n")
    else:
        pending_judges.append((label, output_dir))
        msg = f"[JUDGE SUBMIT] {label}: batch submitted (will collect later)"
        print(msg)
        log_file.write(msg + "\n")

    log_file.flush()


def collect_all_judges(pending_judges: list, dry_run: bool, log_file):
    """Collect all pending judge batches (blocking — runs after all GPU work is done)."""
    if not pending_judges:
        return

    print(f"\n{'='*70}")
    print(f"COLLECTING {len(pending_judges)} JUDGE BATCHES")
    print(f"{'='*70}")
    log_file.write(f"\nCollecting {len(pending_judges)} judge batches\n")

    for label, output_dir in pending_judges:
        if is_judge_complete(output_dir):
            msg = f"[JUDGE SKIP] {label}: already complete"
            print(msg)
            log_file.write(msg + "\n")
            continue

        msg = f"[JUDGE COLLECT] {label}: waiting for results..."
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

        if dry_run:
            continue

        judge_cmd = build_judge_collect_command(output_dir)
        result = subprocess.run(
            judge_cmd, env={**os.environ, "PYTHONPATH": "."},
            capture_output=False,
        )

        if result.returncode != 0:
            msg = f"[WARN] {label}: judge collection failed (exit {result.returncode})"
        else:
            msg = f"[JUDGE DONE] {label}: results collected"
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()


# ── Main ─────────────────────────────────────────────────────────────────

def deduplicate_experiments(experiments: list[dict]) -> list[dict]:
    """Remove duplicate experiments that would produce identical results.

    Experiments with the same model, context_types, and lengths but different
    output_subdirs are duplicates. Keep the first occurrence and create
    symlinks for the rest after the first one completes.
    """
    seen = {}  # (model, ctx_types_tuple, lengths_tuple) -> first exp
    deduped = []
    symlink_plan = []  # (source_subdir, target_subdir)

    for exp in experiments:
        key = (
            exp["model"],
            tuple(exp["context_types"]),
            tuple(exp["lengths"]),
            exp.get("enable_thinking", False),
        )
        if key in seen:
            first = seen[key]
            symlink_plan.append((first["output_subdir"], exp["output_subdir"]))
        else:
            seen[key] = exp
            deduped.append(exp)

    return deduped, symlink_plan


def create_symlinks(symlink_plan: list[tuple[str, str]]):
    """Create symlinks for deduplicated experiments."""
    for source_subdir, target_subdir in symlink_plan:
        source = Path(BASE_OUTPUT_DIR) / source_subdir
        target = Path(BASE_OUTPUT_DIR) / target_subdir
        if target.exists() or target.is_symlink():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        # Use relative symlink
        rel = os.path.relpath(source, target.parent)
        target.symlink_to(rel)
        print(f"[LINK] {target_subdir} -> {source_subdir}")


def main():
    sweep_choices = list(SWEEP_BUILDERS.keys()) + ["all"]
    parser = argparse.ArgumentParser(
        description="Safety Collapse Sweep v2: star-shaped experiment design",
    )
    parser.add_argument(
        "--sweep", type=str, default="all",
        choices=sweep_choices,
        help="Which sweep dimension to run",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument(
        "--gpu", type=str, default=None,
        help="Pin experiments to this GPU (sets CUDA_VISIBLE_DEVICES). "
             "E.g. --gpu 0 or --gpu 1",
    )
    parser.add_argument(
        "--include", type=str, default=None,
        help="Comma-separated prefixes to include (e.g. 'context_type,model_size/qwen35_0.8b')",
    )
    parser.add_argument(
        "--exclude", type=str, default=None,
        help="Comma-separated prefixes to exclude",
    )
    parser.add_argument(
        "--multi-gpu-only", action="store_true",
        help="Only run experiments that require multiple GPUs (27B, 70B)",
    )
    parser.add_argument(
        "--single-gpu-only", action="store_true",
        help="Only run experiments that fit on a single GPU",
    )
    args = parser.parse_args()

    # Build experiments for selected sweeps
    if args.sweep == "all":
        sweep_names = list(SWEEP_BUILDERS.keys())
    else:
        sweep_names = [args.sweep]

    all_experiments = []
    for sweep_name in sweep_names:
        exps = SWEEP_BUILDERS[sweep_name]()
        all_experiments.extend(exps)

    # Deduplicate identical experiments across sweep dimensions
    all_experiments, symlink_plan = deduplicate_experiments(all_experiments)
    if symlink_plan:
        print(f"Deduplicated: {len(symlink_plan)} redundant experiments "
              f"(will symlink after source completes)")

    # Filter by GPU requirement
    if args.single_gpu_only:
        all_experiments = [e for e in all_experiments
                          if not needs_multi_gpu(e["model"])]
    elif args.multi_gpu_only:
        all_experiments = [e for e in all_experiments
                          if needs_multi_gpu(e["model"])]

    # Filter by include/exclude patterns
    if args.include:
        prefixes = [p.strip() for p in args.include.split(",")]
        all_experiments = [
            e for e in all_experiments
            if any(e["output_subdir"].startswith(p) for p in prefixes)
        ]
    if args.exclude:
        prefixes = [p.strip() for p in args.exclude.split(",")]
        all_experiments = [
            e for e in all_experiments
            if not any(e["output_subdir"].startswith(p) for p in prefixes)
        ]

    # Setup logging
    log_dir = Path(BASE_OUTPUT_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    gpu_suffix = f"_gpu{args.gpu}" if args.gpu else ""
    log_path = log_dir / f"sweep_v2{gpu_suffix}.log"

    print("=" * 70)
    print("SAFETY COLLAPSE SWEEP v2 (Star-Shaped Design)")
    print("=" * 70)
    print(f"Sweeps: {sweep_names}")
    print(f"GPU: {args.gpu or 'all (auto)'}")
    print(f"Total experiments: {len(all_experiments)}")
    print(f"Log: {log_path}")
    if args.dry_run:
        print("[DRY RUN MODE]")
    print()

    # Print experiment summary
    for exp in all_experiments:
        n_types = len(exp["context_types"])
        n_lengths = len(exp["lengths"])
        est = n_types * n_lengths * 3 * 50
        status = "DONE" if is_experiment_complete(get_output_dir(exp)) else "TODO"
        multi = " [MULTI-GPU]" if needs_multi_gpu(exp["model"]) else ""
        print(f"  [{status}] {exp['output_subdir']}: "
              f"{exp['model']} | {n_types}x{n_lengths} (~{est} evals){multi}")
    print()

    with open(log_path, "a") as log_file:
        log_file.write(f"\n{'='*70}\n")
        log_file.write(f"Sweep started: {datetime.now().isoformat()}\n")
        log_file.write(f"Sweeps: {sweep_names}, GPU: {args.gpu or 'auto'}\n")
        log_file.write(f"{'='*70}\n\n")

        n_success = 0
        n_fail = 0
        n_skip = 0
        pending_judges = []  # (label, output_dir) pairs awaiting judge results
        sweep_start = time.time()

        # Phase 1: Run GPU experiments + submit judge batches (non-blocking)
        for exp in all_experiments:
            output_dir = get_output_dir(exp)
            if is_experiment_complete(output_dir) and is_judge_complete(output_dir):
                n_skip += 1
                print(f"[SKIP] {exp['output_subdir']}: fully complete")
                continue

            success = run_single_experiment(
                exp, args.dry_run, log_file, pending_judges, gpu=args.gpu)
            if success:
                n_success += 1
            else:
                n_fail += 1

        # Create symlinks for deduplicated experiments
        if symlink_plan and not args.dry_run:
            create_symlinks(symlink_plan)

        # Phase 2: Collect all pending judge results
        collect_all_judges(pending_judges, args.dry_run, log_file)

        elapsed = time.time() - sweep_start
        summary = (
            f"\nSweep complete: {n_success} succeeded, {n_fail} failed, "
            f"{n_skip} skipped in {elapsed/3600:.1f}h"
        )
        print(summary)
        log_file.write(summary + "\n")


if __name__ == "__main__":
    main()
