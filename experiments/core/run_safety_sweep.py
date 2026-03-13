#!/usr/bin/env python3
"""
Safety Collapse Sweep: Run safety experiments across multiple models, vocab sizes,
and context lengths.

Orchestrates calls to run_safety_collapse_experiment.py and run_llm_judge_safety.py
for each configuration. Resumable — skips experiments that already have results.

Usage:
    # Run all Tier 1 experiments (fast, high-value)
    PYTHONPATH=. python experiments/core/run_safety_sweep.py --tier 1

    # Run a specific sweep dimension
    PYTHONPATH=. python experiments/core/run_safety_sweep.py --sweep model_size

    # Run everything
    PYTHONPATH=. python experiments/core/run_safety_sweep.py --sweep all

    # Dry run (show commands without executing)
    PYTHONPATH=. python experiments/core/run_safety_sweep.py --sweep all --dry-run
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# ── Sweep Configurations ──────────────────────────────────────────────────

BASE_OUTPUT_DIR = "results/safety_collapse_sweep"

# Standard context lengths (every 5K up to 50K)
STANDARD_LENGTHS = "0,5000,10000,15000,20000,25000,30000,40000,50000"
# Reduced lengths for very large models (KV cache memory constraint)
LARGE_MODEL_LENGTHS = "0,5000,10000,15000,20000,25000,30000"
# Vocab sweep lengths
VOCAB_LENGTHS = "0,5000,10000,15000,20000,30000"

SWEEP_CONFIGS = {
    "model_size": {
        "description": "Qwen2.5-Instruct family: 0.5B to 72B",
        "experiments": [
            {
                "model": "Qwen/Qwen2.5-0.5B-Instruct",
                "short": "qwen25_0.5b",
                "lengths": STANDARD_LENGTHS,
                "tier": 1,
            },
            {
                "model": "Qwen/Qwen2.5-3B-Instruct",
                "short": "qwen25_3b",
                "lengths": STANDARD_LENGTHS,
                "tier": 1,
            },
            {
                "model": "Qwen/Qwen2.5-14B-Instruct",
                "short": "qwen25_14b",
                "lengths": STANDARD_LENGTHS,
                "tier": 2,
            },
            {
                "model": "Qwen/Qwen2.5-72B-Instruct",
                "short": "qwen25_72b",
                "lengths": LARGE_MODEL_LENGTHS,
                "tier": 3,
            },
        ],
    },
    "architecture": {
        "description": "Llama instruct models for cross-architecture comparison",
        "experiments": [
            {
                "model": "meta-llama/Llama-3.1-8B-Instruct",
                "short": "llama31_8b",
                "lengths": STANDARD_LENGTHS,
                "tier": 2,
            },
            {
                "model": "meta-llama/Llama-3.3-70B-Instruct",
                "short": "llama33_70b",
                "lengths": LARGE_MODEL_LENGTHS,
                "tier": 3,
            },
        ],
    },
    "model_gen": {
        "description": "Different Qwen generations at ~7-8B size",
        "experiments": [
            {
                "model": "Qwen/Qwen3-8B",
                "short": "qwen3_8b",
                "lengths": STANDARD_LENGTHS,
                "tier": 4,
            },
        ],
    },
    "vocab_size": {
        "description": "Vocab size sweep on Qwen2.5-7B-Instruct",
        "experiments": [
            {
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "short": "vocab50",
                "lengths": VOCAB_LENGTHS,
                "vocab_size": 50,
                "tier": 1,
            },
            {
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "short": "vocab200",
                "lengths": VOCAB_LENGTHS,
                "vocab_size": 200,
                "tier": 1,
            },
            {
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "short": "vocab1000",
                "lengths": VOCAB_LENGTHS,
                "vocab_size": 1000,
                "tier": 1,
            },
        ],
    },
    "context_granularity": {
        "description": "Fill-in context lengths for Qwen2.5-7B (15K, 25K)",
        "experiments": [
            {
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "short": "qwen25_7b_fillin",
                "lengths": "15000,25000",
                "tier": 1,
            },
        ],
    },
}


def get_output_dir(sweep_name: str, short_name: str) -> Path:
    return Path(BASE_OUTPUT_DIR) / sweep_name / short_name


def is_experiment_complete(output_dir: Path) -> bool:
    """Check if an experiment has already completed."""
    return (output_dir / "all_results.json").exists()


def is_judge_complete(output_dir: Path) -> bool:
    """Check if the LLM judge has already run."""
    return (output_dir / "judge" / "judge_results.json").exists()


def build_experiment_command(exp: dict, sweep_name: str, use_vllm: bool = False) -> list[str]:
    """Build the command to run a single experiment."""
    output_dir = get_output_dir(sweep_name, exp["short"])

    cmd = [
        sys.executable, "-u",
        "experiments/core/run_safety_collapse_experiment.py",
        "--model", exp["model"],
        "--context-types", "structured_walk,natural_books",
        "--wrapping-modes", "raw",
        "--context-lengths", exp["lengths"],
        "--n-trials", "3",
        "--output-dir", str(output_dir),
    ]

    if "vocab_size" in exp:
        cmd.extend(["--vocab-size", str(exp["vocab_size"])])

    if use_vllm:
        cmd.append("--use-vllm")

    return cmd


def build_judge_command(output_dir: Path) -> list[str]:
    """Build the command to run the LLM judge."""
    return [
        sys.executable, "-u",
        "experiments/core/run_llm_judge_safety.py",
        "--results-dir", str(output_dir),
    ]


def run_single_experiment(exp: dict, sweep_name: str, dry_run: bool, log_file, use_vllm: bool = False) -> bool:
    """Run a single experiment + judge. Returns True if successful."""
    output_dir = get_output_dir(sweep_name, exp["short"])
    label = f"{sweep_name}/{exp['short']}"

    # Check if already done
    if is_experiment_complete(output_dir):
        msg = f"[SKIP] {label}: already complete"
        print(msg)
        log_file.write(msg + "\n")

        # Still run judge if needed
        if not is_judge_complete(output_dir):
            msg = f"[JUDGE] {label}: running judge..."
            print(msg)
            log_file.write(msg + "\n")

            if not dry_run:
                judge_cmd = build_judge_command(output_dir)
                result = subprocess.run(
                    judge_cmd, env={**os.environ, "PYTHONPATH": "."},
                    capture_output=False,
                )
                if result.returncode != 0:
                    msg = f"[FAIL] {label}: judge failed (exit {result.returncode})"
                    print(msg)
                    log_file.write(msg + "\n")
                    return False
        return True

    # Build and run experiment
    cmd = build_experiment_command(exp, sweep_name, use_vllm=use_vllm)
    msg = f"[RUN] {label}: {exp['model']} | lengths={exp['lengths']}"
    if "vocab_size" in exp:
        msg += f" | vocab={exp['vocab_size']}"
    print(msg)
    log_file.write(msg + "\n")
    log_file.write(f"  CMD: {' '.join(cmd)}\n")
    log_file.flush()

    if dry_run:
        print(f"  DRY RUN: {' '.join(cmd)}")
        return True

    start = time.time()
    result = subprocess.run(
        cmd, env={**os.environ, "PYTHONPATH": "."},
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

    # Run judge
    msg = f"[JUDGE] {label}: running judge..."
    print(msg)
    log_file.write(msg + "\n")
    log_file.flush()

    judge_cmd = build_judge_command(output_dir)
    result = subprocess.run(
        judge_cmd, env={**os.environ, "PYTHONPATH": "."},
        capture_output=False,
    )

    if result.returncode != 0:
        msg = f"[WARN] {label}: judge failed (exit {result.returncode})"
        print(msg)
        log_file.write(msg + "\n")
        # Don't fail the whole sweep for judge errors
    else:
        msg = f"[JUDGE] {label}: judge complete"
        print(msg)
        log_file.write(msg + "\n")

    log_file.flush()
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Safety Collapse Sweep: run experiments across models and configs",
    )
    parser.add_argument(
        "--sweep", type=str, default="all",
        choices=list(SWEEP_CONFIGS.keys()) + ["all"],
        help="Which sweep dimension to run",
    )
    parser.add_argument(
        "--tier", type=int, default=None,
        help="Only run experiments up to this tier (1=fast, 4=slow)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--use-vllm", action="store_true",
                        help="Use vLLM for batched safety evaluation (faster)")
    args = parser.parse_args()

    # Determine which sweeps to run
    if args.sweep == "all":
        sweep_names = list(SWEEP_CONFIGS.keys())
    else:
        sweep_names = [args.sweep]

    max_tier = args.tier if args.tier else 4

    # Collect all experiments, sorted by tier
    all_experiments = []
    for sweep_name in sweep_names:
        sweep = SWEEP_CONFIGS[sweep_name]
        for exp in sweep["experiments"]:
            if exp["tier"] <= max_tier:
                all_experiments.append((sweep_name, exp))

    all_experiments.sort(key=lambda x: x[1]["tier"])

    # Setup logging
    log_dir = Path(BASE_OUTPUT_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "sweep.log"

    print("=" * 70)
    print("SAFETY COLLAPSE SWEEP")
    print("=" * 70)
    print(f"Sweeps: {sweep_names}")
    print(f"Max tier: {max_tier}")
    print(f"Total experiments: {len(all_experiments)}")
    print(f"Log: {log_path}")
    if args.dry_run:
        print("[DRY RUN MODE]")
    print()

    with open(log_path, "a") as log_file:
        log_file.write(f"\n{'='*70}\n")
        log_file.write(f"Sweep started: {datetime.now().isoformat()}\n")
        log_file.write(f"Sweeps: {sweep_names}, Max tier: {max_tier}\n")
        log_file.write(f"{'='*70}\n\n")

        n_success = 0
        n_fail = 0
        n_skip = 0
        sweep_start = time.time()

        for sweep_name, exp in all_experiments:
            output_dir = get_output_dir(sweep_name, exp["short"])
            if is_experiment_complete(output_dir) and is_judge_complete(output_dir):
                n_skip += 1
                print(f"[SKIP] {sweep_name}/{exp['short']}: fully complete")
                continue

            success = run_single_experiment(exp, sweep_name, args.dry_run, log_file, use_vllm=getattr(args, 'use_vllm', False))
            if success:
                n_success += 1
            else:
                n_fail += 1

        elapsed = time.time() - sweep_start
        summary = (
            f"\nSweep complete: {n_success} succeeded, {n_fail} failed, "
            f"{n_skip} skipped in {elapsed/3600:.1f}h"
        )
        print(summary)
        log_file.write(summary + "\n")


if __name__ == "__main__":
    main()
