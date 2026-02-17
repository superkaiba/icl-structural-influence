#!/usr/bin/env python3
"""
Collapse Trajectory: How do representations collapse as context accumulates?

Measures cos_sim and effective_dim at regular intervals throughout context
processing, producing a trajectory that shows the dynamics of collapse.

Usage:
    # Quick test
    PYTHONPATH=. python experiments/core/run_collapse_trajectory.py --quick-test

    # Full run (20K tokens, all conditions)
    PYTHONPATH=. python experiments/core/run_collapse_trajectory.py
"""

import argparse
import json
import gc
from collections import deque
from datetime import datetime
from pathlib import Path

import torch
import numpy as np

from src.models import HookedLLM
from src.data.dual_interpretation_graph import (
    DualInterpretationGraph,
    DualInterpretationConfig,
)
from src.data.natural_language_loader import (
    NaturalLanguageLoader,
    NaturalLanguageConfig,
)
from src.metrics.collapse_metrics import compute_collapse_metrics
import random


# ── Context generation (reused from probing experiment) ───────────────────

def generate_context_tokens(
    context_type: str,
    context_length: int,
    model: HookedLLM,
    graph: DualInterpretationGraph,
    nl_loader: NaturalLanguageLoader,
    trial_idx: int = 0,
) -> list[int]:
    """Generate raw context tokens for a given type and length."""
    if context_type == "structured_walk":
        walk_length = context_length * 2
        prompt, _, _ = graph.generate_h1_only_walk(
            length=walk_length, return_nodes=True
        )
        tokens = model.tokenizer.encode(prompt, add_special_tokens=False)
        return tokens[:context_length]

    elif context_type == "natural_books":
        nl_loader.rng = random.Random(42 + trial_idx)
        tokens = nl_loader.load_book(context_length)
        return tokens[:context_length]

    elif context_type == "repeated_token":
        token_id = model.tokenizer.encode(" the", add_special_tokens=False)[0]
        return [token_id] * context_length

    else:
        raise ValueError(f"Unknown context type: {context_type}")


# ── Trajectory measurement ────────────────────────────────────────────────

def measure_collapse_trajectory(
    model: HookedLLM,
    token_ids: list[int],
    layers: list[int],
    chunk_size: int = 256,
    window_size: int = 50,
    measure_every: int = 256,
) -> list[dict]:
    """
    Process context tokens in chunks and measure collapse at regular intervals.

    Returns a list of dicts, each with:
        - position: token index where measurement was taken
        - layer_metrics: {layer: {cos_sim, eff_dim, spread, ...}}
    """
    if len(token_ids) == 0:
        return []

    past_kvs = None
    # Keep a sliding window of representations per layer
    recent_reps = {layer: deque(maxlen=window_size) for layer in layers}
    trajectory = []

    tokens_processed = 0
    last_measured_at = 0

    for start in range(0, len(token_ids), chunk_size):
        end = min(start + chunk_size, len(token_ids))
        chunk = token_ids[start:end]
        input_ids = torch.tensor([chunk]).to(model.device)

        _, cache, past_kvs = model.forward_incremental(
            input_ids, layers=layers, past_key_values=past_kvs,
        )

        # Store per-token representations
        for layer in layers:
            rep = cache.get_residual_stream(layer)
            if rep is not None:
                for pos in range(rep.shape[1]):
                    rep_np = rep[0, pos].cpu().float().numpy()
                    recent_reps[layer].append(rep_np)

        tokens_processed = end

        # Measure collapse if we've passed the next measurement point
        if (tokens_processed - last_measured_at >= measure_every) or end == len(token_ids):
            layer_metrics = {}
            for layer in layers:
                reps_list = list(recent_reps[layer])
                if len(reps_list) >= 10:
                    metrics = compute_collapse_metrics(reps_list)
                    layer_metrics[layer] = {
                        "cos_sim": float(metrics.avg_cos_sim) if metrics.avg_cos_sim is not None else None,
                        "eff_dim": float(metrics.effective_dim) if metrics.effective_dim is not None else 0.0,
                        "spread": float(metrics.spread) if metrics.spread is not None else 0.0,
                        "intrinsic_dim": float(metrics.intrinsic_dim) if metrics.intrinsic_dim is not None else 0.0,
                    }

            if layer_metrics:
                trajectory.append({
                    "position": tokens_processed,
                    "layer_metrics": layer_metrics,
                })
                last_measured_at = tokens_processed

    return trajectory


# ── Main ──────────────────────────────────────────────────────────────────

def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("COLLAPSE TRAJECTORY EXPERIMENT")
    print("=" * 70)

    # Load model
    print(f"\nLoading model: {args.model}")
    model = HookedLLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    n_layers = model.model.config.num_hidden_layers
    layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    print(f"Layers: {layers}")

    # Data sources
    graph_config = DualInterpretationConfig(
        vocab_size=15, clusters_per_interpretation=3, seed=42,
    )
    graph = DualInterpretationGraph(graph_config)

    nl_loader = NaturalLanguageLoader(model.tokenizer, NaturalLanguageConfig(seed=42))

    context_types = args.context_types
    max_length = args.max_length
    n_trials = args.n_trials

    print(f"Context types: {context_types}")
    print(f"Max length: {max_length:,} tokens")
    print(f"Trials: {n_trials}")
    print(f"Measure every: {args.measure_every} tokens")
    print(f"Window size: {args.window_size} tokens")

    all_results = {}

    for ctx_type in context_types:
        print(f"\n{'='*60}")
        print(f"Context type: {ctx_type}")
        print(f"{'='*60}")

        trial_trajectories = []

        for trial in range(n_trials):
            print(f"\n  Trial {trial + 1}/{n_trials}")

            # Generate context
            tokens = generate_context_tokens(
                ctx_type, max_length, model, graph, nl_loader, trial_idx=trial
            )
            actual_len = len(tokens)
            print(f"    Generated {actual_len:,} tokens")

            # Measure trajectory
            with torch.no_grad():
                trajectory = measure_collapse_trajectory(
                    model, tokens, layers,
                    chunk_size=args.chunk_size,
                    window_size=args.window_size,
                    measure_every=args.measure_every,
                )

            trial_trajectories.append(trajectory)

            # Print a few checkpoints
            for t in trajectory[::max(1, len(trajectory) // 5)]:
                l27 = t["layer_metrics"].get(layers[-1], {})
                print(f"    pos={t['position']:>6d}  "
                      f"L{layers[-1]} cos={l27.get('cos_sim', 0):.3f}  "
                      f"eff_dim={l27.get('eff_dim', 0):.1f}")

            # Clear KV cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        all_results[ctx_type] = trial_trajectories

    # Save results
    results_file = output_dir / "trajectory_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "config": {
                "model": args.model,
                "layers": layers,
                "context_types": context_types,
                "max_length": max_length,
                "n_trials": n_trials,
                "measure_every": args.measure_every,
                "window_size": args.window_size,
                "chunk_size": args.chunk_size,
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            },
            "trajectories": all_results,
        }, f, indent=2)

    print(f"\nResults saved to {results_file}")
    return all_results, layers


def main():
    parser = argparse.ArgumentParser(description="Collapse trajectory measurement")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output-dir", default="results/collapse_trajectory")
    parser.add_argument("--max-length", type=int, default=20000)
    parser.add_argument("--n-trials", type=int, default=3)
    parser.add_argument("--context-types", nargs="+",
                        default=["structured_walk", "natural_books", "repeated_token"])
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--measure-every", type=int, default=256)
    parser.add_argument("--quick-test", action="store_true")
    args = parser.parse_args()

    if args.quick_test:
        args.max_length = 2000
        args.n_trials = 1
        args.measure_every = 256
        args.output_dir = "results/collapse_trajectory_test"

    run(args)


if __name__ == "__main__":
    main()
