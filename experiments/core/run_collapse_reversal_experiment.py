#!/usr/bin/env python3
"""
Collapse Reversal Experiment: Can Representational Collapse Be Reversed?

This experiment tests whether representational collapse can be reversed by
injecting contradicting information after a non-ambiguous sequence has caused
collapse.

Hypothesis:
If representations collapse toward a single interpretation during unambiguous
context, injecting contradicting information should:
- Cause representations to "un-collapse" (spread out again)
- Or shift toward a different collapsed state
- Or have no effect (collapse is irreversible)

Experimental Design:
- Phase 1 (5000 tokens): Induce collapse with H1-only walk
- Phase 2 (5000 tokens): Inject one of several types of contradicting information

Conditions:
- control_h1_continuous: Continue H1-only walk (no injection)
- inject_h2_same_tokens: Switch to H2-only walk (same tokens, different structure)
- inject_different_graph: Switch to H1-only on a completely different graph
- inject_natural_books: Switch to natural language (books)
- inject_natural_wikipedia: Switch to natural language (Wikipedia)

Usage:
    # Quick test (2000 tokens, 1 trial)
    python run_collapse_reversal_experiment.py --model gpt2 --n-trials 1 \
        --phase1-length 1000 --phase2-length 1000

    # Medium test (5000 tokens, 3 trials)
    python run_collapse_reversal_experiment.py --model Qwen/Qwen2.5-7B \
        --n-trials 3 --phase1-length 2500 --phase2-length 2500

    # Full experiment (10000 tokens, 5 trials per condition)
    python run_collapse_reversal_experiment.py --model Qwen/Qwen2.5-7B \
        --n-trials 5

References:
    - Park et al. (2024) arXiv:2501.00070 "ICLR: In-Context Learning of Representations"
"""

import argparse
import json
import os
import gc
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional
import warnings

import torch
import numpy as np
from tqdm import tqdm

# Local imports
from src.data.dual_interpretation_graph import (
    DualInterpretationGraph,
    DualInterpretationConfig,
    create_graph_with_vocab_size,
    generate_extended_vocabulary,
)
from src.data.natural_language_loader import (
    NaturalLanguageLoader,
    NaturalLanguageConfig,
)
from src.models import HookedLLM
from src.metrics.collapse_metrics import (
    compute_collapse_metrics,
    CollapseMetrics,
)


# Experimental conditions
CONDITIONS = {
    # Control: no injection, continue H1
    "control_h1_continuous": {
        "phase1": {"type": "structured", "walk_type": "h1_only", "graph": "graph1"},
        "phase2": {"type": "structured", "walk_type": "h1_only", "graph": "graph1"},
        "description": "Control: continue H1-only walk throughout",
    },

    # Condition A: Same tokens, switch to H2 structure
    "inject_h2_same_tokens": {
        "phase1": {"type": "structured", "walk_type": "h1_only", "graph": "graph1"},
        "phase2": {"type": "structured", "walk_type": "h2_only", "graph": "graph1"},
        "description": "Inject H2 structure using same token vocabulary",
    },

    # Condition B: Different graph with different tokens
    "inject_different_graph": {
        "phase1": {"type": "structured", "walk_type": "h1_only", "graph": "graph1"},
        "phase2": {"type": "structured", "walk_type": "h1_only", "graph": "graph2"},
        "description": "Inject H1 walk from completely different graph/vocabulary",
    },

    # Condition C: Switch to natural language
    "inject_natural_books": {
        "phase1": {"type": "structured", "walk_type": "h1_only", "graph": "graph1"},
        "phase2": {"type": "natural", "source": "book"},
        "description": "Inject natural language from Project Gutenberg books",
    },
    "inject_natural_wikipedia": {
        "phase1": {"type": "structured", "walk_type": "h1_only", "graph": "graph1"},
        "phase2": {"type": "natural", "source": "wikipedia"},
        "description": "Inject natural language from Wikipedia articles",
    },
}


def get_checkpoint_schedule(injection_point: int, total_length: int) -> list[int]:
    """
    Generate checkpoint schedule with dense sampling around injection point.

    Args:
        injection_point: Token position where injection occurs
        total_length: Total sequence length

    Returns:
        Sorted list of checkpoint positions
    """
    checkpoints = set()

    # Phase 1: Collapse buildup (before injection)
    phase1_sparse = [100, 500, 1000, 2000, 3000, 4000]
    checkpoints.update([cp for cp in phase1_sparse if cp < injection_point])

    # Dense around injection point
    dense_before = [
        injection_point - 500,
        injection_point - 250,
        injection_point - 100,
        injection_point - 50,
        injection_point - 10,
    ]
    checkpoints.update([cp for cp in dense_before if cp > 0])

    # Injection point itself (last token before injection)
    checkpoints.add(injection_point)

    # Dense after injection
    dense_after = [
        injection_point + 10,
        injection_point + 50,
        injection_point + 100,
        injection_point + 250,
        injection_point + 500,
    ]
    checkpoints.update([cp for cp in dense_after if cp <= total_length])

    # Phase 2: Recovery tracking (sparse)
    phase2_sparse = [
        injection_point + 1000,
        injection_point + 2000,
        injection_point + 3000,
        injection_point + 4000,
        injection_point + 5000,
    ]
    checkpoints.update([cp for cp in phase2_sparse if cp <= total_length])

    return sorted(list(checkpoints))


def setup_dtype(dtype_str: str) -> torch.dtype:
    """Convert string to torch dtype."""
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }[dtype_str]


def clear_gpu_memory():
    """Clear GPU memory between operations."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def convert_numpy(obj):
    """Convert numpy arrays and types to JSON-serializable formats."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, CollapseMetrics):
        return convert_numpy(obj.to_dict())
    return obj


def generate_phase_tokens(
    phase_config: dict,
    target_length: int,
    graph1: DualInterpretationGraph,
    graph2: DualInterpretationGraph,
    nl_loader: NaturalLanguageLoader,
    tokenizer,
    trial_idx: int,
) -> tuple[list[int], dict]:
    """
    Generate tokens for a single phase of the experiment.

    Args:
        phase_config: Configuration for this phase (type, walk_type, etc.)
        target_length: Target number of tokens
        graph1: Primary graph for H1/H2 walks
        graph2: Secondary graph for "different graph" condition
        nl_loader: Natural language loader
        tokenizer: Model tokenizer
        trial_idx: Trial index for reproducibility

    Returns:
        Tuple of (token_ids, metadata)
    """
    if phase_config["type"] == "structured":
        # Select the appropriate graph
        graph = graph1 if phase_config["graph"] == "graph1" else graph2

        # Generate walk based on type
        walk_length = target_length * 2  # Generate extra for tokenization overhead

        if phase_config["walk_type"] == "h1_only":
            prompt, nodes, meta = graph.generate_h1_only_walk(
                length=walk_length,
                return_nodes=True,
            )
        elif phase_config["walk_type"] == "h2_only":
            prompt, nodes, meta = graph.generate_h2_only_walk(
                length=walk_length,
                return_nodes=True,
            )
        else:
            raise ValueError(f"Unknown walk type: {phase_config['walk_type']}")

        # Tokenize
        tokens = tokenizer.encode(prompt, add_special_tokens=False)

        metadata = {
            "type": "structured",
            "walk_type": phase_config["walk_type"],
            "graph": phase_config["graph"],
            "H1_clusters": meta["H1_clusters"][:target_length],
            "H2_clusters": meta["H2_clusters"][:target_length],
        }

        return tokens[:target_length], metadata

    elif phase_config["type"] == "natural":
        source = phase_config["source"]

        if source == "wikipedia":
            tokens = nl_loader.load_wikipedia(target_length)
        elif source == "book":
            tokens = nl_loader.load_book(target_length)
        else:
            raise ValueError(f"Unknown natural language source: {source}")

        metadata = {
            "type": "natural",
            "source": source,
            "actual_length": len(tokens),
        }

        return tokens[:target_length], metadata

    else:
        raise ValueError(f"Unknown phase type: {phase_config['type']}")


def generate_tokens_with_injection(
    condition_config: dict,
    phase1_length: int,
    phase2_length: int,
    graph1: DualInterpretationGraph,
    graph2: DualInterpretationGraph,
    nl_loader: NaturalLanguageLoader,
    tokenizer,
    trial_idx: int,
) -> tuple[list[int], dict]:
    """
    Generate full token sequence with injection at the midpoint.

    Args:
        condition_config: Full condition configuration with phase1 and phase2
        phase1_length: Length of phase 1 (collapse induction)
        phase2_length: Length of phase 2 (injection/recovery)
        graph1, graph2: Graphs for structured walks
        nl_loader: Natural language loader
        tokenizer: Model tokenizer
        trial_idx: Trial index

    Returns:
        Tuple of (token_ids, metadata)
    """
    # Generate Phase 1 tokens
    phase1_tokens, phase1_meta = generate_phase_tokens(
        phase_config=condition_config["phase1"],
        target_length=phase1_length,
        graph1=graph1,
        graph2=graph2,
        nl_loader=nl_loader,
        tokenizer=tokenizer,
        trial_idx=trial_idx,
    )

    # Generate Phase 2 tokens
    phase2_tokens, phase2_meta = generate_phase_tokens(
        phase_config=condition_config["phase2"],
        target_length=phase2_length,
        graph1=graph1,
        graph2=graph2,
        nl_loader=nl_loader,
        tokenizer=tokenizer,
        trial_idx=trial_idx,
    )

    # Combine tokens
    all_tokens = phase1_tokens + phase2_tokens

    metadata = {
        "injection_point": phase1_length,
        "total_length": len(all_tokens),
        "phase1": {
            "length": len(phase1_tokens),
            **phase1_meta,
        },
        "phase2": {
            "length": len(phase2_tokens),
            **phase2_meta,
        },
    }

    return all_tokens, metadata


def process_with_injection_tracking(
    model: HookedLLM,
    token_ids: list[int],
    checkpoints: list[int],
    injection_point: int,
    layers: list[int],
    window_size: int = 50,
    progress_bar: bool = True,
) -> dict:
    """
    Process tokens incrementally, tracking metrics with injection awareness.

    Separately tracks metrics for before/after injection for analysis.

    Args:
        model: HookedLLM instance
        token_ids: List of token IDs to process
        checkpoints: Context lengths at which to compute metrics
        injection_point: Token position where injection occurs
        layers: Layers to extract representations from
        window_size: Number of recent tokens for collapse metrics
        progress_bar: Whether to show progress bar

    Returns:
        Dict with phase1_results, phase2_results, and transition metrics
    """
    # Filter checkpoints to valid range
    valid_checkpoints = [cp for cp in checkpoints if cp <= len(token_ids)]

    # Initialize storage for recent representations per layer
    recent_reps = {layer: deque(maxlen=window_size) for layer in layers}

    # Separate results for each phase
    phase1_results = {}
    phase2_results = {}

    # Track metrics right before and after injection
    last_before = None
    first_after = None

    # Process incrementally
    past_kvs = None
    iterator = enumerate(token_ids)
    if progress_bar:
        iterator = tqdm(iterator, total=len(token_ids), desc="Processing context", leave=False)

    for i, token_id in iterator:
        # Forward pass with KV cache
        input_ids = torch.tensor([[token_id]]).to(model.device)

        try:
            _, cache, past_kvs = model.forward_incremental(
                input_ids,
                layers=layers,
                past_key_values=past_kvs,
            )

            # Store representation for each layer
            for layer in layers:
                rep = cache.get_residual_stream(layer)
                if rep is not None:
                    rep_np = rep[0, -1].cpu().float().numpy()
                    recent_reps[layer].append(rep_np)

        except Exception as e:
            warnings.warn(f"Error at position {i}: {e}")
            clear_gpu_memory()
            continue

        # At checkpoints, compute collapse metrics
        context_len = i + 1
        if context_len in valid_checkpoints:
            checkpoint_metrics = {}
            for layer in layers:
                reps_list = list(recent_reps[layer])
                if len(reps_list) >= 10:
                    metrics = compute_collapse_metrics(reps_list, compute_diagnostics=True)
                    checkpoint_metrics[layer] = metrics.to_dict()
                else:
                    checkpoint_metrics[layer] = None

            # Assign to appropriate phase
            if context_len <= injection_point:
                phase1_results[context_len] = checkpoint_metrics
                # Track last before injection
                if context_len == injection_point:
                    last_before = checkpoint_metrics
            else:
                phase2_results[context_len] = checkpoint_metrics
                # Track first after injection
                if first_after is None:
                    first_after = checkpoint_metrics

            if progress_bar:
                phase = "Phase 1" if context_len <= injection_point else "Phase 2"
                tqdm.write(f"  Checkpoint {context_len} ({phase}): metrics computed")

        # Periodic memory cleanup
        if (i + 1) % 5000 == 0:
            del cache
            clear_gpu_memory()

    # Compute transition delta
    transition = {
        "last_before": last_before,
        "first_after": first_after,
        "delta": None,
    }

    if last_before and first_after:
        delta = {}
        for layer in layers:
            if last_before.get(layer) and first_after.get(layer):
                layer_delta = {}
                for metric in ["avg_cos_sim", "avg_l2_dist", "spread", "effective_dim"]:
                    before_val = last_before[layer].get(metric)
                    after_val = first_after[layer].get(metric)
                    if before_val is not None and after_val is not None:
                        layer_delta[metric] = after_val - before_val
                delta[layer] = layer_delta
        transition["delta"] = delta

    return {
        "phase1": phase1_results,
        "phase2": phase2_results,
        "transition": transition,
    }


def run_single_trial(
    model: HookedLLM,
    condition: str,
    config: dict,
    phase1_length: int,
    phase2_length: int,
    layers: list[int],
    window_size: int,
    graph1: DualInterpretationGraph,
    graph2: DualInterpretationGraph,
    nl_loader: NaturalLanguageLoader,
    trial_idx: int,
) -> dict:
    """
    Run a single trial for a condition.

    Args:
        model: HookedLLM instance
        condition: Condition name
        config: Condition configuration
        phase1_length: Length of phase 1 (collapse induction)
        phase2_length: Length of phase 2 (injection/recovery)
        layers: Layers to analyze
        window_size: Window size for collapse metrics
        graph1, graph2: Graphs for structured walks
        nl_loader: Natural language loader
        trial_idx: Trial index

    Returns:
        Dict with trial results
    """
    try:
        # Generate tokens with injection
        token_ids, metadata = generate_tokens_with_injection(
            condition_config=config,
            phase1_length=phase1_length,
            phase2_length=phase2_length,
            graph1=graph1,
            graph2=graph2,
            nl_loader=nl_loader,
            tokenizer=model.tokenizer,
            trial_idx=trial_idx,
        )

        injection_point = metadata["injection_point"]
        total_length = len(token_ids)

        # Get checkpoint schedule
        checkpoints = get_checkpoint_schedule(injection_point, total_length)

        # Process with injection tracking
        results = process_with_injection_tracking(
            model=model,
            token_ids=token_ids,
            checkpoints=checkpoints,
            injection_point=injection_point,
            layers=layers,
            window_size=window_size,
            progress_bar=True,
        )

        return {
            "condition": condition,
            "trial_idx": trial_idx,
            "injection_point": injection_point,
            "n_tokens": len(token_ids),
            "metadata": convert_numpy(metadata),
            "checkpoints": checkpoints,
            "results": convert_numpy(results),
            "error": None,
        }

    except Exception as e:
        import traceback
        return {
            "condition": condition,
            "trial_idx": trial_idx,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def save_checkpoint(
    results: dict,
    output_dir: Path,
    condition: str,
    trial_idx: int,
):
    """Save intermediate results to checkpoint file."""
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"checkpoint_{condition}_trial_{trial_idx}.json"
    with open(checkpoint_path, 'w') as f:
        json.dump(convert_numpy(results), f, indent=2)


def run_experiment(args):
    """Main experiment runner."""

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw").mkdir(exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 70)
    print("COLLAPSE REVERSAL EXPERIMENT")
    print("Can Representational Collapse Be Reversed?")
    print("=" * 70)
    print(f"\nTimestamp: {timestamp}")
    print(f"Output directory: {output_dir}")

    # Parse conditions
    if args.conditions:
        conditions = [c.strip() for c in args.conditions.split(',')]
    else:
        conditions = list(CONDITIONS.keys())
    print(f"Conditions: {conditions}")

    # Validate conditions
    for cond in conditions:
        if cond not in CONDITIONS:
            print(f"ERROR: Unknown condition '{cond}'")
            print(f"Available conditions: {list(CONDITIONS.keys())}")
            return

    # Load model
    print("\n" + "-" * 70)
    print(f"Loading Model: {args.model}")
    print("-" * 70)

    try:
        model = HookedLLM.from_pretrained(
            args.model,
            device="auto",
            dtype=setup_dtype(args.dtype),
        )
        print(f"  Loaded successfully")
        print(f"  Layers: {model.num_layers}, Hidden size: {model.hidden_size}")
    except Exception as e:
        print(f"  ERROR loading model: {e}")
        return

    # Determine layers to analyze
    if args.layers:
        layers = [int(x) for x in args.layers.split(',')]
    else:
        # Sample layers across depth
        n_layers = model.num_layers
        layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
        layers = sorted(list(set(layers)))
    print(f"  Layers to analyze: {layers}")

    # Create graphs
    print("\n" + "-" * 70)
    print("Creating Data Sources")
    print("-" * 70)

    # Graph 1: Primary graph for most conditions
    graph1_config = DualInterpretationConfig(
        vocab_size=15,
        clusters_per_interpretation=3,
        seed=args.seed,
    )
    graph1 = DualInterpretationGraph(graph1_config)
    print(f"  Graph 1: {graph1.get_graph_statistics()}")

    # Graph 2: Different graph for "inject_different_graph" condition
    # Use different vocabulary and seed
    graph2_vocab = generate_extended_vocabulary(15, seed=args.seed + 1000)
    graph2_config = DualInterpretationConfig(
        vocab_size=15,
        clusters_per_interpretation=3,
        vocabulary=graph2_vocab,
        seed=args.seed + 1000,
    )
    graph2 = DualInterpretationGraph(graph2_config)
    print(f"  Graph 2: {graph2.get_graph_statistics()}")

    # Natural language loader
    nl_config = NaturalLanguageConfig(seed=args.seed)
    nl_loader = NaturalLanguageLoader(model.tokenizer, nl_config)
    print(f"  Natural language loader: initialized")

    # Calculate injection point and total length
    injection_point = args.phase1_length
    total_length = args.phase1_length + args.phase2_length
    checkpoints = get_checkpoint_schedule(injection_point, total_length)

    # Save experiment config
    experiment_config = {
        "model": args.model,
        "phase1_length": args.phase1_length,
        "phase2_length": args.phase2_length,
        "injection_point": injection_point,
        "total_length": total_length,
        "checkpoints": checkpoints,
        "layers": layers,
        "window_size": args.window_size,
        "n_trials": args.n_trials,
        "conditions": conditions,
        "seed": args.seed,
        "timestamp": timestamp,
        "dtype": args.dtype,
    }

    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(experiment_config, f, indent=2)
    print(f"\n  Config saved to: {config_path}")

    # Run experiment
    print("\n" + "-" * 70)
    print("Running Experiment")
    print("-" * 70)
    print(f"  Phase 1 (collapse induction): {args.phase1_length} tokens")
    print(f"  Injection point: {injection_point}")
    print(f"  Phase 2 (reversal test): {args.phase2_length} tokens")
    print(f"  Total: {total_length} tokens")

    all_results = {
        "config": experiment_config,
        "conditions": {},
    }

    for condition in conditions:
        print(f"\n{'='*60}")
        print(f"Condition: {condition}")
        print(f"{'='*60}")

        config = CONDITIONS[condition]
        print(f"  Description: {config['description']}")
        print(f"  Phase 1: {config['phase1']}")
        print(f"  Phase 2: {config['phase2']}")

        condition_results = []

        for trial_idx in range(args.n_trials):
            print(f"\n  Trial {trial_idx + 1}/{args.n_trials}")

            result = run_single_trial(
                model=model,
                condition=condition,
                config=config,
                phase1_length=args.phase1_length,
                phase2_length=args.phase2_length,
                layers=layers,
                window_size=args.window_size,
                graph1=graph1,
                graph2=graph2,
                nl_loader=nl_loader,
                trial_idx=trial_idx,
            )

            if result.get("error"):
                print(f"    ERROR: {result['error']}")
            else:
                print(f"    Completed: {result['n_tokens']} tokens")
                if result.get("results", {}).get("transition", {}).get("delta"):
                    delta = result["results"]["transition"]["delta"]
                    # Report delta for middle layer
                    mid_layer = layers[len(layers) // 2]
                    if mid_layer in delta:
                        d = delta[mid_layer]
                        print(f"    Transition delta (layer {mid_layer}):")
                        print(f"      cos_sim: {d.get('avg_cos_sim', 0):+.4f}")
                        print(f"      eff_dim: {d.get('effective_dim', 0):+.1f}")

            condition_results.append(result)

            # Save raw trial result
            raw_path = output_dir / "raw" / f"{condition}_trial_{trial_idx}.json"
            with open(raw_path, 'w') as f:
                json.dump(convert_numpy(result), f, indent=2)

            # Save checkpoint
            save_checkpoint(result, output_dir, condition, trial_idx)

            # Clear GPU memory
            clear_gpu_memory()

        all_results["conditions"][condition] = {
            "config": config,
            "n_trials_completed": len([r for r in condition_results if not r.get("error")]),
            "trials_summary": [
                {
                    "trial_idx": r["trial_idx"],
                    "n_tokens": r.get("n_tokens"),
                    "error": r.get("error"),
                }
                for r in condition_results
            ],
        }

    # Save final results
    results_path = output_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(convert_numpy(all_results), f, indent=2)
    print(f"\n  Results saved to: {results_path}")

    # Cleanup
    del model
    clear_gpu_memory()

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"\nResults directory: {output_dir}")
    print("Key outputs:")
    print(f"  - {output_dir}/config.json")
    print(f"  - {output_dir}/results.json")
    print(f"  - {output_dir}/raw/*.json (per-trial results)")
    print(f"  - {output_dir}/checkpoints/*.json (intermediate checkpoints)")


def main():
    parser = argparse.ArgumentParser(
        description="Run collapse reversal experiment - test if representational collapse can be reversed"
    )

    # Model configuration
    parser.add_argument("--model", type=str, default="gpt2",
                       help="HuggingFace model name (default: gpt2)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                       choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--layers", type=str, default=None,
                       help="Comma-separated layer indices (default: sample across depth)")

    # Experiment configuration
    parser.add_argument("--phase1-length", type=int, default=5000,
                       help="Phase 1 length in tokens (collapse induction, default: 5000)")
    parser.add_argument("--phase2-length", type=int, default=5000,
                       help="Phase 2 length in tokens (reversal test, default: 5000)")
    parser.add_argument("--window-size", type=int, default=50,
                       help="Window size for collapse metrics (default: 50)")
    parser.add_argument("--n-trials", type=int, default=5,
                       help="Number of trials per condition (default: 5)")
    parser.add_argument("--conditions", type=str, default=None,
                       help="Comma-separated condition names (default: all)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")

    # Output configuration
    parser.add_argument("--output-dir", type=str, default="results/collapse_reversal",
                       help="Output directory")

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Run experiment
    run_experiment(args)


if __name__ == "__main__":
    main()
