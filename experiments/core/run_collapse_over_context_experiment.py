#!/usr/bin/env python3
"""
Collapse Over Context Experiment: Track Representational Collapse Up to 128K Tokens.

This experiment tests how token representations collapse (converge to similar points)
over context length, comparing:
1. Structured walks with/without ambiguity and disambiguation
2. Natural language contexts (Wikipedia, books, conversations)
3. Different vocabulary sizes (token diversity)

Key hypothesis: Longer context leads to representational collapse regardless of
content structure, but disambiguation timing and token diversity affect the rate.

Collapse Metrics:
- avg_cos_sim: Mean pairwise cosine similarity (increases toward 1.0 as collapse happens)
- spread: Total variance (decreases toward 0 as collapse happens)
- avg_l2_dist: Mean pairwise L2 distance (decreases as collapse happens)
- effective_dim: Participation ratio (decreases as variance concentrates)

Usage:
    # Quick test
    python run_collapse_over_context_experiment.py --model gpt2 --n-trials 1 \
        --checkpoints 100,500,1000 --conditions structured_no_ambig

    # Medium test (up to 8K)
    python run_collapse_over_context_experiment.py --model Qwen/Qwen2.5-7B \
        --n-trials 3 --max-context 8000

    # Full experiment (128K, ~8-24 hours)
    python run_collapse_over_context_experiment.py --model Qwen/Qwen2.5-7B \
        --n-trials 10

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


# Default checkpoint schedule: dense early, sparse late (for 10K context)
DEFAULT_CHECKPOINTS = [
    10, 20, 30, 50, 75, 100, 125, 150, 175, 200,
    250, 300, 350, 400, 450, 500, 600, 700, 800, 900,
    1000, 1250, 1500, 1750, 2000, 2500, 3000, 3500, 4000, 4500,
    5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000
]

# Disambiguation percentages: very dense at start
DISAMBIG_PCTS = [
    0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.12, 0.15,
    0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80,
    0.85, 0.90, 0.95, 0.98, 0.99
]

# Condition configurations
CONDITION_CONFIGS = {
    # Structured walks - no ambiguity
    "structured_no_ambig": {
        "type": "structured",
        "walk_type": "h1_only",
        "description": "H1-only walk (no ambiguity from start)",
    },
    # Structured walks - full ambiguity (never disambiguated)
    "structured_full_ambig": {
        "type": "structured",
        "walk_type": "ambiguous",
        "disambig_pct": None,
        "description": "Fully ambiguous walk (never disambiguated)",
    },
    # Natural language
    "natural_wikipedia": {
        "type": "natural",
        "source": "wikipedia",
        "description": "Wikipedia articles",
    },
    "natural_books": {
        "type": "natural",
        "source": "book",
        "description": "Project Gutenberg books",
    },
    "natural_conversation": {
        "type": "natural",
        "source": "conversation",
        "description": "Multi-turn conversations (OpenAssistant)",
    },
    "natural_wildchat": {
        "type": "natural",
        "source": "wildchat",
        "description": "Long user conversations (WildChat, 10+ turns)",
    },
    # Vocabulary size variations
    "vocab_15": {
        "type": "structured",
        "walk_type": "ambiguous",
        "disambig_pct": 0.50,
        "vocab_size": 15,
        "description": "Minimal vocabulary (15 tokens)",
    },
    "vocab_50": {
        "type": "structured",
        "walk_type": "ambiguous",
        "disambig_pct": 0.50,
        "vocab_size": 50,
        "description": "Small vocabulary (50 tokens)",
    },
    "vocab_200": {
        "type": "structured",
        "walk_type": "ambiguous",
        "disambig_pct": 0.50,
        "vocab_size": 200,
        "description": "Medium vocabulary (200 tokens)",
    },
    "vocab_1000": {
        "type": "structured",
        "walk_type": "ambiguous",
        "disambig_pct": 0.50,
        "vocab_size": 1000,
        "description": "Large vocabulary (1000 tokens)",
    },
}

# Add disambiguation conditions dynamically (25 points)
for pct in DISAMBIG_PCTS:
    pct_str = f"{pct:.1%}".replace(".", "_").replace("%", "pct")
    key = f"structured_disambig_{pct_str}"
    CONDITION_CONFIGS[key] = {
        "type": "structured",
        "walk_type": "ambiguous",
        "disambig_pct": pct,
        "description": f"Disambiguated at {pct:.1%} of context",
    }


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


def generate_structured_tokens(
    graph: DualInterpretationGraph,
    tokenizer,
    target_length: int,
    walk_type: str,
    disambig_pct: Optional[float],
    true_hypothesis: str = "H1",
) -> tuple[list[int], dict]:
    """
    Generate structured walk tokens.

    Args:
        graph: DualInterpretationGraph instance
        tokenizer: HuggingFace tokenizer
        target_length: Target number of tokens
        walk_type: "h1_only" or "ambiguous"
        disambig_pct: Disambiguation position as fraction (None = never)
        true_hypothesis: "H1" or "H2"

    Returns:
        Tuple of (token_ids, metadata)
    """
    # Calculate walk length (may need multiple walks for very long sequences)
    # Each walk word typically becomes 1 token, but some may split
    walk_length = target_length * 2  # Generate extra to account for tokenization

    all_tokens = []
    all_metadata = {
        "walk_type": walk_type,
        "disambig_pct": disambig_pct,
        "true_hypothesis": true_hypothesis,
        "H1_clusters": [],
        "H2_clusters": [],
    }

    while len(all_tokens) < target_length:
        # Calculate disambiguation position for this walk segment
        if disambig_pct is not None:
            disambig_pos = int(walk_length * disambig_pct)
        else:
            disambig_pos = None

        if walk_type == "h1_only":
            prompt, nodes, meta = graph.generate_h1_only_walk(
                length=walk_length,
                return_nodes=True,
            )
        else:  # ambiguous
            prompt, nodes, meta = graph.generate_ambiguous_walk(
                length=walk_length,
                disambig_position=disambig_pos,
                true_hypothesis=true_hypothesis,
                return_nodes=True,
            )

        # Tokenize
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        all_tokens.extend(tokens)

        # Track cluster labels (map tokens to cluster labels)
        # Note: tokenization may not preserve 1:1 mapping, but we track original walk
        all_metadata["H1_clusters"].extend(meta["H1_clusters"])
        all_metadata["H2_clusters"].extend(meta["H2_clusters"])

    # Truncate to target length
    return all_tokens[:target_length], all_metadata


def generate_natural_tokens(
    loader: NaturalLanguageLoader,
    source: str,
    target_length: int,
) -> tuple[list[int], dict]:
    """
    Generate natural language tokens.

    Args:
        loader: NaturalLanguageLoader instance
        source: "wikipedia", "book", "conversation", or "wildchat"
        target_length: Target number of tokens

    Returns:
        Tuple of (token_ids, metadata)
    """
    if source == "wikipedia":
        tokens = loader.load_wikipedia(target_length)
    elif source == "book":
        tokens = loader.load_book(target_length)
    elif source == "conversation":
        tokens = loader.load_conversation(target_length)
    elif source == "wildchat":
        tokens = loader.load_wildchat_conversation(target_length, min_turns=10)
    else:
        raise ValueError(f"Unknown source: {source}")

    metadata = {
        "source": source,
        "actual_length": len(tokens),
    }

    return tokens, metadata


def process_long_context(
    model: HookedLLM,
    token_ids: list[int],
    checkpoints: list[int],
    layers: list[int],
    window_size: int = 50,
    progress_bar: bool = True,
) -> dict:
    """
    Process tokens incrementally, computing collapse metrics at checkpoints.

    Uses KV-cache for efficient incremental processing.

    Args:
        model: HookedLLM instance
        token_ids: List of token IDs to process
        checkpoints: Context lengths at which to compute metrics
        layers: Layers to extract representations from
        window_size: Number of recent tokens to use for collapse metrics
        progress_bar: Whether to show progress bar

    Returns:
        Dict mapping checkpoint -> layer -> CollapseMetrics
    """
    # Filter checkpoints to valid range
    valid_checkpoints = [cp for cp in checkpoints if cp <= len(token_ids)]

    # Initialize storage for recent representations per layer
    recent_reps = {layer: deque(maxlen=window_size) for layer in layers}

    # Results storage
    results = {cp: {} for cp in valid_checkpoints}

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
            # Handle OOM or other errors
            warnings.warn(f"Error at position {i}: {e}")
            clear_gpu_memory()
            continue

        # At checkpoints, compute collapse metrics
        context_len = i + 1
        if context_len in valid_checkpoints:
            for layer in layers:
                reps_list = list(recent_reps[layer])
                if len(reps_list) >= 10:  # Need enough samples
                    metrics = compute_collapse_metrics(reps_list, compute_diagnostics=True)
                    results[context_len][layer] = metrics.to_dict()
                else:
                    results[context_len][layer] = None

            # Optional: print progress
            if progress_bar:
                tqdm.write(f"  Checkpoint {context_len}: computed metrics for {len(layers)} layers")

        # Periodic memory cleanup
        if (i + 1) % 5000 == 0:
            del cache
            clear_gpu_memory()

    return results


def run_single_trial(
    model: HookedLLM,
    condition: str,
    config: dict,
    checkpoints: list[int],
    layers: list[int],
    window_size: int,
    graph: Optional[DualInterpretationGraph],
    nl_loader: Optional[NaturalLanguageLoader],
    max_context: int,
    trial_idx: int,
) -> dict:
    """
    Run a single trial for a condition.

    Args:
        model: HookedLLM instance
        condition: Condition name
        config: Condition configuration
        checkpoints: Context length checkpoints
        layers: Layers to analyze
        window_size: Window size for collapse metrics
        graph: DualInterpretationGraph (for structured conditions)
        nl_loader: NaturalLanguageLoader (for natural language conditions)
        max_context: Maximum context length
        trial_idx: Trial index (for alternating hypotheses)

    Returns:
        Dict with trial results
    """
    try:
        # Generate tokens based on condition type
        if config["type"] == "structured":
            # Create graph with appropriate vocab size
            vocab_size = config.get("vocab_size", 15)
            if vocab_size != 15:
                trial_graph = create_graph_with_vocab_size(
                    vocab_size=vocab_size,
                    seed=42 + trial_idx,
                )
            else:
                trial_graph = graph

            true_hyp = "H1" if trial_idx % 2 == 0 else "H2"
            token_ids, metadata = generate_structured_tokens(
                graph=trial_graph,
                tokenizer=model.tokenizer,
                target_length=max_context,
                walk_type=config["walk_type"],
                disambig_pct=config.get("disambig_pct"),
                true_hypothesis=true_hyp,
            )
        else:  # natural language
            token_ids, metadata = generate_natural_tokens(
                loader=nl_loader,
                source=config["source"],
                target_length=max_context,
            )

        # Filter checkpoints to actual token length
        valid_checkpoints = [cp for cp in checkpoints if cp <= len(token_ids)]

        # Process context and compute metrics
        results = process_long_context(
            model=model,
            token_ids=token_ids,
            checkpoints=valid_checkpoints,
            layers=layers,
            window_size=window_size,
            progress_bar=True,
        )

        return {
            "condition": condition,
            "trial_idx": trial_idx,
            "n_tokens": len(token_ids),
            "metadata": convert_numpy(metadata),
            "results": convert_numpy(results),
            "error": None,
        }

    except Exception as e:
        return {
            "condition": condition,
            "trial_idx": trial_idx,
            "error": str(e),
        }


def save_checkpoint(
    results: dict,
    output_dir: Path,
    condition: str,
    checkpoint_idx: int,
):
    """Save intermediate results to checkpoint file."""
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"checkpoint_{condition}_{checkpoint_idx}.json"
    with open(checkpoint_path, 'w') as f:
        json.dump(convert_numpy(results), f, indent=2)


def load_checkpoint(
    output_dir: Path,
    condition: str,
) -> Optional[dict]:
    """Load latest checkpoint for a condition if exists."""
    checkpoint_dir = output_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return None

    checkpoints = sorted(checkpoint_dir.glob(f"checkpoint_{condition}_*.json"))
    if checkpoints:
        with open(checkpoints[-1]) as f:
            return json.load(f)
    return None


def run_experiment(args):
    """Main experiment runner."""

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw").mkdir(exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 70)
    print("COLLAPSE OVER CONTEXT EXPERIMENT")
    print("Track Representational Collapse Up to 128K Tokens")
    print("=" * 70)
    print(f"\nTimestamp: {timestamp}")
    print(f"Output directory: {output_dir}")

    # Parse checkpoints
    if args.checkpoints:
        checkpoints = [int(x) for x in args.checkpoints.split(',')]
    else:
        checkpoints = [cp for cp in DEFAULT_CHECKPOINTS if cp <= args.max_context]
    print(f"Checkpoints: {checkpoints}")

    # Parse conditions
    if args.conditions:
        conditions = [c.strip() for c in args.conditions.split(',')]
    else:
        conditions = list(CONDITION_CONFIGS.keys())
    print(f"Conditions: {conditions}")

    # Validate conditions
    for cond in conditions:
        if cond not in CONDITION_CONFIGS:
            print(f"ERROR: Unknown condition '{cond}'")
            print(f"Available conditions: {list(CONDITION_CONFIGS.keys())}")
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

    # Create graph for structured conditions
    print("\n" + "-" * 70)
    print("Creating Data Sources")
    print("-" * 70)

    graph_config = DualInterpretationConfig(
        vocab_size=15,
        clusters_per_interpretation=3,
        seed=args.seed,
    )
    graph = DualInterpretationGraph(graph_config)
    print(f"  Structured graph: {graph.get_graph_statistics()}")

    # Create natural language loader
    nl_config = NaturalLanguageConfig(seed=args.seed)
    nl_loader = NaturalLanguageLoader(model.tokenizer, nl_config)
    print(f"  Natural language loader: initialized")

    # Save experiment config
    experiment_config = {
        "model": args.model,
        "max_context": args.max_context,
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

    all_results = {
        "config": experiment_config,
        "conditions": {},
    }

    for condition in conditions:
        print(f"\n{'='*60}")
        print(f"Condition: {condition}")
        print(f"{'='*60}")

        config = CONDITION_CONFIGS[condition]
        print(f"  Description: {config['description']}")

        condition_results = []

        for trial_idx in range(args.n_trials):
            print(f"\n  Trial {trial_idx + 1}/{args.n_trials}")

            result = run_single_trial(
                model=model,
                condition=condition,
                config=config,
                checkpoints=checkpoints,
                layers=layers,
                window_size=args.window_size,
                graph=graph,
                nl_loader=nl_loader,
                max_context=args.max_context,
                trial_idx=trial_idx,
            )

            if result.get("error"):
                print(f"    ERROR: {result['error']}")
            else:
                print(f"    Completed: {result['n_tokens']} tokens")

            condition_results.append(result)

            # Save raw trial result
            raw_path = output_dir / "raw" / f"{condition}_trial_{trial_idx}.json"
            with open(raw_path, 'w') as f:
                json.dump(convert_numpy(result), f, indent=2)

            # Save checkpoint after each trial
            save_checkpoint(
                {"trials": condition_results},
                output_dir,
                condition,
                trial_idx,
            )

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
        description="Run collapse over context experiment - track representational collapse up to 128K tokens"
    )

    # Model configuration
    parser.add_argument("--model", type=str, default="gpt2",
                       help="HuggingFace model name (default: gpt2)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                       choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--layers", type=str, default=None,
                       help="Comma-separated layer indices (default: sample across depth)")

    # Experiment configuration
    parser.add_argument("--max-context", type=int, default=128000,
                       help="Maximum context length (default: 128000)")
    parser.add_argument("--checkpoints", type=str, default=None,
                       help="Comma-separated checkpoint positions (default: standard schedule)")
    parser.add_argument("--window-size", type=int, default=50,
                       help="Window size for collapse metrics (default: 50)")
    parser.add_argument("--n-trials", type=int, default=10,
                       help="Number of trials per condition (default: 10)")
    parser.add_argument("--conditions", type=str, default=None,
                       help="Comma-separated condition names (default: all)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")

    # Output configuration
    parser.add_argument("--output-dir", type=str, default="results/collapse_over_context",
                       help="Output directory")

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Run experiment
    run_experiment(args)


if __name__ == "__main__":
    main()
