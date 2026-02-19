#!/usr/bin/env python3
"""
Context Variation Experiments: Misspellings, Topic Changes, Extended Length

Three experiments testing how context properties affect collapse and knowledge retrieval:

1. MISSPELLINGS: Natural book text with 5%, 10%, 25%, 50% misspelling rates
   - Tests: Is collapse about vocabulary restriction or syntactic/semantic structure?

2. TOPIC CHANGES: Single book vs rapid switching between books (every 300 tokens)
   - Tests: Does topic diversity maintain representational diversity?

3. EXTENDED LENGTH: Natural book text at 30K and 50K tokens
   - Tests: Does natural language's ~97% accuracy hold at very long contexts?

Usage:
    # Quick test
    PYTHONPATH=. python experiments/core/run_context_variation_experiments.py --quick-test

    # Run specific experiment
    PYTHONPATH=. python experiments/core/run_context_variation_experiments.py --experiment misspellings
    PYTHONPATH=. python experiments/core/run_context_variation_experiments.py --experiment topic_changes
    PYTHONPATH=. python experiments/core/run_context_variation_experiments.py --experiment extended_length

    # Run all experiments (default)
    PYTHONPATH=. python experiments/core/run_context_variation_experiments.py --experiment all
"""

import argparse
import json
import gc
import random
import string
from collections import deque
from datetime import datetime
from pathlib import Path

import torch
import numpy as np

from src.models import HookedLLM
from src.data.natural_language_loader import NaturalLanguageLoader, NaturalLanguageConfig
from src.data.dual_interpretation_graph import DualInterpretationGraph, DualInterpretationConfig
from src.metrics.collapse_metrics import compute_collapse_metrics

# Import evaluation infrastructure from existing probing experiment
from experiments.core.run_probing_collapse_performance import (
    QUESTIONS,
    screen_questions,
    format_question,
    check_answer_correct,
    evaluate_question,
    process_context_chunks,
    convert_numpy,
    clear_gpu_memory,
)


# ── Misspelling Functions ─────────────────────────────────────────────────

def _misspell_word(word: str, rng: random.Random) -> str:
    """Apply a random misspelling to a word."""
    chars = list(word)
    action = rng.choice(["swap", "delete", "insert", "substitute"])

    if action == "swap" and len(chars) > 1:
        idx = rng.randint(0, len(chars) - 2)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
    elif action == "delete" and len(chars) > 1:
        idx = rng.randint(0, len(chars) - 1)
        chars.pop(idx)
    elif action == "insert":
        idx = rng.randint(0, len(chars))
        chars.insert(idx, rng.choice(string.ascii_lowercase))
    elif action == "substitute":
        idx = rng.randint(0, len(chars) - 1)
        chars[idx] = rng.choice(string.ascii_lowercase)

    return "".join(chars)


def introduce_misspellings(text: str, rate: float, rng: random.Random) -> str:
    """Introduce random misspellings at the given per-word rate."""
    if rate <= 0:
        return text
    words = text.split()
    result = []
    for word in words:
        # Only misspell words with >2 chars (skip punctuation, short words)
        if len(word) > 2 and word.isalpha() and rng.random() < rate:
            word = _misspell_word(word, rng)
        result.append(word)
    return " ".join(result)


# ── Context Generation ────────────────────────────────────────────────────

def _gather_book_text(
    nl_loader: NaturalLanguageLoader,
    target_chars: int,
    trial_idx: int,
) -> str:
    """Gather enough book text, concatenating multiple books if needed."""
    rng = random.Random(42 + trial_idx)
    book_ids = list(nl_loader.config.gutenberg_ids)
    rng.shuffle(book_ids)

    text_parts = []
    total_chars = 0

    for bid in book_ids:
        book_text = _load_book_text(nl_loader, bid)
        if book_text:
            text_parts.append(book_text)
            total_chars += len(book_text)
            if total_chars >= target_chars:
                break

    return "\n\n".join(text_parts)


def generate_misspelled_book_tokens(
    model: HookedLLM,
    nl_loader: NaturalLanguageLoader,
    target_length: int,
    misspelling_rate: float,
    trial_idx: int,
) -> list[int]:
    """Load book text, introduce misspellings, then tokenize."""
    # Gather enough text (concatenate books if needed, ~6 chars per token)
    needed_chars = target_length * 8
    text = _gather_book_text(nl_loader, needed_chars, trial_idx)

    # Pick a random starting position in the text (character-level)
    rng = random.Random(42 + trial_idx * 7)
    if len(text) > target_length * 6:
        start = rng.randint(0, len(text) - target_length * 6)
        text = text[start : start + target_length * 8]

    # Introduce misspellings
    misspell_rng = random.Random(42 + trial_idx * 13 + int(misspelling_rate * 100))
    text = introduce_misspellings(text, misspelling_rate, misspell_rng)

    # Tokenize
    tokens = model.tokenizer.encode(text, add_special_tokens=False)
    return tokens[:target_length]


def _load_book_text(nl_loader: NaturalLanguageLoader, gutenberg_id: int) -> str:
    """Load raw book text from cache or download."""
    cache_path = Path(nl_loader.config.cache_dir) / f"gutenberg_{gutenberg_id}.txt"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()
    # Trigger download via loader, then read cache
    nl_loader.load_book(100, gutenberg_id=gutenberg_id)
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def generate_multi_topic_tokens(
    model: HookedLLM,
    nl_loader: NaturalLanguageLoader,
    target_length: int,
    trial_idx: int,
    excerpt_tokens: int = 300,
) -> list[int]:
    """Generate tokens switching between different books every excerpt_tokens tokens.

    Samples random positions within each book (not just the beginning).
    """
    book_ids = list(nl_loader.config.gutenberg_ids)
    rng = random.Random(42 + trial_idx * 100)
    rng.shuffle(book_ids)

    # Pre-load all book texts
    book_texts = {}
    for bid in book_ids:
        text = _load_book_text(nl_loader, bid)
        if text:
            book_texts[bid] = text

    if not book_texts:
        raise RuntimeError("Could not load any books for multi-topic generation")

    available_ids = list(book_texts.keys())
    # Estimate chars per excerpt (~5 chars per token)
    chars_per_excerpt = excerpt_tokens * 6

    all_tokens = []
    cycle = 0

    while len(all_tokens) < target_length:
        bid = available_ids[cycle % len(available_ids)]
        text = book_texts[bid]

        # Pick a random position in the book
        if len(text) > chars_per_excerpt:
            start = rng.randint(0, len(text) - chars_per_excerpt)
            chunk_text = text[start : start + chars_per_excerpt]
        else:
            chunk_text = text

        # Tokenize this random excerpt
        excerpt = model.tokenizer.encode(chunk_text, add_special_tokens=False)
        excerpt = excerpt[:excerpt_tokens]

        all_tokens.extend(excerpt)
        cycle += 1

    return all_tokens[:target_length]


def generate_single_topic_tokens(
    model: HookedLLM,
    nl_loader: NaturalLanguageLoader,
    target_length: int,
    trial_idx: int,
) -> list[int]:
    """Generate tokens from a single continuous book (same as natural_books)."""
    nl_loader.rng = random.Random(42 + trial_idx)
    tokens = nl_loader.load_book(target_length)
    return tokens[:target_length]


# ── Shared Evaluation Loop ────────────────────────────────────────────────

def run_single_experiment(
    name: str,
    conditions: list[dict],
    model: HookedLLM,
    encoded_questions: list[dict],
    layers: list[int],
    output_dir: Path,
    chunk_size: int = 512,
):
    """
    Run one experiment with multiple conditions.

    Each condition is a dict with:
        - name: str
        - context_lengths: list[int]
        - n_trials: int
        - generate_fn: callable(length, trial_idx) -> list[int]
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw").mkdir(exist_ok=True)

    all_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save config
    config = {
        "experiment": name,
        "conditions": [
            {
                "name": c["name"],
                "context_lengths": c["context_lengths"],
                "n_trials": c["n_trials"],
            }
            for c in conditions
        ],
        "layers": layers,
        "n_questions": len(encoded_questions),
        "chunk_size": chunk_size,
        "timestamp": timestamp,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    total_contexts = sum(
        len(c["context_lengths"]) * c["n_trials"] for c in conditions
    )
    ctx_count = 0

    for cond in conditions:
        cond_name = cond["name"]
        print(f"\n{'='*60}")
        print(f"Condition: {cond_name}")
        print(f"{'='*60}")

        for ctx_len in cond["context_lengths"]:
            for trial_idx in range(cond["n_trials"]):
                ctx_count += 1
                label = f"{cond_name}_len{ctx_len}_trial{trial_idx}"
                print(f"\n  [{ctx_count}/{total_contexts}] {label}")

                # Generate context tokens
                context_tokens = cond["generate_fn"](ctx_len, trial_idx)
                actual_len = len(context_tokens)
                print(f"    Generated {actual_len:,} tokens")

                # Process context -> KV cache + collapse metrics
                past_kvs, collapse_metrics = process_context_chunks(
                    model,
                    context_tokens,
                    layers,
                    chunk_size=chunk_size,
                    window_size=50,
                )

                if collapse_metrics:
                    last_layer = layers[-1]
                    cm = collapse_metrics.get(last_layer)
                    if cm:
                        print(
                            f"    Collapse L{last_layer}: "
                            f"cos_sim={cm['avg_cos_sim']:.3f}, "
                            f"eff_dim={cm['effective_dim']:.1f}"
                        )

                # Evaluate each question
                trial_results = []
                for eq in encoded_questions:
                    log_prob, generated = evaluate_question(
                        model,
                        eq["prompt_ids"],
                        eq["answer_ids"],
                        context_past_kvs=past_kvs,
                        max_new_tokens=30,
                    )
                    correct = check_answer_correct(generated, eq["a"])

                    result = {
                        "condition": cond_name,
                        "context_length": ctx_len,
                        "actual_context_length": actual_len,
                        "trial_idx": trial_idx,
                        "question": eq["q"],
                        "expected_answer": eq["a"],
                        "category": eq["category"],
                        "generated_answer": generated,
                        "answer_correct": correct,
                        "answer_log_prob": log_prob,
                        "collapse_metrics": convert_numpy(collapse_metrics),
                    }
                    trial_results.append(result)
                    all_results.append(result)

                # Log trial summary
                n_correct = sum(r["answer_correct"] for r in trial_results)
                mean_lp = np.mean([r["answer_log_prob"] for r in trial_results])
                print(
                    f"    Accuracy: {n_correct}/{len(trial_results)} "
                    f"({100 * n_correct / len(trial_results):.1f}%)"
                )
                print(f"    Mean log-prob: {mean_lp:.3f}")

                # Save raw trial
                raw_path = output_dir / "raw" / f"{label}.json"
                with open(raw_path, "w") as f:
                    json.dump(convert_numpy(trial_results), f, indent=2)

                del past_kvs, collapse_metrics
                clear_gpu_memory()

    # ── Aggregate Results ──
    print(f"\n{'='*60}")
    print("AGGREGATING RESULTS")
    print(f"{'='*60}")

    aggregated = {}
    for cond in conditions:
        cond_name = cond["name"]
        aggregated[cond_name] = {}
        for ctx_len in cond["context_lengths"]:
            trials = [
                r
                for r in all_results
                if r["condition"] == cond_name and r["context_length"] == ctx_len
            ]
            if not trials:
                continue

            # Collapse stats from last layer
            collapse_cos_sims = []
            collapse_eff_dims = []
            for r in trials:
                cm = r.get("collapse_metrics", {})
                if cm:
                    layer_cm = cm.get(str(layers[-1]))
                    if layer_cm:
                        collapse_cos_sims.append(layer_cm["avg_cos_sim"])
                        collapse_eff_dims.append(layer_cm["effective_dim"])

            aggregated[cond_name][ctx_len] = {
                "accuracy": float(np.mean([r["answer_correct"] for r in trials])),
                "mean_log_prob": float(np.mean([r["answer_log_prob"] for r in trials])),
                "std_log_prob": float(np.std([r["answer_log_prob"] for r in trials])),
                "n_evaluations": len(trials),
                "collapse_cos_sim_mean": (
                    float(np.mean(collapse_cos_sims)) if collapse_cos_sims else None
                ),
                "collapse_eff_dim_mean": (
                    float(np.mean(collapse_eff_dims)) if collapse_eff_dims else None
                ),
            }

    # Print summary
    print(f"\n{'-'*70}")
    print(f"SUMMARY: {name}")
    print(f"{'-'*70}")
    print(f"{'Condition':<25} {'Length':<8} {'Accuracy':<10} {'LogProb':<10} {'CosSim':<10} {'EffDim':<10}")
    print(f"{'-'*70}")
    for cond_name, lengths in aggregated.items():
        for ctx_len, agg in sorted(lengths.items()):
            cs = f"{agg['collapse_cos_sim_mean']:.3f}" if agg["collapse_cos_sim_mean"] is not None else "N/A"
            ed = f"{agg['collapse_eff_dim_mean']:.1f}" if agg["collapse_eff_dim_mean"] is not None else "N/A"
            print(
                f"{cond_name:<25} {ctx_len:<8} "
                f"{agg['accuracy']:<10.3f} "
                f"{agg['mean_log_prob']:<10.3f} "
                f"{cs:<10} {ed:<10}"
            )

    # Save final results
    final = {
        "config": config,
        "aggregated": convert_numpy(aggregated),
        "n_total_evaluations": len(all_results),
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(final, f, indent=2)
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(convert_numpy(all_results), f, indent=2)

    print(f"\nResults saved to {output_dir}")
    return aggregated


# ── Experiment Definitions ────────────────────────────────────────────────

def run_misspellings_experiment(model, encoded_questions, layers, nl_loader, args):
    """Experiment 1: Natural language with varying misspelling rates."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: MISSPELLINGS")
    print("=" * 70)

    rates = [0.10, 0.25, 0.50]
    lengths = [20000, 50000, 80000, 100000, 128000]
    n_trials = 3

    if args.quick_test:
        rates = [0.10, 0.50]
        lengths = [1000]
        n_trials = 1

    # CLI overrides
    if args.context_lengths:
        lengths = [int(x.strip()) for x in args.context_lengths.split(",")]
    if args.n_trials:
        n_trials = args.n_trials

    conditions = []
    for rate in rates:
        pct = int(rate * 100)
        conditions.append(
            {
                "name": f"misspell_{pct}pct",
                "context_lengths": lengths,
                "n_trials": n_trials,
                "generate_fn": lambda length, trial_idx, r=rate: generate_misspelled_book_tokens(
                    model, nl_loader, length, r, trial_idx
                ),
            }
        )

    output_dir = Path(args.output_dir) / "misspellings"
    return run_single_experiment(
        "misspellings", conditions, model, encoded_questions, layers, output_dir,
        chunk_size=args.chunk_size,
    )


def run_topic_change_experiment(model, encoded_questions, layers, nl_loader, args):
    """Experiment 2: Single topic vs multi-topic context."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: TOPIC CHANGES")
    print("=" * 70)

    lengths = [20000, 50000, 80000, 100000, 128000]
    n_trials = 3

    if args.quick_test:
        lengths = [1000]
        n_trials = 1

    # CLI overrides
    if args.context_lengths:
        lengths = [int(x.strip()) for x in args.context_lengths.split(",")]
    if args.n_trials:
        n_trials = args.n_trials

    conditions = [
        {
            "name": "single_topic",
            "context_lengths": lengths,
            "n_trials": n_trials,
            "generate_fn": lambda length, trial_idx: generate_single_topic_tokens(
                model, nl_loader, length, trial_idx
            ),
        },
        {
            "name": "multi_topic_300",
            "context_lengths": lengths,
            "n_trials": n_trials,
            "generate_fn": lambda length, trial_idx: generate_multi_topic_tokens(
                model, nl_loader, length, trial_idx, excerpt_tokens=300
            ),
        },
        {
            "name": "multi_topic_1000",
            "context_lengths": lengths,
            "n_trials": n_trials,
            "generate_fn": lambda length, trial_idx: generate_multi_topic_tokens(
                model, nl_loader, length, trial_idx, excerpt_tokens=1000
            ),
        },
    ]

    output_dir = Path(args.output_dir) / "topic_changes"
    return run_single_experiment(
        "topic_changes", conditions, model, encoded_questions, layers, output_dir,
        chunk_size=args.chunk_size,
    )


def run_extended_length_experiment(model, encoded_questions, layers, nl_loader, args):
    """Experiment 3: Natural language at extended context lengths."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: EXTENDED LENGTH")
    print("=" * 70)

    lengths = [20000, 30000, 50000]
    n_trials = 3

    if args.quick_test:
        lengths = [2000]
        n_trials = 1

    # CLI override for context lengths
    if args.context_lengths:
        lengths = [int(x.strip()) for x in args.context_lengths.split(",")]
    if args.n_trials:
        n_trials = args.n_trials

    conditions = [
        {
            "name": "natural_books_extended",
            "context_lengths": lengths,
            "n_trials": n_trials,
            "generate_fn": lambda length, trial_idx: generate_single_topic_tokens(
                model, nl_loader, length, trial_idx
            ),
        },
    ]

    output_dir = Path(args.output_dir) / "extended_length"
    return run_single_experiment(
        "extended_length", conditions, model, encoded_questions, layers, output_dir,
        chunk_size=args.chunk_size,
    )


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Context variation experiments: misspellings, topic changes, extended length"
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"]
    )
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument(
        "--output-dir", default="results/context_variation",
    )
    parser.add_argument(
        "--experiment",
        default="all",
        choices=["all", "misspellings", "topic_changes", "extended_length"],
    )
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--context-lengths", type=str, default=None,
                        help="Comma-separated context lengths (for extended_length experiment)")
    parser.add_argument("--n-trials", type=int, default=None,
                        help="Override number of trials")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print("=" * 70)
    print("CONTEXT VARIATION EXPERIMENTS")
    print("=" * 70)

    # Load model
    print(f"\nLoading model: {args.model}")
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    model = HookedLLM.from_pretrained(
        args.model, device="auto", dtype=dtype_map[args.dtype]
    )

    n_layers = model.num_layers
    layers = sorted(set([0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]))
    print(f"  Layers: {layers}")

    # Detect chat template
    use_chat_template = (
        hasattr(model.tokenizer, "chat_template")
        and model.tokenizer.chat_template is not None
    )
    print(f"  Chat template: {use_chat_template}")

    # Screen questions
    questions = QUESTIONS.copy()
    if args.quick_test:
        questions = {cat: qs[:2] for cat, qs in questions.items()}

    screened = screen_questions(model, questions, use_chat_template)
    clear_gpu_memory()

    # Encode questions
    encoded_questions = []
    for cat, qs in screened.items():
        for q_data in qs:
            prompt = format_question(q_data["q"], use_chat_template, model.tokenizer)
            prompt_ids = model.tokenizer.encode(prompt, add_special_tokens=False)
            answer_ids = model.tokenizer.encode(q_data["a"], add_special_tokens=False)
            encoded_questions.append(
                {
                    **q_data,
                    "prompt_ids": prompt_ids,
                    "answer_ids": answer_ids,
                    "category": cat,
                }
            )

    print(f"\n{len(encoded_questions)} questions passed screening")

    # Setup natural language loader
    nl_loader = NaturalLanguageLoader(model.tokenizer, NaturalLanguageConfig(seed=42))

    # Run experiments
    experiments_to_run = (
        ["misspellings", "topic_changes", "extended_length"]
        if args.experiment == "all"
        else [args.experiment]
    )

    for exp_name in experiments_to_run:
        if exp_name == "misspellings":
            run_misspellings_experiment(model, encoded_questions, layers, nl_loader, args)
        elif exp_name == "topic_changes":
            run_topic_change_experiment(model, encoded_questions, layers, nl_loader, args)
        elif exp_name == "extended_length":
            run_extended_length_experiment(model, encoded_questions, layers, nl_loader, args)

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
