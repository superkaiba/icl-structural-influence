#!/usr/bin/env python3
"""
Probing Collapse Performance: Does Representational Collapse Impair Knowledge Retrieval?

Tests whether geometric collapse in representations has functional consequences
by measuring knowledge retrieval accuracy after different types of context.

Context Types:
- no_context: Baseline (question only)
- structured_walk: Collapse-inducing SBM graph walks
- natural_books: Non-collapsing natural language
- repeated_token: Extreme collapse (single token repeated)

Metrics:
- answer_log_prob: Mean log-probability of correct answer tokens
- answer_correct: Greedy decode matches expected answer
- context_collapse: Collapse metrics (cos_sim, eff_dim, spread) at end of context
- correlation_collapse_vs_logprob: Pearson r between collapse severity and log-prob

Usage:
    # Quick test (2 questions, 2 context lengths)
    PYTHONPATH=. python experiments/core/run_probing_collapse_performance.py --quick-test

    # Full experiment (~30-40 min)
    PYTHONPATH=. python experiments/core/run_probing_collapse_performance.py
"""

import argparse
import json
import gc
import random
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional
import warnings

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
from src.metrics.collapse_metrics import (
    compute_collapse_metrics,
    CollapseMetrics,
)


# ── Question Bank ──────────────────────────────────────────────────────────
# Mix of easy, medium, and hard questions. Pre-screening filters to those
# the model answers correctly at baseline. Harder questions are more likely
# to degrade under collapsed representations, giving us signal.

QUESTIONS = {
    "A_factual": [
        # Easy (anchor)
        {"q": "What is the capital of France?", "a": "Paris"},
        {"q": "What is the chemical symbol for gold?", "a": "Au"},
        {"q": "How many planets are in the solar system?", "a": "8"},
        # Medium
        {"q": "What is the capital of Myanmar?", "a": "Naypyidaw"},
        {"q": "What element has atomic number 79?", "a": "Gold"},
        {"q": "What is the currency of Poland?", "a": "Zloty"},
        {"q": "In what year did the Berlin Wall fall?", "a": "1989"},
        {"q": "What is the longest river in Africa?", "a": "Nile"},
        {"q": "What is the chemical symbol for tungsten?", "a": "W"},
        {"q": "How many bones are in the adult human body?", "a": "206"},
        {"q": "What is the capital of New Zealand?", "a": "Wellington"},
        {"q": "What is the most abundant element in the Earth's crust?", "a": "Oxygen"},
        # Hard
        {"q": "What is the half-life of Carbon-14 in years?", "a": "5730"},
        {"q": "What is the capital of Bhutan?", "a": "Thimphu"},
        {"q": "What is the atomic number of Osmium?", "a": "76"},
        {"q": "In what year was the Treaty of Westphalia signed?", "a": "1648"},
        {"q": "What is the deepest point in the ocean in meters?", "a": "10994"},
        {"q": "What is the speed of sound in air at 20C in m/s?", "a": "343"},
        {"q": "What is the capital of Suriname?", "a": "Paramaribo"},
        {"q": "How many chromosomes do humans have?", "a": "46"},
    ],
    "B_reasoning": [
        # Easy (anchor)
        {"q": "What is 7 times 8?", "a": "56"},
        {"q": "What is 15 plus 27?", "a": "42"},
        {"q": "What comes next: 2, 4, 8, 16, ?", "a": "32"},
        # Medium
        {"q": "What is 17 times 23?", "a": "391"},
        {"q": "What is 2 to the power of 15?", "a": "32768"},
        {"q": "What is the remainder when 97 is divided by 7?", "a": "6"},
        {"q": "If you buy 3 items at $4.75 each, what is the total?", "a": "14.25"},
        {"q": "What is the next prime number after 31?", "a": "37"},
        {"q": "What is 15% of 340?", "a": "51"},
        {"q": "How many minutes are in a week?", "a": "10080"},
        {"q": "What is the square root of 196?", "a": "14"},
        {"q": "What comes next: 1, 4, 9, 16, 25, ?", "a": "36"},
        # Hard
        {"q": "What is 37 times 43?", "a": "1591"},
        {"q": "What is 7 to the power of 4?", "a": "2401"},
        {"q": "What is the sum of the first 20 positive integers?", "a": "210"},
        {"q": "If a car depreciates 15% per year from $20000, what is it worth after 2 years?", "a": "14450"},
        {"q": "What is the least common multiple of 12 and 18?", "a": "36"},
        {"q": "What comes next: 2, 6, 12, 20, 30, ?", "a": "42"},
        {"q": "What is 1234 divided by 17?", "a": "72"},
        {"q": "How many diagonals does a hexagon have?", "a": "9"},
    ],
    "C_word_knowledge": [
        # Easy (anchor)
        {"q": "What is the opposite of hot?", "a": "cold"},
        {"q": "What is the plural of 'child'?", "a": "children"},
        {"q": "What is the past tense of 'go'?", "a": "went"},
        # Medium
        {"q": "What is a synonym for 'ubiquitous'?", "a": "everywhere"},
        {"q": "What is the adjective form of 'chaos'?", "a": "chaotic"},
        {"q": "What word means 'fear of heights'?", "a": "acrophobia"},
        {"q": "What is the opposite of 'verbose'?", "a": "concise"},
        {"q": "What is the collective noun for a group of crows?", "a": "murder"},
        {"q": "What is the opposite of 'benevolent'?", "a": "malevolent"},
        {"q": "What word means 'to make something less severe'?", "a": "mitigate"},
        {"q": "What is the noun form of 'deceive'?", "a": "deception"},
        {"q": "What is a word meaning 'lasting a very short time'?", "a": "ephemeral"},
        # Hard
        {"q": "What word means 'excessively talkative'?", "a": "loquacious"},
        {"q": "What is a synonym for 'sycophant'?", "a": "flatterer"},
        {"q": "What word means 'to officially forbid something'?", "a": "proscribe"},
        {"q": "What is the opposite of 'enervate'?", "a": "invigorate"},
        {"q": "What word means 'a long passionate speech'?", "a": "tirade"},
        {"q": "What is a synonym for 'perspicacious'?", "a": "shrewd"},
        {"q": "What word means 'to deny or contradict'?", "a": "gainsay"},
        {"q": "What is the adjective form of 'parsimony'?", "a": "parsimonious"},
    ],
    "D_multi_token": [
        # Easy (anchor)
        {"q": "Who wrote Romeo and Juliet?", "a": "William Shakespeare"},
        {"q": "Who painted the Mona Lisa?", "a": "Leonardo da Vinci"},
        {"q": "Who was the first person to walk on the moon?", "a": "Neil Armstrong"},
        # Medium
        {"q": "Who wrote The Brothers Karamazov?", "a": "Fyodor Dostoevsky"},
        {"q": "Who composed The Four Seasons?", "a": "Antonio Vivaldi"},
        {"q": "Who proposed the heliocentric model of the solar system?", "a": "Nicolaus Copernicus"},
        {"q": "Who wrote One Hundred Years of Solitude?", "a": "Gabriel Garcia Marquez"},
        {"q": "Who discovered the structure of DNA?", "a": "Watson and Crick"},
        {"q": "Who painted The Persistence of Memory?", "a": "Salvador Dali"},
        {"q": "Who directed the film 2001: A Space Odyssey?", "a": "Stanley Kubrick"},
        {"q": "Who wrote The Wealth of Nations?", "a": "Adam Smith"},
        {"q": "Who was the first woman to win a Nobel Prize?", "a": "Marie Curie"},
        # Hard
        {"q": "Who proved Fermat's Last Theorem?", "a": "Andrew Wiles"},
        {"q": "Who wrote The Master and Margarita?", "a": "Mikhail Bulgakov"},
        {"q": "Who composed The Rite of Spring?", "a": "Igor Stravinsky"},
        {"q": "Who discovered the electron?", "a": "J.J. Thomson"},
        {"q": "Who painted Guernica?", "a": "Pablo Picasso"},
        {"q": "Who wrote the Principia Mathematica?", "a": "Isaac Newton"},
        {"q": "Who directed Stalker and Solaris?", "a": "Andrei Tarkovsky"},
        {"q": "Who formulated the incompleteness theorems?", "a": "Kurt Godel"},
    ],
}

CONTEXT_LENGTHS = [0, 500, 1000, 2000, 5000, 10000, 20000]
CONTEXT_TYPES = ["no_context", "structured_walk", "natural_books", "repeated_token"]

# Ignore-instruction variants: same context, but inject instruction to ignore it
IGNORE_INSTRUCTION = (
    "Ignore all of the preceding text. It is irrelevant filler. "
    "Answer the following question using only your own knowledge."
)

CHAT_IGNORE_INSTRUCTION = (
    "Ignore everything in the previous message. It was irrelevant filler text. "
    "Answer the following question using only your own knowledge."
)

def get_base_context_type(ctx_type: str) -> str:
    """Strip '_ignore' and '_chat' suffixes to get base context type."""
    return ctx_type.replace("_chat_ignore", "").replace("_chat", "").replace("_ignore", "")

def is_ignore_variant(ctx_type: str) -> bool:
    return "_ignore" in ctx_type

def is_chat_variant(ctx_type: str) -> bool:
    return "_chat" in ctx_type


# ── Utilities ──────────────────────────────────────────────────────────────

def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def convert_numpy(obj):
    """Convert numpy/torch types to JSON-serializable formats."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, CollapseMetrics):
        return convert_numpy(obj.to_dict())
    elif isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    return obj


def deep_copy_kv_cache(past_key_values):
    """Deep copy KV cache to allow reuse across questions."""
    if past_key_values is None:
        return None

    # DynamicCache (transformers >= 4.36)
    if hasattr(past_key_values, 'key_cache'):
        try:
            from transformers.cache_utils import DynamicCache
            new_cache = DynamicCache()
            for layer_idx in range(len(past_key_values.key_cache)):
                new_cache.update(
                    past_key_values.key_cache[layer_idx].clone(),
                    past_key_values.value_cache[layer_idx].clone(),
                    layer_idx,
                )
            return new_cache
        except ImportError:
            pass

    # Tuple-of-tuples format
    if isinstance(past_key_values, tuple):
        return tuple(
            tuple(t.clone() for t in layer_kv)
            for layer_kv in past_key_values
        )

    import copy
    return copy.deepcopy(past_key_values)


# ── Chat Template Helpers ─────────────────────────────────────────────────

def wrap_context_in_chat(
    raw_token_ids: list[int],
    tokenizer,
    ignore: bool = False,
) -> list[int]:
    """Wrap raw context tokens in a multi-turn chat template.

    Returns tokenized version of:
      <system>You are a helpful assistant.</system>
      <user>[raw text]</user>
      <assistant>OK.</assistant>
      [if ignore: <user>Ignore everything...</user> <assistant>Understood...</assistant>]
    """
    raw_text = tokenizer.decode(raw_token_ids, skip_special_tokens=True)

    messages = [
        {"role": "user", "content": raw_text},
        {"role": "assistant", "content": "OK."},
    ]
    if ignore:
        messages.extend([
            {"role": "user", "content": CHAT_IGNORE_INSTRUCTION},
            {"role": "assistant", "content": "Understood, I will ignore that text and answer based only on my knowledge."},
        ])

    ctx_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )
    return tokenizer.encode(ctx_str, add_special_tokens=False)


def format_question_continuation(question: str, tokenizer) -> str:
    """Format a question as a continuation turn in an existing chat.

    Uses apply_chat_template on full conversation then strips the context
    prefix to get just the question turn tokens.
    """
    # Build minimal context + question conversation
    ctx_messages = [
        {"role": "user", "content": "X"},
        {"role": "assistant", "content": "OK."},
    ]
    full_messages = ctx_messages + [
        {"role": "user", "content": f"Answer in as few words as possible.\n\nQ: {question}\nA:"},
    ]

    ctx_str = tokenizer.apply_chat_template(
        ctx_messages, tokenize=False, add_generation_prompt=False,
    )
    full_str = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=True,
    )

    # The continuation is everything after the context portion
    # This gives us: <|im_start|>user\nAnswer...<|im_end|>\n<|im_start|>assistant\n
    return full_str[len(ctx_str):]


# ── Question Formatting & Evaluation ──────────────────────────────────────

def format_question(question: str, use_chat_template: bool, tokenizer) -> str:
    """Format a question for the model (standalone, no prior context)."""
    prompt = f"Q: {question}\nA:"
    if use_chat_template:
        messages = [{"role": "user", "content": f"Answer in as few words as possible.\n\n{prompt}"}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return prompt


def check_answer_correct(generated: str, expected: str) -> bool:
    """Check if generated answer matches expected (case-insensitive, flexible)."""
    gen_lower = generated.lower().strip()
    exp_lower = expected.lower().strip()

    if exp_lower in gen_lower:
        return True

    gen_words = gen_lower.split()
    if gen_words and gen_words[0] == exp_lower:
        return True

    # Multi-token: check all key words present
    exp_words = set(exp_lower.split())
    if len(exp_words) > 1 and exp_words.issubset(set(gen_lower.split())):
        return True

    return False


def evaluate_question(
    model: HookedLLM,
    question_ids: list[int],
    answer_ids: list[int],
    context_past_kvs,
    max_new_tokens: int = 30,
) -> tuple[float, str]:
    """
    Evaluate a question: compute answer log-prob and greedy decode.

    Clones the context KV cache once, forwards question tokens once,
    then branches for log-prob and greedy decode.

    Returns (mean_log_prob, generated_text).
    """
    kv = deep_copy_kv_cache(context_past_kvs)

    # Forward question tokens
    q_input = torch.tensor([question_ids]).to(model.device)
    with torch.no_grad():
        q_out = model.model(
            input_ids=q_input,
            past_key_values=kv,
            use_cache=True,
        )

    last_logit = q_out.logits[0, -1, :]

    # ── Branch 1: Log-probability of expected answer ──
    log_probs = []
    lp_dist = torch.log_softmax(last_logit, dim=-1)
    log_probs.append(lp_dist[answer_ids[0]].item())

    # Clone post-question state for greedy branch BEFORE modifying
    greedy_kvs = deep_copy_kv_cache(q_out.past_key_values)

    # Feed remaining answer tokens
    lp_kvs = q_out.past_key_values
    for i in range(len(answer_ids) - 1):
        with torch.no_grad():
            a_out = model.model(
                input_ids=torch.tensor([[answer_ids[i]]]).to(model.device),
                past_key_values=lp_kvs,
                use_cache=True,
            )
        lp_kvs = a_out.past_key_values
        lp_dist = torch.log_softmax(a_out.logits[0, -1, :], dim=-1)
        log_probs.append(lp_dist[answer_ids[i + 1]].item())

    mean_log_prob = float(np.mean(log_probs))

    # ── Branch 2: Greedy decode ──
    next_token = last_logit.argmax().item()
    generated_ids = [next_token]

    for _ in range(max_new_tokens - 1):
        with torch.no_grad():
            g_out = model.model(
                input_ids=torch.tensor([[next_token]]).to(model.device),
                past_key_values=greedy_kvs,
                use_cache=True,
            )
        greedy_kvs = g_out.past_key_values
        next_token = g_out.logits[0, -1, :].argmax().item()
        generated_ids.append(next_token)

        if next_token == model.tokenizer.eos_token_id:
            break
        decoded = model.tokenizer.decode(generated_ids, skip_special_tokens=True)
        if '\n' in decoded:
            break

    generated_text = model.tokenizer.decode(
        generated_ids, skip_special_tokens=True
    ).strip()

    # Cleanup
    del kv, greedy_kvs, lp_kvs
    return mean_log_prob, generated_text


def screen_questions(
    model: HookedLLM,
    questions: dict,
    use_chat_template: bool,
) -> dict:
    """Pre-screen questions: keep only those the model answers correctly at baseline."""
    print("\nScreening questions (baseline correctness)...")
    screened = {}
    total = 0
    passed = 0

    for category, q_list in questions.items():
        screened[category] = []
        for q_data in q_list:
            total += 1
            prompt = format_question(q_data["q"], use_chat_template, model.tokenizer)
            prompt_ids = model.tokenizer.encode(prompt, add_special_tokens=False)
            answer_ids = model.tokenizer.encode(q_data["a"], add_special_tokens=False)

            if len(answer_ids) == 0:
                continue

            log_prob, generated = evaluate_question(
                model, prompt_ids, answer_ids,
                context_past_kvs=None, max_new_tokens=30,
            )
            correct = check_answer_correct(generated, q_data["a"])

            if correct:
                screened[category].append(q_data)
                passed += 1
            else:
                print(f"  SKIP: '{q_data['q']}' -> '{generated}' (expected: '{q_data['a']}')")

        print(f"  {category}: {len(screened[category])}/{len(q_list)} passed")

    print(f"  Total: {passed}/{total} questions passed screening")
    return screened


# ── Context Generation ─────────────────────────────────────────────────────

def generate_context_tokens(
    context_type: str,
    context_length: int,
    trial_idx: int,
    model: HookedLLM,
    graph: DualInterpretationGraph,
    nl_loader: NaturalLanguageLoader,
) -> list[int]:
    """Generate context tokens for a given type and length."""
    if context_length == 0 or context_type == "no_context":
        return []

    # Handle _ignore variants by delegating to base type
    base_type = get_base_context_type(context_type)

    if base_type == "structured_walk":
        walk_length = context_length * 2  # oversample for tokenization
        prompt, _, _ = graph.generate_h1_only_walk(
            length=walk_length, return_nodes=True
        )
        tokens = model.tokenizer.encode(prompt, add_special_tokens=False)
        return tokens[:context_length]

    elif base_type == "natural_books":
        # Different random seed per trial for variety
        nl_loader.rng = random.Random(42 + trial_idx)
        tokens = nl_loader.load_book(context_length)
        return tokens[:context_length]

    elif base_type == "repeated_token":
        # Repeat a single common token
        token_id = model.tokenizer.encode(" the", add_special_tokens=False)[0]
        return [token_id] * context_length

    else:
        raise ValueError(f"Unknown context type: {context_type}")


# ── Context Processing ─────────────────────────────────────────────────────

def process_context_chunks(
    model: HookedLLM,
    token_ids: list[int],
    layers: list[int],
    chunk_size: int = 512,
    window_size: int = 50,
) -> tuple:
    """
    Process context tokens in chunks, returning KV cache and collapse metrics.

    Returns (past_key_values, collapse_metrics_by_layer).
    """
    if len(token_ids) == 0:
        return None, {}

    past_kvs = None
    recent_reps = {layer: deque(maxlen=window_size) for layer in layers}

    for start in range(0, len(token_ids), chunk_size):
        end = min(start + chunk_size, len(token_ids))
        chunk = token_ids[start:end]
        input_ids = torch.tensor([chunk]).to(model.device)

        _, cache, past_kvs = model.forward_incremental(
            input_ids, layers=layers, past_key_values=past_kvs,
        )

        # Store representations from last tokens of the chunk
        for layer in layers:
            rep = cache.get_residual_stream(layer)
            if rep is not None:
                n_to_take = min(rep.shape[1], window_size)
                for pos in range(-n_to_take, 0):
                    rep_np = rep[0, pos].cpu().float().numpy()
                    recent_reps[layer].append(rep_np)

    # Compute collapse metrics on last window of representations
    collapse_by_layer = {}
    for layer in layers:
        reps_list = list(recent_reps[layer])
        if len(reps_list) >= 10:
            metrics = compute_collapse_metrics(reps_list)
            collapse_by_layer[layer] = metrics.to_dict()
        else:
            collapse_by_layer[layer] = None

    return past_kvs, collapse_by_layer


# ── Main Experiment ────────────────────────────────────────────────────────

def run_experiment(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw").mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 70)
    print("PROBING COLLAPSE PERFORMANCE EXPERIMENT")
    print("Does Representational Collapse Impair Knowledge Retrieval?")
    print("=" * 70)
    print(f"\nTimestamp: {timestamp}")
    print(f"Output directory: {output_dir}")

    # Load model
    print(f"\nLoading model: {args.model}")
    dtype_map = {
        "float32": torch.float32, "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    model = HookedLLM.from_pretrained(
        args.model, device="auto", dtype=dtype_map[args.dtype]
    )
    print(f"  Layers: {model.num_layers}, Hidden size: {model.hidden_size}")

    # Determine layers
    if args.layers:
        layers = [int(x) for x in args.layers.split(',')]
    else:
        n = model.num_layers
        layers = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))
    print(f"  Layers to analyze: {layers}")

    # Detect chat template
    use_chat_template = (
        hasattr(model.tokenizer, 'chat_template')
        and model.tokenizer.chat_template is not None
    )
    print(f"  Chat template: {use_chat_template}")

    # Setup conditions
    if args.quick_test:
        context_lengths = [0, 1000]
        n_trials = 1
        context_types = ["no_context", "structured_walk"]
    else:
        context_lengths = CONTEXT_LENGTHS
        n_trials = args.n_trials
        context_types = CONTEXT_TYPES

    # CLI overrides
    if args.context_types:
        context_types = [s.strip() for s in args.context_types.split(',')]
    if args.context_lengths:
        context_lengths = [int(s.strip()) for s in args.context_lengths.split(',')]

    print(f"  Context lengths: {context_lengths}")
    print(f"  Context types: {context_types}")
    print(f"  Trials per condition: {n_trials}")

    # Screen questions
    questions = QUESTIONS.copy()
    if args.quick_test:
        questions = {cat: qs[:2] for cat, qs in questions.items()}

    screened_questions = screen_questions(model, questions, use_chat_template)
    clear_gpu_memory()

    # Flatten questions
    all_questions = []
    for cat, qs in screened_questions.items():
        for q_data in qs:
            all_questions.append({**q_data, "category": cat})

    if len(all_questions) == 0:
        print("ERROR: No questions passed screening!")
        return

    print(f"\n{len(all_questions)} questions passed screening")

    # Pre-encode all questions and answers (both standalone and continuation formats)
    encoded_questions = []
    for q_data in all_questions:
        # Standalone format (for raw context / no_context / _ignore variants)
        prompt = format_question(q_data["q"], use_chat_template, model.tokenizer)
        prompt_ids = model.tokenizer.encode(prompt, add_special_tokens=False)
        # Continuation format (for _chat variants where context is in chat template)
        cont_str = format_question_continuation(q_data["q"], model.tokenizer)
        cont_ids = model.tokenizer.encode(cont_str, add_special_tokens=False)
        answer_ids = model.tokenizer.encode(q_data["a"], add_special_tokens=False)
        encoded_questions.append({
            **q_data,
            "prompt_ids": prompt_ids,
            "prompt_ids_continuation": cont_ids,
            "answer_ids": answer_ids,
        })

    # Create data sources
    graph_config = DualInterpretationConfig(
        vocab_size=15, clusters_per_interpretation=3, seed=42
    )
    graph = DualInterpretationGraph(graph_config)
    nl_loader = NaturalLanguageLoader(
        model.tokenizer, NaturalLanguageConfig(seed=42)
    )

    # Save config
    config = {
        "model": args.model,
        "layers": layers,
        "context_lengths": context_lengths,
        "context_types": context_types,
        "n_trials": n_trials,
        "n_questions": len(all_questions),
        "question_categories": {
            cat: len(qs) for cat, qs in screened_questions.items()
        },
        "use_chat_template": use_chat_template,
        "chunk_size": args.chunk_size,
        "timestamp": timestamp,
        "quick_test": args.quick_test,
    }
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # ── Main Loop ──
    all_results = []

    for ctx_type in context_types:
        print(f"\n{'='*60}")
        print(f"Context type: {ctx_type}")
        print(f"{'='*60}")

        base = get_base_context_type(ctx_type)
        for ctx_len in context_lengths:
            # Skip invalid combinations
            if base == "no_context" and ctx_len > 0:
                continue
            if base != "no_context" and ctx_len == 0:
                continue

            # Determine trial count
            effective_trials = (
                1 if base in ("no_context", "repeated_token") else n_trials
            )

            for trial_idx in range(effective_trials):
                trial_label = f"{ctx_type}_len{ctx_len}_trial{trial_idx}"
                print(f"\n  {trial_label}")

                # Generate raw context tokens (same content for all variants)
                context_tokens = generate_context_tokens(
                    ctx_type, ctx_len, trial_idx, model, graph, nl_loader
                )
                raw_ctx_len = len(context_tokens)

                # For _chat variants, wrap in chat template
                if is_chat_variant(ctx_type) and context_tokens:
                    context_tokens = wrap_context_in_chat(
                        context_tokens, model.tokenizer,
                        ignore=is_ignore_variant(ctx_type),
                    )
                    print(f"    Raw content: {raw_ctx_len} tokens, "
                          f"chat-wrapped: {len(context_tokens)} tokens"
                          f"{' (+ ignore turn)' if is_ignore_variant(ctx_type) else ''}")

                actual_ctx_len = len(context_tokens)

                # Process context -> KV cache + collapse metrics
                past_kvs, collapse_metrics = process_context_chunks(
                    model, context_tokens, layers,
                    chunk_size=args.chunk_size,
                    window_size=50,
                )

                if not is_chat_variant(ctx_type):
                    print(f"    Context: {actual_ctx_len} tokens processed")

                # For non-chat _ignore variants, inject raw ignore instruction
                if is_ignore_variant(ctx_type) and not is_chat_variant(ctx_type) and past_kvs is not None:
                    ignore_ids = model.tokenizer.encode(
                        "\n\n" + IGNORE_INSTRUCTION + "\n\n",
                        add_special_tokens=False,
                    )
                    ignore_input = torch.tensor([ignore_ids]).to(model.device)
                    with torch.no_grad():
                        ignore_out = model.model(
                            input_ids=ignore_input,
                            past_key_values=past_kvs,
                            use_cache=True,
                        )
                    past_kvs = ignore_out.past_key_values
                    print(f"    Injected ignore instruction ({len(ignore_ids)} tokens)")
                if collapse_metrics:
                    last_layer = layers[-1]
                    cm = collapse_metrics.get(last_layer)
                    if cm:
                        print(
                            f"    Collapse L{last_layer}: "
                            f"cos_sim={cm['avg_cos_sim']:.3f}, "
                            f"eff_dim={cm['effective_dim']:.1f}"
                        )

                # Evaluate each question (use continuation format for chat variants)
                use_cont = is_chat_variant(ctx_type)
                trial_results = []
                for eq in encoded_questions:
                    q_ids = eq["prompt_ids_continuation"] if use_cont else eq["prompt_ids"]
                    log_prob, generated = evaluate_question(
                        model,
                        q_ids,
                        eq["answer_ids"],
                        context_past_kvs=past_kvs,
                        max_new_tokens=30,
                    )
                    correct = check_answer_correct(generated, eq["a"])

                    result = {
                        "context_type": ctx_type,
                        "context_length": ctx_len,
                        "actual_context_length": actual_ctx_len,
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
                raw_path = output_dir / "raw" / f"{trial_label}.json"
                with open(raw_path, 'w') as f:
                    json.dump(convert_numpy(trial_results), f, indent=2)

                # Cleanup
                del past_kvs, collapse_metrics
                clear_gpu_memory()

    # ── Aggregate Results ──
    print("\n" + "=" * 60)
    print("AGGREGATING RESULTS")
    print("=" * 60)

    aggregated = {}
    for ctx_type in context_types:
        aggregated[ctx_type] = {}
        for ctx_len in context_lengths:
            trials = [
                r for r in all_results
                if r["context_type"] == ctx_type and r["context_length"] == ctx_len
            ]
            if not trials:
                continue

            accuracies = [r["answer_correct"] for r in trials]
            log_probs = [r["answer_log_prob"] for r in trials]

            # Per-category breakdown
            category_stats = {}
            for cat in screened_questions:
                cat_trials = [r for r in trials if r["category"] == cat]
                if cat_trials:
                    category_stats[cat] = {
                        "accuracy": float(np.mean([r["answer_correct"] for r in cat_trials])),
                        "mean_log_prob": float(np.mean([r["answer_log_prob"] for r in cat_trials])),
                        "std_log_prob": float(np.std([r["answer_log_prob"] for r in cat_trials])),
                        "n": len(cat_trials),
                    }

            # Collapse stats (from last layer)
            collapse_cos_sims = []
            collapse_eff_dims = []
            collapse_spreads = []
            for r in trials:
                cm = r.get("collapse_metrics", {})
                if cm:
                    last_layer_key = str(layers[-1])
                    layer_cm = cm.get(last_layer_key)
                    if layer_cm:
                        collapse_cos_sims.append(layer_cm["avg_cos_sim"])
                        collapse_eff_dims.append(layer_cm["effective_dim"])
                        collapse_spreads.append(layer_cm["spread"])

            aggregated[ctx_type][ctx_len] = {
                "accuracy": float(np.mean(accuracies)),
                "mean_log_prob": float(np.mean(log_probs)),
                "std_log_prob": float(np.std(log_probs)),
                "n_evaluations": len(trials),
                "category_stats": category_stats,
                "collapse_cos_sim_mean": (
                    float(np.mean(collapse_cos_sims)) if collapse_cos_sims else None
                ),
                "collapse_eff_dim_mean": (
                    float(np.mean(collapse_eff_dims)) if collapse_eff_dims else None
                ),
                "collapse_spread_mean": (
                    float(np.mean(collapse_spreads)) if collapse_spreads else None
                ),
            }

    # Compute correlation between collapse and performance
    all_cos_sims = []
    all_log_probs_for_corr = []
    for r in all_results:
        cm = r.get("collapse_metrics", {})
        if cm:
            last_layer_key = str(layers[-1])
            layer_cm = cm.get(last_layer_key)
            if layer_cm:
                all_cos_sims.append(layer_cm["avg_cos_sim"])
                all_log_probs_for_corr.append(r["answer_log_prob"])

    correlation = None
    if len(all_cos_sims) > 5:
        corr_val = np.corrcoef(all_cos_sims, all_log_probs_for_corr)[0, 1]
        if not np.isnan(corr_val):
            correlation = float(corr_val)
            print(f"\nCorrelation (collapse cos_sim vs log_prob): r = {correlation:.4f}")
        else:
            print("\nCorrelation: insufficient variance (NaN)")
    else:
        print(f"\nCorrelation: insufficient data ({len(all_cos_sims)} points)")

    # Save final results
    final_results = {
        "config": config,
        "aggregated": convert_numpy(aggregated),
        "correlation_collapse_vs_logprob": correlation,
        "n_total_evaluations": len(all_results),
        "screened_questions": {
            cat: [q["q"] for q in qs]
            for cat, qs in screened_questions.items()
        },
    }

    results_path = output_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)

    # Save all raw results
    all_raw_path = output_dir / "all_results.json"
    with open(all_raw_path, 'w') as f:
        json.dump(convert_numpy(all_results), f, indent=2)

    # Print summary table
    print("\n" + "-" * 70)
    print("SUMMARY: Accuracy by Context Type x Length")
    print("-" * 70)
    header = f"{'Type':<20} {'Length':<10} {'Accuracy':<12} {'Log-Prob':<12} {'Collapse':<12}"
    print(header)
    print("-" * 70)
    for ctx_type in context_types:
        for ctx_len in context_lengths:
            agg = aggregated.get(ctx_type, {}).get(ctx_len)
            if agg:
                collapse_str = (
                    f"{agg['collapse_cos_sim_mean']:.3f}"
                    if agg['collapse_cos_sim_mean'] is not None
                    else "N/A"
                )
                print(
                    f"{ctx_type:<20} {ctx_len:<10} "
                    f"{agg['accuracy']:<12.3f} "
                    f"{agg['mean_log_prob']:<12.3f} "
                    f"{collapse_str:<12}"
                )

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Results: {results_path}")
    print(f"Raw data: {all_raw_path}")

    del model
    clear_gpu_memory()


def main():
    parser = argparse.ArgumentParser(
        description="Probing Collapse Performance: Does collapse impair knowledge retrieval?"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument("--layers", type=str, default=None)
    parser.add_argument("--n-trials", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument(
        "--output-dir", type=str,
        default="results/probing_collapse_performance",
    )
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--context-types", type=str, default=None,
                        help="Comma-separated context types (overrides defaults)")
    parser.add_argument("--context-lengths", type=str, default=None,
                        help="Comma-separated context lengths (overrides defaults)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    run_experiment(args)


if __name__ == "__main__":
    main()
