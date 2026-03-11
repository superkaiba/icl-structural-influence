#!/usr/bin/env python3
"""
Chain-of-Thought Disambiguation Experiment.

Tests whether velocity spikes and preference shifts occur during self-generated
CoT reasoning when the model considers multiple possibilities and commits to an answer.

Conditions:
  - cot_ambig: Ambiguous passage + "Think step by step" (model reasons without disambig word)
  - cot_disambig: Full passage (with disambig word) + "Think step by step"
  - direct_answer: Ambiguous passage + "Answer in one word"
  - external_disambig: Full passage with disambig word, no generation
  - external_cot: Feed saved CoT from cot_ambig as input (no generation)

Usage:
    python run_cot_disambig_experiment.py --model Qwen/Qwen2.5-7B-Instruct

    # After first run (to generate CoTs), re-run with external_cot:
    python run_cot_disambig_experiment.py --model Qwen/Qwen2.5-7B-Instruct \
        --conditions external_cot --saved-cots results/cot_disambig_pilot/saved_cots.json
"""

import argparse
import gc
import json
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.hooked_model import HookedLLM
from src.data.natural_language_disambig import (
    get_pilot_pairs,
    get_stimuli,
    get_reference_passages,
    get_pair_by_name,
    CategoryPair,
    DisambiguationStimulus,
)
from src.data.cot_reasoning_problems import (
    get_reasoning_problems,
    ReasoningProblem,
)
from src.metrics.context_evolution_metrics import (
    compute_velocity_series,
    compute_cumulative_drift,
    detect_velocity_spike,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, (np.float32, np.float64, np.floating)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64, np.integer)):
        return int(obj)
    return obj


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_cot_prompt(
    stimulus: DisambiguationStimulus,
    pair: CategoryPair,
    tokenizer,
    condition: str,
) -> str:
    """Build a chat-template prompt for the given condition."""

    if condition == "cot_ambig":
        passage = stimulus.ambiguous_text
        instruction = (
            f"Read the following passage carefully. Think step by step about "
            f"whether this is about {pair.h1_name} or {pair.h2_name}. Consider "
            f"evidence for both possibilities before giving your final answer."
            f"\n\n\"{passage}\"\n\nThink step by step:"
        )
    elif condition == "cot_disambig":
        passage = stimulus.full_text
        instruction = (
            f"Read the following passage carefully. Think step by step about "
            f"whether this is about {pair.h1_name} or {pair.h2_name}. Consider "
            f"evidence for both possibilities before giving your final answer."
            f"\n\n\"{passage}\"\n\nThink step by step:"
        )
    elif condition == "direct_answer":
        passage = stimulus.ambiguous_text
        instruction = (
            f"Read the following passage. Is it about {pair.h1_name} or "
            f"{pair.h2_name}? Answer in one word.\n\n\"{passage}\""
        )
    elif condition == "external_disambig":
        passage = stimulus.full_text
        instruction = (
            f"Is the following passage about {pair.h1_name} or {pair.h2_name}?"
            f"\n\n{passage}"
        )
    else:
        raise ValueError(f"Unknown condition for Category A: {condition}")

    messages = [{"role": "user", "content": instruction}]
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return f"Q: {instruction}\nA:"


def build_reasoning_prompt(
    problem: ReasoningProblem,
    tokenizer,
    condition: str,
) -> str:
    """Build a chat-template prompt for a reasoning problem."""

    if condition in ("cot_ambig", "cot_disambig"):
        instruction = problem.question
    elif condition == "direct_answer":
        instruction = problem.question.split("Think")[0].strip() + " Answer in one word."
    else:
        instruction = problem.question

    messages = [{"role": "user", "content": instruction}]
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return f"Q: {instruction}\nA:"


# ---------------------------------------------------------------------------
# Commitment point detection
# ---------------------------------------------------------------------------

COMMITMENT_MARKERS = [
    "the answer is", "this is about", "i conclude", "in conclusion",
    "therefore", "my answer", "final answer", "based on this",
    "it must be", "this means it", "so the passage is about",
    "the passage is about", "this is definitely", "i believe this is",
    "thus", "hence", "so it's", "so this is",
]


@dataclass
class CommitmentDetection:
    lexical_position: Optional[int] = None
    answer_position: Optional[int] = None
    representation_position: Optional[int] = None
    chosen_answer: str = ""
    commitment_text: str = ""
    method: str = "none"


def detect_commitment_point(
    generated_text: str,
    generated_token_strings: list[str],
    prompt_len: int,
    h1_name: str,
    h2_name: str,
    min_deliberation_tokens: int = 10,
) -> CommitmentDetection:
    """
    Detect the commitment point in generated CoT text.

    Returns the token position (in the full sequence, including prompt) where
    the model commits to an answer.
    """
    result = CommitmentDetection()
    gen_text_lower = generated_text.lower()

    # Method 1: Lexical markers
    earliest_marker_pos = None
    earliest_marker_text = ""
    for marker in COMMITMENT_MARKERS:
        idx = gen_text_lower.find(marker)
        if idx >= 0:
            # Convert character position to approximate token position
            # Count tokens whose cumulative text reaches this position
            char_count = 0
            tok_pos = None
            for i, tok in enumerate(generated_token_strings):
                char_count += len(tok)
                if char_count >= idx:
                    tok_pos = i
                    break
            if tok_pos is not None and tok_pos >= min_deliberation_tokens:
                if earliest_marker_pos is None or tok_pos < earliest_marker_pos:
                    earliest_marker_pos = tok_pos
                    earliest_marker_text = marker

    if earliest_marker_pos is not None:
        result.lexical_position = prompt_len + earliest_marker_pos
        result.commitment_text = earliest_marker_text
        result.method = "lexical"

    # Method 2: Final answer token detection
    h1_lower = h1_name.lower()
    h2_lower = h2_name.lower()
    last_h1_pos = None
    last_h2_pos = None

    for i, tok in enumerate(generated_token_strings):
        tok_lower = tok.lower().strip()
        if h1_lower in tok_lower:
            last_h1_pos = i
        if h2_lower in tok_lower:
            last_h2_pos = i

    # The answer that appears LAST is likely the committed answer
    if last_h1_pos is not None and last_h2_pos is not None:
        if last_h1_pos > last_h2_pos:
            result.chosen_answer = "H1"
            result.answer_position = prompt_len + last_h1_pos
        else:
            result.chosen_answer = "H2"
            result.answer_position = prompt_len + last_h2_pos
    elif last_h1_pos is not None:
        result.chosen_answer = "H1"
        result.answer_position = prompt_len + last_h1_pos
    elif last_h2_pos is not None:
        result.chosen_answer = "H2"
        result.answer_position = prompt_len + last_h2_pos

    # Use the best available position
    if result.lexical_position is None and result.answer_position is not None:
        result.lexical_position = result.answer_position
        result.method = "answer_token"

    return result


def detect_representation_commitment(
    h1_preferences: list[float],
    prompt_len: int,
    window: int = 5,
) -> Optional[int]:
    """
    Find the position where H1 preference permanently crosses 0.5.

    Returns the token position where the preference trajectory commits
    (stays above or below 0.5 for the rest of the sequence).
    """
    if len(h1_preferences) <= prompt_len + window:
        return None

    gen_prefs = h1_preferences[prompt_len:]
    n = len(gen_prefs)

    for i in range(n - window):
        segment = gen_prefs[i:i + window]
        # Check if all above or all below 0.5
        all_above = all(p > 0.5 for p in segment)
        all_below = all(p < 0.5 for p in segment)
        if all_above or all_below:
            # Check remaining sequence
            remaining = gen_prefs[i:]
            if all_above and all(p > 0.45 for p in remaining):  # Allow small noise
                return prompt_len + i
            if all_below and all(p < 0.55 for p in remaining):
                return prompt_len + i

    return None


# ---------------------------------------------------------------------------
# Centroid computation
# ---------------------------------------------------------------------------

def compute_centroids_category_a(
    model: HookedLLM,
    pairs: list[CategoryPair],
    layers: list[int],
) -> dict:
    """Compute H1/H2 centroids from reference passages (Category A)."""
    centroids = {}
    for pair in pairs:
        print(f"  Computing centroids for {pair.name}...")
        ref_passages = get_reference_passages(pair.name)
        centroids[pair.name] = {}
        for layer in layers:
            h1_reps, h2_reps = [], []
            for passage in ref_passages["H1"]:
                text = _format_ref(passage, pair, model.tokenizer)
                with torch.no_grad():
                    _, cache = model.forward_with_cache(text, layers=[layer])
                    h1_reps.append(cache.get_residual_stream(layer)[0, -1].cpu().float().numpy())
            for passage in ref_passages["H2"]:
                text = _format_ref(passage, pair, model.tokenizer)
                with torch.no_grad():
                    _, cache = model.forward_with_cache(text, layers=[layer])
                    h2_reps.append(cache.get_residual_stream(layer)[0, -1].cpu().float().numpy())
            h1_c = np.mean(h1_reps, axis=0)
            h2_c = np.mean(h2_reps, axis=0)
            centroids[pair.name][layer] = {
                "H1": h1_c, "H2": h2_c,
                "cos_sim": cosine_similarity(h1_c, h2_c),
            }
    return centroids


def compute_centroids_category_b(
    model: HookedLLM,
    problems: list[ReasoningProblem],
    layers: list[int],
) -> dict:
    """Compute H1/H2 centroids from answer-asserting statements (Category B)."""
    centroids = {}
    for prob in problems:
        print(f"  Computing centroids for {prob.problem_id}...")
        centroids[prob.problem_id] = {}
        for layer in layers:
            h1_reps, h2_reps = [], []
            for stmt in prob.h1_reference_statements:
                text = _format_ref_statement(stmt, model.tokenizer)
                with torch.no_grad():
                    _, cache = model.forward_with_cache(text, layers=[layer])
                    h1_reps.append(cache.get_residual_stream(layer)[0, -1].cpu().float().numpy())
            for stmt in prob.h2_reference_statements:
                text = _format_ref_statement(stmt, model.tokenizer)
                with torch.no_grad():
                    _, cache = model.forward_with_cache(text, layers=[layer])
                    h2_reps.append(cache.get_residual_stream(layer)[0, -1].cpu().float().numpy())
            h1_c = np.mean(h1_reps, axis=0)
            h2_c = np.mean(h2_reps, axis=0)
            centroids[prob.problem_id] = centroids.get(prob.problem_id, {})
            centroids[prob.problem_id][layer] = {
                "H1": h1_c, "H2": h2_c,
                "cos_sim": cosine_similarity(h1_c, h2_c),
            }
    return centroids


def _format_ref(passage: str, pair: CategoryPair, tokenizer) -> str:
    question = f"Is the following passage about {pair.h1_name} or {pair.h2_name}?"
    messages = [{"role": "user", "content": f"{question}\n\n{passage}"}]
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"Q: {question}\n\n{passage}\nA:"


def _format_ref_statement(stmt: str, tokenizer) -> str:
    messages = [{"role": "assistant", "content": stmt}]
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": "What is the answer?"}] + messages,
            tokenize=False, add_generation_prompt=False,
        )
    return stmt


# ---------------------------------------------------------------------------
# Preference computation
# ---------------------------------------------------------------------------

def compute_preference_trajectory(
    representations: list[np.ndarray],
    h1_centroid: np.ndarray,
    h2_centroid: np.ndarray,
) -> dict:
    h1_prefs, h1_sims, h2_sims = [], [], []
    for rep in representations:
        h1_sim = cosine_similarity(rep, h1_centroid)
        h2_sim = cosine_similarity(rep, h2_centroid)
        total = abs(h1_sim) + abs(h2_sim)
        pref = h1_sim / total if total > 1e-8 else 0.5
        h1_prefs.append(pref)
        h1_sims.append(h1_sim)
        h2_sims.append(h2_sim)
    return {"h1_preference": h1_prefs, "h1_sim": h1_sims, "h2_sim": h2_sims}


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def run_trial(
    model: HookedLLM,
    prompt_text: str,
    problem_key: str,
    centroids: dict,
    layers: list[int],
    condition: str,
    h1_name: str,
    h2_name: str,
    max_new_tokens: int = 300,
    generates: bool = True,
    external_text: str | None = None,
) -> dict:
    """Run a single trial with optional generation + representation extraction."""

    tokenizer = model.tokenizer

    if external_text is not None:
        # external_cot condition: use saved text directly
        full_text = external_text
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_len = len(prompt_ids)
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        generated_text = tokenizer.decode(full_ids[prompt_len:], skip_special_tokens=True)
        generated_token_strings = [tokenizer.decode([tid]) for tid in full_ids[prompt_len:]]
        n_generated = len(full_ids) - prompt_len
    elif generates:
        # Generation conditions: generate text
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        prompt_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = model.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        full_ids_tensor = outputs[0]
        full_ids = full_ids_tensor.tolist()
        generated_ids = full_ids[prompt_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        generated_token_strings = [tokenizer.decode([tid]) for tid in generated_ids]
        n_generated = len(generated_ids)
        full_text = tokenizer.decode(full_ids, skip_special_tokens=True)
    else:
        # No generation: process prompt only (external_disambig)
        full_text = prompt_text
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        prompt_len = len(full_ids)
        generated_text = ""
        generated_token_strings = []
        n_generated = 0

    try:
        # Detect commitment point (for conditions with generated/external text)
        if n_generated > 0:
            commitment = detect_commitment_point(
                generated_text, generated_token_strings,
                prompt_len, h1_name, h2_name,
            )
        else:
            commitment = CommitmentDetection()

        # Extract representations via single forward pass
        with torch.no_grad():
            _, cache = model.forward_with_cache(full_text, layers=layers)

        # Compute metrics per layer
        layer_results = {}
        for layer in layers:
            reps_tensor = cache.get_residual_stream(layer)
            if reps_tensor is None:
                continue
            reps_np = reps_tensor[0].cpu().float().numpy()  # (seq_len, hidden_dim)
            representations = [reps_np[i] for i in range(reps_np.shape[0])]

            h1_centroid = centroids[problem_key][layer]["H1"]
            h2_centroid = centroids[problem_key][layer]["H2"]

            # Velocity
            velocities = compute_velocity_series(representations)
            cumulative_drift = compute_cumulative_drift(representations)

            # Preference trajectory
            pref_traj = compute_preference_trajectory(representations, h1_centroid, h2_centroid)

            # Velocity spike detection
            spike_pos, spike_mag = detect_velocity_spike(
                velocities, baseline_window=min(20, prompt_len), threshold_std=2.0
            )

            # Representation commitment point
            rep_commit = detect_representation_commitment(
                pref_traj["h1_preference"], prompt_len
            )

            # Pre/post commitment stats
            commit_pos = commitment.lexical_position or commitment.answer_position
            if commit_pos is not None and commit_pos < len(velocities):
                pre_vel = velocities[prompt_len:commit_pos] if commit_pos > prompt_len else []
                post_vel = velocities[commit_pos:] if commit_pos < len(velocities) else []
                pre_pref = pref_traj["h1_preference"][prompt_len:commit_pos]
                post_pref = pref_traj["h1_preference"][commit_pos:]
                commit_velocity = velocities[commit_pos] if commit_pos < len(velocities) else 0.0
            else:
                pre_vel = velocities[prompt_len:]
                post_vel = []
                pre_pref = pref_traj["h1_preference"][prompt_len:]
                post_pref = []
                commit_velocity = 0.0

            layer_results[str(layer)] = {
                "velocities": velocities[prompt_len:],  # Only generated portion
                "h1_preference": pref_traj["h1_preference"][prompt_len:],
                "spike_position": spike_pos,
                "spike_magnitude": spike_mag,
                "rep_commitment_position": rep_commit,
                "pre_vel_mean": float(np.mean(pre_vel)) if pre_vel else 0.0,
                "post_vel_mean": float(np.mean(post_vel)) if post_vel else 0.0,
                "pre_pref_mean": float(np.mean(pre_pref)) if pre_pref else 0.5,
                "post_pref_mean": float(np.mean(post_pref)) if post_pref else 0.5,
                "pref_change": (float(np.mean(post_pref)) - float(np.mean(pre_pref))) if post_pref and pre_pref else 0.0,
                "commit_velocity": commit_velocity,
            }

        # Timing analysis: offset between velocity spike and lexical commitment
        commit_pos = commitment.lexical_position or commitment.answer_position
        spike_commit_offset = None
        if spike_pos is not None and commit_pos is not None:
            spike_commit_offset = spike_pos - commit_pos  # negative = spike before text

        return {
            "problem_key": problem_key,
            "condition": condition,
            "prompt_len": prompt_len,
            "n_generated": n_generated,
            "generated_text": generated_text[:500],
            "full_text_preview": full_text[:200] + "...",
            "commitment": asdict(commitment),
            "spike_commit_offset": spike_commit_offset,
            "layers": layer_results,
            "error": None,
        }

    except Exception as e:
        return {
            "problem_key": problem_key,
            "condition": condition,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_results(trials: list[dict], layers: list[int]) -> dict:
    successful = [t for t in trials if t.get("error") is None]
    if not successful:
        return {"n_trials": 0, "n_errors": len(trials)}

    agg = {"n_trials": len(successful), "n_errors": len(trials) - len(successful), "layers": {}}

    # Commitment detection stats
    n_lexical = sum(1 for t in successful if t["commitment"]["lexical_position"] is not None)
    n_answer = sum(1 for t in successful if t["commitment"]["answer_position"] is not None)
    offsets = [t["spike_commit_offset"] for t in successful if t["spike_commit_offset"] is not None]

    agg["commitment_stats"] = {
        "lexical_detection_rate": n_lexical / len(successful),
        "answer_detection_rate": n_answer / len(successful),
        "spike_commit_offset_mean": float(np.mean(offsets)) if offsets else None,
        "spike_commit_offset_std": float(np.std(offsets)) if offsets else None,
    }

    for layer in layers:
        ls = str(layer)
        pref_changes = [t["layers"][ls]["pref_change"] for t in successful if ls in t["layers"]]
        commit_vels = [t["layers"][ls]["commit_velocity"] for t in successful if ls in t["layers"]]
        spike_detected = [t["layers"][ls]["spike_position"] is not None for t in successful if ls in t["layers"]]

        agg["layers"][ls] = {
            "pref_change_mean": float(np.mean(pref_changes)) if pref_changes else 0.0,
            "pref_change_std": float(np.std(pref_changes)) if pref_changes else 0.0,
            "commit_vel_mean": float(np.mean(commit_vels)) if commit_vels else 0.0,
            "commit_vel_std": float(np.std(commit_vels)) if commit_vels else 0.0,
            "spike_detection_rate": float(np.mean(spike_detected)) if spike_detected else 0.0,
        }

    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CoT Disambiguation Experiment")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--layers", type=str, default=None)
    parser.add_argument("--conditions", type=str, default=None,
                        help="Comma-separated conditions (default: all except external_cot)")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--output-dir", type=str, default="results/cot_disambig_pilot")
    parser.add_argument("--saved-cots", type=str, default=None,
                        help="Path to saved CoTs for external_cot condition")
    parser.add_argument("--categories", type=str, default="A,B",
                        help="Which categories to run: A, B, or A,B")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(exist_ok=True)

    # Conditions
    default_conditions = ["cot_ambig", "cot_disambig", "direct_answer", "external_disambig"]
    if args.saved_cots:
        default_conditions.append("external_cot")
    conditions = args.conditions.split(",") if args.conditions else default_conditions

    categories = args.categories.split(",")

    # Load model
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    print("=" * 60)
    print("COT DISAMBIGUATION EXPERIMENT")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Conditions: {conditions}")
    print(f"Categories: {categories}")
    print(f"Max new tokens: {args.max_new_tokens}")

    print(f"\nLoading model...")
    model = HookedLLM.from_pretrained(args.model, dtype=dtype_map[args.dtype])
    print(f"Loaded. Layers: {model.num_layers}, Hidden: {model.hidden_size}")

    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    else:
        n = model.num_layers
        layers = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    print(f"Layers: {layers}")

    # Compute centroids
    print("\nComputing centroids...")
    cat_a_centroids = {}
    cat_b_centroids = {}

    if "A" in categories:
        pairs = get_pilot_pairs()
        cat_a_centroids = compute_centroids_category_a(model, pairs, layers)
        for pair in pairs:
            sim = cat_a_centroids[pair.name][layers[-1]]["cos_sim"]
            print(f"  {pair.name} L{layers[-1]}: cos_sim = {sim:.4f}")

    if "B" in categories:
        problems = get_reasoning_problems()
        cat_b_centroids = compute_centroids_category_b(model, problems, layers)
        for prob in problems:
            sim = cat_b_centroids[prob.problem_id][layers[-1]]["cos_sim"]
            print(f"  {prob.problem_id} L{layers[-1]}: cos_sim = {sim:.4f}")

    # Load saved CoTs if needed
    saved_cots = {}
    if args.saved_cots and Path(args.saved_cots).exists():
        with open(args.saved_cots) as f:
            saved_cots = json.load(f)
        print(f"Loaded {len(saved_cots)} saved CoTs")

    # Save config
    config = {
        "model": args.model,
        "layers": layers,
        "conditions": conditions,
        "categories": categories,
        "max_new_tokens": args.max_new_tokens,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Run experiment
    print("\n" + "=" * 60)
    print("RUNNING EXPERIMENT")
    print("=" * 60)

    all_results = {}
    cots_to_save = {}  # For saving generated CoTs for external_cot condition

    for condition in conditions:
        print(f"\n--- Condition: {condition} ---")
        condition_trials = []
        generates = condition in ("cot_ambig", "cot_disambig", "direct_answer")

        # Category A trials
        if "A" in categories:
            pairs = get_pilot_pairs()
            for pair in pairs:
                stimuli = get_stimuli(pair.name)
                for stim in tqdm(stimuli, desc=f"{condition} {pair.name}", leave=False):

                    if condition == "external_cot":
                        # Use saved CoT
                        key = stim.stimulus_id
                        if key not in saved_cots:
                            continue
                        prompt_text = build_cot_prompt(stim, pair, model.tokenizer, "cot_ambig")
                        result = run_trial(
                            model, prompt_text, pair.name, cat_a_centroids,
                            layers, condition, pair.h1_name, pair.h2_name,
                            generates=False, external_text=saved_cots[key],
                        )
                    else:
                        prompt_text = build_cot_prompt(stim, pair, model.tokenizer, condition)
                        result = run_trial(
                            model, prompt_text, pair.name, cat_a_centroids,
                            layers, condition, pair.h1_name, pair.h2_name,
                            max_new_tokens=args.max_new_tokens, generates=generates,
                        )

                    result["stimulus_id"] = stim.stimulus_id
                    result["category"] = "A"
                    condition_trials.append(result)

                    # Save CoT for external_cot condition later
                    if condition == "cot_ambig" and result.get("error") is None:
                        full_text = prompt_text
                        if result.get("generated_text"):
                            gen_ids = model.tokenizer.encode(prompt_text, add_special_tokens=False)
                            # Reconstruct full text
                            full_gen = result.get("generated_text", "")
                            cots_to_save[stim.stimulus_id] = prompt_text + full_gen

                    # Save raw trial
                    fname = f"{stim.stimulus_id}_{condition}.json"
                    with open(raw_dir / fname, "w") as f:
                        json.dump(convert_numpy(result), f, indent=2)

                    clear_gpu_memory()

        # Category B trials
        if "B" in categories and condition != "external_cot":
            problems = get_reasoning_problems()
            for prob in tqdm(problems, desc=f"{condition} cat_b", leave=False):
                prompt_text = build_reasoning_prompt(prob, model.tokenizer, condition)
                result = run_trial(
                    model, prompt_text, prob.problem_id, cat_b_centroids,
                    layers, condition, prob.h1_answer, prob.h2_answer,
                    max_new_tokens=args.max_new_tokens, generates=generates,
                )
                result["problem_id"] = prob.problem_id
                result["category"] = "B"
                result["correct_answer"] = prob.correct
                condition_trials.append(result)

                fname = f"{prob.problem_id}_{condition}.json"
                with open(raw_dir / fname, "w") as f:
                    json.dump(convert_numpy(result), f, indent=2)

                clear_gpu_memory()

        # Aggregate
        agg = aggregate_results(condition_trials, layers)
        all_results[condition] = {
            "aggregate": agg,
            "trials": [convert_numpy(t) for t in condition_trials],
        }

        # Print summary
        n_ok = agg.get("n_trials", 0)
        print(f"  Completed: {n_ok}/{n_ok + agg.get('n_errors', 0)}")
        if n_ok > 0 and "layers" in agg:
            cs = agg.get("commitment_stats", {})
            print(f"  Commitment detection: lexical={cs.get('lexical_detection_rate', 0):.0%}, "
                  f"answer={cs.get('answer_detection_rate', 0):.0%}")
            if cs.get("spike_commit_offset_mean") is not None:
                print(f"  Spike-commit offset: {cs['spike_commit_offset_mean']:+.1f} tokens")
            for layer in layers:
                la = agg["layers"][str(layer)]
                print(f"  Layer {layer}: pref_change={la['pref_change_mean']:+.4f}, "
                      f"commit_vel={la['commit_vel_mean']:.1f}, "
                      f"spike_rate={la['spike_detection_rate']:.0%}")

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(convert_numpy(all_results), f, indent=2)

    # Save CoTs for external_cot condition
    if cots_to_save:
        with open(output_dir / "saved_cots.json", "w") as f:
            json.dump(cots_to_save, f, indent=2)
        print(f"\nSaved {len(cots_to_save)} CoTs for external_cot condition")

    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'Condition':<20} {'Layer':<8} {'Pref Change':<16} {'Commit Vel':<14} {'Spike Rate'}")
    print("-" * 72)
    for condition in conditions:
        agg = all_results[condition]["aggregate"]
        if agg.get("n_trials", 0) == 0:
            continue
        for layer in layers:
            la = agg["layers"][str(layer)]
            print(f"{condition:<20} {layer:<8} {la['pref_change_mean']:+.4f} +/- {la['pref_change_std']:.4f}"
                  f"  {la['commit_vel_mean']:>8.1f}  {la['spike_detection_rate']:>8.0%}")

    print(f"\nAll outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
