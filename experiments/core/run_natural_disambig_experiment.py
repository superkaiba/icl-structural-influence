#!/usr/bin/env python3
"""
Natural Language Disambiguation Experiment.

Tests whether the velocity spike and preference shift phenomena from synthetic
graph walk experiments generalize to natural language. Uses hand-crafted passages
that are semantically ambiguous between two interpretations until a single
disambiguating word.

Runs two model arms:
  - Base model (Qwen2.5-7B): raw passage text
  - Instruct model (Qwen2.5-7B-Instruct): question prefix + chat template

Usage:
    # Quick test with gpt2
    python run_natural_disambig_experiment.py --model gpt2 --arm base

    # Full experiment
    python run_natural_disambig_experiment.py --model Qwen/Qwen2.5-7B --arm base
    python run_natural_disambig_experiment.py --model Qwen/Qwen2.5-7B-Instruct --arm instruct
"""

import argparse
import gc
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.hooked_model import HookedLLM
from src.data.natural_language_disambig import (
    get_pilot_pairs,
    get_all_stimuli,
    get_stimuli,
    get_reference_passages,
    get_pair_by_name,
    tokenize_stimulus,
    validate_stimulus,
    build_instruct_prompt,
    save_stimuli_to_json,
    DisambiguationStimulus,
    CategoryPair,
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
    """Convert numpy types to JSON-serializable formats."""
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
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Reference centroid computation
# ---------------------------------------------------------------------------

def compute_reference_centroids(
    model: HookedLLM,
    pairs: list[CategoryPair],
    layers: list[int],
    arm: str,
) -> dict:
    """
    Compute H1 and H2 reference centroids from unambiguous passages.

    For each category pair and layer, runs the reference passages through
    the model and averages the last-token representations.

    Returns:
        Dict: {pair_name: {layer: {"H1": np.ndarray, "H2": np.ndarray, "cos_sim": float}}}
    """
    centroids = {}

    for pair in pairs:
        print(f"\n  Computing centroids for {pair.name}...")
        ref_passages = get_reference_passages(pair.name)
        centroids[pair.name] = {}

        for layer in layers:
            h1_reps = []
            h2_reps = []

            # Process H1 reference passages
            for passage in ref_passages["H1"]:
                text = _format_text_for_arm(passage, pair, model.tokenizer, arm)
                with torch.no_grad():
                    _, cache = model.forward_with_cache(text, layers=[layer])
                    rep = cache.get_residual_stream(layer)[0, -1].cpu().float().numpy()
                    h1_reps.append(rep)

            # Process H2 reference passages
            for passage in ref_passages["H2"]:
                text = _format_text_for_arm(passage, pair, model.tokenizer, arm)
                with torch.no_grad():
                    _, cache = model.forward_with_cache(text, layers=[layer])
                    rep = cache.get_residual_stream(layer)[0, -1].cpu().float().numpy()
                    h2_reps.append(rep)

            h1_centroid = np.mean(h1_reps, axis=0)
            h2_centroid = np.mean(h2_reps, axis=0)
            sim = cosine_similarity(h1_centroid, h2_centroid)

            centroids[pair.name][layer] = {
                "H1": h1_centroid,
                "H2": h2_centroid,
                "cos_sim": sim,
            }

            print(f"    Layer {layer}: centroid cos_sim = {sim:.4f}")

    return centroids


def _format_text_for_arm(
    text: str,
    pair: CategoryPair,
    tokenizer,
    arm: str,
) -> str:
    """Format text for the given model arm (base or instruct)."""
    if arm == "instruct":
        question = f"Is the following passage about {pair.h1_name} or {pair.h2_name}?"
        messages = [{"role": "user", "content": f"{question}\n\n{text}"}]
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            return f"Q: {question}\n\n{text}\nA:"
    else:
        return text


# ---------------------------------------------------------------------------
# Preference computation
# ---------------------------------------------------------------------------

def compute_preference_trajectory(
    representations: list[np.ndarray],
    h1_centroid: np.ndarray,
    h2_centroid: np.ndarray,
) -> dict:
    """
    Compute H1 preference at each position using cosine similarity to centroids.

    Returns:
        Dict with 'h1_preference', 'h1_sim', 'h2_sim' lists.
    """
    h1_prefs = []
    h1_sims = []
    h2_sims = []

    for rep in representations:
        h1_sim = cosine_similarity(rep, h1_centroid)
        h2_sim = cosine_similarity(rep, h2_centroid)
        total = abs(h1_sim) + abs(h2_sim)
        if total < 1e-8:
            pref = 0.5
        else:
            pref = h1_sim / total

        h1_prefs.append(pref)
        h1_sims.append(h1_sim)
        h2_sims.append(h2_sim)

    return {
        "h1_preference": h1_prefs,
        "h1_sim": h1_sims,
        "h2_sim": h2_sims,
    }


# ---------------------------------------------------------------------------
# Incremental representation extraction
# ---------------------------------------------------------------------------

def extract_incremental_representations(
    model: HookedLLM,
    text: str,
    layers: list[int],
) -> dict[int, list[np.ndarray]]:
    """
    Process text token-by-token with KV-cache, collecting last-token
    representations at each step for each layer.

    Returns:
        Dict: {layer: [rep_at_step_0, rep_at_step_1, ...]}
    """
    token_ids = model.tokenizer.encode(text, add_special_tokens=False)
    reps = {layer: [] for layer in layers}
    past_kvs = None

    for i in range(len(token_ids)):
        input_ids = torch.tensor([[token_ids[i]]]).to(model.device)
        with torch.no_grad():
            _, cache, past_kvs = model.forward_incremental(
                input_ids,
                layers=layers,
                past_key_values=past_kvs,
            )
        for layer in layers:
            rep = cache.get_residual_stream(layer)[0, -1].cpu().float().numpy()
            reps[layer].append(rep)

    return reps


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def run_trial(
    model: HookedLLM,
    stimulus: DisambiguationStimulus,
    pair: CategoryPair,
    centroids: dict,
    layers: list[int],
    arm: str,
    condition: str,
) -> dict:
    """
    Run a single trial: extract representations and compute metrics.

    Args:
        condition: one of "disambig_50pct", "disambig_25pct", "disambig_75pct",
                   "no_disambig", "unambiguous_H1", "unambiguous_H2"
    """
    # Build the text for this condition
    # For timing conditions, truncate the ambiguous prefix so the disambig
    # word appears at the target fraction of the total passage.
    if condition == "no_disambig":
        raw_text = stimulus.ambiguous_text
    elif condition.startswith("unambiguous_"):
        hyp = "H1" if condition == "unambiguous_H1" else "H2"
        refs = get_reference_passages(pair.name)
        idx = hash(stimulus.stimulus_id) % len(refs[hyp])
        raw_text = refs[hyp][idx]
    elif condition.startswith("disambig_"):
        # Parse target fraction (e.g., "disambig_25pct" -> 0.25)
        frac_str = condition.replace("disambig_", "").replace("pct", "")
        target_frac = int(frac_str) / 100.0

        # Tokenize the parts separately to control timing
        ambig_tokens = model.tokenizer.encode(stimulus.ambiguous_text, add_special_tokens=False)
        disambig_tokens = model.tokenizer.encode(
            " " + stimulus.disambig_word, add_special_tokens=False
        )
        post_tokens = model.tokenizer.encode(
            " " + stimulus.post_disambig_text, add_special_tokens=False
        )

        # Total passage length with full ambiguous text
        full_len = len(ambig_tokens) + len(disambig_tokens) + len(post_tokens)
        # How many ambig tokens to keep so disambig word is at target_frac
        target_ambig_len = max(
            5,  # minimum context
            int(target_frac * full_len) - len(disambig_tokens)
        )
        target_ambig_len = min(target_ambig_len, len(ambig_tokens))  # can't exceed original

        # Truncate ambiguous tokens and decode back to text
        truncated_ambig_ids = ambig_tokens[:target_ambig_len]
        truncated_ambig_text = model.tokenizer.decode(truncated_ambig_ids)

        raw_text = f"{truncated_ambig_text} {stimulus.disambig_word} {stimulus.post_disambig_text}"
    else:
        raw_text = stimulus.full_text

    # Format for the model arm
    text = _format_text_for_arm(raw_text, pair, model.tokenizer, arm)

    # Find disambiguation token position in formatted text
    if condition.startswith("disambig_"):
        # Compute disambig position by tokenizing the prefix
        if condition.startswith("disambig_"):
            prefix_text = _format_text_for_arm(
                truncated_ambig_text, pair, model.tokenizer, arm
            )
        else:
            prefix_text = _format_text_for_arm(
                stimulus.ambiguous_text, pair, model.tokenizer, arm
            )
        pre_tokens = model.tokenizer.encode(prefix_text, add_special_tokens=False)
        full_tokens = model.tokenizer.encode(text, add_special_tokens=False)
        disambig_token_pos = len(pre_tokens)
        total_tokens = len(full_tokens)
    else:
        tokens = model.tokenizer.encode(text, add_special_tokens=False)
        disambig_token_pos = None
        total_tokens = len(tokens)

    try:
        # Extract representations incrementally
        reps = extract_incremental_representations(model, text, layers)

        # Compute metrics per layer
        layer_results = {}
        for layer in layers:
            layer_reps = reps[layer]
            h1_centroid = centroids[pair.name][layer]["H1"]
            h2_centroid = centroids[pair.name][layer]["H2"]

            # Velocity
            velocities = compute_velocity_series(layer_reps)
            cumulative_drift = compute_cumulative_drift(layer_reps)

            # Preference trajectory
            pref_traj = compute_preference_trajectory(layer_reps, h1_centroid, h2_centroid)

            # Velocity spike detection
            spike_pos, spike_mag = detect_velocity_spike(
                velocities, baseline_window=10, threshold_std=2.0
            )

            # Pre/post disambiguation stats
            if disambig_token_pos is not None and disambig_token_pos < len(velocities):
                pre_vel = velocities[1:disambig_token_pos] if disambig_token_pos > 1 else []
                post_vel = velocities[disambig_token_pos+1:] if disambig_token_pos+1 < len(velocities) else []
                pre_pref = pref_traj["h1_preference"][:disambig_token_pos]
                post_pref = pref_traj["h1_preference"][disambig_token_pos:]

                pre_vel_mean = float(np.mean(pre_vel)) if pre_vel else 0.0
                post_vel_mean = float(np.mean(post_vel)) if post_vel else 0.0
                pre_pref_mean = float(np.mean(pre_pref)) if pre_pref else 0.5
                post_pref_mean = float(np.mean(post_pref)) if post_pref else 0.5
                disambig_velocity = velocities[disambig_token_pos] if disambig_token_pos < len(velocities) else 0.0
            else:
                pre_vel_mean = float(np.mean(velocities[1:])) if len(velocities) > 1 else 0.0
                post_vel_mean = pre_vel_mean
                pre_pref_mean = float(np.mean(pref_traj["h1_preference"]))
                post_pref_mean = pre_pref_mean
                disambig_velocity = 0.0

            layer_results[str(layer)] = {
                "velocities": velocities,
                "cumulative_drift": cumulative_drift,
                "h1_preference": pref_traj["h1_preference"],
                "h1_sim": pref_traj["h1_sim"],
                "h2_sim": pref_traj["h2_sim"],
                "spike_position": spike_pos,
                "spike_magnitude": spike_mag,
                "pre_vel_mean": pre_vel_mean,
                "post_vel_mean": post_vel_mean,
                "pre_pref_mean": pre_pref_mean,
                "post_pref_mean": post_pref_mean,
                "pref_change": post_pref_mean - pre_pref_mean,
                "disambig_velocity": disambig_velocity,
            }

        return {
            "stimulus_id": stimulus.stimulus_id,
            "category_pair": pair.name,
            "true_hypothesis": stimulus.true_hypothesis,
            "condition": condition,
            "disambig_word": stimulus.disambig_word,
            "disambig_token_pos": disambig_token_pos,
            "total_tokens": total_tokens,
            "layers": layer_results,
            "error": None,
        }

    except Exception as e:
        return {
            "stimulus_id": stimulus.stimulus_id,
            "category_pair": pair.name,
            "condition": condition,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_results(trial_results: list[dict], layers: list[int]) -> dict:
    """Aggregate results across trials for a condition."""
    successful = [r for r in trial_results if r.get("error") is None]
    if not successful:
        return {"n_trials": 0, "n_errors": len(trial_results)}

    agg = {
        "n_trials": len(successful),
        "n_errors": len(trial_results) - len(successful),
        "layers": {},
    }

    for layer in layers:
        layer_str = str(layer)

        # Collect per-trial metrics
        pre_vel_means = []
        post_vel_means = []
        pre_pref_means = []
        post_pref_means = []
        pref_changes = []
        disambig_velocities = []
        spike_detected = []
        spike_magnitudes = []

        for r in successful:
            lr = r["layers"][layer_str]
            pre_vel_means.append(lr["pre_vel_mean"])
            post_vel_means.append(lr["post_vel_mean"])
            pre_pref_means.append(lr["pre_pref_mean"])
            post_pref_means.append(lr["post_pref_mean"])
            pref_changes.append(lr["pref_change"])
            disambig_velocities.append(lr["disambig_velocity"])
            spike_detected.append(lr["spike_position"] is not None)
            if lr["spike_position"] is not None:
                spike_magnitudes.append(lr["spike_magnitude"])

        agg["layers"][layer_str] = {
            "pre_vel_mean": float(np.mean(pre_vel_means)),
            "post_vel_mean": float(np.mean(post_vel_means)),
            "vel_change": float(np.mean(post_vel_means)) - float(np.mean(pre_vel_means)),
            "pre_pref_mean": float(np.mean(pre_pref_means)),
            "pre_pref_std": float(np.std(pre_pref_means)),
            "post_pref_mean": float(np.mean(post_pref_means)),
            "post_pref_std": float(np.std(post_pref_means)),
            "pref_change_mean": float(np.mean(pref_changes)),
            "pref_change_std": float(np.std(pref_changes)),
            "disambig_vel_mean": float(np.mean(disambig_velocities)),
            "disambig_vel_std": float(np.std(disambig_velocities)),
            "spike_detection_rate": float(np.mean(spike_detected)),
            "spike_magnitude_mean": float(np.mean(spike_magnitudes)) if spike_magnitudes else 0.0,
        }

    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Natural Language Disambiguation Experiment")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B",
                        help="Model name or path")
    parser.add_argument("--arm", type=str, choices=["base", "instruct"], default="base",
                        help="Experimental arm: base (raw text) or instruct (question prefix)")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices (default: auto-detect 5 layers)")
    parser.add_argument("--pairs", type=str, default=None,
                        help="Comma-separated pair names (default: all pilot pairs)")
    parser.add_argument("--conditions", type=str, default=None,
                        help="Comma-separated conditions (default: all)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--output-dir", type=str, default="results/natural_disambig_pilot")
    parser.add_argument("--validate-stimuli", action="store_true",
                        help="Run stimulus validation before experiment")
    parser.add_argument("--skip-centroids", action="store_true",
                        help="Skip centroid computation (load from previous run)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / f"{args.arm}_model"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(exist_ok=True)

    # Conditions
    all_conditions = [
        "disambig_50pct", "disambig_25pct", "disambig_75pct",
        "no_disambig", "unambiguous_H1", "unambiguous_H2",
    ]
    conditions = args.conditions.split(",") if args.conditions else all_conditions

    # Category pairs
    all_pairs = get_pilot_pairs()
    if args.pairs:
        pair_names = args.pairs.split(",")
        pairs = [get_pair_by_name(n) for n in pair_names]
    else:
        pairs = all_pairs

    # Load model
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    print("=" * 60)
    print("NATURAL LANGUAGE DISAMBIGUATION EXPERIMENT")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Arm: {args.arm}")
    print(f"Dtype: {args.dtype}")
    print(f"Pairs: {[p.name for p in pairs]}")
    print(f"Conditions: {conditions}")
    print(f"Output: {output_dir}")

    print(f"\nLoading model {args.model}...")
    model = HookedLLM.from_pretrained(args.model, dtype=dtype)
    print(f"Model loaded. Layers: {model.num_layers}, Hidden: {model.hidden_size}")

    # Auto-detect layers
    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    else:
        n = model.num_layers
        layers = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    print(f"Layers: {layers}")

    # Get and tokenize stimuli
    print("\nPreparing stimuli...")
    stimuli = []
    for pair in pairs:
        pair_stim = get_stimuli(pair.name)
        for s in pair_stim:
            tokenize_stimulus(s, model.tokenizer)
        stimuli.extend(pair_stim)
        print(f"  {pair.name}: {len(pair_stim)} stimuli, "
              f"avg tokens: {np.mean([s.total_tokens for s in pair_stim]):.0f}")

    # Save stimuli for reproducibility
    stim_path = Path(args.output_dir) / "stimuli.json"
    if not stim_path.exists():
        save_stimuli_to_json(stimuli, stim_path)
        print(f"Saved stimuli to {stim_path}")

    # Validate stimuli (optional)
    if args.validate_stimuli:
        print("\nValidating stimuli (checking for information leakage)...")
        for s in stimuli:
            pair = get_pair_by_name(s.category_pair)
            text = _format_text_for_arm(s.ambiguous_text, pair, model.tokenizer, args.arm)
            is_valid, rank, top_k = validate_stimulus(s, model, model.tokenizer)
            status = "OK" if is_valid else f"LEAK (rank {rank})"
            print(f"  {s.stimulus_id}: {status} | disambig='{s.disambig_word}' | top5={top_k[:5]}")

    # Compute reference centroids
    print("\nComputing reference centroids...")
    centroids = compute_reference_centroids(model, pairs, layers, args.arm)

    # Check centroid separation
    print("\nCentroid separation check:")
    for pair in pairs:
        for layer in layers:
            sim = centroids[pair.name][layer]["cos_sim"]
            flag = " WARNING: LOW SEPARATION" if sim > 0.85 else ""
            print(f"  {pair.name} L{layer}: cos_sim = {sim:.4f}{flag}")

    # Save config
    config = {
        "model": args.model,
        "arm": args.arm,
        "dtype": args.dtype,
        "layers": layers,
        "pairs": [p.name for p in pairs],
        "conditions": conditions,
        "n_stimuli": len(stimuli),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "centroid_separation": {
            pair.name: {
                str(layer): centroids[pair.name][layer]["cos_sim"]
                for layer in layers
            }
            for pair in pairs
        },
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Run experiment
    print("\n" + "=" * 60)
    print("RUNNING EXPERIMENT")
    print("=" * 60)

    all_results = {}
    for condition in conditions:
        print(f"\n--- Condition: {condition} ---")
        condition_trials = []

        for stim in tqdm(stimuli, desc=condition):
            pair = get_pair_by_name(stim.category_pair)
            result = run_trial(
                model, stim, pair, centroids, layers, args.arm, condition
            )
            condition_trials.append(result)

            # Save raw trial data
            trial_fname = f"{stim.category_pair}_{condition}_{stim.stimulus_id}.json"
            with open(raw_dir / trial_fname, "w") as f:
                json.dump(convert_numpy(result), f, indent=2)

            clear_gpu_memory()

        # Aggregate
        agg = aggregate_results(condition_trials, layers)
        all_results[condition] = {
            "aggregate": agg,
            "trials": [convert_numpy(r) for r in condition_trials],
        }

        # Print summary
        n_ok = agg.get("n_trials", 0)
        n_err = agg.get("n_errors", 0)
        print(f"  Completed: {n_ok}/{n_ok + n_err}")
        if n_ok > 0 and "layers" in agg:
            for layer in layers:
                la = agg["layers"][str(layer)]
                print(f"  Layer {layer}: pref_change={la['pref_change_mean']:+.4f}, "
                      f"disambig_vel={la['disambig_vel_mean']:.1f}, "
                      f"spike_rate={la['spike_detection_rate']:.0%}")

    # Save all results
    with open(output_dir / "results.json", "w") as f:
        json.dump(convert_numpy(all_results), f, indent=2)
    print(f"\nSaved results to {output_dir / 'results.json'}")

    # Print final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'Condition':<20} {'Layer':<8} {'Pref Change':<14} {'Disambig Vel':<14} {'Spike Rate'}")
    print("-" * 70)
    for condition in conditions:
        agg = all_results[condition]["aggregate"]
        if agg.get("n_trials", 0) == 0:
            print(f"{condition:<20} {'N/A':<8} {'no data'}")
            continue
        for layer in layers:
            la = agg["layers"][str(layer)]
            print(f"{condition:<20} {layer:<8} {la['pref_change_mean']:+.4f} ± {la['pref_change_std']:.4f}"
                  f"  {la['disambig_vel_mean']:>8.1f}"
                  f"  {la['spike_detection_rate']:>8.0%}")

    print(f"\nAll outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
