"""Track representation collapse during seahorse emoji generation.

Strategy: Use model.generate() with KV cache for speed, then re-run
forward passes at checkpoints to extract representations.
"""

import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_metrics(reps: np.ndarray) -> dict:
    """Compute collapse metrics on a set of representations."""
    n = reps.shape[0]
    if n < 3:
        return {"avg_cos_sim": 0, "avg_l2_dist": 0, "spread": 0,
                "effective_dim": 0, "intrinsic_dim": None}

    cos_sim_matrix = cosine_similarity(reps)
    mask = np.triu(np.ones_like(cos_sim_matrix, dtype=bool), k=1)
    avg_cos_sim = float(cos_sim_matrix[mask].mean())

    dists = pdist(reps, metric="euclidean")
    avg_l2_dist = float(dists.mean())

    centered = reps - reps.mean(axis=0)
    spread = float(np.trace(centered.T @ centered) / n)

    cov = np.cov(reps.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 0]
    if len(eigenvalues) > 0:
        total = eigenvalues.sum()
        effective_dim = float((total ** 2) / (eigenvalues ** 2).sum())
    else:
        effective_dim = 0.0

    intrinsic_dim = None
    if n >= 10:
        try:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=3).fit(reps)
            distances, _ = nn.kneighbors(reps)
            mu = distances[:, 2] / (distances[:, 1] + 1e-10)
            mu = np.sort(mu)
            mu = mu[mu > 1]
            if len(mu) > 2:
                intrinsic_dim = float(len(mu) / np.sum(np.log(mu)))
        except Exception:
            pass

    return {
        "avg_cos_sim": avg_cos_sim,
        "avg_l2_dist": avg_l2_dist,
        "spread": spread,
        "effective_dim": effective_dim,
        "intrinsic_dim": intrinsic_dim,
    }


def extract_representations_at_checkpoints(
    model, input_ids, layers, window_size=30, checkpoint_every=5
):
    """Run a single forward pass over the full sequence and extract reps at all positions.

    Then compute sliding window metrics at each checkpoint.
    """
    model_layers = model.model.layers

    # Register hooks for all layers at once
    hooks = []
    layer_outputs = {}

    def make_hook(layer_idx):
        def hook_fn(module, inp, output):
            hidden = output[0] if isinstance(output, tuple) else output
            layer_outputs[layer_idx] = hidden.detach().cpu().float()
        return hook_fn

    for layer_idx in layers:
        h = model_layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    # Single forward pass over entire sequence
    with torch.no_grad():
        model(input_ids=input_ids)

    for h in hooks:
        h.remove()

    # Extract per-position representations and compute sliding window metrics
    seq_len = input_ids.shape[1]
    per_step_metrics = {layer: [] for layer in layers}
    checkpoint_positions = list(range(window_size, seq_len, checkpoint_every))

    for layer_idx in layers:
        all_reps = layer_outputs[layer_idx][0].numpy()  # (seq_len, hidden_dim)

        for pos in checkpoint_positions:
            start = max(0, pos - window_size)
            window = all_reps[start:pos]
            if len(window) >= 5:
                metrics = compute_metrics(window)
            else:
                metrics = {k: None for k in ["avg_cos_sim", "avg_l2_dist", "spread", "effective_dim", "intrinsic_dim"]}
            per_step_metrics[layer_idx].append(metrics)

    return per_step_metrics, checkpoint_positions


def detect_loop_onset(tokens, min_repeats=3):
    """Find the token index where repetition begins."""
    text = "".join(tokens)
    for pattern_len in range(1, min(30, len(text) // min_repeats)):
        for start in range(len(text) - pattern_len * min_repeats):
            pattern = text[start:start + pattern_len]
            if len(pattern.strip()) == 0:
                continue
            count = 0
            pos = start
            while pos + pattern_len <= len(text) and text[pos:pos + pattern_len] == pattern:
                count += 1
                pos += pattern_len
            if count >= min_repeats:
                char_count = 0
                for token_idx, tok in enumerate(tokens):
                    char_count += len(tok)
                    if char_count >= start:
                        return token_idx
    return None


def plot_results(results_dict, output_dir):
    """Plot collapse metrics over generation steps for all prompts."""
    layers = [0, 7, 14, 21, 27]
    metric_names = ["avg_cos_sim", "avg_l2_dist", "spread", "effective_dim"]
    metric_labels = {
        "avg_cos_sim": "Cosine Similarity",
        "avg_l2_dist": "L2 Distance",
        "spread": "Spread (Variance)",
        "effective_dim": "Effective Dimension",
    }

    for prompt_name, data in results_dict.items():
        if data["n_generated"] < 10:
            continue

        fig, axes = plt.subplots(len(metric_names), len(layers),
                                figsize=(4 * len(layers), 3.5 * len(metric_names)))

        loop_onset = data.get("loop_onset")
        positions = data["checkpoint_positions"]

        for row, m_name in enumerate(metric_names):
            for col, layer in enumerate(layers):
                ax = axes[row, col]
                values = [s[m_name] for s in data["metrics"][str(layer)]]
                values = [v if v is not None else np.nan for v in values]

                ax.plot(positions[:len(values)], values, linewidth=1.5)

                if loop_onset is not None:
                    ax.axvline(x=loop_onset, color='red', linestyle='--',
                              alpha=0.8, label=f'Loop @tok {loop_onset}')

                if row == 0:
                    ax.set_title(f"Layer {layer}")
                if col == 0:
                    ax.set_ylabel(metric_labels[m_name])
                if row == len(metric_names) - 1:
                    ax.set_xlabel("Position in Sequence")
                if loop_onset and col == len(layers) - 1 and row == 0:
                    ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

        status = "LOOP" if loop_onset is not None else "no loop"
        plt.suptitle(f"Collapse During Generation: {prompt_name} ({status})\n"
                     f"Window={30}, {data['n_generated']} tokens generated",
                     fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / f"collapse_{prompt_name}.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: collapse_{prompt_name}.png")

    # Comparison plot: layer 27 cos sim for all prompts
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for prompt_name, data in results_dict.items():
        if data["n_generated"] < 10:
            continue
        positions = data["checkpoint_positions"]
        for ax_idx, (m_name, m_label) in enumerate([
            ("avg_cos_sim", "Cosine Similarity"),
            ("effective_dim", "Effective Dimension"),
        ]):
            ax = axes[ax_idx]
            values = [s[m_name] for s in data["metrics"]["27"]]
            values = [v if v is not None else np.nan for v in values]
            label = prompt_name
            loop_onset = data.get("loop_onset")
            if loop_onset is not None:
                label += f" (loop@{loop_onset})"
            style = '--' if 'control' in prompt_name or 'dolphin' in prompt_name else '-'
            ax.plot(positions[:len(values)], values, style, label=label, linewidth=1.5)
            ax.set_ylabel(m_label)
            ax.set_xlabel("Position in Sequence")
            ax.grid(True, alpha=0.3)

    axes[0].set_title("Layer 27: Cosine Similarity")
    axes[1].set_title("Layer 27: Effective Dimension")
    axes[0].legend(fontsize=7, loc='best')
    plt.suptitle("Seahorse vs Control: Representation Collapse Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "collapse_comparison_layer27.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: collapse_comparison_layer27.png")


def main():
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    layers = [0, 7, 14, 21, 27]
    max_new_tokens = 300
    window_size = 30

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    prompts = {
        "seahorse_list": [
            {"role": "user", "content": "List every single sea creature emoji one by one. Include seahorse, dolphin, whale, shark, octopus, squid, fish, blowfish, jellyfish, crab, lobster, shrimp, and any others you can think of. Show each emoji followed by its name."}
        ],
        "seahorse_repeat": [
            {"role": "user", "content": "Show me the seahorse emoji. Just output the emoji character repeated 50 times."}
        ],
        "seahorse_explain": [
            {"role": "user", "content": "What is the seahorse emoji? Show it to me and explain when to use it."}
        ],
        "dolphin_list": [
            {"role": "user", "content": "List every single sea creature emoji one by one. Include dolphin, whale, shark, octopus, squid, fish, blowfish, jellyfish, crab, lobster, shrimp, and any others you can think of. Show each emoji followed by its name."}
        ],
        "dolphin_repeat": [
            {"role": "user", "content": "Show me the dolphin emoji. Just output the emoji character repeated 50 times."}
        ],
        "platypus_repeat": [
            {"role": "user", "content": "Show me the platypus emoji. Just output the emoji character repeated 50 times."}
        ],
        "narwhal_repeat": [
            {"role": "user", "content": "Show me the narwhal emoji. Just output the emoji character repeated 50 times."}
        ],
    }

    output_dir = Path("results/seahorse_emoji_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for name, messages in prompts.items():
        print(f"\n{'='*60}")
        print(f"Generating: {name}")
        print(f"{'='*60}")

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = inputs.input_ids.shape[1]

        # Step 1: Fast generation with KV cache
        print(f"  Generating {max_new_tokens} tokens...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        full_ids = outputs[0]
        generated_ids = full_ids[prompt_len:]
        token_strings = [tokenizer.decode([tid]) for tid in generated_ids]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        n_generated = len(generated_ids)

        print(f"  Generated {n_generated} tokens")
        print(f"  Text: {generated_text[:200]}")

        loop_onset = detect_loop_onset(token_strings)
        if loop_onset is not None:
            # Adjust to be relative to full sequence
            loop_onset_seq = prompt_len + loop_onset
            print(f"  *** LOOP at generated token {loop_onset} (seq pos {loop_onset_seq})")
        else:
            loop_onset_seq = None
            print(f"  No loop detected")

        # Step 2: Single forward pass to extract ALL representations
        print(f"  Extracting representations (single forward pass)...")
        per_step_metrics, checkpoint_positions = extract_representations_at_checkpoints(
            model, full_ids.unsqueeze(0), layers,
            window_size=window_size, checkpoint_every=5
        )

        all_results[name] = {
            "tokens": token_strings,
            "generated_text": generated_text,
            "n_generated": n_generated,
            "prompt_len": prompt_len,
            "loop_onset": loop_onset_seq,
            "checkpoint_positions": checkpoint_positions,
            "metrics": {str(k): v for k, v in per_step_metrics.items()},
        }

        # Quick summary
        if per_step_metrics[27]:
            final = per_step_metrics[27][-1]
            print(f"  Final L27: cos_sim={final['avg_cos_sim']:.3f} eff_dim={final['effective_dim']:.1f}")

    # Save results
    save_data = {}
    for name, data in all_results.items():
        save_data[name] = {
            "tokens": data["tokens"][:50],  # First 50 tokens only
            "generated_text": data["generated_text"][:500],
            "n_generated": data["n_generated"],
            "prompt_len": data["prompt_len"],
            "loop_onset": data["loop_onset"],
            "checkpoint_positions": data["checkpoint_positions"],
            "metrics": {
                layer: [{k: float(v) if v is not None else None for k, v in step.items()}
                        for step in steps]
                for layer, steps in data["metrics"].items()
            },
        }

    with open(output_dir / "collapse_during_generation.json", "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {output_dir / 'collapse_during_generation.json'}")

    # Plot
    print("\nPlotting...")
    plot_results(all_results, output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
