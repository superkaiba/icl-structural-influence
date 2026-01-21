#!/usr/bin/env python3
"""
Compare three LOO influence metrics:
1. Original Ratio: cross_cluster_dist / within_cluster_dist
2. Dirichlet Energy: E(X) = Σ A_{ij} ||x_i - x_j||² (Park et al.)
3. CSS: -Cov(loss, φ) (Lee et al.)

This script runs the same LOO experiment with all three metrics to compare them.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).parent))
from run_hierarchy_and_semantic_experiments import SemanticConflictGraph


def load_model(model_name="meta-llama/Llama-3.1-8B"):
    """Load model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"Model loaded: {n_layers} layers")
    return model, tokenizer, n_layers


def get_representations_and_logits(model, tokenizer, context_tokens, layer_idx):
    """Get representations at specified layer AND logits for loss computation."""
    prompt = " ".join(context_tokens)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    representations = {}

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            representations['hidden'] = output[0].detach()
        else:
            representations['hidden'] = output.detach()

    if layer_idx == 0:
        handle = model.model.embed_tokens.register_forward_hook(hook_fn)
    else:
        handle = model.model.layers[layer_idx - 1].register_forward_hook(hook_fn)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]  # [seq_len, vocab_size]

    handle.remove()

    hidden = representations['hidden'][0]

    # Map tokens to representations
    token_reps = {}
    token_positions = defaultdict(list)
    current_pos = 0

    for i, token in enumerate(context_tokens):
        word_tokens = tokenizer.encode(" " + token, add_special_tokens=False)
        n_subtokens = len(word_tokens)

        if current_pos + n_subtokens <= hidden.shape[0]:
            rep = hidden[current_pos:current_pos + n_subtokens].mean(dim=0)
            if token not in token_reps:
                token_reps[token] = []
            token_reps[token].append(rep)
            token_positions[token].append(current_pos)
            current_pos += n_subtokens

    # Average repeated tokens
    for token in token_reps:
        token_reps[token] = torch.stack(token_reps[token]).mean(dim=0)

    # Compute per-token loss
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    if logits.shape[0] > 1:
        shift_logits = logits[:-1].float()
        shift_labels = inputs['input_ids'][0, 1:]
        token_losses = loss_fn(shift_logits, shift_labels).cpu().numpy()
    else:
        token_losses = np.array([])

    return token_reps, hidden, token_losses, inputs['input_ids'][0]


# =============================================================================
# METRIC 1: Original Ratio (cross/within cluster distance)
# =============================================================================

def compute_ratio_metric(token_reps, graph):
    """Compute original ratio metric."""
    available_tokens = set(token_reps.keys())

    # Cross-cluster pairs (semantic pairs in different clusters)
    cross_pairs = [(t1, t2) for t1, t2, _ in graph.get_semantic_pairs()
                   if t1 in available_tokens and t2 in available_tokens]

    # Within-cluster pairs
    within_pairs = []
    for cid, tokens in graph.graph_clusters.items():
        for i, t1 in enumerate(tokens):
            for t2 in tokens[i+1:]:
                if t1 in available_tokens and t2 in available_tokens:
                    within_pairs.append((t1, t2))

    def mean_dist(pairs):
        if not pairs:
            return np.nan
        dists = []
        for t1, t2 in pairs:
            dists.append(torch.norm(token_reps[t1] - token_reps[t2]).item())
        return np.mean(dists) if dists else np.nan

    cross_dist = mean_dist(cross_pairs)
    within_dist = mean_dist(within_pairs)

    # Handle edge cases with relative epsilon to bound ratio (max ~1000)
    if np.isnan(cross_dist) or np.isnan(within_dist):
        ratio = np.nan
    else:
        eps = 1e-3
        min_within = max(eps * cross_dist, 1e-10)
        within_dist_bounded = max(within_dist, min_within)
        ratio = cross_dist / within_dist_bounded

    return {
        'ratio': ratio,
        'cross_dist': cross_dist,
        'within_dist': within_dist
    }


# =============================================================================
# METRIC 2: Dirichlet Energy (Park et al.)
# =============================================================================

def compute_dirichlet_energy(token_reps, graph):
    """
    Compute Dirichlet Energy: E(X) = Σ_{i,j} A_{i,j} ||x_i - x_j||²

    Lower energy = representations respect graph structure (adjacent nodes close)

    Modified: Only compute over tokens that appear in context (not full graph)
    """
    # Build adjacency from graph clusters (within-cluster = adjacent)
    available_tokens = set(token_reps.keys())

    energy = 0.0
    edge_count = 0

    # Within-cluster pairs (adjacent in graph)
    for cid, tokens in graph.graph_clusters.items():
        for i, t1 in enumerate(tokens):
            for t2 in tokens[i+1:]:
                if t1 in available_tokens and t2 in available_tokens:
                    diff = token_reps[t1] - token_reps[t2]
                    energy += torch.sum(diff ** 2).item()
                    edge_count += 1

    # Normalize by edge count
    if edge_count > 0:
        energy /= edge_count
        return energy
    else:
        # Fallback: compute average pairwise distance among available tokens
        tokens_list = list(available_tokens)
        if len(tokens_list) < 2:
            return np.nan

        total_dist = 0.0
        pair_count = 0
        for i, t1 in enumerate(tokens_list):
            for t2 in tokens_list[i+1:]:
                diff = token_reps[t1] - token_reps[t2]
                total_dist += torch.sum(diff ** 2).item()
                pair_count += 1

        return total_dist / pair_count if pair_count > 0 else np.nan


# =============================================================================
# METRIC 3: CSS - Covariance Sample Significance (Lee et al.)
# =============================================================================

def compute_css_components(token_reps, token_losses, context_tokens, graph):
    """
    Compute CSS components: loss and phi (between-cluster variance).

    CSS = -Cov(loss, phi) computed across samples
    For single sample, return the components.
    """
    # Compute cluster centroids
    cluster_reps = defaultdict(list)
    for token in context_tokens:
        if token in token_reps:
            cid = graph.token_to_graph_cluster.get(token)
            if cid is not None:
                cluster_reps[cid].append(token_reps[token])

    cluster_means = {}
    for cid, reps in cluster_reps.items():
        if reps:
            cluster_means[cid] = torch.stack(reps).mean(dim=0)

    # Phi = between-cluster variance
    if len(cluster_means) >= 2:
        global_mean = torch.stack(list(cluster_means.values())).mean(dim=0)
        phi = sum(torch.sum((m - global_mean)**2).item() for m in cluster_means.values())
        phi /= len(cluster_means)
    else:
        phi = 0.0

    # Mean loss
    mean_loss = np.mean(token_losses) if len(token_losses) > 0 else 0.0

    return mean_loss, phi


# =============================================================================
# LOO Influence Computation
# =============================================================================

def compute_all_loo_influences(model, tokenizer, context_tokens, graph, layer_idx, pos):
    """Compute LOO influence using all three metrics."""

    # Full context
    full_reps, full_hidden, full_losses, _ = get_representations_and_logits(
        model, tokenizer, context_tokens, layer_idx
    )

    full_ratio = compute_ratio_metric(full_reps, graph)
    full_energy = compute_dirichlet_energy(full_reps, graph)
    full_loss, full_phi = compute_css_components(full_reps, full_losses, context_tokens, graph)

    # LOO context (remove position)
    loo_tokens = context_tokens[:pos] + context_tokens[pos+1:]
    if len(loo_tokens) < 2:
        return None

    loo_reps, loo_hidden, loo_losses, _ = get_representations_and_logits(
        model, tokenizer, loo_tokens, layer_idx
    )

    loo_ratio = compute_ratio_metric(loo_reps, graph)
    loo_energy = compute_dirichlet_energy(loo_reps, graph)
    loo_loss, loo_phi = compute_css_components(loo_reps, loo_losses, loo_tokens, graph)

    # Compute influences
    # Ratio: positive = removing hurts (increases ratio means better structure)
    ratio_influence = full_ratio['ratio'] - loo_ratio['ratio']

    # Dirichlet: positive = removing hurts (increases energy means worse structure)
    # Note: Lower energy is better, so influence = loo_energy - full_energy
    energy_influence = loo_energy - full_energy

    # CSS components (will aggregate across samples)
    return {
        'ratio_influence': ratio_influence,
        'energy_influence': energy_influence,
        'full_loss': full_loss,
        'full_phi': full_phi,
        'loo_loss': loo_loss,
        'loo_phi': loo_phi,
        'cross_dist_influence': full_ratio['cross_dist'] - loo_ratio['cross_dist'],
        'within_dist_influence': full_ratio['within_dist'] - loo_ratio['within_dist'],
    }


def identify_token_types(context_tokens, graph):
    """Identify bridge and anchor positions."""
    bridge_positions = []
    anchor_positions = []

    prev_cluster = None
    for i, token in enumerate(context_tokens):
        cluster = graph.token_to_graph_cluster.get(token)
        if prev_cluster is not None and cluster != prev_cluster:
            bridge_positions.append(i)
        else:
            anchor_positions.append(i)
        prev_cluster = cluster

    return bridge_positions, anchor_positions


# =============================================================================
# Main Experiment
# =============================================================================

def run_comparison_experiment(
    context_lengths=[4, 6, 7, 8, 10, 14, 20, 30, 50],
    n_trials=5,
    layer_idx=16,
    output_dir="results/loo_metric_comparison"
):
    """Run LOO experiment with all three metrics."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model, tokenizer, n_layers = load_model()
    graph = SemanticConflictGraph(seed=42, use_semantic_tokens=True)

    results = {
        'context_lengths': context_lengths,
        'n_trials': n_trials,
        'layer_idx': layer_idx,
        'by_N': {}
    }

    for N in context_lengths:
        print(f"\n{'='*60}")
        print(f"Context length N={N}")
        print(f"{'='*60}")

        # Collect influences by token type
        bridge_influences = defaultdict(list)
        anchor_influences = defaultdict(list)

        # For CSS: collect (loss, phi) pairs SEPARATELY for bridge and anchor
        css_bridge_data = {'losses': [], 'phis': []}
        css_anchor_data = {'losses': [], 'phis': []}
        css_all_data = {'losses': [], 'phis': []}

        for trial in tqdm(range(n_trials), desc=f"N={N}"):
            # Generate random walk
            graph_trial = SemanticConflictGraph(seed=42 + trial * 100 + N, use_semantic_tokens=True)
            context, _ = graph_trial.generate_random_walk(length=N, return_nodes=True)
            context_tokens = context.split()

            # Identify token types
            bridge_pos, anchor_pos, = identify_token_types(context_tokens, graph_trial)

            # Sample positions to test (all for small N, sample for large N)
            if N > 20:
                test_positions = list(range(0, N, 3))[:10]
            else:
                test_positions = list(range(N))

            for pos in test_positions:
                influences = compute_all_loo_influences(
                    model, tokenizer, context_tokens, graph_trial, layer_idx, pos
                )

                if influences is None:
                    continue

                # Store by token type
                if pos in bridge_pos:
                    for key, val in influences.items():
                        if 'influence' in key or 'dist' in key:
                            bridge_influences[key].append(val)
                    # Store CSS components for bridge
                    css_bridge_data['losses'].append(influences['full_loss'])
                    css_bridge_data['phis'].append(influences['full_phi'])
                else:
                    for key, val in influences.items():
                        if 'influence' in key or 'dist' in key:
                            anchor_influences[key].append(val)
                    # Store CSS components for anchor
                    css_anchor_data['losses'].append(influences['full_loss'])
                    css_anchor_data['phis'].append(influences['full_phi'])

                # Store CSS components for all positions
                css_all_data['losses'].append(influences['full_loss'])
                css_all_data['phis'].append(influences['full_phi'])

        # Compute CSS for this context length - separately for bridge and anchor
        def compute_css(data):
            if len(data['losses']) > 1:
                cov_matrix = np.cov(data['losses'], data['phis'])
                if cov_matrix.shape == (2, 2):
                    return -cov_matrix[0, 1]
            return np.nan

        css_bridge = compute_css(css_bridge_data)
        css_anchor = compute_css(css_anchor_data)
        css_all = compute_css(css_all_data)

        # Aggregate results
        results['by_N'][N] = {
            'bridge': {k: {'mean': np.mean(v), 'std': np.std(v), 'n': len(v)}
                      for k, v in bridge_influences.items() if v},
            'anchor': {k: {'mean': np.mean(v), 'std': np.std(v), 'n': len(v)}
                      for k, v in anchor_influences.items() if v},
            'css': css_all,
            'css_bridge': css_bridge,
            'css_anchor': css_anchor,
            'css_bridge_n': len(css_bridge_data['losses']),
            'css_anchor_n': len(css_anchor_data['losses']),
        }

        # Print summary
        print(f"\n  Bridge (n={len(bridge_influences.get('ratio_influence', []))}):")
        for metric in ['ratio_influence', 'energy_influence']:
            if metric in bridge_influences:
                vals = bridge_influences[metric]
                print(f"    {metric}: {np.mean(vals):+.4f} ± {np.std(vals):.4f}")

        print(f"  Anchor (n={len(anchor_influences.get('ratio_influence', []))}):")
        for metric in ['ratio_influence', 'energy_influence']:
            if metric in anchor_influences:
                vals = anchor_influences[metric]
                print(f"    {metric}: {np.mean(vals):+.4f} ± {np.std(vals):.4f}")

        print(f"  CSS (all): {css_all:.4f}")
        print(f"  CSS (bridge, n={len(css_bridge_data['losses'])}): {css_bridge:.4f}")
        print(f"  CSS (anchor, n={len(css_anchor_data['losses'])}): {css_anchor:.4f}")

    # Save results
    results_path = output_path / "comparison_results.json"

    # Convert to serializable format
    def to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        return obj

    with open(results_path, 'w') as f:
        json.dump(to_serializable(results), f, indent=2)
    print(f"\nSaved results to {results_path}")

    # Create comparison plot
    create_comparison_plot(results, output_path)

    return results


def create_comparison_plot(results, output_path):
    """Create side-by-side comparison of all three metrics."""

    context_lengths = results['context_lengths']

    # Extract data
    ratio_bridge = []
    ratio_anchor = []
    energy_bridge = []
    energy_anchor = []
    css_values = []
    css_bridge_values = []
    css_anchor_values = []

    ratio_bridge_std = []
    ratio_anchor_std = []
    energy_bridge_std = []
    energy_anchor_std = []

    for N in context_lengths:
        data = results['by_N'][N]

        # Ratio
        if 'ratio_influence' in data['bridge']:
            ratio_bridge.append(data['bridge']['ratio_influence']['mean'])
            ratio_bridge_std.append(data['bridge']['ratio_influence']['std'])
        else:
            ratio_bridge.append(np.nan)
            ratio_bridge_std.append(0)

        if 'ratio_influence' in data['anchor']:
            ratio_anchor.append(data['anchor']['ratio_influence']['mean'])
            ratio_anchor_std.append(data['anchor']['ratio_influence']['std'])
        else:
            ratio_anchor.append(np.nan)
            ratio_anchor_std.append(0)

        # Energy
        if 'energy_influence' in data['bridge']:
            energy_bridge.append(data['bridge']['energy_influence']['mean'])
            energy_bridge_std.append(data['bridge']['energy_influence']['std'])
        else:
            energy_bridge.append(np.nan)
            energy_bridge_std.append(0)

        if 'energy_influence' in data['anchor']:
            energy_anchor.append(data['anchor']['energy_influence']['mean'])
            energy_anchor_std.append(data['anchor']['energy_influence']['std'])
        else:
            energy_anchor.append(np.nan)
            energy_anchor_std.append(0)

        # CSS - now with bridge and anchor
        css_values.append(data.get('css', np.nan))
        css_bridge_values.append(data.get('css_bridge', np.nan))
        css_anchor_values.append(data.get('css_anchor', np.nan))

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Original Ratio Metric
    ax1 = axes[0, 0]
    ax1.errorbar(context_lengths, ratio_bridge, yerr=ratio_bridge_std,
                 fmt='o-', capsize=3, label='Bridge', color='tab:red', linewidth=2)
    ax1.errorbar(context_lengths, ratio_anchor, yerr=ratio_anchor_std,
                 fmt='s-', capsize=3, label='Anchor', color='tab:blue', linewidth=2)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Context Length (N)')
    ax1.set_ylabel('LOO Influence')
    ax1.set_title('Original Metric: Ratio (cross/within dist)\nPositive = removing hurts structure')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')

    # Plot 2: Dirichlet Energy
    ax2 = axes[0, 1]
    ax2.errorbar(context_lengths, energy_bridge, yerr=energy_bridge_std,
                 fmt='o-', capsize=3, label='Bridge', color='tab:red', linewidth=2)
    ax2.errorbar(context_lengths, energy_anchor, yerr=energy_anchor_std,
                 fmt='s-', capsize=3, label='Anchor', color='tab:blue', linewidth=2)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Context Length (N)')
    ax2.set_ylabel('LOO Influence (Energy Change)')
    ax2.set_title('Park et al.: Dirichlet Energy\nPositive = removing increases energy (hurts)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')

    # Plot 3: CSS - now with bridge vs anchor
    ax3 = axes[1, 0]
    ax3.plot(context_lengths, css_bridge_values, 'o-', color='tab:red', linewidth=2, markersize=8, label='Bridge')
    ax3.plot(context_lengths, css_anchor_values, 's-', color='tab:blue', linewidth=2, markersize=8, label='Anchor')
    ax3.plot(context_lengths, css_values, 'd--', color='tab:green', linewidth=1, markersize=6, alpha=0.5, label='All')
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Context Length (N)')
    ax3.set_ylabel('CSS = -Cov(loss, φ)')
    ax3.set_title('Lee et al.: CSS (Covariance Sample Significance)\nPositive = structure helps prediction')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')

    # Plot 4: Metric Correlation
    ax4 = axes[1, 1]

    # Flatten bridge+anchor for correlation
    all_ratio = ratio_bridge + ratio_anchor
    all_energy = energy_bridge + energy_anchor

    # Remove NaN pairs
    valid = [(r, e) for r, e in zip(all_ratio, all_energy) if not (np.isnan(r) or np.isnan(e))]
    if len(valid) > 2:
        r_vals, e_vals = zip(*valid)
        corr = np.corrcoef(r_vals, e_vals)[0, 1]

        ax4.scatter(r_vals, e_vals, alpha=0.6, s=100)
        ax4.set_xlabel('Ratio Influence')
        ax4.set_ylabel('Energy Influence')
        ax4.set_title(f'Metric Correlation: r = {corr:.3f}')
        ax4.grid(True, alpha=0.3)

        # Add trend line
        z = np.polyfit(r_vals, e_vals, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(r_vals), max(r_vals), 100)
        ax4.plot(x_line, p(x_line), 'r--', alpha=0.5, label='Trend')
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'Insufficient data\nfor correlation',
                ha='center', va='center', transform=ax4.transAxes)

    plt.suptitle(f'LOO Influence Metric Comparison (Layer {results["layer_idx"]})\n'
                 f'{results["n_trials"]} trials per context length',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save
    plot_path = output_path / "metric_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.savefig(plot_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Saved plot to {plot_path}")

    plt.close()

    # Create additional detailed plot
    create_detailed_plot(results, output_path)


def create_detailed_plot(results, output_path):
    """Create detailed comparison with separate distance components."""

    context_lengths = results['context_lengths']

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Extract all metrics
    metrics_bridge = defaultdict(list)
    metrics_anchor = defaultdict(list)

    for N in context_lengths:
        data = results['by_N'][N]

        for metric in ['ratio_influence', 'energy_influence', 'cross_dist_influence', 'within_dist_influence']:
            if metric in data['bridge']:
                metrics_bridge[metric].append(data['bridge'][metric]['mean'])
            else:
                metrics_bridge[metric].append(np.nan)

            if metric in data['anchor']:
                metrics_anchor[metric].append(data['anchor'][metric]['mean'])
            else:
                metrics_anchor[metric].append(np.nan)

    css_values = [results['by_N'][N].get('css', np.nan) for N in context_lengths]
    css_bridge = [results['by_N'][N].get('css_bridge', np.nan) for N in context_lengths]
    css_anchor = [results['by_N'][N].get('css_anchor', np.nan) for N in context_lengths]

    plot_configs = [
        ('ratio_influence', 'Original: Ratio Influence', axes[0, 0]),
        ('energy_influence', 'Park: Dirichlet Energy Influence', axes[0, 1]),
        ('cross_dist_influence', 'Cross-Cluster Distance Change', axes[0, 2]),
        ('within_dist_influence', 'Within-Cluster Distance Change', axes[1, 0]),
    ]

    for metric, title, ax in plot_configs:
        ax.plot(context_lengths, metrics_bridge[metric], 'o-',
               label='Bridge', color='tab:red', linewidth=2)
        ax.plot(context_lengths, metrics_anchor[metric], 's-',
               label='Anchor', color='tab:blue', linewidth=2)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Context Length (N)')
        ax.set_ylabel('Influence')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')

    # CSS plot - now with bridge vs anchor
    ax_css = axes[1, 1]
    ax_css.plot(context_lengths, css_bridge, 'o-', color='tab:red', linewidth=2, markersize=8, label='Bridge')
    ax_css.plot(context_lengths, css_anchor, 's-', color='tab:blue', linewidth=2, markersize=8, label='Anchor')
    ax_css.plot(context_lengths, css_values, 'd--', color='tab:green', linewidth=1, markersize=6, alpha=0.5, label='All')
    ax_css.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax_css.set_xlabel('Context Length (N)')
    ax_css.set_ylabel('CSS = -Cov(loss, φ)')
    ax_css.set_title('Lee: CSS (Bridge vs Anchor)')
    ax_css.legend()
    ax_css.grid(True, alpha=0.3)
    ax_css.set_xscale('log')

    # Summary text
    ax_text = axes[1, 2]
    ax_text.axis('off')

    summary = """
METRIC INTERPRETATIONS:

1. Ratio Influence (Original)
   Positive: removing token decreases ratio
            (hurts structure)

2. Energy Influence (Park et al.)
   Positive: removing token increases energy
            (hurts structure alignment)

3. Cross-Dist Influence
   Positive: removing token decreases
            cross-cluster distance

4. Within-Dist Influence
   Positive: removing token decreases
            within-cluster distance

5. CSS (Lee et al.)
   Positive: structure helps prediction
   Negative: structure hurts prediction

KEY: Bridge tokens should show different
     patterns than anchor tokens if they
     play different structural roles.
"""
    ax_text.text(0.1, 0.9, summary, transform=ax_text.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Detailed LOO Metric Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()

    plot_path = output_path / "metric_comparison_detailed.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.savefig(plot_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Saved detailed plot to {plot_path}")

    plt.close()


if __name__ == "__main__":
    # Extended context lengths up to 500
    context_lengths = [6, 7, 8, 10, 12, 15, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500]

    results = run_comparison_experiment(
        context_lengths=context_lengths,
        n_trials=10,
        layer_idx=16,
    )
