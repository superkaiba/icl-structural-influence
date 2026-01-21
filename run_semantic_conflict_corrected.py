#!/usr/bin/env python3
"""
Re-run semantic conflict experiment with CORRECTED metrics:
- Centered cosine similarity (subtract mean before computing)
- L2 distance
- Extended context lengths to N=10,000

Key question: At what point does the model abandon pretrained semantics
for in-context learned structure?
"""

import json
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))
from models import HookedLLM
from run_hierarchy_and_semantic_experiments import SemanticConflictGraph


def collect_representations(model, tokenizer, graph, context_lengths, n_samples=50, layer_idx=-5):
    """Collect token representations at each context length."""
    token_reps = {ctx: {} for ctx in context_lengths}

    for ctx_len in context_lengths:
        print(f"  N={ctx_len}...", end=" ", flush=True)
        token_representations = defaultdict(list)

        for _ in range(n_samples):
            prompt, node_sequence = graph.generate_random_walk(length=ctx_len, return_nodes=True)
            tokens = tokenizer.encode(prompt, add_special_tokens=False)
            input_ids = torch.tensor([tokens]).to(model.device)

            with torch.no_grad():
                outputs = model(input_ids, output_hidden_states=True)
                hidden_states = outputs.hidden_states[layer_idx][0]

            token_texts = prompt.split()
            for pos, (node, token_text) in enumerate(zip(node_sequence, token_texts)):
                if pos < hidden_states.shape[0]:
                    rep = hidden_states[pos].cpu().float().numpy()
                    token_representations[token_text].append(rep)

        for token_text, reps in token_representations.items():
            if reps:
                token_reps[ctx_len][token_text] = np.mean(reps, axis=0)

        print(f"({len(token_reps[ctx_len])} tokens)")

    return token_reps


def compute_centered_cosine(reps_dict):
    """Compute centered cosine similarity matrix."""
    tokens = list(reps_dict.keys())
    X = np.array([reps_dict[t] for t in tokens])

    # Center the representations
    mean_rep = np.mean(X, axis=0)
    X_centered = X - mean_rep

    # Compute cosine similarity of centered representations
    cos_sim = cosine_similarity(X_centered)

    return {(tokens[i], tokens[j]): cos_sim[i, j]
            for i in range(len(tokens)) for j in range(len(tokens)) if i != j}


def compute_l2_distance(reps_dict):
    """Compute L2 distance matrix."""
    tokens = list(reps_dict.keys())
    X = np.array([reps_dict[t] for t in tokens])

    distances = {}
    for i, t1 in enumerate(tokens):
        for j, t2 in enumerate(tokens):
            if i != j:
                distances[(t1, t2)] = np.linalg.norm(X[i] - X[j])

    return distances


def analyze_semantic_vs_graph(token_reps, graph, context_lengths):
    """
    Compare semantic similarity vs graph similarity using CORRECTED metrics.
    """
    semantic_pairs = graph.get_semantic_pairs()  # (t1, t2, semantic_group)

    # Get graph-same pairs (semantically different but same graph cluster)
    graph_same_pairs = []
    for cluster_id, cluster_tokens in graph.graph_clusters.items():
        for i, t1 in enumerate(cluster_tokens):
            for t2 in cluster_tokens[i+1:]:
                if graph.token_to_semantic_group[t1] != graph.token_to_semantic_group[t2]:
                    graph_same_pairs.append((t1, t2))

    results = {
        "context_lengths": context_lengths,
        # Centered cosine similarity
        "centered_cos_semantic_same": [],
        "centered_cos_graph_same": [],
        # L2 distance
        "l2_semantic_same": [],
        "l2_graph_same": [],
        # Raw cosine for comparison
        "raw_cos_semantic_same": [],
        "raw_cos_graph_same": [],
        # All pairwise (for baseline)
        "centered_cos_all": [],
        "l2_all": [],
    }

    for ctx_len in context_lengths:
        reps = token_reps[ctx_len]
        if len(reps) < 4:
            for key in results:
                if key != "context_lengths":
                    results[key].append(np.nan)
            continue

        # Compute metrics
        centered_cos = compute_centered_cosine(reps)
        l2_dist = compute_l2_distance(reps)

        # Raw cosine for comparison
        tokens = list(reps.keys())
        X = np.array([reps[t] for t in tokens])
        raw_cos_matrix = cosine_similarity(X)
        raw_cos = {(tokens[i], tokens[j]): raw_cos_matrix[i, j]
                   for i in range(len(tokens)) for j in range(len(tokens)) if i != j}

        # Semantic-same, graph-different
        sem_same_centered = []
        sem_same_l2 = []
        sem_same_raw = []
        for t1, t2, _ in semantic_pairs:
            if t1 in reps and t2 in reps:
                sem_same_centered.append(centered_cos.get((t1, t2), np.nan))
                sem_same_l2.append(l2_dist.get((t1, t2), np.nan))
                sem_same_raw.append(raw_cos.get((t1, t2), np.nan))

        # Graph-same, semantic-different
        graph_same_centered = []
        graph_same_l2 = []
        graph_same_raw = []
        for t1, t2 in graph_same_pairs:
            if t1 in reps and t2 in reps:
                graph_same_centered.append(centered_cos.get((t1, t2), np.nan))
                graph_same_l2.append(l2_dist.get((t1, t2), np.nan))
                graph_same_raw.append(raw_cos.get((t1, t2), np.nan))

        # All pairs (for baseline)
        all_centered = list(centered_cos.values())
        all_l2 = list(l2_dist.values())

        results["centered_cos_semantic_same"].append(np.nanmean(sem_same_centered))
        results["centered_cos_graph_same"].append(np.nanmean(graph_same_centered))
        results["l2_semantic_same"].append(np.nanmean(sem_same_l2))
        results["l2_graph_same"].append(np.nanmean(graph_same_l2))
        results["raw_cos_semantic_same"].append(np.nanmean(sem_same_raw))
        results["raw_cos_graph_same"].append(np.nanmean(graph_same_raw))
        results["centered_cos_all"].append(np.nanmean(all_centered))
        results["l2_all"].append(np.nanmean(all_l2))

    return results


def find_crossover(x_vals, y1_vals, y2_vals, metric="higher_better"):
    """Find where y2 crosses y1.

    For centered cosine (higher = more similar): crossover when graph_same > semantic_same
    For L2 distance (lower = more similar): crossover when graph_same < semantic_same
    """
    for i, x in enumerate(x_vals):
        if metric == "higher_better":
            if y2_vals[i] > y1_vals[i]:
                return x, i
        else:  # lower_better (L2 distance)
            if y2_vals[i] < y1_vals[i]:
                return x, i
    return None, None


def main():
    print("=" * 70)
    print("SEMANTIC CONFLICT EXPERIMENT (CORRECTED METRICS)")
    print("=" * 70)

    # Configuration
    model_name = "meta-llama/Llama-3.1-8B"
    # Extended context lengths
    context_lengths = (
        [5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000] +
        list(range(1500, 10001, 500))
    )
    n_samples = 50
    layer_idx = -5
    output_dir = Path("results/semantic_conflict_corrected")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\nLoading model: {model_name}...")
    hooked_model = HookedLLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = hooked_model.model
    tokenizer = hooked_model.tokenizer
    model.eval()

    # Create graph
    print("\nCreating semantic conflict graph...")
    graph = SemanticConflictGraph(seed=42)

    print("\nGraph structure:")
    print("  Semantic groups: animals, electronics, vegetables, furniture")
    print("  Graph clusters:")
    for cid, tokens in graph.graph_clusters.items():
        print(f"    Cluster {['A','B','C'][cid]}: {tokens}")

    print("\nSemantic pairs in DIFFERENT graph clusters:")
    for t1, t2, group in graph.get_semantic_pairs():
        c1 = ["A", "B", "C"][graph.token_to_graph_cluster[t1]]
        c2 = ["A", "B", "C"][graph.token_to_graph_cluster[t2]]
        print(f"    {t1} ({c1}) - {t2} ({c2}) [{group}]")

    # Collect representations
    print(f"\nCollecting representations at {len(context_lengths)} context lengths...")
    token_reps = collect_representations(model, tokenizer, graph, context_lengths, n_samples, layer_idx)

    # Analyze with corrected metrics
    print("\nAnalyzing semantic vs graph similarity...")
    results = analyze_semantic_vs_graph(token_reps, graph, context_lengths)

    # Find crossover points
    crossover_centered, idx_centered = find_crossover(
        context_lengths,
        results["centered_cos_semantic_same"],
        results["centered_cos_graph_same"],
        metric="higher_better"
    )

    crossover_l2, idx_l2 = find_crossover(
        context_lengths,
        results["l2_semantic_same"],
        results["l2_graph_same"],
        metric="lower_better"
    )

    print(f"\n--- CROSSOVER POINTS ---")
    print(f"Centered Cosine: N={crossover_centered}")
    print(f"L2 Distance: N={crossover_l2}")

    # Create visualizations
    print("\nCreating visualizations...")

    # Figure 1: Main comparison plot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Centered Cosine Similarity (higher = more similar)",
            "L2 Distance (lower = more similar)",
            "Raw Cosine Similarity (for reference - DON'T USE)",
            "Dominance Ratio"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # Panel 1: Centered cosine
    fig.add_trace(go.Scatter(
        x=context_lengths,
        y=results["centered_cos_semantic_same"],
        mode='lines+markers',
        name='Semantic-same (e.g., cat-dog)',
        line=dict(color='#d62728', width=3),
        marker=dict(size=8),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=context_lengths,
        y=results["centered_cos_graph_same"],
        mode='lines+markers',
        name='Graph-same (e.g., cat-computer)',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8),
    ), row=1, col=1)

    if crossover_centered:
        fig.add_vline(x=crossover_centered, line_dash="dash", line_color="green",
                      annotation_text=f"Crossover N={crossover_centered}", row=1, col=1)

    # Panel 2: L2 distance
    fig.add_trace(go.Scatter(
        x=context_lengths,
        y=results["l2_semantic_same"],
        mode='lines+markers',
        name='Semantic-same',
        line=dict(color='#d62728', width=3),
        marker=dict(size=8),
        showlegend=False,
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=context_lengths,
        y=results["l2_graph_same"],
        mode='lines+markers',
        name='Graph-same',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8),
        showlegend=False,
    ), row=1, col=2)

    if crossover_l2:
        fig.add_vline(x=crossover_l2, line_dash="dash", line_color="green",
                      annotation_text=f"Crossover N={crossover_l2}", row=1, col=2)

    # Panel 3: Raw cosine (for reference)
    fig.add_trace(go.Scatter(
        x=context_lengths,
        y=results["raw_cos_semantic_same"],
        mode='lines+markers',
        name='Semantic-same (raw)',
        line=dict(color='#d62728', width=2, dash='dot'),
        marker=dict(size=6),
        showlegend=False,
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=context_lengths,
        y=results["raw_cos_graph_same"],
        mode='lines+markers',
        name='Graph-same (raw)',
        line=dict(color='#1f77b4', width=2, dash='dot'),
        marker=dict(size=6),
        showlegend=False,
    ), row=2, col=1)

    # Panel 4: Dominance ratio
    # For centered cosine: graph/semantic (>1 = graph dominates)
    ratio_centered = [g/s if s != 0 else np.nan
                      for g, s in zip(results["centered_cos_graph_same"],
                                      results["centered_cos_semantic_same"])]

    # For L2: semantic/graph (>1 = graph dominates, since lower L2 = more similar)
    ratio_l2 = [s/g if g != 0 else np.nan
                for s, g in zip(results["l2_semantic_same"],
                                results["l2_graph_same"])]

    fig.add_trace(go.Scatter(
        x=context_lengths,
        y=ratio_l2,
        mode='lines+markers',
        name='L2 Ratio (semantic/graph)',
        line=dict(color='#9467bd', width=3),
        marker=dict(size=8),
    ), row=2, col=2)

    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=2, col=2)
    fig.add_annotation(x=np.log10(1000), y=1.0, text="Ratio=1 (equal)",
                       showarrow=False, yshift=15, row=2, col=2)

    fig.update_layout(
        title="Semantic Conflict: When Does Graph Override Pretrained Semantics?<br>" +
              "<sup>Using CORRECTED metrics (centered cosine, L2 distance)</sup>",
        width=1200,
        height=900,
        legend=dict(x=0.02, y=0.98),
    )

    fig.update_xaxes(title_text="Context Length (N)", type="log", row=1, col=1)
    fig.update_xaxes(title_text="Context Length (N)", type="log", row=1, col=2)
    fig.update_xaxes(title_text="Context Length (N)", type="log", row=2, col=1)
    fig.update_xaxes(title_text="Context Length (N)", type="log", row=2, col=2)

    fig.update_yaxes(title_text="Centered Cosine Sim", row=1, col=1)
    fig.update_yaxes(title_text="L2 Distance", row=1, col=2)
    fig.update_yaxes(title_text="Raw Cosine Sim", row=2, col=1)
    fig.update_yaxes(title_text="Ratio (>1 = graph wins)", row=2, col=2)

    fig.write_html(output_dir / "semantic_conflict_corrected.html")
    print(f"  Saved: {output_dir / 'semantic_conflict_corrected.html'}")

    # Save results
    results["crossover_centered_cos"] = crossover_centered
    results["crossover_l2"] = crossover_l2

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"  Saved: {output_dir / 'results.json'}")

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nCROSSOVER POINTS (where graph structure > pretrained semantics):")
    print(f"  Centered Cosine: N = {crossover_centered}")
    print(f"  L2 Distance: N = {crossover_l2}")

    print(f"\nAt N=5 (early):")
    print(f"  Semantic-same L2: {results['l2_semantic_same'][0]:.2f}")
    print(f"  Graph-same L2: {results['l2_graph_same'][0]:.2f}")
    print(f"  Ratio: {ratio_l2[0]:.3f} ({'Semantic' if ratio_l2[0] < 1 else 'Graph'} wins)")

    last_idx = -1
    print(f"\nAt N={context_lengths[last_idx]} (late):")
    print(f"  Semantic-same L2: {results['l2_semantic_same'][last_idx]:.2f}")
    print(f"  Graph-same L2: {results['l2_graph_same'][last_idx]:.2f}")
    print(f"  Ratio: {ratio_l2[last_idx]:.3f} ({'Semantic' if ratio_l2[last_idx] < 1 else 'Graph'} wins)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
