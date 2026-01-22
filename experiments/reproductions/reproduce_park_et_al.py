#!/usr/bin/env python3
"""
Reproduce Park et al. (2024) "ICLR: In-Context Learning of Representations"
arXiv:2501.00070

This script faithfully reproduces the key experiment from the paper:
1. Graph Tracing Task: Random walks on a grid graph
2. Dirichlet Energy: E(X) = sum_{i,j} A_{i,j} ||x_i - x_j||^2
3. Phase Transition: Track representation reorganization with context length

Key findings to reproduce:
- Representations start aligned with pretraining semantics
- As context increases, sudden reorganization to graph structure
- Dirichlet energy decreases as model learns graph topology
"""

import argparse
import json
import gc
from datetime import datetime
from pathlib import Path
from typing import Optional
import warnings

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tqdm import tqdm

# Local imports
from src.models import HookedLLM


# =============================================================================
# Graph Implementation (following Park et al.)
# =============================================================================

class GridGraph:
    """
    Simple grid graph as used in Park et al.

    Nodes are arranged in an m x m grid with edges between adjacent cells.
    Each node is assigned a concept word (vocabulary token).
    """

    # Default vocabulary from the paper (common nouns with minimal semantic overlap)
    VOCABULARY = [
        "apple", "truck", "sand", "river", "lamp", "chair", "stone", "cloud",
        "book", "glass", "bridge", "wheel", "needle", "mirror", "basket", "anchor",
        "forest", "mountain", "valley", "desert", "ocean", "island", "meadow", "cliff",
        "castle", "garden", "market", "harbor", "tunnel", "tower", "temple", "cave",
        "crystal", "feather", "copper", "silver", "marble", "canvas", "velvet", "bamboo",
    ]

    def __init__(self, grid_size: int = 4, seed: int = 42):
        """
        Args:
            grid_size: Side length of square grid (total nodes = grid_size^2)
            seed: Random seed for reproducibility
        """
        self.grid_size = grid_size
        self.num_nodes = grid_size * grid_size
        self.rng = np.random.default_rng(seed)

        if self.num_nodes > len(self.VOCABULARY):
            raise ValueError(f"Grid too large: {self.num_nodes} nodes but only {len(self.VOCABULARY)} words")

        # Build adjacency matrix
        self.adjacency_matrix = self._build_adjacency()

        # Assign vocabulary to nodes (random permutation)
        vocab_perm = self.rng.permutation(len(self.VOCABULARY))[:self.num_nodes]
        self.node_to_token = {i: self.VOCABULARY[vocab_perm[i]] for i in range(self.num_nodes)}
        self.token_to_node = {v: k for k, v in self.node_to_token.items()}

        # Compute graph Laplacian for Dirichlet energy
        self.degree_matrix = np.diag(self.adjacency_matrix.sum(axis=1))
        self.laplacian = self.degree_matrix - self.adjacency_matrix

    def _build_adjacency(self) -> np.ndarray:
        """Build adjacency matrix for grid graph."""
        adj = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)

        for i in range(self.num_nodes):
            row, col = i // self.grid_size, i % self.grid_size

            # Connect to neighbors (up, down, left, right)
            neighbors = []
            if row > 0:
                neighbors.append((row - 1) * self.grid_size + col)  # up
            if row < self.grid_size - 1:
                neighbors.append((row + 1) * self.grid_size + col)  # down
            if col > 0:
                neighbors.append(row * self.grid_size + (col - 1))  # left
            if col < self.grid_size - 1:
                neighbors.append(row * self.grid_size + (col + 1))  # right

            for j in neighbors:
                adj[i, j] = 1.0

        return adj

    def get_neighbors(self, node: int) -> list[int]:
        """Get neighbors of a node."""
        return list(np.where(self.adjacency_matrix[node] > 0)[0])

    def generate_random_walk(self, length: int, start_node: Optional[int] = None) -> tuple[str, list[int]]:
        """
        Generate a random walk on the grid.

        Args:
            length: Number of steps
            start_node: Starting node (random if None)

        Returns:
            (prompt_string, node_sequence)
        """
        if start_node is None:
            current = self.rng.integers(0, self.num_nodes)
        else:
            current = start_node

        nodes = [current]
        tokens = [self.node_to_token[current]]

        for _ in range(length - 1):
            neighbors = self.get_neighbors(current)
            if neighbors:
                current = self.rng.choice(neighbors)
            else:
                # Shouldn't happen in connected grid, but fallback
                current = self.rng.integers(0, self.num_nodes)

            nodes.append(current)
            tokens.append(self.node_to_token[current])

        return " ".join(tokens), nodes

    def visualize(self, save_path: Optional[str] = None):
        """Visualize the grid graph."""
        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot nodes
        for i in range(self.num_nodes):
            row, col = i // self.grid_size, i % self.grid_size
            ax.scatter(col, -row, s=500, c='steelblue', zorder=2)
            ax.text(col, -row, self.node_to_token[i], ha='center', va='center',
                   fontsize=8, color='white', fontweight='bold')

        # Plot edges
        for i in range(self.num_nodes):
            for j in self.get_neighbors(i):
                if i < j:
                    row_i, col_i = i // self.grid_size, i % self.grid_size
                    row_j, col_j = j // self.grid_size, j % self.grid_size
                    ax.plot([col_i, col_j], [-row_i, -row_j], 'gray', alpha=0.5, zorder=1)

        ax.set_title(f"Grid Graph ({self.grid_size}x{self.grid_size})")
        ax.axis('equal')
        ax.axis('off')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.close()
        return fig


class RingGraph:
    """
    Ring graph as also used in Park et al.

    Nodes arranged in a circle with edges to adjacent nodes.
    """

    VOCABULARY = [
        "apple", "truck", "sand", "river", "lamp", "chair", "stone", "cloud",
        "book", "glass", "bridge", "wheel", "needle", "mirror", "basket", "anchor",
    ]

    def __init__(self, num_nodes: int = 10, seed: int = 42):
        self.num_nodes = num_nodes
        self.rng = np.random.default_rng(seed)

        # Build circular adjacency
        self.adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for i in range(num_nodes):
            self.adjacency_matrix[i, (i + 1) % num_nodes] = 1.0
            self.adjacency_matrix[i, (i - 1) % num_nodes] = 1.0

        # Assign vocabulary
        vocab_perm = self.rng.permutation(len(self.VOCABULARY))[:num_nodes]
        self.node_to_token = {i: self.VOCABULARY[vocab_perm[i]] for i in range(num_nodes)}
        self.token_to_node = {v: k for k, v in self.node_to_token.items()}

        # Laplacian
        self.degree_matrix = np.diag(self.adjacency_matrix.sum(axis=1))
        self.laplacian = self.degree_matrix - self.adjacency_matrix

    def get_neighbors(self, node: int) -> list[int]:
        return [(node - 1) % self.num_nodes, (node + 1) % self.num_nodes]

    def generate_random_walk(self, length: int, start_node: Optional[int] = None) -> tuple[str, list[int]]:
        if start_node is None:
            current = self.rng.integers(0, self.num_nodes)
        else:
            current = start_node

        nodes = [current]
        tokens = [self.node_to_token[current]]

        for _ in range(length - 1):
            # Random step left or right
            direction = self.rng.choice([-1, 1])
            current = (current + direction) % self.num_nodes
            nodes.append(current)
            tokens.append(self.node_to_token[current])

        return " ".join(tokens), nodes


# =============================================================================
# Dirichlet Energy (exact formula from Park et al.)
# =============================================================================

def compute_dirichlet_energy(
    representations: torch.Tensor,
    adjacency_matrix: np.ndarray,
    node_sequence: list[int],
) -> float:
    """
    Compute Dirichlet Energy as defined in Park et al.

    E(X) = sum_{i,j} A_{i,j} ||x_i - x_j||^2

    We compute per-node mean representations, then energy over the full graph.

    Args:
        representations: (seq_len, hidden_dim) - token representations
        adjacency_matrix: (num_nodes, num_nodes) - graph adjacency
        node_sequence: list of node indices for each token

    Returns:
        Dirichlet energy (scalar)
    """
    num_nodes = adjacency_matrix.shape[0]

    # Compute mean representation per node (averaging over occurrences)
    node_reps = {}
    node_counts = {}

    for i, node in enumerate(node_sequence):
        if i >= len(representations):
            break
        if node not in node_reps:
            node_reps[node] = representations[i].clone()
            node_counts[node] = 1
        else:
            node_reps[node] += representations[i]
            node_counts[node] += 1

    # Average
    for node in node_reps:
        node_reps[node] /= node_counts[node]

    # Compute energy over all graph edges
    energy = 0.0
    edge_count = 0

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adjacency_matrix[i, j] > 0 and i in node_reps and j in node_reps:
                diff = node_reps[i] - node_reps[j]
                energy += torch.sum(diff ** 2).item()
                edge_count += 1

    # Normalize by number of edges to make comparable
    if edge_count > 0:
        energy /= edge_count

    return energy


def compute_dirichlet_energy_full(
    representations: torch.Tensor,
    adjacency_matrix: np.ndarray,
    node_to_rep_idx: dict,
) -> float:
    """
    Compute full Dirichlet Energy over all graph edges.

    E(X) = sum_{i,j} A_{i,j} ||x_i - x_j||^2

    This is the full formula from the paper (Equation 2).

    Args:
        representations: Dict or tensor mapping nodes to representations
        adjacency_matrix: (num_nodes, num_nodes)
        node_to_rep_idx: Mapping from node index to representation index

    Returns:
        Normalized Dirichlet energy
    """
    num_nodes = adjacency_matrix.shape[0]

    # Compute mean representation per node
    node_reps = {}
    for node, rep_idx in node_to_rep_idx.items():
        if isinstance(rep_idx, list):
            # Average over multiple occurrences
            node_reps[node] = torch.stack([representations[i] for i in rep_idx]).mean(dim=0)
        else:
            node_reps[node] = representations[rep_idx]

    # Compute energy over all edges
    energy = 0.0
    edge_count = 0

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adjacency_matrix[i, j] > 0:
                if i in node_reps and j in node_reps:
                    diff = node_reps[i] - node_reps[j]
                    energy += torch.sum(diff ** 2).item()
                    edge_count += 1

    # Normalize by number of edges
    if edge_count > 0:
        energy /= edge_count

    return energy


def compute_normalized_energy(
    representations: torch.Tensor,
    node_sequence: list[int],
    num_nodes: int,
) -> float:
    """
    Compute Dirichlet energy normalized by representation variance.

    This makes the metric comparable across different context lengths.
    """
    seq_len = representations.shape[0]

    # Sequential energy
    energy = 0.0
    for i in range(seq_len - 1):
        diff = representations[i] - representations[i + 1]
        energy += torch.sum(diff ** 2).item()

    # Normalize by total variance
    total_var = representations.var(dim=0).sum().item()
    if total_var > 1e-10:
        energy /= total_var

    return energy


# =============================================================================
# Experiment Runner
# =============================================================================

def run_reproduction_experiment(
    model_id: str = "meta-llama/Llama-3.1-8B",
    graph_type: str = "grid",
    grid_size: int = 4,
    context_lengths: list[int] = None,
    n_samples_per_length: int = 50,
    layer: int = 26,  # Deep layer as in paper
    output_dir: str = "results/park_reproduction",
    seed: int = 42,
):
    """
    Run the Park et al. reproduction experiment.

    Args:
        model_id: HuggingFace model ID
        graph_type: "grid" or "ring"
        grid_size: Size of grid (for grid graph)
        context_lengths: List of context lengths to test
        n_samples_per_length: Number of random walks per context length
        layer: Which layer to extract representations from
        output_dir: Where to save results
        seed: Random seed
    """
    if context_lengths is None:
        # Default: test phase transition region
        context_lengths = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print("REPRODUCING Park et al. (2024) ICLR")
    print("In-Context Learning of Representations")
    print("=" * 70)
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    print(f"Model: {model_id}")
    print(f"Graph: {graph_type} (size {grid_size}x{grid_size})")
    print(f"Context lengths: {context_lengths}")
    print(f"Samples per length: {n_samples_per_length}")
    print(f"Layer: {layer}")

    # Create graph
    print("\n" + "-" * 70)
    print("Creating Graph")
    print("-" * 70)

    if graph_type == "grid":
        graph = GridGraph(grid_size=grid_size, seed=seed)
    else:
        graph = RingGraph(num_nodes=grid_size * grid_size, seed=seed)

    print(f"  Nodes: {graph.num_nodes}")
    print(f"  Vocabulary: {list(graph.node_to_token.values())[:5]}...")

    # Save graph visualization
    graph.visualize(save_path=str(output_path / "graph_structure.png"))

    # Load model
    print("\n" + "-" * 70)
    print("Loading Model")
    print("-" * 70)

    model = HookedLLM.from_pretrained(
        model_id,
        device="auto",
        dtype=torch.bfloat16,
    )
    print(f"  Loaded: {model_id}")
    print(f"  Layers: {model.num_layers}")
    print(f"  Hidden size: {model.hidden_size}")

    # Adjust layer if needed
    if layer >= model.num_layers:
        layer = model.num_layers - 1
        print(f"  Adjusted layer to: {layer}")

    # Run experiment
    print("\n" + "-" * 70)
    print("Running Experiment")
    print("-" * 70)

    results = {
        "model": model_id,
        "graph_type": graph_type,
        "graph_size": grid_size,
        "num_nodes": graph.num_nodes,
        "layer": layer,
        "n_samples": n_samples_per_length,
        "context_results": {},
    }

    all_energies = {ctx_len: [] for ctx_len in context_lengths}
    all_normalized_energies = {ctx_len: [] for ctx_len in context_lengths}

    for ctx_len in tqdm(context_lengths, desc="Context lengths"):
        ctx_energies = []
        ctx_norm_energies = []

        for _ in range(n_samples_per_length):
            # Generate random walk
            prompt, nodes = graph.generate_random_walk(length=ctx_len)

            try:
                # Get representations
                _, cache = model.forward_with_cache(prompt, layers=[layer])
                residual = cache.get_residual_stream(layer)

                if residual is None:
                    continue

                # Remove batch dimension
                reps = residual.squeeze(0).cpu()

                # Compute Dirichlet energy (over consecutive tokens)
                energy = compute_dirichlet_energy(
                    reps,
                    graph.adjacency_matrix,
                    nodes
                )
                ctx_energies.append(energy)

                # Normalized version
                norm_energy = compute_normalized_energy(
                    reps,
                    nodes,
                    graph.num_nodes
                )
                ctx_norm_energies.append(norm_energy)

            except Exception as e:
                warnings.warn(f"Error at ctx_len={ctx_len}: {e}")
                continue

        all_energies[ctx_len] = ctx_energies
        all_normalized_energies[ctx_len] = ctx_norm_energies

        # Store results
        results["context_results"][str(ctx_len)] = {
            "energy_mean": float(np.mean(ctx_energies)) if ctx_energies else 0,
            "energy_std": float(np.std(ctx_energies)) if ctx_energies else 0,
            "energy_median": float(np.median(ctx_energies)) if ctx_energies else 0,
            "normalized_energy_mean": float(np.mean(ctx_norm_energies)) if ctx_norm_energies else 0,
            "normalized_energy_std": float(np.std(ctx_norm_energies)) if ctx_norm_energies else 0,
            "n_valid_samples": len(ctx_energies),
        }

        print(f"  N={ctx_len}: Energy={np.mean(ctx_energies):.2f} +/- {np.std(ctx_energies):.2f}")

    # Cleanup model
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save raw results
    results_path = output_path / "reproduction_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Generate visualizations
    print("\n" + "-" * 70)
    print("Generating Visualizations")
    print("-" * 70)

    generate_reproduction_plots(results, output_path)

    print("\n" + "=" * 70)
    print("REPRODUCTION COMPLETE")
    print("=" * 70)

    return results


def generate_reproduction_plots(results: dict, output_path: Path):
    """Generate plots matching Park et al. figures."""

    ctx_results = results["context_results"]
    context_lengths = sorted([int(k) for k in ctx_results.keys()])

    energies = [ctx_results[str(c)]["energy_mean"] for c in context_lengths]
    energy_stds = [ctx_results[str(c)]["energy_std"] for c in context_lengths]
    norm_energies = [ctx_results[str(c)]["normalized_energy_mean"] for c in context_lengths]
    norm_stds = [ctx_results[str(c)]["normalized_energy_std"] for c in context_lengths]

    # Figure 1: Main phase transition plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Raw Dirichlet Energy
    ax1 = axes[0]
    ax1.errorbar(context_lengths, energies, yerr=energy_stds,
                 fmt='o-', capsize=3, color='steelblue', linewidth=2)
    ax1.set_xlabel("Context Length (N)", fontsize=12)
    ax1.set_ylabel("Dirichlet Energy E(X)", fontsize=12)
    ax1.set_title("A. Raw Dirichlet Energy vs Context Length", fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Normalized Energy
    ax2 = axes[1]
    ax2.errorbar(context_lengths, norm_energies, yerr=norm_stds,
                 fmt='o-', capsize=3, color='darkorange', linewidth=2)
    ax2.set_xlabel("Context Length (N)", fontsize=12)
    ax2.set_ylabel("Normalized Energy E(X)/Var(X)", fontsize=12)
    ax2.set_title("B. Normalized Dirichlet Energy", fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Rate of change (looking for phase transition)
    ax3 = axes[2]
    if len(context_lengths) > 1:
        # Compute discrete derivative
        energy_changes = []
        for i in range(1, len(context_lengths)):
            delta_e = energies[i] - energies[i-1]
            delta_n = context_lengths[i] - context_lengths[i-1]
            rate = delta_e / delta_n if delta_n > 0 else 0
            energy_changes.append((context_lengths[i], rate))

        if energy_changes:
            change_x, change_y = zip(*energy_changes)
            ax3.bar(change_x, change_y, width=5, color='seagreen', alpha=0.7)
            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.3)

    ax3.set_xlabel("Context Length (N)", fontsize=12)
    ax3.set_ylabel("dE/dN (Energy Rate of Change)", fontsize=12)
    ax3.set_title("C. Rate of Change (Phase Transition Detection)", fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    fig.suptitle(f"Park et al. Reproduction: {results['model']} on {results['graph_type']} graph",
                 fontsize=14, fontweight='bold', y=1.02)

    plt.savefig(output_path / "phase_transition_reproduction.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_path / "phase_transition_reproduction.pdf", bbox_inches='tight')
    print(f"  Saved: phase_transition_reproduction.png")

    plt.close()

    # Figure 2: Energy per token (to match paper's Figure 4)
    fig2, ax = plt.subplots(figsize=(10, 6))

    # Plot energy per token
    energy_per_token = [e / c if c > 0 else 0 for e, c in zip(energies, context_lengths)]
    ax.plot(context_lengths, energy_per_token, 'o-', color='purple', linewidth=2, markersize=8)
    ax.set_xlabel("Context Length (N)", fontsize=12)
    ax.set_ylabel("Dirichlet Energy per Token", fontsize=12)
    ax.set_title("Dirichlet Energy per Token vs Context Length", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "energy_per_token.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: energy_per_token.png")

    plt.close()

    # Print summary statistics
    print("\n" + "=" * 70)
    print("REPRODUCTION SUMMARY")
    print("=" * 70)
    print(f"\nDirichlet Energy across context lengths:")
    print(f"{'N':>6} | {'Energy':>12} | {'Normalized':>12}")
    print("-" * 36)
    for c in context_lengths:
        e = ctx_results[str(c)]["energy_mean"]
        n = ctx_results[str(c)]["normalized_energy_mean"]
        print(f"{c:>6} | {e:>12.2f} | {n:>12.4f}")

    # Find potential phase transition
    if len(context_lengths) > 2:
        energy_diffs = np.diff(energies)
        max_change_idx = np.argmax(np.abs(energy_diffs))
        print(f"\nLargest energy change between N={context_lengths[max_change_idx]} and N={context_lengths[max_change_idx+1]}")
        print(f"  Change magnitude: {energy_diffs[max_change_idx]:.2f}")


# =============================================================================
# Layer Analysis (matching paper's layer-wise analysis)
# =============================================================================

def run_layer_analysis(
    model_id: str = "meta-llama/Llama-3.1-8B",
    context_length: int = 100,
    n_samples: int = 50,
    output_dir: str = "results/park_reproduction",
    seed: int = 42,
):
    """
    Analyze Dirichlet energy across layers (like Park et al. Appendix).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    print("\n" + "=" * 70)
    print("LAYER-WISE ANALYSIS")
    print("=" * 70)

    # Create graph
    graph = GridGraph(grid_size=4, seed=seed)

    # Load model
    model = HookedLLM.from_pretrained(
        model_id,
        device="auto",
        dtype=torch.bfloat16,
    )

    # Analyze all layers
    all_layers = list(range(model.num_layers))
    layer_energies = {layer: [] for layer in all_layers}

    print(f"Analyzing {len(all_layers)} layers with context_length={context_length}")

    for _ in tqdm(range(n_samples), desc="Samples"):
        prompt, nodes = graph.generate_random_walk(length=context_length)

        try:
            _, cache = model.forward_with_cache(prompt, layers=all_layers)

            for layer in all_layers:
                residual = cache.get_residual_stream(layer)
                if residual is not None:
                    reps = residual.squeeze(0).cpu()
                    energy = compute_dirichlet_energy(reps, graph.adjacency_matrix, nodes)
                    layer_energies[layer].append(energy)
        except Exception as e:
            warnings.warn(f"Error: {e}")
            continue

    # Plot layer-wise energy
    fig, ax = plt.subplots(figsize=(12, 6))

    layers = sorted(layer_energies.keys())
    means = [np.mean(layer_energies[l]) for l in layers]
    stds = [np.std(layer_energies[l]) for l in layers]

    ax.errorbar(layers, means, yerr=stds, fmt='o-', capsize=3, color='steelblue', linewidth=2)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Dirichlet Energy", fontsize=12)
    ax.set_title(f"Dirichlet Energy Across Layers (N={context_length})", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "layer_analysis.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved: layer_analysis.png")

    # Cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return layer_energies


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Reproduce Park et al. (2024)")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B",
                       help="Model ID")
    parser.add_argument("--graph", type=str, default="grid", choices=["grid", "ring"],
                       help="Graph type")
    parser.add_argument("--grid-size", type=int, default=4,
                       help="Grid size (4x4 = 16 nodes)")
    parser.add_argument("--context-lengths", type=str, default="5,10,15,20,25,30,40,50,75,100",
                       help="Comma-separated context lengths")
    parser.add_argument("--n-samples", type=int, default=50,
                       help="Samples per context length")
    parser.add_argument("--layer", type=int, default=26,
                       help="Layer to analyze")
    parser.add_argument("--output-dir", type=str, default="results/park_reproduction",
                       help="Output directory")
    parser.add_argument("--layer-analysis", action="store_true",
                       help="Run layer-wise analysis")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    context_lengths = [int(x) for x in args.context_lengths.split(",")]

    if args.layer_analysis:
        run_layer_analysis(
            model_id=args.model,
            context_length=max(context_lengths),
            n_samples=args.n_samples,
            output_dir=args.output_dir,
            seed=args.seed,
        )
    else:
        run_reproduction_experiment(
            model_id=args.model,
            graph_type=args.graph,
            grid_size=args.grid_size,
            context_lengths=context_lengths,
            n_samples_per_length=args.n_samples,
            layer=args.layer,
            output_dir=args.output_dir,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
