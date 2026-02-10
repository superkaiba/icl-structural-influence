"""
Dual Interpretation Graph for Hypothesis Superposition Experiments.

This module implements a graph structure where the SAME tokens can be
interpreted under TWO different valid clusterings (H1 and H2). This enables
experiments on whether LLMs maintain multiple hypotheses in superposition
before a disambiguating token forces collapse to one interpretation.

Core concept:
- Same vocabulary of arbitrary tokens
- Two different Stochastic Block Model graphs (G1, G2) over these tokens
- Sequences are constructed to be ambiguous (valid under both) until a
  disambiguating token that only makes sense under one interpretation

References:
- Park et al. (2024) arXiv:2501.00070 "ICLR: In-Context Learning of Representations"
- Inspired by linguistic garden-path sentences and syntactic ambiguity
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import random
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

from .hierarchical_graph import DEFAULT_VOCABULARY


@dataclass
class DualInterpretationConfig:
    """Configuration for dual interpretation graph generation."""

    # Vocabulary (shared between both interpretations)
    vocab_size: int = 15
    clusters_per_interpretation: int = 3

    # Edge probabilities for both graphs
    p_intra_cluster: float = 0.8   # Probability of edge within same cluster
    p_inter_cluster: float = 0.15  # Probability of edge between clusters

    # Vocabulary
    vocabulary: list[str] = field(default_factory=lambda: DEFAULT_VOCABULARY.copy())

    # Reproducibility
    seed: Optional[int] = None

    def __post_init__(self):
        if len(self.vocabulary) < self.vocab_size:
            raise ValueError(
                f"Vocabulary size ({len(self.vocabulary)}) must be >= "
                f"vocab_size ({self.vocab_size})"
            )
        if self.vocab_size % self.clusters_per_interpretation != 0:
            raise ValueError(
                f"vocab_size ({self.vocab_size}) must be divisible by "
                f"clusters_per_interpretation ({self.clusters_per_interpretation})"
            )


class DualInterpretationGraph:
    """
    Graph where same tokens have two valid cluster interpretations.

    This enables testing whether models maintain multiple hypotheses
    in superposition during ambiguous contexts.

    Attributes:
        config: DualInterpretationConfig with graph parameters
        tokens: List of token strings
        H1_clusters: Dict mapping token_idx -> cluster_id under H1
        H2_clusters: Dict mapping token_idx -> cluster_id under H2
        G1_adj: Adjacency matrix for interpretation H1
        G2_adj: Adjacency matrix for interpretation H2
    """

    def __init__(self, config: Optional[DualInterpretationConfig] = None):
        self.config = config or DualInterpretationConfig()
        self.rng = np.random.default_rng(self.config.seed)

        self.num_tokens = self.config.vocab_size
        self.num_clusters = self.config.clusters_per_interpretation
        self.nodes_per_cluster = self.num_tokens // self.num_clusters

        # Assign tokens
        self._assign_tokens()

        # Build two different clusterings
        self._build_dual_clusterings()

        # Build adjacency matrices for each interpretation
        self._build_adjacency_matrices()

    def _assign_tokens(self):
        """Assign vocabulary tokens to node indices."""
        shuffled_vocab = self.config.vocabulary[:self.num_tokens].copy()
        self.rng.shuffle(shuffled_vocab)

        self.tokens = shuffled_vocab
        self.token_to_idx = {t: i for i, t in enumerate(self.tokens)}
        self.idx_to_token = {i: t for i, t in enumerate(self.tokens)}

    def _build_dual_clusterings(self):
        """
        Create two different valid clusterings of the same tokens.

        H1: Sequential assignment (tokens 0-4 -> cluster 0, etc.)
        H2: Interleaved assignment designed to maximize disagreement with H1
        """
        # H1: Sequential clustering
        self.H1_clusters = {}
        for i in range(self.num_tokens):
            self.H1_clusters[i] = i // self.nodes_per_cluster

        # H2: Maximally different clustering
        # Interleave so tokens that are same-cluster in H1 are different in H2
        self.H2_clusters = {}
        indices = list(range(self.num_tokens))
        self.rng.shuffle(indices)  # Shuffle to randomize H2

        # Assign to clusters in a way that maximizes disagreement
        for new_idx, old_idx in enumerate(indices):
            self.H2_clusters[old_idx] = new_idx // self.nodes_per_cluster

        # Verify we have disagreement
        self._verify_clustering_difference()

    def _verify_clustering_difference(self):
        """Verify that H1 and H2 have meaningful differences."""
        agreements = 0
        total_pairs = 0

        for i in range(self.num_tokens):
            for j in range(i + 1, self.num_tokens):
                h1_same = (self.H1_clusters[i] == self.H1_clusters[j])
                h2_same = (self.H2_clusters[i] == self.H2_clusters[j])
                if h1_same == h2_same:
                    agreements += 1
                total_pairs += 1

        agreement_rate = agreements / total_pairs
        if agreement_rate > 0.8:
            # Re-shuffle H2 if too similar
            self._build_dual_clusterings()

    def _build_adjacency_matrices(self):
        """Build separate adjacency matrices for G1 (H1) and G2 (H2)."""
        n = self.num_tokens

        # Build G1 based on H1 clustering
        self.G1_adj = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                if self.H1_clusters[i] == self.H1_clusters[j]:
                    p = self.config.p_intra_cluster
                else:
                    p = self.config.p_inter_cluster

                if self.rng.random() < p:
                    self.G1_adj[i, j] = 1.0
                    self.G1_adj[j, i] = 1.0

        # Build G2 based on H2 clustering
        self.G2_adj = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                if self.H2_clusters[i] == self.H2_clusters[j]:
                    p = self.config.p_intra_cluster
                else:
                    p = self.config.p_inter_cluster

                if self.rng.random() < p:
                    self.G2_adj[i, j] = 1.0
                    self.G2_adj[j, i] = 1.0

        # Ensure both graphs are connected
        self._ensure_connectivity(self.G1_adj)
        self._ensure_connectivity(self.G2_adj)

    def _ensure_connectivity(self, adj_matrix: np.ndarray):
        """Add minimal edges to ensure graph is connected."""
        sparse_adj = csr_matrix(adj_matrix)
        n_components, labels = connected_components(sparse_adj, directed=False)

        while n_components > 1:
            for comp in range(1, n_components):
                nodes_comp0 = np.where(labels == 0)[0]
                nodes_comp = np.where(labels == comp)[0]

                i = self.rng.choice(nodes_comp0)
                j = self.rng.choice(nodes_comp)
                adj_matrix[i, j] = 1.0
                adj_matrix[j, i] = 1.0

            sparse_adj = csr_matrix(adj_matrix)
            n_components, labels = connected_components(sparse_adj, directed=False)

    def get_neighbors(self, node: int, hypothesis: str) -> list[int]:
        """Get neighbors of a node under a specific hypothesis."""
        adj = self.G1_adj if hypothesis == "H1" else self.G2_adj
        return list(np.where(adj[node] > 0)[0])

    def get_cluster(self, node: int, hypothesis: str) -> int:
        """Get cluster assignment for a node under a specific hypothesis."""
        return self.H1_clusters[node] if hypothesis == "H1" else self.H2_clusters[node]

    def is_valid_transition(self, from_node: int, to_node: int, hypothesis: str) -> bool:
        """Check if a transition is valid under a specific hypothesis."""
        adj = self.G1_adj if hypothesis == "H1" else self.G2_adj
        return adj[from_node, to_node] > 0

    def find_ambiguous_tokens(self, from_node: int) -> list[int]:
        """
        Find tokens that are valid transitions from from_node under BOTH hypotheses.

        These are the tokens that maintain ambiguity.
        """
        h1_neighbors = set(self.get_neighbors(from_node, "H1"))
        h2_neighbors = set(self.get_neighbors(from_node, "H2"))
        return list(h1_neighbors & h2_neighbors)

    def find_disambiguating_tokens(self, from_node: int, true_hypothesis: str) -> list[int]:
        """
        Find tokens that are ONLY valid under the true hypothesis.

        These are tokens that reveal which interpretation is correct.
        """
        h1_neighbors = set(self.get_neighbors(from_node, "H1"))
        h2_neighbors = set(self.get_neighbors(from_node, "H2"))

        if true_hypothesis == "H1":
            # Valid in H1 but NOT in H2
            return list(h1_neighbors - h2_neighbors)
        else:
            # Valid in H2 but NOT in H1
            return list(h2_neighbors - h1_neighbors)

    def generate_h1_only_walk(
        self,
        length: int,
        return_nodes: bool = False,
    ) -> str | tuple[str, list[int], dict]:
        """
        Generate a walk that follows ONLY H1-valid transitions (no ambiguity).

        This creates sequences that are unambiguously consistent with H1 from
        the start, serving as a baseline for collapse experiments where there's
        no competing hypothesis.

        Args:
            length: Total walk length
            return_nodes: If True, also return node indices and metadata

        Returns:
            If return_nodes=False: Prompt string
            If return_nodes=True: (prompt, nodes, metadata)
        """
        # Start from a random node
        current = self.rng.integers(0, self.num_tokens)
        walk_nodes = [current]

        for pos in range(1, length):
            # Only use H1 neighbors
            h1_neighbors = self.get_neighbors(current, "H1")

            if h1_neighbors:
                current = self.rng.choice(h1_neighbors)
            else:
                # No H1 neighbors - pick a random node in the same H1 cluster
                current_cluster = self.H1_clusters[current]
                same_cluster_nodes = [
                    n for n in range(self.num_tokens)
                    if self.H1_clusters[n] == current_cluster and n != current
                ]
                if same_cluster_nodes:
                    current = self.rng.choice(same_cluster_nodes)
                else:
                    # Fall back to any node (rare edge case)
                    current = self.rng.integers(0, self.num_tokens)

            walk_nodes.append(current)

        # Convert to tokens
        walk_tokens = [self.idx_to_token[n] for n in walk_nodes]
        prompt = " ".join(walk_tokens)

        if return_nodes:
            metadata = {
                "true_hypothesis": "H1",
                "disambig_position": 0,  # Disambiguated from start
                "disambig_achieved": True,
                "H1_clusters": [self.H1_clusters[n] for n in walk_nodes],
                "H2_clusters": [self.H2_clusters[n] for n in walk_nodes],
                "walk_type": "h1_only",
            }
            return prompt, walk_nodes, metadata
        return prompt

    def generate_h2_only_walk(
        self,
        length: int,
        return_nodes: bool = False,
    ) -> str | tuple[str, list[int], dict]:
        """
        Generate a walk that follows ONLY H2-valid transitions (no ambiguity).

        This creates sequences that are unambiguously consistent with H2 from
        the start. Used in collapse reversal experiments to inject contradicting
        structure after H1-only context has induced collapse.

        Args:
            length: Total walk length
            return_nodes: If True, also return node indices and metadata

        Returns:
            If return_nodes=False: Prompt string
            If return_nodes=True: (prompt, nodes, metadata)
        """
        # Start from a random node
        current = self.rng.integers(0, self.num_tokens)
        walk_nodes = [current]

        for pos in range(1, length):
            # Only use H2 neighbors
            h2_neighbors = self.get_neighbors(current, "H2")

            if h2_neighbors:
                current = self.rng.choice(h2_neighbors)
            else:
                # No H2 neighbors - pick a random node in the same H2 cluster
                current_cluster = self.H2_clusters[current]
                same_cluster_nodes = [
                    n for n in range(self.num_tokens)
                    if self.H2_clusters[n] == current_cluster and n != current
                ]
                if same_cluster_nodes:
                    current = self.rng.choice(same_cluster_nodes)
                else:
                    # Fall back to any node (rare edge case)
                    current = self.rng.integers(0, self.num_tokens)

            walk_nodes.append(current)

        # Convert to tokens
        walk_tokens = [self.idx_to_token[n] for n in walk_nodes]
        prompt = " ".join(walk_tokens)

        if return_nodes:
            metadata = {
                "true_hypothesis": "H2",
                "disambig_position": 0,  # Disambiguated from start
                "disambig_achieved": True,
                "H1_clusters": [self.H1_clusters[n] for n in walk_nodes],
                "H2_clusters": [self.H2_clusters[n] for n in walk_nodes],
                "walk_type": "h2_only",
            }
            return prompt, walk_nodes, metadata
        return prompt

    def generate_ambiguous_walk(
        self,
        length: int,
        disambig_position: Optional[int],
        true_hypothesis: str,
        return_nodes: bool = False,
    ) -> str | tuple[str, list[int], dict]:
        """
        Generate a walk that's ambiguous until disambig_position.

        Args:
            length: Total walk length
            disambig_position: Position where disambiguation occurs (None = never)
            true_hypothesis: "H1" or "H2" - the true interpretation
            return_nodes: If True, also return node indices and metadata

        Returns:
            If return_nodes=False: Prompt string
            If return_nodes=True: (prompt, nodes, metadata)
        """
        if true_hypothesis not in ["H1", "H2"]:
            raise ValueError("true_hypothesis must be 'H1' or 'H2'")

        # Start from a random node
        current = self.rng.integers(0, self.num_tokens)
        walk_nodes = [current]

        disambig_achieved = False

        for pos in range(1, length):
            if disambig_position is not None and pos == disambig_position and not disambig_achieved:
                # Try to disambiguate
                disambig_tokens = self.find_disambiguating_tokens(current, true_hypothesis)

                if disambig_tokens:
                    current = self.rng.choice(disambig_tokens)
                    disambig_achieved = True
                else:
                    # No disambiguating token available, use ambiguous
                    ambig_tokens = self.find_ambiguous_tokens(current)
                    if ambig_tokens:
                        current = self.rng.choice(ambig_tokens)
                    else:
                        # Fall back to any neighbor under true hypothesis
                        neighbors = self.get_neighbors(current, true_hypothesis)
                        if neighbors:
                            current = self.rng.choice(neighbors)
                        else:
                            current = self.rng.integers(0, self.num_tokens)
            elif disambig_position is None or pos < disambig_position:
                # Ambiguous region: choose tokens valid under BOTH hypotheses
                ambig_tokens = self.find_ambiguous_tokens(current)

                if ambig_tokens:
                    current = self.rng.choice(ambig_tokens)
                else:
                    # No ambiguous tokens, use true hypothesis neighbors
                    neighbors = self.get_neighbors(current, true_hypothesis)
                    if neighbors:
                        current = self.rng.choice(neighbors)
                    else:
                        current = self.rng.integers(0, self.num_tokens)
            else:
                # After disambiguation: follow true hypothesis
                neighbors = self.get_neighbors(current, true_hypothesis)
                if neighbors:
                    current = self.rng.choice(neighbors)
                else:
                    current = self.rng.integers(0, self.num_tokens)

            walk_nodes.append(current)

        # Convert to tokens
        walk_tokens = [self.idx_to_token[n] for n in walk_nodes]
        prompt = " ".join(walk_tokens)

        if return_nodes:
            metadata = {
                "true_hypothesis": true_hypothesis,
                "disambig_position": disambig_position,
                "disambig_achieved": disambig_achieved,
                "H1_clusters": [self.H1_clusters[n] for n in walk_nodes],
                "H2_clusters": [self.H2_clusters[n] for n in walk_nodes],
            }
            return prompt, walk_nodes, metadata
        return prompt

    def generate_batch(
        self,
        batch_size: int,
        length: int,
        disambig_position: Optional[int],
    ) -> list[dict]:
        """
        Generate a batch of walks with balanced H1/H2 true hypotheses.

        Returns:
            List of dicts with walk data and metadata
        """
        batch = []

        for i in range(batch_size):
            # Alternate true hypothesis
            true_hyp = "H1" if i % 2 == 0 else "H2"

            prompt, nodes, metadata = self.generate_ambiguous_walk(
                length=length,
                disambig_position=disambig_position,
                true_hypothesis=true_hyp,
                return_nodes=True,
            )

            batch.append({
                "prompt": prompt,
                "nodes": nodes,
                **metadata,
            })

        return batch

    def compute_interpretation_consistency(self, nodes: list[int], hypothesis: str) -> float:
        """
        Compute what fraction of transitions are valid under a hypothesis.

        Returns value in [0, 1]. Higher = more consistent with hypothesis.
        """
        if len(nodes) < 2:
            return 1.0

        valid = 0
        total = len(nodes) - 1

        for i in range(total):
            if self.is_valid_transition(nodes[i], nodes[i+1], hypothesis):
                valid += 1

        return valid / total

    def get_graph_statistics(self) -> dict:
        """Return statistics about both graph structures."""
        # Count edges
        g1_edges = int(self.G1_adj.sum() / 2)
        g2_edges = int(self.G2_adj.sum() / 2)

        # Count clustering agreement
        agreements = 0
        for i in range(self.num_tokens):
            for j in range(i + 1, self.num_tokens):
                h1_same = (self.H1_clusters[i] == self.H1_clusters[j])
                h2_same = (self.H2_clusters[i] == self.H2_clusters[j])
                if h1_same == h2_same:
                    agreements += 1
        total_pairs = self.num_tokens * (self.num_tokens - 1) // 2

        # Count ambiguous edges (in both graphs)
        ambiguous_edges = int((self.G1_adj * self.G2_adj).sum() / 2)

        return {
            "num_tokens": self.num_tokens,
            "num_clusters": self.num_clusters,
            "g1_edges": g1_edges,
            "g2_edges": g2_edges,
            "ambiguous_edges": ambiguous_edges,
            "clustering_agreement_rate": agreements / total_pairs,
            "g1_mean_degree": float(self.G1_adj.sum(axis=1).mean()),
            "g2_mean_degree": float(self.G2_adj.sum(axis=1).mean()),
        }

    def visualize(
        self,
        save_path: Optional[str] = None,
        hypothesis: str = "both"
    ):
        """
        Visualize the graph structure(s).

        Args:
            save_path: If provided, save figure to this path
            hypothesis: "H1", "H2", or "both"
        """
        import matplotlib.pyplot as plt
        import networkx as nx

        if hypothesis == "both":
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))

            for idx, (hyp, adj) in enumerate([("H1", self.G1_adj), ("H2", self.G2_adj)]):
                ax = axes[idx]
                G = nx.from_numpy_array(adj)

                # Color by cluster
                clusters = self.H1_clusters if hyp == "H1" else self.H2_clusters
                colors = plt.cm.tab10(np.array([clusters[i] for i in range(self.num_tokens)]) / self.num_clusters)

                pos = nx.spring_layout(G, seed=42, k=2)

                nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=500, ax=ax)
                nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5, ax=ax)

                labels = {i: self.idx_to_token[i] for i in range(self.num_tokens)}
                nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)

                ax.set_title(f"Interpretation {hyp}\n({self.num_clusters} clusters)")
                ax.axis('off')

        else:
            fig, ax = plt.subplots(figsize=(10, 8))
            adj = self.G1_adj if hypothesis == "H1" else self.G2_adj
            G = nx.from_numpy_array(adj)

            clusters = self.H1_clusters if hypothesis == "H1" else self.H2_clusters
            colors = plt.cm.tab10(np.array([clusters[i] for i in range(self.num_tokens)]) / self.num_clusters)

            pos = nx.spring_layout(G, seed=42, k=2)

            nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=500, ax=ax)
            nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5, ax=ax)

            labels = {i: self.idx_to_token[i] for i in range(self.num_tokens)}
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)

            ax.set_title(f"Interpretation {hypothesis}")
            ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig


def generate_extended_vocabulary(size: int, seed: Optional[int] = None) -> list[str]:
    """
    Generate an extended vocabulary for large vocab experiments.

    Creates unique tokens by combining base words with numeric suffixes
    when the vocabulary size exceeds the default vocabulary.

    Args:
        size: Desired vocabulary size
        seed: Random seed for reproducibility

    Returns:
        List of unique token strings
    """
    rng = np.random.default_rng(seed)

    # Start with default vocabulary
    vocab = DEFAULT_VOCABULARY.copy()

    # If we need more tokens, generate them
    if size > len(vocab):
        # Use additional base words
        additional_bases = [
            "oak", "pine", "elm", "birch", "maple",
            "lion", "tiger", "bear", "wolf", "fox",
            "red", "blue", "green", "gold", "white",
            "north", "south", "east", "west", "center",
            "alpha", "beta", "gamma", "delta", "sigma",
            "spark", "flame", "frost", "mist", "storm",
            "swift", "bold", "calm", "keen", "true",
            "dawn", "dusk", "noon", "night", "day",
        ]
        vocab.extend(additional_bases)

        # Add numbered variants if still need more
        base_count = len(vocab)
        suffix_num = 1
        while len(vocab) < size:
            for base in vocab[:base_count]:
                if len(vocab) >= size:
                    break
                vocab.append(f"{base}{suffix_num}")
            suffix_num += 1

    # Shuffle and return requested size
    rng.shuffle(vocab)
    return vocab[:size]


def create_graph_with_vocab_size(
    vocab_size: int,
    seed: Optional[int] = None,
    p_intra_cluster: float = 0.8,
    p_inter_cluster: float = 0.15,
) -> "DualInterpretationGraph":
    """
    Create a DualInterpretationGraph with a specific vocabulary size.

    This is a convenience function for collapse experiments that vary
    vocabulary size to study how token diversity affects collapse.

    Args:
        vocab_size: Number of unique tokens (must be divisible by 3)
        seed: Random seed for reproducibility
        p_intra_cluster: Intra-cluster edge probability
        p_inter_cluster: Inter-cluster edge probability

    Returns:
        DualInterpretationGraph with the specified vocabulary size
    """
    # Ensure vocab_size is divisible by clusters
    clusters = 3  # Default number of clusters
    if vocab_size % clusters != 0:
        # Round up to next multiple
        vocab_size = ((vocab_size // clusters) + 1) * clusters

    # Generate extended vocabulary if needed
    vocab = generate_extended_vocabulary(vocab_size, seed)

    config = DualInterpretationConfig(
        vocab_size=vocab_size,
        clusters_per_interpretation=clusters,
        p_intra_cluster=p_intra_cluster,
        p_inter_cluster=p_inter_cluster,
        vocabulary=vocab,
        seed=seed,
    )

    return DualInterpretationGraph(config)


def demo():
    """Demonstrate the DualInterpretationGraph functionality."""
    print("=" * 60)
    print("Dual Interpretation Graph Demo")
    print("=" * 60)

    # Create graph
    config = DualInterpretationConfig(
        vocab_size=15,
        clusters_per_interpretation=3,
        seed=42
    )

    graph = DualInterpretationGraph(config)

    # Print statistics
    stats = graph.get_graph_statistics()
    print("\nGraph Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Generate sample walks
    print("\n" + "-" * 60)
    print("Sample Walks:")
    print("-" * 60)

    for disambig in [None, 5, 10]:
        for true_hyp in ["H1", "H2"]:
            prompt, nodes, meta = graph.generate_ambiguous_walk(
                length=15,
                disambig_position=disambig,
                true_hypothesis=true_hyp,
                return_nodes=True,
            )

            print(f"\nDisambig at pos {disambig}, True: {true_hyp}")
            print(f"  Tokens: {prompt[:50]}...")
            print(f"  H1 consistency: {graph.compute_interpretation_consistency(nodes, 'H1'):.2f}")
            print(f"  H2 consistency: {graph.compute_interpretation_consistency(nodes, 'H2'):.2f}")
            print(f"  Disambig achieved: {meta['disambig_achieved']}")

    return graph


if __name__ == "__main__":
    graph = demo()
