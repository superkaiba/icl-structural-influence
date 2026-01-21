"""
Hierarchical Graph for ICL Representation Influence Experiments.

This module implements a hierarchical graph structure (Stochastic Block Model style)
for the "Graph Tracing" task described in Park et al. (2024) "ICLR: In-Context
Learning of Representations", extended with hierarchical constraints.

The graph consists of:
- Level 1: Super-clusters (distinct communities)
- Level 2: Sub-nodes within each super-cluster

Random walks on this graph tend to stay within super-clusters before traversing
"bridges" to other clusters, enabling us to test hierarchical learning hypotheses.

References:
- Park et al. (2024) arXiv:2501.00070
- Lee et al. (2025) arXiv:2510.12071 (Influence Dynamics)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import random


# Default vocabulary: common nouns that avoid semantic shortcuts
# These are arbitrary labels with no inherent graph structure
DEFAULT_VOCABULARY = [
    # Objects
    "apple", "truck", "sand", "river", "lamp", "chair", "stone", "cloud",
    "book", "glass", "bridge", "wheel", "needle", "mirror", "basket", "anchor",
    # Nature
    "forest", "mountain", "valley", "desert", "ocean", "island", "meadow", "cliff",
    # Misc
    "castle", "garden", "market", "harbor", "tunnel", "tower", "temple", "cave",
    "crystal", "feather", "copper", "silver", "marble", "canvas", "velvet", "bamboo",
    # Additional fillers
    "lantern", "compass", "ribbon", "pillar", "barrel", "curtain", "ladder", "fountain",
    # Extended for larger hierarchies (72+ nodes)
    "pocket", "button", "handle", "socket", "panel", "frame", "switch", "sensor",
    "filter", "valve", "shaft", "gear", "spring", "hinge", "bracket", "clamp",
    "joint", "seal", "blade", "nozzle", "probe", "gauge", "dial", "lever",
    "plate", "bolt", "nut", "screw", "washer", "pin", "rod", "hook",
]


@dataclass
class HierarchicalGraphConfig:
    """Configuration for hierarchical graph generation."""

    num_superclusters: int = 3
    nodes_per_cluster: int = 5

    # Edge probabilities (Stochastic Block Model style)
    p_intra_cluster: float = 0.8   # Probability of edge within same cluster
    p_inter_cluster: float = 0.1   # Probability of edge between clusters (bridge)

    # Random walk parameters
    walk_length: int = 50
    bridge_penalty: float = 0.3    # Reduces probability of crossing clusters

    # Vocabulary
    vocabulary: list[str] = field(default_factory=lambda: DEFAULT_VOCABULARY.copy())

    # Reproducibility
    seed: Optional[int] = None

    def __post_init__(self):
        total_nodes = self.num_superclusters * self.nodes_per_cluster
        if len(self.vocabulary) < total_nodes:
            raise ValueError(
                f"Vocabulary size ({len(self.vocabulary)}) must be >= "
                f"total nodes ({total_nodes})"
            )


class HierarchicalGraph:
    """
    A hierarchical graph with clusters-of-clusters structure.

    Designed for ICL graph tracing experiments where we want to test
    whether models learn global structure (super-clusters) before
    local structure (intra-cluster connections).

    Attributes:
        config: HierarchicalGraphConfig with graph parameters
        adj_matrix: NxN adjacency matrix
        node_to_token: Mapping from node index to vocabulary token
        token_to_node: Mapping from vocabulary token to node index
        cluster_assignments: Array mapping node index to cluster ID
    """

    def __init__(self, config: Optional[HierarchicalGraphConfig] = None):
        self.config = config or HierarchicalGraphConfig()
        self.rng = np.random.default_rng(self.config.seed)

        self.num_nodes = self.config.num_superclusters * self.config.nodes_per_cluster

        # Build the graph
        self._build_adjacency_matrix()
        self._assign_tokens()
        self._build_cluster_assignments()

    def _build_adjacency_matrix(self):
        """
        Build adjacency matrix using Stochastic Block Model.

        Edges within clusters have probability p_intra_cluster.
        Edges between clusters have probability p_inter_cluster.
        """
        n = self.num_nodes
        k = self.config.nodes_per_cluster

        self.adj_matrix = np.zeros((n, n), dtype=np.float32)

        for i in range(n):
            for j in range(i + 1, n):
                cluster_i = i // k
                cluster_j = j // k

                if cluster_i == cluster_j:
                    # Same cluster: high connectivity
                    p = self.config.p_intra_cluster
                else:
                    # Different clusters: sparse bridges
                    p = self.config.p_inter_cluster

                if self.rng.random() < p:
                    self.adj_matrix[i, j] = 1.0
                    self.adj_matrix[j, i] = 1.0

        # Ensure graph is connected: add minimal bridges if needed
        self._ensure_connectivity()

    def _ensure_connectivity(self):
        """Add minimal edges to ensure the graph is connected."""
        from scipy.sparse.csgraph import connected_components
        from scipy.sparse import csr_matrix

        sparse_adj = csr_matrix(self.adj_matrix)
        n_components, labels = connected_components(sparse_adj, directed=False)

        while n_components > 1:
            # Find nodes from different components and connect them
            for comp in range(1, n_components):
                nodes_comp0 = np.where(labels == 0)[0]
                nodes_comp = np.where(labels == comp)[0]

                # Add edge between random nodes from each component
                i = self.rng.choice(nodes_comp0)
                j = self.rng.choice(nodes_comp)
                self.adj_matrix[i, j] = 1.0
                self.adj_matrix[j, i] = 1.0

            sparse_adj = csr_matrix(self.adj_matrix)
            n_components, labels = connected_components(sparse_adj, directed=False)

    def _assign_tokens(self):
        """Randomly assign vocabulary tokens to nodes."""
        shuffled_vocab = self.config.vocabulary.copy()
        self.rng.shuffle(shuffled_vocab)

        self.node_to_token = {i: shuffled_vocab[i] for i in range(self.num_nodes)}
        self.token_to_node = {v: k for k, v in self.node_to_token.items()}

    def _build_cluster_assignments(self):
        """Build array mapping each node to its cluster ID."""
        k = self.config.nodes_per_cluster
        self.cluster_assignments = np.array([i // k for i in range(self.num_nodes)])

    def get_cluster(self, node: int) -> int:
        """Get the super-cluster ID for a node."""
        return self.cluster_assignments[node]

    def get_neighbors(self, node: int) -> list[int]:
        """Get all neighbors of a node."""
        return list(np.where(self.adj_matrix[node] > 0)[0])

    def generate_random_walk(
        self,
        length: Optional[int] = None,
        start_node: Optional[int] = None,
        return_nodes: bool = False,
    ) -> str | tuple[str, list[int]]:
        """
        Generate a random walk on the hierarchical graph.

        The walk uses a biased transition probability that penalizes
        crossing cluster boundaries, encouraging walks to stay within
        clusters for extended periods.

        Args:
            length: Walk length (defaults to config.walk_length)
            start_node: Starting node (random if None)
            return_nodes: If True, also return the node sequence

        Returns:
            Prompt string of space-separated tokens (e.g., "apple truck sand...")
            If return_nodes=True, also returns list of node indices
        """
        length = length or self.config.walk_length

        if start_node is None:
            current = self.rng.integers(0, self.num_nodes)
        else:
            current = start_node

        walk_nodes = [current]
        walk_tokens = [self.node_to_token[current]]

        for _ in range(length - 1):
            neighbors = self.get_neighbors(current)

            if not neighbors:
                # Dead end: restart from random node
                current = self.rng.integers(0, self.num_nodes)
            else:
                # Compute transition probabilities with bridge penalty
                current_cluster = self.get_cluster(current)
                probs = []

                for neighbor in neighbors:
                    neighbor_cluster = self.get_cluster(neighbor)
                    if neighbor_cluster == current_cluster:
                        probs.append(1.0)
                    else:
                        # Penalize cluster-crossing transitions
                        probs.append(self.config.bridge_penalty)

                # Normalize
                probs = np.array(probs)
                probs = probs / probs.sum()

                # Sample next node
                current = self.rng.choice(neighbors, p=probs)

            walk_nodes.append(current)
            walk_tokens.append(self.node_to_token[current])

        prompt = " ".join(walk_tokens)

        if return_nodes:
            return prompt, walk_nodes
        return prompt

    def generate_batch(
        self,
        batch_size: int,
        length: Optional[int] = None,
    ) -> list[dict]:
        """
        Generate a batch of random walks with metadata.

        Args:
            batch_size: Number of walks to generate
            length: Walk length (defaults to config.walk_length)

        Returns:
            List of dicts with 'prompt', 'nodes', 'clusters' keys
        """
        batch = []
        for _ in range(batch_size):
            prompt, nodes = self.generate_random_walk(
                length=length,
                return_nodes=True
            )
            clusters = [self.get_cluster(n) for n in nodes]
            batch.append({
                "prompt": prompt,
                "nodes": nodes,
                "clusters": clusters,
            })
        return batch

    def get_cluster_transition_points(self, nodes: list[int]) -> list[int]:
        """
        Find indices where the walk transitions between clusters.

        Useful for identifying "bridge" tokens that may have
        different influence dynamics.

        Args:
            nodes: List of node indices from a random walk

        Returns:
            List of indices where cluster changes
        """
        transitions = []
        for i in range(1, len(nodes)):
            if self.get_cluster(nodes[i]) != self.get_cluster(nodes[i-1]):
                transitions.append(i)
        return transitions

    def get_graph_statistics(self) -> dict:
        """Return statistics about the graph structure."""
        degrees = self.adj_matrix.sum(axis=1)

        # Count intra vs inter cluster edges
        intra_edges = 0
        inter_edges = 0
        k = self.config.nodes_per_cluster

        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if self.adj_matrix[i, j] > 0:
                    if i // k == j // k:
                        intra_edges += 1
                    else:
                        inter_edges += 1

        return {
            "num_nodes": self.num_nodes,
            "num_superclusters": self.config.num_superclusters,
            "nodes_per_cluster": self.config.nodes_per_cluster,
            "total_edges": int(self.adj_matrix.sum() / 2),
            "intra_cluster_edges": intra_edges,
            "inter_cluster_edges": inter_edges,
            "mean_degree": float(degrees.mean()),
            "min_degree": int(degrees.min()),
            "max_degree": int(degrees.max()),
        }

    def visualize(self, save_path: Optional[str] = None):
        """
        Visualize the hierarchical graph with cluster coloring.

        Args:
            save_path: If provided, save figure to this path
        """
        import matplotlib.pyplot as plt
        import networkx as nx

        G = nx.from_numpy_array(self.adj_matrix)

        # Color by cluster
        colors = plt.cm.tab10(self.cluster_assignments / self.config.num_superclusters)

        # Layout that respects clusters
        pos = nx.spring_layout(G, seed=self.config.seed or 42, k=2)

        fig, ax = plt.subplots(figsize=(10, 8))

        nx.draw_networkx_nodes(
            G, pos,
            node_color=colors,
            node_size=500,
            ax=ax
        )

        nx.draw_networkx_edges(
            G, pos,
            edge_color='gray',
            alpha=0.5,
            ax=ax
        )

        # Add token labels
        labels = {i: self.node_to_token[i] for i in range(self.num_nodes)}
        nx.draw_networkx_labels(
            G, pos,
            labels=labels,
            font_size=8,
            ax=ax
        )

        ax.set_title(f"Hierarchical Graph: {self.config.num_superclusters} clusters × "
                     f"{self.config.nodes_per_cluster} nodes")
        ax.axis('off')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig, ax


# =============================================================================
# Deep (N-Level) Hierarchical Graph
# =============================================================================

@dataclass
class DeepHierarchyConfig:
    """Configuration for N-level hierarchical graphs with configurable depth.

    The hierarchy structure is defined by branching_factors:
    - branching_factors=[2, 2, 4] creates:
      - 2 super-clusters (Level 1)
      - 2 mid-clusters per super (Level 2)
      - 4 leaf nodes per mid-cluster (Level 3)
      - Total nodes = 2 * 2 * 4 = 16

    Example hierarchy with [2, 2, 4]:

    Level 0 (Root):           ALL
                            /     \\
    Level 1 (Super):      S0       S1
                         / \\       / \\
    Level 2 (Mid):     M00 M01   M10 M11
                      /|||\\     /|||\\
    Level 3 (Leaf):  Nodes 0-3, 4-7, 8-11, 12-15
    """

    # Hierarchy structure: list of branching factors from top to bottom
    branching_factors: list[int] = field(default_factory=lambda: [2, 2, 4])

    # Edge probability parameters
    p_same_level: float = 0.9       # Edge probability within same leaf cluster
    p_decay_per_level: float = 0.3  # Multiply by this for each level of separation

    # Random walk parameters
    walk_length: int = 100
    bridge_penalty_per_level: float = 0.5  # Penalty multiplied per level crossed

    # Vocabulary (auto-generated if None)
    vocabulary: Optional[list[str]] = None

    # Reproducibility
    seed: Optional[int] = None

    @property
    def num_levels(self) -> int:
        """Number of hierarchy levels (including root)."""
        return len(self.branching_factors) + 1

    @property
    def total_nodes(self) -> int:
        """Total number of leaf nodes."""
        result = 1
        for bf in self.branching_factors:
            result *= bf
        return result

    def __post_init__(self):
        # Generate default vocabulary if needed
        if self.vocabulary is None:
            self.vocabulary = DEFAULT_VOCABULARY.copy()

        if len(self.vocabulary) < self.total_nodes:
            raise ValueError(
                f"Vocabulary size ({len(self.vocabulary)}) must be >= "
                f"total nodes ({self.total_nodes})"
            )


class DeepHierarchicalGraph:
    """
    N-level hierarchical graph with configurable depth.

    Extends the basic 2-level HierarchicalGraph to support arbitrary hierarchy depth,
    enabling experiments on multi-scale structural learning.

    Key features:
    - Configurable depth via branching_factors
    - Per-level cluster assignments
    - Hierarchy path for each node (e.g., node 5 -> (0, 1, 1))
    - Hierarchy distance computation between nodes

    Attributes:
        config: DeepHierarchyConfig with graph parameters
        hierarchy_paths: Dict mapping node -> tuple of cluster indices at each level
        level_assignments: List of dicts, level_assignments[level][node] = cluster_id
        adj_matrix: NxN adjacency matrix
        node_to_token: Mapping from node index to vocabulary token
        token_to_node: Mapping from vocabulary token to node index
    """

    def __init__(self, config: Optional[DeepHierarchyConfig] = None):
        self.config = config or DeepHierarchyConfig()
        self.rng = np.random.default_rng(self.config.seed)

        self.num_nodes = self.config.total_nodes
        self.num_levels = self.config.num_levels

        # Build hierarchy structure
        self._build_hierarchy_structure()
        self._build_adjacency_matrix()
        self._assign_tokens()

    def _compute_divisors(self) -> list[int]:
        """Compute divisors for converting node index to hierarchy path.

        For branching_factors=[2, 2, 4]:
        - divisors = [8, 4, 1]
        - node 5: 5//8=0, (5%8)//4=1, (5%4)//1=1 -> path (0, 1, 1)
        """
        divisors = []
        prod = 1
        for bf in reversed(self.config.branching_factors):
            divisors.append(prod)
            prod *= bf
        divisors.reverse()
        return divisors

    def _compute_hierarchy_path(self, node: int) -> tuple:
        """Convert node index to hierarchy path.

        Args:
            node: Node index (0 to total_nodes-1)

        Returns:
            Tuple of cluster indices at each level (excluding root).
            E.g., for branching_factors=[2,2,4], node 5 -> (0, 1, 1)
        """
        divisors = self._compute_divisors()
        path = []
        remaining = node

        for i, bf in enumerate(self.config.branching_factors):
            level_idx = remaining // divisors[i]
            path.append(level_idx)
            remaining = remaining % divisors[i]

        return tuple(path)

    def _path_to_cluster_id(self, path: tuple, level: int) -> int:
        """Convert hierarchy path prefix to unique cluster ID at a level.

        Args:
            path: Full hierarchy path
            level: Target level (1 to num_levels-1)

        Returns:
            Unique cluster ID at the specified level
        """
        if level == 0:
            return 0  # Root level - all nodes in same cluster

        # Use path prefix up to this level
        path_prefix = path[:level]

        # Compute unique ID from path prefix
        cluster_id = 0
        multiplier = 1
        for i in range(level - 1, -1, -1):
            cluster_id += path_prefix[i] * multiplier
            multiplier *= self.config.branching_factors[i]

        return cluster_id

    def _build_hierarchy_structure(self):
        """Build hierarchy paths and level assignments for all nodes."""
        self.hierarchy_paths = {}
        # level_assignments[level] = {node: cluster_id}
        # Level 0 is root (all nodes -> cluster 0)
        # Levels 1 to num_levels-1 are intermediate/leaf levels
        self.level_assignments = [{} for _ in range(self.num_levels)]

        for node in range(self.num_nodes):
            path = self._compute_hierarchy_path(node)
            self.hierarchy_paths[node] = path

            # Assign cluster ID at each level
            for level in range(self.num_levels):
                cluster_id = self._path_to_cluster_id(path, level)
                self.level_assignments[level][node] = cluster_id

    def hierarchy_distance(self, node_a: int, node_b: int) -> int:
        """Compute hierarchy distance between two nodes.

        Distance = number of levels from the deepest common ancestor to leaves.
        - Same leaf cluster: distance = 0
        - Different leaf, same parent: distance = 1
        - etc.

        Args:
            node_a: First node index
            node_b: Second node index

        Returns:
            Number of hierarchy levels separating the nodes
        """
        path_a = self.hierarchy_paths[node_a]
        path_b = self.hierarchy_paths[node_b]

        # Find first divergence point
        for i, (a, b) in enumerate(zip(path_a, path_b)):
            if a != b:
                return len(path_a) - i

        return 0  # Same path = same leaf cluster

    def get_cluster_at_level(self, node: int, level: int) -> int:
        """Get cluster assignment for a node at a specific hierarchy level.

        Args:
            node: Node index
            level: Hierarchy level (0=root, 1=first split, etc.)

        Returns:
            Cluster ID at the specified level
        """
        return self.level_assignments[level][node]

    def get_level_labels(self, level: int) -> dict[int, int]:
        """Get all node -> cluster_id mappings for a level.

        Args:
            level: Hierarchy level

        Returns:
            Dictionary mapping node index to cluster ID at that level
        """
        return self.level_assignments[level].copy()

    def get_level_labels_array(self, level: int) -> np.ndarray:
        """Get cluster labels for all nodes at a level as numpy array.

        Args:
            level: Hierarchy level

        Returns:
            Array of shape (num_nodes,) with cluster IDs
        """
        return np.array([self.level_assignments[level][i] for i in range(self.num_nodes)])

    def _build_adjacency_matrix(self):
        """Build adjacency matrix with hierarchy-aware edge probabilities.

        Edge probability decreases exponentially with hierarchy distance:
        p_edge = p_same_level * (p_decay_per_level ^ hierarchy_distance)
        """
        n = self.num_nodes
        self.adj_matrix = np.zeros((n, n), dtype=np.float32)

        for i in range(n):
            for j in range(i + 1, n):
                dist = self.hierarchy_distance(i, j)

                # Edge probability decays with hierarchy distance
                p = self.config.p_same_level * (self.config.p_decay_per_level ** dist)

                if self.rng.random() < p:
                    self.adj_matrix[i, j] = 1.0
                    self.adj_matrix[j, i] = 1.0

        # Ensure connectivity
        self._ensure_connectivity()

    def _ensure_connectivity(self):
        """Add minimal edges to ensure the graph is connected."""
        from scipy.sparse.csgraph import connected_components
        from scipy.sparse import csr_matrix

        sparse_adj = csr_matrix(self.adj_matrix)
        n_components, labels = connected_components(sparse_adj, directed=False)

        while n_components > 1:
            for comp in range(1, n_components):
                nodes_comp0 = np.where(labels == 0)[0]
                nodes_comp = np.where(labels == comp)[0]

                i = self.rng.choice(nodes_comp0)
                j = self.rng.choice(nodes_comp)
                self.adj_matrix[i, j] = 1.0
                self.adj_matrix[j, i] = 1.0

            sparse_adj = csr_matrix(self.adj_matrix)
            n_components, labels = connected_components(sparse_adj, directed=False)

    def _assign_tokens(self):
        """Randomly assign vocabulary tokens to nodes."""
        shuffled_vocab = self.config.vocabulary.copy()
        self.rng.shuffle(shuffled_vocab)

        self.node_to_token = {i: shuffled_vocab[i] for i in range(self.num_nodes)}
        self.token_to_node = {v: k for k, v in self.node_to_token.items()}

    def get_neighbors(self, node: int) -> list[int]:
        """Get all neighbors of a node."""
        return list(np.where(self.adj_matrix[node] > 0)[0])

    def generate_random_walk(
        self,
        length: Optional[int] = None,
        start_node: Optional[int] = None,
        return_nodes: bool = False,
    ) -> str | tuple[str, list[int]]:
        """Generate a random walk with hierarchy-aware transition probabilities.

        Transition probability penalizes crossing hierarchy levels:
        penalty = bridge_penalty_per_level ^ hierarchy_distance
        """
        length = length or self.config.walk_length

        if start_node is None:
            current = self.rng.integers(0, self.num_nodes)
        else:
            current = start_node

        walk_nodes = [current]
        walk_tokens = [self.node_to_token[current]]

        for _ in range(length - 1):
            neighbors = self.get_neighbors(current)

            if not neighbors:
                current = self.rng.integers(0, self.num_nodes)
            else:
                probs = []
                for neighbor in neighbors:
                    dist = self.hierarchy_distance(current, neighbor)
                    # Penalty increases with hierarchy distance
                    penalty = self.config.bridge_penalty_per_level ** dist
                    probs.append(penalty)

                probs = np.array(probs)
                probs = probs / probs.sum()
                current = self.rng.choice(neighbors, p=probs)

            walk_nodes.append(current)
            walk_tokens.append(self.node_to_token[current])

        prompt = " ".join(walk_tokens)

        if return_nodes:
            return prompt, walk_nodes
        return prompt

    def generate_batch(
        self,
        batch_size: int,
        length: Optional[int] = None,
    ) -> list[dict]:
        """Generate a batch of random walks with multi-level cluster metadata.

        Returns:
            List of dicts with 'prompt', 'nodes', 'level_clusters' keys
            level_clusters is a dict mapping level -> list of cluster IDs
        """
        batch = []
        for _ in range(batch_size):
            prompt, nodes = self.generate_random_walk(
                length=length,
                return_nodes=True
            )

            # Get cluster assignments at each level
            level_clusters = {}
            for level in range(self.num_levels):
                level_clusters[level] = [self.get_cluster_at_level(n, level) for n in nodes]

            batch.append({
                "prompt": prompt,
                "nodes": nodes,
                "level_clusters": level_clusters,
            })
        return batch

    def get_level_transition_points(self, nodes: list[int], level: int) -> list[int]:
        """Find indices where walk transitions between clusters at a specific level.

        Args:
            nodes: List of node indices from a random walk
            level: Hierarchy level to check transitions

        Returns:
            List of indices where cluster changes at this level
        """
        transitions = []
        for i in range(1, len(nodes)):
            if self.get_cluster_at_level(nodes[i], level) != self.get_cluster_at_level(nodes[i-1], level):
                transitions.append(i)
        return transitions

    def get_graph_statistics(self) -> dict:
        """Return statistics about the graph structure."""
        degrees = self.adj_matrix.sum(axis=1)

        # Count edges at each hierarchy distance
        edge_counts_by_distance = {}
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if self.adj_matrix[i, j] > 0:
                    dist = self.hierarchy_distance(i, j)
                    edge_counts_by_distance[dist] = edge_counts_by_distance.get(dist, 0) + 1

        return {
            "num_nodes": self.num_nodes,
            "num_levels": self.num_levels,
            "branching_factors": self.config.branching_factors,
            "total_edges": int(self.adj_matrix.sum() / 2),
            "edges_by_hierarchy_distance": edge_counts_by_distance,
            "mean_degree": float(degrees.mean()),
            "min_degree": int(degrees.min()),
            "max_degree": int(degrees.max()),
        }

    def visualize(self, save_path: Optional[str] = None, color_level: int = 1):
        """Visualize the graph with coloring by hierarchy level.

        Args:
            save_path: If provided, save figure to this path
            color_level: Hierarchy level to use for node coloring (default: 1)
        """
        import matplotlib.pyplot as plt
        import networkx as nx

        G = nx.from_numpy_array(self.adj_matrix)

        # Color by specified hierarchy level
        level_labels = self.get_level_labels_array(color_level)
        num_clusters = len(set(level_labels))
        colors = plt.cm.tab10(level_labels / max(num_clusters, 1))

        pos = nx.spring_layout(G, seed=self.config.seed or 42, k=2)

        fig, ax = plt.subplots(figsize=(12, 10))

        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=500, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5, ax=ax)

        labels = {i: self.node_to_token[i] for i in range(self.num_nodes)}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)

        ax.set_title(
            f"Deep Hierarchical Graph: {self.num_levels} levels, "
            f"branching={self.config.branching_factors}, "
            f"colored by level {color_level}"
        )
        ax.axis('off')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig, ax


def demo():
    """Demonstrate the HierarchicalGraph functionality."""

    print("=" * 60)
    print("Hierarchical Graph for ICL Representation Influence")
    print("=" * 60)

    # Create graph with default config
    config = HierarchicalGraphConfig(
        num_superclusters=3,
        nodes_per_cluster=5,
        walk_length=30,
        seed=42
    )

    graph = HierarchicalGraph(config)

    # Print statistics
    stats = graph.get_graph_statistics()
    print("\nGraph Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Generate sample walks
    print("\n" + "-" * 60)
    print("Sample Random Walks:")
    print("-" * 60)

    for i in range(3):
        prompt, nodes = graph.generate_random_walk(length=15, return_nodes=True)
        clusters = [graph.get_cluster(n) for n in nodes]
        transitions = graph.get_cluster_transition_points(nodes)

        print(f"\nWalk {i+1}:")
        print(f"  Tokens: {prompt}")
        print(f"  Clusters: {clusters}")
        print(f"  Cluster transitions at positions: {transitions}")

    # Generate batch
    print("\n" + "-" * 60)
    print("Batch Generation:")
    print("-" * 60)

    batch = graph.generate_batch(batch_size=5, length=20)
    print(f"Generated {len(batch)} walks")
    print(f"First walk prompt: {batch[0]['prompt'][:50]}...")

    return graph


if __name__ == "__main__":
    graph = demo()
