#!/usr/bin/env python3
"""
Two experiments to replicate Lee et al. stagewise learning:

1. 3-LEVEL HIERARCHICAL GRAPH:
   - Shows different hierarchy levels emerging at different context lengths
   - Replicates Lee et al.'s stagewise branching

2. SEMANTIC CONFLICT EXPERIMENT:
   - Semantically similar words placed in different graph clusters
   - Shows when model overrides pretraining semantics with task structure
"""

import json
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

sys.path.insert(0, str(Path(__file__).parent / "src"))
from models import HookedLLM


# =============================================================================
# EXPERIMENT 1: 3-Level Hierarchical Graph
# =============================================================================

class HierarchicalGraph3Level:
    """
    3-level hierarchical graph structure:

    Level 0 (Root):     ALL
                       /   \
    Level 1 (Super):  S_A   S_B
                     / \    / \
    Level 2 (Mid):  M1 M2  M3 M4
                    /\ /\  /\ /\
    Level 3 (Leaf): Individual nodes (4 per mid-cluster = 16 total)
    """

    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)

        # Vocabulary - 16 words, 4 per mid-cluster (expanded for denser plots)
        self.vocabulary = [
            "crystal", "marble", "diamond", "granite",    # M1 (under S_A)
            "lantern", "castle", "beacon", "fortress",    # M2 (under S_A)
            "cloud", "canvas", "mist", "fabric",          # M3 (under S_B)
            "pillar", "tunnel", "column", "passage",      # M4 (under S_B)
        ]

        self.num_nodes = len(self.vocabulary)
        self.node_to_token = {i: self.vocabulary[i] for i in range(self.num_nodes)}
        self.token_to_node = {v: k for k, v in self.node_to_token.items()}

        # Hierarchy assignments
        # Level 1 (Super): 0-7 = S_A, 8-15 = S_B
        self.level1_cluster = {i: 0 if i < 8 else 1 for i in range(16)}

        # Level 2 (Mid): 0-3 = M1, 4-7 = M2, 8-11 = M3, 12-15 = M4
        self.level2_cluster = {i: i // 4 for i in range(16)}

        # Level 3 (Leaf): Each node is its own cluster
        self.level3_cluster = {i: i for i in range(16)}

        # Build adjacency matrix with hierarchical edge probabilities
        self._build_adjacency_matrix()

    def _build_adjacency_matrix(self):
        """Build adjacency with deterministic hierarchical structure.

        Uses deterministic edges to guarantee hierarchy is preserved:
        - All nodes fully connected within mid-cluster
        - Exactly 2 edges between sibling mid-clusters (within same super)
        - Exactly 1 edge between super-clusters (for connectivity)

        This prevents random edge placement from distorting the hierarchy.
        """
        self.adj_matrix = np.zeros((self.num_nodes, self.num_nodes))

        # STEP 1: Fully connect all nodes within each mid-cluster
        # This gives strongest within-mid-cluster signal
        for mid_cluster in range(4):
            nodes_in_cluster = [i for i in range(16) if self.level2_cluster[i] == mid_cluster]
            for i in range(len(nodes_in_cluster)):
                for j in range(i + 1, len(nodes_in_cluster)):
                    n1, n2 = nodes_in_cluster[i], nodes_in_cluster[j]
                    self.adj_matrix[n1, n2] = 1
                    self.adj_matrix[n2, n1] = 1

        # STEP 2: Add controlled edges between sibling mid-clusters
        # M_A1 (0-3) <-> M_A2 (4-7): connect first nodes of each
        # M_B1 (8-11) <-> M_B2 (12-15): connect first nodes of each
        sibling_edges = [
            (0, 4), (1, 5),   # M_A1 <-> M_A2 (2 edges)
            (8, 12), (9, 13), # M_B1 <-> M_B2 (2 edges)
        ]
        for i, j in sibling_edges:
            self.adj_matrix[i, j] = 1
            self.adj_matrix[j, i] = 1

        # STEP 3: Add single edge between super-clusters for connectivity
        # Connect through one specific pair to minimize cross-super influence
        cross_super_edge = (3, 11)  # granite (M_A1) <-> fabric (M_B1)
        self.adj_matrix[cross_super_edge[0], cross_super_edge[1]] = 1
        self.adj_matrix[cross_super_edge[1], cross_super_edge[0]] = 1

        # No need for _ensure_connectivity - graph is now deterministically connected

    def _ensure_connectivity(self):
        """Ensure graph is connected."""
        from scipy.sparse.csgraph import connected_components
        from scipy.sparse import csr_matrix

        sparse_adj = csr_matrix(self.adj_matrix)
        n_components, labels = connected_components(sparse_adj, directed=False)

        while n_components > 1:
            for comp in range(1, n_components):
                nodes_0 = np.where(labels == 0)[0]
                nodes_c = np.where(labels == comp)[0]
                i, j = self.rng.choice(nodes_0), self.rng.choice(nodes_c)
                self.adj_matrix[i, j] = self.adj_matrix[j, i] = 1

            sparse_adj = csr_matrix(self.adj_matrix)
            n_components, labels = connected_components(sparse_adj, directed=False)

    def get_neighbors(self, node):
        return list(np.where(self.adj_matrix[node] > 0)[0])

    def generate_random_walk(self, length, return_nodes=False):
        """Generate random walk."""
        current = self.rng.integers(0, self.num_nodes)
        walk_nodes = [current]
        walk_tokens = [self.node_to_token[current]]

        for _ in range(length - 1):
            neighbors = self.get_neighbors(current)
            if neighbors:
                current = self.rng.choice(neighbors)
            else:
                current = self.rng.integers(0, self.num_nodes)
            walk_nodes.append(current)
            walk_tokens.append(self.node_to_token[current])

        prompt = " ".join(walk_tokens)
        if return_nodes:
            return prompt, walk_nodes
        return prompt

    def get_hierarchy_labels(self):
        """Return hierarchy labels for visualization."""
        return {
            "level1": self.level1_cluster,  # Super-cluster
            "level2": self.level2_cluster,  # Mid-cluster
            "level3": self.level3_cluster,  # Leaf
            "super_names": {0: "Super_A", 1: "Super_B"},
            "mid_names": {0: "Mid_A1", 1: "Mid_A2", 2: "Mid_B1", 3: "Mid_B2"},
        }


# =============================================================================
# EXPERIMENT 2: Semantic Conflict Graph
# =============================================================================

class SemanticConflictGraph:
    """
    Graph where tokens are assigned to clusters.

    Two modes:
    1. use_semantic_tokens=True: Semantically similar words in DIFFERENT clusters
       - Animals: cat, dog, bird (in clusters A, B, C)
       - Electronics: computer, television, radio
       - etc.

    2. use_semantic_tokens=False: Unrelated tokens (control condition)
       - Group A: piano, river, hammer
       - Group B: cloud, pencil, mountain
       - etc.
    """

    def __init__(self, seed=42, use_semantic_tokens=True):
        self.rng = np.random.default_rng(seed)
        self.use_semantic_tokens = use_semantic_tokens

        if use_semantic_tokens:
            # SEMANTIC tokens: similar words placed in DIFFERENT clusters
            self.semantic_groups = {
                "animals": ["cat", "dog", "bird"],
                "electronics": ["computer", "television", "radio"],
                "vegetables": ["tomato", "potato", "carrot"],
                "furniture": ["chair", "table", "desk"],
            }
            self.graph_clusters = {
                0: ["cat", "computer", "tomato", "chair"],       # Cluster A
                1: ["dog", "television", "potato", "table"],     # Cluster B
                2: ["bird", "radio", "carrot", "desk"],          # Cluster C
            }
        else:
            # UNRELATED tokens: no semantic relationship (control)
            self.semantic_groups = {
                "group_a": ["piano", "river", "hammer"],
                "group_b": ["cloud", "pencil", "mountain"],
                "group_c": ["window", "coffee", "bicycle"],
                "group_d": ["garden", "mirror", "blanket"],
            }
            self.graph_clusters = {
                0: ["piano", "cloud", "window", "garden"],       # Cluster A
                1: ["river", "pencil", "coffee", "mirror"],      # Cluster B
                2: ["hammer", "mountain", "bicycle", "blanket"], # Cluster C
            }

        # Build vocabulary and mappings
        self.vocabulary = []
        self.token_to_graph_cluster = {}
        self.token_to_semantic_group = {}

        for cluster_id, tokens in self.graph_clusters.items():
            for token in tokens:
                self.vocabulary.append(token)
                self.token_to_graph_cluster[token] = cluster_id

        for group_name, tokens in self.semantic_groups.items():
            for token in tokens:
                self.token_to_semantic_group[token] = group_name

        self.num_nodes = len(self.vocabulary)
        self.node_to_token = {i: self.vocabulary[i] for i in range(self.num_nodes)}
        self.token_to_node = {v: k for k, v in self.node_to_token.items()}

        # Build adjacency matrix based on GRAPH clusters (not semantic!)
        self._build_adjacency_matrix()

    def _build_adjacency_matrix(self):
        """Build adjacency based on graph clusters."""
        self.adj_matrix = np.zeros((self.num_nodes, self.num_nodes))

        p_same_cluster = 0.8   # High connectivity within graph cluster
        p_diff_cluster = 0.1   # Low connectivity between clusters

        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                token_i = self.node_to_token[i]
                token_j = self.node_to_token[j]

                if self.token_to_graph_cluster[token_i] == self.token_to_graph_cluster[token_j]:
                    p = p_same_cluster
                else:
                    p = p_diff_cluster

                if self.rng.random() < p:
                    self.adj_matrix[i, j] = 1
                    self.adj_matrix[j, i] = 1

        self._ensure_connectivity()

    def _ensure_connectivity(self):
        """Ensure graph is connected."""
        from scipy.sparse.csgraph import connected_components
        from scipy.sparse import csr_matrix

        sparse_adj = csr_matrix(self.adj_matrix)
        n_components, labels = connected_components(sparse_adj, directed=False)

        while n_components > 1:
            for comp in range(1, n_components):
                nodes_0 = np.where(labels == 0)[0]
                nodes_c = np.where(labels == comp)[0]
                i, j = self.rng.choice(nodes_0), self.rng.choice(nodes_c)
                self.adj_matrix[i, j] = self.adj_matrix[j, i] = 1

            sparse_adj = csr_matrix(self.adj_matrix)
            n_components, labels = connected_components(sparse_adj, directed=False)

    def get_neighbors(self, node):
        return list(np.where(self.adj_matrix[node] > 0)[0])

    def generate_random_walk(self, length, return_nodes=False):
        """Generate random walk."""
        current = self.rng.integers(0, self.num_nodes)
        walk_nodes = [current]
        walk_tokens = [self.node_to_token[current]]

        for _ in range(length - 1):
            neighbors = self.get_neighbors(current)
            if neighbors:
                current = self.rng.choice(neighbors)
            else:
                current = self.rng.integers(0, self.num_nodes)
            walk_nodes.append(current)
            walk_tokens.append(self.node_to_token[current])

        prompt = " ".join(walk_tokens)
        if return_nodes:
            return prompt, walk_nodes
        return prompt

    def get_semantic_pairs(self):
        """Return pairs of semantically similar tokens in different clusters."""
        pairs = []
        for group_name, tokens in self.semantic_groups.items():
            for i, t1 in enumerate(tokens):
                for t2 in tokens[i+1:]:
                    c1 = self.token_to_graph_cluster[t1]
                    c2 = self.token_to_graph_cluster[t2]
                    if c1 != c2:
                        pairs.append((t1, t2, group_name))
        return pairs


# =============================================================================
# Data Collection
# =============================================================================

def collect_representations(model, tokenizer, graph, context_lengths, n_samples=50, layer_idx=-5):
    """Collect token representations at each context length."""
    token_reps = {ctx: {} for ctx in context_lengths}

    for ctx_len in context_lengths:
        print(f"    N={ctx_len}...", end=" ", flush=True)
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


# =============================================================================
# Analysis Functions
# =============================================================================

def compute_hierarchy_distances(token_reps, graph, context_lengths):
    """
    Compute distances at each hierarchy level across context lengths.

    Returns distances for:
    - Within mid-cluster (closest)
    - Within super-cluster, across mid (medium)
    - Across super-cluster (farthest)
    """
    hierarchy = graph.get_hierarchy_labels()
    level1 = hierarchy["level1"]
    level2 = hierarchy["level2"]

    results = {
        "within_mid": [],      # Same mid-cluster
        "within_super": [],    # Same super, diff mid
        "across_super": [],    # Different super
        "context_lengths": context_lengths,
    }

    for ctx_len in context_lengths:
        reps = token_reps[ctx_len]
        tokens = list(reps.keys())

        within_mid_dists = []
        within_super_dists = []
        across_super_dists = []

        for i, t1 in enumerate(tokens):
            for t2 in tokens[i+1:]:
                n1 = graph.token_to_node[t1]
                n2 = graph.token_to_node[t2]

                dist = np.linalg.norm(reps[t1] - reps[t2])

                if level2[n1] == level2[n2]:
                    within_mid_dists.append(dist)
                elif level1[n1] == level1[n2]:
                    within_super_dists.append(dist)
                else:
                    across_super_dists.append(dist)

        results["within_mid"].append(np.mean(within_mid_dists) if within_mid_dists else 0)
        results["within_super"].append(np.mean(within_super_dists) if within_super_dists else 0)
        results["across_super"].append(np.mean(across_super_dists) if across_super_dists else 0)

    return results


def compute_semantic_vs_graph_similarity(token_reps, graph, context_lengths):
    """
    Compare semantic similarity vs graph similarity across context.

    Key question: At what point does graph structure override semantic structure?
    """
    semantic_pairs = graph.get_semantic_pairs()

    results = {
        "semantic_same_graph_diff": [],   # Semantically similar, different graph clusters
        "semantic_diff_graph_same": [],   # Semantically different, same graph cluster
        "context_lengths": context_lengths,
    }

    for ctx_len in context_lengths:
        reps = token_reps[ctx_len]

        # Semantic similar, graph different (should start close, end far)
        sem_same_graph_diff = []
        for t1, t2, _ in semantic_pairs:
            if t1 in reps and t2 in reps:
                sim = cosine_similarity([reps[t1]], [reps[t2]])[0, 0]
                sem_same_graph_diff.append(sim)

        # Semantic different, graph same (should start far, end close)
        sem_diff_graph_same = []
        for cluster_id, cluster_tokens in graph.graph_clusters.items():
            for i, t1 in enumerate(cluster_tokens):
                for t2 in cluster_tokens[i+1:]:
                    # Check if semantically different
                    if graph.token_to_semantic_group[t1] != graph.token_to_semantic_group[t2]:
                        if t1 in reps and t2 in reps:
                            sim = cosine_similarity([reps[t1]], [reps[t2]])[0, 0]
                            sem_diff_graph_same.append(sim)

        results["semantic_same_graph_diff"].append(np.mean(sem_same_graph_diff) if sem_same_graph_diff else 0)
        results["semantic_diff_graph_same"].append(np.mean(sem_diff_graph_same) if sem_diff_graph_same else 0)

    return results


# =============================================================================
# Visualization
# =============================================================================

def create_hierarchy_figure(token_reps, graph, context_lengths, distances, output_dir):
    """Create visualization for 3-level hierarchy experiment."""
    fig = plt.figure(figsize=(18, 14))

    hierarchy = graph.get_hierarchy_labels()

    # Color schemes
    super_colors = {0: '#e41a1c', 1: '#377eb8'}
    mid_colors = {0: '#e41a1c', 1: '#ff7f7f', 2: '#377eb8', 3: '#7fbfff'}

    cmap = plt.cm.viridis
    norm = Normalize(vmin=min(context_lengths), vmax=max(context_lengths))

    # =========================================================================
    # Panel 1: Hierarchy structure diagram
    # =========================================================================
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.axis('off')

    hierarchy_text = """
    3-LEVEL HIERARCHY STRUCTURE
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Level 0 (Root):        ALL
                          /   \\
    Level 1 (Super):   S_A     S_B
                      /  \\    /  \\
    Level 2 (Mid):  M_A1 M_A2 M_B1 M_B2
                    |  |  |  |  |  |  |  |
    Level 3:       crystal marble lantern castle cloud canvas pillar tunnel


    Edge Probabilities:
    • Same mid-cluster: 90%
    • Same super, diff mid: 30%
    • Different super: 5%

    Expected Stagewise Learning:
    1. First: Super-clusters separate (S_A vs S_B)
    2. Then: Mid-clusters separate within super
    """
    ax1.text(0.05, 0.95, hierarchy_text, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    ax1.set_title("A. Hierarchy Structure", fontsize=12, fontweight='bold')

    # =========================================================================
    # Panel 2: Distance by hierarchy level
    # =========================================================================
    ax2 = fig.add_subplot(2, 3, 2)

    ax2.plot(context_lengths, distances["within_mid"], 'o-', color='green',
             linewidth=2.5, markersize=8, label='Within Mid-Cluster')
    ax2.plot(context_lengths, distances["within_super"], 's-', color='orange',
             linewidth=2.5, markersize=8, label='Within Super (diff Mid)')
    ax2.plot(context_lengths, distances["across_super"], '^-', color='red',
             linewidth=2.5, markersize=8, label='Across Super-Clusters')

    ax2.set_xlabel("Context Length (N)", fontsize=11)
    ax2.set_ylabel("Mean Distance", fontsize=11)
    ax2.set_title("B. Representation Distance by Hierarchy Level", fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 3: Hierarchy ratio (between/within)
    # =========================================================================
    ax3 = fig.add_subplot(2, 3, 3)

    # Ratio of across-super to within-mid (should increase = hierarchy emerging)
    ratio_super = [a/w if w > 0 else 0 for a, w in
                   zip(distances["across_super"], distances["within_mid"])]
    ratio_mid = [a/w if w > 0 else 0 for a, w in
                 zip(distances["within_super"], distances["within_mid"])]

    ax3.plot(context_lengths, ratio_super, 'o-', color='red', linewidth=2.5,
             markersize=8, label='Across-Super / Within-Mid')
    ax3.plot(context_lengths, ratio_mid, 's-', color='orange', linewidth=2.5,
             markersize=8, label='Within-Super / Within-Mid')
    ax3.axhline(y=1, color='gray', linestyle='--', alpha=0.5)

    ax3.set_xlabel("Context Length (N)", fontsize=11)
    ax3.set_ylabel("Distance Ratio", fontsize=11)
    ax3.set_title("C. Hierarchy Emergence\n(Higher = More Separation)", fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 4-5: PCA at early vs late context
    # =========================================================================
    for idx, ctx_len in enumerate([context_lengths[0], context_lengths[-1]]):
        ax = fig.add_subplot(2, 3, 4 + idx)

        reps = token_reps[ctx_len]
        tokens = list(reps.keys())
        X = np.array([reps[t] for t in tokens])

        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)

        for i, token in enumerate(tokens):
            node = graph.token_to_node[token]
            mid_cluster = hierarchy["level2"][node]
            super_cluster = hierarchy["level1"][node]

            color = mid_colors[mid_cluster]
            marker = 'o' if super_cluster == 0 else 's'

            ax.scatter(X_pca[i, 0], X_pca[i, 1], c=color, marker=marker, s=150,
                      edgecolors='black', linewidths=1)
            ax.annotate(token, (X_pca[i, 0], X_pca[i, 1]), fontsize=8,
                       ha='center', va='bottom')

        ax.set_xlabel("PC1", fontsize=10)
        ax.set_ylabel("PC2", fontsize=10)
        title = f"{'D' if idx == 0 else 'E'}. PCA at N={ctx_len}\n(○=Super_A, □=Super_B)"
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 6: Summary
    # =========================================================================
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    # Find transition points
    super_emergence = None
    mid_emergence = None
    for i, ctx in enumerate(context_lengths[1:], 1):
        if ratio_super[i] > ratio_super[0] * 1.5 and super_emergence is None:
            super_emergence = ctx
        if ratio_mid[i] > ratio_mid[0] * 1.2 and mid_emergence is None:
            mid_emergence = ctx

    summary_text = f"""
    STAGEWISE LEARNING RESULTS
    ━━━━━━━━━━━━━━━━━━━━━━━━━━

    LEE ET AL. PREDICTION:
    ──────────────────────
    Different hierarchy levels should
    emerge at different context lengths:

    1. Super-cluster distinction first
    2. Mid-cluster distinction later

    OUR FINDINGS:
    ──────────────────────
    • Super-cluster emergence: ~N={super_emergence or 'gradual'}
    • Mid-cluster emergence: ~N={mid_emergence or 'gradual'}

    Distance at N={context_lengths[-1]}:
    • Within-mid: {distances['within_mid'][-1]:.1f}
    • Within-super: {distances['within_super'][-1]:.1f}
    • Across-super: {distances['across_super'][-1]:.1f}

    Ratio increase from N={context_lengths[0]} to N={context_lengths[-1]}:
    • Super/Mid ratio: {ratio_super[0]:.2f} → {ratio_super[-1]:.2f}
    """

    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))
    ax6.set_title("F. Summary", fontsize=12, fontweight='bold')

    plt.tight_layout()
    fig.suptitle("Experiment 1: 3-Level Hierarchical Graph\n" +
                 "Replicating Lee et al. Stagewise Learning",
                 fontsize=14, fontweight='bold', y=1.02)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "hierarchy_3level_experiment.png", dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)

    return ratio_super, ratio_mid


def create_semantic_conflict_figure(token_reps, graph, context_lengths, similarities, output_dir):
    """Create visualization for semantic conflict experiment."""
    fig = plt.figure(figsize=(18, 14))

    cmap = plt.cm.viridis
    norm = Normalize(vmin=min(context_lengths), vmax=max(context_lengths))

    # Colors for graph clusters
    graph_colors = {0: '#e41a1c', 1: '#377eb8', 2: '#4daf4a'}
    # Colors for semantic groups
    semantic_colors = {
        'animals': '#ff7f0e',
        'electronics': '#9467bd',
        'vegetables': '#8c564b',
        'furniture': '#17becf'
    }

    # =========================================================================
    # Panel 1: Experiment setup
    # =========================================================================
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.axis('off')

    setup_text = """
    SEMANTIC CONFLICT EXPERIMENT
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    SEMANTIC GROUPS (pretraining knowledge):
    ────────────────────────────────────────
    • Animals:     cat,    dog,        bird
    • Electronics: computer, television, radio
    • Vegetables:  tomato,  potato,     carrot
    • Furniture:   chair,   table,      desk

    GRAPH CLUSTERS (task structure):
    ────────────────────────────────────────
    • Cluster A: cat, computer, tomato, chair
    • Cluster B: dog, television, potato, table
    • Cluster C: bird, radio, carrot, desk

    KEY CONFLICT:
    ────────────────────────────────────────
    cat & dog are semantically similar (animals)
    but in DIFFERENT graph clusters (A vs B)

    QUESTION: At what context length does the
    model forget "cat ≈ dog" and learn
    "cat ≈ computer" (same graph cluster)?
    """

    ax1.text(0.02, 0.98, setup_text, transform=ax1.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    ax1.set_title("A. Experiment Setup", fontsize=12, fontweight='bold')

    # =========================================================================
    # Panel 2: Semantic vs Graph similarity over context
    # =========================================================================
    ax2 = fig.add_subplot(2, 3, 2)

    ax2.plot(context_lengths, similarities["semantic_same_graph_diff"], 'o-',
             color='red', linewidth=2.5, markersize=8,
             label='Semantic Similar, Graph Different\n(e.g., cat-dog)')
    ax2.plot(context_lengths, similarities["semantic_diff_graph_same"], 's-',
             color='blue', linewidth=2.5, markersize=8,
             label='Semantic Different, Graph Same\n(e.g., cat-computer)')

    # Find crossover point
    crossover = None
    for i, ctx in enumerate(context_lengths):
        if similarities["semantic_diff_graph_same"][i] > similarities["semantic_same_graph_diff"][i]:
            crossover = ctx
            break

    if crossover:
        ax2.axvline(x=crossover, color='green', linestyle='--', linewidth=2,
                   label=f'Crossover at N={crossover}')

    ax2.set_xlabel("Context Length (N)", fontsize=11)
    ax2.set_ylabel("Cosine Similarity", fontsize=11)
    ax2.set_title("B. When Does Graph Override Semantics?", fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8, loc='best')
    ax2.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 3: Similarity ratio
    # =========================================================================
    ax3 = fig.add_subplot(2, 3, 3)

    ratio = [g/s if s > 0 else 0 for g, s in
             zip(similarities["semantic_diff_graph_same"],
                 similarities["semantic_same_graph_diff"])]

    ax3.plot(context_lengths, ratio, 'o-', color='purple', linewidth=2.5, markersize=8)
    ax3.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='Equality line')
    ax3.fill_between(context_lengths, ratio, 1, where=[r > 1 for r in ratio],
                     color='blue', alpha=0.2, label='Graph > Semantic')
    ax3.fill_between(context_lengths, ratio, 1, where=[r <= 1 for r in ratio],
                     color='red', alpha=0.2, label='Semantic > Graph')

    ax3.set_xlabel("Context Length (N)", fontsize=11)
    ax3.set_ylabel("Graph Similarity / Semantic Similarity", fontsize=11)
    ax3.set_title("C. Structure Dominance Ratio", fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 4-5: PCA at early vs late context
    # =========================================================================
    for idx, ctx_len in enumerate([context_lengths[0], context_lengths[-1]]):
        ax = fig.add_subplot(2, 3, 4 + idx)

        reps = token_reps[ctx_len]
        tokens = list(reps.keys())
        X = np.array([reps[t] for t in tokens])

        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)

        # Plot by GRAPH cluster (color) and SEMANTIC group (marker)
        markers = {'animals': 'o', 'electronics': 's', 'vegetables': '^', 'furniture': 'D'}

        for i, token in enumerate(tokens):
            graph_cluster = graph.token_to_graph_cluster[token]
            semantic_group = graph.token_to_semantic_group[token]

            ax.scatter(X_pca[i, 0], X_pca[i, 1],
                      c=graph_colors[graph_cluster],
                      marker=markers[semantic_group],
                      s=150, edgecolors='black', linewidths=1)
            ax.annotate(token, (X_pca[i, 0], X_pca[i, 1]), fontsize=7,
                       ha='center', va='bottom')

        # Add legend
        if idx == 0:
            for gc, color in graph_colors.items():
                ax.scatter([], [], c=color, s=60, label=f'Graph Cluster {["A","B","C"][gc]}')
            ax.legend(loc='upper left', fontsize=7, title='Color = Graph')

        ax.set_xlabel("PC1", fontsize=10)
        ax.set_ylabel("PC2", fontsize=10)
        title = f"{'D' if idx == 0 else 'E'}. PCA at N={ctx_len}\n(Color=Graph, Shape=Semantic)"
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 6: Summary
    # =========================================================================
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    summary_text = f"""
    SEMANTIC OVERRIDE RESULTS
    ━━━━━━━━━━━━━━━━━━━━━━━━━

    QUESTION: When does the model "forget"
    pretraining semantics and adopt task structure?

    FINDINGS:
    ──────────────────────────────────
    At N={context_lengths[0]} (early context):
    • Semantic-similar pairs: {similarities['semantic_same_graph_diff'][0]:.3f}
    • Graph-similar pairs:    {similarities['semantic_diff_graph_same'][0]:.3f}
    • Ratio: {ratio[0]:.2f} ({"Semantic" if ratio[0] < 1 else "Graph"} dominates)

    At N={context_lengths[-1]} (late context):
    • Semantic-similar pairs: {similarities['semantic_same_graph_diff'][-1]:.3f}
    • Graph-similar pairs:    {similarities['semantic_diff_graph_same'][-1]:.3f}
    • Ratio: {ratio[-1]:.2f} ({"Semantic" if ratio[-1] < 1 else "Graph"} dominates)

    CROSSOVER POINT: {f"N={crossover}" if crossover else "Not reached"}

    INTERPRETATION:
    ──────────────────────────────────
    {"✓ Graph structure overrides semantics!" if ratio[-1] > 1 else "✗ Semantic structure persists"}
    {"  Model learns task > pretraining" if ratio[-1] > 1 else "  Pretraining knowledge is strong"}
    """

    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))
    ax6.set_title("F. Summary", fontsize=12, fontweight='bold')

    plt.tight_layout()
    fig.suptitle("Experiment 2: Semantic Conflict\n" +
                 "When Does Task Structure Override Pretraining Semantics?",
                 fontsize=14, fontweight='bold', y=1.02)

    output_dir = Path(output_dir)
    fig.savefig(output_dir / "semantic_conflict_experiment.png", dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)

    return crossover, ratio


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("HIERARCHY AND SEMANTIC CONFLICT EXPERIMENTS")
    print("=" * 70)

    # Configuration
    model_name = "meta-llama/Llama-3.1-8B"
    context_lengths = [5, 10, 15, 20, 30, 50, 75, 100]
    n_samples = 50
    layer_idx = -5
    output_dir = Path("results/hierarchy_semantic_experiments")
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

    # =========================================================================
    # EXPERIMENT 1: 3-Level Hierarchy
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: 3-Level Hierarchical Graph")
    print("=" * 70)

    graph_hier = HierarchicalGraph3Level(seed=42)
    print("\nHierarchy structure:")
    hierarchy = graph_hier.get_hierarchy_labels()
    for node in range(graph_hier.num_nodes):
        token = graph_hier.node_to_token[node]
        super_c = hierarchy["super_names"][hierarchy["level1"][node]]
        mid_c = hierarchy["mid_names"][hierarchy["level2"][node]]
        print(f"  {token}: {super_c} → {mid_c}")

    print("\nCollecting representations...")
    token_reps_hier = collect_representations(
        model, tokenizer, graph_hier, context_lengths, n_samples, layer_idx
    )

    print("\nAnalyzing hierarchy distances...")
    distances = compute_hierarchy_distances(token_reps_hier, graph_hier, context_lengths)

    print("\nCreating hierarchy figure...")
    ratio_super, ratio_mid = create_hierarchy_figure(
        token_reps_hier, graph_hier, context_lengths, distances, output_dir
    )

    # =========================================================================
    # EXPERIMENT 2: Semantic Conflict
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Semantic Conflict")
    print("=" * 70)

    graph_sem = SemanticConflictGraph(seed=42)
    print("\nSemantic pairs in different graph clusters:")
    for t1, t2, group in graph_sem.get_semantic_pairs():
        c1 = ["A", "B", "C"][graph_sem.token_to_graph_cluster[t1]]
        c2 = ["A", "B", "C"][graph_sem.token_to_graph_cluster[t2]]
        print(f"  {t1} (Cluster {c1}) - {t2} (Cluster {c2}) [{group}]")

    print("\nCollecting representations...")
    token_reps_sem = collect_representations(
        model, tokenizer, graph_sem, context_lengths, n_samples, layer_idx
    )

    print("\nAnalyzing semantic vs graph similarity...")
    similarities = compute_semantic_vs_graph_similarity(
        token_reps_sem, graph_sem, context_lengths
    )

    print("\nCreating semantic conflict figure...")
    crossover, ratio = create_semantic_conflict_figure(
        token_reps_sem, graph_sem, context_lengths, similarities, output_dir
    )

    # =========================================================================
    # Save Results
    # =========================================================================
    results = {
        "experiment1_hierarchy": {
            "distances": distances,
            "ratio_super": ratio_super,
            "ratio_mid": ratio_mid,
        },
        "experiment2_semantic": {
            "similarities": similarities,
            "crossover_point": crossover,
            "ratio": ratio,
        },
        "config": {
            "model": model_name,
            "context_lengths": context_lengths,
            "n_samples": n_samples,
            "layer_idx": layer_idx,
        }
    }

    with open(output_dir / "experiment_results.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)

    print("\n" + "=" * 70)
    print("EXPERIMENTS COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - hierarchy_3level_experiment.png")
    print(f"  - semantic_conflict_experiment.png")
    print(f"  - experiment_results.json")

    return results


if __name__ == "__main__":
    main()
