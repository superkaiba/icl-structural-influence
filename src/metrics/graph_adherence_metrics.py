"""
Graph Adherence Metrics for Measuring H1 vs H2 Structure in Representations.

This module provides metrics to measure how much token representations
adhere to one graph structure (H1) versus another (H2).

Metrics:
1. Dirichlet Energy Ratio - smoothness over graph edges
2. Cluster Separation Ratio - cluster quality under each interpretation
3. Linear Probe Accuracy - decodability of cluster labels
4. Neighbor Consistency - k-NN overlap with graph neighbors
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings


@dataclass
class GraphAdherenceMetrics:
    """Container for all graph adherence metrics."""

    # Dirichlet Energy
    dirichlet_h1: float  # Energy on H1 graph (lower = smoother)
    dirichlet_h2: float  # Energy on H2 graph
    dirichlet_ratio: float  # H1 / (H1 + H2), higher = more H2-adherent

    # Cluster Separation
    separation_h1: float  # Inter/intra cluster ratio for H1
    separation_h2: float  # Inter/intra cluster ratio for H2
    separation_ratio: float  # H1 / (H1 + H2), higher = more H1-adherent

    # Linear Probe
    probe_acc_h1: float  # Classification accuracy for H1 labels
    probe_acc_h2: float  # Classification accuracy for H2 labels
    probe_diff: float  # H1 - H2, positive = H1 more decodable

    # Neighbor Consistency
    neighbor_h1: float  # k-NN overlap with H1 graph neighbors
    neighbor_h2: float  # k-NN overlap with H2 graph neighbors
    neighbor_ratio: float  # H1 / (H1 + H2), higher = more H1-adherent

    def to_dict(self) -> dict:
        return {
            "dirichlet_h1": self.dirichlet_h1,
            "dirichlet_h2": self.dirichlet_h2,
            "dirichlet_ratio": self.dirichlet_ratio,
            "separation_h1": self.separation_h1,
            "separation_h2": self.separation_h2,
            "separation_ratio": self.separation_ratio,
            "probe_acc_h1": self.probe_acc_h1,
            "probe_acc_h2": self.probe_acc_h2,
            "probe_diff": self.probe_diff,
            "neighbor_h1": self.neighbor_h1,
            "neighbor_h2": self.neighbor_h2,
            "neighbor_ratio": self.neighbor_ratio,
        }


def compute_dirichlet_energy(
    representations: np.ndarray,
    adjacency: np.ndarray,
    node_indices: list[int],
    use_cosine: bool = True,
) -> float:
    """
    Compute Dirichlet energy: sum of squared distances over graph edges.

    Lower energy = representations are smoother over the graph structure.

    Args:
        representations: (n_tokens, hidden_dim) array of representations
        adjacency: (n_nodes, n_nodes) adjacency matrix of the graph
        node_indices: list mapping token position -> node index
        use_cosine: if True, use cosine distance instead of L2

    Returns:
        Dirichlet energy (lower = more adherent to graph structure)
    """
    n_tokens = len(representations)
    energy = 0.0
    n_edges = 0

    for i in range(n_tokens):
        for j in range(i + 1, n_tokens):
            node_i = node_indices[i]
            node_j = node_indices[j]

            # Check if there's an edge between these nodes
            if adjacency[node_i, node_j] > 0:
                if use_cosine:
                    # Cosine distance = 1 - cosine similarity
                    rep_i = representations[i]
                    rep_j = representations[j]
                    norm_i = np.linalg.norm(rep_i)
                    norm_j = np.linalg.norm(rep_j)
                    if norm_i > 1e-8 and norm_j > 1e-8:
                        cos_sim = np.dot(rep_i, rep_j) / (norm_i * norm_j)
                        dist_sq = (1 - cos_sim) ** 2
                    else:
                        dist_sq = 1.0
                else:
                    dist_sq = np.sum((representations[i] - representations[j]) ** 2)

                energy += dist_sq
                n_edges += 1

    # Normalize by number of edges
    if n_edges > 0:
        energy /= n_edges

    return float(energy)


def compute_cluster_separation(
    representations: np.ndarray,
    cluster_labels: np.ndarray,
    use_cosine: bool = True,
) -> float:
    """
    Compute cluster separation ratio: inter-cluster / intra-cluster distance.

    Higher ratio = better separated clusters = stronger adherence to that clustering.

    Args:
        representations: (n_tokens, hidden_dim) array
        cluster_labels: (n_tokens,) array of cluster assignments
        use_cosine: if True, use cosine distance

    Returns:
        Separation ratio (higher = better clustering)
    """
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)

    if n_clusters < 2:
        return 0.0

    def distance(rep_i, rep_j):
        if use_cosine:
            norm_i = np.linalg.norm(rep_i)
            norm_j = np.linalg.norm(rep_j)
            if norm_i > 1e-8 and norm_j > 1e-8:
                return 1 - np.dot(rep_i, rep_j) / (norm_i * norm_j)
            return 1.0
        else:
            return np.linalg.norm(rep_i - rep_j)

    # Compute intra-cluster distances
    intra_dists = []
    for c in unique_clusters:
        mask = cluster_labels == c
        cluster_reps = representations[mask]
        if len(cluster_reps) >= 2:
            for i in range(len(cluster_reps)):
                for j in range(i + 1, len(cluster_reps)):
                    intra_dists.append(distance(cluster_reps[i], cluster_reps[j]))

    # Compute inter-cluster distances (between cluster centroids)
    centroids = []
    for c in unique_clusters:
        mask = cluster_labels == c
        centroid = representations[mask].mean(axis=0)
        centroids.append(centroid)

    inter_dists = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            inter_dists.append(distance(centroids[i], centroids[j]))

    avg_intra = np.mean(intra_dists) if intra_dists else 1e-8
    avg_inter = np.mean(inter_dists) if inter_dists else 0.0

    # Avoid division by zero
    if avg_intra < 1e-8:
        avg_intra = 1e-8

    return float(avg_inter / avg_intra)


def compute_linear_probe_accuracy(
    representations: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 3,
) -> float:
    """
    Compute linear probe accuracy using cross-validation.

    Args:
        representations: (n_tokens, hidden_dim) array
        labels: (n_tokens,) array of labels
        n_splits: number of CV splits

    Returns:
        Mean classification accuracy
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.5

    # Check minimum samples per class
    min_samples = min(np.sum(labels == l) for l in unique_labels)
    if min_samples < 2:
        return 0.5

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(representations)
    y = labels

    # Simple holdout validation (faster than CV for this use case)
    n = len(X)
    accuracies = []

    for split in range(n_splits):
        # Random 70/30 split
        np.random.seed(split)
        indices = np.random.permutation(n)
        train_size = int(0.7 * n)
        train_idx = indices[:train_size]
        test_idx = indices[train_size:]

        if len(test_idx) < 2:
            continue

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Check if test set has multiple classes
        if len(np.unique(y_test)) < 2:
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf = LogisticRegression(max_iter=500, solver='lbfgs', multi_class='multinomial')
                clf.fit(X_train, y_train)
                acc = clf.score(X_test, y_test)
                accuracies.append(acc)
        except Exception:
            continue

    return float(np.mean(accuracies)) if accuracies else 0.5


def compute_neighbor_consistency(
    representations: np.ndarray,
    adjacency: np.ndarray,
    node_indices: list[int],
    k: int = 5,
) -> float:
    """
    Compute k-NN consistency with graph neighbors.

    For each token, check what fraction of its k nearest neighbors
    in representation space are actual graph neighbors.

    Args:
        representations: (n_tokens, hidden_dim) array
        adjacency: (n_nodes, n_nodes) adjacency matrix
        node_indices: list mapping token position -> node index
        k: number of nearest neighbors to consider

    Returns:
        Average overlap fraction (higher = rep geometry matches graph)
    """
    n_tokens = len(representations)
    if n_tokens < k + 1:
        k = max(1, n_tokens - 1)

    # Compute pairwise cosine similarities
    norms = np.linalg.norm(representations, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    normed = representations / norms
    sim_matrix = normed @ normed.T

    consistencies = []

    for i in range(n_tokens):
        # Get k nearest neighbors in representation space (excluding self)
        sims = sim_matrix[i].copy()
        sims[i] = -np.inf  # Exclude self
        knn_indices = np.argsort(sims)[-k:]

        # Get graph neighbors
        node_i = node_indices[i]
        graph_neighbors = set()
        for j in range(n_tokens):
            if j != i:
                node_j = node_indices[j]
                if adjacency[node_i, node_j] > 0:
                    graph_neighbors.add(j)

        if len(graph_neighbors) == 0:
            continue

        # Compute overlap
        knn_set = set(knn_indices)
        overlap = len(knn_set & graph_neighbors) / min(k, len(graph_neighbors))
        consistencies.append(overlap)

    return float(np.mean(consistencies)) if consistencies else 0.0


def compute_graph_adherence_metrics(
    representations: np.ndarray,
    adjacency_h1: np.ndarray,
    adjacency_h2: np.ndarray,
    labels_h1: np.ndarray,
    labels_h2: np.ndarray,
    node_indices: list[int],
    k_neighbors: int = 5,
    use_cosine: bool = True,
) -> GraphAdherenceMetrics:
    """
    Compute all graph adherence metrics comparing H1 vs H2.

    Args:
        representations: (n_tokens, hidden_dim) array
        adjacency_h1: H1 graph adjacency matrix
        adjacency_h2: H2 graph adjacency matrix
        labels_h1: H1 cluster labels for each token
        labels_h2: H2 cluster labels for each token
        node_indices: list mapping token position -> node index
        k_neighbors: k for neighbor consistency metric
        use_cosine: whether to use cosine distance

    Returns:
        GraphAdherenceMetrics with all computed values
    """
    # 1. Dirichlet Energy
    dirichlet_h1 = compute_dirichlet_energy(representations, adjacency_h1, node_indices, use_cosine)
    dirichlet_h2 = compute_dirichlet_energy(representations, adjacency_h2, node_indices, use_cosine)
    dirichlet_total = dirichlet_h1 + dirichlet_h2
    dirichlet_ratio = dirichlet_h1 / dirichlet_total if dirichlet_total > 1e-8 else 0.5

    # 2. Cluster Separation
    separation_h1 = compute_cluster_separation(representations, labels_h1, use_cosine)
    separation_h2 = compute_cluster_separation(representations, labels_h2, use_cosine)
    sep_total = separation_h1 + separation_h2
    separation_ratio = separation_h1 / sep_total if sep_total > 1e-8 else 0.5

    # 3. Linear Probe Accuracy
    probe_acc_h1 = compute_linear_probe_accuracy(representations, labels_h1)
    probe_acc_h2 = compute_linear_probe_accuracy(representations, labels_h2)
    probe_diff = probe_acc_h1 - probe_acc_h2

    # 4. Neighbor Consistency
    neighbor_h1 = compute_neighbor_consistency(representations, adjacency_h1, node_indices, k_neighbors)
    neighbor_h2 = compute_neighbor_consistency(representations, adjacency_h2, node_indices, k_neighbors)
    neighbor_total = neighbor_h1 + neighbor_h2
    neighbor_ratio = neighbor_h1 / neighbor_total if neighbor_total > 1e-8 else 0.5

    return GraphAdherenceMetrics(
        dirichlet_h1=dirichlet_h1,
        dirichlet_h2=dirichlet_h2,
        dirichlet_ratio=dirichlet_ratio,
        separation_h1=separation_h1,
        separation_h2=separation_h2,
        separation_ratio=separation_ratio,
        probe_acc_h1=probe_acc_h1,
        probe_acc_h2=probe_acc_h2,
        probe_diff=probe_diff,
        neighbor_h1=neighbor_h1,
        neighbor_h2=neighbor_h2,
        neighbor_ratio=neighbor_ratio,
    )


def compute_adherence_over_context(
    all_representations: list[np.ndarray],
    adjacency_h1: np.ndarray,
    adjacency_h2: np.ndarray,
    labels_h1: list[int],
    labels_h2: list[int],
    node_indices: list[int],
    window_size: int = 50,
    checkpoints: Optional[list[int]] = None,
) -> dict[int, GraphAdherenceMetrics]:
    """
    Compute graph adherence metrics at checkpoints over context.

    Args:
        all_representations: list of (hidden_dim,) arrays, one per token
        adjacency_h1, adjacency_h2: graph adjacency matrices
        labels_h1, labels_h2: cluster labels per token
        node_indices: node index per token
        window_size: number of recent tokens to use for metrics
        checkpoints: positions to compute metrics at

    Returns:
        Dict mapping checkpoint -> GraphAdherenceMetrics
    """
    results = {}
    n_tokens = len(all_representations)

    if checkpoints is None:
        checkpoints = list(range(window_size, n_tokens, window_size))

    for cp in checkpoints:
        if cp >= n_tokens:
            continue

        # Get window of representations
        start = max(0, cp - window_size + 1)
        window_reps = np.array(all_representations[start:cp + 1])
        window_labels_h1 = np.array(labels_h1[start:cp + 1])
        window_labels_h2 = np.array(labels_h2[start:cp + 1])
        window_nodes = node_indices[start:cp + 1]

        if len(window_reps) < 10:
            continue

        metrics = compute_graph_adherence_metrics(
            window_reps,
            adjacency_h1,
            adjacency_h2,
            window_labels_h1,
            window_labels_h2,
            window_nodes,
        )
        results[cp] = metrics

    return results
