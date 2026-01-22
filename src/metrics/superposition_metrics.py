"""
Superposition and Collapse Metrics for Hypothesis Disambiguation Experiments.

This module provides metrics for measuring:
1. Superposition: Whether representations sit "between" two hypothesis centroids
2. Collapse: How much representations shift after a disambiguating token
3. Hypothesis commitment: Which interpretation the model is leaning toward

Core metrics:
- Superposition Score: Distance from midpoint between H1 and H2 centroids
- Hypothesis Ratio: Ratio of distances to each hypothesis centroid
- Collapse Magnitude: Norm of representation shift after disambiguation
- Representation Velocity: Rate of change in representation space

References:
- Park et al. (2024) arXiv:2501.00070 "ICLR: In-Context Learning of Representations"
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class SuperpositionResult:
    """Container for superposition analysis results at a single position."""

    position: int
    representation: np.ndarray  # The actual representation vector

    # Distances to each hypothesis centroid
    dist_H1: float
    dist_H2: float

    # Derived metrics
    superposition_score: float  # Distance to midpoint
    hypothesis_ratio: float     # dist_H1 / dist_H2
    velocity: float             # Change from previous position

    # Optional cluster info
    H1_cluster: Optional[int] = None
    H2_cluster: Optional[int] = None


@dataclass
class CollapseAnalysis:
    """Results of collapse analysis around the disambiguation point."""

    disambig_position: int

    # Pre-disambiguation metrics (averaged over positions before disambig)
    pre_superposition_mean: float
    pre_superposition_std: float
    pre_ratio_mean: float
    pre_ratio_std: float

    # Post-disambiguation metrics
    post_superposition_mean: float
    post_superposition_std: float
    post_ratio_mean: float
    post_ratio_std: float

    # Collapse magnitude
    collapse_distance: float          # ||rep_before - rep_after||
    velocity_at_disambig: float       # Velocity spike at disambiguation
    velocity_mean_elsewhere: float    # Average velocity elsewhere

    # Direction of collapse
    collapsed_to_correct: bool        # Did it collapse to the true hypothesis?
    final_hypothesis: str             # "H1" or "H2"


def compute_centroid(
    representations: torch.Tensor,
    cluster_labels: torch.Tensor,
    cluster_id: int,
) -> torch.Tensor:
    """
    Compute centroid of representations for a specific cluster.

    Args:
        representations: (seq_len, hidden_dim)
        cluster_labels: (seq_len,) cluster assignments
        cluster_id: Which cluster to compute centroid for

    Returns:
        Centroid tensor of shape (hidden_dim,)
    """
    mask = cluster_labels == cluster_id
    if mask.sum() == 0:
        # Return zeros if no tokens in cluster
        return torch.zeros(representations.shape[1], device=representations.device)

    cluster_reps = representations[mask]
    return cluster_reps.mean(dim=0)


def compute_all_centroids(
    representations: torch.Tensor,
    cluster_labels: torch.Tensor,
) -> dict[int, torch.Tensor]:
    """
    Compute centroids for all clusters.

    Returns:
        Dict mapping cluster_id -> centroid tensor
    """
    unique_clusters = torch.unique(cluster_labels)
    centroids = {}

    for c in unique_clusters:
        centroids[c.item()] = compute_centroid(representations, cluster_labels, c.item())

    return centroids


def compute_hypothesis_centroid(
    representations: torch.Tensor,
    cluster_labels: torch.Tensor,
) -> torch.Tensor:
    """
    Compute overall centroid for a hypothesis (mean of all tokens).

    This gives the "center" of where representations should be
    if they fully committed to this hypothesis.
    """
    return representations.mean(dim=0)


def compute_superposition_score(
    rep: torch.Tensor | np.ndarray,
    H1_centroid: torch.Tensor | np.ndarray,
    H2_centroid: torch.Tensor | np.ndarray,
) -> float:
    """
    Compute how close a representation is to the midpoint between hypotheses.

    Low score = In superposition (between both hypotheses)
    High score = Committed to one side

    Args:
        rep: Single representation vector
        H1_centroid: Centroid of H1 interpretation
        H2_centroid: Centroid of H2 interpretation

    Returns:
        Distance from midpoint (lower = more superposition)
    """
    # Convert to numpy if needed
    if isinstance(rep, torch.Tensor):
        rep = rep.float().cpu().numpy()
    if isinstance(H1_centroid, torch.Tensor):
        H1_centroid = H1_centroid.float().cpu().numpy()
    if isinstance(H2_centroid, torch.Tensor):
        H2_centroid = H2_centroid.float().cpu().numpy()

    midpoint = (H1_centroid + H2_centroid) / 2
    return float(np.linalg.norm(rep - midpoint))


def compute_hypothesis_ratio(
    rep: torch.Tensor | np.ndarray,
    H1_centroid: torch.Tensor | np.ndarray,
    H2_centroid: torch.Tensor | np.ndarray,
    eps: float = 1e-8,
) -> float:
    """
    Compute ratio of distances to each hypothesis centroid.

    Ratio = dist_H1 / dist_H2

    - Ratio = 1.0: Equal commitment to both (or equidistant)
    - Ratio > 1.0: Closer to H2 (committed to H2)
    - Ratio < 1.0: Closer to H1 (committed to H1)

    Args:
        rep: Single representation vector
        H1_centroid: Centroid of H1 interpretation
        H2_centroid: Centroid of H2 interpretation
        eps: Small constant for numerical stability

    Returns:
        Ratio of distances
    """
    if isinstance(rep, torch.Tensor):
        rep = rep.float().cpu().numpy()
    if isinstance(H1_centroid, torch.Tensor):
        H1_centroid = H1_centroid.float().cpu().numpy()
    if isinstance(H2_centroid, torch.Tensor):
        H2_centroid = H2_centroid.float().cpu().numpy()

    d1 = np.linalg.norm(rep - H1_centroid)
    d2 = np.linalg.norm(rep - H2_centroid)

    return float(d1 / (d2 + eps))


def compute_collapse_magnitude(
    rep_before: torch.Tensor | np.ndarray,
    rep_after: torch.Tensor | np.ndarray,
) -> float:
    """
    Compute how much representations shifted after disambiguation.

    Args:
        rep_before: Representation just before disambiguating token
        rep_after: Representation just after disambiguating token

    Returns:
        L2 distance between representations
    """
    if isinstance(rep_before, torch.Tensor):
        rep_before = rep_before.float().cpu().numpy()
    if isinstance(rep_after, torch.Tensor):
        rep_after = rep_after.float().cpu().numpy()

    return float(np.linalg.norm(rep_after - rep_before))


def compute_representation_velocity(
    reps: torch.Tensor | np.ndarray,
) -> np.ndarray:
    """
    Compute the velocity (rate of change) of representations over positions.

    Velocity[i] = ||rep[i+1] - rep[i]||

    Args:
        reps: Representations of shape (seq_len, hidden_dim)

    Returns:
        Array of velocities, shape (seq_len-1,)
    """
    if isinstance(reps, torch.Tensor):
        reps = reps.float().cpu().numpy()

    velocities = []
    for i in range(len(reps) - 1):
        v = np.linalg.norm(reps[i+1] - reps[i])
        velocities.append(v)

    return np.array(velocities)


def analyze_position_trajectory(
    representations: torch.Tensor | np.ndarray,
    H1_labels: torch.Tensor | np.ndarray,
    H2_labels: torch.Tensor | np.ndarray,
    H1_centroid: torch.Tensor | np.ndarray,
    H2_centroid: torch.Tensor | np.ndarray,
) -> list[SuperpositionResult]:
    """
    Analyze representation trajectory at each position.

    Args:
        representations: (seq_len, hidden_dim)
        H1_labels: Cluster labels under H1 interpretation
        H2_labels: Cluster labels under H2 interpretation
        H1_centroid: Overall centroid for H1
        H2_centroid: Overall centroid for H2

    Returns:
        List of SuperpositionResult for each position
    """
    if isinstance(representations, torch.Tensor):
        representations = representations.float().cpu().numpy()
    if isinstance(H1_labels, torch.Tensor):
        H1_labels = H1_labels.cpu().numpy()
    if isinstance(H2_labels, torch.Tensor):
        H2_labels = H2_labels.cpu().numpy()
    if isinstance(H1_centroid, torch.Tensor):
        H1_centroid = H1_centroid.float().cpu().numpy()
    if isinstance(H2_centroid, torch.Tensor):
        H2_centroid = H2_centroid.float().cpu().numpy()

    seq_len = len(representations)
    results = []

    for pos in range(seq_len):
        rep = representations[pos]

        dist_H1 = float(np.linalg.norm(rep - H1_centroid))
        dist_H2 = float(np.linalg.norm(rep - H2_centroid))

        superposition = compute_superposition_score(rep, H1_centroid, H2_centroid)
        ratio = dist_H1 / (dist_H2 + 1e-8)

        # Compute velocity
        if pos > 0:
            velocity = float(np.linalg.norm(rep - representations[pos - 1]))
        else:
            velocity = 0.0

        results.append(SuperpositionResult(
            position=pos,
            representation=rep,
            dist_H1=dist_H1,
            dist_H2=dist_H2,
            superposition_score=superposition,
            hypothesis_ratio=ratio,
            velocity=velocity,
            H1_cluster=int(H1_labels[pos]) if pos < len(H1_labels) else None,
            H2_cluster=int(H2_labels[pos]) if pos < len(H2_labels) else None,
        ))

    return results


def analyze_collapse(
    position_results: list[SuperpositionResult],
    disambig_position: int,
    true_hypothesis: str,
    window_size: int = 3,
) -> CollapseAnalysis:
    """
    Analyze collapse behavior around the disambiguation point.

    Args:
        position_results: Results from analyze_position_trajectory
        disambig_position: Position of disambiguating token
        true_hypothesis: "H1" or "H2"
        window_size: Number of positions before/after to average

    Returns:
        CollapseAnalysis with pre/post comparison
    """
    seq_len = len(position_results)

    # Pre-disambiguation metrics
    pre_start = max(0, disambig_position - window_size)
    pre_end = disambig_position
    pre_results = position_results[pre_start:pre_end]

    if pre_results:
        pre_superposition = [r.superposition_score for r in pre_results]
        pre_ratio = [r.hypothesis_ratio for r in pre_results]
        pre_superposition_mean = np.mean(pre_superposition)
        pre_superposition_std = np.std(pre_superposition)
        pre_ratio_mean = np.mean(pre_ratio)
        pre_ratio_std = np.std(pre_ratio)
    else:
        pre_superposition_mean = pre_superposition_std = 0.0
        pre_ratio_mean = pre_ratio_std = 0.0

    # Post-disambiguation metrics
    post_start = min(disambig_position + 1, seq_len)
    post_end = min(disambig_position + 1 + window_size, seq_len)
    post_results = position_results[post_start:post_end]

    if post_results:
        post_superposition = [r.superposition_score for r in post_results]
        post_ratio = [r.hypothesis_ratio for r in post_results]
        post_superposition_mean = np.mean(post_superposition)
        post_superposition_std = np.std(post_superposition)
        post_ratio_mean = np.mean(post_ratio)
        post_ratio_std = np.std(post_ratio)
    else:
        post_superposition_mean = post_superposition_std = 0.0
        post_ratio_mean = post_ratio_std = 0.0

    # Collapse distance
    if disambig_position > 0 and disambig_position < seq_len - 1:
        collapse_distance = compute_collapse_magnitude(
            position_results[disambig_position - 1].representation,
            position_results[disambig_position + 1].representation,
        )
    else:
        collapse_distance = 0.0

    # Velocity at disambiguation
    if disambig_position < seq_len:
        velocity_at_disambig = position_results[disambig_position].velocity
    else:
        velocity_at_disambig = 0.0

    # Mean velocity elsewhere
    other_velocities = [
        r.velocity for i, r in enumerate(position_results)
        if i != disambig_position and r.velocity > 0
    ]
    velocity_mean_elsewhere = np.mean(other_velocities) if other_velocities else 0.0

    # Determine final hypothesis
    final_ratio = post_ratio_mean if post_results else pre_ratio_mean
    if final_ratio < 1.0:
        final_hypothesis = "H1"
    else:
        final_hypothesis = "H2"

    collapsed_to_correct = (final_hypothesis == true_hypothesis)

    return CollapseAnalysis(
        disambig_position=disambig_position,
        pre_superposition_mean=pre_superposition_mean,
        pre_superposition_std=pre_superposition_std,
        pre_ratio_mean=pre_ratio_mean,
        pre_ratio_std=pre_ratio_std,
        post_superposition_mean=post_superposition_mean,
        post_superposition_std=post_superposition_std,
        post_ratio_mean=post_ratio_mean,
        post_ratio_std=post_ratio_std,
        collapse_distance=collapse_distance,
        velocity_at_disambig=velocity_at_disambig,
        velocity_mean_elsewhere=velocity_mean_elsewhere,
        collapsed_to_correct=collapsed_to_correct,
        final_hypothesis=final_hypothesis,
    )


def aggregate_trial_results(
    trial_results: list[dict],
) -> dict:
    """
    Aggregate results across multiple trials.

    Args:
        trial_results: List of dicts with position_results and collapse_analysis

    Returns:
        Aggregated statistics
    """
    n_trials = len(trial_results)

    # Aggregate collapse analyses
    collapse_distances = []
    velocity_spikes = []
    collapsed_correct = []

    pre_superposition = []
    post_superposition = []

    for trial in trial_results:
        if "collapse_analysis" in trial and trial["collapse_analysis"] is not None:
            ca = trial["collapse_analysis"]
            collapse_distances.append(ca.collapse_distance)
            velocity_spikes.append(ca.velocity_at_disambig / (ca.velocity_mean_elsewhere + 1e-8))
            collapsed_correct.append(1 if ca.collapsed_to_correct else 0)
            pre_superposition.append(ca.pre_superposition_mean)
            post_superposition.append(ca.post_superposition_mean)

    return {
        "n_trials": n_trials,
        "collapse_distance_mean": np.mean(collapse_distances) if collapse_distances else 0.0,
        "collapse_distance_std": np.std(collapse_distances) if collapse_distances else 0.0,
        "velocity_spike_mean": np.mean(velocity_spikes) if velocity_spikes else 0.0,
        "velocity_spike_std": np.std(velocity_spikes) if velocity_spikes else 0.0,
        "collapse_accuracy": np.mean(collapsed_correct) if collapsed_correct else 0.0,
        "pre_superposition_mean": np.mean(pre_superposition) if pre_superposition else 0.0,
        "post_superposition_mean": np.mean(post_superposition) if post_superposition else 0.0,
        "superposition_change": (
            np.mean(post_superposition) - np.mean(pre_superposition)
            if pre_superposition and post_superposition else 0.0
        ),
    }


def demo():
    """Demonstrate superposition metrics."""
    print("=" * 60)
    print("Superposition Metrics Demo")
    print("=" * 60)

    # Create synthetic data
    np.random.seed(42)
    seq_len = 20
    hidden_dim = 64

    # Create two hypothesis centroids
    H1_centroid = np.random.randn(hidden_dim)
    H2_centroid = np.random.randn(hidden_dim) + 2.0  # Offset

    print(f"\nH1-H2 centroid distance: {np.linalg.norm(H2_centroid - H1_centroid):.2f}")

    # Create a trajectory that starts in superposition and collapses at position 10
    representations = []
    disambig_pos = 10

    for i in range(seq_len):
        if i < disambig_pos:
            # Superposition: near midpoint with noise
            midpoint = (H1_centroid + H2_centroid) / 2
            rep = midpoint + np.random.randn(hidden_dim) * 0.5
        else:
            # Collapsed to H1
            rep = H1_centroid + np.random.randn(hidden_dim) * 0.3

        representations.append(rep)

    representations = np.array(representations)

    # Fake cluster labels
    H1_labels = np.array([i % 3 for i in range(seq_len)])
    H2_labels = np.array([(i + 1) % 3 for i in range(seq_len)])

    # Analyze trajectory
    print("\nAnalyzing trajectory...")
    position_results = analyze_position_trajectory(
        representations, H1_labels, H2_labels, H1_centroid, H2_centroid
    )

    print(f"\nPosition metrics (first 5):")
    for r in position_results[:5]:
        print(f"  Pos {r.position}: superposition={r.superposition_score:.2f}, "
              f"ratio={r.hypothesis_ratio:.2f}, velocity={r.velocity:.2f}")

    # Analyze collapse
    collapse = analyze_collapse(position_results, disambig_pos, "H1")

    print(f"\nCollapse Analysis:")
    print(f"  Pre-superposition: {collapse.pre_superposition_mean:.2f} +/- {collapse.pre_superposition_std:.2f}")
    print(f"  Post-superposition: {collapse.post_superposition_mean:.2f} +/- {collapse.post_superposition_std:.2f}")
    print(f"  Collapse distance: {collapse.collapse_distance:.2f}")
    print(f"  Velocity spike: {collapse.velocity_at_disambig:.2f} (vs mean {collapse.velocity_mean_elsewhere:.2f})")
    print(f"  Collapsed to correct: {collapse.collapsed_to_correct}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo()
