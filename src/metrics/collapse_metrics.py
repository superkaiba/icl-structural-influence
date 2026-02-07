"""
Collapse Metrics for Measuring Representational Convergence Over Long Contexts.

This module provides metrics for detecting and quantifying representational
collapse - the phenomenon where token representations converge to similar points
as context grows, potentially indicating "context rot" or information saturation.

Core Metrics:
- Average Cosine Similarity: Increases toward 1.0 as representations collapse
- Spread (Total Variance): Decreases toward 0 as representations collapse
- Mean Pairwise L2 Distance: Decreases as representations converge
- Effective Dimension (Participation Ratio): Decreases as variance concentrates

References:
- Park et al. (2024) arXiv:2501.00070 "ICLR: In-Context Learning of Representations"
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
import warnings


@dataclass
class CollapseMetrics:
    """Container for collapse-related metrics computed over a window of representations."""

    avg_cos_sim: float  # Mean pairwise cosine similarity [0, 1]
    avg_l2_dist: float  # Mean pairwise L2 distance
    spread: float       # Total variance (trace of covariance)
    effective_dim: float  # Participation ratio (linear)
    intrinsic_dim: Optional[float] = None  # Two-NN estimator (nonlinear)

    # Optional diagnostic metrics
    centroid_norm: Optional[float] = None  # Norm of the centroid
    max_pairwise_dist: Optional[float] = None  # Maximum pairwise distance
    min_pairwise_dist: Optional[float] = None  # Minimum pairwise distance

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "avg_cos_sim": self.avg_cos_sim,
            "avg_l2_dist": self.avg_l2_dist,
            "spread": self.spread,
            "effective_dim": self.effective_dim,
            "intrinsic_dim": self.intrinsic_dim,
            "centroid_norm": self.centroid_norm,
            "max_pairwise_dist": self.max_pairwise_dist,
            "min_pairwise_dist": self.min_pairwise_dist,
        }


def _compute_intrinsic_dim_twonn(reps: np.ndarray) -> Optional[float]:
    """
    Compute intrinsic dimension using Two-NN estimator (Facco et al., 2017).

    This is a nonlinear estimator that uses the ratio of distances to the
    2nd and 1st nearest neighbors. More accurate than PCA-based methods
    when data lies on a curved manifold.

    Falls back to None if skdim is not installed or computation fails.
    """
    if len(reps) < 10:
        return None

    try:
        from skdim.id import TwoNN
        estimator = TwoNN()
        return float(estimator.fit_transform(reps))
    except ImportError:
        # skdim not installed - compute manually using the Two-NN formula
        # ID = n / Σ log(r2/r1) where r1, r2 are distances to 1st, 2nd NN
        try:
            from scipy.spatial.distance import cdist

            # Compute pairwise distances
            dists = cdist(reps, reps)
            np.fill_diagonal(dists, np.inf)  # Exclude self

            # For each point, get distances to 1st and 2nd nearest neighbors
            sorted_dists = np.sort(dists, axis=1)
            r1 = sorted_dists[:, 0]  # Distance to 1st NN
            r2 = sorted_dists[:, 1]  # Distance to 2nd NN

            # Filter out zero distances (identical points)
            valid = (r1 > 1e-10) & (r2 > 1e-10)
            if valid.sum() < 5:
                return None

            r1, r2 = r1[valid], r2[valid]

            # Two-NN estimator: d = n / Σ log(r2/r1)
            mu = r2 / r1
            # MLE for Pareto distribution
            intrinsic_dim = len(mu) / np.sum(np.log(mu))

            return float(intrinsic_dim)
        except Exception:
            return None
    except Exception:
        return None


def compute_collapse_metrics(
    representations: list[np.ndarray],
    compute_diagnostics: bool = False,
) -> CollapseMetrics:
    """
    Compute all collapse-related metrics for a window of representations.

    These metrics quantify how "collapsed" or converged a set of representations
    are. Higher avg_cos_sim, lower spread, lower avg_l2_dist, and lower
    effective_dim all indicate more collapse.

    Args:
        representations: List of representation vectors (numpy arrays)
        compute_diagnostics: Whether to compute optional diagnostic metrics

    Returns:
        CollapseMetrics dataclass with all computed values
    """
    if len(representations) < 2:
        return CollapseMetrics(
            avg_cos_sim=1.0,
            avg_l2_dist=0.0,
            spread=0.0,
            effective_dim=0.0,
            intrinsic_dim=None,
        )

    reps = np.array(representations)
    n = len(reps)

    # 1. Mean pairwise cosine similarity
    # Normalize representations
    norms = np.linalg.norm(reps, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)  # Avoid division by zero
    normed = reps / norms

    # Compute cosine similarity matrix
    cos_sim_matrix = normed @ normed.T

    # Extract upper triangle (excluding diagonal)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    avg_cos_sim = float(cos_sim_matrix[mask].mean()) if mask.sum() > 0 else 1.0

    # 2. Mean pairwise L2 distance
    l2_dists = []
    for i in range(n):
        for j in range(i + 1, n):
            l2_dists.append(np.linalg.norm(reps[i] - reps[j]))
    avg_l2_dist = float(np.mean(l2_dists)) if l2_dists else 0.0

    # 3. Spread (total variance)
    # Compute as trace of covariance matrix = sum of variances across dimensions
    centered = reps - reps.mean(axis=0)
    spread = float(np.var(centered, axis=0).sum())

    # 4. Effective dimension (participation ratio)
    # PR = (sum of eigenvalues)^2 / sum(eigenvalues^2)
    # This measures how many dimensions carry significant variance
    if n > 1:
        cov = np.cov(reps.T)
        # Handle 1D case
        if cov.ndim == 0:
            eigenvalues = np.array([float(cov)])
        else:
            eigenvalues = np.linalg.eigvalsh(cov)

        # Ensure non-negative (numerical stability)
        eigenvalues = np.maximum(eigenvalues, 0)
        total = eigenvalues.sum()

        if total > 1e-8:
            eff_dim = float((total ** 2) / (eigenvalues ** 2).sum())
        else:
            eff_dim = 0.0
    else:
        eff_dim = 0.0

    # 5. Intrinsic dimension (Two-NN estimator - nonlinear)
    intrinsic_dim = _compute_intrinsic_dim_twonn(reps)

    # Optional diagnostic metrics
    centroid_norm = None
    max_pairwise_dist = None
    min_pairwise_dist = None

    if compute_diagnostics:
        centroid = reps.mean(axis=0)
        centroid_norm = float(np.linalg.norm(centroid))

        if l2_dists:
            max_pairwise_dist = float(np.max(l2_dists))
            min_pairwise_dist = float(np.min(l2_dists))

    return CollapseMetrics(
        avg_cos_sim=avg_cos_sim,
        avg_l2_dist=avg_l2_dist,
        spread=spread,
        effective_dim=eff_dim,
        intrinsic_dim=intrinsic_dim,
        centroid_norm=centroid_norm,
        max_pairwise_dist=max_pairwise_dist,
        min_pairwise_dist=min_pairwise_dist,
    )


def compute_collapse_trajectory(
    representations: list[np.ndarray],
    window_size: int = 50,
    compute_diagnostics: bool = False,
) -> list[CollapseMetrics]:
    """
    Compute collapse metrics over time using a sliding window.

    At each position, computes metrics using the most recent `window_size`
    representations. This tracks how collapse evolves as context grows.

    Args:
        representations: Full list of representations at each context position
        window_size: Number of recent representations to include in each computation
        compute_diagnostics: Whether to compute optional diagnostic metrics

    Returns:
        List of CollapseMetrics, one per position (starting from position 1)
    """
    trajectory = []

    for i in range(1, len(representations)):
        # Use window of representations up to current position
        start = max(0, i - window_size + 1)
        window_reps = representations[start:i + 1]

        metrics = compute_collapse_metrics(window_reps, compute_diagnostics)
        trajectory.append(metrics)

    return trajectory


def compute_collapse_at_checkpoints(
    representations: list[np.ndarray],
    checkpoints: list[int],
    window_size: int = 50,
    compute_diagnostics: bool = False,
) -> dict[int, CollapseMetrics]:
    """
    Compute collapse metrics at specific checkpoint positions.

    This is more efficient than computing a full trajectory when only
    specific positions are of interest (e.g., for long context experiments).

    Args:
        representations: Full list of representations at each context position
        checkpoints: List of positions to compute metrics at
        window_size: Number of recent representations to include
        compute_diagnostics: Whether to compute optional diagnostic metrics

    Returns:
        Dict mapping checkpoint position to CollapseMetrics
    """
    results = {}

    for cp in checkpoints:
        if cp < 1 or cp >= len(representations):
            continue

        # Use window of representations up to checkpoint
        start = max(0, cp - window_size + 1)
        window_reps = representations[start:cp + 1]

        metrics = compute_collapse_metrics(window_reps, compute_diagnostics)
        results[cp] = metrics

    return results


def aggregate_collapse_results(
    results_by_layer: dict[int, list[dict]],
) -> dict:
    """
    Aggregate collapse results across multiple trials and layers.

    Args:
        results_by_layer: Dict mapping layer -> list of trial results.
                         Each trial result has checkpoint -> CollapseMetrics mapping.

    Returns:
        Dict with aggregated statistics per layer per checkpoint
    """
    aggregated = {}

    for layer, trials in results_by_layer.items():
        if not trials:
            continue

        # Get all checkpoints
        all_checkpoints = set()
        for trial in trials:
            all_checkpoints.update(trial.keys())

        layer_agg = {}
        for cp in sorted(all_checkpoints):
            # Collect metrics from all trials at this checkpoint
            cos_sims = []
            l2_dists = []
            spreads = []
            eff_dims = []

            for trial in trials:
                if cp in trial:
                    metrics = trial[cp]
                    if isinstance(metrics, dict):
                        cos_sims.append(metrics.get("avg_cos_sim", 0))
                        l2_dists.append(metrics.get("avg_l2_dist", 0))
                        spreads.append(metrics.get("spread", 0))
                        eff_dims.append(metrics.get("effective_dim", 0))
                    elif hasattr(metrics, "avg_cos_sim"):
                        cos_sims.append(metrics.avg_cos_sim)
                        l2_dists.append(metrics.avg_l2_dist)
                        spreads.append(metrics.spread)
                        eff_dims.append(metrics.effective_dim)

            if cos_sims:
                layer_agg[cp] = {
                    "avg_cos_sim_mean": float(np.mean(cos_sims)),
                    "avg_cos_sim_std": float(np.std(cos_sims)),
                    "avg_l2_dist_mean": float(np.mean(l2_dists)),
                    "avg_l2_dist_std": float(np.std(l2_dists)),
                    "spread_mean": float(np.mean(spreads)),
                    "spread_std": float(np.std(spreads)),
                    "effective_dim_mean": float(np.mean(eff_dims)),
                    "effective_dim_std": float(np.std(eff_dims)),
                    "n_trials": len(cos_sims),
                }

        aggregated[layer] = layer_agg

    return aggregated


def demo():
    """Demonstrate collapse metrics functionality."""
    print("=" * 60)
    print("Collapse Metrics Demo")
    print("=" * 60)

    np.random.seed(42)

    # Simulate representations that collapse over context
    hidden_dim = 64
    seq_len = 100

    # Create representations that gradually collapse to a point
    target = np.random.randn(hidden_dim)  # Collapse target

    representations = []
    for i in range(seq_len):
        # Progress toward collapse (0 = fully spread, 1 = fully collapsed)
        progress = i / seq_len

        # Start spread out, end collapsed
        noise_scale = 2.0 * (1 - progress) + 0.1 * progress
        rep = target + np.random.randn(hidden_dim) * noise_scale
        representations.append(rep)

    print("\nSimulating gradual collapse over 100 positions...")
    print("(Representations converge to a common point)")

    # Compute at specific checkpoints
    checkpoints = [10, 25, 50, 75, 99]
    results = compute_collapse_at_checkpoints(
        representations, checkpoints, window_size=20, compute_diagnostics=True
    )

    print("\nCollapse Metrics at Checkpoints:")
    print(f"{'Position':<12} {'Cos Sim':<12} {'L2 Dist':<12} {'Spread':<12} {'Eff Dim':<12}")
    print("-" * 60)

    for cp in checkpoints:
        m = results[cp]
        print(f"{cp:<12} {m.avg_cos_sim:<12.3f} {m.avg_l2_dist:<12.3f} "
              f"{m.spread:<12.3f} {m.effective_dim:<12.1f}")

    print("\nInterpretation:")
    print("- Cos Sim increases (representations becoming more similar)")
    print("- L2 Dist decreases (representations converging)")
    print("- Spread decreases (variance collapsing)")
    print("- Eff Dim decreases (fewer effective dimensions)")

    print("\n" + "=" * 60)
    print("Demo Complete")
    print("=" * 60)


if __name__ == "__main__":
    demo()
