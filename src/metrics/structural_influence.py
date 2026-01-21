"""
Structural Influence Metrics for ICL Representation Analysis.

This module implements Context Sensitivity Scores for In-Context Learning,
inspired by (but distinct from) Bayesian Influence Functions from Lee et al.
(2025) "Influence Dynamics and Stagewise Data Attribution".

Core Idea:
    CSS(z_i, Φ) = -Cov_{contexts}(L(z_i), Φ)

Where:
    - z_i: A specific context token
    - L(z_i): Loss of that token (next-token prediction)
    - Φ: A structural metric (e.g., Dirichlet Energy, cluster separation)
    - Cov: Covariance computed across different contexts (NOT weight samples)

IMPORTANT DISTINCTION:
    True BIF requires sampling from the weight posterior via SGLD, which is
    infeasible for large models. Our Context Sensitivity Score instead measures
    covariance across different input contexts with FROZEN weights. This is a
    correlational measure, not causal influence.

    High CSS = "When this token position has high loss, the structural metric
                tends to be different across contexts"

References:
    - Lee et al. (2025) arXiv:2510.12071 "Influence Dynamics" (theoretical basis)
    - Park et al. (2024) arXiv:2501.00070 "ICLR: In-Context Learning of Representations"
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable, Literal
from abc import ABC, abstractmethod


@dataclass
class SensitivityResult:
    """Container for context sensitivity computation results."""

    # Token-level sensitivity scores
    sensitivity_scores: torch.Tensor  # Shape: (seq_len,)

    # Raw components
    token_losses: torch.Tensor      # Shape: (seq_len,)
    structural_metric: float        # Scalar Φ value

    # Optional decomposition
    within_cluster_sensitivity: Optional[torch.Tensor] = None
    between_cluster_sensitivity: Optional[torch.Tensor] = None

    # Metadata
    layer: Optional[int] = None
    metric_name: Optional[str] = None


# =============================================================================
# Structural Metrics (Φ)
# =============================================================================

class StructuralMetric(ABC):
    """Base class for structural metrics Φ."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name identifier for this metric."""
        pass

    @abstractmethod
    def compute(
        self,
        representations: torch.Tensor,
        cluster_labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> float:
        """
        Compute the structural metric from representations.

        Args:
            representations: Tensor of shape (num_tokens, hidden_dim)
            cluster_labels: Optional cluster assignments for each token
            **kwargs: Additional metric-specific arguments

        Returns:
            Scalar metric value
        """
        pass


class DirichletEnergy(StructuralMetric):
    """
    Dirichlet Energy: Measures smoothness of representations over the graph.

    E_D = Σ_{(i,j) ∈ E} ||f(i) - f(j)||²

    Lower energy = smoother representation (nearby nodes have similar embeddings)
    Higher energy = representations respect graph structure

    This captures whether the model has learned the graph topology.
    """

    @property
    def name(self) -> str:
        return "dirichlet_energy"

    def compute(
        self,
        representations: torch.Tensor,
        cluster_labels: Optional[torch.Tensor] = None,
        adjacency_matrix: Optional[torch.Tensor] = None,
        node_sequence: Optional[list[int]] = None,
        **kwargs
    ) -> float:
        """
        Compute Dirichlet energy.

        For a random walk sequence, we compute energy over consecutive tokens
        (which are adjacent in the graph by construction).

        Args:
            representations: (seq_len, hidden_dim)
            adjacency_matrix: Optional (num_nodes, num_nodes) for full computation
            node_sequence: Node indices corresponding to each token

        Returns:
            Dirichlet energy (scalar)
        """
        if representations.dim() == 1:
            representations = representations.unsqueeze(0)

        seq_len = representations.shape[0]

        # Compute energy over consecutive tokens (random walk edges)
        energy = 0.0
        for i in range(seq_len - 1):
            diff = representations[i] - representations[i + 1]
            energy += torch.sum(diff ** 2).item()

        return energy


class ClusterSeparation(StructuralMetric):
    """
    Inter-cluster separation: Measures how well-separated cluster centroids are.

    S = mean_{i≠j} ||μ_i - μ_j||² / mean_i Var_i

    Higher = better separation between clusters relative to within-cluster variance.
    This captures whether the model has learned the super-cluster structure.
    """

    @property
    def name(self) -> str:
        return "cluster_separation"

    def compute(
        self,
        representations: torch.Tensor,
        cluster_labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> float:
        """
        Compute cluster separation metric.

        Args:
            representations: (seq_len, hidden_dim)
            cluster_labels: (seq_len,) cluster assignment for each token

        Returns:
            Separation score (scalar)
        """
        if cluster_labels is None:
            raise ValueError("cluster_labels required for ClusterSeparation metric")

        cluster_labels = torch.as_tensor(cluster_labels)
        unique_clusters = torch.unique(cluster_labels)

        if len(unique_clusters) < 2:
            return 0.0

        # Compute cluster centroids and variances
        centroids = {}
        variances = {}
        single_token_clusters = []

        # First pass: compute variances for clusters with 2+ tokens
        for c in unique_clusters:
            mask = cluster_labels == c
            if mask.sum() > 0:
                cluster_reps = representations[mask]
                centroids[c.item()] = cluster_reps.mean(dim=0)
                if mask.sum() > 1:
                    variances[c.item()] = cluster_reps.var(dim=0).mean().item()
                else:
                    single_token_clusters.append(c.item())

        # Second pass: handle single-token clusters using pooled variance
        if single_token_clusters:
            if variances:
                # Use mean variance from clusters with 2+ tokens (pooled estimate)
                pooled_var = np.mean(list(variances.values()))
            else:
                # All clusters have only 1 token - use global variance as fallback
                global_var = representations.var(dim=0).mean().item()
                pooled_var = global_var if global_var > 1e-10 else 1.0

            for c in single_token_clusters:
                variances[c] = pooled_var

        # Inter-cluster distances
        cluster_ids = list(centroids.keys())
        inter_distances = []
        for i, c1 in enumerate(cluster_ids):
            for c2 in cluster_ids[i+1:]:
                dist = torch.sum((centroids[c1] - centroids[c2]) ** 2).item()
                inter_distances.append(dist)

        if not inter_distances:
            return 0.0

        mean_inter = np.mean(inter_distances)
        mean_intra_var = np.mean(list(variances.values()))

        # Use relative epsilon to bound the ratio
        # This ensures Φ ≤ 1/eps (default max ~1000)
        # Prevents extreme outliers when intra-cluster variance is near zero
        eps = 1e-3
        min_intra_var = max(eps * mean_inter, 1e-10)
        if mean_intra_var < min_intra_var:
            mean_intra_var = min_intra_var

        return mean_inter / mean_intra_var


class LevelSpecificClusterSeparation(StructuralMetric):
    """
    Cluster separation metric computed at a specific hierarchy level.

    This allows measuring CSS separately for different hierarchy levels:
    - Level 1: Super-cluster separation (coarse structure)
    - Level 2: Mid-cluster separation (intermediate structure)
    - Level 3: Sub-cluster separation (fine structure)

    Uses the same formula as ClusterSeparation but with level-specific labels.
    """

    def __init__(self, level: int):
        """
        Args:
            level: Hierarchy level to compute separation for (1, 2, 3, ...)
        """
        self.level = level

    @property
    def name(self) -> str:
        return f"cluster_separation_level_{self.level}"

    def compute(
        self,
        representations: torch.Tensor,
        cluster_labels: Optional[torch.Tensor] = None,
        level_labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> float:
        """
        Compute cluster separation at a specific hierarchy level.

        Args:
            representations: (seq_len, hidden_dim)
            cluster_labels: Legacy parameter, ignored if level_labels provided
            level_labels: (seq_len,) cluster assignments at this hierarchy level

        Returns:
            Separation score (scalar)
        """
        # Use level_labels if provided, otherwise fall back to cluster_labels
        labels = level_labels if level_labels is not None else cluster_labels

        if labels is None:
            raise ValueError("level_labels or cluster_labels required")

        labels = torch.as_tensor(labels)
        unique_clusters = torch.unique(labels)

        if len(unique_clusters) < 2:
            return 0.0

        # Compute cluster centroids and variances
        centroids = {}
        variances = {}
        single_token_clusters = []

        for c in unique_clusters:
            mask = labels == c
            if mask.sum() > 0:
                cluster_reps = representations[mask]
                centroids[c.item()] = cluster_reps.mean(dim=0)
                if mask.sum() > 1:
                    variances[c.item()] = cluster_reps.var(dim=0).mean().item()
                else:
                    single_token_clusters.append(c.item())

        # Handle single-token clusters with pooled variance
        if single_token_clusters:
            if variances:
                pooled_var = np.mean(list(variances.values()))
            else:
                global_var = representations.var(dim=0).mean().item()
                pooled_var = global_var if global_var > 1e-10 else 1.0

            for c in single_token_clusters:
                variances[c] = pooled_var

        # Inter-cluster distances
        cluster_ids = list(centroids.keys())
        inter_distances = []
        for i, c1 in enumerate(cluster_ids):
            for c2 in cluster_ids[i+1:]:
                dist = torch.sum((centroids[c1] - centroids[c2]) ** 2).item()
                inter_distances.append(dist)

        if not inter_distances:
            return 0.0

        mean_inter = np.mean(inter_distances)
        mean_intra_var = np.mean(list(variances.values()))

        if mean_intra_var < 1e-10:
            mean_intra_var = 1e-10

        return mean_inter / mean_intra_var


class RepresentationCoherence(StructuralMetric):
    """
    Representation Coherence: Measures alignment with graph connectivity.

    Uses cosine similarity weighted by graph distance. Tokens that are
    close in the graph should have similar representations.

    C = Σ_{i,j} w_ij * cos_sim(r_i, r_j)

    where w_ij = 1 if same cluster, -1 if different cluster
    """

    @property
    def name(self) -> str:
        return "representation_coherence"

    def compute(
        self,
        representations: torch.Tensor,
        cluster_labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> float:
        """
        Compute representation coherence.

        Args:
            representations: (seq_len, hidden_dim)
            cluster_labels: (seq_len,) cluster assignment for each token

        Returns:
            Coherence score (scalar)
        """
        if cluster_labels is None:
            raise ValueError("cluster_labels required for RepresentationCoherence")

        cluster_labels = torch.as_tensor(cluster_labels)
        seq_len = representations.shape[0]

        # Normalize for cosine similarity
        norms = representations.norm(dim=1, keepdim=True)
        norms = torch.clamp(norms, min=1e-8)
        normalized = representations / norms

        # Compute pairwise cosine similarities
        sim_matrix = torch.mm(normalized, normalized.t())

        # Weight by cluster membership
        coherence = 0.0
        count = 0

        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                weight = 1.0 if cluster_labels[i] == cluster_labels[j] else -0.5
                coherence += weight * sim_matrix[i, j].item()
                count += 1

        return coherence / max(count, 1)


class LayerWiseProgress(StructuralMetric):
    """
    Layer-wise Learning Progress: Measures how much structural information
    emerges across layers.

    Computes the ratio of structure (cluster separation) at layer L compared
    to the input embedding layer.
    """

    def __init__(self, baseline_separation: float = 1.0):
        self.baseline_separation = baseline_separation
        self._cluster_sep = ClusterSeparation()

    @property
    def name(self) -> str:
        return "layerwise_progress"

    def compute(
        self,
        representations: torch.Tensor,
        cluster_labels: Optional[torch.Tensor] = None,
        baseline_representations: Optional[torch.Tensor] = None,
        **kwargs
    ) -> float:
        """
        Compute progress relative to baseline.

        Args:
            representations: Current layer representations
            cluster_labels: Cluster assignments
            baseline_representations: Input/embedding layer representations

        Returns:
            Progress ratio (>1 means improvement)
        """
        current_sep = self._cluster_sep.compute(representations, cluster_labels)

        if baseline_representations is not None:
            baseline_sep = self._cluster_sep.compute(
                baseline_representations, cluster_labels
            )
            if baseline_sep < 1e-10:
                baseline_sep = self.baseline_separation
        else:
            baseline_sep = self.baseline_separation

        return current_sep / baseline_sep


# =============================================================================
# Context Sensitivity Score (CSS)
# =============================================================================

class ContextSensitivityScore:
    """
    Context Sensitivity Score for ICL.

    Measures how token loss covaries with structural metrics across contexts:

        CSS(token_i, Φ) = -Cov_{contexts}(L(token_i), Φ)

    IMPORTANT: This is NOT true Bayesian influence (which requires weight sampling).
    Instead, we measure correlational sensitivity across different input contexts
    with frozen model weights.

    Interpretation:
        High CSS = Token positions where loss variation correlates with
                   structural metric variation across contexts.

    This tells us which token positions are most associated with structural
    changes, but does NOT establish causal influence.
    """

    def __init__(
        self,
        structural_metric: StructuralMetric,
        n_bootstrap: int = 100,
    ):
        """
        Args:
            structural_metric: The Φ metric to use
            n_bootstrap: Number of bootstrap samples for confidence intervals
        """
        self.metric = structural_metric
        self.n_bootstrap = n_bootstrap

    def compute_single_sample(
        self,
        token_losses: torch.Tensor,
        representations: torch.Tensor,
        cluster_labels: Optional[torch.Tensor] = None,
        **metric_kwargs
    ) -> SensitivityResult:
        """
        Compute sensitivity scores for a single context.

        This provides a local sensitivity estimate without cross-context
        covariance. Use compute_batch for the full CSS computation.

        Args:
            token_losses: Per-token losses, shape (seq_len,)
            representations: Token representations, shape (seq_len, hidden_dim)
            cluster_labels: Optional cluster assignments

        Returns:
            SensitivityResult with per-token sensitivity scores
        """
        seq_len = token_losses.shape[0]

        # Compute structural metric
        phi = self.metric.compute(
            representations,
            cluster_labels=cluster_labels,
            **metric_kwargs
        )

        # Local sensitivity: normalized loss as proxy
        # For proper CSS, use compute_batch with multiple contexts

        # Normalize losses
        losses_centered = token_losses - token_losses.mean()
        losses_std = losses_centered.std()
        if losses_std > 1e-8:
            losses_normalized = losses_centered / losses_std
        else:
            losses_normalized = losses_centered

        # Higher loss = less predictable = potentially more structurally important
        sensitivity_scores = -losses_normalized

        return SensitivityResult(
            sensitivity_scores=sensitivity_scores,
            token_losses=token_losses,
            structural_metric=phi,
            metric_name=self.metric.name,
        )

    def compute_batch(
        self,
        all_token_losses: list[torch.Tensor],
        all_representations: list[torch.Tensor],
        all_cluster_labels: Optional[list[torch.Tensor]] = None,
        **metric_kwargs
    ) -> dict:
        """
        Compute Context Sensitivity Scores across multiple contexts.

        This is the core CSS computation: estimate covariance between
        token losses and structural metric across different contexts.

        Args:
            all_token_losses: List of per-token loss tensors (one per context)
            all_representations: List of representation tensors (one per context)
            all_cluster_labels: Optional list of cluster assignments

        Returns:
            Dict with sensitivity analysis results
        """
        n_samples = len(all_token_losses)

        if n_samples < 2:
            raise ValueError("Need at least 2 samples for covariance estimation")

        # Compute structural metric for each context
        phi_values = []
        for i, reps in enumerate(all_representations):
            clusters = all_cluster_labels[i] if all_cluster_labels else None
            phi = self.metric.compute(reps, cluster_labels=clusters, **metric_kwargs)
            phi_values.append(phi)

        phi_values = np.array(phi_values)
        phi_mean = phi_values.mean()
        phi_centered = phi_values - phi_mean

        # Find common sequence length (or handle variable lengths)
        seq_lengths = [losses.shape[0] for losses in all_token_losses]
        min_len = min(seq_lengths)

        # Compute covariance for each token position
        position_sensitivities = []

        for pos in range(min_len):
            # Gather losses at this position across contexts
            position_losses = np.array([
                losses[pos].item() if pos < len(losses) else 0.0
                for losses in all_token_losses
            ])

            loss_centered = position_losses - position_losses.mean()

            # CSS = -Cov(L, Φ) across contexts
            covariance = np.mean(loss_centered * phi_centered)
            css = -covariance

            position_sensitivities.append(css)

        return {
            "position_sensitivities": np.array(position_sensitivities),
            "phi_values": phi_values,
            "phi_mean": phi_mean,
            "phi_std": phi_values.std(),
            "n_contexts": n_samples,
            "metric_name": self.metric.name,
        }

    def compute_hierarchical_decomposition(
        self,
        all_token_losses: list[torch.Tensor],
        all_representations: list[torch.Tensor],
        all_cluster_labels: list[torch.Tensor],
        **metric_kwargs
    ) -> dict:
        """
        Decompose sensitivity into within-cluster and between-cluster components.

        This tests the hierarchical learning hypothesis: are token positions
        differentially sensitive to global structure (between-cluster) vs
        local structure (within-cluster)?

        Returns:
            Dict with decomposed sensitivity scores
        """
        n_samples = len(all_token_losses)

        # Separate metrics for within vs between cluster
        within_metric = RepresentationCoherence()
        between_metric = ClusterSeparation()

        within_phi = []
        between_phi = []

        for i, reps in enumerate(all_representations):
            clusters = all_cluster_labels[i]
            within_phi.append(within_metric.compute(reps, cluster_labels=clusters))
            between_phi.append(between_metric.compute(reps, cluster_labels=clusters))

        within_phi = np.array(within_phi)
        between_phi = np.array(between_phi)

        # Compute CSS for each component
        seq_lengths = [losses.shape[0] for losses in all_token_losses]
        min_len = min(seq_lengths)

        within_sensitivities = []
        between_sensitivities = []

        within_centered = within_phi - within_phi.mean()
        between_centered = between_phi - between_phi.mean()

        for pos in range(min_len):
            position_losses = np.array([
                losses[pos].item() if pos < len(losses) else 0.0
                for losses in all_token_losses
            ])
            loss_centered = position_losses - position_losses.mean()

            within_sensitivities.append(-np.mean(loss_centered * within_centered))
            between_sensitivities.append(-np.mean(loss_centered * between_centered))

        return {
            "within_cluster_sensitivities": np.array(within_sensitivities),
            "between_cluster_sensitivities": np.array(between_sensitivities),
            "within_phi_values": within_phi,
            "between_phi_values": between_phi,
            "n_contexts": n_samples,
        }

    def compute_multilevel_decomposition(
        self,
        all_token_losses: list[torch.Tensor],
        all_representations: list[torch.Tensor],
        level_labels: dict[int, list[torch.Tensor]],
        num_levels: int = 3,
        **metric_kwargs
    ) -> dict:
        """
        Decompose sensitivity scores by hierarchy level for N-level hierarchies.

        This extends compute_hierarchical_decomposition to handle arbitrary
        hierarchy depths, computing separate CSS scores for each level.

        Args:
            all_token_losses: List of per-token loss tensors (one per context)
            all_representations: List of representation tensors (one per context)
            level_labels: Dict mapping level (1, 2, 3, ...) to list of label
                          tensors (one per context). Level 0 (root) is skipped
                          as all nodes belong to the same cluster.
            num_levels: Number of hierarchy levels (excluding root)

        Returns:
            Dict with keys:
            - 'css_level_{i}': Per-position CSS array for level i
            - 'phi_level_{i}': Phi values across contexts for level i
            - 'phi_mean_level_{i}': Mean Phi for level i
            - 'phi_std_level_{i}': Std Phi for level i
            - 'n_contexts': Number of contexts used
        """
        n_samples = len(all_token_losses)

        if n_samples < 2:
            raise ValueError("Need at least 2 samples for covariance estimation")

        results = {'n_contexts': n_samples}

        # Find common sequence length
        seq_lengths = [losses.shape[0] for losses in all_token_losses]
        min_len = min(seq_lengths)

        # Compute CSS for each hierarchy level
        for level in range(1, num_levels + 1):
            if level not in level_labels:
                continue

            # Create metric for this level
            metric = LevelSpecificClusterSeparation(level)

            # Compute Phi for each context at this level
            phi_values = []
            for i, reps in enumerate(all_representations):
                labels = level_labels[level][i]
                phi = metric.compute(reps, level_labels=labels)
                phi_values.append(phi)

            phi_values = np.array(phi_values)
            phi_centered = phi_values - phi_values.mean()

            # Compute CSS for each position
            position_css = []
            for pos in range(min_len):
                position_losses = np.array([
                    losses[pos].item() if pos < len(losses) else 0.0
                    for losses in all_token_losses
                ])
                loss_centered = position_losses - position_losses.mean()

                # CSS = -Cov(L, Φ)
                css = -np.mean(loss_centered * phi_centered)
                position_css.append(css)

            results[f'css_level_{level}'] = np.array(position_css)
            results[f'phi_level_{level}'] = phi_values
            results[f'phi_mean_level_{level}'] = float(phi_values.mean())
            results[f'phi_std_level_{level}'] = float(phi_values.std())

        return results


def compute_levelwise_phi_trajectory(
    representations_by_context_length: dict[int, list[torch.Tensor]],
    level_labels_by_context_length: dict[int, dict[int, list[torch.Tensor]]],
    context_lengths: list[int],
    num_levels: int = 3,
) -> dict:
    """
    Track how Phi (cluster separation) evolves across context lengths at each level.

    This is a standalone function for computing Phi trajectories without CSS,
    useful for the multi-line plot visualization.

    Args:
        representations_by_context_length: Dict mapping context length to list
            of representation tensors
        level_labels_by_context_length: Dict mapping context length to dict of
            {level: list of label tensors}
        context_lengths: List of context lengths to analyze
        num_levels: Number of hierarchy levels (excluding root)

    Returns:
        Dict with keys:
        - 'context_lengths': List of context lengths
        - 'phi_trajectory_level_{i}': List of mean Phi values at each context length
        - 'phi_std_level_{i}': List of Phi std values at each context length
    """
    results = {'context_lengths': context_lengths}

    for level in range(1, num_levels + 1):
        phi_means = []
        phi_stds = []

        for ctx_len in context_lengths:
            if ctx_len not in representations_by_context_length:
                phi_means.append(np.nan)
                phi_stds.append(np.nan)
                continue

            reps_list = representations_by_context_length[ctx_len]
            labels_dict = level_labels_by_context_length.get(ctx_len, {})

            if level not in labels_dict:
                phi_means.append(np.nan)
                phi_stds.append(np.nan)
                continue

            labels_list = labels_dict[level]

            # Compute Phi for each sample at this context length
            metric = LevelSpecificClusterSeparation(level)
            phi_values = []

            for reps, labels in zip(reps_list, labels_list):
                try:
                    phi = metric.compute(reps, level_labels=labels)
                    phi_values.append(phi)
                except Exception:
                    continue

            if phi_values:
                phi_means.append(np.mean(phi_values))
                phi_stds.append(np.std(phi_values))
            else:
                phi_means.append(np.nan)
                phi_stds.append(np.nan)

        results[f'phi_trajectory_level_{level}'] = phi_means
        results[f'phi_std_level_{level}'] = phi_stds

    return results


# =============================================================================
# High-level Experiment Interface
# =============================================================================

class ContextSensitivityExperiment:
    """
    High-level interface for running context sensitivity experiments.

    Coordinates between:
    - HierarchicalGraph (data generation)
    - HookedLLM (model and activation extraction)
    - ContextSensitivityScore (sensitivity computation)
    """

    def __init__(
        self,
        model,  # HookedLLM instance
        graph,  # HierarchicalGraph instance
        layers: list[int],
        metrics: Optional[list[StructuralMetric]] = None,
    ):
        """
        Args:
            model: HookedLLM for activation extraction
            graph: HierarchicalGraph for data generation
            layers: Which layers to analyze
            metrics: Structural metrics to compute (default: all)
        """
        self.model = model
        self.graph = graph
        self.layers = layers

        if metrics is None:
            self.metrics = [
                DirichletEnergy(),
                ClusterSeparation(),
                RepresentationCoherence(),
            ]
        else:
            self.metrics = metrics

    def run_single_walk(
        self,
        walk_length: Optional[int] = None
    ) -> dict:
        """
        Run analysis on a single random walk.

        Returns:
            Dict with per-layer, per-metric sensitivity results
        """
        prompt, nodes = self.graph.generate_random_walk(
            length=walk_length,
            return_nodes=True
        )
        clusters = torch.tensor([self.graph.get_cluster(n) for n in nodes])

        # Get losses and representations
        token_losses = self.model.compute_per_token_loss(prompt)
        _, cache = self.model.forward_with_cache(prompt, layers=self.layers)

        results = {"prompt": prompt, "nodes": nodes, "clusters": clusters.tolist()}

        for layer in self.layers:
            residual = cache.get_residual_stream(layer)
            if residual is None:
                continue

            # Remove batch dimension
            reps = residual.squeeze(0)

            layer_results = {}
            for metric in self.metrics:
                css = ContextSensitivityScore(metric)
                # Note: losses are shifted by 1 (next-token prediction)
                # Align with representations
                sensitivity = css.compute_single_sample(
                    token_losses.squeeze(0),
                    reps[:-1],  # Exclude last token (no loss for it)
                    cluster_labels=clusters[1:]  # Align with losses
                )
                layer_results[metric.name] = {
                    "sensitivity_scores": sensitivity.sensitivity_scores.tolist(),
                    "structural_metric": sensitivity.structural_metric,
                }

            results[f"layer_{layer}"] = layer_results

        return results

    def run_batch_analysis(
        self,
        n_contexts: int = 100,
        walk_length: Optional[int] = None
    ) -> dict:
        """
        Run full CSS analysis across multiple contexts.

        This computes the covariance-based sensitivity properly.

        Returns:
            Dict with aggregated sensitivity results
        """
        all_losses = []
        all_representations = {layer: [] for layer in self.layers}
        all_clusters = []

        for _ in range(n_contexts):
            prompt, nodes = self.graph.generate_random_walk(
                length=walk_length,
                return_nodes=True
            )
            clusters = torch.tensor([self.graph.get_cluster(n) for n in nodes])

            token_losses = self.model.compute_per_token_loss(prompt)
            _, cache = self.model.forward_with_cache(prompt, layers=self.layers)

            all_losses.append(token_losses.squeeze(0))
            all_clusters.append(clusters[1:])  # Align with losses

            for layer in self.layers:
                residual = cache.get_residual_stream(layer)
                if residual is not None:
                    all_representations[layer].append(residual.squeeze(0)[:-1])

        # Compute CSS for each layer and metric
        results = {"n_contexts": n_contexts}

        for layer in self.layers:
            layer_results = {}

            for metric in self.metrics:
                css = ContextSensitivityScore(metric)
                sensitivity_result = css.compute_batch(
                    all_losses,
                    all_representations[layer],
                    all_clusters
                )
                layer_results[metric.name] = sensitivity_result

                # Also compute hierarchical decomposition
                if metric.name == "cluster_separation":
                    decomp = css.compute_hierarchical_decomposition(
                        all_losses,
                        all_representations[layer],
                        all_clusters
                    )
                    layer_results["hierarchical_decomposition"] = decomp

            results[f"layer_{layer}"] = layer_results

        return results


# Backwards compatibility alias
StructuralInfluenceExperiment = ContextSensitivityExperiment


def demo():
    """Demonstrate the context sensitivity metrics."""
    print("=" * 60)
    print("Context Sensitivity Score (CSS) Demo")
    print("=" * 60)

    # Create synthetic data
    torch.manual_seed(42)
    seq_len = 20
    hidden_dim = 64
    n_clusters = 3

    # Random representations
    representations = torch.randn(seq_len, hidden_dim)

    # Cluster labels (cyclic)
    cluster_labels = torch.tensor([i % n_clusters for i in range(seq_len)])

    # Test each metric
    metrics = [
        DirichletEnergy(),
        ClusterSeparation(),
        RepresentationCoherence(),
    ]

    print("\nStructural Metrics on Random Data:")
    print("-" * 40)

    for metric in metrics:
        try:
            value = metric.compute(representations, cluster_labels=cluster_labels)
            print(f"  {metric.name}: {value:.4f}")
        except Exception as e:
            print(f"  {metric.name}: Error - {e}")

    # Test CSS computation
    print("\n" + "-" * 40)
    print("Context Sensitivity Score Test:")
    print("-" * 40)

    # Generate multiple contexts
    n_contexts = 10
    all_losses = [torch.rand(seq_len) for _ in range(n_contexts)]
    all_reps = [torch.randn(seq_len, hidden_dim) for _ in range(n_contexts)]
    all_clusters = [cluster_labels for _ in range(n_contexts)]

    css = ContextSensitivityScore(ClusterSeparation())
    result = css.compute_batch(all_losses, all_reps, all_clusters)

    print(f"  Contexts: {result['n_contexts']}")
    print(f"  Φ mean: {result['phi_mean']:.4f}")
    print(f"  Φ std: {result['phi_std']:.4f}")
    print(f"  Position sensitivities shape: {result['position_sensitivities'].shape}")
    print(f"  Max sensitivity position: {np.argmax(np.abs(result['position_sensitivities']))}")

    # Test hierarchical decomposition
    decomp = css.compute_hierarchical_decomposition(all_losses, all_reps, all_clusters)
    print("\nHierarchical Decomposition:")
    print(f"  Within-cluster Φ mean: {decomp['within_phi_values'].mean():.4f}")
    print(f"  Between-cluster Φ mean: {decomp['between_phi_values'].mean():.4f}")

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
