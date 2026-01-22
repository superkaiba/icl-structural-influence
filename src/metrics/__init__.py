"""Structural metrics and context sensitivity scores for ICL analysis."""

from .structural_influence import (
    # Base classes
    StructuralMetric,
    SensitivityResult,
    # Metrics
    DirichletEnergy,
    ClusterSeparation,
    RepresentationCoherence,
    LayerWiseProgress,
    LevelSpecificClusterSeparation,
    # Context Sensitivity Score (renamed from BIF)
    ContextSensitivityScore,
    # Experiment interface
    ContextSensitivityExperiment,
    # Standalone functions
    compute_levelwise_phi_trajectory,
    # Backwards compatibility
    StructuralInfluenceExperiment,
)

from .superposition_metrics import (
    # Result containers
    SuperpositionResult,
    CollapseAnalysis,
    # Core metric functions
    compute_centroid,
    compute_all_centroids,
    compute_hypothesis_centroid,
    compute_superposition_score,
    compute_hypothesis_ratio,
    compute_collapse_magnitude,
    compute_representation_velocity,
    # Analysis functions
    analyze_position_trajectory,
    analyze_collapse,
    aggregate_trial_results,
)

__all__ = [
    # Structural influence
    "StructuralMetric",
    "SensitivityResult",
    "DirichletEnergy",
    "ClusterSeparation",
    "RepresentationCoherence",
    "LayerWiseProgress",
    "LevelSpecificClusterSeparation",
    "ContextSensitivityScore",
    "ContextSensitivityExperiment",
    "compute_levelwise_phi_trajectory",
    "StructuralInfluenceExperiment",  # alias for backwards compat
    # Superposition metrics
    "SuperpositionResult",
    "CollapseAnalysis",
    "compute_centroid",
    "compute_all_centroids",
    "compute_hypothesis_centroid",
    "compute_superposition_score",
    "compute_hypothesis_ratio",
    "compute_collapse_magnitude",
    "compute_representation_velocity",
    "analyze_position_trajectory",
    "analyze_collapse",
    "aggregate_trial_results",
]
