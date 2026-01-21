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

__all__ = [
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
]
