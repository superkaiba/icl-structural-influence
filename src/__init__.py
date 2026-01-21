"""
Structural Influence Framework for ICL Representation Analysis.

A framework for measuring how individual in-context examples contribute
to the emergence of geometric structures in LLM residual streams.

Modules:
    data: Hierarchical graph generation and random walk sampling
    models: Hooked LLM infrastructure for activation extraction
    metrics: Structural influence metrics (BIF, Dirichlet Energy, etc.)
    utils: Helper utilities

References:
    - Park et al. (2024) "ICLR: In-Context Learning of Representations"
    - Lee et al. (2025) "Influence Dynamics and Stagewise Data Attribution"
"""

__version__ = "0.1.0"
