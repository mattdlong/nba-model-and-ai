"""Machine learning models for NBA predictions.

This module contains the neural network architectures and training infrastructure
for the NBA prediction system:

- GameFlowTransformer: Sequence model for game event patterns
- PlayerInteractionGNN: GATv2 for lineup interaction modeling
- TwoTowerFusion: Combined architecture with multi-task outputs

Submodules:
    transformer: Transformer encoder sequence model
    gnn: Graph Attention Network (GATv2) model
    fusion: Two-Tower fusion architecture
    trainer: Training loop and optimization
    dataset: PyTorch Dataset implementations
    registry: Model versioning and storage

Example:
    >>> from nba_model.models import TwoTowerFusion
    >>> model = TwoTowerFusion(context_dim=32)
    >>> outputs = model(context, transformer_out, gnn_out)
"""

from __future__ import annotations

# Public API - will be populated in Phase 4
__all__: list[str] = []
