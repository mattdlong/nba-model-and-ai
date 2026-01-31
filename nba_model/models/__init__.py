"""Machine learning models for NBA predictions.

This module contains the neural network architectures and training infrastructure
for the NBA prediction system:

- GameFlowTransformer: Sequence model for game event patterns
- PlayerInteractionGNN: GATv2 for lineup interaction modeling
- TwoTowerFusion: Combined architecture with multi-task outputs
- FusionTrainer: Training loop with early stopping and checkpointing
- NBADataset: PyTorch Dataset for game data
- ModelRegistry: Versioned model storage

Submodules:
    transformer: Transformer encoder sequence model
    gnn: Graph Attention Network (GATv2) model
    fusion: Two-Tower fusion architecture
    trainer: Training loop and optimization
    dataset: PyTorch Dataset implementations
    registry: Model versioning and storage

Example:
    >>> from nba_model.models import TwoTowerFusion, ModelRegistry
    >>> model = TwoTowerFusion(context_dim=32)
    >>> outputs = model(context, transformer_out, gnn_out)
    >>> registry = ModelRegistry()
    >>> registry.save_model("1.0.0", {"fusion": model}, metrics, config)
"""

from __future__ import annotations

# Transformer module
from nba_model.models.transformer import (
    GameFlowTransformer,
    EventTokenizer,
    PositionalEncoding,
    TokenizedSequence,
    collate_sequences,
    EVENT_VOCAB,
    DEFAULT_D_MODEL,
    DEFAULT_NHEAD,
    DEFAULT_NUM_LAYERS,
    DEFAULT_MAX_SEQ_LEN,
)

# GNN module
from nba_model.models.gnn import (
    PlayerInteractionGNN,
    LineupGraphBuilder,
    PlayerFeatures,
    create_empty_graph,
    DEFAULT_NODE_FEATURES,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_OUTPUT_DIM,
    DEFAULT_NUM_HEADS,
)

# Fusion module
from nba_model.models.fusion import (
    TwoTowerFusion,
    ContextFeatureBuilder,
    MultiTaskLoss,
    FusionOutput,
    GameContext,
    CONTEXT_FEATURES,
    create_dummy_inputs,
    create_dummy_labels,
)

# Trainer module
from nba_model.models.trainer import (
    FusionTrainer,
    TrainingConfig,
    EpochMetrics,
    TrainingHistory,
)

# Dataset module
from nba_model.models.dataset import (
    NBADataset,
    GameSample,
    nba_collate_fn,
    temporal_split,
    create_data_loader,
)

# Registry module
from nba_model.models.registry import (
    ModelRegistry,
    VersionInfo,
    VersionComparison,
)

__all__ = [
    # Transformer
    "GameFlowTransformer",
    "EventTokenizer",
    "PositionalEncoding",
    "TokenizedSequence",
    "collate_sequences",
    "EVENT_VOCAB",
    "DEFAULT_D_MODEL",
    "DEFAULT_NHEAD",
    "DEFAULT_NUM_LAYERS",
    "DEFAULT_MAX_SEQ_LEN",
    # GNN
    "PlayerInteractionGNN",
    "LineupGraphBuilder",
    "PlayerFeatures",
    "create_empty_graph",
    "DEFAULT_NODE_FEATURES",
    "DEFAULT_HIDDEN_DIM",
    "DEFAULT_OUTPUT_DIM",
    "DEFAULT_NUM_HEADS",
    # Fusion
    "TwoTowerFusion",
    "ContextFeatureBuilder",
    "MultiTaskLoss",
    "FusionOutput",
    "GameContext",
    "CONTEXT_FEATURES",
    "create_dummy_inputs",
    "create_dummy_labels",
    # Trainer
    "FusionTrainer",
    "TrainingConfig",
    "EpochMetrics",
    "TrainingHistory",
    # Dataset
    "NBADataset",
    "GameSample",
    "nba_collate_fn",
    "temporal_split",
    "create_data_loader",
    # Registry
    "ModelRegistry",
    "VersionInfo",
    "VersionComparison",
]
