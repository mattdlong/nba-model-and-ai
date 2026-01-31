"""Fixtures for model tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data

if TYPE_CHECKING:
    from nba_model.models.transformer import GameFlowTransformer
    from nba_model.models.gnn import PlayerInteractionGNN
    from nba_model.models.fusion import TwoTowerFusion


@pytest.fixture
def sample_events() -> torch.Tensor:
    """Sample event indices for transformer testing."""
    return torch.randint(1, 14, (2, 50))  # batch=2, seq_len=50


@pytest.fixture
def sample_times() -> torch.Tensor:
    """Sample time features for transformer testing."""
    return torch.rand(2, 50, 1)


@pytest.fixture
def sample_scores() -> torch.Tensor:
    """Sample score features for transformer testing."""
    return torch.randn(2, 50, 1)


@pytest.fixture
def sample_lineups() -> torch.Tensor:
    """Sample lineup features for transformer testing."""
    return torch.zeros(2, 50, 20)


@pytest.fixture
def sample_mask() -> torch.Tensor:
    """Sample attention mask for transformer testing."""
    return torch.zeros(2, 50, dtype=torch.bool)


@pytest.fixture
def sample_graph() -> Data:
    """Sample PyG Data object for GNN testing."""
    x = torch.randn(10, 16)  # 10 players, 16 features

    # Create edges: fully connected within each team + some cross-team
    src = []
    dst = []
    for i in range(5):
        for j in range(5):
            if i != j:
                src.extend([i, i + 5])
                dst.extend([j, j + 5])
    # Add some cross-team edges
    for i in range(5):
        src.extend([i, i + 5])
        dst.extend([i + 5, i])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


@pytest.fixture
def sample_context() -> torch.Tensor:
    """Sample context features for fusion testing."""
    return torch.randn(2, 32)


@pytest.fixture
def sample_transformer_output() -> torch.Tensor:
    """Sample transformer output for fusion testing."""
    return torch.randn(2, 128)


@pytest.fixture
def sample_gnn_output() -> torch.Tensor:
    """Sample GNN output for fusion testing."""
    return torch.randn(2, 128)


@pytest.fixture
def sample_labels() -> dict[str, torch.Tensor]:
    """Sample labels for training testing."""
    return {
        "win": torch.randint(0, 2, (2, 1)).float(),
        "margin": torch.randn(2, 1) * 10,
        "total": torch.randn(2, 1) * 15 + 220,
    }


@pytest.fixture
def sample_plays_df() -> pd.DataFrame:
    """Sample plays DataFrame for tokenizer testing."""
    return pd.DataFrame({
        "game_id": ["001"] * 10,
        "event_num": list(range(10)),
        "period": [1] * 10,
        "pc_time": ["12:00", "11:45", "11:30", "11:15", "11:00",
                    "10:45", "10:30", "10:15", "10:00", "09:45"],
        "event_type": [1, 2, 4, 5, 1, 2, 3, 4, 1, 2],
        "score_home": [0, 0, 0, 0, 2, 2, 2, 2, 4, 4],
        "score_away": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    })


@pytest.fixture
def sample_games_df() -> pd.DataFrame:
    """Sample games DataFrame for dataset testing."""
    return pd.DataFrame({
        "game_id": ["001", "002", "003"],
        "season_id": ["2023-24"] * 3,
        "home_team_id": [1610612747, 1610612738, 1610612749],
        "away_team_id": [1610612738, 1610612749, 1610612747],
        "home_score": [110, 105, 120],
        "away_score": [105, 112, 115],
    })


@pytest.fixture
def transformer_model() -> GameFlowTransformer:
    """Create a small transformer model for testing."""
    from nba_model.models.transformer import GameFlowTransformer
    return GameFlowTransformer(vocab_size=15, d_model=64, nhead=2, num_layers=1)


@pytest.fixture
def gnn_model() -> PlayerInteractionGNN:
    """Create a small GNN model for testing."""
    from nba_model.models.gnn import PlayerInteractionGNN
    return PlayerInteractionGNN(
        node_features=16, hidden_dim=32, output_dim=64, num_heads=2, num_layers=1
    )


@pytest.fixture
def fusion_model() -> TwoTowerFusion:
    """Create a small fusion model for testing."""
    from nba_model.models.fusion import TwoTowerFusion
    return TwoTowerFusion(
        context_dim=32, transformer_dim=64, gnn_dim=64, hidden_dim=64
    )
