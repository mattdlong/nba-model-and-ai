"""Tests for NBADataset and data loading utilities.

Tests cover:
- Dataset initialization
- Sample retrieval
- Collate function
- Temporal splitting
"""

from __future__ import annotations

import pytest
import torch
import pandas as pd
from torch_geometric.data import Data

from nba_model.models.dataset import (
    NBADataset,
    GameSample,
    nba_collate_fn,
    temporal_split,
    create_data_loader,
)


class TestNBADataset:
    """Tests for NBADataset class."""

    def test_initialization_minimal(
        self,
        sample_games_df: pd.DataFrame,
    ) -> None:
        """Dataset should initialize with just games DataFrame."""
        dataset = NBADataset(sample_games_df)
        assert len(dataset) == 3
        assert len(dataset.game_ids) == 3

    def test_initialization_with_plays(
        self,
        sample_games_df: pd.DataFrame,
        sample_plays_df: pd.DataFrame,
    ) -> None:
        """Dataset should accept plays DataFrame."""
        dataset = NBADataset(sample_games_df, plays_df=sample_plays_df)
        assert dataset.plays_df is not None

    def test_getitem_returns_sample(
        self,
        sample_games_df: pd.DataFrame,
    ) -> None:
        """__getitem__ should return GameSample."""
        dataset = NBADataset(sample_games_df)
        sample = dataset[0]

        assert isinstance(sample, GameSample)
        assert sample.game_id == "001"

    def test_getitem_labels(
        self,
        sample_games_df: pd.DataFrame,
    ) -> None:
        """Labels should be computed correctly."""
        dataset = NBADataset(sample_games_df)
        sample = dataset[0]  # home_score=110, away_score=105

        assert sample.win_label == 1.0  # Home won
        assert sample.margin_label == 5.0  # 110 - 105
        assert sample.total_label == 215.0  # 110 + 105

    def test_getitem_tensor_types(
        self,
        sample_games_df: pd.DataFrame,
    ) -> None:
        """All tensors should have correct types."""
        dataset = NBADataset(sample_games_df)
        sample = dataset[0]

        assert isinstance(sample.events, torch.Tensor)
        assert isinstance(sample.times, torch.Tensor)
        assert isinstance(sample.scores, torch.Tensor)
        assert isinstance(sample.lineups, torch.Tensor)
        assert isinstance(sample.mask, torch.Tensor)
        assert isinstance(sample.graph, Data)
        assert isinstance(sample.context, torch.Tensor)

    def test_from_dataframes_factory(
        self,
        sample_games_df: pd.DataFrame,
    ) -> None:
        """from_dataframes should create dataset."""
        dataset = NBADataset.from_dataframes(sample_games_df, seq_len=30)
        assert len(dataset) == 3
        assert dataset.seq_len == 30


class TestGameSample:
    """Tests for GameSample dataclass."""

    def test_all_fields(self) -> None:
        """GameSample should have all required fields."""
        sample = GameSample(
            game_id="001",
            events=torch.zeros(50, dtype=torch.long),
            times=torch.zeros(50, 1),
            scores=torch.zeros(50, 1),
            lineups=torch.zeros(50, 10, dtype=torch.long),
            mask=torch.zeros(50, dtype=torch.bool),
            graph=Data(x=torch.zeros(10, 16), edge_index=torch.zeros(2, 0, dtype=torch.long)),
            context=torch.zeros(32),
            win_label=1.0,
            margin_label=5.0,
            total_label=215.0,
        )

        assert sample.game_id == "001"
        assert sample.win_label == 1.0


class TestNbaCollateFn:
    """Tests for nba_collate_fn."""

    def test_collate_multiple_samples(
        self,
        sample_games_df: pd.DataFrame,
    ) -> None:
        """Should collate multiple samples into batch."""
        dataset = NBADataset(sample_games_df)
        samples = [dataset[0], dataset[1], dataset[2]]

        batch = nba_collate_fn(samples)

        assert isinstance(batch, dict)
        assert "events" in batch
        assert "graphs" in batch
        assert "context" in batch
        assert "win_label" in batch

    def test_collate_tensor_shapes(
        self,
        sample_games_df: pd.DataFrame,
    ) -> None:
        """Batch tensors should have correct shapes."""
        dataset = NBADataset(sample_games_df, seq_len=50)
        samples = [dataset[0], dataset[1]]

        batch = nba_collate_fn(samples)

        assert batch["events"].shape[0] == 2  # Batch size
        assert batch["events"].shape[1] == 50  # Seq len
        assert batch["context"].shape == (2, 32)
        assert batch["win_label"].shape == (2, 1)
        assert batch["margin_label"].shape == (2, 1)
        assert batch["total_label"].shape == (2, 1)

    def test_collate_game_ids(
        self,
        sample_games_df: pd.DataFrame,
    ) -> None:
        """Should preserve game IDs."""
        dataset = NBADataset(sample_games_df)
        samples = [dataset[0], dataset[1]]

        batch = nba_collate_fn(samples)

        assert "game_ids" in batch
        assert len(batch["game_ids"]) == 2
        assert batch["game_ids"][0] == "001"

    def test_collate_empty_raises(self) -> None:
        """Should raise error for empty list."""
        with pytest.raises(ValueError, match="empty"):
            nba_collate_fn([])


class TestTemporalSplit:
    """Tests for temporal_split function."""

    def test_split_basic(
        self,
        sample_games_df: pd.DataFrame,
    ) -> None:
        """Should split dataset temporally."""
        dataset = NBADataset(sample_games_df)
        # Use val_ratio=0.5 so int(3 * 0.5) = 1 goes to validation
        train, val = temporal_split(dataset, val_ratio=0.5)

        # With 3 games and 50% validation, expect 2 train, 1 val
        assert len(train) == 2
        assert len(val) == 1

    def test_split_no_overlap(
        self,
        sample_games_df: pd.DataFrame,
    ) -> None:
        """Train and val should have no overlapping games."""
        dataset = NBADataset(sample_games_df)
        train, val = temporal_split(dataset, val_ratio=0.33)

        train_ids = set(train.game_ids)
        val_ids = set(val.game_ids)

        assert len(train_ids & val_ids) == 0

    def test_split_preserves_seq_len(
        self,
        sample_games_df: pd.DataFrame,
    ) -> None:
        """Split should preserve seq_len setting."""
        dataset = NBADataset(sample_games_df, seq_len=30)
        train, val = temporal_split(dataset, val_ratio=0.33)

        assert train.seq_len == 30
        assert val.seq_len == 30


class TestCreateDataLoader:
    """Tests for create_data_loader function."""

    def test_creates_loader(
        self,
        sample_games_df: pd.DataFrame,
    ) -> None:
        """Should create DataLoader with correct settings."""
        dataset = NBADataset(sample_games_df)
        loader = create_data_loader(dataset, batch_size=2, shuffle=False)

        # Should be able to iterate
        batch = next(iter(loader))
        assert "events" in batch

    def test_uses_collate_fn(
        self,
        sample_games_df: pd.DataFrame,
    ) -> None:
        """Should use custom collate function."""
        dataset = NBADataset(sample_games_df)
        loader = create_data_loader(dataset, batch_size=2)

        batch = next(iter(loader))
        # nba_collate_fn adds 'game_ids' key
        assert "game_ids" in batch
