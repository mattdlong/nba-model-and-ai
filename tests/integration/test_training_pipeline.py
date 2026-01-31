"""Integration tests for Phase 4 model training pipeline.

Tests end-to-end training flow, dataset behavior, and model registry operations.
"""

from __future__ import annotations

import json
import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import pandas as pd

# Skip all tests if torch has OpenMP issues
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available"),
]


class TestNBADataset:
    """Integration tests for NBADataset and collate function."""

    def test_dataset_from_dataframes_creates_samples(self) -> None:
        """Dataset should create valid samples from DataFrames."""
        from nba_model.models import NBADataset

        # Create minimal games DataFrame
        games_df = pd.DataFrame({
            "game_id": ["0022300001", "0022300002", "0022300003"],
            "home_score": [110, 105, 120],
            "away_score": [100, 108, 115],
        })

        dataset = NBADataset.from_dataframes(games_df)

        assert len(dataset) == 3

        # Get first sample
        sample = dataset[0]
        assert sample.game_id == "0022300001"
        assert sample.win_label == 1.0  # home won
        assert sample.margin_label == 10.0
        assert sample.total_label == 210.0

    def test_dataset_with_plays_tokenizes_correctly(self) -> None:
        """Dataset should tokenize play-by-play data correctly."""
        from nba_model.models import NBADataset

        # Create games DataFrame
        games_df = pd.DataFrame({
            "game_id": ["0022300001"],
            "home_score": [110],
            "away_score": [100],
        })

        # Create plays DataFrame
        plays_df = pd.DataFrame({
            "game_id": ["0022300001"] * 5,
            "event_num": [1, 2, 3, 4, 5],
            "period": [1, 1, 1, 1, 1],
            "pc_time": ["12:00", "11:45", "11:30", "11:15", "11:00"],
            "event_type": [10, 1, 2, 4, 1],  # Jump ball, made shot, missed, rebound, made
            "score_home": [0, 2, 2, 2, 4],
            "score_away": [0, 0, 0, 0, 0],
        })

        dataset = NBADataset.from_dataframes(games_df, plays_df)
        sample = dataset[0]

        # Check sequence tensors have correct shapes
        assert sample.events.shape[0] == 50  # Padded to max_seq_len
        assert sample.times.shape == (50, 1)
        assert sample.scores.shape == (50, 1)
        assert sample.lineups.shape == (50, 20)

    def test_collate_fn_batches_correctly(self) -> None:
        """Collate function should properly batch samples."""
        from nba_model.models import NBADataset, nba_collate_fn

        games_df = pd.DataFrame({
            "game_id": ["0022300001", "0022300002"],
            "home_score": [110, 105],
            "away_score": [100, 108],
        })

        dataset = NBADataset.from_dataframes(games_df)
        samples = [dataset[0], dataset[1]]

        batch = nba_collate_fn(samples)

        assert batch["events"].shape == (2, 50)
        assert batch["context"].shape == (2, 32)
        assert batch["win_label"].shape == (2, 1)
        assert len(batch["game_ids"]) == 2

    def test_temporal_split_preserves_order(self) -> None:
        """Temporal split should keep later games in validation."""
        from nba_model.models import NBADataset, temporal_split

        games_df = pd.DataFrame({
            "game_id": [f"002230000{i}" for i in range(10)],
            "game_date": pd.date_range("2024-01-01", periods=10),
            "home_score": [100] * 10,
            "away_score": [95] * 10,
        })

        dataset = NBADataset.from_dataframes(games_df)
        train_ds, val_ds = temporal_split(dataset, val_ratio=0.2, games_df=games_df)

        assert len(train_ds) == 8
        assert len(val_ds) == 2


class TestModelComponents:
    """Integration tests for model components working together."""

    def test_transformer_forward_pass(self) -> None:
        """Transformer should produce correct output shape."""
        from nba_model.models import GameFlowTransformer

        model = GameFlowTransformer(vocab_size=15, d_model=128)

        # Create dummy inputs
        batch_size = 4
        seq_len = 50
        events = torch.randint(0, 15, (batch_size, seq_len))
        times = torch.rand(batch_size, seq_len, 1)
        scores = torch.randn(batch_size, seq_len, 1)
        lineups = torch.zeros(batch_size, seq_len, 20)

        output = model(events, times, scores, lineups)

        assert output.shape == (batch_size, 128)

    def test_gnn_forward_pass(self) -> None:
        """GNN should produce correct output shape."""
        from nba_model.models import PlayerInteractionGNN, create_empty_graph
        from torch_geometric.data import Batch

        model = PlayerInteractionGNN(
            node_features=16,
            hidden_dim=64,
            output_dim=128,
        )

        # Create batch of graphs
        graphs = [create_empty_graph() for _ in range(4)]
        batch = Batch.from_data_list(graphs)

        output = model(batch)

        assert output.shape == (4, 128)

    def test_fusion_forward_pass(self) -> None:
        """Fusion model should produce all three outputs."""
        from nba_model.models import TwoTowerFusion

        model = TwoTowerFusion(
            context_dim=32,
            transformer_dim=128,
            gnn_dim=128,
        )

        batch_size = 4
        context = torch.randn(batch_size, 32)
        transformer_out = torch.randn(batch_size, 128)
        gnn_out = torch.randn(batch_size, 128)

        outputs = model(context, transformer_out, gnn_out)

        assert outputs["win_prob"].shape == (batch_size, 1)
        assert outputs["margin"].shape == (batch_size, 1)
        assert outputs["total"].shape == (batch_size, 1)
        assert torch.all(outputs["win_prob"] >= 0) and torch.all(outputs["win_prob"] <= 1)

    def test_multi_task_loss_computation(self) -> None:
        """Multi-task loss should compute all components."""
        from nba_model.models import MultiTaskLoss

        loss_fn = MultiTaskLoss()

        outputs = {
            "win_prob": torch.tensor([[0.6], [0.4]]),
            "margin": torch.tensor([[5.0], [-3.0]]),
            "total": torch.tensor([[210.0], [205.0]]),
        }
        labels = {
            "win": torch.tensor([[1.0], [0.0]]),
            "margin": torch.tensor([[7.0], [-5.0]]),
            "total": torch.tensor([[215.0], [200.0]]),
        }

        loss, components = loss_fn(outputs, labels)

        assert loss.item() > 0
        assert "win_loss" in components
        assert "margin_loss" in components
        assert "total_loss" in components


class TestTrainingPipeline:
    """Integration tests for the training pipeline."""

    def test_trainer_initialization(self) -> None:
        """Trainer should initialize all components correctly."""
        from nba_model.models import (
            GameFlowTransformer,
            PlayerInteractionGNN,
            TwoTowerFusion,
            FusionTrainer,
            TrainingConfig,
        )

        transformer = GameFlowTransformer(vocab_size=15)
        gnn = PlayerInteractionGNN(node_features=16)
        fusion = TwoTowerFusion(context_dim=32)
        config = TrainingConfig(epochs=5, patience=3)

        trainer = FusionTrainer(transformer, gnn, fusion, config)

        assert trainer.transformer is not None
        assert trainer.gnn is not None
        assert trainer.fusion is not None
        assert trainer.optimizer is not None

    def test_trainer_single_batch_forward(self) -> None:
        """Trainer should process a single batch without error."""
        from nba_model.models import (
            GameFlowTransformer,
            PlayerInteractionGNN,
            TwoTowerFusion,
            FusionTrainer,
            create_empty_graph,
        )
        from torch_geometric.data import Batch

        transformer = GameFlowTransformer(vocab_size=15)
        gnn = PlayerInteractionGNN(node_features=16)
        fusion = TwoTowerFusion(context_dim=32)
        trainer = FusionTrainer(transformer, gnn, fusion)

        # Create minimal batch
        batch = {
            "events": torch.randint(0, 15, (2, 50)),
            "times": torch.rand(2, 50, 1),
            "scores": torch.randn(2, 50, 1),
            "lineups": torch.zeros(2, 50, 20),
            "mask": torch.zeros(2, 50, dtype=torch.bool),
            "graphs": Batch.from_data_list([create_empty_graph(), create_empty_graph()]),
            "context": torch.randn(2, 32),
            "win_label": torch.tensor([[1.0], [0.0]]),
            "margin_label": torch.tensor([[10.0], [-5.0]]),
            "total_label": torch.tensor([[210.0], [200.0]]),
        }

        # Should not raise
        outputs = trainer._forward_batch(batch)
        assert "win_prob" in outputs

    def test_save_and_load_models(self) -> None:
        """Trainer should save and load model weights correctly."""
        from nba_model.models import (
            GameFlowTransformer,
            PlayerInteractionGNN,
            TwoTowerFusion,
            FusionTrainer,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create and save
            transformer = GameFlowTransformer(vocab_size=15)
            gnn = PlayerInteractionGNN(node_features=16)
            fusion = TwoTowerFusion(context_dim=32)
            trainer = FusionTrainer(transformer, gnn, fusion)

            # Modify a weight to verify save/load
            with torch.no_grad():
                transformer.event_embedding.weight[0, 0] = 999.0

            trainer.save_models(output_dir)

            # Create fresh models and load
            transformer2 = GameFlowTransformer(vocab_size=15)
            gnn2 = PlayerInteractionGNN(node_features=16)
            fusion2 = TwoTowerFusion(context_dim=32)
            trainer2 = FusionTrainer(transformer2, gnn2, fusion2)

            trainer2.load_models(output_dir)

            # Verify weights match
            assert transformer2.event_embedding.weight[0, 0].item() == 999.0


class TestModelRegistry:
    """Integration tests for model registry."""

    def test_registry_save_and_load(self) -> None:
        """Registry should save and load models correctly."""
        from nba_model.models import (
            GameFlowTransformer,
            PlayerInteractionGNN,
            TwoTowerFusion,
            ModelRegistry,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(base_dir=Path(tmpdir))

            transformer = GameFlowTransformer(vocab_size=15)
            gnn = PlayerInteractionGNN(node_features=16)
            fusion = TwoTowerFusion(context_dim=32)

            models = {
                "transformer": transformer,
                "gnn": gnn,
                "fusion": fusion,
            }
            metrics = {"accuracy": 0.58, "loss": 0.65}
            config = {"d_model": 128}

            registry.save_model("1.0.0", models, metrics, config)

            # Load and verify
            loaded_weights = registry.load_model("1.0.0")
            assert "transformer" in loaded_weights
            assert "gnn" in loaded_weights
            assert "fusion" in loaded_weights

    def test_registry_list_versions(self) -> None:
        """Registry should list all saved versions."""
        from nba_model.models import (
            GameFlowTransformer,
            ModelRegistry,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(base_dir=Path(tmpdir))

            model = GameFlowTransformer(vocab_size=15)

            registry.save_model("1.0.0", {"transformer": model}, {}, {})
            registry.save_model("1.1.0", {"transformer": model}, {}, {})
            registry.save_model("2.0.0", {"transformer": model}, {}, {})

            versions = registry.list_versions()

            assert len(versions) == 3
            assert any(v.version == "1.0.0" for v in versions)
            assert any(v.version == "2.0.0" and v.is_latest for v in versions)

    def test_registry_latest_symlink(self) -> None:
        """Registry should create 'latest' symlink to newest version."""
        from nba_model.models import (
            GameFlowTransformer,
            ModelRegistry,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(base_dir=Path(tmpdir))

            model = GameFlowTransformer(vocab_size=15)

            registry.save_model("1.0.0", {"transformer": model}, {}, {})

            # Load via "latest"
            loaded = registry.load_model("latest")
            assert "transformer" in loaded


class TestLineupEncoding:
    """Integration tests for lineup encoding in tokenizer."""

    def test_lineup_encoding_with_stints(self) -> None:
        """Tokenizer should encode lineups from stint data."""
        from nba_model.models.transformer import EventTokenizer

        tokenizer = EventTokenizer(max_seq_len=10)

        # Create plays DataFrame
        plays_df = pd.DataFrame({
            "event_num": [1, 2, 3],
            "period": [1, 1, 1],
            "pc_time": ["12:00", "11:30", "11:00"],
            "event_type": [10, 1, 2],
            "score_home": [0, 2, 2],
            "score_away": [0, 0, 0],
        })

        # Create stints DataFrame
        stints_df = pd.DataFrame({
            "period": [1],
            "start_time": ["12:00"],
            "end_time": ["10:00"],
            "home_lineup": [json.dumps([1, 2, 3, 4, 5])],
            "away_lineup": [json.dumps([6, 7, 8, 9, 10])],
        })

        tokens = tokenizer.tokenize_game(plays_df, stints_df)

        # Lineups should have non-zero values for the 3 plays
        assert tokens.lineups.shape == (3, 20)
        # First 10 slots should indicate players on court
        assert tokens.lineups[0, :10].sum() > 0

    def test_lineup_encoding_without_stints_returns_zeros(self) -> None:
        """Tokenizer should return zeros when no stint data available."""
        from nba_model.models.transformer import EventTokenizer

        tokenizer = EventTokenizer(max_seq_len=10)

        plays_df = pd.DataFrame({
            "event_num": [1, 2],
            "period": [1, 1],
            "pc_time": ["12:00", "11:30"],
            "event_type": [10, 1],
            "score_home": [0, 2],
            "score_away": [0, 0],
        })

        tokens = tokenizer.tokenize_game(plays_df, stints_df=None)

        # Should be all zeros without stint data
        assert tokens.lineups.sum() == 0
