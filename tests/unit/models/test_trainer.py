"""Tests for FusionTrainer and training utilities.

Tests cover:
- Trainer initialization
- Training loop execution
- Metrics calculation
- Checkpoint save/load
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from nba_model.models.trainer import (
    FusionTrainer,
    TrainingConfig,
    EpochMetrics,
    TrainingHistory,
)
from nba_model.models.transformer import GameFlowTransformer
from nba_model.models.gnn import PlayerInteractionGNN
from nba_model.models.fusion import TwoTowerFusion


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_values(self) -> None:
        """Config should have sensible defaults."""
        config = TrainingConfig()
        assert config.learning_rate == 1e-4
        assert config.weight_decay == 1e-5
        assert config.epochs == 50
        assert config.patience == 10
        assert config.gradient_clip == 1.0
        assert config.batch_size == 32

    def test_custom_values(self) -> None:
        """Config should accept custom values."""
        config = TrainingConfig(
            learning_rate=1e-3,
            epochs=100,
            patience=20,
        )
        assert config.learning_rate == 1e-3
        assert config.epochs == 100
        assert config.patience == 20

    def test_from_settings(self) -> None:
        """Should create config from settings."""
        config = TrainingConfig.from_settings()
        assert config.learning_rate > 0
        assert config.batch_size > 0


class TestTrainingHistory:
    """Tests for TrainingHistory container."""

    def test_initialization(self) -> None:
        """History should start empty."""
        history = TrainingHistory()
        assert len(history.epochs) == 0
        assert history.best_epoch == 0
        assert history.best_val_loss == float("inf")

    def test_add_epoch(self) -> None:
        """add_epoch should append metrics."""
        history = TrainingHistory()
        metrics = EpochMetrics(
            epoch=0,
            train_loss=1.0,
            val_loss=0.8,
            win_accuracy=0.55,
            margin_mae=8.0,
            total_mae=12.0,
            brier_score=0.24,
            log_loss=0.65,
            learning_rate=1e-4,
        )
        history.add_epoch(metrics)

        assert len(history.epochs) == 1
        assert history.best_epoch == 0
        assert history.best_val_loss == 0.8

    def test_best_tracking(self) -> None:
        """Should track best validation loss."""
        history = TrainingHistory()

        history.add_epoch(EpochMetrics(epoch=0, train_loss=1.0, val_loss=0.8))
        history.add_epoch(EpochMetrics(epoch=1, train_loss=0.9, val_loss=0.7))
        history.add_epoch(EpochMetrics(epoch=2, train_loss=0.8, val_loss=0.75))

        assert history.best_epoch == 1
        assert history.best_val_loss == 0.7


class TestFusionTrainer:
    """Tests for FusionTrainer class."""

    @pytest.fixture
    def small_models(self) -> tuple:
        """Create small models for testing."""
        transformer = GameFlowTransformer(
            vocab_size=15, d_model=32, nhead=2, num_layers=1
        )
        gnn = PlayerInteractionGNN(
            node_features=16, hidden_dim=16, output_dim=32, num_heads=2, num_layers=1
        )
        fusion = TwoTowerFusion(
            context_dim=16, transformer_dim=32, gnn_dim=32, hidden_dim=32
        )
        return transformer, gnn, fusion

    @pytest.fixture
    def trainer(self, small_models: tuple) -> FusionTrainer:
        """Create trainer with small models."""
        transformer, gnn, fusion = small_models
        config = TrainingConfig(epochs=2, patience=2)
        return FusionTrainer(transformer, gnn, fusion, config)

    def test_initialization(self, small_models: tuple) -> None:
        """Trainer should initialize correctly."""
        transformer, gnn, fusion = small_models
        trainer = FusionTrainer(transformer, gnn, fusion)

        assert trainer.transformer is transformer
        assert trainer.gnn is gnn
        assert trainer.fusion is fusion
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None

    def test_device_detection(self, small_models: tuple) -> None:
        """Should auto-detect device."""
        transformer, gnn, fusion = small_models
        trainer = FusionTrainer(transformer, gnn, fusion)

        # Should be CPU or CUDA
        assert trainer.device.type in ["cpu", "cuda"]

    def test_calculate_metrics(self, trainer: FusionTrainer) -> None:
        """Should calculate validation metrics correctly."""
        # win_probs > 0.5 maps to predictions: [1, 1, 0, 1]
        win_probs = [0.6, 0.7, 0.4, 0.8]
        # labels: [1, 0, 0, 1] - 3 out of 4 correct (0.75 accuracy)
        win_labels = [1, 0, 0, 1]
        margins = [5.0, -3.0, 2.0, 10.0]
        margin_labels = [4.0, -5.0, 0.0, 8.0]
        totals = [220.0, 200.0, 215.0, 230.0]
        total_labels = [218.0, 202.0, 210.0, 225.0]

        metrics = trainer._calculate_metrics(
            win_probs, win_labels, margins, margin_labels, totals, total_labels
        )

        assert "accuracy" in metrics
        assert "margin_mae" in metrics
        assert "total_mae" in metrics
        assert "brier_score" in metrics
        assert "log_loss" in metrics

        # Accuracy: predictions [1,1,0,1] vs labels [1,0,0,1] = 3/4 correct
        assert metrics["accuracy"] == 0.75

    def test_save_load_checkpoint(
        self,
        trainer: FusionTrainer,
        tmp_path: Path,
    ) -> None:
        """Should save and load checkpoints correctly."""
        checkpoint_path = tmp_path / "test_checkpoint.pt"

        # Save
        trainer._save_checkpoint(checkpoint_path)
        assert checkpoint_path.exists()

        # Modify weights
        with torch.no_grad():
            trainer.transformer.event_embedding.weight.fill_(999.0)

        original_weight = trainer.transformer.event_embedding.weight[0, 0].item()
        assert original_weight == 999.0

        # Load
        trainer.load_checkpoint(checkpoint_path)
        loaded_weight = trainer.transformer.event_embedding.weight[0, 0].item()

        # Weights should be restored (not 999.0)
        assert loaded_weight != 999.0

    def test_save_models(
        self,
        trainer: FusionTrainer,
        tmp_path: Path,
    ) -> None:
        """Should save individual model weights."""
        trainer.save_models(tmp_path)

        assert (tmp_path / "transformer.pt").exists()
        assert (tmp_path / "gnn.pt").exists()
        assert (tmp_path / "fusion.pt").exists()

    def test_load_models(
        self,
        trainer: FusionTrainer,
        tmp_path: Path,
    ) -> None:
        """Should load individual model weights."""
        # Save first
        trainer.save_models(tmp_path)

        # Modify weights
        with torch.no_grad():
            trainer.transformer.event_embedding.weight.fill_(999.0)

        # Load
        trainer.load_models(tmp_path)

        # Weights should be restored
        weight = trainer.transformer.event_embedding.weight[0, 0].item()
        assert weight != 999.0


class TestEpochMetrics:
    """Tests for EpochMetrics dataclass."""

    def test_all_fields(self) -> None:
        """Should store all metric fields."""
        metrics = EpochMetrics(
            epoch=5,
            train_loss=0.5,
            val_loss=0.6,
            win_accuracy=0.58,
            margin_mae=7.5,
            total_mae=11.0,
            brier_score=0.22,
            log_loss=0.62,
            learning_rate=5e-5,
        )

        assert metrics.epoch == 5
        assert metrics.train_loss == 0.5
        assert metrics.val_loss == 0.6
        assert metrics.win_accuracy == 0.58
        assert metrics.margin_mae == 7.5
        assert metrics.total_mae == 11.0
        assert metrics.brier_score == 0.22
        assert metrics.log_loss == 0.62
        assert metrics.learning_rate == 5e-5

    def test_default_values(self) -> None:
        """Should have defaults for optional fields."""
        metrics = EpochMetrics(epoch=0, train_loss=1.0, val_loss=1.0)

        assert metrics.win_accuracy == 0.0
        assert metrics.margin_mae == 0.0
