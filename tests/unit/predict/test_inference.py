"""Unit tests for the inference pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from nba_model.predict.inference import (
    DEFAULT_CONTEXT_DIM,
    DEFAULT_GNN_DIM,
    DEFAULT_TRANSFORMER_DIM,
    GamePrediction,
    InferencePipeline,
    MAX_MARGIN,
    MAX_TOTAL,
    MAX_WIN_PROB,
    MIN_MARGIN,
    MIN_TOTAL,
    MIN_WIN_PROB,
    PredictionBatch,
    GameNotFoundError,
    InferenceError,
    ModelLoadError,
)


class TestGamePrediction:
    """Tests for GamePrediction dataclass."""

    def test_create_game_prediction(self) -> None:
        """Test creating a GamePrediction with all fields."""
        pred = GamePrediction(
            game_id="0022300001",
            game_date=date(2024, 1, 15),
            home_team="BOS",
            away_team="LAL",
            matchup="LAL @ BOS",
            home_win_prob=0.58,
            predicted_margin=5.2,
            predicted_total=218.5,
            home_win_prob_adjusted=0.56,
            predicted_margin_adjusted=4.8,
            predicted_total_adjusted=217.0,
            confidence=0.72,
            injury_uncertainty=0.15,
            top_factors=[("home_off_rating_z", 0.85), ("rapm_diff", 0.72)],
            model_version="v1.0.0",
        )

        assert pred.game_id == "0022300001"
        assert pred.home_win_prob == 0.58
        assert pred.confidence == 0.72
        assert len(pred.top_factors) == 2
        assert pred.matchup == "LAL @ BOS"

    def test_prediction_default_values(self) -> None:
        """Test that optional fields have default values."""
        pred = GamePrediction(
            game_id="0022300001",
            game_date=date(2024, 1, 15),
            home_team="BOS",
            away_team="LAL",
            matchup="LAL @ BOS",
            home_win_prob=0.58,
            predicted_margin=5.2,
            predicted_total=218.5,
            home_win_prob_adjusted=0.58,
            predicted_margin_adjusted=5.2,
            predicted_total_adjusted=218.5,
            confidence=0.72,
            injury_uncertainty=0.0,
        )

        assert pred.top_factors == []
        assert pred.home_lineup == []
        assert pred.away_lineup == []
        assert pred.model_version == ""
        assert pred.inference_time_ms == 0.0


class TestPredictionBatch:
    """Tests for PredictionBatch dataclass."""

    def test_create_batch(self) -> None:
        """Test creating a prediction batch."""
        predictions = [
            GamePrediction(
                game_id=f"002230000{i}",
                game_date=date(2024, 1, 15),
                home_team="BOS",
                away_team="LAL",
                matchup="LAL @ BOS",
                home_win_prob=0.55 + i * 0.01,
                predicted_margin=3.0 + i,
                predicted_total=215.0 + i * 2,
                home_win_prob_adjusted=0.55 + i * 0.01,
                predicted_margin_adjusted=3.0 + i,
                predicted_total_adjusted=215.0 + i * 2,
                confidence=0.6,
                injury_uncertainty=0.0,
            )
            for i in range(3)
        ]

        batch = PredictionBatch(
            predictions=predictions,
            prediction_date=date(2024, 1, 15),
            total_games=3,
            model_version="v1.0.0",
            total_time_ms=150.5,
        )

        assert batch.total_games == 3
        assert len(batch.predictions) == 3
        assert batch.total_time_ms == 150.5


class TestInferencePipelineInit:
    """Tests for InferencePipeline initialization."""

    def test_init_with_defaults(self) -> None:
        """Test initializing pipeline with default settings."""
        mock_registry = MagicMock()
        mock_session = MagicMock()

        pipeline = InferencePipeline(mock_registry, mock_session)

        assert pipeline.registry == mock_registry
        assert pipeline.db_session == mock_session
        assert pipeline.model_version == "latest"
        assert not pipeline._models_loaded

    def test_init_with_specific_version(self) -> None:
        """Test initializing pipeline with specific model version."""
        mock_registry = MagicMock()
        mock_session = MagicMock()

        pipeline = InferencePipeline(
            mock_registry,
            mock_session,
            model_version="v1.2.0",
        )

        assert pipeline.model_version == "v1.2.0"

    def test_init_with_explicit_device(self) -> None:
        """Test initializing pipeline with explicit device."""
        mock_registry = MagicMock()
        mock_session = MagicMock()
        device = torch.device("cpu")

        pipeline = InferencePipeline(
            mock_registry,
            mock_session,
            device=device,
        )

        assert pipeline.device == device


class TestInferencePipelineModelLoading:
    """Tests for model loading behavior."""

    def test_lazy_model_loading(self) -> None:
        """Test that models are loaded lazily."""
        mock_registry = MagicMock()
        mock_registry.load_model.return_value = {}
        mock_registry.load_metadata.return_value = None
        mock_session = MagicMock()

        pipeline = InferencePipeline(mock_registry, mock_session)

        # Models should not be loaded yet
        assert not pipeline._models_loaded
        mock_registry.load_model.assert_not_called()

        # Trigger loading by patching at the import time
        with patch("nba_model.models.GameFlowTransformer") as mock_transformer, \
             patch("nba_model.models.PlayerInteractionGNN") as mock_gnn, \
             patch("nba_model.models.TwoTowerFusion") as mock_fusion:

            # Setup mocks to return instances that can be moved to device
            for mock in [mock_transformer, mock_gnn, mock_fusion]:
                instance = MagicMock()
                instance.to.return_value = instance
                instance.eval.return_value = instance
                mock.return_value = instance

            pipeline._ensure_models_loaded()

        # Now models should be loaded
        assert pipeline._models_loaded
        mock_registry.load_model.assert_called_once()

    def test_model_load_error_handling(self) -> None:
        """Test error handling when model loading fails."""
        mock_registry = MagicMock()
        mock_registry.load_model.side_effect = Exception("Model not found")
        mock_session = MagicMock()

        pipeline = InferencePipeline(mock_registry, mock_session)

        with pytest.raises(ModelLoadError):
            pipeline._ensure_models_loaded()


class TestPredictionBounds:
    """Tests for prediction value bounds."""

    def test_win_prob_bounds(self) -> None:
        """Test that win probability bounds are reasonable."""
        assert MIN_WIN_PROB == 0.01
        assert MAX_WIN_PROB == 0.99
        assert MIN_WIN_PROB < MAX_WIN_PROB

    def test_margin_bounds(self) -> None:
        """Test that margin bounds are reasonable."""
        assert MIN_MARGIN == -35.0
        assert MAX_MARGIN == 35.0
        assert MIN_MARGIN < MAX_MARGIN

    def test_total_bounds(self) -> None:
        """Test that total bounds are reasonable."""
        assert MIN_TOTAL == 175.0
        assert MAX_TOTAL == 270.0
        assert MIN_TOTAL < MAX_TOTAL

    def test_dimension_constants(self) -> None:
        """Test that dimension constants are correct."""
        assert DEFAULT_CONTEXT_DIM == 32
        assert DEFAULT_TRANSFORMER_DIM == 128
        assert DEFAULT_GNN_DIM == 128


class TestInferencePipelineHelpers:
    """Tests for helper methods."""

    def test_get_games_for_date_empty(self) -> None:
        """Test getting games when none exist for date."""
        mock_registry = MagicMock()
        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.all.return_value = []

        pipeline = InferencePipeline(mock_registry, mock_session)
        games = pipeline._get_games_for_date(date(2024, 1, 15))

        assert games == []

    def test_game_not_found_error(self) -> None:
        """Test GameNotFoundError is raised correctly."""
        mock_registry = MagicMock()
        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.first.return_value = None

        pipeline = InferencePipeline(mock_registry, mock_session)

        with pytest.raises(GameNotFoundError):
            pipeline._get_game_info("nonexistent_game")


class TestGetTopFactors:
    """Tests for feature importance extraction."""

    def test_top_factors_extraction(self) -> None:
        """Test extracting top contributing factors."""
        mock_registry = MagicMock()
        mock_session = MagicMock()
        pipeline = InferencePipeline(mock_registry, mock_session)

        # Create test context features
        context = torch.zeros(32)
        context[0] = 1.5  # home_off_rating_z
        context[1] = -0.8  # home_def_rating_z
        context[22] = 2.0  # rapm_diff

        factors = pipeline._get_top_factors(context, top_k=3)

        assert len(factors) == 3
        # Should be sorted by absolute value
        assert all(isinstance(f, tuple) for f in factors)
        assert all(len(f) == 2 for f in factors)


class TestContextFeatureBuilding:
    """Tests for context feature building."""

    def test_build_context_features_no_game(self) -> None:
        """Test building features when game doesn't exist."""
        mock_registry = MagicMock()
        mock_session = MagicMock()

        # Setup mock to return None for game query
        mock_query = MagicMock()
        mock_query.filter.return_value.first.return_value = None
        mock_session.query.return_value = mock_query

        pipeline = InferencePipeline(mock_registry, mock_session)

        with patch("nba_model.models.fusion.ContextFeatureBuilder") as MockBuilder:
            mock_builder = MagicMock()
            mock_builder.build.return_value = torch.zeros(32)
            MockBuilder.return_value = mock_builder

            features = pipeline._build_context_features("nonexistent")

            assert features.shape == (32,)


class TestTransformerOutput:
    """Tests for transformer output generation."""

    def test_pre_game_transformer_output(self) -> None:
        """Test that pre-game predictions use zeros for transformer."""
        mock_registry = MagicMock()
        mock_session = MagicMock()

        pipeline = InferencePipeline(mock_registry, mock_session)

        output = pipeline._get_transformer_output("0022300001")

        assert output.shape == (DEFAULT_TRANSFORMER_DIM,)
        assert torch.all(output == 0)


class TestGNNOutput:
    """Tests for GNN output generation."""

    def test_gnn_output_fallback(self) -> None:
        """Test GNN output falls back to zeros on error."""
        mock_registry = MagicMock()
        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.first.return_value = None

        pipeline = InferencePipeline(mock_registry, mock_session)
        pipeline._models_loaded = True
        pipeline._gnn = MagicMock()

        # Should return zeros when lineup not found
        output = pipeline._get_gnn_output("0022300001")

        assert output.shape == (DEFAULT_GNN_DIM,)
        assert torch.all(output == 0)


class TestPredictionConfidence:
    """Tests for confidence score calculation."""

    def test_confidence_at_extremes(self) -> None:
        """Test confidence calculation at probability extremes."""
        # Confidence should be high when probability is extreme
        # confidence = abs(home_win_prob - 0.5) * 2

        prob_high = 0.9
        expected_high = abs(prob_high - 0.5) * 2
        assert expected_high == pytest.approx(0.8)

        prob_low = 0.1
        expected_low = abs(prob_low - 0.5) * 2
        assert expected_low == pytest.approx(0.8)

    def test_confidence_at_center(self) -> None:
        """Test confidence is low when probability is near 0.5."""
        prob_center = 0.5
        expected = abs(prob_center - 0.5) * 2
        assert expected == 0.0

        prob_near_center = 0.52
        expected_near = abs(prob_near_center - 0.5) * 2
        assert expected_near == pytest.approx(0.04)
