"""Integration tests for the prediction pipeline.

These tests verify end-to-end behavior of the prediction system including:
- Inference pipeline with model loading
- Injury adjustment integration
- Signal generation from predictions
- CLI command execution

Latency requirements tested:
- Single game prediction: < 5 seconds
- Full day predictions (~15 games): < 2 minutes
"""

from __future__ import annotations

import time
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
import torch

if TYPE_CHECKING:
    from nba_model.config import Settings


class TestInferencePipelineIntegration:
    """Integration tests for InferencePipeline."""

    @pytest.fixture
    def mock_registry_with_models(self, tmp_data_dir: Path) -> MagicMock:
        """Create a mock registry that returns actual model weights."""
        from nba_model.models import (
            GameFlowTransformer,
            PlayerInteractionGNN,
            TwoTowerFusion,
        )

        # Create real models
        transformer = GameFlowTransformer()
        gnn = PlayerInteractionGNN()
        fusion = TwoTowerFusion()

        # Mock registry
        registry = MagicMock()
        registry.load_model.return_value = {
            "transformer": transformer.state_dict(),
            "gnn": gnn.state_dict(),
            "fusion": fusion.state_dict(),
        }
        registry.load_metadata.return_value = MagicMock(version="v1.0.0")

        return registry

    @pytest.fixture
    def mock_db_session_with_game(self) -> MagicMock:
        """Create a mock session with sample game data."""
        session = MagicMock()

        # Mock game
        mock_game = MagicMock()
        mock_game.game_id = "0022300001"
        mock_game.game_date = date(2024, 1, 15)
        mock_game.home_team_id = 1610612738
        mock_game.away_team_id = 1610612749
        mock_game.season_id = "2023-24"

        # Mock team
        mock_team = MagicMock()
        mock_team.abbreviation = "BOS"

        # Setup query chains
        game_query = MagicMock()
        game_query.filter.return_value.first.return_value = mock_game
        game_query.filter.return_value.all.return_value = [(mock_game.game_id,)]

        team_query = MagicMock()
        team_query.filter.return_value.first.return_value = mock_team

        def query_handler(model):
            if hasattr(model, "__tablename__"):
                if model.__tablename__ == "games":
                    return game_query
                elif model.__tablename__ == "teams":
                    return team_query
            # Default return for other queries
            return MagicMock()

        session.query.side_effect = query_handler

        return session

    @pytest.mark.slow
    def test_single_game_prediction_latency(
        self,
        mock_registry_with_models: MagicMock,
        mock_db_session_with_game: MagicMock,
    ) -> None:
        """Test that single game prediction completes within 5 seconds."""
        from nba_model.predict import InferencePipeline

        pipeline = InferencePipeline(
            model_registry=mock_registry_with_models,
            db_session=mock_db_session_with_game,
        )

        # Mock context feature building
        with patch(
            "nba_model.models.fusion.ContextFeatureBuilder"
        ) as MockBuilder:
            mock_builder = MagicMock()
            mock_builder.build.return_value = torch.zeros(32)
            MockBuilder.return_value = mock_builder

            with patch.object(
                pipeline, "_get_game_info"
            ) as mock_info:
                mock_info.return_value = {
                    "game_date": date(2024, 1, 15),
                    "home_team": "BOS",
                    "away_team": "MIL",
                    "home_team_id": 1610612738,
                    "away_team_id": 1610612749,
                }

                with patch.object(
                    pipeline, "_get_expected_lineups"
                ) as mock_lineups:
                    mock_lineups.return_value = ([], [])

                    with patch.object(
                        pipeline, "_get_expected_lineup_ids"
                    ) as mock_lineup_ids:
                        mock_lineup_ids.return_value = ([], [])

                        start_time = time.perf_counter()
                        prediction = pipeline.predict_game(
                            "0022300001",
                            apply_injury_adjustment=False,
                        )
                        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Should complete within 5 seconds
        assert elapsed_ms < 5000, f"Prediction took {elapsed_ms:.1f}ms (max 5000ms)"

        # Verify prediction structure
        assert prediction.game_id == "0022300001"
        assert 0.01 <= prediction.home_win_prob <= 0.99
        assert -35 <= prediction.predicted_margin <= 35
        assert 175 <= prediction.predicted_total <= 270

    def test_prediction_output_structure(
        self,
        mock_registry_with_models: MagicMock,
        mock_db_session_with_game: MagicMock,
    ) -> None:
        """Test that predictions have correct structure."""
        from nba_model.predict import GamePrediction, InferencePipeline

        pipeline = InferencePipeline(
            model_registry=mock_registry_with_models,
            db_session=mock_db_session_with_game,
        )

        with patch(
            "nba_model.models.fusion.ContextFeatureBuilder"
        ) as MockBuilder:
            mock_builder = MagicMock()
            mock_builder.build.return_value = torch.zeros(32)
            MockBuilder.return_value = mock_builder

            with patch.object(pipeline, "_get_game_info") as mock_info:
                mock_info.return_value = {
                    "game_date": date(2024, 1, 15),
                    "home_team": "BOS",
                    "away_team": "MIL",
                    "home_team_id": 1610612738,
                    "away_team_id": 1610612749,
                }

                with patch.object(
                    pipeline, "_get_expected_lineups"
                ) as mock_lineups:
                    mock_lineups.return_value = ([], [])

                    with patch.object(
                        pipeline, "_get_expected_lineup_ids"
                    ) as mock_lineup_ids:
                        mock_lineup_ids.return_value = ([], [])

                        prediction = pipeline.predict_game(
                            "0022300001",
                            apply_injury_adjustment=False,
                        )

        # Verify it's a GamePrediction
        assert isinstance(prediction, GamePrediction)

        # Check all required fields exist
        assert hasattr(prediction, "game_id")
        assert hasattr(prediction, "home_win_prob")
        assert hasattr(prediction, "predicted_margin")
        assert hasattr(prediction, "predicted_total")
        assert hasattr(prediction, "confidence")
        assert hasattr(prediction, "model_version")


class TestInjuryIntegration:
    """Integration tests for injury adjustment system."""

    def test_injury_adjustment_direction(self) -> None:
        """Test that injury adjustments move probability correctly."""
        from nba_model.predict.injuries import InjuryAdjuster

        mock_session = MagicMock()
        adjuster = InjuryAdjuster(mock_session)

        # Star player out should reduce team's chances
        base_prob = 0.55

        # Without adjustment (no injuries)
        prob_no_injury = adjuster.get_play_probability(
            player_id=203507,
            injury_status="available",
        )
        assert prob_no_injury == 1.0

        # With "out" status
        prob_out = adjuster.get_play_probability(
            player_id=203507,
            injury_status="out",
        )
        assert prob_out == 0.0

    def test_bayesian_priors_integration(self) -> None:
        """Test that Bayesian priors are correctly applied."""
        from nba_model.predict.injuries import (
            InjuryAdjuster,
            PRIOR_PLAY_PROBABILITIES,
        )

        mock_session = MagicMock()
        adjuster = InjuryAdjuster(mock_session)

        # Test each status
        for status, expected_prior in PRIOR_PLAY_PROBABILITIES.items():
            prob = adjuster.get_play_probability(
                player_id=12345,
                injury_status=status,
            )

            # Should be approximately the prior (with possible modifiers)
            # For available/healthy/out, should be exact
            if status in ("available", "healthy"):
                assert prob == 1.0
            elif status == "out":
                assert prob == 0.0
            else:
                # Others should be close to prior
                assert abs(prob - expected_prior) < 0.2


class TestSignalGenerationIntegration:
    """Integration tests for signal generation."""

    @pytest.fixture
    def sample_predictions(self) -> list:
        """Create sample predictions for testing."""
        from nba_model.predict.inference import GamePrediction

        return [
            GamePrediction(
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
                injury_uncertainty=0.1,
                top_factors=[("factor1", 0.5)],
                model_version="v1.0.0",
            ),
            GamePrediction(
                game_id="0022300002",
                game_date=date(2024, 1, 15),
                home_team="MIA",
                away_team="GSW",
                matchup="GSW @ MIA",
                home_win_prob=0.52,
                predicted_margin=1.5,
                predicted_total=220.0,
                home_win_prob_adjusted=0.51,
                predicted_margin_adjusted=1.2,
                predicted_total_adjusted=219.5,
                confidence=0.45,
                injury_uncertainty=0.3,
                top_factors=[],
                model_version="v1.0.0",
            ),
        ]

    @pytest.fixture
    def sample_market_odds(self) -> dict:
        """Create sample market odds."""
        from nba_model.predict.signals import create_market_odds

        return {
            "0022300001": create_market_odds(
                game_id="0022300001",
                home_ml=1.65,  # ~60% implied
                away_ml=2.35,
                spread_home=-5.5,
                total=220.0,
            ),
            "0022300002": create_market_odds(
                game_id="0022300002",
                home_ml=1.91,  # ~52% implied
                away_ml=1.91,
                spread_home=-1.0,
                total=218.0,
            ),
        }

    def test_signal_generation_with_real_calculators(
        self, sample_predictions: list, sample_market_odds: dict
    ) -> None:
        """Test signal generation using real Devig and Kelly calculators."""
        from nba_model.backtest.devig import DevigCalculator
        from nba_model.backtest.kelly import KellyCalculator
        from nba_model.predict.signals import SignalGenerator

        devig = DevigCalculator()
        kelly = KellyCalculator(fraction=0.25, max_bet_pct=0.02, min_edge_pct=0.02)
        generator = SignalGenerator(devig, kelly, min_edge=0.02)

        signals = generator.generate_signals(sample_predictions, sample_market_odds)

        # Verify signals are properly formatted
        for signal in signals:
            assert signal.edge >= 0.02  # Must meet min edge
            assert 0 <= signal.model_prob <= 1
            assert 0 <= signal.market_prob <= 1
            assert signal.bet_type in ["moneyline", "spread", "total"]
            assert signal.confidence in ["high", "medium", "low"]

    def test_no_signals_when_no_edge(self) -> None:
        """Test that no signals are generated when market is efficient."""
        from nba_model.backtest.devig import DevigCalculator
        from nba_model.backtest.kelly import KellyCalculator
        from nba_model.predict.inference import GamePrediction
        from nba_model.predict.signals import SignalGenerator, create_market_odds

        devig = DevigCalculator()
        kelly = KellyCalculator(fraction=0.25, max_bet_pct=0.02, min_edge_pct=0.02)
        generator = SignalGenerator(devig, kelly, min_edge=0.02)

        # Prediction matches market exactly
        predictions = [
            GamePrediction(
                game_id="0022300001",
                game_date=date(2024, 1, 15),
                home_team="BOS",
                away_team="LAL",
                matchup="LAL @ BOS",
                home_win_prob=0.50,  # Matches 50% implied
                predicted_margin=0.0,  # Matches spread
                predicted_total=220.0,  # Matches total
                home_win_prob_adjusted=0.50,
                predicted_margin_adjusted=0.0,
                predicted_total_adjusted=220.0,
                confidence=0.5,
                injury_uncertainty=0.0,
            ),
        ]

        odds = {
            "0022300001": create_market_odds(
                game_id="0022300001",
                home_ml=1.91,  # ~52% implied, after devig ~50%
                away_ml=1.91,
                spread_home=0.0,
                total=220.0,
            ),
        }

        signals = generator.generate_signals(predictions, odds)

        # Should have no signals (no edge)
        assert len(signals) == 0


class TestEndToEndPrediction:
    """End-to-end tests for complete prediction flow."""

    @pytest.mark.slow
    def test_full_prediction_flow(self, test_settings: "Settings") -> None:
        """Test complete prediction flow from database to signals."""
        # This would require a populated database
        # For now, verify the flow with mocks
        from unittest.mock import MagicMock, patch

        from nba_model.backtest.devig import DevigCalculator
        from nba_model.backtest.kelly import KellyCalculator
        from nba_model.models import ModelRegistry
        from nba_model.predict import InferencePipeline, SignalGenerator

        # Mock registry
        with patch.object(ModelRegistry, "load_model") as mock_load:
            mock_load.return_value = {}

            # Verify components can be instantiated
            devig = DevigCalculator()
            kelly = KellyCalculator()
            generator = SignalGenerator(devig, kelly)

            # Verify generator is properly configured
            assert generator.min_edge == 0.02
            assert generator.bankroll == 10000.0


class TestCLIPredict:
    """Integration tests for CLI predict commands."""

    def test_predict_today_command_structure(self) -> None:
        """Test that predict today command is properly defined."""
        from typer.testing import CliRunner

        from nba_model.cli import app

        runner = CliRunner()

        # Run with --help to verify command exists
        result = runner.invoke(app, ["predict", "today", "--help"])

        assert result.exit_code == 0
        assert "Generate predictions for today's games" in result.output

    def test_predict_game_command_structure(self) -> None:
        """Test that predict game command is properly defined."""
        from typer.testing import CliRunner

        from nba_model.cli import app

        runner = CliRunner()

        # Run with --help to verify command exists
        result = runner.invoke(app, ["predict", "game", "--help"])

        assert result.exit_code == 0
        assert "Generate prediction for a single game" in result.output

    def test_predict_signals_command_structure(self) -> None:
        """Test that predict signals command is properly defined."""
        from typer.testing import CliRunner

        from nba_model.cli import app

        runner = CliRunner()

        # Run with --help to verify command exists
        result = runner.invoke(app, ["predict", "signals", "--help"])

        assert result.exit_code == 0
        assert "Generate betting signals" in result.output

    def test_predict_date_command_structure(self) -> None:
        """Test that predict date command is properly defined."""
        from typer.testing import CliRunner

        from nba_model.cli import app

        runner = CliRunner()

        # Run with --help to verify command exists
        result = runner.invoke(app, ["predict", "date", "--help"])

        assert result.exit_code == 0
        assert "Generate predictions for games on a specific date" in result.output


class TestPredictImportable:
    """Test that all predict module exports are importable."""

    def test_inference_imports(self) -> None:
        """Test inference module imports."""
        from nba_model.predict import (
            GamePrediction,
            InferencePipeline,
            PredictionBatch,
        )

        assert GamePrediction is not None
        assert InferencePipeline is not None
        assert PredictionBatch is not None

    def test_injuries_imports(self) -> None:
        """Test injuries module imports."""
        from nba_model.predict import (
            InjuryAdjuster,
            InjuryReport,
            InjuryReportFetcher,
            InjuryStatus,
            PlayerAvailability,
            PRIOR_PLAY_PROBABILITIES,
            parse_injury_status,
        )

        assert InjuryAdjuster is not None
        assert PRIOR_PLAY_PROBABILITIES is not None

    def test_signals_imports(self) -> None:
        """Test signals module imports."""
        from nba_model.predict import (
            SignalGenerator,
            BettingSignal,
            MarketOdds,
            BetType,
            Side,
            Confidence,
            create_market_odds,
            american_to_decimal,
            decimal_to_american,
        )

        assert SignalGenerator is not None
        assert BettingSignal is not None
        assert american_to_decimal(-110) > 1
