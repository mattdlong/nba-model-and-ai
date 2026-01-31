"""Tests for type definitions module."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from nba_model.types import (
    BacktestMetrics,
    Bet,
    BettingSignal,
    DataCollectionError,
    DataCollector,
    DriftDetectedError,
    DriftResult,
    FatigueIndicators,
    FeatureCalculator,
    GameId,
    GameInfo,
    GameNotFound,
    InsufficientDataError,
    ModelMetadata,
    ModelNotFoundError,
    ModelPredictor,
    NBAModelError,
    PlayerId,
    PredictionResult,
    RAPMCoefficients,
    RateLimitExceeded,
    SeasonId,
    SpacingMetrics,
    TeamId,
)


# =============================================================================
# Type Alias Tests
# =============================================================================


class TestTypeAliases:
    """Tests for type alias definitions."""

    def test_player_id_is_int(self) -> None:
        """PlayerId should be an int alias."""
        player_id: PlayerId = 203507
        assert isinstance(player_id, int)

    def test_team_id_is_int(self) -> None:
        """TeamId should be an int alias."""
        team_id: TeamId = 1610612738
        assert isinstance(team_id, int)

    def test_game_id_is_str(self) -> None:
        """GameId should be a str alias."""
        game_id: GameId = "0022300001"
        assert isinstance(game_id, str)

    def test_season_id_is_str(self) -> None:
        """SeasonId should be a str alias."""
        season_id: SeasonId = "2023-24"
        assert isinstance(season_id, str)


# =============================================================================
# Protocol Tests
# =============================================================================


class MockFeatureCalculator:
    """Mock class implementing FeatureCalculator protocol."""

    def fit(self, data: pd.DataFrame) -> None:
        """Mock fit method."""
        pass

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """Mock transform method."""
        return np.array([1.0, 2.0, 3.0])

    def save(self, path: Path) -> None:
        """Mock save method."""
        pass

    def load(self, path: Path) -> None:
        """Mock load method."""
        pass


class MockDataCollector:
    """Mock class implementing DataCollector protocol."""

    def collect(self, season: SeasonId) -> pd.DataFrame:
        """Mock collect method."""
        return pd.DataFrame({"game_id": ["0022300001"]})

    def update(self) -> pd.DataFrame:
        """Mock update method."""
        return pd.DataFrame({"game_id": ["0022300002"]})


class MockModelPredictor:
    """Mock class implementing ModelPredictor protocol."""

    def predict(self, game_id: GameId) -> dict[str, float]:
        """Mock predict method."""
        return {"home_win_prob": 0.55}

    def predict_batch(self, game_ids: list[GameId]) -> list[dict[str, float]]:
        """Mock predict_batch method."""
        return [{"home_win_prob": 0.55} for _ in game_ids]


class TestFeatureCalculatorProtocol:
    """Tests for FeatureCalculator protocol."""

    def test_mock_satisfies_protocol(self) -> None:
        """Mock class should satisfy FeatureCalculator protocol."""
        calculator: FeatureCalculator = MockFeatureCalculator()
        assert hasattr(calculator, "fit")
        assert hasattr(calculator, "transform")
        assert hasattr(calculator, "save")
        assert hasattr(calculator, "load")

    def test_fit_accepts_dataframe(self) -> None:
        """fit should accept a DataFrame."""
        calculator = MockFeatureCalculator()
        df = pd.DataFrame({"col": [1, 2, 3]})
        calculator.fit(df)  # Should not raise

    def test_transform_returns_ndarray(self) -> None:
        """transform should return numpy array."""
        calculator = MockFeatureCalculator()
        df = pd.DataFrame({"col": [1, 2, 3]})
        result = calculator.transform(df)
        assert isinstance(result, np.ndarray)

    def test_save_accepts_path(self) -> None:
        """save should accept a Path."""
        calculator = MockFeatureCalculator()
        calculator.save(Path("/tmp/test"))  # Should not raise

    def test_load_accepts_path(self) -> None:
        """load should accept a Path."""
        calculator = MockFeatureCalculator()
        calculator.load(Path("/tmp/test"))  # Should not raise


class TestDataCollectorProtocol:
    """Tests for DataCollector protocol."""

    def test_mock_satisfies_protocol(self) -> None:
        """Mock class should satisfy DataCollector protocol."""
        collector: DataCollector = MockDataCollector()
        assert hasattr(collector, "collect")
        assert hasattr(collector, "update")

    def test_collect_returns_dataframe(self) -> None:
        """collect should return a DataFrame."""
        collector = MockDataCollector()
        result = collector.collect("2023-24")
        assert isinstance(result, pd.DataFrame)

    def test_update_returns_dataframe(self) -> None:
        """update should return a DataFrame."""
        collector = MockDataCollector()
        result = collector.update()
        assert isinstance(result, pd.DataFrame)


class TestModelPredictorProtocol:
    """Tests for ModelPredictor protocol."""

    def test_mock_satisfies_protocol(self) -> None:
        """Mock class should satisfy ModelPredictor protocol."""
        predictor: ModelPredictor = MockModelPredictor()
        assert hasattr(predictor, "predict")
        assert hasattr(predictor, "predict_batch")

    def test_predict_returns_dict(self) -> None:
        """predict should return a dict."""
        predictor = MockModelPredictor()
        result = predictor.predict("0022300001")
        assert isinstance(result, dict)
        assert "home_win_prob" in result

    def test_predict_batch_returns_list(self) -> None:
        """predict_batch should return a list of dicts."""
        predictor = MockModelPredictor()
        result = predictor.predict_batch(["0022300001", "0022300002"])
        assert isinstance(result, list)
        assert len(result) == 2


# =============================================================================
# TypedDict Tests
# =============================================================================


class TestRAPMCoefficients:
    """Tests for RAPMCoefficients TypedDict."""

    def test_create_rapm_coefficients(self) -> None:
        """Should create RAPMCoefficients with all fields."""
        coefs: RAPMCoefficients = {
            "player_id": 203507,
            "orapm": 2.1,
            "drapm": 0.8,
            "total_rapm": 2.9,
            "sample_stints": 500,
        }
        assert coefs["player_id"] == 203507
        assert coefs["orapm"] == 2.1
        assert coefs["drapm"] == 0.8
        assert coefs["total_rapm"] == 2.9
        assert coefs["sample_stints"] == 500


class TestSpacingMetrics:
    """Tests for SpacingMetrics TypedDict."""

    def test_create_spacing_metrics(self) -> None:
        """Should create SpacingMetrics with all fields."""
        metrics: SpacingMetrics = {
            "hull_area": 650.5,
            "centroid_x": 10.2,
            "centroid_y": 150.8,
            "avg_distance": 12.5,
            "corner_density": 0.15,
            "shot_count": 100,
        }
        assert metrics["hull_area"] == 650.5
        assert metrics["centroid_x"] == 10.2
        assert metrics["centroid_y"] == 150.8
        assert metrics["avg_distance"] == 12.5
        assert metrics["corner_density"] == 0.15
        assert metrics["shot_count"] == 100


class TestFatigueIndicators:
    """Tests for FatigueIndicators TypedDict."""

    def test_create_fatigue_indicators(self) -> None:
        """Should create FatigueIndicators with all fields."""
        fatigue: FatigueIndicators = {
            "rest_days": 2,
            "back_to_back": False,
            "three_in_four": False,
            "four_in_five": False,
            "travel_miles": 1500.0,
            "home_stand": 3,
            "road_trip": 0,
        }
        assert fatigue["rest_days"] == 2
        assert fatigue["back_to_back"] is False
        assert fatigue["three_in_four"] is False
        assert fatigue["four_in_five"] is False
        assert fatigue["travel_miles"] == 1500.0
        assert fatigue["home_stand"] == 3
        assert fatigue["road_trip"] == 0


class TestDriftResult:
    """Tests for DriftResult TypedDict."""

    def test_create_drift_result(self) -> None:
        """Should create DriftResult with all fields."""
        result: DriftResult = {
            "has_drift": True,
            "features_drifted": ["pace", "ortg"],
            "details": {
                "pace": {"ks_stat": 0.15, "p_value": 0.01},
                "ortg": {"ks_stat": 0.12, "p_value": 0.03},
            },
        }
        assert result["has_drift"] is True
        assert len(result["features_drifted"]) == 2
        assert "pace" in result["features_drifted"]
        assert result["details"]["pace"]["p_value"] == 0.01


class TestPredictionResult:
    """Tests for PredictionResult TypedDict."""

    def test_create_prediction_result(self) -> None:
        """Should create PredictionResult with all fields."""
        pred: PredictionResult = {
            "game_id": "0022300001",
            "home_win_prob": 0.58,
            "predicted_margin": 4.5,
            "predicted_total": 218.5,
            "confidence": 0.72,
            "model_version": "v1.0.0",
        }
        assert pred["game_id"] == "0022300001"
        assert pred["home_win_prob"] == 0.58
        assert pred["predicted_margin"] == 4.5
        assert pred["predicted_total"] == 218.5
        assert pred["confidence"] == 0.72
        assert pred["model_version"] == "v1.0.0"


class TestBettingSignal:
    """Tests for BettingSignal TypedDict."""

    def test_create_betting_signal(self) -> None:
        """Should create BettingSignal with all fields."""
        signal: BettingSignal = {
            "game_id": "0022300001",
            "bet_type": "moneyline",
            "side": "home",
            "model_prob": 0.58,
            "market_prob": 0.52,
            "edge": 0.06,
            "kelly_fraction": 0.025,
            "recommended_stake_pct": 0.00625,
            "confidence": "medium",
        }
        assert signal["game_id"] == "0022300001"
        assert signal["bet_type"] == "moneyline"
        assert signal["side"] == "home"
        assert signal["model_prob"] == 0.58
        assert signal["market_prob"] == 0.52
        assert signal["edge"] == 0.06
        assert signal["kelly_fraction"] == 0.025
        assert signal["recommended_stake_pct"] == 0.00625
        assert signal["confidence"] == "medium"


# =============================================================================
# Dataclass Tests
# =============================================================================


class TestGameInfo:
    """Tests for GameInfo dataclass."""

    def test_create_game_info(self) -> None:
        """Should create GameInfo with all fields."""
        game = GameInfo(
            game_id="0022300001",
            season_id="2023-24",
            game_date=date(2024, 1, 15),
            home_team_id=1610612738,
            away_team_id=1610612749,
            home_team="Boston Celtics",
            away_team="Milwaukee Bucks",
        )
        assert game.game_id == "0022300001"
        assert game.season_id == "2023-24"
        assert game.game_date == date(2024, 1, 15)
        assert game.home_team_id == 1610612738
        assert game.away_team_id == 1610612749
        assert game.home_team == "Boston Celtics"
        assert game.away_team == "Milwaukee Bucks"

    def test_game_info_is_frozen(self) -> None:
        """GameInfo should be immutable (frozen)."""
        game = GameInfo(
            game_id="0022300001",
            season_id="2023-24",
            game_date=date(2024, 1, 15),
            home_team_id=1610612738,
            away_team_id=1610612749,
            home_team="Boston Celtics",
            away_team="Milwaukee Bucks",
        )
        with pytest.raises(FrozenInstanceError):
            game.game_id = "different"  # type: ignore[misc]

    def test_game_info_equality(self) -> None:
        """GameInfo instances with same data should be equal."""
        game1 = GameInfo(
            game_id="0022300001",
            season_id="2023-24",
            game_date=date(2024, 1, 15),
            home_team_id=1610612738,
            away_team_id=1610612749,
            home_team="Boston Celtics",
            away_team="Milwaukee Bucks",
        )
        game2 = GameInfo(
            game_id="0022300001",
            season_id="2023-24",
            game_date=date(2024, 1, 15),
            home_team_id=1610612738,
            away_team_id=1610612749,
            home_team="Boston Celtics",
            away_team="Milwaukee Bucks",
        )
        assert game1 == game2


class TestBet:
    """Tests for Bet dataclass."""

    def test_create_bet_with_required_fields(self) -> None:
        """Should create Bet with required fields."""
        bet = Bet(
            game_id="0022300001",
            timestamp=datetime(2024, 1, 15, 12, 0, 0),
            bet_type="moneyline",
            side="home",
            model_prob=0.58,
            market_odds=1.91,
            market_prob=0.52,
            edge=0.06,
            kelly_fraction=0.025,
            bet_amount=100.0,
        )
        assert bet.game_id == "0022300001"
        assert bet.timestamp == datetime(2024, 1, 15, 12, 0, 0)
        assert bet.bet_type == "moneyline"
        assert bet.side == "home"
        assert bet.model_prob == 0.58
        assert bet.market_odds == 1.91
        assert bet.market_prob == 0.52
        assert bet.edge == 0.06
        assert bet.kelly_fraction == 0.025
        assert bet.bet_amount == 100.0
        assert bet.result is None
        assert bet.profit is None

    def test_create_bet_with_result(self) -> None:
        """Should create Bet with result and profit."""
        bet = Bet(
            game_id="0022300001",
            timestamp=datetime(2024, 1, 15, 12, 0, 0),
            bet_type="moneyline",
            side="home",
            model_prob=0.58,
            market_odds=1.91,
            market_prob=0.52,
            edge=0.06,
            kelly_fraction=0.025,
            bet_amount=100.0,
            result="win",
            profit=91.0,
        )
        assert bet.result == "win"
        assert bet.profit == 91.0

    def test_bet_is_mutable(self) -> None:
        """Bet should be mutable (can update result after creation)."""
        bet = Bet(
            game_id="0022300001",
            timestamp=datetime(2024, 1, 15, 12, 0, 0),
            bet_type="moneyline",
            side="home",
            model_prob=0.58,
            market_odds=1.91,
            market_prob=0.52,
            edge=0.06,
            kelly_fraction=0.025,
            bet_amount=100.0,
        )
        bet.result = "win"
        bet.profit = 91.0
        assert bet.result == "win"
        assert bet.profit == 91.0


class TestBacktestMetrics:
    """Tests for BacktestMetrics dataclass."""

    def test_create_backtest_metrics(self) -> None:
        """Should create BacktestMetrics with all fields."""
        metrics = BacktestMetrics(
            total_return=0.25,
            cagr=0.15,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            max_drawdown=0.10,
            win_rate=0.55,
            total_bets=500,
            avg_edge=0.03,
            avg_clv=0.02,
            roi=0.05,
        )
        assert metrics.total_return == 0.25
        assert metrics.cagr == 0.15
        assert metrics.sharpe_ratio == 1.5
        assert metrics.sortino_ratio == 2.0
        assert metrics.max_drawdown == 0.10
        assert metrics.win_rate == 0.55
        assert metrics.total_bets == 500
        assert metrics.avg_edge == 0.03
        assert metrics.avg_clv == 0.02
        assert metrics.roi == 0.05


class TestModelMetadata:
    """Tests for ModelMetadata dataclass."""

    def test_create_model_metadata_with_required_fields(self) -> None:
        """Should create ModelMetadata with required fields."""
        metadata = ModelMetadata(
            version="v1.0.0",
            training_date=datetime(2024, 1, 15, 10, 0, 0),
            training_data_start=date(2020, 10, 1),
            training_data_end=date(2024, 1, 1),
            hyperparameters={"learning_rate": 0.001, "batch_size": 32},
            validation_metrics={"brier_score": 0.22, "log_loss": 0.65},
        )
        assert metadata.version == "v1.0.0"
        assert metadata.training_date == datetime(2024, 1, 15, 10, 0, 0)
        assert metadata.training_data_start == date(2020, 10, 1)
        assert metadata.training_data_end == date(2024, 1, 1)
        assert metadata.hyperparameters["learning_rate"] == 0.001
        assert metadata.validation_metrics["brier_score"] == 0.22
        assert metadata.git_commit is None

    def test_create_model_metadata_with_git_commit(self) -> None:
        """Should create ModelMetadata with git commit."""
        metadata = ModelMetadata(
            version="v1.0.0",
            training_date=datetime(2024, 1, 15, 10, 0, 0),
            training_data_start=date(2020, 10, 1),
            training_data_end=date(2024, 1, 1),
            hyperparameters={},
            validation_metrics={},
            git_commit="abc123def456",
        )
        assert metadata.git_commit == "abc123def456"


# =============================================================================
# Exception Tests
# =============================================================================


class TestNBAModelError:
    """Tests for NBAModelError exception."""

    def test_raise_nba_model_error(self) -> None:
        """Should be able to raise and catch NBAModelError."""
        with pytest.raises(NBAModelError, match="test error"):
            raise NBAModelError("test error")

    def test_nba_model_error_is_exception(self) -> None:
        """NBAModelError should be a subclass of Exception."""
        assert issubclass(NBAModelError, Exception)


class TestDataCollectionError:
    """Tests for DataCollectionError exception."""

    def test_raise_data_collection_error(self) -> None:
        """Should be able to raise and catch DataCollectionError."""
        with pytest.raises(DataCollectionError, match="collection failed"):
            raise DataCollectionError("collection failed")

    def test_data_collection_error_is_nba_model_error(self) -> None:
        """DataCollectionError should be a subclass of NBAModelError."""
        assert issubclass(DataCollectionError, NBAModelError)

    def test_catch_as_nba_model_error(self) -> None:
        """DataCollectionError should be catchable as NBAModelError."""
        with pytest.raises(NBAModelError):
            raise DataCollectionError("test")


class TestRateLimitExceeded:
    """Tests for RateLimitExceeded exception."""

    def test_raise_rate_limit_exceeded(self) -> None:
        """Should be able to raise and catch RateLimitExceeded."""
        with pytest.raises(RateLimitExceeded, match="rate limit"):
            raise RateLimitExceeded("rate limit hit")

    def test_rate_limit_exceeded_is_data_collection_error(self) -> None:
        """RateLimitExceeded should be a subclass of DataCollectionError."""
        assert issubclass(RateLimitExceeded, DataCollectionError)


class TestGameNotFound:
    """Tests for GameNotFound exception."""

    def test_raise_game_not_found(self) -> None:
        """Should be able to raise and catch GameNotFound."""
        with pytest.raises(GameNotFound, match="0022300001"):
            raise GameNotFound("Game 0022300001 not found")

    def test_game_not_found_is_data_collection_error(self) -> None:
        """GameNotFound should be a subclass of DataCollectionError."""
        assert issubclass(GameNotFound, DataCollectionError)


class TestInsufficientDataError:
    """Tests for InsufficientDataError exception."""

    def test_raise_insufficient_data_error(self) -> None:
        """Should be able to raise and catch InsufficientDataError."""
        with pytest.raises(InsufficientDataError, match="not enough"):
            raise InsufficientDataError("not enough data")

    def test_insufficient_data_error_is_nba_model_error(self) -> None:
        """InsufficientDataError should be a subclass of NBAModelError."""
        assert issubclass(InsufficientDataError, NBAModelError)


class TestModelNotFoundError:
    """Tests for ModelNotFoundError exception."""

    def test_raise_model_not_found_error(self) -> None:
        """Should be able to raise and catch ModelNotFoundError."""
        with pytest.raises(ModelNotFoundError, match="v1.0.0"):
            raise ModelNotFoundError("Model v1.0.0 not found")

    def test_model_not_found_error_is_nba_model_error(self) -> None:
        """ModelNotFoundError should be a subclass of NBAModelError."""
        assert issubclass(ModelNotFoundError, NBAModelError)


class TestDriftDetectedError:
    """Tests for DriftDetectedError exception."""

    def test_raise_drift_detected_error(self) -> None:
        """Should be able to raise and catch DriftDetectedError."""
        with pytest.raises(DriftDetectedError, match="drift detected"):
            raise DriftDetectedError("drift detected in features")

    def test_drift_detected_error_is_nba_model_error(self) -> None:
        """DriftDetectedError should be a subclass of NBAModelError."""
        assert issubclass(DriftDetectedError, NBAModelError)
