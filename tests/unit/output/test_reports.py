"""Unit tests for ReportGenerator class."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING

import pytest

from nba_model.output.reports import (
    InvalidPeriodError,
    ReportGenerator,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# Mock Data Classes
# =============================================================================


@dataclass
class MockGamePrediction:
    """Mock GamePrediction for testing."""

    game_id: str = "0022400001"
    game_date: date = field(default_factory=date.today)
    home_team: str = "BOS"
    away_team: str = "LAL"
    matchup: str = "LAL @ BOS"
    home_win_prob: float = 0.65
    predicted_margin: float = 5.5
    predicted_total: float = 220.5
    home_win_prob_adjusted: float = 0.62
    predicted_margin_adjusted: float = 4.8
    predicted_total_adjusted: float = 218.0
    confidence: float = 0.75
    injury_uncertainty: float = 0.15
    top_factors: list = field(default_factory=lambda: [("home_court", 0.12), ("rest", 0.08)])
    model_version: str = "v1.0.0"
    prediction_timestamp: datetime = field(default_factory=datetime.now)
    home_lineup: list = field(default_factory=list)
    away_lineup: list = field(default_factory=list)
    inference_time_ms: float = 150.0


@dataclass
class MockBettingSignal:
    """Mock BettingSignal for testing."""

    game_id: str = "0022400001"
    game_date: date = field(default_factory=date.today)
    matchup: str = "LAL @ BOS"
    bet_type: str = "moneyline"
    side: str = "home"
    line: float | None = None
    model_prob: float = 0.62
    market_prob: float = 0.55
    edge: float = 0.07
    recommended_odds: float = 1.82
    kelly_fraction: float = 0.08
    recommended_stake_pct: float = 0.02
    confidence: str = "high"
    key_factors: list = field(default_factory=lambda: ["home_court", "rest_advantage"])
    injury_notes: list = field(default_factory=list)
    model_confidence: float = 0.75
    injury_uncertainty: float = 0.15


@dataclass
class MockFullBacktestMetrics:
    """Mock FullBacktestMetrics for testing."""

    total_return: float = 0.12
    cagr: float = 0.08
    avg_bet_return: float = 0.015
    volatility: float = 0.05
    sharpe_ratio: float = 1.2
    sortino_ratio: float = 1.5
    max_drawdown: float = 0.08
    max_drawdown_duration: int = 5
    total_bets: int = 100
    win_rate: float = 0.54
    avg_edge: float = 0.03
    avg_odds: float = 1.91
    roi: float = 0.05
    brier_score: float = 0.22
    log_loss: float = 0.65
    avg_clv: float = 0.012
    clv_positive_rate: float = 0.62
    metrics_by_type: dict = field(default_factory=lambda: {
        "moneyline": {"count": 50, "win_rate": 0.55, "roi": 0.06},
        "spread": {"count": 50, "win_rate": 0.53, "roi": 0.04},
    })
    total_wagered: float = 5000.0
    total_profit: float = 250.0
    win_count: int = 54
    loss_count: int = 46
    push_count: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_return": self.total_return,
            "cagr": self.cagr,
            "avg_bet_return": self.avg_bet_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration": self.max_drawdown_duration,
            "total_bets": self.total_bets,
            "win_rate": self.win_rate,
            "avg_edge": self.avg_edge,
            "avg_odds": self.avg_odds,
            "roi": self.roi,
            "brier_score": self.brier_score,
            "log_loss": self.log_loss,
            "avg_clv": self.avg_clv,
            "clv_positive_rate": self.clv_positive_rate,
            "metrics_by_type": self.metrics_by_type,
            "total_wagered": self.total_wagered,
            "total_profit": self.total_profit,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "push_count": self.push_count,
        }


@dataclass
class MockBet:
    """Mock Bet for testing."""

    game_id: str = "0022400001"
    timestamp: datetime = field(default_factory=datetime.now)
    bet_type: str = "moneyline"
    side: str = "home"
    model_prob: float = 0.55
    market_odds: float = 1.91
    market_prob: float = 0.52
    edge: float = 0.03
    kelly_fraction: float = 0.03
    bet_amount: float = 100.0
    result: str | None = "win"
    profit: float | None = 91.0


# =============================================================================
# Test ReportGenerator
# =============================================================================


class TestReportGenerator:
    """Tests for ReportGenerator class."""

    @pytest.fixture
    def generator(self) -> ReportGenerator:
        """Create ReportGenerator instance."""
        return ReportGenerator()

    @pytest.fixture
    def sample_predictions(self) -> list[MockGamePrediction]:
        """Create sample predictions."""
        return [
            MockGamePrediction(game_id="001", matchup="LAL @ BOS"),
            MockGamePrediction(game_id="002", matchup="GSW @ PHX", home_win_prob=0.45),
            MockGamePrediction(game_id="003", matchup="MIA @ NYK", confidence=0.85),
        ]

    @pytest.fixture
    def sample_signals(self) -> list[MockBettingSignal]:
        """Create sample signals."""
        return [
            MockBettingSignal(game_id="001", edge=0.07, confidence="high"),
            MockBettingSignal(game_id="002", edge=0.04, confidence="medium"),
            MockBettingSignal(game_id="003", edge=0.025, confidence="low"),
        ]

    # -------------------------------------------------------------------------
    # Daily Predictions Report Tests
    # -------------------------------------------------------------------------

    def test_daily_predictions_report_returns_dict(
        self,
        generator: ReportGenerator,
        sample_predictions: list[MockGamePrediction],
        sample_signals: list[MockBettingSignal],
    ) -> None:
        """Daily report should return dictionary with required keys."""
        report = generator.daily_predictions_report(sample_predictions, sample_signals)

        assert isinstance(report, dict)
        assert "date" in report
        assert "generated_at" in report
        assert "games" in report
        assert "signals" in report
        assert "summary" in report

    def test_daily_predictions_report_formats_games(
        self,
        generator: ReportGenerator,
        sample_predictions: list[MockGamePrediction],
        sample_signals: list[MockBettingSignal],
    ) -> None:
        """Daily report should format game predictions correctly."""
        report = generator.daily_predictions_report(sample_predictions, sample_signals)

        assert len(report["games"]) == 3
        game = report["games"][0]
        assert "game_id" in game
        assert "matchup" in game
        assert "home_win_prob" in game
        assert "predicted_margin" in game
        assert "confidence" in game

    def test_daily_predictions_report_formats_signals(
        self,
        generator: ReportGenerator,
        sample_predictions: list[MockGamePrediction],
        sample_signals: list[MockBettingSignal],
    ) -> None:
        """Daily report should format signals correctly."""
        report = generator.daily_predictions_report(sample_predictions, sample_signals)

        assert len(report["signals"]) == 3
        signal = report["signals"][0]
        assert "game_id" in signal
        assert "bet_type" in signal
        assert "edge" in signal
        assert "confidence" in signal
        assert "recommended_stake_pct" in signal

    def test_daily_predictions_report_calculates_summary(
        self,
        generator: ReportGenerator,
        sample_predictions: list[MockGamePrediction],
        sample_signals: list[MockBettingSignal],
    ) -> None:
        """Daily report should calculate summary statistics."""
        report = generator.daily_predictions_report(sample_predictions, sample_signals)

        summary = report["summary"]
        assert summary["total_games"] == 3
        assert summary["total_signals"] == 3
        assert summary["high_confidence_signals"] == 1
        assert summary["medium_confidence_signals"] == 1

    def test_daily_predictions_report_empty_lists(
        self,
        generator: ReportGenerator,
    ) -> None:
        """Daily report should handle empty lists gracefully."""
        report = generator.daily_predictions_report([], [])

        assert report["games"] == []
        assert report["signals"] == []
        assert report["summary"]["total_games"] == 0
        assert report["summary"]["total_signals"] == 0

    # -------------------------------------------------------------------------
    # Performance Report Tests
    # -------------------------------------------------------------------------

    def test_performance_report_with_metrics(
        self,
        generator: ReportGenerator,
    ) -> None:
        """Performance report should format metrics correctly."""
        metrics = MockFullBacktestMetrics()
        report = generator.performance_report("week", metrics=metrics)

        assert report["period"] == "week"
        assert report["total_predictions"] == 100
        assert report["accuracy"] == 0.54
        assert report["roi"] == 0.05
        assert "by_bet_type" in report

    def test_performance_report_with_bets(
        self,
        generator: ReportGenerator,
    ) -> None:
        """Performance report should calculate from bets list."""
        bets = [
            MockBet(result="win", profit=91.0, bet_amount=100.0),
            MockBet(result="win", profit=91.0, bet_amount=100.0),
            MockBet(result="loss", profit=-100.0, bet_amount=100.0),
        ]
        report = generator.performance_report("month", bets=bets)

        assert report["period"] == "month"
        assert report["total_predictions"] == 3
        assert report["win_count"] == 2
        assert report["loss_count"] == 1

    def test_performance_report_invalid_period(
        self,
        generator: ReportGenerator,
    ) -> None:
        """Performance report should raise on invalid period."""
        with pytest.raises(InvalidPeriodError, match="Invalid period"):
            generator.performance_report("invalid_period")

    def test_performance_report_valid_periods(
        self,
        generator: ReportGenerator,
    ) -> None:
        """Performance report should accept valid periods."""
        for period in ["week", "month", "season"]:
            report = generator.performance_report(period)
            assert report["period"] == period

    def test_performance_report_empty(
        self,
        generator: ReportGenerator,
    ) -> None:
        """Performance report should handle no data gracefully."""
        report = generator.performance_report("week")

        assert report["total_predictions"] == 0
        assert report["accuracy"] == 0.0
        assert report["roi"] == 0.0

    # -------------------------------------------------------------------------
    # Model Health Report Tests
    # -------------------------------------------------------------------------

    def test_model_health_report_healthy(
        self,
        generator: ReportGenerator,
    ) -> None:
        """Model health report should show healthy status when no drift."""
        drift_result = {
            "has_drift": False,
            "features_drifted": [],
            "details": {},
        }
        metrics = {"accuracy": 0.55, "brier_score": 0.22}

        report = generator.model_health_report(drift_result, metrics)

        assert report["status"] == "healthy"
        assert report["drift_detected"] is False
        assert report["retraining_recommended"] is False

    def test_model_health_report_with_drift(
        self,
        generator: ReportGenerator,
    ) -> None:
        """Model health report should detect drift."""
        drift_result = {
            "has_drift": True,
            "features_drifted": ["pace", "offensive_rating"],
            "details": {
                "pace": {"ks_stat": 0.15, "p_value": 0.02, "psi": 0.25},
                "offensive_rating": {"ks_stat": 0.12, "p_value": 0.03, "psi": 0.18},
            },
        }
        metrics = {"accuracy": 0.50, "brier_score": 0.24}

        report = generator.model_health_report(drift_result, metrics)

        assert report["drift_detected"] is True
        assert len(report["features_drifted"]) == 2
        assert report["retraining_recommended"] is True
        assert "covariate drift" in report["recommendation_reason"].lower()

    def test_model_health_report_critical_status(
        self,
        generator: ReportGenerator,
    ) -> None:
        """Model health report should show critical on severe issues."""
        drift_result = {
            "has_drift": True,
            "features_drifted": ["pace", "offensive_rating", "rest_days"],
            "details": {},
        }
        metrics = {"accuracy": 0.45, "brier_score": 0.32}

        report = generator.model_health_report(drift_result, metrics)

        assert report["status"] == "critical"
        assert report["retraining_recommended"] is True

    def test_model_health_report_feature_stability(
        self,
        generator: ReportGenerator,
    ) -> None:
        """Model health report should include feature stability metrics."""
        drift_result = {
            "has_drift": False,
            "features_drifted": [],
            "details": {
                "pace": {"ks_stat": 0.05, "p_value": 0.15, "psi": 0.08},
            },
        }

        report = generator.model_health_report(drift_result)

        assert "feature_stability" in report
        assert "pace" in report["feature_stability"]
        assert report["feature_stability"]["pace"]["status"] == "stable"

    def test_model_health_report_none_inputs(
        self,
        generator: ReportGenerator,
    ) -> None:
        """Model health report should handle None inputs gracefully."""
        report = generator.model_health_report(None, None)

        assert report["status"] == "healthy"
        assert report["drift_detected"] is False
        assert "generated_at" in report
