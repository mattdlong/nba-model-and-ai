"""Integration tests for output pipeline.

Tests the full flow from predictions through report generation
to dashboard building.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import pytest


# =============================================================================
# Mock Data Classes
# =============================================================================


@dataclass
class MockGamePrediction:
    """Mock GamePrediction for integration testing."""

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
    top_factors: list = field(default_factory=lambda: [("home_court", 0.12)])
    model_version: str = "v1.0.0"
    prediction_timestamp: datetime = field(default_factory=datetime.now)
    home_lineup: list = field(default_factory=list)
    away_lineup: list = field(default_factory=list)
    inference_time_ms: float = 150.0


@dataclass
class MockBettingSignal:
    """Mock BettingSignal for integration testing."""

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
    key_factors: list = field(default_factory=lambda: ["home_court"])
    injury_notes: list = field(default_factory=list)
    model_confidence: float = 0.75
    injury_uncertainty: float = 0.15


@dataclass
class MockBet:
    """Mock Bet for integration testing."""

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
# Integration Tests
# =============================================================================


@pytest.mark.integration
class TestOutputPipeline:
    """Integration tests for the complete output pipeline."""

    @pytest.fixture
    def sample_predictions(self) -> list[MockGamePrediction]:
        """Create sample predictions."""
        return [
            MockGamePrediction(
                game_id="001",
                matchup="LAL @ BOS",
                home_win_prob=0.65,
                confidence=0.80,
            ),
            MockGamePrediction(
                game_id="002",
                matchup="GSW @ PHX",
                home_win_prob=0.45,
                confidence=0.70,
            ),
            MockGamePrediction(
                game_id="003",
                matchup="MIA @ NYK",
                home_win_prob=0.52,
                confidence=0.60,
            ),
        ]

    @pytest.fixture
    def sample_signals(self) -> list[MockBettingSignal]:
        """Create sample signals."""
        return [
            MockBettingSignal(
                game_id="001",
                matchup="LAL @ BOS",
                bet_type="moneyline",
                edge=0.07,
                confidence="high",
            ),
            MockBettingSignal(
                game_id="002",
                matchup="GSW @ PHX",
                bet_type="spread",
                edge=0.04,
                confidence="medium",
            ),
        ]

    @pytest.fixture
    def sample_bets(self) -> list[MockBet]:
        """Create sample historical bets."""
        bets = []
        for i in range(50):
            result = "win" if i % 2 == 0 else "loss"
            profit = 91.0 if result == "win" else -100.0
            bets.append(
                MockBet(
                    game_id=f"00{i:05d}",
                    timestamp=datetime(2024, 1, 1 + i // 2),
                    result=result,
                    profit=profit,
                )
            )
        return bets

    def test_full_dashboard_generation_flow(
        self,
        tmp_path: Path,
        sample_predictions: list[MockGamePrediction],
        sample_signals: list[MockBettingSignal],
        sample_bets: list[MockBet],
    ) -> None:
        """Test complete flow from predictions to dashboard."""
        from nba_model.output import ChartGenerator, DashboardBuilder, ReportGenerator

        output_dir = tmp_path / "docs"

        # Step 1: Generate reports
        report_gen = ReportGenerator()
        daily_report = report_gen.daily_predictions_report(
            sample_predictions, sample_signals
        )

        assert daily_report["summary"]["total_games"] == 3
        assert daily_report["summary"]["total_signals"] == 2

        # Step 2: Generate chart data
        chart_gen = ChartGenerator()
        bankroll_history = [10000.0, 10250.0, 10180.0, 10500.0]
        bankroll_chart = chart_gen.bankroll_chart(bankroll_history)

        assert len(bankroll_chart["labels"]) == 4

        # Step 3: Build dashboard
        builder = DashboardBuilder(output_dir=output_dir)
        builder.update_predictions(sample_predictions, sample_signals)
        file_count = builder.build_full_site()

        assert file_count > 0
        assert (output_dir / "api" / "today.json").exists()
        assert (output_dir / "api" / "signals.json").exists()

        # Verify JSON content
        today_data = json.loads((output_dir / "api" / "today.json").read_text())
        assert len(today_data["games"]) == 3
        assert len(today_data["signals"]) == 2

    def test_report_to_dashboard_data_integrity(
        self,
        tmp_path: Path,
        sample_predictions: list[MockGamePrediction],
        sample_signals: list[MockBettingSignal],
    ) -> None:
        """Test that data flows correctly from reports to dashboard."""
        from nba_model.output import DashboardBuilder, ReportGenerator

        output_dir = tmp_path / "docs"

        # Generate report
        report_gen = ReportGenerator()
        daily_report = report_gen.daily_predictions_report(
            sample_predictions, sample_signals
        )

        # Build dashboard
        builder = DashboardBuilder(output_dir=output_dir)
        builder.update_predictions(sample_predictions, sample_signals)

        # Read and compare
        today_data = json.loads((output_dir / "api" / "today.json").read_text())

        # Verify game data matches
        assert len(today_data["games"]) == len(daily_report["games"])
        for i, game in enumerate(today_data["games"]):
            assert game["game_id"] == daily_report["games"][i]["game_id"]
            assert game["matchup"] == daily_report["games"][i]["matchup"]

        # Verify signals match
        assert len(today_data["signals"]) == len(daily_report["signals"])

    def test_multiple_day_archiving(
        self,
        tmp_path: Path,
        sample_predictions: list[MockGamePrediction],
        sample_signals: list[MockBettingSignal],
    ) -> None:
        """Test archiving predictions over multiple days."""
        from nba_model.output import DashboardBuilder

        output_dir = tmp_path / "docs"
        builder = DashboardBuilder(output_dir=output_dir)

        # Day 1
        builder.update_predictions(sample_predictions, sample_signals)
        builder.archive_day(date(2024, 1, 1))

        # Day 2 - fewer predictions
        day2_preds = sample_predictions[:2]
        builder.update_predictions(day2_preds, sample_signals[:1])
        builder.archive_day(date(2024, 1, 2))

        # Verify archives
        history_dir = output_dir / "api" / "history"
        assert (history_dir / "2024-01-01.json").exists()
        assert (history_dir / "2024-01-02.json").exists()

        # Verify content differs
        day1_data = json.loads((history_dir / "2024-01-01.json").read_text())
        day2_data = json.loads((history_dir / "2024-01-02.json").read_text())

        assert len(day1_data["games"]) == 3
        assert len(day2_data["games"]) == 2

    def test_chart_generation_with_betting_data(
        self,
        sample_bets: list[MockBet],
    ) -> None:
        """Test chart generation with realistic betting data."""
        from nba_model.output import ChartGenerator

        chart_gen = ChartGenerator()

        # Generate charts
        roi_chart = chart_gen.roi_by_month_chart(sample_bets)
        win_trend = chart_gen.win_rate_trend_chart(sample_bets, window=20)

        # ROI chart should have monthly data
        assert len(roi_chart["labels"]) > 0
        assert "metadata" in roi_chart

        # Win rate trend should have rolling data
        assert len(win_trend["datasets"]) >= 2  # Actual + break-even

    def test_model_health_integration(
        self,
        tmp_path: Path,
    ) -> None:
        """Test model health report integration with dashboard."""
        from nba_model.output import DashboardBuilder, ReportGenerator

        output_dir = tmp_path / "docs"

        # Create drift results
        drift_result = {
            "has_drift": True,
            "features_drifted": ["pace"],
            "details": {
                "pace": {"ks_stat": 0.15, "p_value": 0.02, "psi": 0.22},
                "offensive_rating": {"ks_stat": 0.08, "p_value": 0.12, "psi": 0.09},
            },
        }
        recent_metrics = {
            "accuracy": 0.51,
            "brier_score": 0.24,
            "roi": 0.02,
        }

        # Generate health report
        report_gen = ReportGenerator()
        health_report = report_gen.model_health_report(drift_result, recent_metrics)

        assert health_report["status"] == "warning"
        assert health_report["drift_detected"] is True

        # Update dashboard
        builder = DashboardBuilder(output_dir=output_dir)
        builder.update_model_health(
            drift_results=drift_result,
            recent_metrics=recent_metrics,
        )

        # Build site
        builder.build_full_site()

        assert output_dir.exists()

    def test_calibration_chart_with_realistic_data(self) -> None:
        """Test calibration chart with diverse predictions."""
        from nba_model.output import ChartGenerator

        # Create predictions across the probability spectrum
        predictions = []
        actuals = []

        # Well-calibrated model simulation
        import random

        random.seed(42)
        for _ in range(200):
            pred = random.uniform(0.2, 0.8)
            actual = 1 if random.random() < pred else 0
            predictions.append(pred)
            actuals.append(actual)

        chart_gen = ChartGenerator()
        cal_chart = chart_gen.calibration_chart(predictions, actuals)

        # Should have binned data
        assert len(cal_chart["labels"]) == 10
        assert cal_chart["metadata"]["total_predictions"] == 200

        # Actual rates should exist for populated bins
        actual_data = cal_chart["datasets"][0]["data"]
        non_null_rates = [r for r in actual_data if r is not None]
        assert len(non_null_rates) > 0

    def test_dashboard_json_files_are_valid(
        self,
        tmp_path: Path,
        sample_predictions: list[MockGamePrediction],
        sample_signals: list[MockBettingSignal],
    ) -> None:
        """Test that all generated JSON files are valid and parseable."""
        from nba_model.output import DashboardBuilder

        output_dir = tmp_path / "docs"
        builder = DashboardBuilder(output_dir=output_dir)

        builder.update_predictions(sample_predictions, sample_signals)
        builder.update_performance(bankroll_history=[10000.0, 10500.0])
        builder.build_full_site()

        # Check all JSON files
        json_files = [
            output_dir / "api" / "today.json",
            output_dir / "api" / "signals.json",
            output_dir / "api" / "performance.json",
        ]

        for json_file in json_files:
            assert json_file.exists(), f"{json_file} should exist"
            content = json_file.read_text()

            # Should be valid JSON
            data = json.loads(content)
            assert isinstance(data, dict)

    def test_empty_state_handling(
        self,
        tmp_path: Path,
    ) -> None:
        """Test dashboard handles empty/initial state gracefully."""
        from nba_model.output import DashboardBuilder

        output_dir = tmp_path / "docs"
        builder = DashboardBuilder(output_dir=output_dir)

        # Build with no data
        file_count = builder.build_full_site()

        assert file_count > 0
        assert (output_dir / "api" / "today.json").exists()

        # JSON should have empty but valid structure
        today_data = json.loads((output_dir / "api" / "today.json").read_text())
        assert "games" in today_data
        assert "signals" in today_data
        assert today_data["games"] == []
