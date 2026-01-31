"""Unit tests for ChartGenerator class."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import pytest

from nba_model.output.charts import ChartGenerator


# =============================================================================
# Mock Data Classes
# =============================================================================


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
# Test ChartGenerator
# =============================================================================


class TestChartGenerator:
    """Tests for ChartGenerator class."""

    @pytest.fixture
    def generator(self) -> ChartGenerator:
        """Create ChartGenerator instance."""
        return ChartGenerator()

    # -------------------------------------------------------------------------
    # Bankroll Chart Tests
    # -------------------------------------------------------------------------

    def test_bankroll_chart_returns_chartjs_structure(
        self,
        generator: ChartGenerator,
    ) -> None:
        """Bankroll chart should return Chart.js compatible structure."""
        history = [10000.0, 10250.0, 10180.0, 10500.0, 10800.0]

        result = generator.bankroll_chart(history)

        assert "labels" in result
        assert "datasets" in result
        assert len(result["datasets"]) > 0
        assert "data" in result["datasets"][0]

    def test_bankroll_chart_labels_match_data_length(
        self,
        generator: ChartGenerator,
    ) -> None:
        """Bankroll chart labels should match data length."""
        history = [10000.0, 10250.0, 10500.0]

        result = generator.bankroll_chart(history)

        assert len(result["labels"]) == len(history)
        assert len(result["datasets"][0]["data"]) == len(history)

    def test_bankroll_chart_with_dates(
        self,
        generator: ChartGenerator,
    ) -> None:
        """Bankroll chart should use provided dates as labels."""
        history = [10000.0, 10500.0]
        dates = [date(2024, 1, 1), date(2024, 1, 2)]

        result = generator.bankroll_chart(history, dates=dates)

        assert result["labels"] == ["2024-01-01", "2024-01-02"]

    def test_bankroll_chart_empty_data(
        self,
        generator: ChartGenerator,
    ) -> None:
        """Bankroll chart should handle empty data."""
        result = generator.bankroll_chart([])

        assert result["labels"] == []
        assert result["datasets"][0]["data"] == []

    def test_bankroll_chart_color_for_profit(
        self,
        generator: ChartGenerator,
    ) -> None:
        """Bankroll chart should use green for profit."""
        history = [10000.0, 10500.0]  # Profitable

        result = generator.bankroll_chart(history)

        # Check dataset color contains green (secondary color)
        assert "rgb(16, 185, 129)" in result["datasets"][0]["borderColor"]

    def test_bankroll_chart_color_for_loss(
        self,
        generator: ChartGenerator,
    ) -> None:
        """Bankroll chart should use red for loss."""
        history = [10000.0, 9500.0]  # Loss

        result = generator.bankroll_chart(history)

        # Check dataset color contains red (danger color)
        assert "rgb(239, 68, 68)" in result["datasets"][0]["borderColor"]

    # -------------------------------------------------------------------------
    # Calibration Chart Tests
    # -------------------------------------------------------------------------

    def test_calibration_chart_returns_chartjs_structure(
        self,
        generator: ChartGenerator,
    ) -> None:
        """Calibration chart should return Chart.js compatible structure."""
        predictions = [0.55, 0.65, 0.45, 0.75, 0.35]
        actuals = [1, 1, 0, 1, 0]

        result = generator.calibration_chart(predictions, actuals)

        assert "labels" in result
        assert "datasets" in result
        assert len(result["datasets"]) >= 2  # Actual + Perfect line

    def test_calibration_chart_bins_predictions(
        self,
        generator: ChartGenerator,
    ) -> None:
        """Calibration chart should bin predictions correctly."""
        predictions = [0.15, 0.25, 0.55, 0.75, 0.85]
        actuals = [0, 0, 1, 1, 1]

        result = generator.calibration_chart(predictions, actuals)

        # Should have 10 bins
        assert len(result["labels"]) == 10
        assert "metadata" in result
        assert "total_predictions" in result["metadata"]
        assert result["metadata"]["total_predictions"] == 5

    def test_calibration_chart_perfect_line(
        self,
        generator: ChartGenerator,
    ) -> None:
        """Calibration chart should include perfect calibration line."""
        predictions = [0.5, 0.5]
        actuals = [1, 0]

        result = generator.calibration_chart(predictions, actuals)

        # Find perfect calibration dataset
        perfect_dataset = None
        for ds in result["datasets"]:
            if "Perfect" in ds.get("label", ""):
                perfect_dataset = ds
                break

        assert perfect_dataset is not None
        assert perfect_dataset["borderDash"] == [5, 5]

    def test_calibration_chart_mismatched_lengths(
        self,
        generator: ChartGenerator,
    ) -> None:
        """Calibration chart should raise on mismatched lengths."""
        with pytest.raises(ValueError, match="same length"):
            generator.calibration_chart([0.5, 0.6], [1])

    def test_calibration_chart_empty_data(
        self,
        generator: ChartGenerator,
    ) -> None:
        """Calibration chart should handle empty data."""
        result = generator.calibration_chart([], [])

        assert result["metadata"]["total_predictions"] == 0

    # -------------------------------------------------------------------------
    # ROI by Month Chart Tests
    # -------------------------------------------------------------------------

    def test_roi_by_month_returns_chartjs_structure(
        self,
        generator: ChartGenerator,
    ) -> None:
        """ROI by month chart should return Chart.js compatible structure."""
        bets = [
            MockBet(timestamp=datetime(2024, 1, 15), profit=50.0, bet_amount=100.0),
            MockBet(timestamp=datetime(2024, 1, 20), profit=-30.0, bet_amount=100.0),
            MockBet(timestamp=datetime(2024, 2, 10), profit=80.0, bet_amount=100.0),
        ]

        result = generator.roi_by_month_chart(bets)

        assert "labels" in result
        assert "datasets" in result
        assert len(result["datasets"]) > 0

    def test_roi_by_month_aggregates_correctly(
        self,
        generator: ChartGenerator,
    ) -> None:
        """ROI by month should aggregate by calendar month."""
        bets = [
            MockBet(timestamp=datetime(2024, 1, 5), profit=50.0, bet_amount=100.0),
            MockBet(timestamp=datetime(2024, 1, 25), profit=50.0, bet_amount=100.0),
            MockBet(timestamp=datetime(2024, 2, 10), profit=-50.0, bet_amount=100.0),
        ]

        result = generator.roi_by_month_chart(bets)

        # Should have 2 months
        assert len(result["labels"]) == 2
        assert "2024-01" in result["labels"]
        assert "2024-02" in result["labels"]

    def test_roi_by_month_sorts_chronologically(
        self,
        generator: ChartGenerator,
    ) -> None:
        """ROI by month should sort months chronologically."""
        bets = [
            MockBet(timestamp=datetime(2024, 3, 1), profit=30.0, bet_amount=100.0),
            MockBet(timestamp=datetime(2024, 1, 1), profit=50.0, bet_amount=100.0),
            MockBet(timestamp=datetime(2024, 2, 1), profit=40.0, bet_amount=100.0),
        ]

        result = generator.roi_by_month_chart(bets)

        assert result["labels"] == ["2024-01", "2024-02", "2024-03"]

    def test_roi_by_month_color_codes_positive_negative(
        self,
        generator: ChartGenerator,
    ) -> None:
        """ROI by month should color code positive/negative months."""
        bets = [
            MockBet(timestamp=datetime(2024, 1, 1), profit=50.0, bet_amount=100.0),  # +
            MockBet(timestamp=datetime(2024, 2, 1), profit=-50.0, bet_amount=100.0),  # -
        ]

        result = generator.roi_by_month_chart(bets)

        colors = result["datasets"][0]["backgroundColor"]
        assert len(colors) == 2
        # First should be green, second should be red
        assert "16, 185, 129" in colors[0]  # Green
        assert "239, 68, 68" in colors[1]  # Red

    def test_roi_by_month_includes_metadata(
        self,
        generator: ChartGenerator,
    ) -> None:
        """ROI by month should include monthly stats metadata."""
        bets = [
            MockBet(timestamp=datetime(2024, 1, 1), profit=50.0, bet_amount=100.0),
            MockBet(timestamp=datetime(2024, 1, 15), profit=50.0, bet_amount=100.0),
        ]

        result = generator.roi_by_month_chart(bets)

        assert "metadata" in result
        assert "monthly_stats" in result["metadata"]
        assert "2024-01" in result["metadata"]["monthly_stats"]

        jan_stats = result["metadata"]["monthly_stats"]["2024-01"]
        assert jan_stats["bets"] == 2
        assert jan_stats["profit"] == 100.0
        assert jan_stats["wagered"] == 200.0

    def test_roi_by_month_empty_data(
        self,
        generator: ChartGenerator,
    ) -> None:
        """ROI by month should handle empty data."""
        result = generator.roi_by_month_chart([])

        assert result["labels"] == []
        assert result["datasets"][0]["data"] == []

    # -------------------------------------------------------------------------
    # Win Rate Trend Chart Tests
    # -------------------------------------------------------------------------

    def test_win_rate_trend_returns_chartjs_structure(
        self,
        generator: ChartGenerator,
    ) -> None:
        """Win rate trend should return Chart.js compatible structure."""
        bets = [MockBet(result="win" if i % 2 == 0 else "loss") for i in range(100)]

        result = generator.win_rate_trend_chart(bets, window=50)

        assert "labels" in result
        assert "datasets" in result

    def test_win_rate_trend_rolling_window(
        self,
        generator: ChartGenerator,
    ) -> None:
        """Win rate trend should calculate rolling window correctly."""
        # 60 wins followed by 40 losses
        bets = (
            [MockBet(result="win") for _ in range(60)]
            + [MockBet(result="loss") for _ in range(40)]
        )

        result = generator.win_rate_trend_chart(bets, window=50)

        data = result["datasets"][0]["data"]
        # First point (at bet 50) should have high win rate
        # Last point should have low win rate
        assert data[0] > 0.8  # Mostly wins at start
        assert data[-1] < 0.5  # Mostly losses at end

    def test_win_rate_trend_includes_breakeven_line(
        self,
        generator: ChartGenerator,
    ) -> None:
        """Win rate trend should include break-even reference line."""
        bets = [MockBet(result="win") for _ in range(100)]

        result = generator.win_rate_trend_chart(bets, window=50)

        # Find break-even line dataset
        breakeven = None
        for ds in result["datasets"]:
            if "Break-even" in ds.get("label", "") or "52.4%" in ds.get("label", ""):
                breakeven = ds
                break

        assert breakeven is not None
        assert breakeven["data"][0] == 0.524

    def test_win_rate_trend_insufficient_data(
        self,
        generator: ChartGenerator,
    ) -> None:
        """Win rate trend should handle insufficient data."""
        bets = [MockBet() for _ in range(10)]  # Less than window

        result = generator.win_rate_trend_chart(bets, window=50)

        assert result["labels"] == []

    # -------------------------------------------------------------------------
    # Edge Distribution Chart Tests
    # -------------------------------------------------------------------------

    def test_edge_distribution_returns_chartjs_structure(
        self,
        generator: ChartGenerator,
    ) -> None:
        """Edge distribution should return Chart.js compatible structure."""
        bets = [MockBet(edge=i * 0.01) for i in range(1, 11)]

        result = generator.edge_distribution_chart(bets)

        assert "labels" in result
        assert "datasets" in result
        assert len(result["datasets"]) > 0

    def test_edge_distribution_bins_correctly(
        self,
        generator: ChartGenerator,
    ) -> None:
        """Edge distribution should bin edges into ranges."""
        bets = [MockBet(edge=0.02)] * 5 + [MockBet(edge=0.05)] * 3

        result = generator.edge_distribution_chart(bets)

        # Should have histogram bins
        assert len(result["labels"]) > 0
        assert all("-" in label for label in result["labels"])

    def test_edge_distribution_empty_data(
        self,
        generator: ChartGenerator,
    ) -> None:
        """Edge distribution should handle empty data."""
        result = generator.edge_distribution_chart([])

        assert result["labels"] == []
