"""Tests for backtest performance metrics."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from nba_model.backtest.metrics import (
    BacktestMetricsCalculator,
    CLVResult,
    FullBacktestMetrics,
    calculate_calibration_curve,
)
from nba_model.types import Bet


class TestFullBacktestMetrics:
    """Tests for FullBacktestMetrics dataclass."""

    def test_default_values(self) -> None:
        """Test default metric values."""
        metrics = FullBacktestMetrics()
        assert metrics.total_return == 0.0
        assert metrics.total_bets == 0
        assert metrics.win_rate == 0.0
        assert isinstance(metrics.metrics_by_type, dict)

    def test_custom_values(self) -> None:
        """Test custom metric values."""
        metrics = FullBacktestMetrics(
            total_return=0.15,
            sharpe_ratio=1.5,
            win_rate=0.55,
            total_bets=100,
        )
        assert metrics.total_return == 0.15
        assert metrics.sharpe_ratio == 1.5
        assert metrics.win_rate == 0.55
        assert metrics.total_bets == 100


class TestBacktestMetricsCalculator:
    """Tests for BacktestMetricsCalculator class."""

    @pytest.fixture
    def calc(self) -> BacktestMetricsCalculator:
        """Create calculator fixture."""
        return BacktestMetricsCalculator()

    @pytest.fixture
    def sample_bets(self) -> list[Bet]:
        """Create sample bet history with mixed outcomes."""
        bets = []
        # 10 bets: 6 wins, 4 losses
        for i in range(10):
            won = i < 6
            bets.append(
                Bet(
                    game_id=f"G{i}",
                    timestamp=datetime(2023, 1, i + 1),
                    bet_type="moneyline",
                    side="home",
                    model_prob=0.55,
                    market_odds=1.91,
                    market_prob=0.523,
                    edge=0.027,
                    kelly_fraction=0.05,
                    bet_amount=100.0,
                    result="win" if won else "loss",
                    profit=91.0 if won else -100.0,
                )
            )
        return bets

    def test_empty_bets_returns_default(
        self, calc: BacktestMetricsCalculator
    ) -> None:
        """Test empty bet list returns default metrics."""
        metrics = calc.calculate_from_bets([])
        assert metrics.total_bets == 0
        assert metrics.win_rate == 0.0

    def test_calculates_win_rate(
        self, calc: BacktestMetricsCalculator, sample_bets: list[Bet]
    ) -> None:
        """Test win rate calculation."""
        metrics = calc.calculate_from_bets(sample_bets)
        # 6 wins out of 10 bets
        assert metrics.win_rate == 0.6
        assert metrics.win_count == 6
        assert metrics.loss_count == 4

    def test_calculates_roi(
        self, calc: BacktestMetricsCalculator, sample_bets: list[Bet]
    ) -> None:
        """Test ROI calculation."""
        metrics = calc.calculate_from_bets(sample_bets)
        # 6 wins * 91 + 4 losses * -100 = 546 - 400 = 146
        # Total wagered = 1000
        # ROI = 146/1000 = 0.146
        assert abs(metrics.roi - 0.146) < 0.01
        assert metrics.total_wagered == 1000.0
        assert abs(metrics.total_profit - 146.0) < 0.1

    def test_calculates_total_return(
        self, calc: BacktestMetricsCalculator, sample_bets: list[Bet]
    ) -> None:
        """Test total return calculation."""
        metrics = calc.calculate_from_bets(sample_bets, initial_bankroll=10000.0)
        # Profit = 146, Initial = 10000
        # Return = 146/10000 = 0.0146
        assert abs(metrics.total_return - 0.0146) < 0.001

    def test_calculates_avg_edge(
        self, calc: BacktestMetricsCalculator, sample_bets: list[Bet]
    ) -> None:
        """Test average edge calculation."""
        metrics = calc.calculate_from_bets(sample_bets)
        assert abs(metrics.avg_edge - 0.027) < 0.001

    def test_calculates_avg_odds(
        self, calc: BacktestMetricsCalculator, sample_bets: list[Bet]
    ) -> None:
        """Test average odds calculation."""
        metrics = calc.calculate_from_bets(sample_bets)
        assert abs(metrics.avg_odds - 1.91) < 0.001

    def test_calculates_volatility(
        self, calc: BacktestMetricsCalculator, sample_bets: list[Bet]
    ) -> None:
        """Test volatility calculation."""
        metrics = calc.calculate_from_bets(sample_bets)
        # With mixed wins/losses, should have some volatility
        assert metrics.volatility > 0

    def test_calculates_brier_score(
        self, calc: BacktestMetricsCalculator, sample_bets: list[Bet]
    ) -> None:
        """Test Brier score calculation."""
        metrics = calc.calculate_from_bets(sample_bets)
        # Brier = mean((prob - outcome)^2)
        # prob = 0.55 for all
        # For wins (6): (0.55 - 1)^2 = 0.2025
        # For losses (4): (0.55 - 0)^2 = 0.3025
        # Mean = (6*0.2025 + 4*0.3025) / 10 = 0.2425
        assert abs(metrics.brier_score - 0.2425) < 0.01

    def test_calculates_log_loss(
        self, calc: BacktestMetricsCalculator, sample_bets: list[Bet]
    ) -> None:
        """Test log loss calculation."""
        metrics = calc.calculate_from_bets(sample_bets)
        # Log loss should be positive
        assert metrics.log_loss > 0

    def test_calculates_metrics_by_type(
        self, calc: BacktestMetricsCalculator
    ) -> None:
        """Test metrics segmentation by bet type."""
        bets = [
            Bet(
                game_id="G1",
                timestamp=datetime(2023, 1, 1),
                bet_type="moneyline",
                side="home",
                model_prob=0.55,
                market_odds=1.91,
                market_prob=0.523,
                edge=0.027,
                kelly_fraction=0.05,
                bet_amount=100.0,
                result="win",
                profit=91.0,
            ),
            Bet(
                game_id="G2",
                timestamp=datetime(2023, 1, 2),
                bet_type="spread",
                side="home",
                model_prob=0.52,
                market_odds=1.91,
                market_prob=0.50,
                edge=0.02,
                kelly_fraction=0.03,
                bet_amount=100.0,
                result="loss",
                profit=-100.0,
            ),
        ]
        metrics = calc.calculate_from_bets(bets)

        assert "moneyline" in metrics.metrics_by_type
        assert "spread" in metrics.metrics_by_type
        assert metrics.metrics_by_type["moneyline"]["win_rate"] == 1.0
        assert metrics.metrics_by_type["spread"]["win_rate"] == 0.0


class TestCLVCalculation:
    """Tests for CLV calculation."""

    @pytest.fixture
    def calc(self) -> BacktestMetricsCalculator:
        """Create calculator fixture."""
        return BacktestMetricsCalculator()

    def test_calculate_clv_positive(
        self, calc: BacktestMetricsCalculator
    ) -> None:
        """Test positive CLV when closing line moved favorably."""
        bet = Bet(
            game_id="G1",
            timestamp=datetime(2023, 1, 1),
            bet_type="moneyline",
            side="home",
            model_prob=0.55,
            market_odds=2.0,  # 50% implied
            market_prob=0.50,
            edge=0.05,
            kelly_fraction=0.1,
            bet_amount=100.0,
            result="win",
            profit=100.0,
        )
        # Closing odds 1.8 = 55.6% implied
        result = calc.calculate_clv(bet, closing_odds=1.8)

        assert result is not None
        assert result.clv > 0  # Positive CLV
        assert abs(result.bet_implied_prob - 0.5) < 0.01
        assert abs(result.closing_implied_prob - 0.556) < 0.01

    def test_calculate_clv_negative(
        self, calc: BacktestMetricsCalculator
    ) -> None:
        """Test negative CLV when closing line moved against."""
        bet = Bet(
            game_id="G1",
            timestamp=datetime(2023, 1, 1),
            bet_type="moneyline",
            side="home",
            model_prob=0.55,
            market_odds=1.8,  # 55.6% implied
            market_prob=0.556,
            edge=0.0,
            kelly_fraction=0.0,
            bet_amount=100.0,
            result="loss",
            profit=-100.0,
        )
        # Closing odds 2.0 = 50% implied (moved against)
        result = calc.calculate_clv(bet, closing_odds=2.0)

        assert result is not None
        assert result.clv < 0  # Negative CLV

    def test_calculate_clv_invalid_odds(
        self, calc: BacktestMetricsCalculator
    ) -> None:
        """Test CLV returns None for invalid odds."""
        bet = Bet(
            game_id="G1",
            timestamp=datetime(2023, 1, 1),
            bet_type="moneyline",
            side="home",
            model_prob=0.55,
            market_odds=1.0,  # Invalid
            market_prob=1.0,
            edge=0.0,
            kelly_fraction=0.0,
            bet_amount=0.0,
            result=None,
            profit=None,
        )
        result = calc.calculate_clv(bet, closing_odds=2.0)
        assert result is None

    def test_clv_metrics_with_closing_odds(
        self, calc: BacktestMetricsCalculator
    ) -> None:
        """Test CLV metrics are calculated when closing odds provided."""
        bets = [
            Bet(
                game_id="G1",
                timestamp=datetime(2023, 1, 1),
                bet_type="moneyline",
                side="home",
                model_prob=0.55,
                market_odds=2.0,
                market_prob=0.50,
                edge=0.05,
                kelly_fraction=0.1,
                bet_amount=100.0,
                result="win",
                profit=100.0,
            ),
        ]
        closing_odds = {"G1": 1.8}  # Favorable movement

        metrics = calc.calculate_from_bets(bets, closing_odds=closing_odds)
        assert metrics.avg_clv > 0
        assert metrics.clv_positive_rate == 1.0


class TestDrawdownCalculation:
    """Tests for drawdown calculation."""

    @pytest.fixture
    def calc(self) -> BacktestMetricsCalculator:
        """Create calculator fixture."""
        return BacktestMetricsCalculator()

    def test_max_drawdown_calculation(
        self, calc: BacktestMetricsCalculator
    ) -> None:
        """Test max drawdown is calculated correctly."""
        bankroll_history = [10000, 11000, 10500, 9500, 10200]
        # Peak 11000, trough 9500 = 13.6% drawdown

        drawdown, duration = calc._calculate_drawdown_metrics(bankroll_history)
        assert abs(drawdown - 0.136) < 0.01

    def test_no_drawdown(self, calc: BacktestMetricsCalculator) -> None:
        """Test no drawdown when always increasing."""
        bankroll_history = [10000, 10500, 11000, 11500]
        drawdown, duration = calc._calculate_drawdown_metrics(bankroll_history)
        assert drawdown == 0.0

    def test_empty_history(self, calc: BacktestMetricsCalculator) -> None:
        """Test empty history returns zero."""
        assert calc._calculate_drawdown_metrics([]) == (0.0, 0)
        assert calc._calculate_drawdown_metrics([10000]) == (0.0, 0)


class TestReportGeneration:
    """Tests for report generation."""

    @pytest.fixture
    def calc(self) -> BacktestMetricsCalculator:
        """Create calculator fixture."""
        return BacktestMetricsCalculator()

    def test_generate_report_format(
        self, calc: BacktestMetricsCalculator
    ) -> None:
        """Test report generation produces formatted output."""
        metrics = FullBacktestMetrics(
            total_return=0.15,
            sharpe_ratio=1.5,
            win_rate=0.55,
            total_bets=100,
            roi=0.05,
        )
        report = calc.generate_report(metrics)

        assert "RETURNS" in report
        assert "RISK METRICS" in report
        assert "BETTING STATS" in report
        assert "15.00%" in report  # total return
        assert "1.50" in report  # sharpe ratio
        assert "55.00%" in report  # win rate


class TestCalibrationCurve:
    """Tests for calibration curve calculation."""

    def test_calibration_curve_bins(self) -> None:
        """Test calibration curve produces correct bins."""
        bets = [
            Bet(
                game_id=f"G{i}",
                timestamp=datetime(2023, 1, 1),
                bet_type="moneyline",
                side="home",
                model_prob=0.1 * i,  # 0 to 0.9
                market_odds=2.0,
                market_prob=0.5,
                edge=0.0,
                kelly_fraction=0.0,
                bet_amount=100.0,
                result="win" if i >= 5 else "loss",
                profit=100.0 if i >= 5 else -100.0,
            )
            for i in range(1, 10)
        ]

        centers, rates, counts = calculate_calibration_curve(bets, n_bins=5)

        assert len(centers) > 0
        assert len(rates) == len(centers)
        assert len(counts) == len(centers)
        assert sum(counts) == len(bets)

    def test_calibration_curve_empty(self) -> None:
        """Test calibration curve handles empty input."""
        centers, rates, counts = calculate_calibration_curve([])
        assert centers == []
        assert rates == []
        assert counts == []
