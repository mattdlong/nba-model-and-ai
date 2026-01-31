"""Tests for Kelly Criterion bet sizing."""

from __future__ import annotations

from datetime import datetime

import pytest

from nba_model.backtest.kelly import (
    InvalidInputError,
    KellyCalculator,
    KellyOptimizationResult,
    KellyResult,
    SimulationResult,
)
from nba_model.types import Bet


class TestKellyCalculator:
    """Tests for KellyCalculator class."""

    @pytest.fixture
    def calc(self) -> KellyCalculator:
        """Create calculator with default settings."""
        return KellyCalculator(fraction=0.25, max_bet_pct=0.02, min_edge_pct=0.02)

    @pytest.fixture
    def full_kelly_calc(self) -> KellyCalculator:
        """Create calculator with full Kelly."""
        return KellyCalculator(fraction=1.0, max_bet_pct=1.0, min_edge_pct=0.0)

    def test_full_kelly_formula(self, full_kelly_calc: KellyCalculator) -> None:
        """Test full Kelly calculation matches formula."""
        # f* = (bp - q) / b
        # p=0.6, odds=2.0, b=1.0, q=0.4
        # f* = (1.0 * 0.6 - 0.4) / 1.0 = 0.2
        kelly = full_kelly_calc.calculate_full_kelly(0.6, 2.0)
        assert abs(kelly - 0.2) < 0.001

    def test_full_kelly_negative_edge(self, full_kelly_calc: KellyCalculator) -> None:
        """Test negative Kelly when no edge."""
        # p=0.4, odds=2.0, b=1.0
        # f* = (1.0 * 0.4 - 0.6) / 1.0 = -0.2
        kelly = full_kelly_calc.calculate_full_kelly(0.4, 2.0)
        assert kelly < 0

    def test_fractional_kelly_applied(self, calc: KellyCalculator) -> None:
        """Test fractional Kelly multiplier is applied."""
        result = calc.calculate(10000, 0.6, 2.0)
        # Full Kelly = 0.2, quarter Kelly = 0.05
        assert result.adjusted_kelly == pytest.approx(0.05, abs=0.001)

    def test_max_bet_cap(self) -> None:
        """Test maximum bet percentage cap."""
        calc = KellyCalculator(fraction=1.0, max_bet_pct=0.02, min_edge_pct=0.0)
        result = calc.calculate(10000, 0.7, 2.0)
        # Full Kelly would be 0.4, but capped at 0.02
        assert result.bet_fraction == 0.02
        assert result.bet_amount == 200.0

    def test_min_edge_filter(self, calc: KellyCalculator) -> None:
        """Test minimum edge filter."""
        # Edge less than 2%
        result = calc.calculate(10000, 0.52, 1.91)  # ~1% edge
        assert not result.has_edge
        assert result.bet_amount == 0.0

    def test_positive_edge_bet_placed(self, calc: KellyCalculator) -> None:
        """Test bet is placed with sufficient edge."""
        # 55% prob at 1.91 odds = ~2.6% edge
        result = calc.calculate(10000, 0.55, 1.91)
        assert result.has_edge
        assert result.bet_amount > 0

    def test_zero_return_negative_kelly(self, calc: KellyCalculator) -> None:
        """Test zero bet when Kelly is negative."""
        result = calc.calculate(10000, 0.4, 2.0)
        assert not result.has_edge
        assert result.bet_amount == 0.0

    def test_kelly_result_properties(self, calc: KellyCalculator) -> None:
        """Test KellyResult contains correct values."""
        result = calc.calculate(10000, 0.6, 2.0)
        assert result.full_kelly > 0
        assert result.adjusted_kelly == result.full_kelly * calc.fraction
        assert result.bet_fraction <= calc.max_bet_pct
        assert result.bet_amount == result.bet_fraction * 10000
        assert result.edge > 0
        assert result.has_edge

    def test_calculate_bet_size_shortcut(self, calc: KellyCalculator) -> None:
        """Test calculate_bet_size returns same as calculate().bet_amount."""
        full_result = calc.calculate(10000, 0.6, 2.0)
        shortcut = calc.calculate_bet_size(10000, 0.6, 2.0)
        assert full_result.bet_amount == shortcut

    def test_invalid_probability_raises(self, calc: KellyCalculator) -> None:
        """Test invalid probability raises error."""
        with pytest.raises(InvalidInputError):
            calc.calculate_full_kelly(-0.1, 2.0)
        with pytest.raises(InvalidInputError):
            calc.calculate_full_kelly(1.1, 2.0)

    def test_invalid_odds_raises(self, calc: KellyCalculator) -> None:
        """Test invalid odds raises error."""
        with pytest.raises(InvalidInputError):
            calc.calculate_full_kelly(0.5, 1.0)
        with pytest.raises(InvalidInputError):
            calc.calculate_full_kelly(0.5, 0.5)

    def test_invalid_bankroll_raises(self, calc: KellyCalculator) -> None:
        """Test invalid bankroll raises error."""
        with pytest.raises(InvalidInputError):
            calc.calculate(0, 0.5, 2.0)
        with pytest.raises(InvalidInputError):
            calc.calculate(-1000, 0.5, 2.0)

    def test_invalid_fraction_raises(self) -> None:
        """Test invalid Kelly fraction raises error."""
        with pytest.raises(InvalidInputError):
            KellyCalculator(fraction=0)
        with pytest.raises(InvalidInputError):
            KellyCalculator(fraction=1.5)

    def test_invalid_max_bet_raises(self) -> None:
        """Test invalid max bet raises error."""
        with pytest.raises(InvalidInputError):
            KellyCalculator(max_bet_pct=0)
        with pytest.raises(InvalidInputError):
            KellyCalculator(max_bet_pct=1.5)


class TestKellySimulation:
    """Tests for Kelly bankroll simulation."""

    @pytest.fixture
    def sample_bets(self) -> list[Bet]:
        """Create sample bet history."""
        return [
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
                result="win" if i % 2 == 0 else "loss",
                profit=91.0 if i % 2 == 0 else -100.0,
            )
            for i in range(10)
        ]

    def test_simulate_bankroll(
        self, sample_bets: list[Bet]
    ) -> None:
        """Test bankroll simulation runs without error."""
        calc = KellyCalculator()
        result = calc.simulate_bankroll(sample_bets, initial_bankroll=10000.0)

        assert isinstance(result, SimulationResult)
        assert len(result.bankroll_history) > 0
        assert result.final_bankroll > 0

    def test_simulate_tracks_max_min(
        self, sample_bets: list[Bet]
    ) -> None:
        """Test simulation tracks max/min bankroll."""
        calc = KellyCalculator()
        result = calc.simulate_bankroll(sample_bets, initial_bankroll=10000.0)

        assert result.max_bankroll >= result.final_bankroll
        assert result.min_bankroll <= result.final_bankroll
        assert result.max_bankroll >= 10000.0 or result.min_bankroll <= 10000.0

    def test_simulate_calculates_metrics(
        self, sample_bets: list[Bet]
    ) -> None:
        """Test simulation calculates metrics."""
        calc = KellyCalculator()
        result = calc.simulate_bankroll(sample_bets, initial_bankroll=10000.0)

        assert isinstance(result.total_return, float)
        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.max_drawdown, float)
        assert 0 <= result.max_drawdown <= 1.0

    def test_optimize_fraction(self, sample_bets: list[Bet]) -> None:
        """Test Kelly fraction optimization."""
        calc = KellyCalculator()
        result = calc.optimize_fraction(
            sample_bets,
            fractions=[0.1, 0.2, 0.3],
            metric="sharpe",
        )

        assert isinstance(result, KellyOptimizationResult)
        assert result.best_fraction in [0.1, 0.2, 0.3]
        assert len(result.results_by_fraction) == 3
        assert result.metric_name == "sharpe"

    def test_optimize_fraction_growth_metric(self, sample_bets: list[Bet]) -> None:
        """Test optimization with growth metric."""
        calc = KellyCalculator()
        result = calc.optimize_fraction(
            sample_bets,
            fractions=[0.1, 0.25, 0.5],
            metric="growth",
        )

        assert result.metric_name == "growth"

    def test_optimize_invalid_metric_raises(self, sample_bets: list[Bet]) -> None:
        """Test invalid metric raises error."""
        calc = KellyCalculator()
        with pytest.raises(ValueError):
            calc.optimize_fraction(sample_bets, metric="invalid")


class TestMaxDrawdown:
    """Tests for max drawdown calculation."""

    def test_no_drawdown(self) -> None:
        """Test no drawdown when always increasing."""
        calc = KellyCalculator()
        history = [100, 110, 120, 130]
        drawdown = calc._calculate_max_drawdown(history)
        assert drawdown == 0.0

    def test_full_drawdown(self) -> None:
        """Test drawdown calculation."""
        calc = KellyCalculator()
        history = [100, 150, 120, 130]  # Peak 150, trough 120 = 20% DD
        drawdown = calc._calculate_max_drawdown(history)
        assert abs(drawdown - 0.2) < 0.001

    def test_empty_history(self) -> None:
        """Test empty history returns zero."""
        calc = KellyCalculator()
        assert calc._calculate_max_drawdown([]) == 0.0
        assert calc._calculate_max_drawdown([100]) == 0.0


class TestEdgeCalculations:
    """Tests for edge-related calculations."""

    def test_edge_required(self) -> None:
        """Test minimum edge required calculation."""
        calc = KellyCalculator(min_edge_pct=0.02)
        # At 1.91 odds (52.36% implied), need 54.36% to have 2% edge
        min_prob = calc.calculate_edge_required(1.91)
        assert abs(min_prob - 0.5436) < 0.01

    def test_odds_for_edge(self) -> None:
        """Test minimum odds for edge calculation."""
        calc = KellyCalculator(min_edge_pct=0.02)
        # With 55% prob and 2% target edge, need odds implying 53%
        min_odds = calc.calculate_odds_for_edge(0.55)
        # 1/0.53 = 1.887
        assert abs(min_odds - 1.887) < 0.01
