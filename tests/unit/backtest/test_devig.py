"""Tests for devigging methods."""

from __future__ import annotations

import pytest

from nba_model.backtest.devig import (
    ConvergenceError,
    DevigCalculator,
    FairProbabilities,
    InvalidOddsError,
    calculate_overround,
    implied_probability,
    solve_power_k,
    solve_shin_z,
)


class TestImpliedProbability:
    """Tests for implied_probability function."""

    def test_standard_odds(self) -> None:
        """Test conversion of standard odds."""
        # -110 line = 1.91 decimal
        assert abs(implied_probability(1.91) - 0.5236) < 0.001

    def test_even_money(self) -> None:
        """Test even money odds."""
        assert implied_probability(2.0) == 0.5

    def test_heavy_favorite(self) -> None:
        """Test heavy favorite odds."""
        # -500 = 1.2 decimal
        assert abs(implied_probability(1.2) - 0.8333) < 0.001

    def test_underdog(self) -> None:
        """Test underdog odds."""
        # +200 = 3.0 decimal
        assert abs(implied_probability(3.0) - 0.3333) < 0.001

    def test_invalid_odds_raises(self) -> None:
        """Test that odds <= 1 raise error."""
        with pytest.raises(InvalidOddsError):
            implied_probability(1.0)

        with pytest.raises(InvalidOddsError):
            implied_probability(0.5)


class TestCalculateOverround:
    """Tests for calculate_overround function."""

    def test_standard_vig(self) -> None:
        """Test typical -110/-110 vig calculation."""
        vig = calculate_overround(1.91, 1.91)
        # 1/1.91 + 1/1.91 - 1 = 0.0471
        assert abs(vig - 0.0471) < 0.001

    def test_no_vig(self) -> None:
        """Test no-vig line."""
        vig = calculate_overround(2.0, 2.0)
        assert abs(vig) < 0.001

    def test_unbalanced_line(self) -> None:
        """Test unbalanced favorite/underdog."""
        # -200/+180 = 1.5/2.8
        vig = calculate_overround(1.5, 2.8)
        # 1/1.5 + 1/2.8 - 1 = 0.024
        assert abs(vig - 0.024) < 0.01


class TestSolvePowerK:
    """Tests for power method k solver."""

    def test_balanced_odds(self) -> None:
        """Test k solver with balanced odds."""
        k = solve_power_k([1.91, 1.91])
        # For balanced odds with vig, k > 1
        assert k > 1.0
        assert k < 1.1

    def test_convergence(self) -> None:
        """Test that solution converges properly."""
        k = solve_power_k([1.5, 2.8])
        # Verify solution: (1/1.5)^k + (1/2.8)^k should ≈ 1
        total = (1 / 1.5) ** k + (1 / 2.8) ** k
        assert abs(total - 1.0) < 1e-5

    def test_invalid_odds_raises(self) -> None:
        """Test that invalid odds raise error."""
        with pytest.raises(InvalidOddsError):
            solve_power_k([1.0, 2.0])


class TestSolveShinZ:
    """Tests for Shin's method z solver."""

    def test_balanced_odds(self) -> None:
        """Test z solver with balanced odds."""
        z = solve_shin_z([1.91, 1.91])
        # z should be small positive value
        assert 0 < z < 0.1

    def test_convergence(self) -> None:
        """Test that solution produces valid probabilities."""
        z = solve_shin_z([1.5, 2.8])
        # z should be in reasonable range
        assert 0 < z < 0.5


class TestFairProbabilities:
    """Tests for FairProbabilities dataclass."""

    def test_valid_probabilities(self) -> None:
        """Test that valid probabilities are accepted."""
        fp = FairProbabilities(home=0.5, away=0.5, vig=0.0, method="test")
        assert fp.home == 0.5
        assert fp.away == 0.5

    def test_nearly_sum_to_one(self) -> None:
        """Test that probabilities can sum to nearly 1."""
        # Should accept 0.99 to 1.01 range
        fp = FairProbabilities(home=0.505, away=0.495, vig=0.0, method="test")
        assert fp.home + fp.away == 1.0

    def test_invalid_sum_raises(self) -> None:
        """Test that invalid probability sum raises error."""
        with pytest.raises(ValueError):
            FairProbabilities(home=0.6, away=0.6, vig=0.0, method="test")


class TestDevigCalculator:
    """Tests for DevigCalculator class."""

    @pytest.fixture
    def calc(self) -> DevigCalculator:
        """Create calculator fixture."""
        return DevigCalculator()

    def test_multiplicative_balanced(self, calc: DevigCalculator) -> None:
        """Test multiplicative devig with balanced odds."""
        result = calc.multiplicative_devig(1.91, 1.91)
        assert abs(result.home - 0.5) < 0.001
        assert abs(result.away - 0.5) < 0.001
        assert result.method == "multiplicative"

    def test_multiplicative_unbalanced(self, calc: DevigCalculator) -> None:
        """Test multiplicative devig with unbalanced odds."""
        result = calc.multiplicative_devig(1.5, 2.8)
        # Home implied: 0.667, Away implied: 0.357, sum: 1.024
        # Fair: 0.667/1.024 = 0.651, 0.357/1.024 = 0.349
        assert abs(result.home - 0.651) < 0.01
        assert abs(result.away - 0.349) < 0.01
        assert abs(result.home + result.away - 1.0) < 0.001

    def test_power_method_balanced(self, calc: DevigCalculator) -> None:
        """Test power method with balanced odds."""
        result = calc.power_method_devig(1.91, 1.91)
        assert abs(result.home - 0.5) < 0.001
        assert abs(result.away - 0.5) < 0.001
        assert result.method == "power"

    def test_power_method_sums_to_one(self, calc: DevigCalculator) -> None:
        """Test power method probabilities sum to 1."""
        result = calc.power_method_devig(1.5, 2.8)
        assert abs(result.home + result.away - 1.0) < 0.001

    def test_shin_method_balanced(self, calc: DevigCalculator) -> None:
        """Test Shin's method with balanced odds."""
        result = calc.shin_method_devig(1.91, 1.91)
        assert abs(result.home - 0.5) < 0.001
        assert abs(result.away - 0.5) < 0.001
        assert result.method == "shin"

    def test_shin_method_sums_to_one(self, calc: DevigCalculator) -> None:
        """Test Shin's method probabilities sum to 1."""
        result = calc.shin_method_devig(1.5, 2.8)
        assert abs(result.home + result.away - 1.0) < 0.001

    def test_devig_method_selection(self, calc: DevigCalculator) -> None:
        """Test devig method selection."""
        mult = calc.devig(1.91, 1.91, method="multiplicative")
        assert mult.method == "multiplicative"

        power = calc.devig(1.91, 1.91, method="power")
        assert power.method == "power"

        shin = calc.devig(1.91, 1.91, method="shin")
        assert shin.method == "shin"

    def test_invalid_method_raises(self, calc: DevigCalculator) -> None:
        """Test that invalid method raises error."""
        with pytest.raises(ValueError):
            calc.devig(1.91, 1.91, method="invalid")

    def test_calculate_edge(self) -> None:
        """Test edge calculation."""
        edge = DevigCalculator.calculate_edge(0.55, 0.50)
        assert edge == pytest.approx(0.05)

        edge = DevigCalculator.calculate_edge(0.45, 0.50)
        assert edge == pytest.approx(-0.05)

    def test_decimal_to_american(self) -> None:
        """Test decimal to American conversion."""
        # -110 = 1.909...
        assert DevigCalculator.decimal_to_american(1.909) == -110
        # +150 = 2.5
        assert DevigCalculator.decimal_to_american(2.5) == 150
        # +100 = 2.0
        assert DevigCalculator.decimal_to_american(2.0) == 100

    def test_american_to_decimal(self) -> None:
        """Test American to decimal conversion."""
        assert abs(DevigCalculator.american_to_decimal(-110) - 1.909) < 0.01
        assert DevigCalculator.american_to_decimal(150) == 2.5
        assert DevigCalculator.american_to_decimal(100) == 2.0

    def test_power_and_shin_similar_for_balanced(self, calc: DevigCalculator) -> None:
        """Test that power and Shin produce similar results for balanced odds."""
        power = calc.power_method_devig(1.91, 1.91)
        shin = calc.shin_method_devig(1.91, 1.91)

        # Should be very close for balanced odds
        assert abs(power.home - shin.home) < 0.01
        assert abs(power.away - shin.away) < 0.01
