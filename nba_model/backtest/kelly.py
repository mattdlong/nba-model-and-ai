"""Kelly Criterion bet sizing for optimal bankroll growth.

This module implements the Kelly Criterion for calculating optimal bet sizes,
with support for fractional Kelly to reduce variance and configurable caps.

Full Kelly Formula:
    f* = (bp - q) / b
    where:
        b = decimal_odds - 1 (net odds)
        p = probability of winning
        q = 1 - p (probability of losing)

Example:
    >>> calc = KellyCalculator(fraction=0.25, max_bet_pct=0.02)
    >>> bet_size = calc.calculate_bet_size(10000, 0.55, 1.91)
    >>> print(f"Recommended bet: ${bet_size:.2f}")
    Recommended bet: $125.00
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from nba_model.types import Bet

# =============================================================================
# Constants
# =============================================================================

DEFAULT_KELLY_FRACTION: float = 0.25  # Quarter Kelly
DEFAULT_MAX_BET_PCT: float = 0.02  # 2% max bet
DEFAULT_MIN_EDGE_PCT: float = 0.02  # 2% min edge to bet


# =============================================================================
# Exceptions
# =============================================================================


class KellyError(Exception):
    """Base exception for Kelly calculation errors."""


class InvalidInputError(KellyError):
    """Raised when input parameters are invalid."""


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class KellyResult:
    """Container for Kelly calculation results.

    Attributes:
        full_kelly: Full Kelly fraction before adjustments.
        adjusted_kelly: Kelly fraction after applying fractional multiplier.
        bet_fraction: Final bet fraction after caps.
        bet_amount: Actual bet amount in currency units.
        edge: Model edge over market.
        has_edge: Whether a positive edge exists.
    """

    full_kelly: float
    adjusted_kelly: float
    bet_fraction: float
    bet_amount: float
    edge: float
    has_edge: bool


@dataclass
class KellyOptimizationResult:
    """Results from Kelly fraction optimization.

    Attributes:
        best_fraction: Optimal Kelly fraction.
        best_metric: Value of optimization metric at best fraction.
        results_by_fraction: Dict mapping fraction to metric value.
        metric_name: Name of the optimized metric.
    """

    best_fraction: float
    best_metric: float
    results_by_fraction: dict[float, float]
    metric_name: str


@dataclass
class SimulationResult:
    """Results from bankroll simulation.

    Attributes:
        final_bankroll: Ending bankroll value.
        max_bankroll: Peak bankroll during simulation.
        min_bankroll: Minimum bankroll during simulation.
        total_return: Total return percentage.
        sharpe_ratio: Risk-adjusted return.
        max_drawdown: Maximum peak-to-trough decline.
        bankroll_history: List of bankroll values over time.
    """

    final_bankroll: float
    max_bankroll: float
    min_bankroll: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    bankroll_history: list[float] = field(default_factory=list)


# =============================================================================
# Main Calculator Class
# =============================================================================


class KellyCalculator:
    """Kelly Criterion bet sizing calculator.

    Calculates optimal bet sizes using Kelly Criterion with fractional
    adjustment and configurable caps for variance reduction.

    Attributes:
        fraction: Kelly fraction multiplier (e.g., 0.25 for quarter Kelly).
        max_bet_pct: Maximum bet as fraction of bankroll.
        min_edge_pct: Minimum edge required to place a bet.

    Example:
        >>> calc = KellyCalculator(fraction=0.25, max_bet_pct=0.02, min_edge_pct=0.02)
        >>> result = calc.calculate(10000, 0.55, 1.91)
        >>> print(f"Bet: ${result.bet_amount:.2f} ({result.bet_fraction:.2%})")
    """

    def __init__(
        self,
        fraction: float = DEFAULT_KELLY_FRACTION,
        max_bet_pct: float = DEFAULT_MAX_BET_PCT,
        min_edge_pct: float = DEFAULT_MIN_EDGE_PCT,
    ) -> None:
        """Initialize KellyCalculator.

        Args:
            fraction: Kelly fraction multiplier (default 0.25 = quarter Kelly).
            max_bet_pct: Maximum bet as percentage of bankroll (default 2%).
            min_edge_pct: Minimum edge to place bet (default 2%).

        Raises:
            InvalidInputError: If parameters are out of valid ranges.
        """
        if not 0 < fraction <= 1:
            raise InvalidInputError(f"Kelly fraction must be in (0, 1], got {fraction}")
        if not 0 < max_bet_pct <= 1:
            raise InvalidInputError(
                f"Max bet percentage must be in (0, 1], got {max_bet_pct}"
            )
        if min_edge_pct < 0:
            raise InvalidInputError(
                f"Min edge percentage must be >= 0, got {min_edge_pct}"
            )

        self.fraction = fraction
        self.max_bet_pct = max_bet_pct
        self.min_edge_pct = min_edge_pct

    def calculate_full_kelly(
        self,
        model_prob: float,
        decimal_odds: float,
    ) -> float:
        """Calculate full Kelly fraction.

        Full Kelly: f* = (bp - q) / b
        where b = decimal_odds - 1, p = model_prob, q = 1 - p

        Args:
            model_prob: Model's probability estimate (0 to 1).
            decimal_odds: Decimal odds (e.g., 1.91 for -110).

        Returns:
            Full Kelly fraction (can be negative if no edge).

        Raises:
            InvalidInputError: If inputs are out of valid ranges.
        """
        if not 0 <= model_prob <= 1:
            raise InvalidInputError(
                f"Model probability must be in [0, 1], got {model_prob}"
            )
        if decimal_odds <= 1:
            raise InvalidInputError(f"Decimal odds must be > 1, got {decimal_odds}")

        b = decimal_odds - 1  # Net odds
        p = model_prob
        q = 1 - p

        return (b * p - q) / b

    def calculate_bet_size(
        self,
        bankroll: float,
        model_prob: float,
        decimal_odds: float,
    ) -> float:
        """Calculate recommended bet size in currency units.

        This is the main method for determining how much to bet.

        Args:
            bankroll: Current bankroll in currency units.
            model_prob: Model's probability estimate (0 to 1).
            decimal_odds: Decimal odds (e.g., 1.91 for -110).

        Returns:
            Bet amount in currency units (0 if no edge or negative Kelly).

        Raises:
            InvalidInputError: If inputs are out of valid ranges.
        """
        result = self.calculate(bankroll, model_prob, decimal_odds)
        return result.bet_amount

    def calculate(
        self,
        bankroll: float,
        model_prob: float,
        decimal_odds: float,
    ) -> KellyResult:
        """Calculate Kelly bet with full details.

        Args:
            bankroll: Current bankroll in currency units.
            model_prob: Model's probability estimate (0 to 1).
            decimal_odds: Decimal odds (e.g., 1.91 for -110).

        Returns:
            KellyResult with full calculation details.

        Raises:
            InvalidInputError: If inputs are out of valid ranges.
        """
        if bankroll <= 0:
            raise InvalidInputError(f"Bankroll must be > 0, got {bankroll}")

        # Calculate edge
        implied_prob = 1.0 / decimal_odds
        edge = model_prob - implied_prob

        # Calculate full Kelly
        full_kelly = self.calculate_full_kelly(model_prob, decimal_odds)

        # Check if we have sufficient edge
        has_edge = edge >= self.min_edge_pct and full_kelly > 0

        if not has_edge:
            return KellyResult(
                full_kelly=full_kelly,
                adjusted_kelly=0.0,
                bet_fraction=0.0,
                bet_amount=0.0,
                edge=edge,
                has_edge=False,
            )

        # Apply fractional Kelly
        adjusted_kelly = full_kelly * self.fraction

        # Apply cap
        bet_fraction = min(adjusted_kelly, self.max_bet_pct)

        # Calculate bet amount
        bet_amount = bankroll * bet_fraction

        return KellyResult(
            full_kelly=full_kelly,
            adjusted_kelly=adjusted_kelly,
            bet_fraction=bet_fraction,
            bet_amount=bet_amount,
            edge=edge,
            has_edge=True,
        )

    def simulate_bankroll(
        self,
        bets: list[Bet],
        initial_bankroll: float = 10000.0,
        kelly_fraction: float | None = None,
    ) -> SimulationResult:
        """Simulate bankroll evolution over historical bets.

        Args:
            bets: List of Bet objects with model_prob, market_odds, result.
            initial_bankroll: Starting bankroll.
            kelly_fraction: Kelly fraction to use (default: self.fraction).

        Returns:
            SimulationResult with bankroll history and metrics.
        """
        fraction = kelly_fraction if kelly_fraction is not None else self.fraction

        # Create a temporary calculator with the specified fraction
        temp_calc = KellyCalculator(
            fraction=fraction,
            max_bet_pct=self.max_bet_pct,
            min_edge_pct=self.min_edge_pct,
        )

        bankroll = initial_bankroll
        bankroll_history = [bankroll]
        max_bankroll = bankroll
        min_bankroll = bankroll

        returns = []

        for bet in bets:
            # Calculate bet size
            result = temp_calc.calculate(bankroll, bet.model_prob, bet.market_odds)

            if result.has_edge and result.bet_amount > 0:
                # Simulate outcome
                if bet.result == "win":
                    profit = result.bet_amount * (bet.market_odds - 1)
                elif bet.result == "loss":
                    profit = -result.bet_amount
                else:  # push
                    profit = 0.0

                bankroll += profit
                returns.append(
                    profit / bankroll_history[-1] if bankroll_history[-1] > 0 else 0
                )

            bankroll_history.append(bankroll)
            max_bankroll = max(max_bankroll, bankroll)
            min_bankroll = min(min_bankroll, bankroll)

        # Calculate metrics
        total_return = (bankroll - initial_bankroll) / initial_bankroll

        # Sharpe ratio (annualized, assuming ~1000 bets/year for daily betting)
        if returns:
            returns_array = np.array(returns)
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array)
            sharpe_ratio = (
                mean_return / std_return * np.sqrt(len(returns))
                if std_return > 0
                else 0.0
            )
        else:
            sharpe_ratio = 0.0

        # Max drawdown
        max_drawdown = self._calculate_max_drawdown(bankroll_history)

        return SimulationResult(
            final_bankroll=bankroll,
            max_bankroll=max_bankroll,
            min_bankroll=min_bankroll,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            bankroll_history=bankroll_history,
        )

    def optimize_fraction(
        self,
        historical_bets: list[Bet],
        fractions: list[float] | None = None,
        initial_bankroll: float = 10000.0,
        metric: str = "sharpe",
    ) -> KellyOptimizationResult:
        """Find optimal Kelly fraction via historical simulation.

        Tests different Kelly fractions and finds the one that maximizes
        the chosen metric (Sharpe ratio or geometric growth rate).

        Args:
            historical_bets: List of historical Bet objects.
            fractions: Kelly fractions to test (default: [0.1, 0.15, ..., 0.5]).
            initial_bankroll: Starting bankroll for simulation.
            metric: Optimization metric ('sharpe' or 'growth').

        Returns:
            KellyOptimizationResult with optimal fraction and comparison data.

        Raises:
            ValueError: If metric is unknown.
        """
        if fractions is None:
            fractions = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

        valid_metrics = {"sharpe", "growth"}
        if metric not in valid_metrics:
            raise ValueError(f"Unknown metric '{metric}'. Valid: {valid_metrics}")

        results_by_fraction: dict[float, float] = {}

        for frac in fractions:
            sim_result = self.simulate_bankroll(
                historical_bets,
                initial_bankroll,
                kelly_fraction=frac,
            )

            if metric == "sharpe":
                results_by_fraction[frac] = sim_result.sharpe_ratio
            else:  # growth
                # Geometric growth rate
                n_bets = len(historical_bets)
                if n_bets > 0 and sim_result.final_bankroll > 0:
                    growth_rate = (sim_result.final_bankroll / initial_bankroll) ** (
                        1 / n_bets
                    ) - 1
                else:
                    growth_rate = 0.0
                results_by_fraction[frac] = growth_rate

        # Find best fraction
        best_fraction = max(results_by_fraction, key=lambda f: results_by_fraction[f])
        best_metric = results_by_fraction[best_fraction]

        return KellyOptimizationResult(
            best_fraction=best_fraction,
            best_metric=best_metric,
            results_by_fraction=results_by_fraction,
            metric_name=metric,
        )

    @staticmethod
    def _calculate_max_drawdown(bankroll_history: list[float]) -> float:
        """Calculate maximum drawdown from bankroll history.

        Args:
            bankroll_history: List of bankroll values.

        Returns:
            Maximum drawdown as a positive decimal (e.g., 0.15 for 15%).
        """
        if not bankroll_history or len(bankroll_history) < 2:
            return 0.0

        peak = bankroll_history[0]
        max_drawdown = 0.0

        for value in bankroll_history[1:]:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak > 0 else 0.0
            max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown

    def calculate_edge_required(self, decimal_odds: float) -> float:
        """Calculate minimum edge required to meet min_edge_pct threshold.

        Args:
            decimal_odds: Decimal odds.

        Returns:
            Minimum model probability to have sufficient edge.
        """
        implied_prob = 1.0 / decimal_odds
        return implied_prob + self.min_edge_pct

    def calculate_odds_for_edge(
        self,
        model_prob: float,
        target_edge: float | None = None,
    ) -> float:
        """Calculate minimum decimal odds needed to have target edge.

        Args:
            model_prob: Model's probability estimate.
            target_edge: Target edge (default: self.min_edge_pct).

        Returns:
            Minimum decimal odds required.
        """
        edge = target_edge if target_edge is not None else self.min_edge_pct
        required_implied = model_prob - edge
        if required_implied <= 0:
            return float("inf")
        return 1.0 / required_implied
