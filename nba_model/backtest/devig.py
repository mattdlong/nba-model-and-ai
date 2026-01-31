"""Vig removal methods for extracting true implied probabilities.

This module provides methods to remove the bookmaker's margin (vig) from
betting odds to recover fair probabilities. Three methods are implemented:

1. Multiplicative: Simple but inferior accuracy
2. Power Method: Better handling of longshot bias
3. Shin's Method: Gold standard for liquid markets

Example:
    >>> calc = DevigCalculator()
    >>> home_prob, away_prob = calc.power_method_devig(1.91, 1.91)
    >>> print(f"Fair probabilities: {home_prob:.3f}, {away_prob:.3f}")
    Fair probabilities: 0.500, 0.500
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import brentq

if TYPE_CHECKING:
    pass

# =============================================================================
# Constants
# =============================================================================

DEFAULT_TOLERANCE: float = 1e-6
MAX_ITERATIONS: int = 100


# =============================================================================
# Exceptions
# =============================================================================


class DevigError(Exception):
    """Base exception for devigging errors."""


class InvalidOddsError(DevigError):
    """Raised when odds are invalid (<=1 or non-positive)."""


class ConvergenceError(DevigError):
    """Raised when numerical method fails to converge."""


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class FairProbabilities:
    """Container for fair probabilities after vig removal.

    Attributes:
        home: Fair probability for home outcome.
        away: Fair probability for away outcome.
        vig: Original vig (overround) in the market.
        method: Devigging method used.
    """

    home: float
    away: float
    vig: float
    method: str

    def __post_init__(self) -> None:
        """Validate probabilities sum to approximately 1."""
        total = self.home + self.away
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Probabilities must sum to ~1, got {total:.4f}")


# =============================================================================
# Helper Functions
# =============================================================================


def implied_probability(decimal_odds: float) -> float:
    """Convert decimal odds to implied probability.

    Args:
        decimal_odds: Decimal odds (e.g., 1.91 for -110).

    Returns:
        Implied probability (includes vig).

    Raises:
        InvalidOddsError: If odds are <= 1.
    """
    if decimal_odds <= 1:
        raise InvalidOddsError(f"Decimal odds must be > 1, got {decimal_odds}")
    return 1.0 / decimal_odds


def calculate_overround(odds_home: float, odds_away: float) -> float:
    """Calculate the overround (vig) in a two-way market.

    Args:
        odds_home: Decimal odds for home.
        odds_away: Decimal odds for away.

    Returns:
        Overround as a decimal (e.g., 0.048 for 4.8% vig).
    """
    return implied_probability(odds_home) + implied_probability(odds_away) - 1.0


def solve_power_k(
    odds: list[float],
    tol: float = DEFAULT_TOLERANCE,
) -> float:
    """Find exponent k for power method devigging using Brent's method.

    Solves for k such that: sum((1/o)^k for o in odds) = 1

    Args:
        odds: List of decimal odds.
        tol: Convergence tolerance.

    Returns:
        The exponent k.

    Raises:
        ConvergenceError: If method fails to converge.
    """
    # Validate odds
    for odd in odds:
        if odd <= 1:
            raise InvalidOddsError(f"All odds must be > 1, got {odd}")

    def objective(k: float) -> float:
        return sum((1.0 / o) ** k for o in odds) - 1.0

    # k > 1 reduces implied probs, k < 1 increases them
    # For typical vig, k is slightly > 1
    try:
        # Search in a reasonable range
        k = brentq(objective, 0.5, 2.0, xtol=tol)
        return float(k)
    except ValueError as e:
        raise ConvergenceError(f"Power method failed to converge: {e}") from e


def _shin_probabilities(z: float, odds: list[float]) -> list[float]:
    """Calculate true probabilities using Shin's formula.

    Uses the standard Shin (1991, 1992) formulation for betting markets.
    The formula for fair probability is:
        p_i = (sqrt(z² + 4(1-z)q_i) - z) / (2(1-z))
    where q_i is the implied probability from odds.

    Args:
        z: Proportion of informed bettors (0 < z < 1).
        odds: List of decimal odds.

    Returns:
        List of fair probabilities (should sum to 1 when z is solved correctly).
    """
    # Calculate implied probabilities
    implied = [1.0 / o for o in odds]

    # Handle edge case of z very close to 0 or 1
    if z <= 1e-10:
        # z = 0 means no informed bettors, use implied probs normalized
        total_impl = sum(implied)
        return [p / total_impl for p in implied]

    if z >= 1 - 1e-10:
        # z = 1 is degenerate case, fallback to multiplicative
        total_impl = sum(implied)
        return [p / total_impl for p in implied]

    # Use the exact Shin formula:
    # p_i = (sqrt(z² + 4(1-z)q_i) - z) / (2(1-z))
    probs = []
    for q in implied:
        inner = z**2 + 4 * (1 - z) * q
        p = (np.sqrt(inner) - z) / (2 * (1 - z))
        probs.append(p)

    # Normalize to ensure sum = 1 (handles small numerical errors)
    total = sum(probs)
    if total > 0:
        probs = [p / total for p in probs]

    return probs


def solve_shin_z(
    odds: list[float],
    tol: float = DEFAULT_TOLERANCE,
) -> float:
    """Find Shin's z parameter (proportion of informed bettors) iteratively.

    In Shin's model (1991, 1992), z represents the proportion of bettors
    who are informed (have private information). This function iteratively
    solves for z such that the derived fair probabilities sum to 1.

    The Shin formula for fair probability is:
        p_i = (sqrt(z² + 4(1-z)q_i) - z) / (2(1-z))
    where q_i is the implied probability from odds.

    We find z by solving: sum(p_i) = 1

    Args:
        odds: List of decimal odds.
        tol: Convergence tolerance for Brent's method.

    Returns:
        The z parameter (proportion of informed bettors).

    Raises:
        InvalidOddsError: If odds are invalid.
        ConvergenceError: If numerical method fails to converge.
    """
    # Validate odds
    for odd in odds:
        if odd <= 1:
            raise InvalidOddsError(f"All odds must be > 1, got {odd}")

    # Calculate implied probabilities
    implied = [1.0 / o for o in odds]
    overround = sum(implied) - 1.0

    # If no overround (fair odds), z = 0
    if overround <= 0:
        return 0.0

    def shin_prob_sum(z: float) -> float:
        """Calculate sum of Shin probabilities minus 1.

        Returns 0 when z gives probabilities that sum to 1.
        """
        if z <= 0 or z >= 1:
            return float("inf")

        total = 0.0
        for q in implied:
            inner = z**2 + 4 * (1 - z) * q
            if inner < 0:
                return float("inf")
            p = (np.sqrt(inner) - z) / (2 * (1 - z))
            total += p
        return total - 1.0

    # Search for z in (0, 1) where probabilities sum to 1
    # z typically lies between 0 and the overround
    try:
        # Use a wider search range to ensure convergence
        # z should be positive and less than 1
        z_min = 1e-6
        z_max = min(0.5, overround + 0.1)  # Reasonable upper bound

        # Check if solution exists in range
        f_min = shin_prob_sum(z_min)
        f_max = shin_prob_sum(z_max)

        # If signs are the same, expand search range
        if f_min * f_max > 0:
            z_max = 0.99  # Try wider range
            f_max = shin_prob_sum(z_max)

        if f_min * f_max > 0:
            # Fallback: use overround approximation when iterative fails
            return min(max(overround, 0.001), 0.5)

        z = brentq(shin_prob_sum, z_min, z_max, xtol=tol)
        return float(z)
    except (ValueError, RuntimeError) as e:
        raise ConvergenceError(f"Shin method failed to converge: {e}") from e


# =============================================================================
# Main Calculator Class
# =============================================================================


class DevigCalculator:
    """Calculator for removing bookmaker vig from betting odds.

    Provides three devigging methods:
    1. Multiplicative: Simple normalization (p_fair = p_implied / sum)
    2. Power Method: Solves for k where sum((1/odds)^k) = 1
    3. Shin's Method: Models informed bettor proportion

    Attributes:
        tolerance: Numerical convergence tolerance.

    Example:
        >>> calc = DevigCalculator()
        >>> result = calc.power_method_devig(1.91, 1.91)
        >>> print(f"Home: {result.home:.3f}, Away: {result.away:.3f}")
        Home: 0.500, Away: 0.500
    """

    def __init__(self, tolerance: float = DEFAULT_TOLERANCE) -> None:
        """Initialize DevigCalculator.

        Args:
            tolerance: Numerical convergence tolerance for iterative methods.
        """
        self.tolerance = tolerance

    def multiplicative_devig(
        self,
        odds_home: float,
        odds_away: float,
    ) -> FairProbabilities:
        """Simple multiplicative devigging.

        Fair probability = implied probability / sum of implied probabilities.

        This method is fast but inferior for handling longshot bias.

        Args:
            odds_home: Decimal odds for home outcome.
            odds_away: Decimal odds for away outcome.

        Returns:
            FairProbabilities with fair home and away probabilities.

        Raises:
            InvalidOddsError: If any odds are <= 1.
        """
        impl_home = implied_probability(odds_home)
        impl_away = implied_probability(odds_away)
        total = impl_home + impl_away

        vig = total - 1.0

        return FairProbabilities(
            home=impl_home / total,
            away=impl_away / total,
            vig=vig,
            method="multiplicative",
        )

    def power_method_devig(
        self,
        odds_home: float,
        odds_away: float,
    ) -> FairProbabilities:
        """Power method devigging for better longshot bias handling.

        Solves for k such that: (1/home_odds)^k + (1/away_odds)^k = 1
        Fair probability = (1/odds)^k

        Args:
            odds_home: Decimal odds for home outcome.
            odds_away: Decimal odds for away outcome.

        Returns:
            FairProbabilities with fair home and away probabilities.

        Raises:
            InvalidOddsError: If any odds are <= 1.
            ConvergenceError: If numerical method fails to converge.
        """
        vig = calculate_overround(odds_home, odds_away)

        # Solve for k
        k = solve_power_k([odds_home, odds_away], self.tolerance)

        # Calculate fair probabilities
        home_prob = (1.0 / odds_home) ** k
        away_prob = (1.0 / odds_away) ** k

        return FairProbabilities(
            home=home_prob,
            away=away_prob,
            vig=vig,
            method="power",
        )

    def shin_method_devig(
        self,
        odds_home: float,
        odds_away: float,
    ) -> FairProbabilities:
        """Shin's method devigging - gold standard for liquid markets.

        Models the market as containing informed bettors. Derives true
        probabilities by solving for the proportion of informed bettors (z).

        Args:
            odds_home: Decimal odds for home outcome.
            odds_away: Decimal odds for away outcome.

        Returns:
            FairProbabilities with fair home and away probabilities.

        Raises:
            InvalidOddsError: If any odds are <= 1.
            ConvergenceError: If numerical method fails to converge.
        """
        vig = calculate_overround(odds_home, odds_away)
        odds = [odds_home, odds_away]

        # Solve for z (informed bettor proportion)
        z = solve_shin_z(odds, self.tolerance)

        # Calculate fair probabilities using Shin formula
        probs = _shin_probabilities(z, odds)

        return FairProbabilities(
            home=probs[0],
            away=probs[1],
            vig=vig,
            method="shin",
        )

    def devig(
        self,
        odds_home: float,
        odds_away: float,
        method: str = "power",
    ) -> FairProbabilities:
        """Devig odds using specified method.

        Args:
            odds_home: Decimal odds for home outcome.
            odds_away: Decimal odds for away outcome.
            method: Devigging method ('multiplicative', 'power', 'shin').

        Returns:
            FairProbabilities with fair home and away probabilities.

        Raises:
            ValueError: If method is unknown.
        """
        methods = {
            "multiplicative": self.multiplicative_devig,
            "power": self.power_method_devig,
            "shin": self.shin_method_devig,
        }

        if method not in methods:
            valid = ", ".join(methods.keys())
            raise ValueError(f"Unknown method '{method}'. Valid: {valid}")

        return methods[method](odds_home, odds_away)

    @staticmethod
    def calculate_edge(model_prob: float, market_prob: float) -> float:
        """Calculate betting edge.

        Edge = model_prob - market_prob

        Positive edge indicates value bet opportunity.

        Args:
            model_prob: Model's estimated probability.
            market_prob: Fair market probability after devigging.

        Returns:
            Edge as a decimal (e.g., 0.05 for 5% edge).
        """
        return model_prob - market_prob

    @staticmethod
    def decimal_to_american(decimal_odds: float) -> int:
        """Convert decimal odds to American format.

        Args:
            decimal_odds: Decimal odds (e.g., 1.91).

        Returns:
            American odds (e.g., -110).
        """
        if decimal_odds >= 2.0:
            return round((decimal_odds - 1) * 100)
        return round(-100 / (decimal_odds - 1))

    @staticmethod
    def american_to_decimal(american_odds: int) -> float:
        """Convert American odds to decimal format.

        Args:
            american_odds: American odds (e.g., -110).

        Returns:
            Decimal odds (e.g., 1.909).
        """
        if american_odds > 0:
            return 1 + american_odds / 100
        return 1 - 100 / american_odds
