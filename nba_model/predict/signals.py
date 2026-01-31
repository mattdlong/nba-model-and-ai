"""Betting signal generation from NBA game predictions.

This module transforms game predictions into actionable betting signals by
comparing model probabilities against market odds, applying devigging, and
calculating Kelly criterion sizing.

Signal Generation Flow:
    1. Obtain market odds for moneyline, spread, and total
    2. Apply devigging to extract fair market probabilities
    3. Compare model probability against fair market probability
    4. Calculate edge = model_prob - market_prob
    5. Filter to signals where edge > min_edge threshold
    6. Compute Kelly fraction for position sizing
    7. Assign confidence level based on model confidence and edge

Example:
    >>> from nba_model.predict import SignalGenerator
    >>> from nba_model.backtest import DevigCalculator, KellyCalculator
    >>> generator = SignalGenerator(DevigCalculator(), KellyCalculator())
    >>> signals = generator.generate_signals(predictions, market_odds)
    >>> for signal in signals:
    ...     print(f"{signal.matchup}: {signal.bet_type} {signal.side} ({signal.edge:.1%} edge)")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import TYPE_CHECKING

from nba_model.logging import get_logger
from nba_model.types import GameId

if TYPE_CHECKING:
    from nba_model.backtest.devig import DevigCalculator
    from nba_model.backtest.kelly import KellyCalculator
    from nba_model.predict.inference import GamePrediction

logger = get_logger(__name__)

# =============================================================================
# Constants
# =============================================================================

DEFAULT_MIN_EDGE: float = 0.02  # 2% minimum edge
DEFAULT_BANKROLL: float = 10000.0

# Confidence thresholds
HIGH_CONFIDENCE_EDGE: float = 0.05  # 5%+ edge
MEDIUM_CONFIDENCE_EDGE: float = 0.03  # 3-5% edge

# Standard deviation for spread-to-probability conversion
SPREAD_STD_DEV: float = 13.0  # Historical std dev of NBA margins


# =============================================================================
# Enums
# =============================================================================


class BetType(Enum):
    """Type of bet."""

    MONEYLINE = "moneyline"
    SPREAD = "spread"
    TOTAL = "total"


class Side(Enum):
    """Side of bet."""

    HOME = "home"
    AWAY = "away"
    OVER = "over"
    UNDER = "under"


class Confidence(Enum):
    """Confidence level for signal."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class MarketOdds:
    """Market odds for a game.

    Attributes:
        game_id: NBA game identifier.
        home_ml: Home moneyline decimal odds.
        away_ml: Away moneyline decimal odds.
        spread_home: Home spread line (e.g., -5.5).
        spread_home_odds: Decimal odds for home spread.
        spread_away_odds: Decimal odds for away spread.
        total: Total points line (e.g., 224.5).
        over_odds: Decimal odds for over.
        under_odds: Decimal odds for under.
        timestamp: When odds were captured.
        source: Odds source (e.g., "pinnacle").
    """

    game_id: GameId
    home_ml: float
    away_ml: float
    spread_home: float
    spread_home_odds: float
    spread_away_odds: float
    total: float
    over_odds: float
    under_odds: float
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "unknown"


@dataclass
class BettingSignal:
    """Actionable betting signal.

    Attributes:
        game_id: NBA game identifier.
        game_date: Date of the game.
        matchup: Formatted matchup string (e.g., "LAL @ BOS").
        bet_type: Type of bet (moneyline, spread, total).
        side: Side of bet (home, away, over, under).
        line: Line for spread/total (None for moneyline).
        model_prob: Model's estimated probability.
        market_prob: Fair market probability after devigging.
        edge: Model probability minus market probability.
        recommended_odds: Decimal odds for the bet.
        kelly_fraction: Full Kelly fraction.
        recommended_stake_pct: Recommended stake as percentage of bankroll.
        confidence: Confidence classification (high, medium, low).
        key_factors: List of key factors driving the prediction.
        injury_notes: Notes about relevant injuries.
        model_confidence: Model's internal confidence score.
        injury_uncertainty: Uncertainty from injury situations.
    """

    game_id: GameId
    game_date: date
    matchup: str
    bet_type: str
    side: str
    line: float | None
    model_prob: float
    market_prob: float
    edge: float
    recommended_odds: float
    kelly_fraction: float
    recommended_stake_pct: float
    confidence: str
    key_factors: list[str] = field(default_factory=list)
    injury_notes: list[str] = field(default_factory=list)
    model_confidence: float = 0.5
    injury_uncertainty: float = 0.0


# =============================================================================
# Signal Generator
# =============================================================================


class SignalGenerator:
    """Generate actionable betting signals from game predictions.

    Transforms predictions into betting signals by comparing model
    probabilities against devigged market odds and applying Kelly
    criterion sizing.

    Attributes:
        devig: DevigCalculator for removing vig from odds.
        kelly: KellyCalculator for position sizing.
        min_edge: Minimum edge threshold to generate signal.
        bankroll: Assumed bankroll for stake calculation.

    Example:
        >>> generator = SignalGenerator(devig, kelly, min_edge=0.02)
        >>> signals = generator.generate_signals(predictions, odds)
    """

    def __init__(
        self,
        devig_calculator: DevigCalculator,
        kelly_calculator: KellyCalculator,
        min_edge: float = DEFAULT_MIN_EDGE,
        bankroll: float = DEFAULT_BANKROLL,
    ) -> None:
        """Initialize SignalGenerator.

        Args:
            devig_calculator: DevigCalculator instance.
            kelly_calculator: KellyCalculator instance.
            min_edge: Minimum edge required to generate signal (default 2%).
            bankroll: Assumed bankroll for stake calculations.
        """
        self.devig = devig_calculator
        self.kelly = kelly_calculator
        self.min_edge = min_edge
        self.bankroll = bankroll

        logger.debug(
            "Initialized SignalGenerator with min_edge={:.1%}",
            min_edge,
        )

    def generate_signals(
        self,
        predictions: list[GamePrediction],
        current_odds: dict[GameId, MarketOdds],
    ) -> list[BettingSignal]:
        """Generate betting signals for all predictions.

        Args:
            predictions: List of GamePrediction objects.
            current_odds: Dictionary mapping game_id to MarketOdds.

        Returns:
            List of BettingSignal objects with positive edge.
        """
        signals = []

        for prediction in predictions:
            game_id = prediction.game_id

            if game_id not in current_odds:
                logger.debug("No odds available for game {}", game_id)
                continue

            odds = current_odds[game_id]
            game_signals = self.generate_game_signals(prediction, odds)
            signals.extend(game_signals)

        # Sort by edge descending
        signals.sort(key=lambda s: s.edge, reverse=True)

        logger.info(
            "Generated {} signals from {} predictions",
            len(signals),
            len(predictions),
        )

        return signals

    def generate_game_signals(
        self,
        prediction: GamePrediction,
        odds: MarketOdds,
    ) -> list[BettingSignal]:
        """Generate all possible signals for a single game.

        Checks moneyline, spread, and total markets for positive edge.

        Args:
            prediction: GamePrediction object.
            odds: MarketOdds for the game.

        Returns:
            List of BettingSignal objects where edge > min_edge.
        """
        signals = []

        # Check moneyline
        ml_signals = self._check_moneyline(prediction, odds)
        signals.extend(ml_signals)

        # Check spread
        spread_signals = self._check_spread(prediction, odds)
        signals.extend(spread_signals)

        # Check total
        total_signals = self._check_total(prediction, odds)
        signals.extend(total_signals)

        return signals

    def _check_moneyline(
        self,
        prediction: GamePrediction,
        odds: MarketOdds,
    ) -> list[BettingSignal]:
        """Check moneyline market for value.

        Args:
            prediction: GamePrediction object.
            odds: MarketOdds for the game.

        Returns:
            List of moneyline signals with positive edge.
        """
        signals = []

        # Devig the moneyline market
        try:
            fair_probs = self.devig.power_method_devig(odds.home_ml, odds.away_ml)
        except Exception as e:
            logger.debug("Devigging failed for {}: {}", prediction.game_id, e)
            return signals

        # Use injury-adjusted probabilities
        home_model_prob = prediction.home_win_prob_adjusted
        away_model_prob = 1.0 - home_model_prob

        # Check home moneyline
        home_edge = home_model_prob - fair_probs.home
        if home_edge >= self.min_edge:
            signal = self._create_signal(
                prediction=prediction,
                odds=odds,
                bet_type=BetType.MONEYLINE.value,
                side=Side.HOME.value,
                line=None,
                model_prob=home_model_prob,
                market_prob=fair_probs.home,
                recommended_odds=odds.home_ml,
            )
            signals.append(signal)

        # Check away moneyline
        away_edge = away_model_prob - fair_probs.away
        if away_edge >= self.min_edge:
            signal = self._create_signal(
                prediction=prediction,
                odds=odds,
                bet_type=BetType.MONEYLINE.value,
                side=Side.AWAY.value,
                line=None,
                model_prob=away_model_prob,
                market_prob=fair_probs.away,
                recommended_odds=odds.away_ml,
            )
            signals.append(signal)

        return signals

    def _check_spread(
        self,
        prediction: GamePrediction,
        odds: MarketOdds,
    ) -> list[BettingSignal]:
        """Check spread market for value.

        Converts predicted margin to cover probability and compares
        against devigged spread market.

        Args:
            prediction: GamePrediction object.
            odds: MarketOdds for the game.

        Returns:
            List of spread signals with positive edge.
        """
        signals = []

        # Devig the spread market
        try:
            fair_probs = self.devig.power_method_devig(
                odds.spread_home_odds,
                odds.spread_away_odds,
            )
        except Exception as e:
            logger.debug("Spread devigging failed: {}", e)
            return signals

        # Convert predicted margin to cover probabilities
        predicted_margin = prediction.predicted_margin_adjusted
        spread_line = odds.spread_home

        # Home covers if margin > spread (spread is typically negative for favorite)
        # e.g., home -5.5, needs to win by 6+
        margin_vs_spread = predicted_margin - spread_line

        # Convert to probability using normal distribution
        home_cover_prob = self._margin_to_probability(margin_vs_spread)
        away_cover_prob = 1.0 - home_cover_prob

        # Check home spread
        home_edge = home_cover_prob - fair_probs.home
        if home_edge >= self.min_edge:
            signal = self._create_signal(
                prediction=prediction,
                odds=odds,
                bet_type=BetType.SPREAD.value,
                side=Side.HOME.value,
                line=spread_line,
                model_prob=home_cover_prob,
                market_prob=fair_probs.home,
                recommended_odds=odds.spread_home_odds,
            )
            signals.append(signal)

        # Check away spread
        away_edge = away_cover_prob - fair_probs.away
        if away_edge >= self.min_edge:
            signal = self._create_signal(
                prediction=prediction,
                odds=odds,
                bet_type=BetType.SPREAD.value,
                side=Side.AWAY.value,
                line=-spread_line,  # Away line is opposite
                model_prob=away_cover_prob,
                market_prob=fair_probs.away,
                recommended_odds=odds.spread_away_odds,
            )
            signals.append(signal)

        return signals

    def _check_total(
        self,
        prediction: GamePrediction,
        odds: MarketOdds,
    ) -> list[BettingSignal]:
        """Check totals market for value.

        Converts predicted total to over/under probabilities and compares
        against devigged totals market.

        Args:
            prediction: GamePrediction object.
            odds: MarketOdds for the game.

        Returns:
            List of total signals with positive edge.
        """
        signals = []

        # Devig the totals market
        try:
            fair_probs = self.devig.power_method_devig(
                odds.over_odds,
                odds.under_odds,
            )
        except Exception as e:
            logger.debug("Totals devigging failed: {}", e)
            return signals

        # Convert predicted total to over/under probabilities
        predicted_total = prediction.predicted_total_adjusted
        total_line = odds.total

        # Over hits if actual > line
        total_diff = predicted_total - total_line

        # Convert to probability (using same std dev as spread)
        over_prob = self._margin_to_probability(total_diff)
        under_prob = 1.0 - over_prob

        # Check over
        over_edge = over_prob - fair_probs.home  # Over is "home" in our devig
        if over_edge >= self.min_edge:
            signal = self._create_signal(
                prediction=prediction,
                odds=odds,
                bet_type=BetType.TOTAL.value,
                side=Side.OVER.value,
                line=total_line,
                model_prob=over_prob,
                market_prob=fair_probs.home,
                recommended_odds=odds.over_odds,
            )
            signals.append(signal)

        # Check under
        under_edge = under_prob - fair_probs.away  # Under is "away" in our devig
        if under_edge >= self.min_edge:
            signal = self._create_signal(
                prediction=prediction,
                odds=odds,
                bet_type=BetType.TOTAL.value,
                side=Side.UNDER.value,
                line=total_line,
                model_prob=under_prob,
                market_prob=fair_probs.away,
                recommended_odds=odds.under_odds,
            )
            signals.append(signal)

        return signals

    def _create_signal(
        self,
        prediction: GamePrediction,
        odds: MarketOdds,
        bet_type: str,
        side: str,
        line: float | None,
        model_prob: float,
        market_prob: float,
        recommended_odds: float,
    ) -> BettingSignal:
        """Create a BettingSignal object.

        Args:
            prediction: GamePrediction object.
            odds: MarketOdds for the game.
            bet_type: Type of bet.
            side: Side of bet.
            line: Line for spread/total.
            model_prob: Model probability.
            market_prob: Fair market probability.
            recommended_odds: Decimal odds.

        Returns:
            BettingSignal object.
        """
        edge = model_prob - market_prob

        # Calculate Kelly
        kelly_result = self.kelly.calculate(
            bankroll=self.bankroll,
            model_prob=model_prob,
            decimal_odds=recommended_odds,
            market_prob=market_prob,
        )

        # Determine confidence level
        confidence = self._determine_confidence(
            edge=edge,
            model_confidence=prediction.confidence,
            injury_uncertainty=prediction.injury_uncertainty,
        )

        # Format key factors
        key_factors = [f"{name}: {value:.2f}" for name, value in prediction.top_factors[:3]]

        return BettingSignal(
            game_id=prediction.game_id,
            game_date=prediction.game_date,
            matchup=prediction.matchup,
            bet_type=bet_type,
            side=side,
            line=line,
            model_prob=model_prob,
            market_prob=market_prob,
            edge=edge,
            recommended_odds=recommended_odds,
            kelly_fraction=kelly_result.full_kelly,
            recommended_stake_pct=kelly_result.bet_fraction,
            confidence=confidence,
            key_factors=key_factors,
            injury_notes=[],  # Would be populated from injury data
            model_confidence=prediction.confidence,
            injury_uncertainty=prediction.injury_uncertainty,
        )

    def _margin_to_probability(self, margin_diff: float) -> float:
        """Convert margin difference to probability using normal distribution.

        Uses historical NBA margin standard deviation to convert point
        differences into win/cover probabilities.

        Args:
            margin_diff: Difference between predicted and line.

        Returns:
            Probability (0 to 1).
        """
        import math

        # Standard normal CDF approximation
        z = margin_diff / SPREAD_STD_DEV

        # Approximation of standard normal CDF
        # Using logistic function as fast approximation
        prob = 1.0 / (1.0 + math.exp(-1.7 * z))

        return max(0.01, min(0.99, prob))

    def _determine_confidence(
        self,
        edge: float,
        model_confidence: float,
        injury_uncertainty: float,
    ) -> str:
        """Determine confidence classification for a signal.

        Args:
            edge: Calculated edge.
            model_confidence: Model's internal confidence.
            injury_uncertainty: Uncertainty from injuries.

        Returns:
            Confidence level string (high, medium, low).
        """
        # Penalize for injury uncertainty
        effective_confidence = model_confidence * (1 - injury_uncertainty * 0.5)

        if edge >= HIGH_CONFIDENCE_EDGE and effective_confidence >= 0.6:
            return Confidence.HIGH.value
        elif edge >= MEDIUM_CONFIDENCE_EDGE and effective_confidence >= 0.4:
            return Confidence.MEDIUM.value
        else:
            return Confidence.LOW.value


# =============================================================================
# Utility Functions
# =============================================================================


def create_market_odds(
    game_id: GameId,
    home_ml: float,
    away_ml: float,
    spread_home: float,
    spread_home_odds: float = 1.91,
    spread_away_odds: float = 1.91,
    total: float = 224.5,
    over_odds: float = 1.91,
    under_odds: float = 1.91,
    source: str = "manual",
) -> MarketOdds:
    """Create MarketOdds object with sensible defaults.

    Args:
        game_id: Game identifier.
        home_ml: Home moneyline decimal odds.
        away_ml: Away moneyline decimal odds.
        spread_home: Home spread line.
        spread_home_odds: Spread home odds (default -110 = 1.91).
        spread_away_odds: Spread away odds (default -110 = 1.91).
        total: Total points line.
        over_odds: Over odds (default -110 = 1.91).
        under_odds: Under odds (default -110 = 1.91).
        source: Odds source.

    Returns:
        MarketOdds object.
    """
    return MarketOdds(
        game_id=game_id,
        home_ml=home_ml,
        away_ml=away_ml,
        spread_home=spread_home,
        spread_home_odds=spread_home_odds,
        spread_away_odds=spread_away_odds,
        total=total,
        over_odds=over_odds,
        under_odds=under_odds,
        source=source,
    )


def american_to_decimal(american: int) -> float:
    """Convert American odds to decimal format.

    Args:
        american: American odds (e.g., -110, +150).

    Returns:
        Decimal odds (e.g., 1.91, 2.50).
    """
    if american > 0:
        return 1 + american / 100
    return 1 - 100 / american


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
