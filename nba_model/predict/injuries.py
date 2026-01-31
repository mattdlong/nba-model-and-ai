"""Bayesian injury probability adjustments for NBA predictions.

This module implements Bayesian updating for player availability probabilities,
accounting for injury status, player history, and team context to adjust
game predictions.

Prior Probabilities (from historical data):
    - Probable: 93% play rate
    - Questionable: 55% play rate
    - Doubtful: 3% play rate
    - Out: 0% play rate
    - Available: 100% play rate

Example:
    >>> from nba_model.predict.injuries import InjuryAdjuster
    >>> from nba_model.data import session_scope
    >>> with session_scope() as session:
    ...     adjuster = InjuryAdjuster(session)
    ...     prob = adjuster.get_play_probability(203507, "questionable")
    ...     print(f"Play probability: {prob:.1%}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import TYPE_CHECKING

import pandas as pd

from nba_model.logging import get_logger
from nba_model.types import GameId, PlayerId, TeamId

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = get_logger(__name__)

# =============================================================================
# Constants
# =============================================================================

# Prior probabilities of playing given injury status
PRIOR_PLAY_PROBABILITIES: dict[str, float] = {
    "probable": 0.93,
    "questionable": 0.55,
    "doubtful": 0.03,
    "out": 0.00,
    "available": 1.00,
    "healthy": 1.00,
    "day-to-day": 0.55,  # Alias for questionable
    "gtd": 0.55,  # Game-time decision
}

# Injury type modifiers (multiplied with prior)
INJURY_TYPE_MODIFIERS: dict[str, float] = {
    "rest": 0.8,  # Rest days often lead to sitting
    "load management": 0.7,
    "ankle": 1.0,
    "knee": 0.95,
    "back": 0.9,
    "illness": 0.85,
    "hamstring": 0.9,
    "calf": 0.9,
    "quad": 0.9,
    "shoulder": 1.0,
    "wrist": 1.0,
    "finger": 1.05,  # Minor, often play through
    "toe": 1.0,
}

# Team context modifiers
CONTEXT_MODIFIERS: dict[str, float] = {
    "back_to_back": 0.85,  # More likely to rest
    "playoff_race": 1.15,  # More likely to play
    "tanking": 0.75,  # More likely to rest stars
    "playoff_game": 1.20,  # Much more likely to play
}


# =============================================================================
# Data Classes
# =============================================================================


class InjuryStatus(Enum):
    """Player injury status classification."""

    PROBABLE = "probable"
    QUESTIONABLE = "questionable"
    DOUBTFUL = "doubtful"
    OUT = "out"
    AVAILABLE = "available"


@dataclass
class InjuryReport:
    """Injury report for a player.

    Attributes:
        player_id: Player identifier.
        player_name: Player full name.
        team_id: Team identifier.
        status: Injury status string.
        injury_description: Description of injury.
        report_date: Date of injury report.
    """

    player_id: PlayerId
    player_name: str
    team_id: TeamId
    status: str
    injury_description: str
    report_date: date


@dataclass
class PlayerAvailability:
    """Computed player availability probability.

    Attributes:
        player_id: Player identifier.
        player_name: Player full name.
        status: Injury status.
        play_probability: Computed probability of playing.
        prior_probability: Base prior probability.
        history_modifier: Modifier from player history.
        context_modifier: Modifier from team context.
        injury_modifier: Modifier from injury type.
        rapm: Player's RAPM value.
        minutes_projection: Expected minutes if playing.
    """

    player_id: PlayerId
    player_name: str
    status: str
    play_probability: float
    prior_probability: float
    history_modifier: float = 1.0
    context_modifier: float = 1.0
    injury_modifier: float = 1.0
    rapm: float = 0.0
    minutes_projection: float = 30.0


@dataclass
class InjuryAdjustmentResult:
    """Result of injury adjustment on prediction.

    Attributes:
        home_win_prob_adjusted: Adjusted home win probability.
        predicted_margin_adjusted: Adjusted predicted margin.
        predicted_total_adjusted: Adjusted predicted total.
        injury_uncertainty: Uncertainty score from injuries.
        home_adjustments: List of home player adjustments.
        away_adjustments: List of away player adjustments.
    """

    home_win_prob_adjusted: float
    predicted_margin_adjusted: float
    predicted_total_adjusted: float
    injury_uncertainty: float
    home_adjustments: list[PlayerAvailability] = field(default_factory=list)
    away_adjustments: list[PlayerAvailability] = field(default_factory=list)


# =============================================================================
# Injury Adjuster
# =============================================================================


class InjuryAdjuster:
    """Bayesian adjustment for player availability uncertainty.

    Implements Bayesian updating to compute play probabilities based on:
    1. Prior probability from injury status
    2. Player historical patterns
    3. Team context (back-to-back, playoffs, etc.)
    4. Injury type

    Attributes:
        db_session: SQLAlchemy database session.

    Example:
        >>> adjuster = InjuryAdjuster(session)
        >>> prob = adjuster.get_play_probability(203507, "questionable")
        >>> print(f"LeBron play probability: {prob:.1%}")
    """

    def __init__(self, db_session: Session) -> None:
        """Initialize InjuryAdjuster.

        Args:
            db_session: SQLAlchemy database session.
        """
        self.db_session = db_session
        self._player_history_cache: dict[PlayerId, float] = {}
        logger.debug("Initialized InjuryAdjuster")

    def get_play_probability(
        self,
        player_id: PlayerId,
        injury_status: str,
        injury_type: str | None = None,
        team_context: dict[str, bool] | None = None,
    ) -> float:
        """Calculate probability that player will play.

        Uses Bayesian updating:
            P(play|context) ∝ P(context|play) × P(play|status)

        Args:
            player_id: Player identifier.
            injury_status: Injury status string (e.g., "questionable").
            injury_type: Type of injury (e.g., "ankle", "rest").
            team_context: Context flags (back_to_back, playoff_race, etc.).

        Returns:
            Probability of playing (0 to 1).
        """
        # Get prior probability
        status_lower = injury_status.lower().strip()
        prior = PRIOR_PLAY_PROBABILITIES.get(status_lower, 0.55)

        # Apply injury type modifier
        injury_modifier = 1.0
        if injury_type:
            injury_lower = injury_type.lower().strip()
            for key, mod in INJURY_TYPE_MODIFIERS.items():
                if key in injury_lower:
                    injury_modifier = mod
                    break

        # Apply team context modifier
        context_modifier = 1.0
        if team_context:
            for key, is_active in team_context.items():
                if is_active and key in CONTEXT_MODIFIERS:
                    context_modifier *= CONTEXT_MODIFIERS[key]

        # Apply player history modifier
        history_modifier = self.calculate_player_history_likelihood(
            player_id, injury_status
        )

        # Compute posterior probability
        posterior = prior * injury_modifier * context_modifier * history_modifier

        # Clamp to valid probability range
        return max(0.0, min(1.0, posterior))

    def calculate_player_history_likelihood(
        self,
        player_id: PlayerId,
        injury_status: str,
    ) -> float:
        """Calculate likelihood modifier based on player's historical patterns.

        Some players (e.g., Anthony Davis) rest more frequently, while others
        (e.g., LeBron) play through minor injuries.

        Args:
            player_id: Player identifier.
            injury_status: Current injury status.

        Returns:
            Likelihood modifier (typically 0.8 to 1.2).
        """
        # Check cache
        if player_id in self._player_history_cache:
            return self._player_history_cache[player_id]

        # Default modifier
        modifier = 1.0

        try:
            from nba_model.data.models import PlayerGameStats

            # Count games played vs expected
            # This is a simplified version - full implementation would track
            # actual injury reports over time
            recent_games = (
                self.db_session.query(PlayerGameStats)
                .filter(PlayerGameStats.player_id == player_id)
                .order_by(PlayerGameStats.id.desc())
                .limit(82)  # One season
                .all()
            )

            if len(recent_games) >= 20:
                # Calculate average minutes
                avg_minutes = sum(g.minutes or 0 for g in recent_games) / len(recent_games)

                # Players with high minutes tend to play more
                if avg_minutes > 32:
                    modifier = 1.1  # Iron man
                elif avg_minutes < 20:
                    modifier = 0.9  # Injury prone or role player

        except Exception as e:
            logger.debug("Could not calculate player history: {}", e)

        self._player_history_cache[player_id] = modifier
        return modifier

    def adjust_prediction(
        self,
        base_prediction,
        game_id: GameId,
    ):
        """Adjust a GamePrediction based on injury report.

        For each questionable player:
        1. Compute play probability via Bayesian update
        2. Run model for both scenarios (plays / sits)
        3. Compute expected values weighted by probability

        Args:
            base_prediction: Base GamePrediction to adjust.
            game_id: Game identifier.

        Returns:
            Adjusted GamePrediction with uncertainty scores.
        """
        adjustment = self.adjust_prediction_values(
            game_id=game_id,
            base_home_win_prob=base_prediction.home_win_prob,
            base_margin=base_prediction.predicted_margin,
            base_total=base_prediction.predicted_total,
        )

        # Update prediction fields
        base_prediction.home_win_prob_adjusted = adjustment["home_win_prob_adjusted"]
        base_prediction.predicted_margin_adjusted = adjustment["predicted_margin_adjusted"]
        base_prediction.predicted_total_adjusted = adjustment["predicted_total_adjusted"]
        base_prediction.injury_uncertainty = adjustment["injury_uncertainty"]

        return base_prediction

    def adjust_prediction_values(
        self,
        game_id: GameId,
        base_home_win_prob: float,
        base_margin: float,
        base_total: float,
    ) -> dict:
        """Adjust prediction values based on injury report.

        Args:
            game_id: Game identifier.
            base_home_win_prob: Base home win probability.
            base_margin: Base predicted margin.
            base_total: Base predicted total.

        Returns:
            Dictionary with adjusted values and uncertainty.
        """
        # Get game teams
        from nba_model.data.models import Game

        game = (
            self.db_session.query(Game)
            .filter(Game.game_id == game_id)
            .first()
        )

        if game is None:
            return {
                "home_win_prob_adjusted": base_home_win_prob,
                "predicted_margin_adjusted": base_margin,
                "predicted_total_adjusted": base_total,
                "injury_uncertainty": 0.0,
            }

        # Get injury reports for both teams
        home_injuries = self._get_team_injuries(game.home_team_id)
        away_injuries = self._get_team_injuries(game.away_team_id)

        # Calculate adjustments
        home_adjustment, home_uncertainty = self._calculate_team_adjustment(
            home_injuries, game.home_team_id
        )
        away_adjustment, away_uncertainty = self._calculate_team_adjustment(
            away_injuries, game.away_team_id
        )

        # Adjust predictions
        # Win probability adjustment based on relative team strength changes
        net_adjustment = home_adjustment - away_adjustment
        adjusted_win_prob = base_home_win_prob + net_adjustment * 0.1

        # Margin adjustment (RAPM-based)
        adjusted_margin = base_margin + net_adjustment * 2.0

        # Total adjustment (both teams affected)
        total_adjustment = (home_adjustment + away_adjustment) * -0.5
        adjusted_total = base_total + total_adjustment

        # Clamp values
        adjusted_win_prob = max(0.01, min(0.99, adjusted_win_prob))
        adjusted_margin = max(-35, min(35, adjusted_margin))
        adjusted_total = max(175, min(270, adjusted_total))

        # Combined uncertainty
        total_uncertainty = (home_uncertainty + away_uncertainty) / 2

        return {
            "home_win_prob_adjusted": adjusted_win_prob,
            "predicted_margin_adjusted": adjusted_margin,
            "predicted_total_adjusted": adjusted_total,
            "injury_uncertainty": total_uncertainty,
        }

    def calculate_replacement_impact(
        self,
        player_id: PlayerId,
        replacement_id: PlayerId | None,
        team_id: TeamId,
    ) -> float:
        """Calculate win probability impact of player replacement.

        Uses RAPM differential between starter and replacement.

        Args:
            player_id: ID of player potentially missing.
            replacement_id: ID of likely replacement (None for unknown).
            team_id: Team identifier.

        Returns:
            Impact on win probability (negative = hurts team).
        """
        from nba_model.data.models import PlayerRAPM

        # Get starter RAPM
        starter_rapm = (
            self.db_session.query(PlayerRAPM)
            .filter(PlayerRAPM.player_id == player_id)
            .order_by(PlayerRAPM.calculation_date.desc())
            .first()
        )

        starter_value = starter_rapm.rapm if starter_rapm else 0.0

        # Get replacement RAPM
        replacement_value = 0.0
        if replacement_id:
            replacement_rapm = (
                self.db_session.query(PlayerRAPM)
                .filter(PlayerRAPM.player_id == replacement_id)
                .order_by(PlayerRAPM.calculation_date.desc())
                .first()
            )
            replacement_value = replacement_rapm.rapm if replacement_rapm else -1.0
        else:
            # Assume replacement-level player
            replacement_value = -2.0

        # RAPM is per 100 possessions, convert to approximate win probability impact
        # Roughly 2.7 RAPM = 1 win over 82 games
        rapm_diff = starter_value - replacement_value
        impact = rapm_diff * 0.003  # Approximate per-game impact

        return impact

    def _get_team_injuries(self, team_id: TeamId) -> list[InjuryReport]:
        """Get current injury reports for a team.

        Note: This is a simplified implementation. Full version would
        fetch from NBA API or web scraping.
        """
        # In production, this would query current injury data
        # For now, return empty list (no injuries)
        return []

    def _calculate_team_adjustment(
        self,
        injuries: list[InjuryReport],
        team_id: TeamId,
    ) -> tuple[float, float]:
        """Calculate team-level adjustment and uncertainty from injuries.

        Returns:
            Tuple of (adjustment_value, uncertainty_score).
        """
        if not injuries:
            return 0.0, 0.0

        total_adjustment = 0.0
        total_uncertainty = 0.0

        for injury in injuries:
            # Get play probability
            play_prob = self.get_play_probability(
                injury.player_id,
                injury.status,
                injury.injury_description,
            )

            # Get player impact
            impact = self.calculate_replacement_impact(
                injury.player_id,
                None,  # Unknown replacement
                team_id,
            )

            # Weighted adjustment by probability of sitting
            sit_prob = 1.0 - play_prob
            total_adjustment += impact * sit_prob

            # Uncertainty from questionable players (not certain outcomes)
            if 0.1 < play_prob < 0.9:
                total_uncertainty += abs(impact) * min(play_prob, sit_prob)

        return total_adjustment, total_uncertainty


# =============================================================================
# Injury Report Fetcher
# =============================================================================


class InjuryReportFetcher:
    """Fetch current injury status data from external sources.

    Note: Full implementation would use NBA API or web scraping.
    This is a simplified version for the Phase 7 implementation.
    """

    def __init__(self, api_delay: float = 0.6) -> None:
        """Initialize InjuryReportFetcher.

        Args:
            api_delay: Delay between API calls in seconds.
        """
        self.api_delay = api_delay
        logger.debug("Initialized InjuryReportFetcher")

    def get_current_injuries(self) -> pd.DataFrame:
        """Get current injury report.

        Returns:
            DataFrame with columns:
                player_id, team_id, status, injury_description, report_date
        """
        # In production, this would fetch from NBA API
        # For now, return empty DataFrame
        return pd.DataFrame(
            columns=[
                "player_id",
                "player_name",
                "team_id",
                "status",
                "injury_description",
                "report_date",
            ]
        )

    def get_team_injuries(self, team_id: TeamId) -> pd.DataFrame:
        """Get injury report for a specific team.

        Args:
            team_id: Team identifier.

        Returns:
            Filtered injury DataFrame for the team.
        """
        all_injuries = self.get_current_injuries()
        return all_injuries[all_injuries["team_id"] == team_id]


# =============================================================================
# Utility Functions
# =============================================================================


def parse_injury_status(status_str: str) -> str:
    """Parse injury status string to standardized format.

    Args:
        status_str: Raw injury status string.

    Returns:
        Standardized status string.
    """
    status_lower = status_str.lower().strip()

    # Exact matches first (most specific)
    exact_mappings = {
        "out": "out",
        "o": "out",
        "doubtful": "doubtful",
        "d": "doubtful",
        "questionable": "questionable",
        "q": "questionable",
        "gtd": "questionable",
        "probable": "probable",
        "p": "probable",
        "available": "available",
        "active": "available",
        "healthy": "available",
    }

    # Check exact match first
    if status_lower in exact_mappings:
        return exact_mappings[status_lower]

    # Partial matches (ordered by priority - most specific first)
    partial_mappings = [
        ("doubtful", "doubtful"),
        ("questionable", "questionable"),
        ("game time decision", "questionable"),
        ("day-to-day", "questionable"),
        ("probable", "probable"),
        ("available", "available"),
        ("active", "available"),
        ("healthy", "available"),
        ("out", "out"),  # "out" last to avoid matching "doubtful"
    ]

    for key, value in partial_mappings:
        if key in status_lower:
            return value

    return "questionable"  # Default to questionable if unknown
