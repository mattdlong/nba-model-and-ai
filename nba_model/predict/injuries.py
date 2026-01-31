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
from typing import TYPE_CHECKING, ClassVar, TypedDict

import pandas as pd

from nba_model.logging import get_logger
from nba_model.types import GameId, PlayerId, TeamId

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from nba_model.predict.inference import GamePrediction

logger = get_logger(__name__)


class AdjustmentValuesDict(TypedDict):
    """Type for injury adjustment values dictionary."""

    home_win_prob_adjusted: float
    predicted_margin_adjusted: float
    predicted_total_adjusted: float
    injury_uncertainty: float


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
                avg_minutes = sum(g.minutes or 0 for g in recent_games) / len(
                    recent_games
                )

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
        base_prediction: GamePrediction,
        game_id: GameId,
    ) -> GamePrediction:
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
        base_prediction.predicted_margin_adjusted = adjustment[
            "predicted_margin_adjusted"
        ]
        base_prediction.predicted_total_adjusted = adjustment[
            "predicted_total_adjusted"
        ]
        base_prediction.injury_uncertainty = adjustment["injury_uncertainty"]

        return base_prediction

    def adjust_prediction_values(
        self,
        game_id: GameId,
        base_home_win_prob: float,
        base_margin: float,
        base_total: float,
    ) -> AdjustmentValuesDict:
        """Adjust prediction values using Bayesian scenario-based expected values.

        Implements the two-scenario adjustment algorithm:
        1. For each player with uncertain status, compute play probability
        2. Compute predictions for both scenarios (plays / sits)
        3. Calculate expected value: E[X] = P(plays)*X_plays + P(sits)*X_sits
        4. Calculate uncertainty from variance of possible outcomes

        Args:
            game_id: Game identifier.
            base_home_win_prob: Base home win probability (assumes all play).
            base_margin: Base predicted margin (assumes all play).
            base_total: Base predicted total (assumes all play).

        Returns:
            Dictionary with adjusted values and uncertainty.
        """
        from nba_model.data.models import Game

        game = self.db_session.query(Game).filter(Game.game_id == game_id).first()

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

        # Calculate scenario-based expected values for each team
        home_result = self._calculate_scenario_expected_values(
            injuries=home_injuries,
            team_id=game.home_team_id,
            base_value=base_home_win_prob,
            is_home_team=True,
        )
        away_result = self._calculate_scenario_expected_values(
            injuries=away_injuries,
            team_id=game.away_team_id,
            base_value=1.0 - base_home_win_prob,  # Away perspective
            is_home_team=False,
        )

        # Compute expected win probability
        # For home team: start with base and apply adjustments
        # The adjustment is the difference between expected value and base
        home_expected = home_result["expected_adjustment"]
        away_expected = away_result["expected_adjustment"]

        # Net adjustment to home win probability
        # Positive home adjustment increases home win prob
        # Positive away adjustment (helps away) decreases home win prob
        net_win_prob_adjustment = home_expected - away_expected
        adjusted_win_prob = base_home_win_prob + net_win_prob_adjustment

        # Margin adjustment follows win probability
        # RAPM difference of ~3 points = ~10% win probability change
        margin_factor = 3.0 / 0.10  # ~30 points per 100% win prob
        adjusted_margin = base_margin + net_win_prob_adjustment * margin_factor

        # Total adjustment based on expected scoring changes
        # Missing players reduce total scoring
        home_total_adj = home_result["expected_total_adjustment"]
        away_total_adj = away_result["expected_total_adjustment"]
        adjusted_total = base_total + home_total_adj + away_total_adj

        # Clamp to reasonable bounds
        adjusted_win_prob = max(0.01, min(0.99, adjusted_win_prob))
        adjusted_margin = max(-35, min(35, adjusted_margin))
        adjusted_total = max(175, min(270, adjusted_total))

        # Combined uncertainty from both teams
        total_uncertainty = (
            home_result["uncertainty"] + away_result["uncertainty"]
        ) / 2.0

        return {
            "home_win_prob_adjusted": adjusted_win_prob,
            "predicted_margin_adjusted": adjusted_margin,
            "predicted_total_adjusted": adjusted_total,
            "injury_uncertainty": total_uncertainty,
        }

    def _calculate_scenario_expected_values(
        self,
        injuries: list[InjuryReport],
        team_id: TeamId,
        base_value: float,
        is_home_team: bool,
    ) -> dict[str, float]:
        """Calculate expected values across play/sit scenarios for a team.

        For each uncertain player, computes:
        E[adjustment] = P(plays) * adj_if_plays + P(sits) * adj_if_sits

        Where adj_if_plays = 0 (base case) and adj_if_sits = -RAPM_impact

        Args:
            injuries: List of injury reports for the team.
            team_id: Team identifier.
            base_value: Base prediction value (assumes all play).
            is_home_team: Whether this is the home team.

        Returns:
            Dict with 'expected_adjustment', 'expected_total_adjustment',
            and 'uncertainty'.
        """
        if not injuries:
            return {
                "expected_adjustment": 0.0,
                "expected_total_adjustment": 0.0,
                "uncertainty": 0.0,
            }

        total_expected_adjustment = 0.0
        total_expected_total_adj = 0.0
        total_variance = 0.0

        for injury in injuries:
            # Get play probability for this player
            play_prob = self.get_play_probability(
                injury.player_id,
                injury.status,
                injury.injury_description,
            )

            # Calculate the RAPM-based impact if player sits
            # This is the win probability impact of replacing this player
            replacement_impact = self.calculate_replacement_impact(
                injury.player_id,
                None,  # Unknown replacement
                team_id,
            )

            # Two scenarios:
            # 1. Player plays (prob = play_prob): adjustment = 0 (base case)
            # 2. Player sits (prob = 1 - play_prob): adjustment = -replacement_impact
            #
            # Expected adjustment = play_prob * 0 + (1 - play_prob) * (-replacement_impact)
            #                     = -(1 - play_prob) * replacement_impact
            sit_prob = 1.0 - play_prob
            expected_adjustment = -sit_prob * replacement_impact

            total_expected_adjustment += expected_adjustment

            # Total points adjustment (players contribute to scoring)
            # Estimate ~0.5 points per game per RAPM point
            starter_rapm = self._get_player_rapm(injury.player_id)
            points_impact = starter_rapm * 0.5  # Approximate points contribution
            expected_total_adj = -sit_prob * points_impact
            total_expected_total_adj += expected_total_adj

            # Calculate variance for uncertainty
            # Variance of binary outcome: p(1-p) * (diff between outcomes)^2
            outcome_diff = abs(replacement_impact)
            variance = play_prob * sit_prob * (outcome_diff**2)
            total_variance += variance

        # Uncertainty is sqrt of total variance, normalized
        uncertainty = min(1.0, (total_variance**0.5) * 10)

        return {
            "expected_adjustment": total_expected_adjustment,
            "expected_total_adjustment": total_expected_total_adj,
            "uncertainty": uncertainty,
        }

    def _get_player_rapm(self, player_id: PlayerId) -> float:
        """Get a player's RAPM value.

        Args:
            player_id: Player identifier.

        Returns:
            Player's RAPM value, or 0.0 if not found.
        """
        from nba_model.data.models import PlayerRAPM

        rapm = (
            self.db_session.query(PlayerRAPM)
            .filter(PlayerRAPM.player_id == player_id)
            .order_by(PlayerRAPM.calculation_date.desc())
            .first()
        )

        return rapm.rapm if rapm else 0.0

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

        Fetches from the NBA injury report endpoint and converts
        to InjuryReport objects.

        Args:
            team_id: Team identifier.

        Returns:
            List of InjuryReport objects for the team.
        """
        fetcher = InjuryReportFetcher()
        df = fetcher.get_team_injuries(team_id)

        if df.empty:
            return []

        injuries = []
        for _, row in df.iterrows():
            injuries.append(
                InjuryReport(
                    player_id=row["player_id"],
                    player_name=row["player_name"],
                    team_id=row["team_id"],
                    status=row["status"],
                    injury_description=row["injury_description"],
                    report_date=row["report_date"],
                )
            )

        return injuries


# =============================================================================
# Injury Report Fetcher
# =============================================================================


class InjuryReportFetcher:
    """Fetch current injury status data from NBA sources.

    Fetches injury data from the NBA's official injury report endpoint.
    Uses rate limiting and error handling to ensure reliable data retrieval.

    Attributes:
        api_delay: Delay between API calls in seconds.
        timeout: Request timeout in seconds.
        _cache: Cached injury data with timestamp.
        _cache_ttl: Cache time-to-live in seconds.
    """

    # NBA injury report endpoint
    INJURY_REPORT_URL = (
        "https://cdn.nba.com/static/json/liveData/injuries/injuries.json"
    )

    # Team ID mapping from NBA tricode to team_id
    TEAM_TRICODE_TO_ID: ClassVar[dict[str, int]] = {
        "ATL": 1610612737,
        "BOS": 1610612738,
        "BKN": 1610612751,
        "CHA": 1610612766,
        "CHI": 1610612741,
        "CLE": 1610612739,
        "DAL": 1610612742,
        "DEN": 1610612743,
        "DET": 1610612765,
        "GSW": 1610612744,
        "HOU": 1610612745,
        "IND": 1610612754,
        "LAC": 1610612746,
        "LAL": 1610612747,
        "MEM": 1610612763,
        "MIA": 1610612748,
        "MIL": 1610612749,
        "MIN": 1610612750,
        "NOP": 1610612740,
        "NYK": 1610612752,
        "OKC": 1610612760,
        "ORL": 1610612753,
        "PHI": 1610612755,
        "PHX": 1610612756,
        "POR": 1610612757,
        "SAC": 1610612758,
        "SAS": 1610612759,
        "TOR": 1610612761,
        "UTA": 1610612762,
        "WAS": 1610612764,
    }

    def __init__(
        self,
        api_delay: float = 0.6,
        timeout: float = 10.0,
        cache_ttl: float = 300.0,
    ) -> None:
        """Initialize InjuryReportFetcher.

        Args:
            api_delay: Delay between API calls in seconds.
            timeout: Request timeout in seconds.
            cache_ttl: Cache time-to-live in seconds (default 5 minutes).
        """
        self.api_delay = api_delay
        self.timeout = timeout
        self._cache: pd.DataFrame | None = None
        self._cache_time: float = 0.0
        self._cache_ttl = cache_ttl
        logger.debug("Initialized InjuryReportFetcher")

    def _fetch_from_nba(self) -> pd.DataFrame:
        """Fetch injury data from NBA's official endpoint.

        Returns:
            DataFrame with injury data.

        Raises:
            Exception: If fetching fails.
        """
        import time

        import requests

        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json",
            "Referer": "https://www.nba.com/",
        }

        time.sleep(self.api_delay)

        response = requests.get(
            self.INJURY_REPORT_URL,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()

        data = response.json()
        injuries = []

        # Parse the JSON response
        # The NBA injury JSON format has a "teams" array with nested player data
        teams_data = data.get("league", {}).get("teams", [])

        for team in teams_data:
            team_tricode = team.get("teamTricode", "")
            team_id = self.TEAM_TRICODE_TO_ID.get(team_tricode, 0)

            players = team.get("players", [])
            for player in players:
                player_id = player.get("personId", 0)
                player_name = f"{player.get('firstName', '')} {player.get('lastName', '')}".strip()
                status = player.get("injuryStatus", "")
                injury_desc = player.get("comment", "")
                report_date_str = player.get("reportDate", "")

                # Parse report date
                try:
                    if report_date_str:
                        report_date = datetime.strptime(
                            report_date_str, "%Y-%m-%d"
                        ).date()
                    else:
                        report_date = date.today()
                except ValueError:
                    report_date = date.today()

                if status:  # Only include players with a status
                    injuries.append(
                        {
                            "player_id": player_id,
                            "player_name": player_name,
                            "team_id": team_id,
                            "status": parse_injury_status(status),
                            "injury_description": injury_desc,
                            "report_date": report_date,
                        }
                    )

        return pd.DataFrame(injuries)

    def get_current_injuries(self) -> pd.DataFrame:
        """Get current injury report.

        Fetches from NBA's official injury endpoint with caching.
        Falls back to empty DataFrame on error.

        Returns:
            DataFrame with columns:
                player_id, player_name, team_id, status, injury_description, report_date
        """
        import time

        # Check cache
        if self._cache is not None:
            cache_age = time.time() - self._cache_time
            if cache_age < self._cache_ttl:
                logger.debug("Using cached injury data (age: {:.1f}s)", cache_age)
                return self._cache.copy()

        try:
            logger.debug("Fetching injury data from NBA endpoint")
            df = self._fetch_from_nba()
            logger.info("Fetched {} injury reports", len(df))

            # Update cache
            self._cache = df
            self._cache_time = time.time()

            return df.copy()

        except Exception as e:
            logger.warning("Failed to fetch injury data: {}", e)

            # Return cached data if available (even if stale)
            if self._cache is not None:
                logger.debug("Returning stale cached injury data")
                return self._cache.copy()

            # Return empty DataFrame as fallback
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
        if all_injuries.empty:
            return all_injuries
        return all_injuries[all_injuries["team_id"] == team_id].copy()


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
