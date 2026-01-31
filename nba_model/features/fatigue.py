"""Fatigue calculator for rest, travel, and load metrics.

This module implements fatigue-related feature calculations including
rest days, travel distance using haversine formula, and schedule flags
(back-to-back, 3-in-4, etc.).

Travel distance is calculated using haversine formula for geodesic
distance between arena coordinates.

Example:
    >>> from nba_model.features.fatigue import FatigueCalculator
    >>> calculator = FatigueCalculator()
    >>> indicators = calculator.calculate_schedule_flags(team_id, game_date, games_df)
    >>> print(indicators['back_to_back'])
    True
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING

import numpy as np
from haversine import Unit, haversine

from nba_model.logging import get_logger
from nba_model.types import FatigueIndicators, PlayerId, TeamId

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)

# Default lookback windows
DEFAULT_TRAVEL_LOOKBACK_DAYS: int = 7
DEFAULT_PLAYER_LOAD_LOOKBACK_GAMES: int = 5


@dataclass
class PlayerLoadMetrics:
    """Container for player load metrics."""

    avg_minutes: float
    total_distance_miles: float
    minutes_trend: float


# NBA Arena Coordinates (latitude, longitude)
# Source: Various public sources, manually verified
ARENA_COORDS: dict[str, tuple[float, float]] = {
    # Eastern Conference - Atlantic
    "BOS": (42.3662, -71.0621),  # TD Garden, Boston
    "BKN": (40.6826, -73.9754),  # Barclays Center, Brooklyn
    "NYK": (40.7505, -73.9934),  # Madison Square Garden, New York
    "PHI": (39.9012, -75.1720),  # Wells Fargo Center, Philadelphia
    "TOR": (43.6435, -79.3791),  # Scotiabank Arena, Toronto
    # Eastern Conference - Central
    "CHI": (41.8807, -87.6742),  # United Center, Chicago
    "CLE": (41.4965, -81.6882),  # Rocket Mortgage FieldHouse, Cleveland
    "DET": (42.3410, -83.0551),  # Little Caesars Arena, Detroit
    "IND": (39.7640, -86.1555),  # Gainbridge Fieldhouse, Indianapolis
    "MIL": (43.0451, -87.9175),  # Fiserv Forum, Milwaukee
    # Eastern Conference - Southeast
    "ATL": (33.7573, -84.3963),  # State Farm Arena, Atlanta
    "CHA": (35.2251, -80.8392),  # Spectrum Center, Charlotte
    "MIA": (25.7814, -80.1870),  # FTX Arena, Miami (now Kaseya Center)
    "ORL": (28.5392, -81.3839),  # Amway Center, Orlando
    "WAS": (38.8981, -77.0209),  # Capital One Arena, Washington
    # Western Conference - Northwest
    "DEN": (39.7487, -105.0077),  # Ball Arena, Denver
    "MIN": (44.9795, -93.2760),  # Target Center, Minneapolis
    "OKC": (35.4634, -97.5151),  # Paycom Center, Oklahoma City
    "POR": (45.5316, -122.6668),  # Moda Center, Portland
    "UTA": (40.7683, -111.9011),  # Delta Center, Salt Lake City
    # Western Conference - Pacific
    "GSW": (37.7680, -122.3877),  # Chase Center, San Francisco
    "LAC": (34.0430, -118.2673),  # Crypto.com Arena, Los Angeles (shared)
    "LAL": (34.0430, -118.2673),  # Crypto.com Arena, Los Angeles (shared)
    "PHX": (33.4457, -112.0712),  # Footprint Center, Phoenix
    "SAC": (38.5802, -121.4997),  # Golden 1 Center, Sacramento
    # Western Conference - Southwest
    "DAL": (32.7905, -96.8103),  # American Airlines Center, Dallas
    "HOU": (29.7508, -95.3621),  # Toyota Center, Houston
    "MEM": (35.1382, -90.0505),  # FedExForum, Memphis
    "NOP": (29.9490, -90.0821),  # Smoothie King Center, New Orleans
    "SAS": (29.4270, -98.4375),  # AT&T Center, San Antonio
}

# Team ID to abbreviation mapping (NBA API team IDs)
TEAM_ID_TO_ABBREV: dict[int, str] = {
    1610612737: "ATL",
    1610612738: "BOS",
    1610612739: "CLE",
    1610612740: "NOP",
    1610612741: "CHI",
    1610612742: "DAL",
    1610612743: "DEN",
    1610612744: "GSW",
    1610612745: "HOU",
    1610612746: "LAC",
    1610612747: "LAL",
    1610612748: "MIA",
    1610612749: "MIL",
    1610612750: "MIN",
    1610612751: "BKN",
    1610612752: "NYK",
    1610612753: "ORL",
    1610612754: "IND",
    1610612755: "PHI",
    1610612756: "PHX",
    1610612757: "POR",
    1610612758: "SAC",
    1610612759: "SAS",
    1610612760: "OKC",
    1610612761: "TOR",
    1610612762: "UTA",
    1610612763: "MEM",
    1610612764: "WAS",
    1610612765: "DET",
    1610612766: "CHA",
}


class FatigueCalculator:
    """Fatigue calculator for rest, travel, and schedule metrics.

    Calculates fatigue-related features for teams and players including:
    - Rest days since last game
    - Travel distance over rolling windows
    - Schedule flags (back-to-back, 3-in-4, 4-in-5)
    - Home stand / road trip indicators
    - Player load metrics (minutes, distance run)

    Attributes:
        arena_coords: Dict mapping team abbreviation to (lat, lon).
        team_id_to_abbrev: Dict mapping team ID to abbreviation.

    Example:
        >>> calculator = FatigueCalculator()
        >>> rest = calculator.calculate_rest_days(team_id, game_date, games_df)
        >>> print(f"Rest days: {rest}")
    """

    def __init__(self) -> None:
        """Initialize fatigue calculator with arena coordinates."""
        self.arena_coords = ARENA_COORDS
        self.team_id_to_abbrev = TEAM_ID_TO_ABBREV

    def _get_team_abbrev(self, team_id: TeamId) -> str | None:
        """Get team abbreviation from ID.

        Args:
            team_id: NBA team ID.

        Returns:
            Team abbreviation or None if not found.
        """
        return self.team_id_to_abbrev.get(team_id)

    def _get_arena_coords(self, team_id: TeamId) -> tuple[float, float] | None:
        """Get arena coordinates for a team.

        Args:
            team_id: NBA team ID.

        Returns:
            Tuple of (latitude, longitude) or None if not found.
        """
        abbrev = self._get_team_abbrev(team_id)
        if abbrev is None:
            return None
        return self.arena_coords.get(abbrev)

    def _get_team_games(
        self,
        team_id: TeamId,
        games_df: pd.DataFrame,
        before_date: date | None = None,
        after_date: date | None = None,
    ) -> pd.DataFrame:
        """Get games for a specific team within date range.

        Args:
            team_id: NBA team ID.
            games_df: DataFrame with columns game_date, home_team_id, away_team_id.
            before_date: Only include games before this date.
            after_date: Only include games after this date.

        Returns:
            DataFrame of team's games sorted by date.
        """
        import pandas as pd

        mask = (games_df["home_team_id"] == team_id) | (
            games_df["away_team_id"] == team_id
        )
        team_games = games_df[mask].copy()

        # Convert game_date if needed
        if not pd.api.types.is_datetime64_any_dtype(team_games["game_date"]):
            team_games["game_date"] = pd.to_datetime(team_games["game_date"])

        if before_date is not None:
            if isinstance(before_date, date) and not isinstance(
                before_date, pd.Timestamp
            ):
                before_date = pd.Timestamp(before_date)
            team_games = team_games[team_games["game_date"] < before_date]

        if after_date is not None:
            if isinstance(after_date, date) and not isinstance(
                after_date, pd.Timestamp
            ):
                after_date = pd.Timestamp(after_date)
            team_games = team_games[team_games["game_date"] > after_date]

        return team_games.sort_values("game_date")

    def calculate_rest_days(
        self,
        team_id: TeamId,
        game_date: date,
        games_df: pd.DataFrame,
    ) -> int:
        """Calculate days since team's last game.

        Args:
            team_id: NBA team ID.
            game_date: Date of current game.
            games_df: DataFrame with game data.

        Returns:
            Number of days since last game. Returns 7 if no prior games found.
        """
        import pandas as pd

        prior_games = self._get_team_games(team_id, games_df, before_date=game_date)

        if len(prior_games) == 0:
            return 7  # Default for season opener

        last_game_date = pd.Timestamp(prior_games["game_date"].max()).date()
        if isinstance(game_date, pd.Timestamp):
            game_date = game_date.date()

        rest_days = (
            game_date - last_game_date
        ).days - 1  # -1 because game day doesn't count
        return max(0, rest_days)

    def calculate_travel_distance(
        self,
        team_id: TeamId,
        game_date: date,
        games_df: pd.DataFrame,
        lookback_days: int = DEFAULT_TRAVEL_LOOKBACK_DAYS,
    ) -> float:
        """Calculate total miles traveled in last N days.

        Uses haversine formula for geodesic distance between arenas.

        Args:
            team_id: NBA team ID.
            game_date: Date of current game.
            games_df: DataFrame with game data.
            lookback_days: Number of days to look back.

        Returns:
            Total miles traveled. Returns 0 if team hasn't traveled.
        """
        import pandas as pd

        if isinstance(game_date, pd.Timestamp):
            game_date = game_date.date()

        lookback_start = game_date - timedelta(days=lookback_days)

        # Get games in lookback window
        recent_games = self._get_team_games(
            team_id,
            games_df,
            before_date=game_date + timedelta(days=1),
            after_date=lookback_start,
        )

        if len(recent_games) < 2:
            return 0.0

        # Calculate travel between consecutive venues
        total_distance = 0.0
        prev_coords: tuple[float, float] | None = None

        for _, game in recent_games.iterrows():
            # Determine venue (home or away)
            if game["home_team_id"] == team_id:
                venue_team_id = team_id  # Playing at home
            else:
                venue_team_id = game["home_team_id"]  # Playing away

            current_coords = self._get_arena_coords(venue_team_id)
            if current_coords is None:
                continue

            if prev_coords is not None:
                # Calculate haversine distance
                distance = haversine(prev_coords, current_coords, unit=Unit.MILES)
                total_distance += distance

            prev_coords = current_coords

        return round(total_distance, 1)

    def calculate_schedule_flags(
        self,
        team_id: TeamId,
        game_date: date,
        games_df: pd.DataFrame,
    ) -> FatigueIndicators:
        """Calculate schedule fatigue flags.

        Args:
            team_id: NBA team ID.
            game_date: Date of current game.
            games_df: DataFrame with game data.

        Returns:
            FatigueIndicators TypedDict with keys:
                - rest_days: Days since last game
                - back_to_back: True if second game in consecutive nights
                - three_in_four: True if third game in four nights
                - four_in_five: True if fourth game in five nights
                - travel_miles: Miles traveled in last 7 days
                - home_stand: Count of consecutive home games (including current)
                - road_trip: Count of consecutive away games (including current)
        """
        import pandas as pd

        if isinstance(game_date, pd.Timestamp):
            game_date = game_date.date()

        rest_days = self.calculate_rest_days(team_id, game_date, games_df)
        travel_miles = self.calculate_travel_distance(team_id, game_date, games_df)

        # Check back-to-back (played yesterday)
        back_to_back = rest_days == 0

        # Check 3-in-4 and 4-in-5
        three_in_four = self._check_n_in_m_nights(team_id, game_date, games_df, 3, 4)
        four_in_five = self._check_n_in_m_nights(team_id, game_date, games_df, 4, 5)

        # Calculate home stand / road trip
        home_stand, road_trip = self._calculate_consecutive_venue(
            team_id, game_date, games_df
        )

        return {
            "rest_days": rest_days,
            "back_to_back": back_to_back,
            "3_in_4": three_in_four,
            "4_in_5": four_in_five,
            "travel_miles": travel_miles,
            "home_stand": home_stand,
            "road_trip": road_trip,
        }

    def _check_n_in_m_nights(
        self,
        team_id: TeamId,
        game_date: date,
        games_df: pd.DataFrame,
        n_games: int,
        m_nights: int,
    ) -> bool:
        """Check if team is playing Nth game in M nights.

        Args:
            team_id: NBA team ID.
            game_date: Date of current game.
            games_df: DataFrame with game data.
            n_games: Number of games (including current).
            m_nights: Number of nights (days span).

        Returns:
            True if playing Nth game in M nights.
        """
        import pandas as pd

        if isinstance(game_date, pd.Timestamp):
            game_date = game_date.date()

        window_start = game_date - timedelta(days=m_nights - 1)

        # Get games in window (including current day)
        window_games = self._get_team_games(
            team_id,
            games_df,
            before_date=game_date + timedelta(days=1),
            after_date=window_start - timedelta(days=1),
        )

        return len(window_games) >= n_games

    def _calculate_consecutive_venue(
        self,
        team_id: TeamId,
        game_date: date,
        games_df: pd.DataFrame,
    ) -> tuple[int, int]:
        """Calculate consecutive home/away game counts.

        Args:
            team_id: NBA team ID.
            game_date: Date of current game.
            games_df: DataFrame with game data.

        Returns:
            Tuple of (home_stand_count, road_trip_count).
            One will be 0, the other positive.
        """
        import pandas as pd

        if isinstance(game_date, pd.Timestamp):
            game_date = game_date.date()

        # Get recent games including current
        recent_games = self._get_team_games(
            team_id,
            games_df,
            before_date=game_date + timedelta(days=1),
        )

        if len(recent_games) == 0:
            return 1, 0  # Assume home game

        # Sort descending (most recent first)
        recent_games = recent_games.sort_values("game_date", ascending=False)

        # Determine if current game is home or away
        current_game = recent_games.iloc[0]
        is_home = current_game["home_team_id"] == team_id

        # Count consecutive same-venue games
        consecutive = 1
        for idx in range(1, len(recent_games)):
            game = recent_games.iloc[idx]
            game_is_home = game["home_team_id"] == team_id
            if game_is_home == is_home:
                consecutive += 1
            else:
                break

        if is_home:
            return consecutive, 0
        else:
            return 0, consecutive

    def calculate_player_load(
        self,
        player_id: PlayerId,
        game_date: date,
        player_stats_df: pd.DataFrame,
        lookback_games: int = DEFAULT_PLAYER_LOAD_LOOKBACK_GAMES,
    ) -> PlayerLoadMetrics:
        """Calculate player load metrics over recent games.

        Args:
            player_id: NBA player ID.
            game_date: Date of current game.
            player_stats_df: DataFrame with columns:
                - player_id, game_date, minutes, distance_miles
            lookback_games: Number of prior games to analyze.

        Returns:
            PlayerLoadMetrics with avg_minutes, total_distance, minutes_trend.
        """
        import pandas as pd

        if isinstance(game_date, pd.Timestamp):
            game_date = game_date.date()

        # Filter to player's games before current date
        mask = player_stats_df["player_id"] == player_id
        player_games = player_stats_df[mask].copy()

        if not pd.api.types.is_datetime64_any_dtype(player_games["game_date"]):
            player_games["game_date"] = pd.to_datetime(player_games["game_date"])

        player_games = player_games[
            player_games["game_date"] < pd.Timestamp(game_date)
        ].sort_values("game_date", ascending=False)

        # Take last N games
        recent_games = player_games.head(lookback_games)

        if len(recent_games) == 0:
            return PlayerLoadMetrics(
                avg_minutes=0.0,
                total_distance_miles=0.0,
                minutes_trend=0.0,
            )

        # Calculate metrics
        minutes = recent_games["minutes"].fillna(0).values
        avg_minutes = float(minutes.mean())

        # Total distance run
        if "distance_miles" in recent_games.columns:
            total_distance = float(recent_games["distance_miles"].fillna(0).sum())
        else:
            total_distance = 0.0

        # Minutes trend (slope of linear fit)
        if len(minutes) >= 2:
            x = np.arange(len(minutes))
            coeffs = np.polyfit(x, minutes, 1)
            minutes_trend = float(coeffs[0])  # Slope
        else:
            minutes_trend = 0.0

        return PlayerLoadMetrics(
            avg_minutes=round(avg_minutes, 1),
            total_distance_miles=round(total_distance, 1),
            minutes_trend=round(minutes_trend, 2),
        )


def calculate_haversine_distance(
    coord1: tuple[float, float],
    coord2: tuple[float, float],
) -> float:
    """Calculate haversine distance between two coordinates.

    Args:
        coord1: (latitude, longitude) of first point.
        coord2: (latitude, longitude) of second point.

    Returns:
        Distance in miles.
    """
    return haversine(coord1, coord2, unit=Unit.MILES)


__all__ = [
    "ARENA_COORDS",
    "DEFAULT_PLAYER_LOAD_LOOKBACK_GAMES",
    "DEFAULT_TRAVEL_LOOKBACK_DAYS",
    "TEAM_ID_TO_ABBREV",
    "FatigueCalculator",
    "PlayerLoadMetrics",
    "calculate_haversine_distance",
]
