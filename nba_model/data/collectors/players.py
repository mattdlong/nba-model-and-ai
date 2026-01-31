"""Players and teams collector for NBA roster data.

This module provides the PlayersCollector class for fetching player
and team data, including hardcoded arena coordinates for travel calculations.

Example:
    >>> from nba_model.data.collectors import PlayersCollector
    >>> collector = PlayersCollector(api_client, session)
    >>> players, player_seasons = collector.collect_rosters("2023-24")
"""
from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

import pandas as pd

from nba_model.data.collectors.base import BaseCollector
from nba_model.data.models import Player, PlayerSeason, Team

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from nba_model.data.api import NBAApiClient


# =============================================================================
# Team Data with Arena Coordinates
# =============================================================================

TEAM_DATA: dict[int, dict[str, Any]] = {
    1610612737: {
        "abbreviation": "ATL",
        "full_name": "Atlanta Hawks",
        "city": "Atlanta",
        "arena_name": "State Farm Arena",
        "arena_lat": 33.757,
        "arena_lon": -84.396,
    },
    1610612738: {
        "abbreviation": "BOS",
        "full_name": "Boston Celtics",
        "city": "Boston",
        "arena_name": "TD Garden",
        "arena_lat": 42.366,
        "arena_lon": -71.062,
    },
    1610612751: {
        "abbreviation": "BKN",
        "full_name": "Brooklyn Nets",
        "city": "Brooklyn",
        "arena_name": "Barclays Center",
        "arena_lat": 40.683,
        "arena_lon": -73.976,
    },
    1610612766: {
        "abbreviation": "CHA",
        "full_name": "Charlotte Hornets",
        "city": "Charlotte",
        "arena_name": "Spectrum Center",
        "arena_lat": 35.225,
        "arena_lon": -80.839,
    },
    1610612741: {
        "abbreviation": "CHI",
        "full_name": "Chicago Bulls",
        "city": "Chicago",
        "arena_name": "United Center",
        "arena_lat": 41.881,
        "arena_lon": -87.674,
    },
    1610612739: {
        "abbreviation": "CLE",
        "full_name": "Cleveland Cavaliers",
        "city": "Cleveland",
        "arena_name": "Rocket Mortgage FieldHouse",
        "arena_lat": 41.496,
        "arena_lon": -81.688,
    },
    1610612742: {
        "abbreviation": "DAL",
        "full_name": "Dallas Mavericks",
        "city": "Dallas",
        "arena_name": "American Airlines Center",
        "arena_lat": 32.790,
        "arena_lon": -96.810,
    },
    1610612743: {
        "abbreviation": "DEN",
        "full_name": "Denver Nuggets",
        "city": "Denver",
        "arena_name": "Ball Arena",
        "arena_lat": 39.749,
        "arena_lon": -105.008,
    },
    1610612765: {
        "abbreviation": "DET",
        "full_name": "Detroit Pistons",
        "city": "Detroit",
        "arena_name": "Little Caesars Arena",
        "arena_lat": 42.341,
        "arena_lon": -83.055,
    },
    1610612744: {
        "abbreviation": "GSW",
        "full_name": "Golden State Warriors",
        "city": "San Francisco",
        "arena_name": "Chase Center",
        "arena_lat": 37.768,
        "arena_lon": -122.388,
    },
    1610612745: {
        "abbreviation": "HOU",
        "full_name": "Houston Rockets",
        "city": "Houston",
        "arena_name": "Toyota Center",
        "arena_lat": 29.751,
        "arena_lon": -95.362,
    },
    1610612754: {
        "abbreviation": "IND",
        "full_name": "Indiana Pacers",
        "city": "Indianapolis",
        "arena_name": "Gainbridge Fieldhouse",
        "arena_lat": 39.764,
        "arena_lon": -86.156,
    },
    1610612746: {
        "abbreviation": "LAC",
        "full_name": "LA Clippers",
        "city": "Los Angeles",
        "arena_name": "Crypto.com Arena",
        "arena_lat": 34.043,
        "arena_lon": -118.267,
    },
    1610612747: {
        "abbreviation": "LAL",
        "full_name": "Los Angeles Lakers",
        "city": "Los Angeles",
        "arena_name": "Crypto.com Arena",
        "arena_lat": 34.043,
        "arena_lon": -118.267,
    },
    1610612763: {
        "abbreviation": "MEM",
        "full_name": "Memphis Grizzlies",
        "city": "Memphis",
        "arena_name": "FedExForum",
        "arena_lat": 35.138,
        "arena_lon": -90.051,
    },
    1610612748: {
        "abbreviation": "MIA",
        "full_name": "Miami Heat",
        "city": "Miami",
        "arena_name": "Kaseya Center",
        "arena_lat": 25.781,
        "arena_lon": -80.188,
    },
    1610612749: {
        "abbreviation": "MIL",
        "full_name": "Milwaukee Bucks",
        "city": "Milwaukee",
        "arena_name": "Fiserv Forum",
        "arena_lat": 43.045,
        "arena_lon": -87.918,
    },
    1610612750: {
        "abbreviation": "MIN",
        "full_name": "Minnesota Timberwolves",
        "city": "Minneapolis",
        "arena_name": "Target Center",
        "arena_lat": 44.980,
        "arena_lon": -93.276,
    },
    1610612740: {
        "abbreviation": "NOP",
        "full_name": "New Orleans Pelicans",
        "city": "New Orleans",
        "arena_name": "Smoothie King Center",
        "arena_lat": 29.949,
        "arena_lon": -90.082,
    },
    1610612752: {
        "abbreviation": "NYK",
        "full_name": "New York Knicks",
        "city": "New York",
        "arena_name": "Madison Square Garden",
        "arena_lat": 40.751,
        "arena_lon": -73.994,
    },
    1610612760: {
        "abbreviation": "OKC",
        "full_name": "Oklahoma City Thunder",
        "city": "Oklahoma City",
        "arena_name": "Paycom Center",
        "arena_lat": 35.463,
        "arena_lon": -97.515,
    },
    1610612753: {
        "abbreviation": "ORL",
        "full_name": "Orlando Magic",
        "city": "Orlando",
        "arena_name": "Amway Center",
        "arena_lat": 28.539,
        "arena_lon": -81.384,
    },
    1610612755: {
        "abbreviation": "PHI",
        "full_name": "Philadelphia 76ers",
        "city": "Philadelphia",
        "arena_name": "Wells Fargo Center",
        "arena_lat": 39.901,
        "arena_lon": -75.172,
    },
    1610612756: {
        "abbreviation": "PHX",
        "full_name": "Phoenix Suns",
        "city": "Phoenix",
        "arena_name": "Footprint Center",
        "arena_lat": 33.446,
        "arena_lon": -112.071,
    },
    1610612757: {
        "abbreviation": "POR",
        "full_name": "Portland Trail Blazers",
        "city": "Portland",
        "arena_name": "Moda Center",
        "arena_lat": 45.532,
        "arena_lon": -122.667,
    },
    1610612758: {
        "abbreviation": "SAC",
        "full_name": "Sacramento Kings",
        "city": "Sacramento",
        "arena_name": "Golden 1 Center",
        "arena_lat": 38.580,
        "arena_lon": -121.500,
    },
    1610612759: {
        "abbreviation": "SAS",
        "full_name": "San Antonio Spurs",
        "city": "San Antonio",
        "arena_name": "Frost Bank Center",
        "arena_lat": 29.427,
        "arena_lon": -98.438,
    },
    1610612761: {
        "abbreviation": "TOR",
        "full_name": "Toronto Raptors",
        "city": "Toronto",
        "arena_name": "Scotiabank Arena",
        "arena_lat": 43.643,
        "arena_lon": -79.379,
    },
    1610612762: {
        "abbreviation": "UTA",
        "full_name": "Utah Jazz",
        "city": "Salt Lake City",
        "arena_name": "Delta Center",
        "arena_lat": 40.768,
        "arena_lon": -111.901,
    },
    1610612764: {
        "abbreviation": "WAS",
        "full_name": "Washington Wizards",
        "city": "Washington",
        "arena_name": "Capital One Arena",
        "arena_lat": 38.898,
        "arena_lon": -77.021,
    },
}


class PlayersCollector(BaseCollector):
    """Collects player information from team rosters.

    Builds Player and PlayerSeason records from roster data,
    and provides static Team data with arena coordinates.
    """

    def __init__(
        self,
        api_client: NBAApiClient,
        db_session: Session | None = None,
    ) -> None:
        """Initialize players collector.

        Args:
            api_client: NBA API client instance.
            db_session: Optional SQLAlchemy session.
        """
        super().__init__(api_client, db_session)

    def collect(
        self,
        season_range: list[str],
        resume_from: str | None = None,
    ) -> tuple[list[Player], list[PlayerSeason]]:
        """Collect players and player-seasons for a range of seasons.

        Args:
            season_range: List of season strings (e.g., ["2022-23", "2023-24"]).
            resume_from: Optional season to resume from.

        Returns:
            Tuple of (all_players, all_player_seasons).
        """
        all_players: dict[int, Player] = {}
        all_player_seasons: list[PlayerSeason] = []

        # Handle resume_from
        start_collecting = resume_from is None

        for season in season_range:
            if not start_collecting:
                if season == resume_from:
                    start_collecting = True
                else:
                    continue

            self.logger.info(f"Collecting rosters for season {season}")
            players, player_seasons = self.collect_rosters(season)

            # Merge players (dedupe by ID)
            for player in players:
                if player.player_id not in all_players:
                    all_players[player.player_id] = player

            all_player_seasons.extend(player_seasons)

            # Set checkpoint after each season
            self.set_checkpoint(season)

        return list(all_players.values()), all_player_seasons

    def collect_game(self, game_id: str) -> list[Player]:
        """Collect players for a single game.

        Note: For players, use collect_rosters instead.

        Args:
            game_id: NBA game ID string.

        Returns:
            Empty list (use collect_rosters instead).
        """
        self.logger.warning(
            "collect_game not supported for PlayersCollector. "
            "Use collect_rosters instead."
        )
        return []

    def collect_rosters(
        self,
        season: str,
        team_ids: list[int] | None = None,
    ) -> tuple[list[Player], list[PlayerSeason]]:
        """Collect all player info from team rosters.

        Args:
            season: Season string (e.g., "2023-24").
            team_ids: Optional list of team IDs (defaults to all 30).

        Returns:
            Tuple of (players, player_seasons).
        """
        if team_ids is None:
            team_ids = list(TEAM_DATA.keys())

        self.logger.info(
            f"Collecting rosters for season {season} ({len(team_ids)} teams)"
        )

        all_players: dict[int, Player] = {}  # Dedup by player_id
        all_player_seasons: list[PlayerSeason] = []

        for i, team_id in enumerate(team_ids, 1):
            self._log_progress(i, len(team_ids), f"team {team_id}")

            try:
                df = self.api.get_team_roster(team_id=team_id, season=season)

                for _, row in df.iterrows():
                    player_id = int(row["PLAYER_ID"])

                    # Only add player if not already seen
                    if player_id not in all_players:
                        player = self._transform_player(row)
                        if player:
                            all_players[player_id] = player

                    # Always add player_season (tracks team changes)
                    player_season = self._transform_player_season(
                        row, season, team_id
                    )
                    if player_season:
                        all_player_seasons.append(player_season)

            except Exception as e:
                self.logger.error(f"Error collecting roster for team {team_id}: {e}")
                continue

        self.logger.info(
            f"Collected {len(all_players)} players, "
            f"{len(all_player_seasons)} player-seasons"
        )

        return list(all_players.values()), all_player_seasons

    def collect_player_details(self, player_id: int) -> Player | None:
        """Collect detailed info for a single player.

        Uses CommonPlayerInfo endpoint for biographical data.

        Args:
            player_id: NBA player ID.

        Returns:
            Player model instance or None if not found.
        """
        self.logger.debug(f"Collecting details for player {player_id}")

        try:
            df = self.api.get_player_info(player_id=player_id)

            if df.empty:
                return None

            row = df.iloc[0]

            # Parse birth date
            birth_date = None
            if pd.notna(row.get("BIRTHDATE")):
                try:
                    birth_date = pd.to_datetime(row["BIRTHDATE"]).date()
                except Exception:
                    pass

            # Parse height (format: "6-8" or similar)
            height_inches = None
            height_str = row.get("HEIGHT", "")
            if height_str and isinstance(height_str, str) and "-" in height_str:
                try:
                    feet, inches = height_str.split("-")
                    height_inches = int(feet) * 12 + int(inches)
                except Exception:
                    pass

            # Parse weight
            weight_lbs = None
            if pd.notna(row.get("WEIGHT")):
                try:
                    weight_lbs = int(row["WEIGHT"])
                except Exception:
                    pass

            return Player(
                player_id=player_id,
                full_name=str(row.get("DISPLAY_FIRST_LAST", "")),
                height_inches=height_inches,
                weight_lbs=weight_lbs,
                birth_date=birth_date,
                draft_year=(
                    int(row["DRAFT_YEAR"])
                    if pd.notna(row.get("DRAFT_YEAR"))
                    and row["DRAFT_YEAR"] != "Undrafted"
                    else None
                ),
                draft_round=(
                    int(row["DRAFT_ROUND"])
                    if pd.notna(row.get("DRAFT_ROUND"))
                    and row["DRAFT_ROUND"] != "Undrafted"
                    else None
                ),
                draft_number=(
                    int(row["DRAFT_NUMBER"])
                    if pd.notna(row.get("DRAFT_NUMBER"))
                    and row["DRAFT_NUMBER"] != "Undrafted"
                    else None
                ),
            )

        except Exception as e:
            self.logger.error(f"Error collecting player {player_id}: {e}")
            return None

    def collect_teams(self) -> list[Team]:
        """Collect all team records.

        Uses hardcoded team data with arena coordinates.
        NBA team IDs are stable across seasons.

        Returns:
            List of Team model instances.
        """
        teams = []
        for team_id, data in TEAM_DATA.items():
            team = Team(
                team_id=team_id,
                abbreviation=data["abbreviation"],
                full_name=data["full_name"],
                city=data["city"],
                arena_name=data["arena_name"],
                arena_lat=data["arena_lat"],
                arena_lon=data["arena_lon"],
            )
            teams.append(team)

        self.logger.info(f"Loaded {len(teams)} teams from static data")
        return teams

    def _transform_player(self, row: pd.Series) -> Player | None:
        """Transform roster row to Player model.

        Args:
            row: DataFrame row from CommonTeamRoster.

        Returns:
            Player model instance or None.
        """
        try:
            player_id = int(row["PLAYER_ID"])
            full_name = str(row.get("PLAYER", ""))

            # Parse height from roster (format varies)
            height_inches = None
            height_str = row.get("HEIGHT", "")
            if height_str and isinstance(height_str, str) and "-" in height_str:
                try:
                    feet, inches = height_str.split("-")
                    height_inches = int(feet) * 12 + int(inches)
                except Exception:
                    pass

            # Parse weight
            weight_lbs = None
            if pd.notna(row.get("WEIGHT")):
                try:
                    weight_lbs = int(row["WEIGHT"])
                except Exception:
                    pass

            # Parse birth date
            birth_date = None
            if pd.notna(row.get("BIRTH_DATE")):
                try:
                    birth_date = pd.to_datetime(row["BIRTH_DATE"]).date()
                except Exception:
                    pass

            return Player(
                player_id=player_id,
                full_name=full_name,
                height_inches=height_inches,
                weight_lbs=weight_lbs,
                birth_date=birth_date,
                draft_year=None,  # Not in roster response
                draft_round=None,
                draft_number=None,
            )

        except Exception as e:
            self.logger.error(f"Error transforming player: {e}")
            return None

    def _transform_player_season(
        self,
        row: pd.Series,
        season: str,
        team_id: int,
    ) -> PlayerSeason | None:
        """Transform roster row to PlayerSeason model.

        Args:
            row: DataFrame row from CommonTeamRoster.
            season: Season string.
            team_id: Team ID.

        Returns:
            PlayerSeason model instance or None.
        """
        try:
            return PlayerSeason(
                player_id=int(row["PLAYER_ID"]),
                season_id=season,
                team_id=team_id,
                position=str(row.get("POSITION", "")) or None,
                jersey_number=str(row.get("NUM", "")) or None,
            )
        except Exception as e:
            self.logger.error(f"Error transforming player_season: {e}")
            return None
