"""Games collector for NBA game schedules and results.

This module provides the GamesCollector class for fetching game data
from the NBA API and transforming it to Game model instances.

Example:
    >>> from nba_model.data.collectors import GamesCollector
    >>> collector = GamesCollector(api_client, session)
    >>> games = collector.collect_season("2023-24")
"""
from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import pandas as pd

from nba_model.data.collectors.base import BaseCollector
from nba_model.data.models import Game

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from nba_model.data.api import NBAApiClient


class GamesCollector(BaseCollector):
    """Collects game records for specified seasons.

    Fetches from LeagueGameFinder and normalizes to Game model format.
    Handles home/away team parsing and game status mapping.
    """

    def __init__(
        self,
        api_client: NBAApiClient,
        db_session: Session | None = None,
    ) -> None:
        """Initialize games collector.

        Args:
            api_client: NBA API client instance.
            db_session: Optional SQLAlchemy session.
        """
        super().__init__(api_client, db_session)

    def collect(
        self,
        season_range: list[str],
        resume_from: str | None = None,
    ) -> list[Game]:
        """Collect games for a range of seasons.

        Args:
            season_range: List of season strings (e.g., ["2022-23", "2023-24"]).
            resume_from: Optional season to resume from.

        Returns:
            List of Game model instances.
        """
        all_games: list[Game] = []

        # Determine starting point
        start_collecting = resume_from is None

        for season in season_range:
            if not start_collecting:
                if season == resume_from:
                    start_collecting = True
                else:
                    continue

            self.logger.info(f"Collecting games for season {season}")
            games = self.collect_season(season)
            all_games.extend(games)

            # Set checkpoint after successful season collection
            self.set_checkpoint(season)

        return all_games

    def collect_game(self, game_id: str) -> list[Game]:
        """Collect data for a single game.

        Note: For games, this returns a list with one item since
        the API returns games in bulk by season.

        Args:
            game_id: NBA game ID string.

        Returns:
            List with single Game instance.
        """
        # For a single game, we'd need to fetch the whole season
        # and filter - not the most efficient approach
        # This method exists to satisfy the abstract base class
        self.logger.warning(
            "collect_game is inefficient for GamesCollector. "
            "Use collect_season or collect_date_range instead."
        )
        return []

    def collect_season(
        self,
        season: str,
        season_type: str = "Regular Season",
    ) -> list[Game]:
        """Collect all games for a season.

        Args:
            season: Season string (e.g., "2023-24").
            season_type: "Regular Season" or "Playoffs".

        Returns:
            List of Game model instances (not yet persisted).
        """
        self.logger.info(f"Collecting games for season {season} ({season_type})")

        df = self.api.get_league_game_finder(
            season=season,
            season_type=season_type,
        )

        if df.empty:
            self.logger.warning(f"No games found for season {season}")
            return []

        # The API returns one row per team per game, so we need to
        # group by GAME_ID and process each unique game
        games = self._transform_games_df(df, season)

        self.logger.info(f"Collected {len(games)} games for season {season}")
        return games

    def collect_date_range(
        self,
        start_date: date,
        end_date: date,
        season: str | None = None,
    ) -> list[Game]:
        """Collect games within a date range.

        Args:
            start_date: Start date (inclusive).
            end_date: End date (inclusive).
            season: Optional season to filter by.

        Returns:
            List of Game model instances.
        """
        self.logger.info(f"Collecting games from {start_date} to {end_date}")

        # If no season provided, determine from date
        if season is None:
            # NBA season starts in October, so use year logic
            year = start_date.year if start_date.month >= 10 else start_date.year - 1
            season = f"{year}-{str(year + 1)[-2:]}"

        df = self.api.get_league_game_finder(
            season=season,
            date_from_nullable=start_date.strftime("%m/%d/%Y"),
            date_to_nullable=end_date.strftime("%m/%d/%Y"),
        )

        if df.empty:
            self.logger.warning(f"No games found in date range")
            return []

        games = self._transform_games_df(df, season)

        # Filter by date range (API filter may not be exact)
        games = [
            g
            for g in games
            if start_date <= g.game_date <= end_date
        ]

        self.logger.info(f"Collected {len(games)} games in date range")
        return games

    def get_game_ids_for_season(
        self,
        season: str,
        season_type: str = "Regular Season",
    ) -> list[str]:
        """Get list of game IDs for a season.

        Useful for other collectors that need game IDs.

        Args:
            season: Season string.
            season_type: "Regular Season" or "Playoffs".

        Returns:
            List of game ID strings.
        """
        df = self.api.get_league_game_finder(
            season=season,
            season_type=season_type,
        )

        if df.empty:
            return []

        game_ids = df["GAME_ID"].unique().tolist()
        self.logger.info(f"Found {len(game_ids)} game IDs for season {season}")
        return game_ids

    def _transform_games_df(self, df: pd.DataFrame, season: str) -> list[Game]:
        """Transform API response DataFrame to Game models.

        The API returns one row per team per game, so we need to
        deduplicate and combine home/away info.

        Args:
            df: DataFrame from LeagueGameFinder.
            season: Season string for the games.

        Returns:
            List of Game model instances.
        """
        games: list[Game] = []
        processed_game_ids: set[str] = set()

        for game_id in df["GAME_ID"].unique():
            if game_id in processed_game_ids:
                continue

            game_rows = df[df["GAME_ID"] == game_id]
            if len(game_rows) < 2:
                # Need both teams to build the game record
                self.logger.warning(f"Incomplete data for game {game_id}")
                continue

            game = self._transform_game(game_rows, game_id, season)
            if game:
                games.append(game)
                processed_game_ids.add(game_id)

        return games

    def _transform_game(
        self,
        game_rows: pd.DataFrame,
        game_id: str,
        season: str,
    ) -> Game | None:
        """Transform game rows to Game model.

        Args:
            game_rows: DataFrame rows for this game (2 rows, one per team).
            game_id: NBA game ID.
            season: Season string.

        Returns:
            Game model instance or None if transformation fails.
        """
        try:
            # Get first row for common game info
            first_row = game_rows.iloc[0]

            # Parse game date
            game_date_str = first_row.get("GAME_DATE", "")
            if isinstance(game_date_str, str):
                game_date = pd.to_datetime(game_date_str).date()
            else:
                game_date = pd.Timestamp(game_date_str).date()

            # Determine home/away from MATCHUP field
            # Format: "LAL vs. BOS" (home) or "BOS @ LAL" (away)
            home_row = None
            away_row = None

            for _, row in game_rows.iterrows():
                matchup = row.get("MATCHUP", "")
                if " vs. " in matchup:
                    home_row = row
                elif " @ " in matchup:
                    away_row = row

            if home_row is None or away_row is None:
                # Fallback: first row with higher PLUS_MINUS is likely home
                # (home teams have slight advantage)
                self.logger.warning(
                    f"Could not determine home/away for game {game_id}, using fallback"
                )
                sorted_rows = game_rows.sort_values("PLUS_MINUS", ascending=False)
                home_row = sorted_rows.iloc[0]
                away_row = sorted_rows.iloc[1]

            # Extract team IDs and scores
            home_team_id = int(home_row["TEAM_ID"])
            away_team_id = int(away_row["TEAM_ID"])
            home_score = int(home_row["PTS"]) if pd.notna(home_row.get("PTS")) else None
            away_score = int(away_row["PTS"]) if pd.notna(away_row.get("PTS")) else None

            # Determine game status
            wl = home_row.get("WL", "")
            if pd.notna(wl) and wl in ("W", "L"):
                status = "completed"
            else:
                status = "scheduled"

            return Game(
                game_id=game_id,
                season_id=season,
                game_date=game_date,
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                home_score=home_score,
                away_score=away_score,
                status=status,
                attendance=None,  # Not available from this endpoint
            )

        except Exception as e:
            self.logger.error(f"Error transforming game {game_id}: {e}")
            return None
