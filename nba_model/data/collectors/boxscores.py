"""Box scores collector for NBA game statistics.

This module provides the BoxScoreCollector class for fetching
box score data (team and player level) from the NBA API.

Example:
    >>> from nba_model.data.collectors import BoxScoreCollector
    >>> collector = BoxScoreCollector(api_client, session)
    >>> game_stats, player_stats = collector.collect_game("0022300001")
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pandas as pd

from nba_model.data.api import NBAApiError
from nba_model.data.collectors.base import BaseCollector
from nba_model.data.models import GameStats, PlayerGameStats

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from nba_model.data.api import NBAApiClient


class BoxScoreCollector(BaseCollector):
    """Collects box score data (team and player level).

    Fetches traditional stats, advanced stats, and player tracking
    metrics where available. Handles missing data gracefully.
    """

    def __init__(
        self,
        api_client: NBAApiClient,
        db_session: Session | None = None,
    ) -> None:
        """Initialize box score collector.

        Args:
            api_client: NBA API client instance.
            db_session: Optional SQLAlchemy session.
        """
        super().__init__(api_client, db_session)

    def collect(
        self,
        season_range: list[str],
        resume_from: str | None = None,
    ) -> tuple[dict[str, list[GameStats]], dict[str, list[PlayerGameStats]]]:
        """Collect box scores for all games in specified seasons.

        Note: This method requires game IDs. Use collect_games() with
        specific game IDs for batch collection.

        Args:
            season_range: List of season strings.
            resume_from: Optional game_id to resume from.

        Returns:
            Tuple of dicts mapping game_id to stats.
        """
        self.logger.info(
            f"collect() for BoxScoreCollector: processing {len(season_range)} seasons"
        )
        # This collector operates on game IDs, not seasons directly
        # Return empty dicts; use collect_games() with explicit game IDs
        return {}, {}

    def collect_game(
        self, game_id: str
    ) -> tuple[list[GameStats], list[PlayerGameStats]]:
        """Collect full box score for a game.

        Args:
            game_id: NBA game ID.

        Returns:
            Tuple of (game_stats, player_game_stats).
        """
        self.logger.debug(f"Collecting box score for game {game_id}")

        # Fetch all stat sources
        traditional_team, traditional_player = self._fetch_traditional(game_id)
        advanced_team, advanced_player = self._fetch_advanced(game_id)
        tracking = self._fetch_tracking(game_id)

        # Build game stats (team level)
        game_stats = self._build_game_stats(
            game_id, traditional_team, advanced_team
        )

        # Build player stats
        player_stats = self._build_player_stats(
            game_id, traditional_player, advanced_player, tracking
        )

        self.logger.debug(
            f"Collected {len(game_stats)} team stats, "
            f"{len(player_stats)} player stats for game {game_id}"
        )

        return game_stats, player_stats

    def collect_games(
        self,
        game_ids: list[str],
        on_error: Literal["raise", "skip", "log"] = "log",
    ) -> tuple[dict[str, list[GameStats]], dict[str, list[PlayerGameStats]]]:
        """Collect box scores for multiple games.

        Args:
            game_ids: List of game IDs.
            on_error: Error handling strategy.

        Returns:
            Tuple of dicts mapping game_id to stats.
        """
        self.logger.info(f"Collecting box scores for {len(game_ids)} games")

        game_stats_results: dict[str, list[GameStats]] = {}
        player_stats_results: dict[str, list[PlayerGameStats]] = {}

        for i, game_id in enumerate(game_ids, 1):
            self._log_progress(i, len(game_ids), game_id)

            try:
                game_stats, player_stats = self.collect_game(game_id)
                game_stats_results[game_id] = game_stats
                player_stats_results[game_id] = player_stats
            except Exception as e:
                self._handle_error(e, game_id, on_error)
                if on_error == "raise":
                    raise

        self.logger.info(
            f"Collected box scores for {len(game_stats_results)}/{len(game_ids)} games"
        )

        return game_stats_results, player_stats_results

    def _fetch_traditional(
        self, game_id: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch traditional box score.

        Args:
            game_id: NBA game ID.

        Returns:
            Tuple of (team_df, player_df).
        """
        try:
            return self.api.get_boxscore_traditional(game_id)
        except NBAApiError as e:
            self.logger.warning(f"Could not fetch traditional box score: {e}")
            return pd.DataFrame(), pd.DataFrame()

    def _fetch_advanced(
        self, game_id: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch advanced box score.

        Args:
            game_id: NBA game ID.

        Returns:
            Tuple of (team_df, player_df).
        """
        try:
            return self.api.get_boxscore_advanced(game_id)
        except NBAApiError as e:
            self.logger.warning(f"Could not fetch advanced box score: {e}")
            return pd.DataFrame(), pd.DataFrame()

    def _fetch_tracking(self, game_id: str) -> pd.DataFrame | None:
        """Fetch player tracking data.

        Note: Player tracking only available from ~2013-14 season.

        Args:
            game_id: NBA game ID.

        Returns:
            DataFrame or None if not available.
        """
        try:
            return self.api.get_player_tracking(game_id)
        except NBAApiError as e:
            self.logger.debug(f"Player tracking not available: {e}")
            return None

    def _build_game_stats(
        self,
        game_id: str,
        traditional: pd.DataFrame,
        advanced: pd.DataFrame,
    ) -> list[GameStats]:
        """Build GameStats models from team data.

        Args:
            game_id: NBA game ID.
            traditional: Traditional team stats.
            advanced: Advanced team stats.

        Returns:
            List of GameStats (2 per game - home and away).
        """
        game_stats = []

        if traditional.empty:
            return game_stats

        # Merge traditional and advanced if both available
        if not advanced.empty:
            merged = traditional.merge(
                advanced,
                on=["TEAM_ID"],
                how="left",
                suffixes=("", "_adv"),
            )
        else:
            merged = traditional

        for _, row in merged.iterrows():
            gs = self._transform_game_stats(row, game_id)
            if gs:
                game_stats.append(gs)

        return game_stats

    def _build_player_stats(
        self,
        game_id: str,
        traditional: pd.DataFrame,
        advanced: pd.DataFrame,
        tracking: pd.DataFrame | None,
    ) -> list[PlayerGameStats]:
        """Build PlayerGameStats models.

        Args:
            game_id: NBA game ID.
            traditional: Traditional player stats.
            advanced: Advanced player stats.
            tracking: Tracking data (may be None).

        Returns:
            List of PlayerGameStats.
        """
        player_stats = []

        if traditional.empty:
            return player_stats

        # Start with traditional stats
        merged = traditional.copy()

        # Merge advanced stats
        if not advanced.empty:
            merged = merged.merge(
                advanced,
                on=["PLAYER_ID", "TEAM_ID"],
                how="left",
                suffixes=("", "_adv"),
            )

        # Merge tracking stats
        if tracking is not None and not tracking.empty:
            merged = merged.merge(
                tracking,
                on=["PLAYER_ID", "TEAM_ID"],
                how="left",
                suffixes=("", "_track"),
            )

        for _, row in merged.iterrows():
            pgs = self._transform_player_game_stats(row, game_id)
            if pgs:
                player_stats.append(pgs)

        return player_stats

    def _transform_game_stats(
        self, row: pd.Series, game_id: str
    ) -> GameStats | None:
        """Transform to GameStats model.

        Args:
            row: Merged team stats row.
            game_id: NBA game ID.

        Returns:
            GameStats model or None.
        """
        try:
            team_id = int(row["TEAM_ID"])

            # Determine if home team (from MATCHUP field if available)
            matchup = str(row.get("MATCHUP", ""))
            is_home = " vs. " in matchup

            return GameStats(
                game_id=game_id,
                team_id=team_id,
                is_home=is_home,
                # Basic stats
                points=self._safe_int(row.get("PTS")),
                rebounds=self._safe_int(row.get("REB")),
                assists=self._safe_int(row.get("AST")),
                steals=self._safe_int(row.get("STL")),
                blocks=self._safe_int(row.get("BLK")),
                turnovers=self._safe_int(row.get("TO")),
                # Advanced stats (may have _adv suffix)
                offensive_rating=self._safe_float(
                    row.get("OFF_RATING") or row.get("OFF_RATING_adv")
                ),
                defensive_rating=self._safe_float(
                    row.get("DEF_RATING") or row.get("DEF_RATING_adv")
                ),
                pace=self._safe_float(row.get("PACE") or row.get("PACE_adv")),
                efg_pct=self._safe_float(
                    row.get("EFG_PCT") or row.get("EFG_PCT_adv")
                ),
                tov_pct=self._safe_float(
                    row.get("TM_TOV_PCT") or row.get("TM_TOV_PCT_adv")
                ),
                orb_pct=self._safe_float(
                    row.get("OREB_PCT") or row.get("OREB_PCT_adv")
                ),
                ft_rate=self._safe_float(
                    row.get("FTA_RATE") or row.get("FTA_RATE_adv")
                ),
            )

        except Exception as e:
            self.logger.error(f"Error transforming game stats: {e}")
            return None

    def _transform_player_game_stats(
        self, row: pd.Series, game_id: str
    ) -> PlayerGameStats | None:
        """Transform to PlayerGameStats model.

        Args:
            row: Merged player stats row.
            game_id: NBA game ID.

        Returns:
            PlayerGameStats model or None.
        """
        try:
            player_id = int(row["PLAYER_ID"])
            team_id = int(row["TEAM_ID"])

            # Parse minutes (format varies: "12:34" or decimal)
            minutes = self._parse_minutes(row.get("MIN"))

            return PlayerGameStats(
                game_id=game_id,
                player_id=player_id,
                team_id=team_id,
                # Box score stats
                minutes=minutes,
                points=self._safe_int(row.get("PTS")),
                rebounds=self._safe_int(row.get("REB")),
                assists=self._safe_int(row.get("AST")),
                steals=self._safe_int(row.get("STL")),
                blocks=self._safe_int(row.get("BLK")),
                turnovers=self._safe_int(row.get("TO")),
                fgm=self._safe_int(row.get("FGM")),
                fga=self._safe_int(row.get("FGA")),
                fg3m=self._safe_int(row.get("FG3M")),
                fg3a=self._safe_int(row.get("FG3A")),
                ftm=self._safe_int(row.get("FTM")),
                fta=self._safe_int(row.get("FTA")),
                plus_minus=self._safe_int(row.get("PLUS_MINUS")),
                # Player tracking (may have _track suffix)
                distance_miles=self._safe_float(
                    row.get("DIST_MILES") or row.get("DIST_MILES_track")
                ),
                speed_avg=self._safe_float(
                    row.get("AVG_SPEED") or row.get("AVG_SPEED_track")
                ),
            )

        except Exception as e:
            self.logger.error(f"Error transforming player game stats: {e}")
            return None

    def _parse_minutes(self, value: object) -> float | None:
        """Parse minutes value from various formats.

        Args:
            value: Minutes value (could be "12:34" or decimal).

        Returns:
            Minutes as float or None.
        """
        if value is None or pd.isna(value):
            return None

        # If it's already a number
        try:
            return float(value)
        except (ValueError, TypeError):
            pass

        # If it's a string like "12:34"
        try:
            str_val = str(value)
            if ":" in str_val:
                parts = str_val.split(":")
                minutes = int(parts[0])
                seconds = int(parts[1]) if len(parts) > 1 else 0
                return minutes + seconds / 60.0
        except (ValueError, IndexError):
            pass

        return None

    def _safe_int(self, value: object) -> int | None:
        """Safely convert value to int."""
        if value is None or pd.isna(value):
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    def _safe_float(self, value: object) -> float | None:
        """Safely convert value to float."""
        if value is None or pd.isna(value):
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
