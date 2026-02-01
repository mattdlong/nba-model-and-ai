"""Shots collector for NBA shot chart data.

This module provides the ShotsCollector class for fetching
shot chart data with court coordinates from the NBA API.

Example:
    >>> from nba_model.data.collectors import ShotsCollector
    >>> collector = ShotsCollector(api_client, session)
    >>> shots = collector.collect_game("0022300001")
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pandas as pd

from nba_model.data.collectors.base import BaseCollector
from nba_model.data.models import Shot

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from nba_model.data.api import NBAApiClient


class ShotsCollector(BaseCollector):
    """Collects shot chart data with court coordinates.

    Shot data includes location (loc_x, loc_y), zone classification,
    and make/miss outcome.

    Court coordinates:
    - LOC_X range: -250 to 250 (left to right from behind basket)
    - LOC_Y range: -50 to 900 (baseline to opposite baseline)
    - Stored as integers (tenths of feet from basket)
    """

    def __init__(
        self,
        api_client: NBAApiClient,
        db_session: Session | None = None,
    ) -> None:
        """Initialize shots collector.

        Args:
            api_client: NBA API client instance.
            db_session: Optional SQLAlchemy session.
        """
        super().__init__(api_client, db_session)

    def collect(
        self,
        season_range: list[str],
        resume_from: str | None = None,
    ) -> dict[str, list[Shot]]:
        """Collect shots for all games in specified seasons.

        Note: This method requires game IDs. Use collect_games() with
        specific game IDs for batch collection.

        Args:
            season_range: List of season strings.
            resume_from: Optional game_id to resume from.

        Returns:
            Dict mapping game_id to list of Shot instances.
        """
        self.logger.info(
            f"collect() for ShotsCollector: processing {len(season_range)} seasons"
        )
        # This collector operates on game IDs, not seasons directly
        # Return empty dict; use collect_games() with explicit game IDs
        return {}

    def collect_game(self, game_id: str) -> list[Shot]:
        """Collect all shots for a single game.

        Args:
            game_id: NBA game ID.

        Returns:
            List of Shot model instances.
        """
        self.logger.debug(f"Collecting shots for game {game_id}")

        try:
            df = self.api.get_shot_chart(game_id=game_id)

            if df.empty:
                self.logger.warning(f"No shot data for game {game_id}")
                return []

            shots = []
            for _, row in df.iterrows():
                shot = self._transform_shot(row)
                if shot:
                    shots.append(shot)

            self.logger.debug(f"Collected {len(shots)} shots for game {game_id}")
            return shots

        except Exception as e:
            self.logger.error(f"Error collecting shots for {game_id}: {e}")
            raise

    def collect_player_season(
        self,
        player_id: int,
        season: str,
    ) -> list[Shot]:
        """Collect all shots for a player in a season.

        Args:
            player_id: NBA player ID.
            season: Season string (e.g., "2023-24").

        Returns:
            List of Shot model instances.
        """
        self.logger.debug(
            f"Collecting shots for player {player_id}, season {season}"
        )

        try:
            df = self.api.get_shot_chart(player_id=player_id, season=season)

            if df.empty:
                self.logger.warning(
                    f"No shot data for player {player_id} in {season}"
                )
                return []

            shots = []
            for _, row in df.iterrows():
                shot = self._transform_shot(row)
                if shot:
                    shots.append(shot)

            self.logger.debug(
                f"Collected {len(shots)} shots for player {player_id}"
            )
            return shots

        except Exception as e:
            self.logger.error(
                f"Error collecting shots for player {player_id}: {e}"
            )
            raise

    def collect_games(
        self,
        game_ids: list[str],
        on_error: Literal["raise", "skip", "log"] = "log",
    ) -> dict[str, list[Shot]]:
        """Collect shots for multiple games.

        Args:
            game_ids: List of game IDs.
            on_error: Error handling strategy.

        Returns:
            Dict mapping game_id to list of Shot instances.
        """
        self.logger.info(f"Collecting shots for {len(game_ids)} games")

        results: dict[str, list[Shot]] = {}

        for i, game_id in enumerate(game_ids, 1):
            self._log_progress(i, len(game_ids), game_id)

            try:
                shots = self.collect_game(game_id)
                results[game_id] = shots
            except Exception as e:
                self._handle_error(e, game_id, on_error)
                if on_error == "raise":
                    raise

        self.logger.info(
            f"Collected shots for {len(results)}/{len(game_ids)} games"
        )
        return results

    def _transform_shot(self, row: pd.Series) -> Shot | None:
        """Transform API response row to Shot model.

        Args:
            row: DataFrame row from ShotChartDetail.

        Returns:
            Shot model instance or None.
        """
        try:
            # Required fields - use _safe_int to handle NaN values
            game_id = str(row.get("GAME_ID", ""))
            player_id = self._safe_int(row.get("PLAYER_ID"))
            team_id = self._safe_int(row.get("TEAM_ID"))

            if not game_id or not player_id or not team_id:
                return None

            # Period and time - use _safe_int with defaults
            period = self._safe_int(row.get("PERIOD")) or 1
            minutes_remaining = self._safe_int(row.get("MINUTES_REMAINING")) or 0
            seconds_remaining = self._safe_int(row.get("SECONDS_REMAINING")) or 0

            # Coordinates (required) - use _safe_int with defaults
            loc_x = self._safe_int(row.get("LOC_X")) or 0
            loc_y = self._safe_int(row.get("LOC_Y")) or 0

            # Shot outcome
            shot_made_flag = row.get("SHOT_MADE_FLAG", 0)
            made = bool(shot_made_flag == 1)

            # Shot classification
            shot_type = str(row.get("SHOT_TYPE", "")) or None
            # Normalize to 2PT/3PT
            if shot_type:
                if "3PT" in shot_type.upper():
                    shot_type = "3PT"
                elif "2PT" in shot_type.upper() or "Field Goal" in shot_type:
                    shot_type = "2PT"

            return Shot(
                game_id=game_id,
                player_id=player_id,
                team_id=team_id,
                period=period,
                minutes_remaining=minutes_remaining,
                seconds_remaining=seconds_remaining,
                action_type=self._safe_str(row.get("ACTION_TYPE")),
                shot_type=shot_type,
                shot_zone_basic=self._safe_str(row.get("SHOT_ZONE_BASIC")),
                shot_zone_area=self._safe_str(row.get("SHOT_ZONE_AREA")),
                shot_zone_range=self._safe_str(row.get("SHOT_ZONE_RANGE")),
                shot_distance=self._safe_int(row.get("SHOT_DISTANCE")),
                loc_x=loc_x,
                loc_y=loc_y,
                made=made,
            )

        except Exception as e:
            self.logger.error(f"Error transforming shot: {e}")
            return None

    def _safe_str(self, value: object) -> str | None:
        """Safely convert value to string.

        Args:
            value: Value to convert.

        Returns:
            String or None if invalid/missing.
        """
        if value is None or pd.isna(value):
            return None
        s = str(value)
        return s if s else None

    def _safe_int(self, value: object) -> int | None:
        """Safely convert value to int.

        Args:
            value: Value to convert.

        Returns:
            Integer or None if invalid/missing.
        """
        if value is None or pd.isna(value):
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None
