"""NBA API client wrapper with rate limiting and retry logic.

This module provides a robust wrapper around the nba_api library with
automatic rate limiting, retry logic with exponential backoff, and
comprehensive error handling.

Example:
    >>> from nba_model.data.api import NBAApiClient
    >>> client = NBAApiClient(delay=0.6)
    >>> games = client.get_league_game_finder(season="2023-24")
"""
from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd
from requests.exceptions import ReadTimeout, RequestException

from nba_model.config import get_settings

logger = logging.getLogger(__name__)


# =============================================================================
# Custom Exceptions
# =============================================================================


class NBAApiError(Exception):
    """Base exception for NBA API errors."""

    pass


class NBAApiRateLimitError(NBAApiError):
    """Rate limit exceeded (HTTP 429)."""

    pass


class NBAApiNotFoundError(NBAApiError):
    """Resource not found (HTTP 404)."""

    pass


class NBAApiTimeoutError(NBAApiError):
    """Request timeout."""

    pass


# =============================================================================
# Status Code Constants
# =============================================================================

RETRIABLE_STATUS_CODES = {429, 500, 502, 503, 504}
NON_RETRIABLE_STATUS_CODES = {400, 404}


# =============================================================================
# NBA API Client
# =============================================================================


class NBAApiClient:
    """Wrapper around nba_api with reliability features.

    Provides:
    - Automatic rate limiting (configurable delay)
    - Retry logic with exponential backoff
    - Error handling and logging
    - Consistent DataFrame output

    Attributes:
        delay: Seconds between API calls.
        max_retries: Maximum retry attempts for transient errors.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        delay: float | None = None,
        max_retries: int | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Initialize API client.

        Args:
            delay: Seconds between API calls (default from settings).
            max_retries: Maximum retry attempts (default from settings).
            timeout: Request timeout in seconds.
        """
        settings = get_settings()
        self.delay = delay if delay is not None else settings.api_delay
        self.max_retries = (
            max_retries if max_retries is not None else settings.api_max_retries
        )
        self.timeout = timeout
        self._last_request_time: float = 0.0

        logger.debug(
            f"NBAApiClient initialized: delay={self.delay}s, "
            f"max_retries={self.max_retries}, timeout={self.timeout}s"
        )

    def _apply_rate_limit(self) -> None:
        """Apply rate limiting delay if needed."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.delay:
            sleep_time = self.delay - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self._last_request_time = time.time()

    def _request_with_retry(
        self,
        endpoint_class: type,
        **kwargs: Any,
    ) -> Any:
        """Execute API request with retry logic.

        Args:
            endpoint_class: The nba_api endpoint class to use.
            **kwargs: Arguments to pass to the endpoint.

        Returns:
            The endpoint instance (use .get_data_frames() to extract data).

        Raises:
            NBAApiError: On permanent failure.
            NBAApiRateLimitError: If rate limit exceeded after retries.
            NBAApiNotFoundError: If resource not found.
            NBAApiTimeoutError: If request times out after retries.
        """
        last_error: Exception | None = None
        last_was_rate_limit = False

        for attempt in range(self.max_retries + 1):
            self._apply_rate_limit()

            try:
                logger.debug(
                    f"API request: {endpoint_class.__name__} "
                    f"(attempt {attempt + 1}/{self.max_retries + 1})"
                )

                # Execute the request
                result = endpoint_class(timeout=self.timeout, **kwargs)
                logger.debug(f"API request successful: {endpoint_class.__name__}")
                return result

            except ReadTimeout as e:
                last_error = e
                last_was_rate_limit = False
                logger.warning(
                    f"Request timeout for {endpoint_class.__name__} "
                    f"(attempt {attempt + 1})"
                )
                if attempt < self.max_retries:
                    backoff = 2**attempt
                    logger.debug(f"Retrying in {backoff}s...")
                    time.sleep(backoff)
                    continue

            except RequestException as e:
                last_error = e
                last_was_rate_limit = False
                # Check for specific status codes
                status_code = getattr(e.response, "status_code", None) if hasattr(e, "response") else None

                if status_code == 404:
                    logger.error(f"Resource not found: {endpoint_class.__name__}")
                    raise NBAApiNotFoundError(
                        f"Resource not found: {endpoint_class.__name__}"
                    ) from e

                if status_code in NON_RETRIABLE_STATUS_CODES:
                    logger.error(
                        f"Non-retriable error {status_code} for {endpoint_class.__name__}"
                    )
                    raise NBAApiError(
                        f"API error {status_code}: {endpoint_class.__name__}"
                    ) from e

                if status_code == 429:
                    last_was_rate_limit = True
                    logger.warning(
                        f"Rate limit hit for {endpoint_class.__name__} "
                        f"(attempt {attempt + 1})"
                    )
                    if attempt < self.max_retries:
                        backoff = 2 ** (attempt + 1)  # Longer backoff for rate limits
                        logger.debug(f"Retrying in {backoff}s...")
                        time.sleep(backoff)
                        continue

                if status_code in RETRIABLE_STATUS_CODES:
                    logger.warning(
                        f"Retriable error {status_code} for {endpoint_class.__name__} "
                        f"(attempt {attempt + 1})"
                    )
                    if attempt < self.max_retries:
                        backoff = 2**attempt
                        logger.debug(f"Retrying in {backoff}s...")
                        time.sleep(backoff)
                        continue

                # Unknown error
                logger.error(f"Request error for {endpoint_class.__name__}: {e}")
                if attempt < self.max_retries:
                    backoff = 2**attempt
                    time.sleep(backoff)
                    continue

            except Exception as e:
                last_error = e
                last_was_rate_limit = False
                logger.error(f"Unexpected error for {endpoint_class.__name__}: {e}")
                if attempt < self.max_retries:
                    backoff = 2**attempt
                    time.sleep(backoff)
                    continue

        # All retries exhausted
        if isinstance(last_error, ReadTimeout):
            raise NBAApiTimeoutError(
                f"Request timeout after {self.max_retries + 1} attempts"
            ) from last_error

        if last_was_rate_limit:
            raise NBAApiRateLimitError(
                f"Rate limit exceeded after {self.max_retries + 1} attempts"
            ) from last_error

        raise NBAApiError(
            f"API request failed after {self.max_retries + 1} attempts: {last_error}"
        ) from last_error

    # =========================================================================
    # Public API Methods
    # =========================================================================

    def get_league_game_finder(
        self,
        season: str,
        season_type: str = "Regular Season",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Fetch games for a season using LeagueGameFinder.

        Args:
            season: Season string (e.g., "2023-24").
            season_type: "Regular Season" or "Playoffs".
            **kwargs: Additional arguments to pass to the endpoint.

        Returns:
            DataFrame with game records.
        """
        from nba_api.stats.endpoints import LeagueGameFinder

        logger.info(f"Fetching games for season {season} ({season_type})")

        result = self._request_with_retry(
            LeagueGameFinder,
            season_nullable=season,
            season_type_nullable=season_type,
            league_id_nullable="00",  # NBA
            **kwargs,
        )

        df = result.get_data_frames()[0]
        logger.info(f"Retrieved {len(df)} game records")
        return df

    def get_play_by_play(self, game_id: str) -> pd.DataFrame:
        """Fetch play-by-play using PlayByPlayV2 with V3 fallback.

        Tries PlayByPlayV2 first for compatibility, then falls back to
        PlayByPlayV3 for historical games where V2 may not work.
        The V3 response is normalized to V2 column format.

        Args:
            game_id: NBA game ID string.

        Returns:
            DataFrame with play events (normalized to V2 format).
        """
        from nba_api.stats.endpoints import PlayByPlayV2, PlayByPlayV3

        logger.debug(f"Fetching play-by-play for game {game_id}")

        # Try V2 first
        try:
            result = self._request_with_retry(PlayByPlayV2, game_id=game_id)
            df = result.get_data_frames()[0]
            logger.debug(f"Retrieved {len(df)} play events for game {game_id} (V2)")
            return df
        except (KeyError, NBAApiError) as e:
            # V2 may fail for historical games with KeyError: 'resultSet'
            # or other API structure issues - fall back to V3
            logger.warning(
                f"PlayByPlayV2 failed for game {game_id}: {e}. "
                f"Falling back to PlayByPlayV3."
            )

        # Fall back to V3
        try:
            result = self._request_with_retry(PlayByPlayV3, game_id=game_id)
            df = result.get_data_frames()[0]
            logger.debug(f"Retrieved {len(df)} play events for game {game_id} (V3)")

            # Normalize V3 columns to V2 format
            df = self._normalize_pbp_v3_to_v2(df, game_id)
            return df
        except Exception as e:
            logger.error(f"PlayByPlayV3 also failed for game {game_id}: {e}")
            # Return empty DataFrame instead of raising
            return pd.DataFrame()

    def _normalize_pbp_v3_to_v2(self, df: pd.DataFrame, game_id: str) -> pd.DataFrame:
        """Normalize PlayByPlayV3 response to V2 column format.

        V3 uses camelCase columns; V2 uses UPPERCASE columns.
        This ensures downstream collectors work with both versions.

        Args:
            df: PlayByPlayV3 DataFrame.
            game_id: Game ID for context.

        Returns:
            DataFrame with V2-compatible column names.
        """
        if df.empty:
            return df

        # Map V3 action types to V2 event message types
        action_type_map = {
            "period": 12,  # PERIOD_START / PERIOD_END
            "Jump Ball": 10,
            "2pt": 1,  # Field goal (use shotResult to determine made/missed)
            "3pt": 1,
            "Free Throw": 3,
            "Rebound": 4,
            "Turnover": 5,
            "Foul": 6,
            "Violation": 7,
            "Substitution": 8,
            "Timeout": 9,
            "Ejection": 11,
            "Instant Replay": 18,
            "Stoppage": 20,
        }

        result_df = pd.DataFrame()

        # Map columns
        result_df["GAME_ID"] = df.get("gameId", game_id)
        result_df["EVENTNUM"] = df.get("actionNumber", range(len(df)))
        result_df["PERIOD"] = df.get("period", 1)

        # Parse clock from "PT12M00.00S" to "12:00" format
        if "clock" in df.columns:
            result_df["PCTIMESTRING"] = df["clock"].apply(self._parse_v3_clock)
        else:
            result_df["PCTIMESTRING"] = None

        result_df["WCTIMESTRING"] = None  # V3 doesn't have wall clock time

        # Map action type to event message type
        def map_event_type(row: pd.Series) -> int:
            action_type = row.get("actionType", "")
            shot_result = row.get("shotResult", "")

            # Check if it's a shot
            if action_type in ("2pt", "3pt"):
                if shot_result == "Made":
                    return 1  # FIELD_GOAL_MADE
                else:
                    return 2  # FIELD_GOAL_MISSED

            # Check for period start/end
            if action_type == "period":
                sub_type = row.get("subType", "")
                if sub_type == "end":
                    return 13  # PERIOD_END
                return 12  # PERIOD_START

            return action_type_map.get(action_type, 0)

        result_df["EVENTMSGTYPE"] = df.apply(map_event_type, axis=1)
        result_df["EVENTMSGACTIONTYPE"] = 0  # Default action type

        # Map descriptions based on location (h=home, v=visitor, empty=neutral)
        def get_home_desc(row: pd.Series) -> str | None:
            if row.get("location") == "h":
                return row.get("description")
            return None

        def get_away_desc(row: pd.Series) -> str | None:
            if row.get("location") == "v":
                return row.get("description")
            return None

        def get_neutral_desc(row: pd.Series) -> str | None:
            if not row.get("location") or row.get("location") == "":
                return row.get("description")
            return None

        result_df["HOMEDESCRIPTION"] = df.apply(get_home_desc, axis=1)
        result_df["VISITORDESCRIPTION"] = df.apply(get_away_desc, axis=1)
        result_df["NEUTRALDESCRIPTION"] = df.apply(get_neutral_desc, axis=1)

        # Format score as "AWAY - HOME"
        def format_score(row: pd.Series) -> str | None:
            score_home = row.get("scoreHome")
            score_away = row.get("scoreAway")
            if score_home and score_away:
                return f"{score_away} - {score_home}"
            return None

        result_df["SCORE"] = df.apply(format_score, axis=1)

        # Map player IDs
        result_df["PLAYER1_ID"] = df.get("personId", 0)
        result_df["PLAYER2_ID"] = 0  # V3 doesn't have secondary players in same row
        result_df["PLAYER3_ID"] = 0
        result_df["PLAYER1_TEAM_ID"] = df.get("teamId", 0)

        logger.debug(f"Normalized {len(result_df)} V3 plays to V2 format")
        return result_df

    def _parse_v3_clock(self, clock_str: str | None) -> str | None:
        """Parse V3 clock format (PT12M00.00S) to V2 format (12:00).

        Args:
            clock_str: Clock string in ISO duration format.

        Returns:
            Time string in "MM:SS" format or None.
        """
        if not clock_str or pd.isna(clock_str):
            return None

        try:
            # Parse "PT12M00.00S" format
            import re

            match = re.match(r"PT(\d+)M([\d.]+)S", str(clock_str))
            if match:
                minutes = int(match.group(1))
                seconds = int(float(match.group(2)))
                return f"{minutes}:{seconds:02d}"
        except (ValueError, AttributeError):
            pass

        return str(clock_str)

    def get_shot_chart(
        self,
        game_id: str | None = None,
        player_id: int | None = None,
        season: str | None = None,
        team_id: int = 0,
        context_measure: str = "FGA",
    ) -> pd.DataFrame:
        """Fetch shot chart using ShotChartDetail.

        Args:
            game_id: Specific game (optional).
            player_id: Specific player (optional, 0 for all).
            season: Season string (optional).
            team_id: Team ID filter (0 for all).
            context_measure: Stat to filter by (default "FGA").

        Returns:
            DataFrame with shot records.
        """
        from nba_api.stats.endpoints import ShotChartDetail

        logger.debug(
            f"Fetching shot chart: game={game_id}, player={player_id}, season={season}"
        )

        # Build params - ShotChartDetail requires specific params
        params: dict[str, Any] = {
            "team_id": team_id,
            "player_id": player_id if player_id else 0,
            "context_measure_simple": context_measure,
        }

        if game_id:
            params["game_id_nullable"] = game_id
        if season:
            params["season_nullable"] = season

        result = self._request_with_retry(ShotChartDetail, **params)

        df = result.get_data_frames()[0]
        logger.debug(f"Retrieved {len(df)} shot records")
        return df

    def get_boxscore_advanced(
        self, game_id: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch advanced boxscore using BoxScoreAdvancedV2.

        Args:
            game_id: NBA game ID string.

        Returns:
            Tuple of (team_stats_df, player_stats_df).
        """
        from nba_api.stats.endpoints import BoxScoreAdvancedV2

        logger.debug(f"Fetching advanced boxscore for game {game_id}")

        result = self._request_with_retry(BoxScoreAdvancedV2, game_id=game_id)

        frames = result.get_data_frames()
        # Frame order: PlayerStats, TeamStats
        player_df = frames[0] if len(frames) > 0 else pd.DataFrame()
        team_df = frames[1] if len(frames) > 1 else pd.DataFrame()

        logger.debug(
            f"Retrieved advanced stats: {len(team_df)} team rows, "
            f"{len(player_df)} player rows"
        )
        return team_df, player_df

    def get_boxscore_traditional(
        self, game_id: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch traditional boxscore using BoxScoreTraditionalV2.

        Args:
            game_id: NBA game ID string.

        Returns:
            Tuple of (team_stats_df, player_stats_df).
        """
        from nba_api.stats.endpoints import BoxScoreTraditionalV2

        logger.debug(f"Fetching traditional boxscore for game {game_id}")

        result = self._request_with_retry(BoxScoreTraditionalV2, game_id=game_id)

        frames = result.get_data_frames()
        # Frame order: PlayerStats, TeamStats
        player_df = frames[0] if len(frames) > 0 else pd.DataFrame()
        team_df = frames[1] if len(frames) > 1 else pd.DataFrame()

        logger.debug(
            f"Retrieved traditional stats: {len(team_df)} team rows, "
            f"{len(player_df)} player rows"
        )
        return team_df, player_df

    def get_player_tracking(self, game_id: str) -> pd.DataFrame:
        """Fetch player tracking using BoxScorePlayerTrackV3.

        Note: Player tracking data is only available from ~2013-14 season.
        Uses V3 endpoint as V2 is not available in the nba_api library.

        Args:
            game_id: NBA game ID string.

        Returns:
            DataFrame with tracking metrics (distance, speed).
        """
        from nba_api.stats.endpoints import BoxScorePlayerTrackV3

        logger.debug(f"Fetching player tracking for game {game_id}")

        result = self._request_with_retry(BoxScorePlayerTrackV3, game_id=game_id)

        df = result.get_data_frames()[0]
        logger.debug(f"Retrieved tracking data for {len(df)} players")
        return df

    def get_team_roster(self, team_id: int, season: str) -> pd.DataFrame:
        """Fetch team roster using CommonTeamRoster.

        Args:
            team_id: NBA team ID.
            season: Season string (e.g., "2023-24").

        Returns:
            DataFrame with player roster info.
        """
        from nba_api.stats.endpoints import CommonTeamRoster

        logger.debug(f"Fetching roster for team {team_id}, season {season}")

        result = self._request_with_retry(
            CommonTeamRoster, team_id=team_id, season=season
        )

        df = result.get_data_frames()[0]
        logger.debug(f"Retrieved {len(df)} players for team {team_id}")
        return df

    def get_player_info(self, player_id: int) -> pd.DataFrame:
        """Fetch player biographical info using CommonPlayerInfo.

        Args:
            player_id: NBA player ID.

        Returns:
            DataFrame with player details.
        """
        from nba_api.stats.endpoints import CommonPlayerInfo

        logger.debug(f"Fetching info for player {player_id}")

        result = self._request_with_retry(CommonPlayerInfo, player_id=player_id)

        df = result.get_data_frames()[0]
        logger.debug(f"Retrieved info for player {player_id}")
        return df
