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
        """Fetch play-by-play using PlayByPlayV2.

        Args:
            game_id: NBA game ID string.

        Returns:
            DataFrame with play events.
        """
        from nba_api.stats.endpoints import PlayByPlayV2

        logger.debug(f"Fetching play-by-play for game {game_id}")

        result = self._request_with_retry(PlayByPlayV2, game_id=game_id)

        df = result.get_data_frames()[0]
        logger.debug(f"Retrieved {len(df)} play events for game {game_id}")
        return df

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
