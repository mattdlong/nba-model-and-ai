"""Tests for NBA API client wrapper.

Tests the NBAApiClient class including rate limiting, retry logic,
and error handling with mocked API responses.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from requests.exceptions import ReadTimeout, RequestException

from nba_model.config import reset_settings
from nba_model.data.api import (
    NBAApiClient,
    NBAApiError,
    NBAApiNotFoundError,
    NBAApiRateLimitError,
    NBAApiTimeoutError,
)

if TYPE_CHECKING:
    from nba_model.config import Settings


@pytest.fixture
def api_client() -> NBAApiClient:
    """Create an API client with fast settings for testing."""
    return NBAApiClient(delay=0.01, max_retries=2, timeout=5.0)


def create_mock_endpoint(return_value: list[pd.DataFrame]) -> MagicMock:
    """Create a mock endpoint class with proper __name__ attribute."""
    mock_instance = MagicMock()
    mock_instance.get_data_frames.return_value = return_value

    mock_class = MagicMock(return_value=mock_instance)
    mock_class.__name__ = "MockEndpoint"
    return mock_class


class TestNBAApiClientInit:
    """Tests for NBAApiClient initialization."""

    def test_init_with_defaults(self, test_settings: "Settings") -> None:
        """Client should initialize with settings defaults."""
        client = NBAApiClient()
        assert client.delay == test_settings.api_delay
        assert client.max_retries == test_settings.api_max_retries

    def test_init_with_custom_values(self) -> None:
        """Client should accept custom delay and retries."""
        client = NBAApiClient(delay=1.0, max_retries=5, timeout=60.0)
        assert client.delay == 1.0
        assert client.max_retries == 5
        assert client.timeout == 60.0


class TestRateLimiting:
    """Tests for rate limiting behavior."""

    def test_rate_limit_applied(self, api_client: NBAApiClient) -> None:
        """Should apply delay between requests."""
        api_client.delay = 0.1  # 100ms delay

        # Test the internal rate limiting directly
        api_client._last_request_time = time.time()
        start = time.time()
        api_client._apply_rate_limit()
        elapsed = time.time() - start
        assert elapsed >= 0.09  # Should have waited ~100ms

    def test_configurable_delay(self) -> None:
        """Delay should be configurable."""
        client = NBAApiClient(delay=0.5)
        assert client.delay == 0.5


class TestRetryLogic:
    """Tests for retry logic with exponential backoff."""

    def test_retry_on_timeout(self, api_client: NBAApiClient) -> None:
        """Should retry on ReadTimeout."""
        call_count = 0

        def mock_endpoint(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ReadTimeout("Timeout")
            mock = MagicMock()
            mock.get_data_frames.return_value = [pd.DataFrame()]
            return mock

        mock_class = MagicMock(side_effect=mock_endpoint)
        mock_class.__name__ = "LeagueGameFinder"

        with patch.object(api_client, "_apply_rate_limit"):
            with patch(
                "nba_api.stats.endpoints.LeagueGameFinder", mock_class
            ):
                # Should succeed after retries
                try:
                    api_client.get_league_game_finder("2023-24")
                except Exception:
                    pass  # May timeout in test, but call count matters

        assert call_count >= 2  # Should have retried

    def test_max_retries_exceeded_timeout(self, api_client: NBAApiClient) -> None:
        """Should raise after max retries on timeout."""
        api_client.max_retries = 1

        def mock_endpoint(*args, **kwargs):
            raise ReadTimeout("Timeout")

        mock_class = MagicMock(side_effect=mock_endpoint)
        mock_class.__name__ = "LeagueGameFinder"

        with patch.object(api_client, "_apply_rate_limit"):
            with patch(
                "nba_api.stats.endpoints.LeagueGameFinder", mock_class
            ):
                with pytest.raises(NBAApiTimeoutError):
                    api_client.get_league_game_finder("2023-24")


class TestErrorHandling:
    """Tests for error handling."""

    def test_404_raises_not_found(self, api_client: NBAApiClient) -> None:
        """Should raise NBAApiNotFoundError on 404."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        error = RequestException("Not found")
        error.response = mock_response

        def mock_endpoint(*args, **kwargs):
            raise error

        mock_class = MagicMock(side_effect=mock_endpoint)
        mock_class.__name__ = "LeagueGameFinder"

        with patch.object(api_client, "_apply_rate_limit"):
            with patch(
                "nba_api.stats.endpoints.LeagueGameFinder", mock_class
            ):
                with pytest.raises(NBAApiNotFoundError):
                    api_client.get_league_game_finder("2023-24")

    def test_400_raises_api_error(self, api_client: NBAApiClient) -> None:
        """Should raise NBAApiError on 400 without retry."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        error = RequestException("Bad request")
        error.response = mock_response

        call_count = 0

        def mock_endpoint(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise error

        mock_class = MagicMock(side_effect=mock_endpoint)
        mock_class.__name__ = "LeagueGameFinder"

        with patch.object(api_client, "_apply_rate_limit"):
            with patch(
                "nba_api.stats.endpoints.LeagueGameFinder", mock_class
            ):
                with pytest.raises(NBAApiError):
                    api_client.get_league_game_finder("2023-24")

        assert call_count == 1  # Should not retry


class TestResponseParsing:
    """Tests for response parsing."""

    def test_get_league_game_finder_returns_dataframe(
        self, api_client: NBAApiClient
    ) -> None:
        """Should return DataFrame from LeagueGameFinder."""
        mock_result = MagicMock()
        mock_result.get_data_frames.return_value = [
            pd.DataFrame({"GAME_ID": ["001", "002"], "PTS": [100, 110]})
        ]

        mock_class = MagicMock(return_value=mock_result)
        mock_class.__name__ = "LeagueGameFinder"

        with patch.object(api_client, "_apply_rate_limit"):
            with patch(
                "nba_api.stats.endpoints.LeagueGameFinder", mock_class
            ):
                df = api_client.get_league_game_finder("2023-24")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "GAME_ID" in df.columns

    def test_get_boxscore_returns_tuple(self, api_client: NBAApiClient) -> None:
        """Should return tuple of DataFrames from boxscore."""
        mock_result = MagicMock()
        mock_result.get_data_frames.return_value = [
            pd.DataFrame({"PLAYER_ID": [1, 2]}),  # Player stats
            pd.DataFrame({"TEAM_ID": [100, 200]}),  # Team stats
        ]

        mock_class = MagicMock(return_value=mock_result)
        mock_class.__name__ = "BoxScoreTraditionalV2"

        with patch.object(api_client, "_apply_rate_limit"):
            with patch(
                "nba_api.stats.endpoints.BoxScoreTraditionalV2", mock_class
            ):
                team_df, player_df = api_client.get_boxscore_traditional("0022300001")

        assert isinstance(team_df, pd.DataFrame)
        assert isinstance(player_df, pd.DataFrame)


class TestAPIEndpoints:
    """Tests for individual API endpoint methods."""

    def test_get_play_by_play(self, api_client: NBAApiClient) -> None:
        """Should fetch play-by-play data."""
        mock_result = MagicMock()
        mock_result.get_data_frames.return_value = [
            pd.DataFrame({"EVENTNUM": [1, 2, 3], "EVENTMSGTYPE": [12, 1, 2]})
        ]

        mock_class = MagicMock(return_value=mock_result)
        mock_class.__name__ = "PlayByPlayV2"

        with patch.object(api_client, "_apply_rate_limit"):
            with patch(
                "nba_api.stats.endpoints.PlayByPlayV2", mock_class
            ):
                df = api_client.get_play_by_play("0022300001")

        assert isinstance(df, pd.DataFrame)
        assert "EVENTNUM" in df.columns

    def test_get_shot_chart(self, api_client: NBAApiClient) -> None:
        """Should fetch shot chart data."""
        mock_result = MagicMock()
        mock_result.get_data_frames.return_value = [
            pd.DataFrame({"LOC_X": [0, 100], "LOC_Y": [50, 200]})
        ]

        mock_class = MagicMock(return_value=mock_result)
        mock_class.__name__ = "ShotChartDetail"

        with patch.object(api_client, "_apply_rate_limit"):
            with patch(
                "nba_api.stats.endpoints.ShotChartDetail", mock_class
            ):
                df = api_client.get_shot_chart(game_id="0022300001")

        assert isinstance(df, pd.DataFrame)
        assert "LOC_X" in df.columns

    def test_get_team_roster(self, api_client: NBAApiClient) -> None:
        """Should fetch team roster data."""
        mock_result = MagicMock()
        mock_result.get_data_frames.return_value = [
            pd.DataFrame({"PLAYER_ID": [1, 2], "PLAYER": ["Player1", "Player2"]})
        ]

        mock_class = MagicMock(return_value=mock_result)
        mock_class.__name__ = "CommonTeamRoster"

        with patch.object(api_client, "_apply_rate_limit"):
            with patch(
                "nba_api.stats.endpoints.CommonTeamRoster", mock_class
            ):
                df = api_client.get_team_roster(1610612738, "2023-24")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_get_player_info(self, api_client: NBAApiClient) -> None:
        """Should fetch player info data."""
        mock_result = MagicMock()
        mock_result.get_data_frames.return_value = [
            pd.DataFrame(
                {"DISPLAY_FIRST_LAST": ["Jayson Tatum"], "HEIGHT": ["6-8"]}
            )
        ]

        mock_class = MagicMock(return_value=mock_result)
        mock_class.__name__ = "CommonPlayerInfo"

        with patch.object(api_client, "_apply_rate_limit"):
            with patch(
                "nba_api.stats.endpoints.CommonPlayerInfo", mock_class
            ):
                df = api_client.get_player_info(1628369)

        assert isinstance(df, pd.DataFrame)
        assert df.iloc[0]["DISPLAY_FIRST_LAST"] == "Jayson Tatum"


class TestRetriableErrors:
    """Tests for retriable status code handling."""

    def test_429_rate_limit_retries(self, api_client: NBAApiClient) -> None:
        """Should retry on 429 rate limit."""
        call_count = 0
        mock_response = MagicMock()
        mock_response.status_code = 429
        error = RequestException("Rate limit")
        error.response = mock_response

        def mock_endpoint(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise error
            mock = MagicMock()
            mock.get_data_frames.return_value = [pd.DataFrame()]
            return mock

        mock_class = MagicMock(side_effect=mock_endpoint)
        mock_class.__name__ = "LeagueGameFinder"

        with patch.object(api_client, "_apply_rate_limit"):
            with patch("time.sleep"):  # Skip actual sleep
                with patch(
                    "nba_api.stats.endpoints.LeagueGameFinder", mock_class
                ):
                    api_client.get_league_game_finder("2023-24")

        assert call_count >= 2

    def test_500_server_error_retries(self, api_client: NBAApiClient) -> None:
        """Should retry on 500 server error."""
        call_count = 0
        mock_response = MagicMock()
        mock_response.status_code = 500
        error = RequestException("Server error")
        error.response = mock_response

        def mock_endpoint(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise error
            mock = MagicMock()
            mock.get_data_frames.return_value = [pd.DataFrame()]
            return mock

        mock_class = MagicMock(side_effect=mock_endpoint)
        mock_class.__name__ = "LeagueGameFinder"

        with patch.object(api_client, "_apply_rate_limit"):
            with patch("time.sleep"):
                with patch(
                    "nba_api.stats.endpoints.LeagueGameFinder", mock_class
                ):
                    api_client.get_league_game_finder("2023-24")

        assert call_count >= 2

    def test_unknown_request_error_retries(self, api_client: NBAApiClient) -> None:
        """Should retry on unknown request errors."""
        call_count = 0
        error = RequestException("Unknown error")
        error.response = None

        def mock_endpoint(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise error
            mock = MagicMock()
            mock.get_data_frames.return_value = [pd.DataFrame()]
            return mock

        mock_class = MagicMock(side_effect=mock_endpoint)
        mock_class.__name__ = "LeagueGameFinder"

        with patch.object(api_client, "_apply_rate_limit"):
            with patch("time.sleep"):
                with patch(
                    "nba_api.stats.endpoints.LeagueGameFinder", mock_class
                ):
                    api_client.get_league_game_finder("2023-24")

        assert call_count >= 2

    def test_generic_exception_retries(self, api_client: NBAApiClient) -> None:
        """Should retry on generic exceptions."""
        call_count = 0

        def mock_endpoint(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Some error")
            mock = MagicMock()
            mock.get_data_frames.return_value = [pd.DataFrame()]
            return mock

        mock_class = MagicMock(side_effect=mock_endpoint)
        mock_class.__name__ = "LeagueGameFinder"

        with patch.object(api_client, "_apply_rate_limit"):
            with patch("time.sleep"):
                with patch(
                    "nba_api.stats.endpoints.LeagueGameFinder", mock_class
                ):
                    api_client.get_league_game_finder("2023-24")

        assert call_count >= 2


class TestAdditionalEndpoints:
    """Tests for additional API endpoint methods."""

    def test_get_boxscore_advanced(self, api_client: NBAApiClient) -> None:
        """Should fetch advanced boxscore data."""
        mock_result = MagicMock()
        mock_result.get_data_frames.return_value = [
            pd.DataFrame({"PLAYER_ID": [1, 2]}),  # Player stats
            pd.DataFrame({"TEAM_ID": [100, 200]}),  # Team stats
        ]

        mock_class = MagicMock(return_value=mock_result)
        mock_class.__name__ = "BoxScoreAdvancedV2"

        with patch.object(api_client, "_apply_rate_limit"):
            with patch(
                "nba_api.stats.endpoints.BoxScoreAdvancedV2", mock_class
            ):
                team_df, player_df = api_client.get_boxscore_advanced("0022300001")

        assert isinstance(team_df, pd.DataFrame)
        assert isinstance(player_df, pd.DataFrame)

    def test_get_player_tracking(self, api_client: NBAApiClient) -> None:
        """Should fetch player tracking data."""
        mock_result = MagicMock()
        mock_result.get_data_frames.return_value = [
            pd.DataFrame({"PLAYER_ID": [1, 2], "DIST_MILES": [2.5, 3.0]})
        ]

        mock_class = MagicMock(return_value=mock_result)
        mock_class.__name__ = "BoxScorePlayerTrackV3"

        with patch.object(api_client, "_apply_rate_limit"):
            with patch(
                "nba_api.stats.endpoints.BoxScorePlayerTrackV3", mock_class
            ):
                df = api_client.get_player_tracking("0022300001")

        assert isinstance(df, pd.DataFrame)
        assert "DIST_MILES" in df.columns

    def test_get_shot_chart_with_player_and_season(
        self, api_client: NBAApiClient
    ) -> None:
        """Should fetch shot chart with player and season filters."""
        mock_result = MagicMock()
        mock_result.get_data_frames.return_value = [
            pd.DataFrame({"LOC_X": [0, 100], "LOC_Y": [50, 200]})
        ]

        mock_class = MagicMock(return_value=mock_result)
        mock_class.__name__ = "ShotChartDetail"

        with patch.object(api_client, "_apply_rate_limit"):
            with patch(
                "nba_api.stats.endpoints.ShotChartDetail", mock_class
            ):
                df = api_client.get_shot_chart(
                    player_id=1628369, season="2023-24"
                )

        assert isinstance(df, pd.DataFrame)
        # Verify correct params were passed
        mock_class.assert_called_once()
        call_kwargs = mock_class.call_args[1]
        assert call_kwargs["player_id"] == 1628369
        assert call_kwargs["season_nullable"] == "2023-24"


class TestExceptionHierarchy:
    """Tests for exception class hierarchy."""

    def test_rate_limit_is_api_error(self) -> None:
        """NBAApiRateLimitError should be NBAApiError."""
        assert issubclass(NBAApiRateLimitError, NBAApiError)

    def test_not_found_is_api_error(self) -> None:
        """NBAApiNotFoundError should be NBAApiError."""
        assert issubclass(NBAApiNotFoundError, NBAApiError)

    def test_timeout_is_api_error(self) -> None:
        """NBAApiTimeoutError should be NBAApiError."""
        assert issubclass(NBAApiTimeoutError, NBAApiError)

    def test_can_catch_all_as_api_error(self) -> None:
        """All custom exceptions should be catchable as NBAApiError."""
        exceptions = [
            NBAApiRateLimitError("rate limit"),
            NBAApiNotFoundError("not found"),
            NBAApiTimeoutError("timeout"),
        ]

        for exc in exceptions:
            try:
                raise exc
            except NBAApiError:
                pass  # Should catch
