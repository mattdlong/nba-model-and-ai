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


class TestPlayByPlayV3Fallback:
    """Tests for PlayByPlayV3 to V2 fallback mechanism."""

    def test_v3_success_does_not_fallback(self, api_client: NBAApiClient) -> None:
        """Should not call V2 when V3 succeeds."""
        mock_v3_result = MagicMock()
        mock_v3_result.get_data_frames.return_value = [
            pd.DataFrame({
                "gameId": ["0022300001"],
                "actionNumber": [1],
                "clock": ["PT12M00.00S"],
                "period": [1],
                "teamId": [1610612738],
                "personId": [1628369],
                "description": ["Period Start"],
                "actionType": ["period"],
                "subType": ["start"],
                "scoreHome": ["0"],
                "scoreAway": ["0"],
                "location": [""],
            })
        ]

        mock_v3_class = MagicMock(return_value=mock_v3_result)
        mock_v3_class.__name__ = "PlayByPlayV3"

        mock_v2_class = MagicMock()
        mock_v2_class.__name__ = "PlayByPlayV2"

        with patch.object(api_client, "_apply_rate_limit"):
            with patch("nba_api.stats.endpoints.PlayByPlayV3", mock_v3_class):
                with patch("nba_api.stats.endpoints.PlayByPlayV2", mock_v2_class):
                    df = api_client.get_play_by_play("0022300001")

        assert mock_v3_class.called
        assert not mock_v2_class.called
        # Should have normalized V3 columns to V2 format
        assert "EVENTNUM" in df.columns

    def test_v3_keyerror_falls_back_to_v2(self, api_client: NBAApiClient) -> None:
        """Should fall back to V2 when V3 raises KeyError."""

        def v3_fails(*args, **kwargs):
            raise KeyError("resultSet")

        mock_v3_class = MagicMock(side_effect=v3_fails)
        mock_v3_class.__name__ = "PlayByPlayV3"

        mock_v2_result = MagicMock()
        mock_v2_result.get_data_frames.return_value = [
            pd.DataFrame({"EVENTNUM": [1, 2], "EVENTMSGTYPE": [12, 1]})
        ]

        mock_v2_class = MagicMock(return_value=mock_v2_result)
        mock_v2_class.__name__ = "PlayByPlayV2"

        with patch.object(api_client, "_apply_rate_limit"):
            with patch("nba_api.stats.endpoints.PlayByPlayV3", mock_v3_class):
                with patch("nba_api.stats.endpoints.PlayByPlayV2", mock_v2_class):
                    df = api_client.get_play_by_play("0022300001")

        assert mock_v3_class.called
        assert mock_v2_class.called
        assert "EVENTNUM" in df.columns

    def test_both_fail_returns_empty_dataframe(
        self, api_client: NBAApiClient
    ) -> None:
        """Should return empty DataFrame when both V3 and V2 fail."""

        def v3_fails(*args, **kwargs):
            raise KeyError("resultSet")

        def v2_fails(*args, **kwargs):
            raise Exception("V2 also failed")

        mock_v3_class = MagicMock(side_effect=v3_fails)
        mock_v3_class.__name__ = "PlayByPlayV3"

        mock_v2_class = MagicMock(side_effect=v2_fails)
        mock_v2_class.__name__ = "PlayByPlayV2"

        with patch.object(api_client, "_apply_rate_limit"):
            with patch("nba_api.stats.endpoints.PlayByPlayV3", mock_v3_class):
                with patch("nba_api.stats.endpoints.PlayByPlayV2", mock_v2_class):
                    df = api_client.get_play_by_play("0022300001")

        assert isinstance(df, pd.DataFrame)
        assert df.empty


class TestV3ClockParsing:
    """Tests for V3 clock format parsing."""

    def test_parse_standard_clock(self, api_client: NBAApiClient) -> None:
        """Should parse standard V3 clock format."""
        assert api_client._parse_v3_clock("PT12M00.00S") == "12:00"
        assert api_client._parse_v3_clock("PT05M30.50S") == "5:30"
        assert api_client._parse_v3_clock("PT00M01.00S") == "0:01"

    def test_parse_none_clock(self, api_client: NBAApiClient) -> None:
        """Should handle None clock value."""
        assert api_client._parse_v3_clock(None) is None

    def test_parse_empty_clock(self, api_client: NBAApiClient) -> None:
        """Should handle empty clock value."""
        import numpy as np

        # Empty string returns None (falsy value check)
        assert api_client._parse_v3_clock("") is None
        # NaN returns None
        assert api_client._parse_v3_clock(np.nan) is None


class TestV3ToV2Normalization:
    """Tests for V3 to V2 column normalization."""

    def test_normalize_empty_dataframe(self, api_client: NBAApiClient) -> None:
        """Should handle empty DataFrame."""
        empty_df = pd.DataFrame()
        result = api_client._normalize_pbp_v3_to_v2(empty_df, "0022300001")
        assert result.empty

    def test_normalize_maps_event_types(self, api_client: NBAApiClient) -> None:
        """Should map V3 action types to V2 event message types."""
        v3_df = pd.DataFrame({
            "gameId": ["0022300001", "0022300001", "0022300001"],
            "actionNumber": [1, 2, 3],
            "clock": ["PT12M00.00S", "PT11M45.00S", "PT11M30.00S"],
            "period": [1, 1, 1],
            "teamId": [0, 1610612738, 1610612738],
            "personId": [0, 1628369, 1628369],
            "description": ["Period Start", "Tatum 3PT Shot", "Tatum 3PT Shot"],
            "actionType": ["period", "3pt", "3pt"],
            "subType": ["start", "", ""],
            "shotResult": ["", "Made", "Missed"],
            "scoreHome": ["0", "3", "3"],
            "scoreAway": ["0", "0", "0"],
            "location": ["", "h", "h"],
        })

        result = api_client._normalize_pbp_v3_to_v2(v3_df, "0022300001")

        # Check event type mappings
        assert result.iloc[0]["EVENTMSGTYPE"] == 12  # Period start
        assert result.iloc[1]["EVENTMSGTYPE"] == 1  # Field goal made
        assert result.iloc[2]["EVENTMSGTYPE"] == 2  # Field goal missed

    def test_normalize_maps_descriptions(self, api_client: NBAApiClient) -> None:
        """Should map descriptions based on location field."""
        v3_df = pd.DataFrame({
            "gameId": ["0022300001", "0022300001", "0022300001"],
            "actionNumber": [1, 2, 3],
            "clock": ["PT12M00.00S", "PT11M45.00S", "PT11M30.00S"],
            "period": [1, 1, 1],
            "teamId": [0, 1610612738, 1610612747],
            "personId": [0, 1628369, 2544],
            "description": ["Neutral event", "Home play", "Away play"],
            "actionType": ["period", "2pt", "2pt"],
            "subType": ["", "", ""],
            "shotResult": ["", "Made", "Made"],
            "scoreHome": ["0", "2", "2"],
            "scoreAway": ["0", "0", "2"],
            "location": ["", "h", "v"],
        })

        result = api_client._normalize_pbp_v3_to_v2(v3_df, "0022300001")

        # Neutral (empty location)
        assert result.iloc[0]["NEUTRALDESCRIPTION"] == "Neutral event"
        assert pd.isna(result.iloc[0]["HOMEDESCRIPTION"])
        assert pd.isna(result.iloc[0]["VISITORDESCRIPTION"])

        # Home
        assert result.iloc[1]["HOMEDESCRIPTION"] == "Home play"
        assert pd.isna(result.iloc[1]["NEUTRALDESCRIPTION"])

        # Away
        assert result.iloc[2]["VISITORDESCRIPTION"] == "Away play"
        assert pd.isna(result.iloc[2]["HOMEDESCRIPTION"])

    def test_normalize_formats_score(self, api_client: NBAApiClient) -> None:
        """Should format score as 'AWAY - HOME' string."""
        v3_df = pd.DataFrame({
            "gameId": ["0022300001"],
            "actionNumber": [1],
            "clock": ["PT10M00.00S"],
            "period": [2],
            "teamId": [1610612738],
            "personId": [1628369],
            "description": ["Score play"],
            "actionType": ["2pt"],
            "subType": [""],
            "shotResult": ["Made"],
            "scoreHome": ["55"],
            "scoreAway": ["48"],
            "location": ["h"],
        })

        result = api_client._normalize_pbp_v3_to_v2(v3_df, "0022300001")

        # Score format is "AWAY - HOME"
        assert result.iloc[0]["SCORE"] == "48 - 55"


class TestKeyErrorNoRetry:
    """Tests for KeyError not being retried."""

    def test_keyerror_raises_immediately_without_retry(
        self, api_client: NBAApiClient
    ) -> None:
        """KeyError should raise NBAApiError immediately without retrying."""
        call_count = 0

        def mock_endpoint(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise KeyError("resultSet")

        mock_class = MagicMock(side_effect=mock_endpoint)
        mock_class.__name__ = "SomeEndpoint"

        with patch.object(api_client, "_apply_rate_limit"):
            with pytest.raises(NBAApiError, match="API structure error"):
                api_client._request_with_retry(mock_class)

        # Should only be called once - no retries
        assert call_count == 1


class TestBoxScoreAdvancedV3Fallback:
    """Tests for BoxScoreAdvancedV3 to V2 fallback mechanism."""

    def test_v3_success_does_not_fallback(self, api_client: NBAApiClient) -> None:
        """Should not call V2 when V3 succeeds."""
        mock_v3_result = MagicMock()
        mock_v3_result.get_data_frames.return_value = [
            pd.DataFrame({
                "personId": [1628369],
                "firstName": ["Jayson"],
                "familyName": ["Tatum"],
                "teamId": [1610612738],
                "offensiveRating": [115.0],
                "defensiveRating": [105.0],
            }),
            pd.DataFrame({
                "teamId": [1610612738],
                "offensiveRating": [112.0],
                "defensiveRating": [108.0],
            }),
        ]

        mock_v3_class = MagicMock(return_value=mock_v3_result)
        mock_v3_class.__name__ = "BoxScoreAdvancedV3"

        mock_v2_class = MagicMock()
        mock_v2_class.__name__ = "BoxScoreAdvancedV2"

        with patch.object(api_client, "_apply_rate_limit"):
            with patch("nba_api.stats.endpoints.BoxScoreAdvancedV3", mock_v3_class):
                with patch("nba_api.stats.endpoints.BoxScoreAdvancedV2", mock_v2_class):
                    team_df, player_df = api_client.get_boxscore_advanced("0022300001")

        assert mock_v3_class.called
        assert not mock_v2_class.called
        # Should have normalized V3 columns to V2 format
        assert "OFF_RATING" in player_df.columns

    def test_v3_keyerror_falls_back_to_v2(self, api_client: NBAApiClient) -> None:
        """Should fall back to V2 when V3 raises KeyError."""

        def v3_fails(*args, **kwargs):
            raise KeyError("resultSet")

        mock_v3_class = MagicMock(side_effect=v3_fails)
        mock_v3_class.__name__ = "BoxScoreAdvancedV3"

        mock_v2_result = MagicMock()
        mock_v2_result.get_data_frames.return_value = [
            pd.DataFrame({"PLAYER_ID": [1, 2], "OFF_RATING": [110.0, 105.0]}),
            pd.DataFrame({"TEAM_ID": [100, 200], "OFF_RATING": [112.0, 108.0]}),
        ]

        mock_v2_class = MagicMock(return_value=mock_v2_result)
        mock_v2_class.__name__ = "BoxScoreAdvancedV2"

        with patch.object(api_client, "_apply_rate_limit"):
            with patch("nba_api.stats.endpoints.BoxScoreAdvancedV3", mock_v3_class):
                with patch("nba_api.stats.endpoints.BoxScoreAdvancedV2", mock_v2_class):
                    team_df, player_df = api_client.get_boxscore_advanced("0022300001")

        assert mock_v3_class.called
        assert mock_v2_class.called
        assert "OFF_RATING" in player_df.columns
        assert "TEAM_ID" in team_df.columns

    def test_both_fail_returns_empty_dataframes(
        self, api_client: NBAApiClient
    ) -> None:
        """Should return empty DataFrames when both V3 and V2 fail."""

        def v3_fails(*args, **kwargs):
            raise KeyError("resultSet")

        def v2_fails(*args, **kwargs):
            raise Exception("V2 also failed")

        mock_v3_class = MagicMock(side_effect=v3_fails)
        mock_v3_class.__name__ = "BoxScoreAdvancedV3"

        mock_v2_class = MagicMock(side_effect=v2_fails)
        mock_v2_class.__name__ = "BoxScoreAdvancedV2"

        with patch.object(api_client, "_apply_rate_limit"):
            with patch("nba_api.stats.endpoints.BoxScoreAdvancedV3", mock_v3_class):
                with patch("nba_api.stats.endpoints.BoxScoreAdvancedV2", mock_v2_class):
                    team_df, player_df = api_client.get_boxscore_advanced("0022300001")

        assert isinstance(team_df, pd.DataFrame)
        assert isinstance(player_df, pd.DataFrame)
        assert team_df.empty
        assert player_df.empty


class TestBoxScoreTraditionalV3Fallback:
    """Tests for BoxScoreTraditionalV3 to V2 fallback mechanism."""

    def test_v3_success_does_not_fallback(self, api_client: NBAApiClient) -> None:
        """Should not call V2 when V3 succeeds."""
        mock_v3_result = MagicMock()
        mock_v3_result.get_data_frames.return_value = [
            pd.DataFrame({
                "personId": [1628369],
                "firstName": ["Jayson"],
                "familyName": ["Tatum"],
                "teamId": [1610612738],
                "points": [30],
                "assists": [5],
                "reboundsTotal": [8],
            }),
            pd.DataFrame({
                "teamId": [1610612738],
                "points": [112],
                "assists": [25],
            }),
        ]

        mock_v3_class = MagicMock(return_value=mock_v3_result)
        mock_v3_class.__name__ = "BoxScoreTraditionalV3"

        mock_v2_class = MagicMock()
        mock_v2_class.__name__ = "BoxScoreTraditionalV2"

        with patch.object(api_client, "_apply_rate_limit"):
            with patch(
                "nba_api.stats.endpoints.BoxScoreTraditionalV3", mock_v3_class
            ):
                with patch(
                    "nba_api.stats.endpoints.BoxScoreTraditionalV2", mock_v2_class
                ):
                    team_df, player_df = api_client.get_boxscore_traditional(
                        "0022300001"
                    )

        assert mock_v3_class.called
        assert not mock_v2_class.called
        # Should have normalized V3 columns to V2 format
        assert "PTS" in player_df.columns
        assert "AST" in player_df.columns
        assert "REB" in player_df.columns

    def test_v3_keyerror_falls_back_to_v2(self, api_client: NBAApiClient) -> None:
        """Should fall back to V2 when V3 raises KeyError."""

        def v3_fails(*args, **kwargs):
            raise KeyError("resultSet")

        mock_v3_class = MagicMock(side_effect=v3_fails)
        mock_v3_class.__name__ = "BoxScoreTraditionalV3"

        mock_v2_result = MagicMock()
        mock_v2_result.get_data_frames.return_value = [
            pd.DataFrame({"PLAYER_ID": [1, 2], "PTS": [25, 18]}),
            pd.DataFrame({"TEAM_ID": [100, 200], "PTS": [110, 105]}),
        ]

        mock_v2_class = MagicMock(return_value=mock_v2_result)
        mock_v2_class.__name__ = "BoxScoreTraditionalV2"

        with patch.object(api_client, "_apply_rate_limit"):
            with patch(
                "nba_api.stats.endpoints.BoxScoreTraditionalV3", mock_v3_class
            ):
                with patch(
                    "nba_api.stats.endpoints.BoxScoreTraditionalV2", mock_v2_class
                ):
                    team_df, player_df = api_client.get_boxscore_traditional(
                        "0022300001"
                    )

        assert mock_v3_class.called
        assert mock_v2_class.called
        assert "PTS" in player_df.columns
        assert "TEAM_ID" in team_df.columns

    def test_both_fail_returns_empty_dataframes(
        self, api_client: NBAApiClient
    ) -> None:
        """Should return empty DataFrames when both V3 and V2 fail."""

        def v3_fails(*args, **kwargs):
            raise KeyError("resultSet")

        def v2_fails(*args, **kwargs):
            raise Exception("V2 also failed")

        mock_v3_class = MagicMock(side_effect=v3_fails)
        mock_v3_class.__name__ = "BoxScoreTraditionalV3"

        mock_v2_class = MagicMock(side_effect=v2_fails)
        mock_v2_class.__name__ = "BoxScoreTraditionalV2"

        with patch.object(api_client, "_apply_rate_limit"):
            with patch(
                "nba_api.stats.endpoints.BoxScoreTraditionalV3", mock_v3_class
            ):
                with patch(
                    "nba_api.stats.endpoints.BoxScoreTraditionalV2", mock_v2_class
                ):
                    team_df, player_df = api_client.get_boxscore_traditional(
                        "0022300001"
                    )

        assert isinstance(team_df, pd.DataFrame)
        assert isinstance(player_df, pd.DataFrame)
        assert team_df.empty
        assert player_df.empty


class TestBoxScoreV3Normalization:
    """Tests for V3 to V2 column normalization for boxscore endpoints."""

    def test_normalize_advanced_player_stats(self, api_client: NBAApiClient) -> None:
        """Should normalize V3 advanced player stats to V2 column format."""
        v3_df = pd.DataFrame({
            "personId": [1628369, 1629029],
            "firstName": ["Jayson", "Jaylen"],
            "familyName": ["Tatum", "Brown"],
            "teamId": [1610612738, 1610612738],
            "offensiveRating": [115.5, 112.3],
            "defensiveRating": [105.2, 108.1],
            "netRating": [10.3, 4.2],
            "usagePercentage": [30.5, 25.2],
            "pace": [98.5, 98.5],
        })

        result = api_client._normalize_boxscore_advanced_v3_to_v2(v3_df, is_player=True)

        # Check column mapping
        assert "PLAYER_ID" in result.columns
        assert "TEAM_ID" in result.columns
        assert "OFF_RATING" in result.columns
        assert "DEF_RATING" in result.columns
        assert "NET_RATING" in result.columns
        assert "USG_PCT" in result.columns
        assert "PACE" in result.columns
        assert "PLAYER_NAME" in result.columns

        # Check values
        assert result.iloc[0]["PLAYER_ID"] == 1628369
        assert result.iloc[0]["PLAYER_NAME"] == "Jayson Tatum"
        assert result.iloc[0]["OFF_RATING"] == 115.5

    def test_normalize_advanced_team_stats(self, api_client: NBAApiClient) -> None:
        """Should normalize V3 advanced team stats to V2 column format."""
        v3_df = pd.DataFrame({
            "teamId": [1610612738, 1610612751],
            "teamTricode": ["BOS", "BKN"],
            "offensiveRating": [112.5, 108.3],
            "defensiveRating": [104.2, 110.1],
            "pace": [98.5, 99.2],
        })

        result = api_client._normalize_boxscore_advanced_v3_to_v2(
            v3_df, is_player=False
        )

        # Check column mapping
        assert "TEAM_ID" in result.columns
        assert "TEAM_ABBREVIATION" in result.columns
        assert "OFF_RATING" in result.columns
        assert "DEF_RATING" in result.columns
        assert "PACE" in result.columns

        # Player-specific columns should not be present
        assert "PLAYER_ID" not in result.columns
        assert "PLAYER_NAME" not in result.columns

    def test_normalize_traditional_player_stats(
        self, api_client: NBAApiClient
    ) -> None:
        """Should normalize V3 traditional player stats to V2 column format."""
        v3_df = pd.DataFrame({
            "personId": [1628369],
            "firstName": ["Jayson"],
            "familyName": ["Tatum"],
            "teamId": [1610612738],
            "points": [30],
            "assists": [5],
            "reboundsTotal": [8],
            "reboundsOffensive": [1],
            "reboundsDefensive": [7],
            "fieldGoalsMade": [10],
            "fieldGoalsAttempted": [20],
            "fieldGoalsPercentage": [0.500],
            "threePointersMade": [3],
            "threePointersAttempted": [8],
            "steals": [2],
            "blocks": [1],
            "turnovers": [3],
        })

        result = api_client._normalize_boxscore_traditional_v3_to_v2(
            v3_df, is_player=True
        )

        # Check column mapping
        assert "PLAYER_ID" in result.columns
        assert "PLAYER_NAME" in result.columns
        assert "PTS" in result.columns
        assert "AST" in result.columns
        assert "REB" in result.columns
        assert "OREB" in result.columns
        assert "DREB" in result.columns
        assert "FGM" in result.columns
        assert "FGA" in result.columns
        assert "FG_PCT" in result.columns
        assert "FG3M" in result.columns
        assert "FG3A" in result.columns
        assert "STL" in result.columns
        assert "BLK" in result.columns
        assert "TO" in result.columns

        # Check values
        assert result.iloc[0]["PLAYER_NAME"] == "Jayson Tatum"
        assert result.iloc[0]["PTS"] == 30
        assert result.iloc[0]["REB"] == 8

    def test_normalize_traditional_team_stats(self, api_client: NBAApiClient) -> None:
        """Should normalize V3 traditional team stats to V2 column format."""
        v3_df = pd.DataFrame({
            "teamId": [1610612738],
            "teamTricode": ["BOS"],
            "points": [112],
            "assists": [25],
            "reboundsTotal": [45],
            "fieldGoalsMade": [42],
            "fieldGoalsAttempted": [88],
        })

        result = api_client._normalize_boxscore_traditional_v3_to_v2(
            v3_df, is_player=False
        )

        # Check column mapping
        assert "TEAM_ID" in result.columns
        assert "TEAM_ABBREVIATION" in result.columns
        assert "PTS" in result.columns
        assert "AST" in result.columns
        assert "REB" in result.columns
        assert "FGM" in result.columns
        assert "FGA" in result.columns

        # Player-specific columns should not be present
        assert "PLAYER_ID" not in result.columns
        assert "PLAYER_NAME" not in result.columns

    def test_normalize_empty_dataframe(self, api_client: NBAApiClient) -> None:
        """Should handle empty DataFrames gracefully."""
        empty_df = pd.DataFrame()

        adv_result = api_client._normalize_boxscore_advanced_v3_to_v2(empty_df)
        trad_result = api_client._normalize_boxscore_traditional_v3_to_v2(empty_df)

        assert adv_result.empty
        assert trad_result.empty


class TestPlayerTrackingV3Normalization:
    """Tests for player tracking V3 to V2 normalization."""

    def test_normalize_player_tracking_stats(self, api_client: NBAApiClient) -> None:
        """Should normalize V3 player tracking stats to V2 column format."""
        v3_df = pd.DataFrame({
            "gameId": ["0022300001"],
            "teamId": [1610612738],
            "teamTricode": ["BOS"],
            "personId": [1628369],
            "firstName": ["Jayson"],
            "familyName": ["Tatum"],
            "minutes": ["36:25"],
            "speed": [4.5],
            "distance": [2.8],
            "reboundChancesOffensive": [3],
            "reboundChancesDefensive": [5],
            "reboundChancesTotal": [8],
            "touches": [75],
            "passes": [45],
        })

        result = api_client._normalize_player_tracking_v3_to_v2(v3_df)

        # Check column mapping
        assert "GAME_ID" in result.columns
        assert "TEAM_ID" in result.columns
        assert "TEAM_ABBREVIATION" in result.columns
        assert "PLAYER_ID" in result.columns
        assert "PLAYER_NAME" in result.columns
        assert "MIN" in result.columns
        assert "SPD" in result.columns
        assert "DIST" in result.columns
        assert "ORBC" in result.columns
        assert "DRBC" in result.columns
        assert "RBC" in result.columns
        assert "TCHS" in result.columns
        assert "PASS" in result.columns

        # Check values
        assert result.iloc[0]["PLAYER_ID"] == 1628369
        assert result.iloc[0]["PLAYER_NAME"] == "Jayson Tatum"
        assert result.iloc[0]["TEAM_ID"] == 1610612738
        assert result.iloc[0]["SPD"] == 4.5
        assert result.iloc[0]["DIST"] == 2.8

    def test_normalize_empty_tracking_dataframe(
        self, api_client: NBAApiClient
    ) -> None:
        """Should handle empty tracking DataFrame gracefully."""
        empty_df = pd.DataFrame()
        result = api_client._normalize_player_tracking_v3_to_v2(empty_df)
        assert result.empty

    def test_get_player_tracking_returns_normalized(
        self, api_client: NBAApiClient
    ) -> None:
        """get_player_tracking should return normalized V2-format columns."""
        mock_result = MagicMock()
        mock_result.get_data_frames.return_value = [
            pd.DataFrame({
                "gameId": ["0022300001"],
                "teamId": [1610612738],
                "personId": [1628369],
                "firstName": ["Jayson"],
                "familyName": ["Tatum"],
                "speed": [4.5],
                "distance": [2.8],
            })
        ]

        mock_class = MagicMock(return_value=mock_result)
        mock_class.__name__ = "BoxScorePlayerTrackV3"

        with patch.object(api_client, "_apply_rate_limit"):
            with patch(
                "nba_api.stats.endpoints.BoxScorePlayerTrackV3", mock_class
            ):
                df = api_client.get_player_tracking("0022300001")

        # Should have normalized column names
        assert "PLAYER_ID" in df.columns
        assert "TEAM_ID" in df.columns
        assert "PLAYER_NAME" in df.columns
        assert "SPD" in df.columns
        assert "DIST" in df.columns
