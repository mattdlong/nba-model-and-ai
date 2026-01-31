"""Tests for PlayByPlayCollector.

Tests play-by-play collection and transformation with mocked API responses.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from nba_model.data.collectors.playbyplay import (
    EVENT_TYPES,
    PlayByPlayCollector,
)
from nba_model.data.models import Play


@pytest.fixture
def mock_api_client() -> MagicMock:
    """Create a mock NBA API client."""
    return MagicMock()


@pytest.fixture
def pbp_collector(mock_api_client: MagicMock) -> PlayByPlayCollector:
    """Create a PlayByPlayCollector with mocked API client."""
    return PlayByPlayCollector(api_client=mock_api_client)


@pytest.fixture
def sample_pbp_df() -> pd.DataFrame:
    """Create sample PlayByPlayV2 response data."""
    return pd.DataFrame({
        "GAME_ID": ["0022300001"] * 5,
        "EVENTNUM": [1, 2, 3, 4, 5],
        "EVENTMSGTYPE": [12, 10, 1, 2, 4],
        "EVENTMSGACTIONTYPE": [0, 1, 1, 1, 0],
        "PERIOD": [1, 1, 1, 1, 1],
        "PCTIMESTRING": ["12:00", "12:00", "11:45", "11:30", "11:28"],
        "WCTIMESTRING": ["7:00 PM", "7:00 PM", "7:01 PM", "7:02 PM", "7:02 PM"],
        "HOMEDESCRIPTION": [
            None, "Jump Ball", "Tatum 3PT", None, "Brown REBOUND"
        ],
        "VISITORDESCRIPTION": [None, None, None, "James MISS", None],
        "NEUTRALDESCRIPTION": ["Period Start", None, None, None, None],
        "SCORE": [None, None, "3 - 0", "3 - 0", "3 - 0"],
        "PLAYER1_ID": [0, 1628369, 1628369, 2544, 1627759],
        "PLAYER1_TEAM_ID": [0, 1610612738, 1610612738, 1610612747, 1610612738],
        "PLAYER2_ID": [0, 2544, 0, 0, 0],
        "PLAYER3_ID": [0, 0, 0, 0, 0],
    })


class TestPlayByPlayCollector:
    """Tests for PlayByPlayCollector."""

    def test_collect_game(
        self,
        pbp_collector: PlayByPlayCollector,
        mock_api_client: MagicMock,
        sample_pbp_df: pd.DataFrame,
    ) -> None:
        """Should collect and transform plays for a game."""
        mock_api_client.get_play_by_play.return_value = sample_pbp_df

        plays = pbp_collector.collect_game("0022300001")

        assert len(plays) == 5
        assert all(isinstance(p, Play) for p in plays)

    def test_collect_game_empty(
        self,
        pbp_collector: PlayByPlayCollector,
        mock_api_client: MagicMock,
    ) -> None:
        """Should handle empty response."""
        mock_api_client.get_play_by_play.return_value = pd.DataFrame()

        plays = pbp_collector.collect_game("0022300001")

        assert len(plays) == 0

    def test_collect_games_multiple(
        self,
        pbp_collector: PlayByPlayCollector,
        mock_api_client: MagicMock,
        sample_pbp_df: pd.DataFrame,
    ) -> None:
        """Should collect plays for multiple games."""
        mock_api_client.get_play_by_play.return_value = sample_pbp_df

        results = pbp_collector.collect_games(["001", "002"])

        assert len(results) == 2
        assert "001" in results
        assert "002" in results

    def test_collect_games_error_handling(
        self,
        pbp_collector: PlayByPlayCollector,
        mock_api_client: MagicMock,
        sample_pbp_df: pd.DataFrame,
    ) -> None:
        """Should handle errors based on strategy."""
        # First call succeeds, second fails
        mock_api_client.get_play_by_play.side_effect = [
            sample_pbp_df,
            Exception("API error"),
        ]

        # With on_error="log", should continue
        results = pbp_collector.collect_games(["001", "002"], on_error="log")

        assert len(results) == 1
        assert "001" in results


class TestTransformPlay:
    """Tests for play transformation logic."""

    def test_transform_play_event_types(
        self,
        pbp_collector: PlayByPlayCollector,
    ) -> None:
        """Should preserve event type codes."""
        row = pd.Series({
            "EVENTNUM": 1,
            "EVENTMSGTYPE": 1,  # Field goal made
            "EVENTMSGACTIONTYPE": 1,
            "PERIOD": 1,
            "PCTIMESTRING": "11:45",
            "WCTIMESTRING": "7:01 PM",
            "HOMEDESCRIPTION": "Made shot",
            "VISITORDESCRIPTION": None,
            "NEUTRALDESCRIPTION": None,
            "SCORE": "2 - 0",
            "PLAYER1_ID": 12345,
            "PLAYER1_TEAM_ID": 100,
            "PLAYER2_ID": 0,
            "PLAYER3_ID": 0,
        })

        play = pbp_collector._transform_play(row, "0022300001")

        assert play is not None
        assert play.event_type == 1
        assert play.event_num == 1
        assert play.period == 1

    def test_transform_play_extracts_players(
        self,
        pbp_collector: PlayByPlayCollector,
    ) -> None:
        """Should extract player IDs correctly."""
        row = pd.Series({
            "EVENTNUM": 1,
            "EVENTMSGTYPE": 10,  # Jump ball
            "EVENTMSGACTIONTYPE": 1,
            "PERIOD": 1,
            "PCTIMESTRING": "12:00",
            "WCTIMESTRING": "7:00 PM",
            "HOMEDESCRIPTION": "Jump ball",
            "VISITORDESCRIPTION": None,
            "NEUTRALDESCRIPTION": None,
            "SCORE": None,
            "PLAYER1_ID": 100,
            "PLAYER1_TEAM_ID": 1000,
            "PLAYER2_ID": 200,
            "PLAYER3_ID": 300,
        })

        play = pbp_collector._transform_play(row, "0022300001")

        assert play is not None
        assert play.player1_id == 100
        assert play.player2_id == 200
        assert play.player3_id == 300

    def test_transform_play_parses_score(
        self,
        pbp_collector: PlayByPlayCollector,
    ) -> None:
        """Should parse score from string format."""
        row = pd.Series({
            "EVENTNUM": 1,
            "EVENTMSGTYPE": 1,
            "EVENTMSGACTIONTYPE": 1,
            "PERIOD": 2,
            "PCTIMESTRING": "5:00",
            "WCTIMESTRING": "8:00 PM",
            "HOMEDESCRIPTION": "Shot made",
            "VISITORDESCRIPTION": None,
            "NEUTRALDESCRIPTION": None,
            "SCORE": "50 - 55",  # Away - Home format
            "PLAYER1_ID": 100,
            "PLAYER1_TEAM_ID": 1000,
            "PLAYER2_ID": 0,
            "PLAYER3_ID": 0,
        })

        play = pbp_collector._transform_play(row, "0022300001")

        assert play is not None
        assert play.score_home == 55
        assert play.score_away == 50

    def test_transform_play_handles_missing_player(
        self,
        pbp_collector: PlayByPlayCollector,
    ) -> None:
        """Should handle missing/zero player IDs."""
        row = pd.Series({
            "EVENTNUM": 1,
            "EVENTMSGTYPE": 12,  # Period start
            "EVENTMSGACTIONTYPE": 0,
            "PERIOD": 1,
            "PCTIMESTRING": "12:00",
            "WCTIMESTRING": "7:00 PM",
            "HOMEDESCRIPTION": None,
            "VISITORDESCRIPTION": None,
            "NEUTRALDESCRIPTION": "Period Start",
            "SCORE": None,
            "PLAYER1_ID": 0,  # No player
            "PLAYER1_TEAM_ID": 0,
            "PLAYER2_ID": None,
            "PLAYER3_ID": None,
        })

        play = pbp_collector._transform_play(row, "0022300001")

        assert play is not None
        assert play.player1_id is None
        assert play.player2_id is None
        assert play.player3_id is None


class TestEventTypes:
    """Tests for event type constants."""

    def test_event_types_defined(self) -> None:
        """Should have key event types defined."""
        assert EVENT_TYPES[1] == "FIELD_GOAL_MADE"
        assert EVENT_TYPES[2] == "FIELD_GOAL_MISSED"
        assert EVENT_TYPES[12] == "PERIOD_START"
        assert EVENT_TYPES[13] == "PERIOD_END"

    def test_event_types_are_strings(self) -> None:
        """All event type values should be strings."""
        for code, name in EVENT_TYPES.items():
            assert isinstance(code, int)
            assert isinstance(name, str)


class TestPlayByPlayCollectorErrors:
    """Tests for error handling in PlayByPlayCollector."""

    def test_collect_game_api_error(
        self,
        pbp_collector: PlayByPlayCollector,
        mock_api_client: MagicMock,
    ) -> None:
        """Should propagate API errors."""
        mock_api_client.get_play_by_play.side_effect = Exception("API error")

        with pytest.raises(Exception, match="API error"):
            pbp_collector.collect_game("0022300001")

    def test_collect_games_with_raise_strategy(
        self,
        pbp_collector: PlayByPlayCollector,
        mock_api_client: MagicMock,
    ) -> None:
        """Should raise on error with raise strategy."""
        mock_api_client.get_play_by_play.side_effect = Exception("API error")

        with pytest.raises(Exception, match="API error"):
            pbp_collector.collect_games(["001"], on_error="raise")

    def test_collect_games_with_skip_strategy(
        self,
        pbp_collector: PlayByPlayCollector,
        mock_api_client: MagicMock,
        sample_pbp_df: pd.DataFrame,
    ) -> None:
        """Should skip errors with skip strategy."""
        mock_api_client.get_play_by_play.side_effect = [
            Exception("API error"),
            sample_pbp_df,
        ]

        results = pbp_collector.collect_games(["001", "002"], on_error="skip")

        assert len(results) == 1
        assert "002" in results


class TestTransformPlayEdgeCases:
    """Tests for edge cases in play transformation."""

    def test_transform_play_no_score(
        self,
        pbp_collector: PlayByPlayCollector,
    ) -> None:
        """Should handle missing score."""
        row = pd.Series({
            "EVENTNUM": 1,
            "EVENTMSGTYPE": 12,
            "EVENTMSGACTIONTYPE": 0,
            "PERIOD": 1,
            "PCTIMESTRING": "12:00",
            "WCTIMESTRING": "7:00 PM",
            "HOMEDESCRIPTION": None,
            "VISITORDESCRIPTION": None,
            "NEUTRALDESCRIPTION": "Period Start",
            "SCORE": None,  # No score yet
            "PLAYER1_ID": 0,
            "PLAYER1_TEAM_ID": 0,
            "PLAYER2_ID": 0,
            "PLAYER3_ID": 0,
        })

        play = pbp_collector._transform_play(row, "0022300001")

        assert play is not None
        assert play.score_home is None
        assert play.score_away is None
