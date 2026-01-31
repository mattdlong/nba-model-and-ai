"""Tests for ShotsCollector.

Tests shot collection and transformation with mocked API responses.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from nba_model.data.collectors.shots import ShotsCollector
from nba_model.data.models import Shot


@pytest.fixture
def mock_api_client() -> MagicMock:
    """Create a mock NBA API client."""
    return MagicMock()


@pytest.fixture
def shots_collector(mock_api_client: MagicMock) -> ShotsCollector:
    """Create a ShotsCollector with mocked API client."""
    return ShotsCollector(api_client=mock_api_client)


@pytest.fixture
def sample_shots_df() -> pd.DataFrame:
    """Create sample ShotChartDetail response data."""
    return pd.DataFrame({
        "GAME_ID": ["0022300001"] * 4,
        "PLAYER_ID": [1628369, 1628369, 2544, 2544],
        "PLAYER_NAME": ["Jayson Tatum", "Jayson Tatum", "LeBron James", "LeBron James"],
        "TEAM_ID": [1610612738, 1610612738, 1610612747, 1610612747],
        "PERIOD": [1, 1, 1, 2],
        "MINUTES_REMAINING": [11, 9, 8, 6],
        "SECONDS_REMAINING": [45, 30, 15, 0],
        "ACTION_TYPE": ["Jump Shot", "Driving Layup", "Fadeaway", "Dunk"],
        "SHOT_TYPE": ["3PT Field Goal", "2PT Field Goal", "2PT Field Goal", "2PT Field Goal"],
        "SHOT_ZONE_BASIC": ["Above the Break 3", "Restricted Area", "Mid-Range", "Restricted Area"],
        "SHOT_ZONE_AREA": ["Center(C)", "Center(C)", "Right Side(R)", "Center(C)"],
        "SHOT_ZONE_RANGE": ["24+ ft.", "Less Than 8 ft.", "16-24 ft.", "Less Than 8 ft."],
        "SHOT_DISTANCE": [25, 2, 18, 0],
        "LOC_X": [100, 5, 150, 0],
        "LOC_Y": [200, 10, 120, 5],
        "SHOT_MADE_FLAG": [1, 0, 1, 1],
    })


class TestShotsCollector:
    """Tests for ShotsCollector."""

    def test_collect_game(
        self,
        shots_collector: ShotsCollector,
        mock_api_client: MagicMock,
        sample_shots_df: pd.DataFrame,
    ) -> None:
        """Should collect and transform shots for a game."""
        mock_api_client.get_shot_chart.return_value = sample_shots_df

        shots = shots_collector.collect_game("0022300001")

        assert len(shots) == 4
        assert all(isinstance(s, Shot) for s in shots)

    def test_collect_game_empty(
        self,
        shots_collector: ShotsCollector,
        mock_api_client: MagicMock,
    ) -> None:
        """Should handle empty response."""
        mock_api_client.get_shot_chart.return_value = pd.DataFrame()

        shots = shots_collector.collect_game("0022300001")

        assert len(shots) == 0

    def test_collect_player_season(
        self,
        shots_collector: ShotsCollector,
        mock_api_client: MagicMock,
        sample_shots_df: pd.DataFrame,
    ) -> None:
        """Should collect shots for a player's season."""
        mock_api_client.get_shot_chart.return_value = sample_shots_df

        shots = shots_collector.collect_player_season(1628369, "2023-24")

        assert len(shots) == 4  # All shots in sample
        mock_api_client.get_shot_chart.assert_called_once()

    def test_collect_games_multiple(
        self,
        shots_collector: ShotsCollector,
        mock_api_client: MagicMock,
        sample_shots_df: pd.DataFrame,
    ) -> None:
        """Should collect shots for multiple games."""
        mock_api_client.get_shot_chart.return_value = sample_shots_df

        results = shots_collector.collect_games(["001", "002"])

        assert len(results) == 2
        assert "001" in results
        assert "002" in results


class TestTransformShot:
    """Tests for shot transformation logic."""

    def test_transform_shot_coordinates(
        self,
        shots_collector: ShotsCollector,
    ) -> None:
        """Should preserve court coordinates."""
        row = pd.Series({
            "GAME_ID": "0022300001",
            "PLAYER_ID": 1628369,
            "TEAM_ID": 1610612738,
            "PERIOD": 1,
            "MINUTES_REMAINING": 10,
            "SECONDS_REMAINING": 30,
            "ACTION_TYPE": "Jump Shot",
            "SHOT_TYPE": "3PT Field Goal",
            "SHOT_ZONE_BASIC": "Above the Break 3",
            "SHOT_ZONE_AREA": "Center(C)",
            "SHOT_ZONE_RANGE": "24+ ft.",
            "SHOT_DISTANCE": 25,
            "LOC_X": 150,
            "LOC_Y": 200,
            "SHOT_MADE_FLAG": 1,
        })

        shot = shots_collector._transform_shot(row)

        assert shot is not None
        assert shot.loc_x == 150
        assert shot.loc_y == 200

    def test_transform_shot_made_flag(
        self,
        shots_collector: ShotsCollector,
    ) -> None:
        """Should convert made flag to boolean."""
        # Made shot
        row_made = pd.Series({
            "GAME_ID": "001",
            "PLAYER_ID": 1,
            "TEAM_ID": 100,
            "PERIOD": 1,
            "MINUTES_REMAINING": 10,
            "SECONDS_REMAINING": 0,
            "ACTION_TYPE": "Shot",
            "SHOT_TYPE": "2PT Field Goal",
            "SHOT_ZONE_BASIC": "Zone",
            "SHOT_ZONE_AREA": "Area",
            "SHOT_ZONE_RANGE": "Range",
            "SHOT_DISTANCE": 10,
            "LOC_X": 0,
            "LOC_Y": 50,
            "SHOT_MADE_FLAG": 1,
        })

        shot_made = shots_collector._transform_shot(row_made)
        assert shot_made is not None
        assert shot_made.made is True

        # Missed shot
        row_missed = row_made.copy()
        row_missed["SHOT_MADE_FLAG"] = 0

        shot_missed = shots_collector._transform_shot(row_missed)
        assert shot_missed is not None
        assert shot_missed.made is False

    def test_transform_shot_type_normalization(
        self,
        shots_collector: ShotsCollector,
    ) -> None:
        """Should normalize shot type to 2PT/3PT."""
        row_3pt = pd.Series({
            "GAME_ID": "001",
            "PLAYER_ID": 1,
            "TEAM_ID": 100,
            "PERIOD": 1,
            "MINUTES_REMAINING": 10,
            "SECONDS_REMAINING": 0,
            "ACTION_TYPE": "Shot",
            "SHOT_TYPE": "3PT Field Goal",
            "SHOT_ZONE_BASIC": "Zone",
            "SHOT_ZONE_AREA": "Area",
            "SHOT_ZONE_RANGE": "Range",
            "SHOT_DISTANCE": 25,
            "LOC_X": 200,
            "LOC_Y": 100,
            "SHOT_MADE_FLAG": 1,
        })

        shot = shots_collector._transform_shot(row_3pt)
        assert shot is not None
        assert shot.shot_type == "3PT"

        row_2pt = row_3pt.copy()
        row_2pt["SHOT_TYPE"] = "2PT Field Goal"

        shot2 = shots_collector._transform_shot(row_2pt)
        assert shot2 is not None
        assert shot2.shot_type == "2PT"

    def test_transform_shot_handles_missing_fields(
        self,
        shots_collector: ShotsCollector,
    ) -> None:
        """Should handle missing optional fields."""
        row = pd.Series({
            "GAME_ID": "001",
            "PLAYER_ID": 1,
            "TEAM_ID": 100,
            "PERIOD": 1,
            "MINUTES_REMAINING": 10,
            "SECONDS_REMAINING": 0,
            "ACTION_TYPE": None,  # Missing
            "SHOT_TYPE": None,  # Missing
            "SHOT_ZONE_BASIC": None,
            "SHOT_ZONE_AREA": None,
            "SHOT_ZONE_RANGE": None,
            "SHOT_DISTANCE": None,  # Missing
            "LOC_X": 0,
            "LOC_Y": 0,
            "SHOT_MADE_FLAG": 0,
        })

        shot = shots_collector._transform_shot(row)

        assert shot is not None
        assert shot.action_type is None
        assert shot.shot_distance is None


class TestCoordinateRanges:
    """Tests for coordinate handling."""

    def test_negative_x_coordinate(
        self,
        shots_collector: ShotsCollector,
    ) -> None:
        """Should handle negative X coordinates (left side of court)."""
        row = pd.Series({
            "GAME_ID": "001",
            "PLAYER_ID": 1,
            "TEAM_ID": 100,
            "PERIOD": 1,
            "MINUTES_REMAINING": 5,
            "SECONDS_REMAINING": 0,
            "ACTION_TYPE": "Shot",
            "SHOT_TYPE": "2PT",
            "SHOT_ZONE_BASIC": "Zone",
            "SHOT_ZONE_AREA": "Left Side(L)",
            "SHOT_ZONE_RANGE": "Range",
            "SHOT_DISTANCE": 15,
            "LOC_X": -150,  # Left side
            "LOC_Y": 100,
            "SHOT_MADE_FLAG": 1,
        })

        shot = shots_collector._transform_shot(row)

        assert shot is not None
        assert shot.loc_x == -150


class TestShotsCollectorErrors:
    """Tests for error handling in ShotsCollector."""

    def test_collect_game_api_error(
        self,
        shots_collector: ShotsCollector,
        mock_api_client: MagicMock,
    ) -> None:
        """Should propagate API errors."""
        mock_api_client.get_shot_chart.side_effect = Exception("API error")

        with pytest.raises(Exception, match="API error"):
            shots_collector.collect_game("0022300001")

    def test_collect_player_season_empty(
        self,
        shots_collector: ShotsCollector,
        mock_api_client: MagicMock,
    ) -> None:
        """Should handle empty player season data."""
        mock_api_client.get_shot_chart.return_value = pd.DataFrame()

        shots = shots_collector.collect_player_season(1628369, "2023-24")

        assert len(shots) == 0

    def test_collect_player_season_api_error(
        self,
        shots_collector: ShotsCollector,
        mock_api_client: MagicMock,
    ) -> None:
        """Should propagate API errors for player season."""
        mock_api_client.get_shot_chart.side_effect = Exception("API error")

        with pytest.raises(Exception, match="API error"):
            shots_collector.collect_player_season(1628369, "2023-24")

    def test_collect_games_with_error_raise(
        self,
        shots_collector: ShotsCollector,
        mock_api_client: MagicMock,
    ) -> None:
        """Should raise on error with raise strategy."""
        mock_api_client.get_shot_chart.side_effect = Exception("API error")

        with pytest.raises(Exception, match="API error"):
            shots_collector.collect_games(["001"], on_error="raise")

    def test_collect_games_with_error_skip(
        self,
        shots_collector: ShotsCollector,
        mock_api_client: MagicMock,
        sample_shots_df: pd.DataFrame,
    ) -> None:
        """Should skip errors with skip strategy."""
        mock_api_client.get_shot_chart.side_effect = [
            Exception("API error"),
            sample_shots_df,
        ]

        results = shots_collector.collect_games(["001", "002"], on_error="skip")

        assert len(results) == 1
        assert "002" in results

    def test_transform_shot_missing_game_id(
        self,
        shots_collector: ShotsCollector,
    ) -> None:
        """Should return None for missing game_id."""
        row = pd.Series({
            "GAME_ID": "",  # Empty
            "PLAYER_ID": 1,
            "TEAM_ID": 100,
            "PERIOD": 1,
            "MINUTES_REMAINING": 5,
            "SECONDS_REMAINING": 0,
            "ACTION_TYPE": "Shot",
            "SHOT_TYPE": "2PT",
            "SHOT_ZONE_BASIC": "Zone",
            "SHOT_ZONE_AREA": "Area",
            "SHOT_ZONE_RANGE": "Range",
            "SHOT_DISTANCE": 15,
            "LOC_X": 0,
            "LOC_Y": 50,
            "SHOT_MADE_FLAG": 1,
        })

        shot = shots_collector._transform_shot(row)

        assert shot is None


class TestSafeIntMethod:
    """Tests for _safe_int helper method."""

    def test_safe_int_with_valid_int(
        self,
        shots_collector: ShotsCollector,
    ) -> None:
        """Should convert valid int."""
        assert shots_collector._safe_int(42) == 42

    def test_safe_int_with_float(
        self,
        shots_collector: ShotsCollector,
    ) -> None:
        """Should convert float to int."""
        assert shots_collector._safe_int(42.5) == 42

    def test_safe_int_with_invalid_string(
        self,
        shots_collector: ShotsCollector,
    ) -> None:
        """Should return None for invalid string."""
        assert shots_collector._safe_int("invalid") is None
