"""Tests for GamesCollector.

Tests game collection and transformation with mocked API responses.
"""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest

from nba_model.data.collectors.games import GamesCollector
from nba_model.data.models import Game


@pytest.fixture
def mock_api_client() -> MagicMock:
    """Create a mock NBA API client."""
    return MagicMock()


@pytest.fixture
def games_collector(mock_api_client: MagicMock) -> GamesCollector:
    """Create a GamesCollector with mocked API client."""
    return GamesCollector(api_client=mock_api_client)


@pytest.fixture
def sample_games_df() -> pd.DataFrame:
    """Create sample LeagueGameFinder response data."""
    return pd.DataFrame({
        "SEASON_ID": ["22023", "22023", "22023", "22023"],
        "TEAM_ID": [1610612738, 1610612747, 1610612744, 1610612748],
        "TEAM_ABBREVIATION": ["BOS", "LAL", "GSW", "MIA"],
        "GAME_ID": ["0022300001", "0022300001", "0022300002", "0022300002"],
        "GAME_DATE": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
        "MATCHUP": ["BOS vs. LAL", "LAL @ BOS", "GSW vs. MIA", "MIA @ GSW"],
        "WL": ["W", "L", "W", "L"],
        "PTS": [115, 108, 120, 115],
        "PLUS_MINUS": [7, -7, 5, -5],
    })


class TestGamesCollector:
    """Tests for GamesCollector."""

    def test_collect_season(
        self,
        games_collector: GamesCollector,
        mock_api_client: MagicMock,
        sample_games_df: pd.DataFrame,
    ) -> None:
        """Should collect and transform games for a season."""
        mock_api_client.get_league_game_finder.return_value = sample_games_df

        games = games_collector.collect_season("2023-24")

        assert len(games) == 2  # Two unique games
        assert all(isinstance(g, Game) for g in games)

        # Verify first game
        game1 = next(g for g in games if g.game_id == "0022300001")
        assert game1.home_team_id == 1610612738  # BOS
        assert game1.away_team_id == 1610612747  # LAL
        assert game1.home_score == 115
        assert game1.away_score == 108
        assert game1.status == "completed"

    def test_collect_season_empty(
        self,
        games_collector: GamesCollector,
        mock_api_client: MagicMock,
    ) -> None:
        """Should handle empty response."""
        mock_api_client.get_league_game_finder.return_value = pd.DataFrame()

        games = games_collector.collect_season("2023-24")

        assert len(games) == 0

    def test_home_away_detection_from_matchup(
        self,
        games_collector: GamesCollector,
        mock_api_client: MagicMock,
    ) -> None:
        """Should correctly identify home/away from matchup string."""
        df = pd.DataFrame({
            "SEASON_ID": ["22023", "22023"],
            "TEAM_ID": [100, 200],
            "GAME_ID": ["001", "001"],
            "GAME_DATE": ["2024-01-01", "2024-01-01"],
            "MATCHUP": ["AAA vs. BBB", "BBB @ AAA"],  # AAA is home
            "WL": ["W", "L"],
            "PTS": [110, 100],
            "PLUS_MINUS": [10, -10],
        })
        mock_api_client.get_league_game_finder.return_value = df

        games = games_collector.collect_season("2023-24")

        assert len(games) == 1
        assert games[0].home_team_id == 100  # AAA
        assert games[0].away_team_id == 200  # BBB

    def test_get_game_ids_for_season(
        self,
        games_collector: GamesCollector,
        mock_api_client: MagicMock,
        sample_games_df: pd.DataFrame,
    ) -> None:
        """Should return unique game IDs."""
        mock_api_client.get_league_game_finder.return_value = sample_games_df

        game_ids = games_collector.get_game_ids_for_season("2023-24")

        assert len(game_ids) == 2
        assert "0022300001" in game_ids
        assert "0022300002" in game_ids

    def test_collect_date_range(
        self,
        games_collector: GamesCollector,
        mock_api_client: MagicMock,
        sample_games_df: pd.DataFrame,
    ) -> None:
        """Should filter games by date range."""
        mock_api_client.get_league_game_finder.return_value = sample_games_df

        games = games_collector.collect_date_range(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 1),
        )

        # Only game from Jan 1
        assert len(games) == 1
        assert games[0].game_id == "0022300001"

    def test_game_status_completed(
        self,
        games_collector: GamesCollector,
        mock_api_client: MagicMock,
    ) -> None:
        """Should set status to completed for finished games."""
        df = pd.DataFrame({
            "SEASON_ID": ["22023", "22023"],
            "TEAM_ID": [100, 200],
            "GAME_ID": ["001", "001"],
            "GAME_DATE": ["2024-01-01", "2024-01-01"],
            "MATCHUP": ["AAA vs. BBB", "BBB @ AAA"],
            "WL": ["W", "L"],  # Has win/loss
            "PTS": [110, 100],
            "PLUS_MINUS": [10, -10],
        })
        mock_api_client.get_league_game_finder.return_value = df

        games = games_collector.collect_season("2023-24")

        assert games[0].status == "completed"

    def test_game_status_scheduled(
        self,
        games_collector: GamesCollector,
        mock_api_client: MagicMock,
    ) -> None:
        """Should set status to scheduled for unfinished games."""
        df = pd.DataFrame({
            "SEASON_ID": ["22023", "22023"],
            "TEAM_ID": [100, 200],
            "GAME_ID": ["001", "001"],
            "GAME_DATE": ["2024-01-01", "2024-01-01"],
            "MATCHUP": ["AAA vs. BBB", "BBB @ AAA"],
            "WL": [None, None],  # No win/loss yet
            "PTS": [None, None],
            "PLUS_MINUS": [0, 0],
        })
        mock_api_client.get_league_game_finder.return_value = df

        games = games_collector.collect_season("2023-24")

        assert games[0].status == "scheduled"


class TestTransformGame:
    """Tests for game transformation logic."""

    def test_transform_game_parses_date(
        self,
        games_collector: GamesCollector,
    ) -> None:
        """Should parse game date correctly."""
        df = pd.DataFrame({
            "TEAM_ID": [100, 200],
            "GAME_ID": ["001", "001"],
            "GAME_DATE": ["2024-01-15", "2024-01-15"],
            "MATCHUP": ["AAA vs. BBB", "BBB @ AAA"],
            "WL": ["W", "L"],
            "PTS": [110, 100],
            "PLUS_MINUS": [10, -10],
        })

        games = games_collector._transform_games_df(df, "2023-24")

        assert len(games) == 1
        assert games[0].game_date == date(2024, 1, 15)


class TestGamesCollectorErrorHandling:
    """Tests for error handling in GamesCollector."""

    def test_collect_season_with_api_error(
        self,
        games_collector: GamesCollector,
        mock_api_client: MagicMock,
    ) -> None:
        """Should propagate API errors."""
        mock_api_client.get_league_game_finder.side_effect = Exception("API error")

        with pytest.raises(Exception, match="API error"):
            games_collector.collect_season("2023-24")


class TestCollectMethod:
    """Tests for the collect() method with season range."""

    def test_collect_multiple_seasons(
        self,
        games_collector: GamesCollector,
        mock_api_client: MagicMock,
        sample_games_df: pd.DataFrame,
    ) -> None:
        """Should collect games for multiple seasons."""
        mock_api_client.get_league_game_finder.return_value = sample_games_df

        games = games_collector.collect(["2022-23", "2023-24"])

        # Called twice, once per season
        assert mock_api_client.get_league_game_finder.call_count == 2
        # Returns games from both seasons (sample_games_df has 2 games)
        assert len(games) == 4

    def test_collect_with_resume_from(
        self,
        games_collector: GamesCollector,
        mock_api_client: MagicMock,
        sample_games_df: pd.DataFrame,
    ) -> None:
        """Should resume from specified season."""
        mock_api_client.get_league_game_finder.return_value = sample_games_df

        games = games_collector.collect(
            ["2021-22", "2022-23", "2023-24"],
            resume_from="2022-23",
        )

        # Should skip 2021-22 and start from 2022-23
        assert mock_api_client.get_league_game_finder.call_count == 2

    def test_collect_sets_checkpoint(
        self,
        games_collector: GamesCollector,
        mock_api_client: MagicMock,
        sample_games_df: pd.DataFrame,
    ) -> None:
        """Should set checkpoint after each season."""
        mock_api_client.get_league_game_finder.return_value = sample_games_df

        games_collector.collect(["2022-23", "2023-24"])

        # Last checkpoint should be last season
        assert games_collector.get_last_checkpoint() == "2023-24"


class TestCollectGameMethod:
    """Tests for the collect_game() method."""

    def test_collect_game_returns_empty_with_warning(
        self,
        games_collector: GamesCollector,
    ) -> None:
        """collect_game is not efficient for games, returns empty."""
        games = games_collector.collect_game("0022300001")
        assert games == []


class TestGetGameIdsForSeason:
    """Tests for get_game_ids_for_season method."""

    def test_empty_response_returns_empty_list(
        self,
        games_collector: GamesCollector,
        mock_api_client: MagicMock,
    ) -> None:
        """Should return empty list for empty response."""
        mock_api_client.get_league_game_finder.return_value = pd.DataFrame()

        game_ids = games_collector.get_game_ids_for_season("2023-24")

        assert game_ids == []


class TestTransformGameEdgeCases:
    """Tests for edge cases in game transformation."""

    def test_incomplete_game_data_skipped(
        self,
        games_collector: GamesCollector,
        mock_api_client: MagicMock,
    ) -> None:
        """Should skip games with incomplete data (only one team row)."""
        df = pd.DataFrame({
            "SEASON_ID": ["22023"],
            "TEAM_ID": [100],
            "GAME_ID": ["001"],
            "GAME_DATE": ["2024-01-01"],
            "MATCHUP": ["AAA vs. BBB"],
            "WL": ["W"],
            "PTS": [110],
            "PLUS_MINUS": [10],
        })
        mock_api_client.get_league_game_finder.return_value = df

        games = games_collector.collect_season("2023-24")

        # Should skip the incomplete game
        assert len(games) == 0

    def test_fallback_home_away_detection(
        self,
        games_collector: GamesCollector,
        mock_api_client: MagicMock,
    ) -> None:
        """Should use fallback when matchup string doesn't match expected pattern."""
        df = pd.DataFrame({
            "SEASON_ID": ["22023", "22023"],
            "TEAM_ID": [100, 200],
            "GAME_ID": ["001", "001"],
            "GAME_DATE": ["2024-01-01", "2024-01-01"],
            "MATCHUP": ["AAA - BBB", "BBB - AAA"],  # Non-standard format
            "WL": ["W", "L"],
            "PTS": [110, 100],
            "PLUS_MINUS": [10, -10],  # Team 100 has higher plus_minus
        })
        mock_api_client.get_league_game_finder.return_value = df

        games = games_collector.collect_season("2023-24")

        # Should use plus_minus fallback (team with higher is home)
        assert len(games) == 1
        assert games[0].home_team_id == 100

    def test_transform_handles_timestamp_game_date(
        self,
        games_collector: GamesCollector,
        mock_api_client: MagicMock,
    ) -> None:
        """Should handle game date as timestamp object."""
        import pandas as pd

        df = pd.DataFrame({
            "SEASON_ID": ["22023", "22023"],
            "TEAM_ID": [100, 200],
            "GAME_ID": ["001", "001"],
            "GAME_DATE": pd.to_datetime(["2024-01-15", "2024-01-15"]),
            "MATCHUP": ["AAA vs. BBB", "BBB @ AAA"],
            "WL": ["W", "L"],
            "PTS": [110, 100],
            "PLUS_MINUS": [10, -10],
        })
        mock_api_client.get_league_game_finder.return_value = df

        games = games_collector.collect_season("2023-24")

        assert len(games) == 1
        assert games[0].game_date == date(2024, 1, 15)

    def test_transform_handles_nan_scores(
        self,
        games_collector: GamesCollector,
        mock_api_client: MagicMock,
    ) -> None:
        """Should handle NaN scores for scheduled games."""
        import numpy as np

        df = pd.DataFrame({
            "SEASON_ID": ["22023", "22023"],
            "TEAM_ID": [100, 200],
            "GAME_ID": ["001", "001"],
            "GAME_DATE": ["2024-01-15", "2024-01-15"],
            "MATCHUP": ["AAA vs. BBB", "BBB @ AAA"],
            "WL": [np.nan, np.nan],
            "PTS": [np.nan, np.nan],
            "PLUS_MINUS": [0, 0],
        })
        mock_api_client.get_league_game_finder.return_value = df

        games = games_collector.collect_season("2023-24")

        assert len(games) == 1
        assert games[0].home_score is None
        assert games[0].away_score is None
        assert games[0].status == "scheduled"


class TestCollectDateRangeSeasonInference:
    """Tests for season inference in collect_date_range."""

    def test_infers_season_from_october_date(
        self,
        games_collector: GamesCollector,
        mock_api_client: MagicMock,
        sample_games_df: pd.DataFrame,
    ) -> None:
        """Should infer season from October date (season start)."""
        mock_api_client.get_league_game_finder.return_value = sample_games_df

        games_collector.collect_date_range(
            start_date=date(2023, 10, 24),
            end_date=date(2023, 10, 31),
        )

        # Should infer 2023-24 season
        call_kwargs = mock_api_client.get_league_game_finder.call_args[1]
        assert call_kwargs["season"] == "2023-24"

    def test_infers_season_from_january_date(
        self,
        games_collector: GamesCollector,
        mock_api_client: MagicMock,
        sample_games_df: pd.DataFrame,
    ) -> None:
        """Should infer season from January date (mid-season)."""
        mock_api_client.get_league_game_finder.return_value = sample_games_df

        games_collector.collect_date_range(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 15),
        )

        # January 2024 is in 2023-24 season
        call_kwargs = mock_api_client.get_league_game_finder.call_args[1]
        assert call_kwargs["season"] == "2023-24"
