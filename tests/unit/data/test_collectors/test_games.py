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
