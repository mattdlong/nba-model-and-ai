"""Tests for PlayersCollector.

Tests player and team collection with mocked API responses.
"""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest

from nba_model.data.collectors.players import (
    PlayersCollector,
    TEAM_DATA,
)
from nba_model.data.models import Player, PlayerSeason, Team


@pytest.fixture
def mock_api_client() -> MagicMock:
    """Create a mock NBA API client."""
    return MagicMock()


@pytest.fixture
def players_collector(mock_api_client: MagicMock) -> PlayersCollector:
    """Create a PlayersCollector with mocked API client."""
    return PlayersCollector(api_client=mock_api_client)


@pytest.fixture
def sample_roster_df() -> pd.DataFrame:
    """Create sample CommonTeamRoster response data."""
    return pd.DataFrame({
        "TeamID": [1610612738, 1610612738],
        "SEASON": ["2023-24", "2023-24"],
        "PLAYER": ["Jayson Tatum", "Jaylen Brown"],
        "PLAYER_ID": [1628369, 1627759],
        "NUM": ["0", "7"],
        "POSITION": ["F", "G-F"],
        "HEIGHT": ["6-8", "6-6"],
        "WEIGHT": ["210", "223"],
        "BIRTH_DATE": ["MAR 03, 1998", "OCT 24, 1996"],
    })


class TestPlayersCollector:
    """Tests for PlayersCollector."""

    def test_collect_rosters(
        self,
        players_collector: PlayersCollector,
        mock_api_client: MagicMock,
        sample_roster_df: pd.DataFrame,
    ) -> None:
        """Should collect players and player_seasons from rosters."""
        mock_api_client.get_team_roster.return_value = sample_roster_df

        players, player_seasons = players_collector.collect_rosters(
            season="2023-24",
            team_ids=[1610612738],  # Just one team for test
        )

        assert len(players) == 2
        assert len(player_seasons) == 2
        assert all(isinstance(p, Player) for p in players)
        assert all(isinstance(ps, PlayerSeason) for ps in player_seasons)

    def test_collect_rosters_deduplicates_players(
        self,
        players_collector: PlayersCollector,
        mock_api_client: MagicMock,
    ) -> None:
        """Should deduplicate players across teams."""
        # Same player on two teams (traded)
        df = pd.DataFrame({
            "TeamID": [100],
            "SEASON": ["2023-24"],
            "PLAYER": ["Player One"],
            "PLAYER_ID": [12345],
            "NUM": ["1"],
            "POSITION": ["G"],
            "HEIGHT": ["6-2"],
            "WEIGHT": ["180"],
            "BIRTH_DATE": ["JAN 01, 1990"],
        })
        mock_api_client.get_team_roster.return_value = df

        players, player_seasons = players_collector.collect_rosters(
            season="2023-24",
            team_ids=[100, 200],  # Two teams
        )

        # Player should only appear once even if on multiple teams
        assert len(players) == 1
        # But should have PlayerSeason for each team
        assert len(player_seasons) == 2

    def test_transform_player_parses_height(
        self,
        players_collector: PlayersCollector,
        mock_api_client: MagicMock,
        sample_roster_df: pd.DataFrame,
    ) -> None:
        """Should parse height from feet-inches format."""
        mock_api_client.get_team_roster.return_value = sample_roster_df

        players, _ = players_collector.collect_rosters(
            season="2023-24",
            team_ids=[1610612738],
        )

        tatum = next(p for p in players if p.player_id == 1628369)
        assert tatum.height_inches == 80  # 6*12 + 8 = 80

    def test_collect_player_details(
        self,
        players_collector: PlayersCollector,
        mock_api_client: MagicMock,
    ) -> None:
        """Should fetch detailed player info."""
        df = pd.DataFrame({
            "DISPLAY_FIRST_LAST": ["Jayson Tatum"],
            "BIRTHDATE": ["1998-03-03T00:00:00"],
            "HEIGHT": ["6-8"],
            "WEIGHT": ["210"],
            "DRAFT_YEAR": ["2017"],
            "DRAFT_ROUND": ["1"],
            "DRAFT_NUMBER": ["3"],
        })
        mock_api_client.get_player_info.return_value = df

        player = players_collector.collect_player_details(1628369)

        assert player is not None
        assert player.full_name == "Jayson Tatum"
        assert player.draft_year == 2017
        assert player.draft_round == 1
        assert player.draft_number == 3


class TestTeamData:
    """Tests for team data."""

    def test_team_data_has_all_30_teams(self) -> None:
        """Should have all 30 NBA teams."""
        assert len(TEAM_DATA) == 30

    def test_team_data_has_required_fields(self) -> None:
        """Each team should have required fields."""
        required_fields = [
            "abbreviation",
            "full_name",
            "city",
            "arena_name",
            "arena_lat",
            "arena_lon",
        ]

        for team_id, data in TEAM_DATA.items():
            for field in required_fields:
                assert field in data, f"Team {team_id} missing {field}"

    def test_collect_teams(
        self,
        players_collector: PlayersCollector,
    ) -> None:
        """Should create Team models from static data."""
        teams = players_collector.collect_teams()

        assert len(teams) == 30
        assert all(isinstance(t, Team) for t in teams)

        # Check one team
        celtics = next(t for t in teams if t.team_id == 1610612738)
        assert celtics.abbreviation == "BOS"
        assert celtics.full_name == "Boston Celtics"
        assert celtics.arena_lat is not None


class TestTransformPlayerSeason:
    """Tests for player season transformation."""

    def test_transform_player_season(
        self,
        players_collector: PlayersCollector,
    ) -> None:
        """Should create PlayerSeason from roster row."""
        row = pd.Series({
            "PLAYER_ID": 1628369,
            "POSITION": "F",
            "NUM": "0",
        })

        ps = players_collector._transform_player_season(row, "2023-24", 1610612738)

        assert ps is not None
        assert ps.player_id == 1628369
        assert ps.season_id == "2023-24"
        assert ps.team_id == 1610612738
        assert ps.position == "F"
        assert ps.jersey_number == "0"


class TestPlayerCollectorErrorHandling:
    """Tests for error handling in PlayersCollector."""

    def test_collect_player_details_empty_response(
        self,
        players_collector: PlayersCollector,
        mock_api_client: MagicMock,
    ) -> None:
        """Should handle empty player info response."""
        mock_api_client.get_player_info.return_value = pd.DataFrame()

        player = players_collector.collect_player_details(99999)

        assert player is None

    def test_transform_player_parses_birthdate(
        self,
        players_collector: PlayersCollector,
        mock_api_client: MagicMock,
    ) -> None:
        """Should parse birth date from roster."""
        df = pd.DataFrame({
            "TeamID": [1610612738],
            "SEASON": ["2023-24"],
            "PLAYER": ["Test Player"],
            "PLAYER_ID": [12345],
            "NUM": ["1"],
            "POSITION": ["G"],
            "HEIGHT": ["6-0"],
            "WEIGHT": ["180"],
            "BIRTH_DATE": ["JAN 15, 1995"],
        })
        mock_api_client.get_team_roster.return_value = df

        players, _ = players_collector.collect_rosters(
            season="2023-24",
            team_ids=[1610612738],
        )

        assert len(players) == 1
        assert players[0].birth_date is not None
