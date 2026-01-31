"""Tests for SQLAlchemy schema and model definitions.

Tests the database schema including table creation, column types,
constraints, and indexes.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING, Generator

import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError

from nba_model.config import reset_settings
from nba_model.data import (
    Base,
    Game,
    GameStats,
    LineupSpacing,
    Odds,
    Play,
    Player,
    PlayerGameStats,
    PlayerRAPM,
    PlayerSeason,
    Season,
    SeasonStats,
    Shot,
    Stint,
    Team,
    get_engine,
    init_db,
    reset_engine,
    session_scope,
)

if TYPE_CHECKING:
    from nba_model.config import Settings


@pytest.fixture(autouse=True)
def reset_db_between_tests(test_settings: "Settings") -> Generator[None, None, None]:
    """Reset database engine before and after each test."""
    reset_engine()
    init_db()
    yield
    reset_engine()


class TestTableCreation:
    """Tests for table creation and structure."""

    def test_all_tables_created(self, test_settings: "Settings") -> None:
        """All 14 tables should be created successfully."""
        engine = get_engine()
        inspector = inspect(engine)
        tables = set(inspector.get_table_names())

        expected_tables = {
            "seasons",
            "teams",
            "players",
            "player_seasons",
            "games",
            "game_stats",
            "player_game_stats",
            "plays",
            "shots",
            "stints",
            "odds",
            "player_rapm",
            "lineup_spacing",
            "season_stats",
        }

        assert expected_tables == tables

    def test_season_table_columns(self, test_settings: "Settings") -> None:
        """Season table should have correct columns."""
        engine = get_engine()
        inspector = inspect(engine)
        columns = {col["name"] for col in inspector.get_columns("seasons")}

        expected = {"season_id", "start_date", "end_date", "games_count"}
        assert expected == columns

    def test_team_table_columns(self, test_settings: "Settings") -> None:
        """Team table should have correct columns."""
        engine = get_engine()
        inspector = inspect(engine)
        columns = {col["name"] for col in inspector.get_columns("teams")}

        expected = {
            "team_id",
            "abbreviation",
            "full_name",
            "city",
            "arena_name",
            "arena_lat",
            "arena_lon",
        }
        assert expected == columns

    def test_player_table_columns(self, test_settings: "Settings") -> None:
        """Player table should have correct columns."""
        engine = get_engine()
        inspector = inspect(engine)
        columns = {col["name"] for col in inspector.get_columns("players")}

        expected = {
            "player_id",
            "full_name",
            "height_inches",
            "weight_lbs",
            "birth_date",
            "draft_year",
            "draft_round",
            "draft_number",
        }
        assert expected == columns

    def test_game_table_columns(self, test_settings: "Settings") -> None:
        """Game table should have correct columns."""
        engine = get_engine()
        inspector = inspect(engine)
        columns = {col["name"] for col in inspector.get_columns("games")}

        expected = {
            "game_id",
            "season_id",
            "game_date",
            "home_team_id",
            "away_team_id",
            "home_score",
            "away_score",
            "status",
            "attendance",
        }
        assert expected == columns


class TestNullableConstraints:
    """Tests for nullable/non-nullable constraints."""

    def test_season_id_not_nullable(self, test_settings: "Settings") -> None:
        """Season.season_id should not be nullable."""
        engine = get_engine()
        inspector = inspect(engine)
        columns = inspector.get_columns("seasons")
        season_id_col = next(c for c in columns if c["name"] == "season_id")
        assert season_id_col["nullable"] is False

    def test_games_count_nullable(self, test_settings: "Settings") -> None:
        """Season.games_count should be nullable."""
        engine = get_engine()
        inspector = inspect(engine)
        columns = inspector.get_columns("seasons")
        games_count_col = next(c for c in columns if c["name"] == "games_count")
        assert games_count_col["nullable"] is True

    def test_team_required_fields(self, test_settings: "Settings") -> None:
        """Team required fields should not be nullable."""
        engine = get_engine()
        inspector = inspect(engine)
        columns = inspector.get_columns("teams")

        for col in columns:
            if col["name"] in ["team_id", "abbreviation", "full_name", "city"]:
                assert col["nullable"] is False, f"{col['name']} should not be nullable"
            elif col["name"] in ["arena_name", "arena_lat", "arena_lon"]:
                assert col["nullable"] is True, f"{col['name']} should be nullable"


class TestForeignKeyConstraints:
    """Tests for foreign key constraint enforcement."""

    def test_game_requires_valid_season(self, test_settings: "Settings") -> None:
        """Game should require valid season_id foreign key."""
        # First create required teams
        with session_scope() as session:
            team1 = Team(
                team_id=1,
                abbreviation="HOM",
                full_name="Home Team",
                city="Home City",
            )
            team2 = Team(
                team_id=2,
                abbreviation="AWY",
                full_name="Away Team",
                city="Away City",
            )
            session.add_all([team1, team2])

        # Try to create game with invalid season
        with pytest.raises(IntegrityError):
            with session_scope() as session:
                game = Game(
                    game_id="0022300001",
                    season_id="INVALID",  # Does not exist
                    game_date=date(2024, 1, 1),
                    home_team_id=1,
                    away_team_id=2,
                    status="scheduled",
                )
                session.add(game)

    def test_game_requires_valid_teams(self, test_settings: "Settings") -> None:
        """Game should require valid team foreign keys."""
        # Create a valid season
        with session_scope() as session:
            season = Season(
                season_id="2023-24",
                start_date=date(2023, 10, 24),
                end_date=date(2024, 4, 14),
            )
            session.add(season)

        # Try to create game with invalid team
        with pytest.raises(IntegrityError):
            with session_scope() as session:
                game = Game(
                    game_id="0022300001",
                    season_id="2023-24",
                    game_date=date(2024, 1, 1),
                    home_team_id=9999,  # Does not exist
                    away_team_id=9998,  # Does not exist
                    status="scheduled",
                )
                session.add(game)

    def test_player_season_requires_valid_references(
        self, test_settings: "Settings"
    ) -> None:
        """PlayerSeason should require valid foreign keys."""
        with pytest.raises(IntegrityError):
            with session_scope() as session:
                ps = PlayerSeason(
                    player_id=9999,  # Invalid
                    season_id="INVALID",  # Invalid
                    team_id=9999,  # Invalid
                )
                session.add(ps)


class TestUniqueConstraints:
    """Tests for unique constraint enforcement."""

    def test_player_season_unique_constraint(
        self, test_settings: "Settings"
    ) -> None:
        """PlayerSeason should have unique (player_id, season_id, team_id)."""
        # Setup: create required records
        with session_scope() as session:
            season = Season(
                season_id="2023-24",
                start_date=date(2023, 10, 24),
                end_date=date(2024, 4, 14),
            )
            team = Team(
                team_id=1,
                abbreviation="TST",
                full_name="Test Team",
                city="Test City",
            )
            player = Player(player_id=1, full_name="Test Player")
            session.add_all([season, team, player])

        # First insert should succeed
        with session_scope() as session:
            ps1 = PlayerSeason(
                player_id=1,
                season_id="2023-24",
                team_id=1,
            )
            session.add(ps1)

        # Duplicate should fail
        with pytest.raises(IntegrityError):
            with session_scope() as session:
                ps2 = PlayerSeason(
                    player_id=1,
                    season_id="2023-24",
                    team_id=1,  # Same combination
                )
                session.add(ps2)

    def test_game_stats_unique_constraint(
        self, test_settings: "Settings"
    ) -> None:
        """GameStats should have unique (game_id, team_id)."""
        # Setup
        with session_scope() as session:
            season = Season(
                season_id="2023-24",
                start_date=date(2023, 10, 24),
                end_date=date(2024, 4, 14),
            )
            team1 = Team(
                team_id=1,
                abbreviation="HOM",
                full_name="Home Team",
                city="Home City",
            )
            team2 = Team(
                team_id=2,
                abbreviation="AWY",
                full_name="Away Team",
                city="Away City",
            )
            session.add_all([season, team1, team2])

        with session_scope() as session:
            game = Game(
                game_id="0022300001",
                season_id="2023-24",
                game_date=date(2024, 1, 1),
                home_team_id=1,
                away_team_id=2,
                status="completed",
            )
            session.add(game)

        # First insert should succeed
        with session_scope() as session:
            gs1 = GameStats(
                game_id="0022300001",
                team_id=1,
                is_home=True,
                points=100,
            )
            session.add(gs1)

        # Duplicate should fail
        with pytest.raises(IntegrityError):
            with session_scope() as session:
                gs2 = GameStats(
                    game_id="0022300001",
                    team_id=1,  # Same game + team
                    is_home=True,
                )
                session.add(gs2)

    def test_play_unique_constraint(self, test_settings: "Settings") -> None:
        """Play should have unique (game_id, event_num)."""
        # Setup
        with session_scope() as session:
            season = Season(
                season_id="2023-24",
                start_date=date(2023, 10, 24),
                end_date=date(2024, 4, 14),
            )
            team1 = Team(team_id=1, abbreviation="HOM", full_name="Home", city="Home")
            team2 = Team(team_id=2, abbreviation="AWY", full_name="Away", city="Away")
            game = Game(
                game_id="0022300001",
                season_id="2023-24",
                game_date=date(2024, 1, 1),
                home_team_id=1,
                away_team_id=2,
                status="completed",
            )
            session.add_all([season, team1, team2, game])

        # First insert should succeed
        with session_scope() as session:
            play1 = Play(
                game_id="0022300001",
                event_num=1,
                period=1,
                event_type=12,
            )
            session.add(play1)

        # Duplicate should fail
        with pytest.raises(IntegrityError):
            with session_scope() as session:
                play2 = Play(
                    game_id="0022300001",
                    event_num=1,  # Same game + event_num
                    period=1,
                    event_type=12,
                )
                session.add(play2)


class TestIndexes:
    """Tests for index creation."""

    def test_game_indexes_exist(self, test_settings: "Settings") -> None:
        """Game table should have date and season indexes."""
        engine = get_engine()
        inspector = inspect(engine)
        indexes = inspector.get_indexes("games")
        index_names = {idx["name"] for idx in indexes}

        assert "idx_games_date" in index_names
        assert "idx_games_season" in index_names

    def test_play_indexes_exist(self, test_settings: "Settings") -> None:
        """Play table should have game index."""
        engine = get_engine()
        inspector = inspect(engine)
        indexes = inspector.get_indexes("plays")
        index_names = {idx["name"] for idx in indexes}

        assert "idx_plays_game" in index_names

    def test_shot_indexes_exist(self, test_settings: "Settings") -> None:
        """Shot table should have game and player indexes."""
        engine = get_engine()
        inspector = inspect(engine)
        indexes = inspector.get_indexes("shots")
        index_names = {idx["name"] for idx in indexes}

        assert "idx_shots_game" in index_names
        assert "idx_shots_player" in index_names

    def test_stint_index_exists(self, test_settings: "Settings") -> None:
        """Stint table should have game index."""
        engine = get_engine()
        inspector = inspect(engine)
        indexes = inspector.get_indexes("stints")
        index_names = {idx["name"] for idx in indexes}

        assert "idx_stints_game" in index_names

    def test_player_game_stats_index_exists(
        self, test_settings: "Settings"
    ) -> None:
        """PlayerGameStats table should have composite index."""
        engine = get_engine()
        inspector = inspect(engine)
        indexes = inspector.get_indexes("player_game_stats")
        index_names = {idx["name"] for idx in indexes}

        assert "idx_player_game_stats" in index_names


class TestTimestampMixin:
    """Tests for TimestampMixin functionality."""

    def test_created_at_auto_set(self, test_settings: "Settings") -> None:
        """created_at should be automatically set on insert."""
        with session_scope() as session:
            season = Season(
                season_id="2023-24",
                start_date=date(2023, 10, 24),
                end_date=date(2024, 4, 14),
            )
            team1 = Team(team_id=1, abbreviation="HOM", full_name="Home", city="Home")
            team2 = Team(team_id=2, abbreviation="AWY", full_name="Away", city="Away")
            game = Game(
                game_id="0022300001",
                season_id="2023-24",
                game_date=date(2024, 1, 1),
                home_team_id=1,
                away_team_id=2,
                status="completed",
            )
            session.add_all([season, team1, team2, game])

        # GameStats uses TimestampMixin
        with session_scope() as session:
            gs = GameStats(
                game_id="0022300001",
                team_id=1,
                is_home=True,
            )
            session.add(gs)

        with session_scope() as session:
            gs = session.query(GameStats).first()
            assert gs is not None
            assert gs.created_at is not None
            assert isinstance(gs.created_at, datetime)

    def test_updated_at_auto_set(self, test_settings: "Settings") -> None:
        """updated_at should be automatically set on insert."""
        with session_scope() as session:
            season = Season(
                season_id="2023-24",
                start_date=date(2023, 10, 24),
                end_date=date(2024, 4, 14),
            )
            team1 = Team(team_id=1, abbreviation="HOM", full_name="Home", city="Home")
            team2 = Team(team_id=2, abbreviation="AWY", full_name="Away", city="Away")
            game = Game(
                game_id="0022300001",
                season_id="2023-24",
                game_date=date(2024, 1, 1),
                home_team_id=1,
                away_team_id=2,
                status="completed",
            )
            session.add_all([season, team1, team2, game])

        with session_scope() as session:
            gs = GameStats(
                game_id="0022300001",
                team_id=1,
                is_home=True,
            )
            session.add(gs)

        with session_scope() as session:
            gs = session.query(GameStats).first()
            assert gs is not None
            assert gs.updated_at is not None
            assert isinstance(gs.updated_at, datetime)
