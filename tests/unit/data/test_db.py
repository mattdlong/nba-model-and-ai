"""Tests for database engine and session management.

Tests the database connection utilities in nba_model.data.db including
engine creation, session management, and database initialization.
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Generator

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from nba_model.config import reset_settings
from nba_model.data import (
    Base,
    get_engine,
    get_session,
    init_db,
    reset_engine,
    session_scope,
    verify_foreign_keys_enabled,
)
from nba_model.data.models import Season, Team

if TYPE_CHECKING:
    from nba_model.config import Settings


@pytest.fixture(autouse=True)
def reset_db_between_tests(test_settings: "Settings") -> Generator[None, None, None]:
    """Reset database engine before and after each test."""
    reset_engine()
    yield
    reset_engine()


class TestGetEngine:
    """Tests for get_engine() function."""

    def test_get_engine_creates_from_settings(
        self, test_settings: "Settings"
    ) -> None:
        """Engine should be created from settings.db_path."""
        engine = get_engine()
        assert engine is not None
        # Verify the URL contains the expected path
        assert str(test_settings.db_path) in str(engine.url)

    def test_get_engine_caches_engine(self, test_settings: "Settings") -> None:
        """Engine should be cached on subsequent calls."""
        engine1 = get_engine()
        engine2 = get_engine()
        assert engine1 is engine2

    def test_get_engine_creates_parent_directories(
        self, tmp_path: Path
    ) -> None:
        """Engine should create parent directories for database file."""
        nested_path = tmp_path / "deep" / "nested" / "dir" / "test.db"
        os.environ["NBA_DB_PATH"] = str(nested_path)
        reset_settings()
        reset_engine()

        try:
            get_engine()
            assert nested_path.parent.exists()
        finally:
            os.environ.pop("NBA_DB_PATH", None)
            reset_settings()


class TestGetSession:
    """Tests for get_session() function."""

    def test_get_session_returns_session(self, test_settings: "Settings") -> None:
        """get_session should return a valid Session object."""
        session = get_session()
        assert session is not None
        # Verify we can execute a simple query
        result = session.execute(text("SELECT 1"))
        assert result.scalar() == 1
        session.close()

    def test_get_session_is_scoped(self, test_settings: "Settings") -> None:
        """Sessions from same thread should be the same instance."""
        session1 = get_session()
        session2 = get_session()
        # Scoped session returns same session in same thread
        assert session1 is session2
        session1.close()


class TestSessionScope:
    """Tests for session_scope() context manager."""

    def test_session_scope_commits_on_success(
        self, test_settings: "Settings"
    ) -> None:
        """Session should commit changes on successful exit."""
        init_db()

        # Add a record inside session scope
        with session_scope() as session:
            season = Season(
                season_id="2023-24",
                start_date=date(2023, 10, 24),
                end_date=date(2024, 4, 14),
            )
            session.add(season)

        # Verify record was committed by reading in new session
        with session_scope() as session:
            found = session.query(Season).filter_by(season_id="2023-24").first()
            assert found is not None
            assert found.season_id == "2023-24"

    def test_session_scope_rolls_back_on_exception(
        self, test_settings: "Settings"
    ) -> None:
        """Session should rollback changes on exception."""
        init_db()

        # Add initial record
        with session_scope() as session:
            team = Team(
                team_id=1,
                abbreviation="TST",
                full_name="Test Team",
                city="Test City",
            )
            session.add(team)

        # Try to add duplicate - should raise
        with pytest.raises(IntegrityError):
            with session_scope() as session:
                # Add a record that should be rolled back
                team2 = Team(
                    team_id=2,
                    abbreviation="TS2",
                    full_name="Test Team 2",
                    city="Test City 2",
                )
                session.add(team2)
                session.flush()

                # This should fail due to primary key constraint
                team_dup = Team(
                    team_id=2,  # Duplicate ID
                    abbreviation="DUP",
                    full_name="Duplicate",
                    city="Dup City",
                )
                session.add(team_dup)
                session.flush()

        # Verify rollback - team2 should not exist
        with session_scope() as session:
            found = session.query(Team).filter_by(team_id=2).first()
            assert found is None

    def test_session_scope_closes_session(
        self, test_settings: "Settings"
    ) -> None:
        """Session should be closed after context manager exits."""
        init_db()

        with session_scope() as session:
            # Use the session
            session.execute(text("SELECT 1"))

        # After exit, we can't use the session
        # (A new get_session() call would be needed)
        # This test mainly verifies no exception during cleanup


class TestInitDb:
    """Tests for init_db() function."""

    def test_init_db_creates_all_tables(self, test_settings: "Settings") -> None:
        """init_db should create all 14 tables."""
        init_db()

        engine = get_engine()

        # Get list of all tables in the database
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        # Verify all expected tables exist
        expected_tables = [
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
        ]

        for table in expected_tables:
            assert table in tables, f"Table '{table}' not found in database"

    def test_init_db_idempotent(self, test_settings: "Settings") -> None:
        """init_db should be safe to call multiple times."""
        init_db()
        init_db()  # Second call should not raise
        init_db()  # Third call should not raise

        # Tables should still exist
        engine = get_engine()
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        assert "seasons" in tables


class TestSQLitePragmas:
    """Tests for SQLite-specific configuration."""

    def test_foreign_keys_enabled(self, test_settings: "Settings") -> None:
        """Foreign key constraints should be enabled."""
        init_db()
        assert verify_foreign_keys_enabled() is True

    def test_wal_mode_enabled(self, test_settings: "Settings") -> None:
        """WAL journal mode should be enabled."""
        init_db()
        engine = get_engine()

        with engine.connect() as conn:
            result = conn.execute(text("PRAGMA journal_mode"))
            mode = result.scalar()
            assert mode == "wal", f"Expected WAL mode, got {mode}"


class TestResetEngine:
    """Tests for reset_engine() function."""

    def test_reset_engine_clears_cache(self, test_settings: "Settings") -> None:
        """reset_engine should allow creating a fresh engine."""
        engine1 = get_engine()
        reset_engine()
        engine2 = get_engine()
        # After reset, should get a different engine instance
        assert engine1 is not engine2

    def test_reset_engine_disposes_connections(
        self, test_settings: "Settings"
    ) -> None:
        """reset_engine should dispose of connection pool."""
        engine = get_engine()
        init_db()

        # Create a session to establish connections
        session = get_session()
        session.execute(text("SELECT 1"))
        session.close()

        # Reset should dispose connections without error
        reset_engine()


class TestDatabaseFileLocation:
    """Tests for database file creation in correct location."""

    def test_database_file_created(self, test_settings: "Settings") -> None:
        """Database file should be created at settings.db_path."""
        init_db()

        # Force a connection to create the file
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        db_path = Path(test_settings.db_path)
        assert db_path.exists(), f"Database file not found at {db_path}"
