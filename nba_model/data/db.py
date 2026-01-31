"""Database engine and session management utilities.

This module provides centralized database connection management including
engine creation, session handling, and database initialization.

Example:
    >>> from nba_model.data.db import session_scope, init_db
    >>> init_db()  # Create all tables
    >>> with session_scope() as session:
    ...     session.add(some_model)
"""
from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.orm import Session, scoped_session, sessionmaker

from nba_model.config import get_settings
from nba_model.data.schema import Base

logger = logging.getLogger(__name__)

# Module-level engine and session factory cache
_engine: Engine | None = None
_session_factory: scoped_session[Session] | None = None


def _set_sqlite_pragmas(
    dbapi_connection: Any,
    connection_record: Any,
) -> None:
    """Set SQLite-specific pragmas for better performance and integrity.

    Args:
        dbapi_connection: Raw DBAPI connection object.
        connection_record: Connection pool record (unused).
    """
    # Type is sqlite3.Connection but we use Any for DBAPI compatibility
    cursor = dbapi_connection.cursor()
    # Enable foreign key enforcement
    cursor.execute("PRAGMA foreign_keys=ON")
    # Enable WAL mode for better concurrent read/write
    cursor.execute("PRAGMA journal_mode=WAL")
    # Set synchronous mode to NORMAL for better performance
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()
    logger.debug("SQLite pragmas applied: foreign_keys=ON, journal_mode=WAL")


def get_engine() -> Engine:
    """Get or create the SQLAlchemy engine from settings.

    Creates a new engine on first call and caches it for subsequent calls.
    The engine is configured based on settings.db_path.

    Returns:
        SQLAlchemy Engine instance.

    Example:
        >>> engine = get_engine()
        >>> with engine.connect() as conn:
        ...     result = conn.execute(text("SELECT 1"))
    """
    global _engine
    if _engine is None:
        settings = get_settings()
        db_path = Path(settings.db_path)

        # Create parent directories if they don't exist
        db_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensuring database directory exists: {db_path.parent}")

        # Create SQLite engine
        db_url = f"sqlite:///{db_path}"
        _engine = create_engine(
            db_url,
            echo=False,  # Set to True for SQL logging
            pool_pre_ping=True,  # Verify connections before use
        )

        # Register SQLite pragma listener
        event.listen(_engine, "connect", _set_sqlite_pragmas)
        logger.debug(f"Created database engine: {db_url}")

    return _engine


def get_session() -> Session:
    """Get a new database session.

    Uses scoped_session pattern for thread safety. Each thread gets
    its own session instance.

    Returns:
        SQLAlchemy Session instance.

    Example:
        >>> session = get_session()
        >>> try:
        ...     session.add(some_model)
        ...     session.commit()
        ... finally:
        ...     session.close()
    """
    global _session_factory
    if _session_factory is None:
        engine = get_engine()
        factory = sessionmaker(bind=engine)
        _session_factory = scoped_session(factory)
        logger.debug("Created scoped session factory")

    return _session_factory()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Context manager for database sessions with auto-commit/rollback.

    Automatically commits on successful exit and rolls back on exception.
    The session is always closed after use.

    Yields:
        SQLAlchemy Session instance.

    Raises:
        Exception: Re-raises any exception after rollback.

    Example:
        >>> with session_scope() as session:
        ...     game = Game(game_id="123", ...)
        ...     session.add(game)
        ... # Auto-commits on exit
    """
    session = get_session()
    try:
        logger.debug("Starting database session")
        yield session
        session.commit()
        logger.debug("Session committed successfully")
    except Exception:
        logger.debug("Session rolling back due to exception")
        session.rollback()
        raise
    finally:
        session.close()
        logger.debug("Session closed")


def init_db() -> None:
    """Initialize the database by creating all tables.

    Creates all tables defined by models that inherit from Base.
    Safe to call multiple times - won't recreate existing tables.

    Example:
        >>> init_db()  # Creates all tables
    """
    # Import models to ensure they're registered with Base
    # This is required for create_all to find them
    from nba_model.data import models  # noqa: F401

    engine = get_engine()
    Base.metadata.create_all(engine)
    logger.info("Database initialized - all tables created")


def reset_engine() -> None:
    """Reset the engine and session factory (for testing).

    Clears the cached engine and session factory so that
    subsequent calls to get_engine() create a fresh connection.
    """
    global _engine, _session_factory
    if _session_factory is not None:
        _session_factory.remove()
        _session_factory = None
    if _engine is not None:
        _engine.dispose()
        _engine = None
    logger.debug("Database engine and session factory reset")


def verify_foreign_keys_enabled() -> bool:
    """Verify that foreign key constraints are enabled.

    Returns:
        True if foreign keys are enabled, False otherwise.
    """
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("PRAGMA foreign_keys"))
        row = result.fetchone()
        return row is not None and row[0] == 1
