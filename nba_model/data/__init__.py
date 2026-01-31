"""Data layer for NBA model.

This module provides data collection, storage, and retrieval functionality
including database engine/session management, SQLAlchemy ORM models, and
(in future phases) NBA API client wrapper and ETL pipelines.

Submodules:
    db: Database engine and session management
    schema: SQLAlchemy base class and mixins
    models: SQLAlchemy ORM model definitions
    collectors: Individual data collectors (Phase 2b)

Example:
    >>> from nba_model.data import init_db, session_scope
    >>> from nba_model.data import Game, Team, Player
    >>> init_db()  # Create all tables
    >>> with session_scope() as session:
    ...     games = session.query(Game).all()
"""
from __future__ import annotations

from nba_model.data.db import (
    get_engine,
    get_session,
    init_db,
    reset_engine,
    session_scope,
    verify_foreign_keys_enabled,
)
from nba_model.data.models import (
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
)
from nba_model.data.schema import Base, TimestampMixin

__all__ = [
    # Base and mixins
    "Base",
    # Game models
    "Game",
    "GameStats",
    "LineupSpacing",
    "Odds",
    # Play-by-play models
    "Play",
    "Player",
    "PlayerGameStats",
    "PlayerRAPM",
    "PlayerSeason",
    # Core reference models
    "Season",
    "SeasonStats",
    "Shot",
    # Derived/feature models
    "Stint",
    "Team",
    "TimestampMixin",
    # Database utilities
    "get_engine",
    "get_session",
    "init_db",
    "reset_engine",
    "session_scope",
    "verify_foreign_keys_enabled",
]
