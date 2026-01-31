"""Data layer for NBA model.

This module provides data collection, storage, and retrieval functionality
including database engine/session management, SQLAlchemy ORM models,
NBA API client wrapper, and data collectors.

Submodules:
    db: Database engine and session management
    schema: SQLAlchemy base class and mixins
    models: SQLAlchemy ORM model definitions
    api: NBA API client with rate limiting and retry logic
    collectors: Individual data collectors

Example:
    >>> from nba_model.data import init_db, session_scope
    >>> from nba_model.data import NBAApiClient, GamesCollector
    >>> init_db()
    >>> client = NBAApiClient()
    >>> collector = GamesCollector(client)
    >>> games = collector.collect_season("2023-24")
"""
from __future__ import annotations

from nba_model.data.api import (
    NBAApiClient,
    NBAApiError,
    NBAApiNotFoundError,
    NBAApiRateLimitError,
    NBAApiTimeoutError,
)
from nba_model.data.collectors import (
    BaseCollector,
    BoxScoreCollector,
    EVENT_TYPES,
    GamesCollector,
    PlayByPlayCollector,
    PlayersCollector,
    ShotsCollector,
    TEAM_DATA,
)
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
    # API client and exceptions
    "NBAApiClient",
    "NBAApiError",
    "NBAApiNotFoundError",
    "NBAApiRateLimitError",
    "NBAApiTimeoutError",
    # Collectors
    "BaseCollector",
    "BoxScoreCollector",
    "EVENT_TYPES",
    "GamesCollector",
    "PlayByPlayCollector",
    "PlayersCollector",
    "ShotsCollector",
    "TEAM_DATA",
    # Database utilities
    "get_engine",
    "get_session",
    "init_db",
    "reset_engine",
    "session_scope",
    "verify_foreign_keys_enabled",
    # Base and mixins
    "Base",
    "TimestampMixin",
    # Core reference models
    "Season",
    "Team",
    "Player",
    "PlayerSeason",
    # Game models
    "Game",
    "GameStats",
    "PlayerGameStats",
    # Play-by-play models
    "Play",
    "Shot",
    # Derived/feature models
    "Stint",
    "Odds",
    "PlayerRAPM",
    "LineupSpacing",
    "SeasonStats",
]
