"""Data layer for NBA model.

This module provides data collection, storage, and retrieval functionality
including database engine/session management, SQLAlchemy ORM models,
NBA API client wrapper, data collectors, and ETL pipelines.

Submodules:
    db: Database engine and session management
    schema: SQLAlchemy base class and mixins
    models: SQLAlchemy ORM model definitions
    api: NBA API client with rate limiting and retry logic
    collectors: Individual data collectors
    checkpoint: Pipeline checkpoint management
    pipelines: ETL pipeline orchestration
    stints: Stint derivation logic
    validation: Data validation utilities

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
from nba_model.data.checkpoint import Checkpoint, CheckpointManager
from nba_model.data.collectors import (
    BaseCollector,
    BoxScoreCollector,
    V3_ACTION_TYPES,
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
    GameStats,  # Alias for TeamBoxScore (backward compatibility)
    LineupSpacing,
    Odds,
    Play,
    Player,
    PlayerBoxScore,
    PlayerGameStats,  # Alias for PlayerBoxScore (backward compatibility)
    PlayerRAPM,
    PlayerSeason,
    Season,
    SeasonStats,
    Shot,
    Stint,
    Team,
    TeamBoxScore,
)
from nba_model.data.pipelines import (
    BatchResult,
    CollectionPipeline,
    PipelineResult,
    PipelineStatus,
)
from nba_model.data.schema import Base, TimestampMixin
from nba_model.data.quality import DataQualityReviewer, QualityIssue, QualityReport
from nba_model.data.stints import LineupChange, StintDeriver
from nba_model.data.validation import DataValidator, ValidationResult

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
    "V3_ACTION_TYPES",
    "GamesCollector",
    "PlayByPlayCollector",
    "PlayersCollector",
    "ShotsCollector",
    "TEAM_DATA",
    # Checkpoint management
    "Checkpoint",
    "CheckpointManager",
    # Pipelines
    "BatchResult",
    "CollectionPipeline",
    "PipelineResult",
    "PipelineStatus",
    # Stint derivation
    "LineupChange",
    "StintDeriver",
    # Quality review
    "DataQualityReviewer",
    "QualityIssue",
    "QualityReport",
    # Validation
    "DataValidator",
    "ValidationResult",
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
    "TeamBoxScore",
    "PlayerBoxScore",
    "GameStats",  # Alias for TeamBoxScore (backward compatibility)
    "PlayerGameStats",  # Alias for PlayerBoxScore (backward compatibility)
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
