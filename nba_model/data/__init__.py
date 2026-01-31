"""Data layer for NBA model.

This module provides data collection, storage, and retrieval functionality
including NBA API client wrapper, SQLAlchemy ORM models, and ETL pipelines.

Submodules:
    api: NBA API client with rate limiting
    models: SQLAlchemy ORM model definitions
    schema: Database schema reference
    pipelines: ETL orchestration
    collectors: Individual data collectors

Example:
    >>> from nba_model.data import NBAApiClient
    >>> client = NBAApiClient(delay=0.6)
    >>> games = client.get_league_game_finder(season="2023-24")
"""
from __future__ import annotations

# Public API - will be populated in Phase 2
__all__: list[str] = []
