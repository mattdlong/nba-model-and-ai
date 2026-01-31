"""Data collectors for NBA model.

This module contains individual collectors for different data types:
- GamesCollector: Game schedules and results
- PlayersCollector: Player rosters and information
- PlayByPlayCollector: Play-by-play event data
- ShotsCollector: Shot chart location data
- BoxScoreCollector: Advanced box score statistics

Each collector implements checkpointing for resumable collection
and handles API rate limiting and error recovery.

Example:
    >>> from nba_model.data.collectors import GamesCollector
    >>> collector = GamesCollector(db_session)
    >>> games = collector.collect(season="2023-24")
"""
from __future__ import annotations

# Public API - will be populated in Phase 2
__all__: list[str] = []
