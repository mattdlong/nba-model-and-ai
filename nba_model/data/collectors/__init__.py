"""Data collectors for NBA model.

This module contains individual collectors for different data types:
- GamesCollector: Game schedules and results
- PlayersCollector: Player rosters and information
- PlayByPlayCollector: Play-by-play event data
- ShotsCollector: Shot chart location data
- BoxScoreCollector: Advanced box score statistics

Each collector handles API rate limiting and error recovery
through the shared NBAApiClient.

Example:
    >>> from nba_model.data.collectors import GamesCollector
    >>> from nba_model.data.api import NBAApiClient
    >>> client = NBAApiClient()
    >>> collector = GamesCollector(client)
    >>> games = collector.collect_season("2023-24")
"""
from __future__ import annotations

from nba_model.data.collectors.base import BaseCollector
from nba_model.data.collectors.boxscores import BoxScoreCollector
from nba_model.data.collectors.games import GamesCollector
from nba_model.data.collectors.players import PlayersCollector, TEAM_DATA
from nba_model.data.collectors.playbyplay import EVENT_TYPES, PlayByPlayCollector
from nba_model.data.collectors.shots import ShotsCollector

__all__ = [
    "BaseCollector",
    "BoxScoreCollector",
    "EVENT_TYPES",
    "GamesCollector",
    "PlayByPlayCollector",
    "PlayersCollector",
    "ShotsCollector",
    "TEAM_DATA",
]
