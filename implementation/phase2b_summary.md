# Phase 2b Implementation Summary

## Overview

Phase 2b implements the data collection layer for the NBA prediction system. This layer is responsible for fetching data from the NBA API and transforming it into the ORM models defined in Phase 2a.

## Components Implemented

### 1. NBA API Client (`nba_model/data/api.py`)

A robust wrapper around the `nba_api` library with:

- **Rate Limiting**: Configurable delay between API calls (default 0.6s)
- **Retry Logic**: Exponential backoff for transient errors (timeout, 429, 5xx)
- **Error Handling**: Custom exception hierarchy for different error types
- **Consistent Interface**: Returns pandas DataFrames for all endpoints

**Key Classes:**
- `NBAApiClient` - Main client with rate limiting and retry
- `NBAApiError` - Base exception for API errors
- `NBAApiRateLimitError` - Rate limit (429) errors
- `NBAApiNotFoundError` - Resource not found (404) errors
- `NBAApiTimeoutError` - Request timeout errors

**Supported Endpoints:**
- `get_league_game_finder()` - Season games
- `get_play_by_play()` - Game play-by-play
- `get_shot_chart()` - Shot chart data
- `get_boxscore_traditional()` - Traditional box scores
- `get_boxscore_advanced()` - Advanced box scores
- `get_player_tracking()` - Player tracking metrics
- `get_team_roster()` - Team rosters
- `get_player_info()` - Player biographical info

### 2. Base Collector (`nba_model/data/collectors/base.py`)

Abstract base class providing common functionality:

- **Progress Logging**: `_log_progress()` for collection status
- **Error Handling**: `_handle_error()` with configurable strategies (raise, skip, log)
- **Dependency Injection**: API client and session injection

### 3. Games Collector (`nba_model/data/collectors/games.py`)

Collects game schedule and results:

- `collect_season()` - All games for a season
- `collect_date_range()` - Games within date range
- `get_game_ids_for_season()` - List of game IDs

**Features:**
- Home/away detection from matchup string
- Game status determination (completed/scheduled)
- Deduplication of two-team game records

### 4. Players Collector (`nba_model/data/collectors/players.py`)

Collects player and team data:

- `collect_rosters()` - Team rosters with player seasons
- `collect_player_details()` - Detailed player information
- `collect_teams()` - All 30 NBA teams from static data

**Features:**
- Player deduplication across teams (trades)
- Height parsing (feet-inches to total inches)
- Birth date parsing from multiple formats
- Static TEAM_DATA with arena coordinates for all 30 teams

### 5. Play-by-Play Collector (`nba_model/data/collectors/playbyplay.py`)

Collects game play-by-play events:

- `collect_game()` - Single game plays
- `collect_games()` - Multiple games with error handling

**Features:**
- Event type code mapping (EVENT_TYPES constant)
- Score parsing from "away - home" format
- Player ID extraction for primary/secondary/tertiary players

### 6. Shots Collector (`nba_model/data/collectors/shots.py`)

Collects shot chart data with court coordinates:

- `collect_game()` - All shots for a game
- `collect_player_season()` - Player's shots for a season
- `collect_games()` - Multiple games

**Features:**
- Court coordinate preservation (LOC_X: -250 to 250, LOC_Y: -50 to 900)
- Shot type normalization (2PT/3PT)
- Zone classification fields

### 7. Box Scores Collector (`nba_model/data/collectors/boxscores.py`)

Collects comprehensive box score data:

- `collect_game()` - Full stats for a game
- `collect_games()` - Multiple games

**Features:**
- Merges traditional + advanced + tracking stats
- Game stats (team-level) transformation
- Player game stats transformation
- Minutes parsing (MM:SS and decimal formats)

## Test Coverage

- **267 tests passing**
- **90.55% overall coverage** (exceeds 90% target)
- All collectors have comprehensive unit tests with mocked API responses

## Files Created/Modified

### Source Files
- `nba_model/data/api.py` - NEW (153 lines)
- `nba_model/data/collectors/base.py` - NEW (102 lines)
- `nba_model/data/collectors/games.py` - NEW (283 lines)
- `nba_model/data/collectors/players.py` - NEW (551 lines)
- `nba_model/data/collectors/playbyplay.py` - NEW (252 lines)
- `nba_model/data/collectors/shots.py` - NEW (250 lines)
- `nba_model/data/collectors/boxscores.py` - NEW (414 lines)
- `nba_model/data/collectors/__init__.py` - UPDATED (exports)
- `nba_model/data/__init__.py` - UPDATED (exports)

### Test Files
- `tests/unit/data/test_api.py` - NEW (18 test classes)
- `tests/unit/data/test_collectors/test_base.py` - NEW
- `tests/unit/data/test_collectors/test_games.py` - NEW (9 test classes)
- `tests/unit/data/test_collectors/test_players.py` - NEW (6 test classes)
- `tests/unit/data/test_collectors/test_playbyplay.py` - NEW (6 test classes)
- `tests/unit/data/test_collectors/test_shots.py` - NEW (7 test classes)
- `tests/unit/data/test_collectors/test_boxscores.py` - NEW (6 test classes)

## Dependencies

- `nba_api>=1.4.0` - NBA Stats API wrapper
- `pandas>=2.0.0` - Data manipulation
- `requests>=2.31.0` - HTTP requests

## Usage Example

```python
from nba_model.data.api import NBAApiClient
from nba_model.data.collectors import GamesCollector, BoxScoreCollector

# Initialize API client with rate limiting
api_client = NBAApiClient(delay=0.6, max_retries=3)

# Collect games for a season
games_collector = GamesCollector(api_client)
games = games_collector.collect_season("2023-24")

# Collect box scores
box_collector = BoxScoreCollector(api_client)
for game in games[:10]:
    game_stats, player_stats = box_collector.collect_game(game.game_id)
```

## Next Steps

Phase 2c will implement:
- ETL pipelines for orchestrated data collection
- Checkpointing for resumable collection
- Database storage integration
