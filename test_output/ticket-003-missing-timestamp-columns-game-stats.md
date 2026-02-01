# Ticket: Missing columns in game_stats table - created_at and updated_at

**Date:** 2026-02-01
**Reporter:** TESTER (opencode-claude-split)
**Severity:** High
**Component:** Database Schema / Feature Engineering

## Steps to Reproduce

1. Activate virtual environment: `source .venv/bin/activate`
2. Run: `python -m nba_model.cli features build`

## Expected Behaviour

The command should build all feature tables (normalization, RAPM, spacing, fatigue) for the available seasons.

## Actual Behaviour

The command crashes with an OperationalError:

```
OperationalError: (sqlite3.OperationalError) no such column: game_stats.created_at
```

Full error:
```
[SQL: SELECT game_stats.id AS game_stats_id, game_stats.game_id AS 
game_stats_game_id, game_stats.team_id AS game_stats_team_id, game_stats.is_home
AS game_stats_is_home, game_stats.points AS game_stats_points, 
game_stats.rebounds AS game_stats_rebounds, game_stats.assists AS 
game_stats_assists, game_stats.steals AS game_stats_steals, game_stats.blocks AS
game_stats_blocks, game_stats.turnovers AS game_stats_turnovers, 
game_stats.offensive_rating AS game_stats_offensive_rating, 
game_stats.defensive_rating AS game_stats_defensive_rating, game_stats.pace AS 
game_stats_pace, game_stats.efg_pct AS game_stats_efg_pct, game_stats.tov_pct AS
game_stats_tov_pct, game_stats.orb_pct AS game_stats_orb_pct, game_stats.ft_rate
AS game_stats_ft_rate, game_stats.created_at AS game_stats_created_at, 
game_stats.updated_at AS game_stats_updated_at, games.season_id AS 
games_season_id 
FROM game_stats JOIN games ON game_stats.game_id = games.game_id 
WHERE games.season_id IN (?, ?, ?)]
[parameters: ('2023-24', '2024-25', '2025-26')]
```

## Why it is incorrect

The code is querying columns `created_at` and `updated_at` from the `game_stats` table, but these columns don't exist in the database schema. The GameStats ORM model appears to expect these timestamp columns for tracking purposes, but they were never added to the database schema.

This is a schema mismatch between:
- The SQLAlchemy ORM model definition (which expects created_at/updated_at)
- The actual SQLite database schema (which doesn't have these columns)

## Impact

This completely blocks the feature engineering workflow described in USAGE.md. Users cannot build features without these columns existing in the database.

## Environment

- OS: macOS
- Python: 3.14.2
- Database: SQLite (data/nba.db)
- Seasons with data: 2023-24, 2024-25, 2025-26
