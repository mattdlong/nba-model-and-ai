# Ticket: Missing columns in season_stats table - created_at and updated_at

**Date:** 2026-02-01
**Reporter:** TESTER (opencode-claude-split)
**Severity:** High
**Component:** Database Schema / Prediction

## Steps to Reproduce

1. Have a trained model (run `python -m nba_model.cli train all --dry-run`)
2. Activate virtual environment: `source .venv/bin/activate`
3. Run: `python -m nba_model.cli predict game GAMEID` (e.g., `python -m nba_model.cli predict game 0022500017`)

## Expected Behaviour

The command should generate a detailed prediction for the specified game, showing win probability, predicted margin, total points, and top factors.

## Actual Behaviour

The command crashes with an OperationalError:

```
OperationalError: (sqlite3.OperationalError) no such column: season_stats.created_at
```

Full error:
```
[SQL: SELECT season_stats.id AS season_stats_id, season_stats.season_id AS 
season_stats_season_id, season_stats.metric_name AS season_stats_metric_name, 
season_stats.mean_value AS season_stats_mean_value, season_stats.std_value AS 
season_stats_std_value, season_stats.min_value AS season_stats_min_value, 
season_stats.max_value AS season_stats_max_value, season_stats.created_at AS 
season_stats_created_at, season_stats.updated_at AS season_stats_updated_at 
FROM season_stats 
WHERE season_stats.season_id = ?]
```

## Why it is incorrect

The SeasonStats ORM model expects `created_at` and `updated_at` columns, but these columns don't exist in the actual database schema. This is the same pattern of issue as ticket-003-missing-timestamp-columns-game-stats.md - the ORM models have timestamp columns defined but the database migration/schema doesn't include them.

## Impact

Users cannot generate predictions for specific games, which is a core functionality described in USAGE.md. The `predict game` command is completely non-functional.

## Related Issue

See ticket-003-missing-timestamp-columns-game-stats.md - same root cause affecting multiple tables.

## Environment

- OS: macOS
- Python: 3.14.2
- Database: SQLite (data/nba.db)
- Model version: 0.1.0

---

## Resolution

**Status:** RESOLVED
**Fixed by:** DEVELOPER (claude-split)
**Fix Date:** 2026-02-01

### Fix Description

Removed the `TimestampMixin` inheritance from the `SeasonStats` class in `nba_model/data/models.py`. The mixin added `created_at` and `updated_at` columns to the ORM model, but these columns don't exist in the actual database schema. Since `SeasonStats` records store normalization statistics that can be recomputed, timestamps are not essential for this model.

This is the same type of fix as ticket-003 - removing unused `TimestampMixin` from models where the database doesn't have the corresponding columns.

### Code Changes

Changed in `nba_model/data/models.py`:
```python
# Before:
class SeasonStats(Base, TimestampMixin):

# After:
class SeasonStats(Base):
```

### Testing Evidence

**Steps followed:**
1. Activated virtual environment: `source .venv/bin/activate`
2. Ran: `python -m nba_model.cli predict game 0022500017`

**Outcome:**
The command now executes successfully without the `OperationalError: no such column: season_stats.created_at` error. The prediction is generated correctly:
```
╭────────────────────────────── Game Prediction ───────────────────────────────╮
│ Matchup: BOS @ DET                                                           │
│ Date: 2026-01-19                                                             │
│                                                                              │
│ Predictions (Injury-Adjusted):                                               │
│   Home Win: 40.6%                                                            │
│   Margin: -0.6 pts                                                           │
│   Total: 175.0 pts                                                           │
│                                                                              │
│ Confidence: 19%                                                              │
│ Model Version: 0.1.0                                                         │
╰──────────────────────────────────────────────────────────────────────────────╯
```

**Confirmation:** Expected behaviour now matches actual behaviour.
