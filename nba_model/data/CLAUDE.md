# Data Layer

## Responsibility

Owns all data acquisition, storage, and retrieval for NBA statistics. Single source of truth for game data, player stats, play-by-play, and shot charts.

## Status

✅ **Phase 2 - Complete**

## Structure

| File | Purpose | Status |
|------|---------|--------|
| `__init__.py` | Public API exports | ✅ Complete |
| `api.py` | NBA API client with rate limiting | ✅ Complete |
| `models.py` | SQLAlchemy ORM models | ✅ Complete |
| `schema.py` | Database schema reference | ✅ Complete |
| `db.py` | Database connection management | ✅ Complete |
| `pipelines.py` | ETL orchestration | ✅ Complete |
| `checkpoint.py` | Collection checkpointing | ✅ Complete |
| `stints.py` | Lineup stint derivation | ✅ Complete |
| `validation.py` | Data validation | ✅ Complete |
| `collectors/` | Entity-specific collectors | ✅ Complete |

## Key Classes

- `NBAApiClient` - Rate-limited API wrapper (0.6s delay, 3 retries)
- `Game`, `Player`, `Stint`, `Shot`, `Play` - SQLAlchemy ORM models
- `CollectionPipeline` - Orchestrates collection with checkpointing
- `StintDeriver` - Derives lineup stints from play-by-play
- `DataValidator` - Validates collected data batches

## Integration Points

- **Upstream:** None (data source)
- **Downstream:** `features/` consumes via ORM models

## Implementation Notes

When implementing Phase 2:
1. Start with `models.py` - define ORM before collectors
2. Use `nba_api` library, not raw API calls
3. Implement checkpointing FIRST in pipelines
4. Rate limit: 0.6s between calls, exponential backoff on 429

## Anti-Patterns

- ❌ Never call NBA API directly - always use `NBAApiClient`
- ❌ Never write raw SQL - use SQLAlchemy ORM
- ❌ Never skip checkpointing - pipeline must be resumable
- ❌ Never store API responses without normalization
