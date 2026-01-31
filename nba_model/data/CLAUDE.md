# Data Layer

## Responsibility

Owns all data acquisition, storage, and retrieval for NBA statistics. Single source of truth for game data, player stats, play-by-play, and shot charts.

## Status

🔲 **Phase 2 - Not Started** (stub `__init__.py` only)

## Planned Structure

| File | Purpose | Status |
|------|---------|--------|
| `__init__.py` | Public API exports | ✅ Exists (stub) |
| `api.py` | NBA API client with rate limiting | 🔲 Planned |
| `models.py` | SQLAlchemy ORM models | 🔲 Planned |
| `schema.py` | Database schema reference | 🔲 Planned |
| `pipelines.py` | ETL orchestration | 🔲 Planned |
| `collectors/` | Entity-specific collectors | 🔲 Planned |

## Planned Key Classes

- `NBAApiClient` - Rate-limited API wrapper (0.6s delay, 3 retries)
- `Game`, `Player`, `Stint`, `Shot` - SQLAlchemy ORM models
- `DataPipeline` - Orchestrates collection with checkpointing

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
