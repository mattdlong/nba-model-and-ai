# Phase 2: Data Collection & Storage

## Phase Context

This phase implements the data layer for the NBA quantitative trading system. It depends on Phase 1 (Application Structure) providing CLI framework, configuration management, and project layout. All subsequent phases depend on the data infrastructure established here.

---

## Objectives

1. Design and implement a comprehensive SQLite schema capturing all NBA data entities required for feature engineering and model training
2. Build a robust NBA API client wrapper with rate limiting, retry logic, and error handling
3. Implement modular data collectors for games, players, play-by-play, shots, and box scores
4. Create ETL pipelines supporting checkpointing, batch processing, and graceful resumption
5. Enable incremental daily updates for production operation

---

## Task Specifications

### Task 2.1: SQLite Schema Implementation

**Location:** `nba_model/data/schema.py` and `nba_model/data/models.py`

**Entities to model:**

Core reference tables:
- `seasons`: Season identifiers with date ranges and game counts
- `teams`: Team metadata including arena coordinates (latitude/longitude) for travel calculations
- `players`: Player biographical data including height, weight, birth date, draft information
- `player_seasons`: Junction table linking players to teams by season with position data

Game tables:
- `games`: Primary game records with home/away teams, scores, status, attendance
- `game_stats`: Team-level per-game statistics including advanced metrics (offensive/defensive rating, pace, four factors)
- `player_game_stats`: Individual player box score data including tracking metrics (distance, speed)

Play-by-play tables:
- `plays`: Event-level records with event types, descriptions, timestamps, player references, running score
- `shots`: Shot chart data with court coordinates (loc_x, loc_y), zone classifications, make/miss

Derived tables:
- `stints`: Lineup tracking derived from play-by-play substitution events; stores 5-player lineup arrays for home/away, stint duration, points scored, possessions

Market data:
- `odds`: Historical odds from various sources with money line, spread, and totals

Feature tables (populated by Phase 3):
- `player_rapm`: RAPM coefficients by player/season
- `lineup_spacing`: Convex hull metrics by lineup
- `season_stats`: Normalization parameters (mean/std) by metric by season

**Schema requirements:**
- Use SQLAlchemy ORM with declarative base
- Implement appropriate foreign key relationships
- Create indexes on frequently queried columns: game_date, season_id, game_id, player_id
- Use UNIQUE constraints to enable idempotent upserts
- Store lineup arrays as JSON strings within TEXT columns

---

### Task 2.2: NBA API Client Wrapper

**Location:** `nba_model/data/api.py`

**Wrapper requirements:**

Rate limiting:
- Configurable delay between API calls (default 0.6 seconds)
- Respect NBA API rate limits to avoid IP blocking

Retry logic:
- Exponential backoff on transient failures
- Maximum retry count (default 3)
- Distinguish retriable vs permanent errors

Error handling:
- Graceful handling of missing data (some historical endpoints return empty)
- Timeout handling for slow responses
- Structured logging of all API interactions

**Endpoints to wrap:**
- `leaguegamefinder`: Season game listings
- `playbyplayv2`: Play-by-play event data
- `shotchartdetail`: Shot location data
- `boxscoreadvancedv2`: Advanced team and player statistics
- `boxscoreplayertrackv2`: Player tracking metrics (distance, speed)
- `commonteamroster`: Team roster by season

Each wrapper method returns pandas DataFrame with standardized column naming.

---

### Task 2.3: Data Collectors

**Location:** `nba_model/data/collectors/`

**Collector modules:**

`games.py` - GamesCollector:
- Fetch all games for specified season range
- Handle regular season and playoffs separately
- Track game status (completed, scheduled, postponed)

`players.py` - PlayersCollector:
- Build player master table from roster data
- Handle mid-season trades (player appears on multiple teams)
- Track biographical data updates

`playbyplay.py` - PlayByPlayCollector:
- Fetch play-by-play for each completed game
- Parse event types and action types
- Extract player references from event descriptions
- Derive stint records from substitution events

`shots.py` - ShotsCollector:
- Fetch shot chart data per game or per player/season
- Normalize court coordinates
- Classify shot zones

`boxscores.py` - BoxScoreCollector:
- Fetch advanced box scores (team and player level)
- Fetch player tracking data where available
- Handle missing tracking data for older seasons

**Collector interface:**
Each collector implements:
- `collect(season_range, resume_from=None)`: Full collection with optional resume point
- `collect_game(game_id)`: Single game collection
- `get_last_checkpoint()`: Return last successfully processed identifier

---

### Task 2.4: ETL Pipeline Orchestration

**Location:** `nba_model/data/pipelines.py`

**Pipeline features:**

Checkpointing:
- Persist last successful game_id after each batch
- Enable resumption from checkpoint on failure
- Store checkpoints in database or local file

Batch processing:
- Process and commit games in configurable batch sizes (default 50)
- Limit memory usage during large historical loads
- Progress reporting via tqdm or similar

Validation:
- Schema validation before database insert
- Data quality checks (null detection, range validation)
- Logging of validation failures without halting pipeline

Deduplication:
- Rely on UNIQUE constraints for idempotent behavior
- Use INSERT OR REPLACE / ON CONFLICT patterns
- Log duplicate detection statistics

**Pipeline commands:**
- `full_historical_load(seasons)`: Complete historical data collection
- `incremental_update()`: Daily update fetching new completed games
- `repair(game_ids)`: Re-fetch specific games

---

### Task 2.5: Stint Derivation Logic

**Location:** Within `nba_model/data/collectors/playbyplay.py` or separate `stints.py`

**Derivation requirements:**

Parse substitution events from play-by-play to track which 5 players are on court for each team at any moment.

Stint definition:
- A stint is a continuous period where lineup remains unchanged
- Stint boundaries: substitutions, period start/end, game start/end

For each stint, compute:
- Duration in seconds
- Points scored by each team during stint
- Approximate possessions (using event counting heuristics)

Store lineups as sorted JSON arrays of 5 player IDs for consistent hashing.

This is complex logic requiring careful handling of edge cases:
- Technical fouls during dead balls
- Simultaneous substitutions
- Period transitions
- Overtime periods

---

## CLI Commands

Implement under `nba-model data` subcommand group:

- `nba-model data collect --seasons 2019-20,2020-21,...` - Full historical collection
- `nba-model data update` - Incremental daily update
- `nba-model data status` - Display database statistics (row counts, date ranges, collection progress)
- `nba-model data repair --game-id XXXX` - Re-fetch specific game data

---

## Success Criteria

1. Schema correctly models all required entities with appropriate relationships and indexes
2. API client successfully fetches data from all required endpoints with proper rate limiting
3. Full historical load completes for 5 seasons (2019-20 through 2023-24) without manual intervention
4. Incremental update correctly identifies and fetches only new completed games
5. Stint derivation produces valid lineup records matching known lineup data
6. Database size approximately matches estimates (~200MB per season)
7. All collectors resume correctly from checkpoints after simulated failures

---

## Testing Requirements

### Unit Tests

**Schema tests (`tests/test_data/test_schema.py`):**
- Verify all tables create successfully
- Test foreign key constraints prevent orphaned records
- Confirm UNIQUE constraints reject duplicates appropriately
- Validate index creation

**API client tests (`tests/test_data/test_api.py`):**
- Mock NBA API responses to test parsing logic
- Verify retry logic triggers on transient errors
- Confirm rate limiting delays are applied
- Test timeout handling

**Collector tests (`tests/test_data/test_collectors.py`):**
- Test each collector with mocked API responses
- Verify checkpoint save/restore functionality
- Confirm batch processing commits at correct intervals
- Test handling of missing/partial data

### Integration Tests

**Database integration (`tests/test_data/test_integration.py`):**
- End-to-end test: fetch single game, store all related data, verify relationships
- Test incremental update identifies correct games to fetch
- Verify stint derivation produces expected lineup records for known game

**API integration (optional, rate-limited):**
- Smoke test against live NBA API for one recent game
- Verify response format matches expected schema

### Data Validation Tests

**Stint validation:**
- Compare derived stints against external lineup data source if available
- Verify stint durations sum to game duration
- Confirm no overlapping stints within same game/team

**Data completeness:**
- Verify expected number of plays per game (typically 300-500 events)
- Confirm shot counts align with box score field goal attempts
- Check player minutes sum appropriately

### Test Fixtures

Create fixtures in `tests/fixtures/`:
- Sample API responses for each endpoint (JSON)
- Pre-populated test database with one complete game
- Edge case data (overtime games, postponed games, missing tracking data)

---

## Data Volume Estimates

| Entity | Rows per Season | Storage per Season |
|--------|-----------------|-------------------|
| games | ~1,300 | ~100 KB |
| plays | ~500,000 | ~50 MB |
| shots | ~100,000 | ~10 MB |
| player_game_stats | ~40,000 | ~5 MB |
| stints | ~50,000 | ~5 MB |

Total per season: ~70 MB (plus indexes ~130 MB overhead)
Total for 5 seasons: ~1 GB

---

## Error Handling Specifications

**Retriable errors (apply exponential backoff):**
- HTTP 429 (rate limited)
- HTTP 500, 502, 503, 504 (server errors)
- Connection timeouts
- DNS resolution failures

**Non-retriable errors (log and skip):**
- HTTP 404 (game not found - may be future game)
- HTTP 400 (bad request - likely invalid game_id format)
- Empty response for historical data (endpoint not available for old games)

**Pipeline failure modes:**
- Single game failure: log error, continue to next game, record failed game_id for retry
- Batch failure: rollback current batch, save checkpoint at last successful batch, exit with error code
- API unavailable: retry with increasing delays, exit after max retries exceeded

---

## Dependencies

**Phase 1 requirements:**
- Typer CLI framework initialized
- SQLAlchemy engine configuration via Pydantic settings
- Loguru logging configured
- Database path configurable via `NBA_DB_PATH` environment variable

**External dependencies:**
- `nba_api` library (v1.4+)
- `sqlalchemy` (v2.0+)
- `pandas` for DataFrame handling
- `tqdm` for progress bars

---

## Completion Instructions

Upon completion, commit all changes with message "feat: implement phase 2 - data collection and storage" and push to origin.
