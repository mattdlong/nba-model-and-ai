# Phase 2a: Data Access, Schema and Models

## Phase Context

This is the first sub-phase of Phase 2 (Data Collection & Storage). It establishes the database foundation required by all subsequent data collection and storage tasks. Phase 2b (Data Collection) and Phase 2c (ETL and Everything Else) depend on this phase being complete.

**Dependencies:**
- Phase 1: Application Structure (complete)

**Dependents:**
- Phase 2b: Data Collection
- Phase 2c: ETL and Everything Else
- All future phases requiring database access

---

## Objectives

1. Implement the SQLite database schema using SQLAlchemy ORM
2. Create all SQLAlchemy model classes with proper relationships
3. Establish database engine and session management utilities
4. Implement database initialization and migration support
5. Achieve 90%+ test coverage on all schema and model code

---

## Task Specifications

### Task 2a.1: Database Engine and Session Management

**Location:** `nba_model/data/db.py`

Create database connection utilities:

```python
# Required exports
def get_engine() -> Engine:
    """Get SQLAlchemy engine from settings."""

def get_session() -> Session:
    """Get a new database session."""

@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Context manager for session with auto-commit/rollback."""

def init_db() -> None:
    """Initialize database with all tables."""
```

**Requirements:**
- Engine configured from `settings.db_path`
- Sessions use scoped_session pattern for thread safety
- Auto-create parent directories for database file
- Support for SQLite-specific pragmas (foreign keys, WAL mode)
- Logging of all database operations at DEBUG level

---

### Task 2a.2: SQLAlchemy Base and Mixins

**Location:** `nba_model/data/schema.py`

Create shared base class and mixins:

```python
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""
    pass

class TimestampMixin:
    """Mixin adding created_at and updated_at columns."""
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]
```

**Requirements:**
- Use SQLAlchemy 2.0 style with `Mapped[]` type hints
- Timestamp mixin with auto-set on insert/update
- Common validation constraints as reusable types

---

### Task 2a.3: Core Reference Models

**Location:** `nba_model/data/models.py`

Implement reference table models:

**Season Model:**
```python
class Season(Base):
    __tablename__ = "seasons"

    season_id: Mapped[str]          # "2023-24" format, PRIMARY KEY
    start_date: Mapped[date]
    end_date: Mapped[date]
    games_count: Mapped[int | None]

    # Relationships
    games: Mapped[list["Game"]] = relationship(back_populates="season")
```

**Team Model:**
```python
class Team(Base):
    __tablename__ = "teams"

    team_id: Mapped[int]            # NBA team ID, PRIMARY KEY
    abbreviation: Mapped[str]       # "LAL", "BOS", etc.
    full_name: Mapped[str]
    city: Mapped[str]
    arena_name: Mapped[str | None]
    arena_lat: Mapped[float | None] # For travel distance calculations
    arena_lon: Mapped[float | None]
```

**Player Model:**
```python
class Player(Base):
    __tablename__ = "players"

    player_id: Mapped[int]          # NBA player ID, PRIMARY KEY
    full_name: Mapped[str]
    height_inches: Mapped[int | None]
    weight_lbs: Mapped[int | None]
    birth_date: Mapped[date | None]
    draft_year: Mapped[int | None]
    draft_round: Mapped[int | None]
    draft_number: Mapped[int | None]
```

**PlayerSeason Model:**
```python
class PlayerSeason(Base):
    __tablename__ = "player_seasons"

    id: Mapped[int]                 # Auto-increment PRIMARY KEY
    player_id: Mapped[int]          # FK to players
    season_id: Mapped[str]          # FK to seasons
    team_id: Mapped[int]            # FK to teams
    position: Mapped[str | None]
    jersey_number: Mapped[str | None]

    # UNIQUE constraint on (player_id, season_id, team_id)
```

---

### Task 2a.4: Game Models

**Location:** `nba_model/data/models.py`

**Game Model:**
```python
class Game(Base):
    __tablename__ = "games"

    game_id: Mapped[str]            # NBA GAME_ID format, PRIMARY KEY
    season_id: Mapped[str]          # FK to seasons
    game_date: Mapped[date]
    home_team_id: Mapped[int]       # FK to teams
    away_team_id: Mapped[int]       # FK to teams
    home_score: Mapped[int | None]
    away_score: Mapped[int | None]
    status: Mapped[str]             # 'scheduled', 'completed', 'postponed'
    attendance: Mapped[int | None]

    # Relationships
    season: Mapped["Season"] = relationship(back_populates="games")
    home_team: Mapped["Team"] = relationship(foreign_keys=[home_team_id])
    away_team: Mapped["Team"] = relationship(foreign_keys=[away_team_id])
    plays: Mapped[list["Play"]] = relationship(back_populates="game")
    shots: Mapped[list["Shot"]] = relationship(back_populates="game")
    stints: Mapped[list["Stint"]] = relationship(back_populates="game")
```

**GameStats Model:**
```python
class GameStats(Base):
    __tablename__ = "game_stats"

    id: Mapped[int]                 # Auto-increment PRIMARY KEY
    game_id: Mapped[str]            # FK to games
    team_id: Mapped[int]            # FK to teams
    is_home: Mapped[bool]

    # Basic stats
    points: Mapped[int | None]
    rebounds: Mapped[int | None]
    assists: Mapped[int | None]
    steals: Mapped[int | None]
    blocks: Mapped[int | None]
    turnovers: Mapped[int | None]

    # Advanced stats (from boxscoreadvancedv2)
    offensive_rating: Mapped[float | None]
    defensive_rating: Mapped[float | None]
    pace: Mapped[float | None]
    efg_pct: Mapped[float | None]
    tov_pct: Mapped[float | None]
    orb_pct: Mapped[float | None]
    ft_rate: Mapped[float | None]

    # UNIQUE constraint on (game_id, team_id)
```

**PlayerGameStats Model:**
```python
class PlayerGameStats(Base):
    __tablename__ = "player_game_stats"

    id: Mapped[int]                 # Auto-increment PRIMARY KEY
    game_id: Mapped[str]            # FK to games
    player_id: Mapped[int]          # FK to players
    team_id: Mapped[int]            # FK to teams

    # Box score stats
    minutes: Mapped[float | None]
    points: Mapped[int | None]
    rebounds: Mapped[int | None]
    assists: Mapped[int | None]
    steals: Mapped[int | None]
    blocks: Mapped[int | None]
    turnovers: Mapped[int | None]
    fgm: Mapped[int | None]
    fga: Mapped[int | None]
    fg3m: Mapped[int | None]
    fg3a: Mapped[int | None]
    ftm: Mapped[int | None]
    fta: Mapped[int | None]
    plus_minus: Mapped[int | None]

    # Player tracking (from boxscoreplayertrackv2)
    distance_miles: Mapped[float | None]
    speed_avg: Mapped[float | None]

    # UNIQUE constraint on (game_id, player_id)
```

---

### Task 2a.5: Play-by-Play Models

**Location:** `nba_model/data/models.py`

**Play Model:**
```python
class Play(Base):
    __tablename__ = "plays"

    id: Mapped[int]                   # Auto-increment PRIMARY KEY
    game_id: Mapped[str]              # FK to games
    event_num: Mapped[int]
    period: Mapped[int]
    pc_time: Mapped[str | None]       # "MM:SS" format
    wc_time: Mapped[str | None]       # Wall clock time
    event_type: Mapped[int]           # EVENTMSGTYPE
    event_action: Mapped[int | None]  # EVENTMSGACTIONTYPE
    home_description: Mapped[str | None]
    away_description: Mapped[str | None]
    neutral_description: Mapped[str | None]
    score_home: Mapped[int | None]
    score_away: Mapped[int | None]
    player1_id: Mapped[int | None]    # FK to players (nullable)
    player2_id: Mapped[int | None]
    player3_id: Mapped[int | None]
    team_id: Mapped[int | None]       # FK to teams (nullable)

    # Relationships
    game: Mapped["Game"] = relationship(back_populates="plays")

    # UNIQUE constraint on (game_id, event_num)
```

**Shot Model:**
```python
class Shot(Base):
    __tablename__ = "shots"

    id: Mapped[int]                   # Auto-increment PRIMARY KEY
    game_id: Mapped[str]              # FK to games
    player_id: Mapped[int]            # FK to players
    team_id: Mapped[int]              # FK to teams
    period: Mapped[int]
    minutes_remaining: Mapped[int]
    seconds_remaining: Mapped[int]
    action_type: Mapped[str | None]
    shot_type: Mapped[str | None]     # '2PT', '3PT'
    shot_zone_basic: Mapped[str | None]
    shot_zone_area: Mapped[str | None]
    shot_zone_range: Mapped[str | None]
    shot_distance: Mapped[int | None]
    loc_x: Mapped[int]                # Court coordinates
    loc_y: Mapped[int]
    made: Mapped[bool]

    # Relationships
    game: Mapped["Game"] = relationship(back_populates="shots")

    # UNIQUE constraint on (game_id, player_id, period, minutes_remaining, seconds_remaining, loc_x, loc_y)
```

---

### Task 2a.6: Derived and Feature Models

**Location:** `nba_model/data/models.py`

**Stint Model:**
```python
class Stint(Base):
    __tablename__ = "stints"

    id: Mapped[int]                   # Auto-increment PRIMARY KEY
    game_id: Mapped[str]              # FK to games
    period: Mapped[int]
    start_time: Mapped[str]           # "MM:SS" format
    end_time: Mapped[str]
    duration_seconds: Mapped[int]
    home_lineup: Mapped[str]          # JSON array of 5 player IDs
    away_lineup: Mapped[str]          # JSON array of 5 player IDs
    home_points: Mapped[int]
    away_points: Mapped[int]
    possessions: Mapped[float | None]

    # Relationships
    game: Mapped["Game"] = relationship(back_populates="stints")
```

**Odds Model:**
```python
class Odds(Base):
    __tablename__ = "odds"

    id: Mapped[int]                   # Auto-increment PRIMARY KEY
    game_id: Mapped[str]              # FK to games
    source: Mapped[str]               # 'pinnacle', 'draftkings', etc.
    timestamp: Mapped[datetime]
    home_ml: Mapped[float | None]     # Money line decimal odds
    away_ml: Mapped[float | None]
    spread_home: Mapped[float | None]
    spread_home_odds: Mapped[float | None]
    spread_away_odds: Mapped[float | None]
    total: Mapped[float | None]
    over_odds: Mapped[float | None]
    under_odds: Mapped[float | None]
```

**PlayerRAPM Model (for Phase 3):**
```python
class PlayerRAPM(Base):
    __tablename__ = "player_rapm"

    id: Mapped[int]                   # Auto-increment PRIMARY KEY
    player_id: Mapped[int]            # FK to players
    season_id: Mapped[str]            # FK to seasons
    calculation_date: Mapped[date]
    orapm: Mapped[float]              # Offensive RAPM
    drapm: Mapped[float]              # Defensive RAPM
    rapm: Mapped[float]               # Total RAPM
    sample_stints: Mapped[int]

    # UNIQUE constraint on (player_id, season_id, calculation_date)
```

**LineupSpacing Model (for Phase 3):**
```python
class LineupSpacing(Base):
    __tablename__ = "lineup_spacing"

    id: Mapped[int]                   # Auto-increment PRIMARY KEY
    season_id: Mapped[str]            # FK to seasons
    lineup_hash: Mapped[str]          # Hash of sorted 5 player IDs
    player_ids: Mapped[str]           # JSON array of 5 player IDs
    hull_area: Mapped[float]
    centroid_x: Mapped[float]
    centroid_y: Mapped[float]
    shot_count: Mapped[int]

    # UNIQUE constraint on (season_id, lineup_hash)
```

**SeasonStats Model (for Phase 3):**
```python
class SeasonStats(Base):
    __tablename__ = "season_stats"

    id: Mapped[int]                   # Auto-increment PRIMARY KEY
    season_id: Mapped[str]            # FK to seasons
    metric_name: Mapped[str]
    mean_value: Mapped[float]
    std_value: Mapped[float]
    min_value: Mapped[float]
    max_value: Mapped[float]

    # UNIQUE constraint on (season_id, metric_name)
```

---

### Task 2a.7: Index Definitions

**Location:** `nba_model/data/models.py`

Create indexes for frequently queried columns:

```python
# Add to appropriate model classes using __table_args__

class Game(Base):
    __table_args__ = (
        Index('idx_games_date', 'game_date'),
        Index('idx_games_season', 'season_id'),
    )

class Play(Base):
    __table_args__ = (
        UniqueConstraint('game_id', 'event_num', name='uq_play_game_event'),
        Index('idx_plays_game', 'game_id'),
    )

class Shot(Base):
    __table_args__ = (
        Index('idx_shots_game', 'game_id'),
        Index('idx_shots_player', 'player_id'),
    )

class Stint(Base):
    __table_args__ = (
        Index('idx_stints_game', 'game_id'),
    )

class PlayerGameStats(Base):
    __table_args__ = (
        UniqueConstraint('game_id', 'player_id', name='uq_player_game'),
        Index('idx_player_game_stats', 'game_id', 'player_id'),
    )
```

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `nba_model/data/db.py` | Create | Database engine and session management |
| `nba_model/data/schema.py` | Create | Base class and mixins |
| `nba_model/data/models.py` | Create | All SQLAlchemy model classes |
| `nba_model/data/__init__.py` | Modify | Export public API |
| `tests/test_data/test_db.py` | Create | Database utility tests |
| `tests/test_data/test_schema.py` | Create | Schema and model tests |

---

## Testing Requirements

### Unit Tests (`tests/test_data/test_db.py`)

**Database engine tests:**
- Verify engine creates from settings
- Test session scope context manager commits on success
- Test session scope rolls back on exception
- Verify database file created in correct location
- Test init_db creates all tables

**Configuration tests:**
- Test engine uses settings.db_path
- Verify SQLite pragmas applied (foreign_keys, WAL mode)

### Schema Tests (`tests/test_data/test_schema.py`)

**Table creation tests:**
- Verify all 14 tables create successfully
- Confirm column types match specifications
- Validate nullable/non-nullable constraints

**Foreign key tests:**
- Test FK constraints prevent orphaned records
- Verify cascade behavior where appropriate

**Unique constraint tests:**
- Test UNIQUE constraints reject duplicates
- Verify INSERT OR REPLACE/upsert patterns work

**Index tests:**
- Confirm all specified indexes created
- Validate index naming conventions

### Model Tests (`tests/test_data/test_models.py`)

**Relationship tests:**
- Test Season -> Games navigation
- Test Game -> Plays, Shots, Stints navigation
- Verify back_populates works bidirectionally

**Data integrity tests:**
- Test inserting valid records
- Verify constraint violations raise appropriate errors
- Test JSON array storage in lineup fields

---

## Success Criteria

1. All 14 tables create successfully with `init_db()`
2. All foreign key relationships enforced correctly
3. UNIQUE constraints prevent duplicate inserts
4. Indexes created on game_date, season_id, game_id, player_id columns
5. Session management handles commits and rollbacks correctly
6. Test coverage >= 90% for all schema and model code
7. All tests pass in CI/CD pipeline

---

## Implementation Checklist

### Database Infrastructure
- [ ] Create `nba_model/data/db.py` with engine and session utilities
- [ ] Implement `get_engine()` with SQLite configuration
- [ ] Implement `get_session()` with proper scoping
- [ ] Implement `session_scope()` context manager
- [ ] Implement `init_db()` for table creation
- [ ] Add SQLite pragmas for foreign keys and WAL mode

### Schema and Base
- [ ] Create `nba_model/data/schema.py` with Base class
- [ ] Implement TimestampMixin with auto-update

### Models - Core Reference
- [ ] Implement Season model
- [ ] Implement Team model
- [ ] Implement Player model
- [ ] Implement PlayerSeason model with unique constraint

### Models - Games
- [ ] Implement Game model with relationships
- [ ] Implement GameStats model
- [ ] Implement PlayerGameStats model

### Models - Play-by-Play
- [ ] Implement Play model
- [ ] Implement Shot model with coordinate fields

### Models - Derived/Feature
- [ ] Implement Stint model with JSON lineup fields
- [ ] Implement Odds model
- [ ] Implement PlayerRAPM model (stub for Phase 3)
- [ ] Implement LineupSpacing model (stub for Phase 3)
- [ ] Implement SeasonStats model (stub for Phase 3)

### Indexes and Constraints
- [ ] Add indexes to Game model
- [ ] Add indexes to Play model
- [ ] Add indexes to Shot model
- [ ] Add indexes to Stint model
- [ ] Add indexes to PlayerGameStats model
- [ ] Define all UNIQUE constraints

### Package Integration
- [ ] Update `nba_model/data/__init__.py` with exports
- [ ] Verify imports work correctly

### Testing
- [ ] Create `tests/test_data/test_db.py`
- [ ] Create `tests/test_data/test_schema.py`
- [ ] Create `tests/test_data/test_models.py`
- [ ] Achieve 90%+ test coverage
- [ ] All tests pass

---

## Completion Instructions

Upon completion:
1. Run full test suite: `pytest tests/test_data/ -v --cov=nba_model.data`
2. Verify coverage >= 90%
3. Run type checking: `mypy nba_model/data/`
4. Run linting: `ruff check nba_model/data/`
5. Commit with message: `feat(data): implement Phase 2a - database schema and models`
