# Phase 2c: ETL and Everything Else

## Phase Context

This is the final sub-phase of Phase 2 (Data Collection & Storage). It implements ETL pipelines for orchestrating data collection, stint derivation logic from play-by-play data, CLI command implementations, and integration tests.

**Dependencies:**
- Phase 1: Application Structure (complete)
- Phase 2a: Data Access, Schema and Models (must be complete)
- Phase 2b: Data Collection (must be complete)

**Dependents:**
- Phase 3: Feature Engineering
- All future phases requiring populated database

---

## Objectives

1. Build ETL pipelines with checkpointing, batch processing, and resumability
2. Implement stint derivation logic from play-by-play substitution events
3. Complete CLI command implementations for data operations
4. Create comprehensive integration tests
5. Successfully complete a full historical data load

---

## Task Specifications

### Task 2c.1: Checkpoint Management

**Location:** `nba_model/data/checkpoint.py`

```python
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json

@dataclass
class Checkpoint:
    """Represents pipeline progress checkpoint."""
    pipeline_name: str
    last_game_id: str | None
    last_season: str | None
    total_processed: int
    last_updated: datetime
    status: str  # 'running', 'completed', 'failed'
    error_message: str | None = None

class CheckpointManager:
    """
    Manages pipeline checkpoints for resumable data collection.

    Stores checkpoints as JSON files or in a database table.
    """

    def __init__(self, storage_path: Path | None = None, session: Session | None = None):
        """
        Initialize checkpoint manager.

        Args:
            storage_path: Path for file-based storage (default: data/checkpoints/)
            session: Database session for DB-based storage
        """
        pass

    def save(self, checkpoint: Checkpoint) -> None:
        """Save checkpoint state."""

    def load(self, pipeline_name: str) -> Checkpoint | None:
        """Load checkpoint for pipeline (returns None if not found)."""

    def clear(self, pipeline_name: str) -> None:
        """Clear checkpoint (for fresh start)."""

    def list_all(self) -> list[Checkpoint]:
        """List all checkpoints."""
```

---

### Task 2c.2: ETL Pipeline Orchestration

**Location:** `nba_model/data/pipelines.py`

```python
from enum import Enum

class PipelineStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"

class CollectionPipeline:
    """
    Orchestrates full data collection with checkpointing and batch processing.
    """

    def __init__(
        self,
        session: Session,
        api_client: NBAApiClient,
        checkpoint_manager: CheckpointManager,
        batch_size: int = 50,
    ):
        """
        Initialize pipeline.

        Args:
            session: Database session
            api_client: NBA API client
            checkpoint_manager: Checkpoint storage
            batch_size: Games per batch commit
        """
        pass

    def full_historical_load(
        self,
        seasons: list[str],
        resume: bool = True,
    ) -> PipelineResult:
        """
        Complete historical data collection for specified seasons.

        Order of operations per season:
        1. Collect all games for season
        2. Collect team rosters (populates players)
        3. For each game batch:
           a. Collect play-by-play
           b. Collect shots
           c. Collect box scores
           d. Derive stints
           e. Commit batch
           f. Save checkpoint

        Args:
            seasons: List of season strings (e.g., ["2019-20", "2020-21"])
            resume: Whether to resume from checkpoint

        Returns:
            PipelineResult with statistics
        """

    def incremental_update(self) -> PipelineResult:
        """
        Daily update for new completed games.

        1. Find games completed since last update
        2. Collect all data for new games
        3. Derive stints
        4. Update checkpoint

        Returns:
            PipelineResult with statistics
        """

    def repair_games(self, game_ids: list[str]) -> PipelineResult:
        """
        Re-fetch specific games (for data repair).

        Args:
            game_ids: List of game IDs to re-fetch

        Returns:
            PipelineResult
        """

    def _collect_game_batch(
        self,
        game_ids: list[str],
    ) -> BatchResult:
        """
        Collect all data for a batch of games.

        Collects: plays, shots, box scores, stints
        """

    def _validate_batch(self, batch: BatchResult) -> ValidationResult:
        """
        Validate batch data before commit.

        Checks:
        - Expected row counts
        - Required fields populated
        - Referential integrity
        """

    def _commit_batch(self, batch: BatchResult) -> None:
        """Commit batch to database."""


@dataclass
class PipelineResult:
    """Results from pipeline execution."""
    status: PipelineStatus
    seasons_processed: list[str]
    games_processed: int
    plays_collected: int
    shots_collected: int
    stints_derived: int
    errors: list[str]
    duration_seconds: float


@dataclass
class BatchResult:
    """Results from single batch collection."""
    game_ids: list[str]
    plays: list[Play]
    shots: list[Shot]
    game_stats: list[GameStats]
    player_game_stats: list[PlayerGameStats]
    stints: list[Stint]
    errors: list[tuple[str, str]]  # (game_id, error_message)
```

---

### Task 2c.3: Stint Derivation Logic

**Location:** `nba_model/data/stints.py`

```python
class StintDeriver:
    """
    Derives lineup stints from play-by-play substitution events.

    A stint is a continuous period where the 5-player lineup remains
    unchanged for each team. Stint boundaries occur at:
    - Substitution events (EVENTMSGTYPE = 8)
    - Period start/end events
    - Game start/end
    """

    def __init__(self):
        self.logger = get_logger("StintDeriver")

    def derive_stints(
        self,
        plays: list[Play],
        game_id: str,
    ) -> list[Stint]:
        """
        Derive all stints for a game from play-by-play data.

        Args:
            plays: List of Play objects for the game (sorted by event_num)
            game_id: Game ID for the stints

        Returns:
            List of Stint objects
        """

    def _get_starting_lineups(
        self,
        plays: list[Play],
    ) -> tuple[list[int], list[int]]:
        """
        Determine starting lineups from period start events.

        Uses first 10 distinct player IDs from period 1 events.

        Returns:
            Tuple of (home_lineup, away_lineup) as sorted player ID lists
        """

    def _track_substitutions(
        self,
        plays: list[Play],
        starting_home: list[int],
        starting_away: list[int],
    ) -> list[LineupChange]:
        """
        Track all lineup changes through the game.

        Parses substitution events to track who enters/exits.

        Returns:
            List of LineupChange objects with timestamps
        """

    def _calculate_stint_outcomes(
        self,
        plays: list[Play],
        stint_start: int,  # event_num
        stint_end: int,    # event_num
    ) -> tuple[int, int, float]:
        """
        Calculate points and possessions for a stint.

        Returns:
            Tuple of (home_points, away_points, possessions)
        """

    def _estimate_possessions(
        self,
        plays: list[Play],
        stint_start: int,
        stint_end: int,
    ) -> float:
        """
        Estimate possessions using event counting heuristics.

        Possession ends on:
        - Made field goal (not and-1)
        - Defensive rebound
        - Turnover
        - Made final free throw

        Returns:
            Estimated possession count
        """

    def _parse_time_to_seconds(self, period: int, pc_time: str) -> int:
        """
        Convert period and clock time to total game seconds.

        Args:
            period: Period number (1-4, or 5+ for OT)
            pc_time: Clock time string "MM:SS"

        Returns:
            Total seconds from game start
        """

    def _lineup_to_json(self, player_ids: list[int]) -> str:
        """Convert sorted player ID list to JSON string."""
        return json.dumps(sorted(player_ids))


@dataclass
class LineupChange:
    """Represents a lineup change event."""
    event_num: int
    period: int
    pc_time: str
    team_id: int
    player_in: int
    player_out: int
```

**Edge cases to handle:**

1. **Technical fouls during dead balls:**
   - Don't create new stint for technical FTs
   - Technical FTs don't indicate lineup change

2. **Simultaneous substitutions:**
   - Multiple subs on same dead ball = single stint boundary
   - Group subs by timestamp, apply all before creating new stint

3. **Period transitions:**
   - End current stint at period end
   - Start new stint at period start with potentially different lineup

4. **Overtime periods:**
   - Each OT is 5 minutes (300 seconds)
   - Lineups may change between regulation and OT

5. **Missing player IDs:**
   - Some events may have NULL player IDs
   - Use surrounding events to infer lineup

6. **Ejections:**
   - Player ejection forces substitution
   - Handle as special substitution event

---

### Task 2c.4: CLI Command Implementation

**Location:** `nba_model/cli.py` (modify existing)

Update the placeholder CLI commands with actual implementations:

```python
@data_app.command("collect")
def data_collect(
    seasons: Annotated[
        Optional[list[str]],
        typer.Option("--seasons", "-s", help="Seasons to collect"),
    ] = None,
    full: Annotated[
        bool,
        typer.Option("--full", "-f", help="Collect all 5 seasons"),
    ] = False,
    resume: Annotated[
        bool,
        typer.Option("--resume/--no-resume", help="Resume from checkpoint"),
    ] = True,
) -> None:
    """Collect data from NBA API."""
    settings = get_settings()
    settings.ensure_directories()

    # Determine seasons
    if full:
        seasons_to_collect = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]
    elif seasons:
        seasons_to_collect = seasons
    else:
        console.print("[red]Error: Specify --seasons or --full[/red]")
        raise typer.Exit(1)

    # Initialize components
    with session_scope() as session:
        api_client = NBAApiClient(
            delay=settings.api_delay,
            max_retries=settings.api_max_retries,
        )
        checkpoint_mgr = CheckpointManager()
        pipeline = CollectionPipeline(
            session=session,
            api_client=api_client,
            checkpoint_manager=checkpoint_mgr,
        )

        # Run collection
        with Progress() as progress:
            task = progress.add_task("Collecting data...", total=None)
            result = pipeline.full_historical_load(
                seasons=seasons_to_collect,
                resume=resume,
            )

        # Display results
        _display_pipeline_result(result)


@data_app.command("update")
def data_update() -> None:
    """Incremental data update for recent games."""
    settings = get_settings()

    with session_scope() as session:
        api_client = NBAApiClient(
            delay=settings.api_delay,
            max_retries=settings.api_max_retries,
        )
        checkpoint_mgr = CheckpointManager()
        pipeline = CollectionPipeline(
            session=session,
            api_client=api_client,
            checkpoint_manager=checkpoint_mgr,
        )

        result = pipeline.incremental_update()
        _display_pipeline_result(result)


@data_app.command("status")
def data_status() -> None:
    """Show database statistics and data freshness."""
    settings = get_settings()

    with session_scope() as session:
        stats = _get_database_stats(session)

        table = Table(title="Database Status")
        table.add_column("Entity", style="cyan")
        table.add_column("Count", justify="right")
        table.add_column("Date Range", style="green")

        for entity, count, date_range in stats:
            table.add_row(entity, str(count), date_range or "N/A")

        console.print(table)

        # Show checkpoint status
        checkpoint_mgr = CheckpointManager()
        checkpoints = checkpoint_mgr.list_all()
        if checkpoints:
            console.print("\n[bold]Checkpoints:[/bold]")
            for cp in checkpoints:
                console.print(f"  {cp.pipeline_name}: {cp.status} ({cp.total_processed} games)")


@data_app.command("repair")
def data_repair(
    game_ids: Annotated[
        list[str],
        typer.Argument(help="Game IDs to re-fetch"),
    ],
) -> None:
    """Re-fetch specific game data."""
    settings = get_settings()

    with session_scope() as session:
        api_client = NBAApiClient(
            delay=settings.api_delay,
            max_retries=settings.api_max_retries,
        )
        checkpoint_mgr = CheckpointManager()
        pipeline = CollectionPipeline(
            session=session,
            api_client=api_client,
            checkpoint_manager=checkpoint_mgr,
        )

        result = pipeline.repair_games(game_ids)
        _display_pipeline_result(result)


def _get_database_stats(session: Session) -> list[tuple[str, int, str | None]]:
    """Get row counts and date ranges for each entity."""
    stats = []

    # Games
    game_count = session.query(func.count(Game.game_id)).scalar()
    if game_count > 0:
        min_date = session.query(func.min(Game.game_date)).scalar()
        max_date = session.query(func.max(Game.game_date)).scalar()
        date_range = f"{min_date} to {max_date}"
    else:
        date_range = None
    stats.append(("Games", game_count, date_range))

    # Similar for other entities...
    stats.append(("Players", session.query(func.count(Player.player_id)).scalar(), None))
    stats.append(("Plays", session.query(func.count(Play.id)).scalar(), None))
    stats.append(("Shots", session.query(func.count(Shot.id)).scalar(), None))
    stats.append(("Stints", session.query(func.count(Stint.id)).scalar(), None))

    return stats


def _display_pipeline_result(result: PipelineResult) -> None:
    """Display pipeline result to console."""
    status_color = {
        PipelineStatus.COMPLETED: "green",
        PipelineStatus.FAILED: "red",
        PipelineStatus.RUNNING: "yellow",
    }.get(result.status, "white")

    console.print(f"\n[{status_color}]Status: {result.status.value}[/{status_color}]")
    console.print(f"Games processed: {result.games_processed}")
    console.print(f"Plays collected: {result.plays_collected}")
    console.print(f"Shots collected: {result.shots_collected}")
    console.print(f"Stints derived: {result.stints_derived}")
    console.print(f"Duration: {result.duration_seconds:.1f}s")

    if result.errors:
        console.print(f"\n[red]Errors ({len(result.errors)}):[/red]")
        for error in result.errors[:10]:
            console.print(f"  - {error}")
        if len(result.errors) > 10:
            console.print(f"  ... and {len(result.errors) - 10} more")
```

---

### Task 2c.5: Data Validation Utilities

**Location:** `nba_model/data/validation.py`

```python
from dataclasses import dataclass

@dataclass
class ValidationResult:
    """Result of data validation."""
    valid: bool
    errors: list[str]
    warnings: list[str]

class DataValidator:
    """
    Validates data integrity and quality.
    """

    def validate_game_completeness(
        self,
        session: Session,
        game_id: str,
    ) -> ValidationResult:
        """
        Validate that all expected data exists for a game.

        Checks:
        - Game record exists
        - Play-by-play has expected row count (300-500)
        - Shots count aligns with FGA from box score
        - Stints exist and durations sum to game time
        - Box scores exist for both teams
        """

    def validate_stints(
        self,
        session: Session,
        game_id: str,
    ) -> ValidationResult:
        """
        Validate stint data for a game.

        Checks:
        - No overlapping stints
        - Stint durations sum to game duration
        - All 5 players in each lineup
        - Players exist in player table
        """

    def validate_season_completeness(
        self,
        session: Session,
        season: str,
    ) -> ValidationResult:
        """
        Validate that season data is complete.

        Checks:
        - Expected number of games (~1230 regular season)
        - All games have play-by-play
        - All games have box scores
        - No orphaned records
        """

    def validate_referential_integrity(
        self,
        session: Session,
    ) -> ValidationResult:
        """
        Validate foreign key relationships.

        Checks:
        - All game team_ids exist in teams
        - All player_game_stats player_ids exist in players
        - All shots player_ids exist in players
        """
```

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `nba_model/data/checkpoint.py` | Create | Checkpoint management |
| `nba_model/data/pipelines.py` | Create | ETL pipeline orchestration |
| `nba_model/data/stints.py` | Create | Stint derivation logic |
| `nba_model/data/validation.py` | Create | Data validation utilities |
| `nba_model/cli.py` | Modify | Implement data commands |
| `nba_model/data/__init__.py` | Modify | Export new modules |
| `tests/test_data/test_pipelines.py` | Create | Pipeline tests |
| `tests/test_data/test_stints.py` | Create | Stint derivation tests |
| `tests/test_data/test_integration.py` | Create | Integration tests |
| `tests/fixtures/complete_game/` | Create | Fixture for one complete game |

---

## Testing Requirements

### Unit Tests

**Checkpoint tests (`tests/test_data/test_checkpoint.py`):**
- Test checkpoint save/load cycle
- Test checkpoint not found returns None
- Test checkpoint clear
- Test listing multiple checkpoints

**Pipeline tests (`tests/test_data/test_pipelines.py`):**
- Test batch processing commits at intervals
- Test checkpoint saves after each batch
- Test resume from checkpoint
- Test error handling continues to next game
- Test validation runs before commit
- Mock all collectors and database

**Stint derivation tests (`tests/test_data/test_stints.py`):**
- Test starting lineup detection
- Test substitution tracking
- Test stint boundary detection at period changes
- Test points/possessions calculation
- Test edge cases (techs, simultaneous subs, OT)
- Use fixture data from real game

**Validation tests (`tests/test_data/test_validation.py`):**
- Test game completeness validation
- Test stint overlap detection
- Test season completeness checks
- Test referential integrity checks

### Integration Tests (`tests/test_data/test_integration.py`)

**Single game end-to-end:**
```python
def test_single_game_collection(
    db_session: Session,
    mock_api_client: NBAApiClient,
):
    """
    End-to-end test: collect one game and verify all data.

    Uses mock API returning fixture data for game 0022300001.
    Verifies:
    - Game record created
    - ~400 plays created
    - ~80 shots created
    - Box scores created
    - Stints derived correctly
    - All relationships valid
    """
```

**Incremental update test:**
```python
def test_incremental_update(
    db_session: Session,
    mock_api_client: NBAApiClient,
):
    """
    Test incremental update finds and fetches new games.

    Setup: Populate DB with games through date X
    Execute: Run incremental update
    Verify: Only games after date X fetched
    """
```

**Resume from checkpoint test:**
```python
def test_resume_from_checkpoint(
    db_session: Session,
    mock_api_client: NBAApiClient,
):
    """
    Test pipeline resumes correctly from checkpoint.

    Setup: Start collection, interrupt after N games
    Execute: Resume collection
    Verify: Continues from checkpoint, no duplicate work
    """
```

### Test Fixtures

**Complete game fixture (`tests/fixtures/complete_game/`):**
```
tests/fixtures/complete_game/
├── game_info.json          # Game metadata
├── playbyplay.json         # ~400 play events
├── shots.json              # ~80 shots
├── boxscore_team.json      # Team stats
├── boxscore_player.json    # Player stats
└── expected_stints.json    # Expected stint output for validation
```

Select a game with known stint data for validation. Preferably:
- Regular season game (no OT initially)
- Multiple substitutions per period
- Clean data (no postponements, ejections)

---

## Success Criteria

1. Full historical load completes for 1 season without manual intervention
2. Incremental update correctly identifies only new games
3. Pipeline resumes correctly from checkpoint after failure
4. Stint derivation produces valid lineups matching external data
5. All stints have durations summing to game time
6. No overlapping stints within game/team
7. Validation catches data quality issues
8. CLI commands display progress and results clearly
9. Test coverage >= 90% for all new code
10. Integration tests pass with fixture data

---

## Implementation Checklist

### Checkpoint Management
- [ ] Create `nba_model/data/checkpoint.py`
- [ ] Implement Checkpoint dataclass
- [ ] Implement CheckpointManager with file storage
- [ ] Add save/load/clear/list methods
- [ ] Create checkpoint tests

### Pipeline Orchestration
- [ ] Create `nba_model/data/pipelines.py`
- [ ] Implement PipelineStatus enum
- [ ] Implement PipelineResult dataclass
- [ ] Implement BatchResult dataclass
- [ ] Implement CollectionPipeline class
- [ ] Implement full_historical_load()
- [ ] Implement incremental_update()
- [ ] Implement repair_games()
- [ ] Implement batch collection logic
- [ ] Implement batch validation
- [ ] Implement batch commit
- [ ] Add progress logging
- [ ] Create pipeline tests

### Stint Derivation
- [ ] Create `nba_model/data/stints.py`
- [ ] Implement LineupChange dataclass
- [ ] Implement StintDeriver class
- [ ] Implement derive_stints()
- [ ] Implement starting lineup detection
- [ ] Implement substitution tracking
- [ ] Implement stint outcome calculation
- [ ] Implement possession estimation
- [ ] Handle edge cases (techs, simultaneous subs)
- [ ] Handle period transitions
- [ ] Handle overtime
- [ ] Create stint derivation tests

### Data Validation
- [ ] Create `nba_model/data/validation.py`
- [ ] Implement ValidationResult dataclass
- [ ] Implement DataValidator class
- [ ] Implement game completeness validation
- [ ] Implement stint validation
- [ ] Implement season completeness validation
- [ ] Implement referential integrity validation
- [ ] Create validation tests

### CLI Commands
- [ ] Implement data_collect command
- [ ] Implement data_update command
- [ ] Implement data_status command
- [ ] Implement data_repair command
- [ ] Add progress display with rich
- [ ] Add result display formatting
- [ ] Test CLI commands

### Package Integration
- [ ] Update `nba_model/data/__init__.py`
- [ ] Verify all imports work

### Testing
- [ ] Create test fixtures for complete game
- [ ] Create `tests/test_data/test_checkpoint.py`
- [ ] Create `tests/test_data/test_pipelines.py`
- [ ] Create `tests/test_data/test_stints.py`
- [ ] Create `tests/test_data/test_validation.py`
- [ ] Create `tests/test_data/test_integration.py`
- [ ] Achieve 90%+ test coverage
- [ ] All tests pass

### Final Validation
- [ ] Run full collection for one season (with live API)
- [ ] Verify data volume matches estimates
- [ ] Validate stint data quality
- [ ] Document any API quirks discovered

---

## Data Volume Estimates (Verification)

After completing one season collection, verify:

| Entity | Expected | Actual |
|--------|----------|--------|
| Games | ~1,300 | ? |
| Plays | ~500,000 | ? |
| Shots | ~100,000 | ? |
| PlayerGameStats | ~40,000 | ? |
| Stints | ~50,000 | ? |
| DB Size | ~200 MB | ? |

---

## Completion Instructions

Upon completion:
1. Run full test suite: `pytest tests/test_data/ -v --cov=nba_model.data`
2. Verify coverage >= 90%
3. Run type checking: `mypy nba_model/data/`
4. Run linting: `ruff check nba_model/data/`
5. Run full historical load for one season (live API test)
6. Document actual data volumes vs estimates
7. Commit with message: `feat(data): implement Phase 2c - ETL pipelines and stint derivation`
8. Update CLAUDE.md Phase 2 status to Complete

---

## Phase 2 Completion Summary

After all three sub-phases are complete:

**Phase 2a deliverables:**
- Database engine and session management
- All SQLAlchemy models (14 tables)
- Indexes and constraints

**Phase 2b deliverables:**
- NBA API client with rate limiting and retries
- 5 data collectors (games, players, playbyplay, shots, boxscores)
- Test fixtures for all API endpoints

**Phase 2c deliverables:**
- ETL pipeline with checkpointing
- Stint derivation logic
- CLI command implementations
- Integration tests
- Validated data for at least one season

**Final Phase 2 commit:**
After all sub-phases complete, create a summary commit:
`feat: complete Phase 2 - data collection and storage`
