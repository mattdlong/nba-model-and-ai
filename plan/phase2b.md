# Phase 2b: Data Collection

## Phase Context

This is the second sub-phase of Phase 2 (Data Collection & Storage). It implements the NBA API client wrapper and all data collectors for fetching data from the NBA API. Phase 2c (ETL and Everything Else) depends on this phase for its ETL pipelines.

**Dependencies:**
- Phase 1: Application Structure (complete)
- Phase 2a: Data Access, Schema and Models (must be complete)

**Dependents:**
- Phase 2c: ETL and Everything Else
- All future phases requiring collected data

---

## Objectives

1. Build a robust NBA API client wrapper with rate limiting, retry logic, and error handling
2. Implement modular data collectors for games, players, play-by-play, shots, and box scores
3. Standardize data transformation from API responses to database-ready formats
4. Achieve 90%+ test coverage on all collection code (using mocked API responses)

---

## Task Specifications

### Task 2b.1: NBA API Client Wrapper

**Location:** `nba_model/data/api.py`

Create a wrapper around the `nba_api` library with reliability features:

```python
class NBAApiClient:
    """
    Wrapper around nba_api with:
    - Automatic rate limiting (configurable delay)
    - Retry logic with exponential backoff
    - Response caching (optional)
    - Error handling and logging
    """

    def __init__(
        self,
        delay: float = 0.6,
        max_retries: int = 3,
        timeout: float = 30.0,
    ):
        """
        Initialize API client.

        Args:
            delay: Seconds between API calls (default 0.6)
            max_retries: Maximum retry attempts for transient errors
            timeout: Request timeout in seconds
        """
        pass
```

**Required methods:**

```python
def get_league_game_finder(
    self,
    season: str,
    season_type: str = "Regular Season",
    **kwargs
) -> pd.DataFrame:
    """
    Fetch games for a season using LeagueGameFinder.

    Args:
        season: Season string (e.g., "2023-24")
        season_type: "Regular Season" or "Playoffs"

    Returns:
        DataFrame with game records
    """

def get_play_by_play(self, game_id: str) -> pd.DataFrame:
    """
    Fetch play-by-play using PlayByPlayV2.

    Args:
        game_id: NBA game ID string

    Returns:
        DataFrame with play events
    """

def get_shot_chart(
    self,
    game_id: str | None = None,
    player_id: int | None = None,
    season: str | None = None,
) -> pd.DataFrame:
    """
    Fetch shot chart using ShotChartDetail.

    Args:
        game_id: Specific game (optional)
        player_id: Specific player (optional)
        season: Season string (optional)

    Returns:
        DataFrame with shot records
    """

def get_boxscore_advanced(
    self, game_id: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch advanced boxscore using BoxScoreAdvancedV2.

    Returns:
        Tuple of (team_stats_df, player_stats_df)
    """

def get_boxscore_traditional(
    self, game_id: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch traditional boxscore using BoxScoreTraditionalV2.

    Returns:
        Tuple of (team_stats_df, player_stats_df)
    """

def get_player_tracking(self, game_id: str) -> pd.DataFrame:
    """
    Fetch player tracking using BoxScorePlayerTrackV2.

    Returns:
        DataFrame with tracking metrics (distance, speed)
    """

def get_team_roster(self, team_id: int, season: str) -> pd.DataFrame:
    """
    Fetch team roster using CommonTeamRoster.

    Returns:
        DataFrame with player roster info
    """

def get_player_info(self, player_id: int) -> pd.DataFrame:
    """
    Fetch player biographical info using CommonPlayerInfo.

    Returns:
        DataFrame with player details
    """
```

**Retry and error handling:**

```python
RETRIABLE_STATUS_CODES = {429, 500, 502, 503, 504}
NON_RETRIABLE_STATUS_CODES = {400, 404}

def _request_with_retry(
    self,
    endpoint_class: type,
    **kwargs
) -> pd.DataFrame:
    """
    Execute API request with retry logic.

    - Applies rate limiting delay before each request
    - Retries on transient errors with exponential backoff
    - Logs all requests and responses
    - Raises APIError on permanent failures
    """
```

**Custom exceptions:**

```python
class NBAApiError(Exception):
    """Base exception for NBA API errors."""
    pass

class NBAApiRateLimitError(NBAApiError):
    """Rate limit exceeded."""
    pass

class NBAApiNotFoundError(NBAApiError):
    """Resource not found (404)."""
    pass

class NBAApiTimeoutError(NBAApiError):
    """Request timeout."""
    pass
```

---

### Task 2b.2: Games Collector

**Location:** `nba_model/data/collectors/games.py`

```python
class GamesCollector:
    """
    Collects game records for specified seasons.

    Fetches from LeagueGameFinder and normalizes to Game model format.
    """

    def __init__(self, api_client: NBAApiClient, db_session: Session):
        pass

    def collect_season(
        self,
        season: str,
        season_type: str = "Regular Season",
    ) -> list[Game]:
        """
        Collect all games for a season.

        Args:
            season: Season string (e.g., "2023-24")
            season_type: "Regular Season" or "Playoffs"

        Returns:
            List of Game model instances (not yet persisted)
        """

    def collect_date_range(
        self,
        start_date: date,
        end_date: date,
    ) -> list[Game]:
        """
        Collect games within a date range.

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            List of Game model instances
        """

    def get_game_ids_for_season(
        self,
        season: str,
        season_type: str = "Regular Season",
    ) -> list[str]:
        """
        Get list of game IDs for a season (useful for other collectors).

        Returns:
            List of game ID strings
        """

    def _transform_game(self, row: pd.Series) -> Game:
        """Transform API response row to Game model."""
```

**Data transformations:**
- Extract game_id from API response
- Parse game_date from string format
- Map team IDs to home/away based on matchup string ("LAL @ BOS")
- Set status based on game completion
- Handle missing attendance data

---

### Task 2b.3: Players Collector

**Location:** `nba_model/data/collectors/players.py`

```python
class PlayersCollector:
    """
    Collects player information from team rosters.

    Builds Player and PlayerSeason records from roster data.
    """

    def __init__(self, api_client: NBAApiClient, db_session: Session):
        pass

    def collect_rosters(
        self,
        season: str,
        team_ids: list[int] | None = None,
    ) -> tuple[list[Player], list[PlayerSeason]]:
        """
        Collect all player info from team rosters.

        Args:
            season: Season string
            team_ids: Optional list of team IDs (defaults to all 30)

        Returns:
            Tuple of (players, player_seasons)
        """

    def collect_player_details(
        self,
        player_id: int,
    ) -> Player:
        """
        Collect detailed info for a single player.

        Uses CommonPlayerInfo endpoint for biographical data.
        """

    def collect_teams(self) -> list[Team]:
        """
        Collect all team records.

        Uses hardcoded team data with arena coordinates.
        """

    def _transform_player(self, row: pd.Series) -> Player:
        """Transform roster row to Player model."""

    def _transform_player_season(
        self, row: pd.Series, season: str
    ) -> PlayerSeason:
        """Transform roster row to PlayerSeason model."""
```

**Team data source:**
- Use hardcoded dictionary of team info including arena coordinates
- NBA team IDs are stable across seasons

```python
TEAM_DATA = {
    1610612737: {
        "abbreviation": "ATL",
        "full_name": "Atlanta Hawks",
        "city": "Atlanta",
        "arena_name": "State Farm Arena",
        "arena_lat": 33.757,
        "arena_lon": -84.396,
    },
    # ... all 30 teams
}
```

---

### Task 2b.4: Play-by-Play Collector

**Location:** `nba_model/data/collectors/playbyplay.py`

```python
class PlayByPlayCollector:
    """
    Collects play-by-play event data for games.

    Each game has ~300-500 events with event types, descriptions,
    timestamps, and player references.
    """

    def __init__(self, api_client: NBAApiClient, db_session: Session):
        pass

    def collect_game(self, game_id: str) -> list[Play]:
        """
        Collect play-by-play for a single game.

        Args:
            game_id: NBA game ID

        Returns:
            List of Play model instances
        """

    def collect_games(
        self,
        game_ids: list[str],
        on_error: Literal["raise", "skip", "log"] = "log",
    ) -> dict[str, list[Play]]:
        """
        Collect play-by-play for multiple games.

        Args:
            game_ids: List of game IDs
            on_error: Error handling strategy

        Returns:
            Dict mapping game_id to list of Play instances
        """

    def _transform_play(self, row: pd.Series, game_id: str) -> Play:
        """Transform API response row to Play model."""

    def _parse_time(self, pctimestring: str) -> str:
        """Parse time string to consistent format."""

    def _extract_player_ids(self, row: pd.Series) -> tuple[int | None, ...]:
        """Extract player IDs from event data."""
```

**Event type reference:**
```python
EVENT_TYPES = {
    1: "FIELD_GOAL_MADE",
    2: "FIELD_GOAL_MISSED",
    3: "FREE_THROW",
    4: "REBOUND",
    5: "TURNOVER",
    6: "FOUL",
    7: "VIOLATION",
    8: "SUBSTITUTION",
    9: "TIMEOUT",
    10: "JUMP_BALL",
    12: "PERIOD_START",
    13: "PERIOD_END",
}
```

---

### Task 2b.5: Shots Collector

**Location:** `nba_model/data/collectors/shots.py`

```python
class ShotsCollector:
    """
    Collects shot chart data with court coordinates.

    Shot data includes location (loc_x, loc_y), zone classification,
    and make/miss outcome.
    """

    def __init__(self, api_client: NBAApiClient, db_session: Session):
        pass

    def collect_game(self, game_id: str) -> list[Shot]:
        """
        Collect all shots for a single game.

        Args:
            game_id: NBA game ID

        Returns:
            List of Shot model instances
        """

    def collect_player_season(
        self,
        player_id: int,
        season: str,
    ) -> list[Shot]:
        """
        Collect all shots for a player in a season.

        Args:
            player_id: NBA player ID
            season: Season string

        Returns:
            List of Shot model instances
        """

    def collect_games(
        self,
        game_ids: list[str],
        on_error: Literal["raise", "skip", "log"] = "log",
    ) -> dict[str, list[Shot]]:
        """
        Collect shots for multiple games.

        Returns:
            Dict mapping game_id to list of Shot instances
        """

    def _transform_shot(self, row: pd.Series) -> Shot:
        """Transform API response row to Shot model."""
```

**Court coordinate normalization:**
- LOC_X range: -250 to 250 (left to right from behind basket)
- LOC_Y range: -50 to 900 (baseline to opposite baseline)
- Store as integers (tenths of feet from basket)

---

### Task 2b.6: Box Scores Collector

**Location:** `nba_model/data/collectors/boxscores.py`

```python
class BoxScoreCollector:
    """
    Collects box score data (team and player level).

    Fetches traditional stats, advanced stats, and player tracking
    metrics where available.
    """

    def __init__(self, api_client: NBAApiClient, db_session: Session):
        pass

    def collect_game(
        self, game_id: str
    ) -> tuple[list[GameStats], list[PlayerGameStats]]:
        """
        Collect full box score for a game.

        Args:
            game_id: NBA game ID

        Returns:
            Tuple of (game_stats, player_game_stats)
        """

    def collect_games(
        self,
        game_ids: list[str],
        on_error: Literal["raise", "skip", "log"] = "log",
    ) -> tuple[dict[str, list[GameStats]], dict[str, list[PlayerGameStats]]]:
        """
        Collect box scores for multiple games.

        Returns:
            Tuple of dicts mapping game_id to stats
        """

    def _fetch_traditional(self, game_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch traditional box score."""

    def _fetch_advanced(self, game_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch advanced box score."""

    def _fetch_tracking(self, game_id: str) -> pd.DataFrame | None:
        """Fetch player tracking (may not exist for old games)."""

    def _merge_player_stats(
        self,
        traditional: pd.DataFrame,
        advanced: pd.DataFrame,
        tracking: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """Merge all player stat sources."""

    def _transform_game_stats(
        self, row: pd.Series, game_id: str
    ) -> GameStats:
        """Transform to GameStats model."""

    def _transform_player_game_stats(
        self, row: pd.Series, game_id: str
    ) -> PlayerGameStats:
        """Transform to PlayerGameStats model."""
```

**Missing data handling:**
- Player tracking only available from ~2013-14 season
- Some older games may have incomplete advanced stats
- Use None for missing fields, never fake data

---

### Task 2b.7: Collector Base Class

**Location:** `nba_model/data/collectors/base.py`

```python
from abc import ABC, abstractmethod

class BaseCollector(ABC):
    """
    Base class for all collectors with common functionality.
    """

    def __init__(self, api_client: NBAApiClient, db_session: Session):
        self.api = api_client
        self.session = db_session
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def collect_game(self, game_id: str) -> Any:
        """Collect data for a single game."""
        pass

    def _log_progress(
        self,
        current: int,
        total: int,
        item_id: str,
    ) -> None:
        """Log collection progress."""

    def _handle_error(
        self,
        error: Exception,
        item_id: str,
        on_error: Literal["raise", "skip", "log"],
    ) -> None:
        """Handle collection error based on strategy."""
```

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `nba_model/data/api.py` | Create | NBA API client wrapper |
| `nba_model/data/collectors/base.py` | Create | Base collector class |
| `nba_model/data/collectors/games.py` | Create | Games collector |
| `nba_model/data/collectors/players.py` | Create | Players/Teams collector |
| `nba_model/data/collectors/playbyplay.py` | Create | Play-by-play collector |
| `nba_model/data/collectors/shots.py` | Create | Shots collector |
| `nba_model/data/collectors/boxscores.py` | Create | Box scores collector |
| `nba_model/data/collectors/__init__.py` | Modify | Export collectors |
| `nba_model/data/__init__.py` | Modify | Export public API |
| `tests/test_data/test_api.py` | Create | API client tests |
| `tests/test_data/test_collectors/` | Create | Collector tests |
| `tests/fixtures/` | Create | Sample API responses |

---

## Testing Requirements

### Test Fixtures (`tests/fixtures/`)

Create JSON fixtures for each API endpoint:

```
tests/fixtures/
├── leaguegamefinder_response.json
├── playbyplayv2_response.json
├── shotchartdetail_response.json
├── boxscoreadvancedv2_response.json
├── boxscoretraditionalv2_response.json
├── boxscoreplayertrackv2_response.json
├── commonteamroster_response.json
└── commonplayerinfo_response.json
```

Each fixture should contain:
- A realistic subset of actual API response
- Edge cases (missing fields, empty arrays)
- Both success and error scenarios

### API Client Tests (`tests/test_data/test_api.py`)

**Rate limiting tests:**
- Verify delay applied between requests
- Test configurable delay values

**Retry tests:**
- Mock transient error (429, 503), verify retry
- Verify exponential backoff timing
- Test max_retries limit reached

**Error handling tests:**
- Test 404 raises NBAApiNotFoundError
- Test 400 raises NBAApiError (non-retriable)
- Test timeout handling

**Response parsing tests:**
- Verify DataFrame structure from each endpoint
- Test column naming standardization

### Collector Tests (`tests/test_data/test_collectors/`)

**GamesCollector tests:**
- Test season collection with mocked API
- Verify home/away parsing from matchup string
- Test date range filtering
- Test handling of postponed games

**PlayersCollector tests:**
- Test roster collection
- Verify Player model population
- Test PlayerSeason creation
- Test team data loading

**PlayByPlayCollector tests:**
- Test single game collection
- Verify event type mapping
- Test player ID extraction
- Test time parsing

**ShotsCollector tests:**
- Test game shot collection
- Verify coordinate normalization
- Test zone classification

**BoxScoreCollector tests:**
- Test stat merging from multiple sources
- Verify handling of missing tracking data
- Test both team and player level stats

---

## Success Criteria

1. API client respects rate limits (0.6s default delay)
2. Retry logic handles transient errors correctly
3. All collectors transform API responses to model instances
4. Edge cases handled gracefully (missing data, empty responses)
5. Test coverage >= 90% using mocked responses
6. No actual API calls in unit tests
7. Logging captures all API interactions

---

## Implementation Checklist

### API Client
- [ ] Create `nba_model/data/api.py`
- [ ] Implement NBAApiClient class with constructor
- [ ] Implement rate limiting with configurable delay
- [ ] Implement retry logic with exponential backoff
- [ ] Implement timeout handling
- [ ] Define custom exception classes
- [ ] Implement get_league_game_finder()
- [ ] Implement get_play_by_play()
- [ ] Implement get_shot_chart()
- [ ] Implement get_boxscore_advanced()
- [ ] Implement get_boxscore_traditional()
- [ ] Implement get_player_tracking()
- [ ] Implement get_team_roster()
- [ ] Implement get_player_info()
- [ ] Add comprehensive logging

### Base Collector
- [ ] Create `nba_model/data/collectors/base.py`
- [ ] Implement BaseCollector abstract class
- [ ] Add progress logging helper
- [ ] Add error handling helper

### Games Collector
- [ ] Create `nba_model/data/collectors/games.py`
- [ ] Implement GamesCollector class
- [ ] Implement collect_season()
- [ ] Implement collect_date_range()
- [ ] Implement get_game_ids_for_season()
- [ ] Implement _transform_game()

### Players Collector
- [ ] Create `nba_model/data/collectors/players.py`
- [ ] Define TEAM_DATA constant with all 30 teams
- [ ] Implement PlayersCollector class
- [ ] Implement collect_rosters()
- [ ] Implement collect_player_details()
- [ ] Implement collect_teams()
- [ ] Implement transform methods

### Play-by-Play Collector
- [ ] Create `nba_model/data/collectors/playbyplay.py`
- [ ] Define EVENT_TYPES constant
- [ ] Implement PlayByPlayCollector class
- [ ] Implement collect_game()
- [ ] Implement collect_games()
- [ ] Implement _transform_play()
- [ ] Implement time parsing

### Shots Collector
- [ ] Create `nba_model/data/collectors/shots.py`
- [ ] Implement ShotsCollector class
- [ ] Implement collect_game()
- [ ] Implement collect_player_season()
- [ ] Implement collect_games()
- [ ] Implement _transform_shot()

### Box Scores Collector
- [ ] Create `nba_model/data/collectors/boxscores.py`
- [ ] Implement BoxScoreCollector class
- [ ] Implement collect_game()
- [ ] Implement collect_games()
- [ ] Implement stat fetching methods
- [ ] Implement stat merging logic
- [ ] Implement transform methods
- [ ] Handle missing tracking data

### Package Integration
- [ ] Update `nba_model/data/collectors/__init__.py`
- [ ] Update `nba_model/data/__init__.py`
- [ ] Verify imports work correctly

### Testing
- [ ] Create test fixtures for each API endpoint
- [ ] Create `tests/test_data/test_api.py`
- [ ] Create `tests/test_data/test_collectors/test_games.py`
- [ ] Create `tests/test_data/test_collectors/test_players.py`
- [ ] Create `tests/test_data/test_collectors/test_playbyplay.py`
- [ ] Create `tests/test_data/test_collectors/test_shots.py`
- [ ] Create `tests/test_data/test_collectors/test_boxscores.py`
- [ ] Achieve 90%+ test coverage
- [ ] All tests pass without network calls

---

## Completion Instructions

Upon completion:
1. Run full test suite: `pytest tests/test_data/ -v --cov=nba_model.data`
2. Verify coverage >= 90%
3. Run type checking: `mypy nba_model/data/`
4. Run linting: `ruff check nba_model/data/`
5. Commit with message: `feat(data): implement Phase 2b - data collection`
