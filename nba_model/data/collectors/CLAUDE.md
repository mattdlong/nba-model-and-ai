# Data Collectors

## Responsibility

Individual data collectors for each NBA data entity type. Each collector handles one specific data source from the NBA API.

## Status

🔲 **Phase 2 - Not Started** (stub `__init__.py` only)

## Planned Files

| File | Purpose | NBA API Endpoints |
|------|---------|-------------------|
| `__init__.py` | Collector exports | - |
| `base.py` | Abstract base collector | - |
| `games.py` | Game schedules + results | `LeagueGameFinder`, `BoxScoreSummaryV2` |
| `players.py` | Rosters + player info | `CommonTeamRoster`, `CommonPlayerInfo` |
| `playbyplay.py` | Play-by-play events | `PlayByPlayV3` |
| `shots.py` | Shot chart data | `ShotChartDetail` |

## Collector Pattern

All collectors must inherit from `BaseCollector`:

```python
class BaseCollector(ABC):
    def __init__(self, api: NBAApiClient, db: Session) -> None: ...
    
    @abstractmethod
    def collect_season(self, season: str) -> CollectionResult: ...
    
    @abstractmethod
    def collect_game(self, game_id: str) -> CollectionResult: ...
    
    def checkpoint(self, key: str, value: Any) -> None: ...
    def get_checkpoint(self, key: str) -> Any | None: ...
```

## Common Tasks

- **Add new collector:** Copy existing, inherit `BaseCollector`, implement abstract methods
- **Handle rate limits:** Use `self.api.request()` which handles delays
- **Resume collection:** Check `get_checkpoint()` before starting

## Anti-Patterns

- ❌ Never bypass `BaseCollector` - all collectors must inherit it
- ❌ Never call `nba_api` directly - use `self.api`
- ❌ Never collect without checkpointing progress
