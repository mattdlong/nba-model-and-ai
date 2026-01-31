# Data Collectors

## Responsibility

Individual data collectors for each NBA data entity type. Each collector handles one specific data source from the NBA API.

## Status

✅ **Phase 2 - Complete**

## Files

| File | Purpose | NBA API Endpoints |
|------|---------|-------------------|
| `__init__.py` | Collector exports | - |
| `base.py` | Abstract base collector | - |
| `games.py` | Game schedules + results | `LeagueGameFinder` |
| `players.py` | Rosters + player info | `CommonTeamRoster`, `CommonPlayerInfo` |
| `playbyplay.py` | Play-by-play events | `PlayByPlayV2` |
| `shots.py` | Shot chart data | `ShotChartDetail` |
| `boxscores.py` | Box score stats | `BoxScoreTraditionalV2`, `BoxScoreAdvancedV2` |

## Collector Pattern

All collectors inherit from `BaseCollector`:

```python
class BaseCollector(ABC):
    def __init__(self, api: NBAApiClient, db: Session) -> None: ...

    def collect(self, season_range: list[str], resume_from: str | None = None) -> Any: ...

    @abstractmethod
    def collect_game(self, game_id: str) -> Any: ...

    def get_last_checkpoint(self) -> str | None: ...
    def set_checkpoint(self, checkpoint: str) -> None: ...
```

## Common Tasks

- **Add new collector:** Copy existing, inherit `BaseCollector`, implement abstract methods
- **Handle rate limits:** Use `self.api.request()` which handles delays
- **Resume collection:** Check `get_checkpoint()` before starting

## Anti-Patterns

- ❌ Never bypass `BaseCollector` - all collectors must inherit it
- ❌ Never call `nba_api` directly - use `self.api`
- ❌ Never collect without checkpointing progress
