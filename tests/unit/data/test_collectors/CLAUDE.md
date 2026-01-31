# Data Collector Tests

## Purpose

Unit tests for the entity-specific data collectors that fetch data from the NBA API.

## Structure

| File | Tests | Coverage |
|------|-------|----------|
| `test_base.py` | BaseCollector rate limiting and retry logic | Core functionality |
| `test_games.py` | GamesCollector season/date filtering | Games collection |
| `test_players.py` | PlayersCollector roster fetching | Players collection |
| `test_playbyplay.py` | PlayByPlayCollector event parsing | Play-by-play collection |
| `test_shots.py` | ShotsCollector location parsing | Shot chart collection |
| `test_boxscores.py` | BoxScoreCollector stat parsing | Box score collection |

## Testing Approach

- All tests use mock API responses (no real network calls)
- Fixtures provide consistent test data
- Rate limiting tested via time measurement
- Error handling tested via mock exceptions

## Anti-Patterns

- Never make real API calls in unit tests
- Never use hardcoded API responses without documentation
- Never skip rate limit validation tests
