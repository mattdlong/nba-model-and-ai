# Test Suite

## Purpose

Comprehensive test coverage for the NBA model package. Mirrors source structure with unit tests, integration tests, and shared fixtures.

## Structure

| Directory | Purpose | Run Command |
|-----------|---------|-------------|
| `unit/` | Isolated unit tests | `pytest tests/unit/` |
| `integration/` | Cross-module tests | `pytest tests/integration/` |
| `fixtures/` | Shared test data | (imported by tests) |
| `conftest.py` | Shared fixtures | (auto-loaded by pytest) |

## Test Coverage Requirements

- **Minimum:** 75% overall coverage
- **Critical paths:** 90% coverage for betting/prediction logic
- **Run with:** `pytest --cov=nba_model --cov-report=term-missing`

## Current Status

| Test File | Tests | Passing |
|-----------|-------|---------|
| `unit/test_cli.py` | 14 | ✅ 14 |
| `unit/test_config.py` | 8 | ✅ 8 |

## Conventions

1. **Test names:** `test_<what>_<condition>` (e.g., `test_kelly_rejects_negative_edge`)
2. **Classes:** `Test<ClassName>` groups related tests
3. **Fixtures:** Define in `conftest.py`, use dependency injection
4. **Markers:** `@pytest.mark.slow`, `@pytest.mark.integration`

## Running Tests

```bash
# All tests
pytest

# Verbose with short traceback
pytest -v --tb=short

# Specific file
pytest tests/unit/test_config.py

# With coverage
pytest --cov=nba_model --cov-fail-under=75

# Skip slow tests
pytest -m "not slow"
```

## Anti-Patterns

- ❌ Never use real NBA API in unit tests (mock it)
- ❌ Never share state between tests (use fixtures)
- ❌ Never use `time.sleep()` in tests (mock time)
- ❌ Never commit test databases
