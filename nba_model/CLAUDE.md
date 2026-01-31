# nba_model Package

## Purpose

Core package containing all production code for the NBA prediction system. Provides CLI interface, configuration management, and subpackages for each system component.

## Module Map

| Module | Responsibility | Key Exports |
|--------|---------------|-------------|
| `__init__.py` | Package metadata, public API | `__version__`, `Settings`, `get_settings` |
| `cli.py` | Typer CLI with command groups | `app` (Typer application) |
| `config.py` | Pydantic Settings configuration | `Settings`, `get_settings`, `reset_settings` |
| `logging.py` | Loguru logging configuration | `setup_logging`, `get_logger` |
| `types.py` | Shared type definitions | TypedDicts, Protocols, type aliases |

## Subpackage Map

| Subpackage | Phase | Status | Responsibility |
|------------|-------|--------|---------------|
| `data/` | 2 | ✅ Complete | Data collection, storage, ETL |
| `features/` | 3 | Stub | Feature engineering (RAPM, spatial) |
| `models/` | 4 | Stub | ML models (Transformer, GNN, Fusion) |
| `backtest/` | 5 | Stub | Backtesting, Kelly sizing, devigging |
| `monitor/` | 6 | Stub | Drift detection, retraining triggers |
| `predict/` | 7 | Stub | Inference pipeline, signal generation |
| `output/` | 8 | Stub | Dashboard, reports |

## Data Flow

```
nba_api → data/ → features/ → models/ → predict/ → output/
                                ↑            ↓
                            backtest/ ← monitor/
```

## CLI Command Groups

```
nba-model
├── data       # collect, status, validate
├── features   # build, export
├── train      # transformer, gnn, fusion
├── backtest   # run, report
├── monitor    # drift, triggers
├── predict    # game, today, signals
└── dashboard  # build, serve
```

## Dependencies

**External:**
- `typer` - CLI framework
- `rich` - Terminal formatting
- `pydantic-settings` - Configuration
- `loguru` - Logging

**Internal:** None (root package)

## Conventions

1. **Subpackages are self-contained** - Minimal cross-subpackage imports
2. **Public API in `__init__.py`** - Each subpackage exports via `__all__`
3. **Config injection** - Use `get_settings()`, never hardcode paths
4. **Logging** - Use `loguru` via `logging.py`, not stdlib `logging`

## Anti-Patterns

- ❌ Never import subpackage internals directly (use public API)
- ❌ Never add business logic to `cli.py` (delegate to subpackages)
- ❌ Never use global state except `_settings` singleton in config
- ❌ Never use `print()` for output (use `rich.console` or logging)
