# Phase 1: Application Structure

## Objective
Foundation layer - CLI app skeleton, config system, logging, project layout.

## Dependencies
- None (foundation phase)

## Deliverables

### 1.1 Project Layout
```
nba-model/
├── pyproject.toml
├── .env.example
├── nba_model/
│   ├── __init__.py
│   ├── cli.py              # Typer entrypoint
│   ├── config.py           # Pydantic Settings
│   ├── data/               # Phase 2
│   ├── features/           # Phase 3
│   ├── models/             # Phase 4
│   ├── backtest/           # Phase 5
│   ├── monitor/            # Phase 6
│   ├── predict/            # Phase 7
│   └── output/             # Phase 8
├── data/                   # gitignored storage
│   ├── nba.db
│   └── models/
├── docs/                   # GitHub Pages
└── tests/
```

### 1.2 Core Dependencies

| Category | Package | Version |
|----------|---------|---------|
| CLI | typer | >=0.9.0 |
| CLI | rich | >=13.0 |
| Data | nba_api | >=1.4 |
| Data | sqlalchemy | >=2.0 |
| Data | pandas | >=2.0 |
| ML | torch | >=2.0 |
| ML | torch-geometric | >=2.4 |
| ML | scikit-learn | >=1.3 |
| Config | pydantic | >=2.0 |
| Config | pydantic-settings | >=2.0 |
| Logging | loguru | >=0.7 |
| Geo | haversine | >=2.8 |
| Template | jinja2 | >=3.1 |

### 1.3 CLI Command Tree
```
nba-model
├── data {collect, update, status}
├── features {build, rapm, spatial}
├── train {transformer, gnn, fusion, all}
├── backtest {run, report, optimize}
├── monitor {drift, trigger, versions}
├── predict {today, game, signals}
└── dashboard {build, deploy}
```

### 1.4 Configuration System

| Setting | Env Var | Default |
|---------|---------|---------|
| DB path | NBA_DB_PATH | data/nba.db |
| API delay | NBA_API_DELAY | 0.6s |
| Log level | LOG_LEVEL | INFO |
| Model dir | MODEL_DIR | data/models |

**Config class**: Pydantic Settings with `.env` file support

### 1.5 Logging Spec
- Framework: Loguru
- Features: File rotation, structured JSON output
- Levels: DEBUG/INFO/WARNING/ERROR

## Acceptance Criteria
- [ ] `nba-model --help` shows all commands
- [ ] Config loads from `.env` and env vars
- [ ] Logging outputs to console and file
- [ ] All submodule directories created with `__init__.py`
- [ ] pyproject.toml valid, `pip install -e .` works

## Complexity
Low
