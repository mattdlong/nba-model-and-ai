# NBA Quantitative Trading Strategy

## Overview

A Python CLI application that predicts NBA game outcomes using machine learning (Transformer + GNN fusion architecture) and generates betting signals with Kelly criterion sizing. The system includes automated data collection, feature engineering, model training, backtesting, and drift detection.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           CLI Interface (Typer)                        │
├─────────────────────────────────────────────────────────────────────────┤
│  Data Layer  │  Features   │  Models    │  Backtest  │  Predictions    │
│  (nba_api)   │  (RAPM,     │  (Xformer, │  (Kelly,   │  (Inference,    │
│  (SQLite)    │   Spatial)  │   GNN)     │   Devig)   │   Signals)      │
├─────────────────────────────────────────────────────────────────────────┤
│                     Monitor (Drift Detection + Retraining)             │
├─────────────────────────────────────────────────────────────────────────┤
│                     Output (Dashboard + Reports)                       │
└─────────────────────────────────────────────────────────────────────────┘
```

## Development Phases

| Phase | Name | Status | Completion |
|-------|------|--------|------------|
| 1 | Project Foundation | ✅ Complete | 100% |
| 2 | Data Collection | ✅ Complete | 100% |
| 3 | Feature Engineering | 🔲 Not Started | 0% |
| 4 | Model Architecture | 🔲 Not Started | 0% |
| 5 | Backtesting Engine | 🔲 Not Started | 0% |
| 6 | Self-Improvement | 🔲 Not Started | 0% |
| 7 | Production Pipeline | 🔲 Not Started | 0% |
| 8 | Output Generation | 🔲 Not Started | 0% |

## Key Decisions

1. **CLI Framework**: Typer (not Click) for modern async support and auto-completion
2. **Database**: SQLite for simplicity; designed for Postgres migration if needed
3. **Config**: Pydantic Settings v2 with environment variable aliases
4. **ML Stack**: PyTorch 2.2.2 + PyTorch Geometric (requires numpy<2)
5. **Testing**: pytest with strict mode, 75% coverage target

## Quick Reference

```bash
# Activate environment
cd ~/Documents/code/nba-model-and-ai
source .venv/bin/activate

# Run CLI
python -m nba_model.cli --help
python -m nba_model.cli data status

# Run tests
pytest -v
pytest --cov=nba_model

# Code quality
black . && ruff check . --fix && mypy nba_model/
```

## Directory Map

| Path | Purpose |
|------|---------|
| `nba_model/` | Main package - all production code |
| `tests/` | Test suite (mirrors nba_model structure) |
| `plan/` | Development phase plans and checklists |
| `implementation/` | Implementation notes and artifacts |
| `docs/` | GitHub Pages dashboard (static site) |
| `data/` | Database + model weights (gitignored) |

## Anti-Patterns

- ❌ Never import from `tests/` in production code
- ❌ Never use `numpy>=2` (PyTorch 2.2.2 incompatible)
- ❌ Never commit `.env` files (use `.env.example`)
- ❌ Never call NBA API without rate limiting
- ❌ Never use `Any` type without justification
- ❌ Never skip type hints on public functions
