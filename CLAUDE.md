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
| 3 | Feature Engineering | ✅ Complete | 100% |
| 4 | Model Architecture | ✅ Complete | 100% |
| 5 | Backtesting Engine | 🔲 Not Started | 0% |
| 6 | Self-Improvement | 🔲 Not Started | 0% |
| 7 | Production Pipeline | 🔲 Not Started | 0% |
| 8 | Output Generation | 🔲 Not Started | 0% |

## Key Decisions

1. **CLI Framework**: Typer (not Click) for modern async support and auto-completion
2. **Database**: SQLite for simplicity; designed for Postgres migration if needed
3. **Config**: Pydantic Settings v2 with environment variable aliases
4. **ML Stack**: PyTorch 2.2.2 + PyTorch Geometric (requires numpy<2)
5. **Testing**: pytest with strict mode, 75% minimum coverage (80% for new code)

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

## Troubleshooting

### OpenMP SHM Error on macOS

If you encounter an OpenMP shared memory error when running tests or importing PyTorch:
```
OMP: Error #179: Function Can't open SHM2 failed
```
or
```
Process exited with signal 6 (SIGABRT)
```

**Automatic Fix (Applied in tests/conftest.py):**
The test suite automatically sets the required environment variables before importing PyTorch. This should resolve the issue in most cases.

**Manual Fix (if automatic fix fails):**
Set these environment variables BEFORE running tests:
```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export KMP_DISABLE_SHM=1
export KMP_DUPLICATE_LIB_OK=TRUE
```

Or run tests with inline variables:
```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DISABLE_SHM=1 pytest -v
```

**Alternative: Conda Environment:**
If issues persist, consider using a Conda environment which manages OpenMP more reliably:
```bash
conda create -n nba-model python=3.11
conda activate nba-model
conda install pytorch -c pytorch
pip install -e ".[dev]"
```

**Root Cause:**
This error occurs due to OpenMP shared memory conflicts between different libraries (PyTorch, NumPy, SciPy) on macOS. Setting `OMP_NUM_THREADS=1` disables OpenMP parallelism, avoiding the conflict.

## Anti-Patterns

- ❌ Never import from `tests/` in production code
- ❌ Never use `numpy>=2` (PyTorch 2.2.2 incompatible)
- ❌ Never commit `.env` files (use `.env.example`)
- ❌ Never call NBA API without rate limiting
- ❌ Never use `Any` type without justification
- ❌ Never skip type hints on public functions
