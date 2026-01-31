# Development Guidelines

## NBA Quantitative Trading System - Development Standards

This document establishes **STRICT** development standards that ALL developers must follow when contributing to the NBA quantitative trading system. Non-compliance will result in rejected pull requests.

---

## Table of Contents

1. [Terminology & Glossary](#1-terminology--glossary)
2. [Coding Standards](#2-coding-standards)
3. [Folder Structure](#3-folder-structure)
4. [Code Organization](#4-code-organization)
5. [Best Practices](#5-best-practices)
6. [Common Libraries](#6-common-libraries)
7. [Testing Expectations](#7-testing-expectations)
8. [Git Workflow](#8-git-workflow)
9. [Definition of Done](#9-definition-of-done)

---

## 1. Terminology & Glossary

### 1.1 Statistical & Mathematical Terms

| Term | Definition |
|------|------------|
| **RAPM** | Regularized Adjusted Plus-Minus. A ridge regression-based metric measuring a player's impact on team point differential per 100 possessions, controlling for teammates and opponents. |
| **ORAPM** | Offensive RAPM. The offensive component of RAPM measuring a player's impact on points scored. |
| **DRAPM** | Defensive RAPM. The defensive component of RAPM measuring a player's impact on points allowed. |
| **Ridge Regression** | Linear regression with L2 regularization (λ\|\|β\|\|²) to prevent overfitting on sparse lineup data. |
| **Design Matrix** | Sparse matrix X where rows are stints, columns are players, values are +1 (home) or -1 (away). |
| **Stint** | A continuous period of play with the same 10 players on the court. The fundamental unit for RAPM calculation. |
| **Possession** | A team's opportunity to score, ending in a shot attempt, turnover, or foul. Used to normalize statistics. |
| **Time Decay** | Exponential weighting: w = exp(-(today - game_date) / τ) where τ is the half-life in days (typically 180). |

### 1.2 Spatial Analytics Terms

| Term | Definition |
|------|------------|
| **Convex Hull** | The smallest convex polygon containing all shot locations for a lineup. Area indicates floor spacing. |
| **Hull Area** | Square footage covered by a lineup's shot distribution. Larger = better spacing (typically 400-800 sq units). |
| **Centroid** | Geometric center of shot locations (x, y). Indicates offensive positioning tendency. |
| **Shot Density (KDE)** | Kernel Density Estimation of shot locations. Creates probability heatmaps for shooting patterns. |
| **Corner Density** | Proportion of shots from corner 3-point zones. Key spacing indicator. |
| **Gravity Overlap** | Degree to which two players' shooting zones overlap. High overlap = poor spacing. |
| **LOC_X / LOC_Y** | NBA API shot coordinates in tenths of feet from the basket center. X: -250 to 250, Y: -50 to 900. |

### 1.3 Machine Learning Terms

| Term | Definition |
|------|------------|
| **Transformer Encoder** | Self-attention-based architecture for sequence modeling (game flow events). |
| **GATv2** | Graph Attention Network v2. Attention-based GNN for modeling player interactions. |
| **Two-Tower Architecture** | Fusion model with separate towers for context (static) and dynamic features. |
| **d_model** | Transformer embedding dimension (default: 128). |
| **nhead** | Number of attention heads in Transformer/GAT (default: 4). |
| **Multi-Task Learning** | Single model predicting multiple outputs (win probability, margin, total). |
| **Huber Loss** | Robust loss function for regression, less sensitive to outliers than MSE. |
| **Brier Score** | Calibration metric: mean((predicted_prob - actual_outcome)²). Lower is better. Target: < 0.24. |
| **Log Loss** | Cross-entropy loss for probability predictions. Measures calibration quality. |

### 1.4 Drift Detection Terms

| Term | Definition |
|------|------------|
| **Covariate Drift** | Change in input feature distributions over time (e.g., league-wide pace increases). |
| **Concept Drift** | Change in the relationship between features and target (e.g., 3-point revolution changing game dynamics). |
| **KS Test** | Kolmogorov-Smirnov test. Non-parametric test comparing two distributions. p < 0.05 indicates drift. |
| **PSI** | Population Stability Index. PSI < 0.1 (stable), 0.1-0.2 (moderate shift), > 0.2 (significant drift). |
| **MMD** | Maximum Mean Discrepancy. Kernel-based distribution comparison for high-dimensional data. |

### 1.5 Betting & Backtesting Terms

| Term | Definition |
|------|------------|
| **Kelly Criterion** | Optimal bet sizing: f* = (bp - q) / b where b = odds - 1, p = win probability, q = 1 - p. |
| **Fractional Kelly** | Conservative Kelly sizing (typically 0.25x) to reduce variance. |
| **Vig (Vigorish)** | Bookmaker's commission embedded in odds. Typically 4-5% on NBA lines. |
| **Devigging** | Removing vig to calculate true implied probabilities. Methods: multiplicative, power, Shin. |
| **Power Method** | Devig method solving for k: Σ(1/odds)^k = 1. Handles longshot bias. |
| **Shin Method** | Devig method modeling informed bettor proportion. Gold standard for liquid markets. |
| **CLV** | Closing Line Value. Edge versus final line before game start. Key profitability indicator. |
| **Walk-Forward Validation** | Time-series CV respecting temporal ordering. Train on past, validate on future. |
| **Drawdown** | Peak-to-trough decline in bankroll. Max drawdown target: < 15%. |
| **Sharpe Ratio** | Risk-adjusted return: (return - risk_free) / volatility. Target: > 1.0. |
| **ROI** | Return on Investment: profit / total_wagered. Target: > 3%. |

### 1.6 NBA-Specific Terms

| Term | Definition |
|------|------------|
| **EVENTMSGTYPE** | Play-by-play event code. 1=made shot, 2=miss, 3=FT, 4=rebound, 5=turnover, 6=foul, etc. |
| **EVENTMSGACTIONTYPE** | Detailed event subtype (e.g., dunk, layup, 3-pointer). |
| **Back-to-Back** | Team playing consecutive days. Significant fatigue factor. |
| **3-in-4** | Three games in four nights. Severe fatigue scenario. |
| **GTD** | Game Time Decision. Player status uncertain until shortly before tip-off. |
| **Pace** | Possessions per 48 minutes. League average ~100. Used for normalization. |
| **ORtg / DRtg** | Offensive/Defensive Rating. Points scored/allowed per 100 possessions. |
| **eFG%** | Effective Field Goal Percentage: (FGM + 0.5 * 3PM) / FGA. Weights 3-pointers. |
| **TOV%** | Turnover percentage: turnovers per 100 possessions. |
| **ORB%** | Offensive Rebound Percentage. Rebounding rate on missed shots. |
| **FT Rate** | Free throws attempted per field goal attempt. |

---

## 2. Coding Standards

### 2.1 Python Version

- **Required:** Python 3.11+
- Use modern syntax features: structural pattern matching, type unions (`X | Y`), `Self` type

### 2.2 Code Formatting

All code MUST pass these checks before commit:

```bash
# Formatting (auto-fix)
black . --line-length 88

# Linting (must pass with zero errors)
ruff check . --fix

# Type checking (must pass)
mypy nba_model/ --strict
```

**Configuration (pyproject.toml):**

```toml
[tool.black]
line-length = 88
target-version = ['py311']

[tool.ruff]
line-length = 88
select = ["E", "F", "I", "N", "W", "UP", "B", "C4", "SIM", "RUF"]
ignore = ["E501"]  # Line length handled by black

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_ignores = true
disallow_untyped_defs = true
```

### 2.3 Type Hints

Type hints are **MANDATORY** for all function signatures and class attributes.

**Required Patterns:**

```python
# Function signatures - ALWAYS annotate parameters and return types
def calculate_rapm(
    stints: pd.DataFrame,
    lambda_: float = 5000,
    min_minutes: int = 100,
) -> dict[int, tuple[float, float, float]]:
    """Calculate RAPM coefficients for all players."""
    ...

# Class attributes - use class-level annotations
class GamePrediction:
    game_id: str
    home_win_prob: float
    predicted_margin: float
    confidence: float

    def __init__(self, game_id: str, ...) -> None:
        ...

# Use dataclasses for data containers
from dataclasses import dataclass

@dataclass
class Bet:
    game_id: str
    bet_type: str
    model_prob: float
    profit: float

# Use Protocol for duck typing
from typing import Protocol

class FeatureCalculator(Protocol):
    def calculate(self, game_id: str) -> np.ndarray: ...

# Use TypedDict for structured dictionaries
from typing import TypedDict

class DriftResult(TypedDict):
    has_drift: bool
    features_drifted: list[str]
    details: dict[str, float]
```

**Forbidden Patterns:**

```python
# NEVER use bare `Any` without justification
def process(data: Any) -> Any:  # FORBIDDEN

# NEVER omit return type
def get_games():  # FORBIDDEN - missing return annotation
    return []

# NEVER use `# type: ignore` without explanation
result = unsafe_call()  # type: ignore  # FORBIDDEN
result = unsafe_call()  # type: ignore[attr-defined]  # OK if explained
```

### 2.4 Docstring Format

Use **Google-style** docstrings for all public functions, classes, and modules.

```python
def calculate_kelly_fraction(
    model_prob: float,
    decimal_odds: float,
    fraction: float = 0.25,
) -> float:
    """Calculate fractional Kelly bet size.

    Computes optimal bet size using Kelly Criterion with conservative
    fractional adjustment to reduce variance.

    Args:
        model_prob: Model's estimated probability of winning (0 to 1).
        decimal_odds: Decimal odds offered by bookmaker (e.g., 1.91 for -110).
        fraction: Kelly fraction multiplier (default 0.25 = quarter Kelly).

    Returns:
        Recommended bet size as fraction of bankroll (0 to max_bet_pct).
        Returns 0 if no edge or negative Kelly.

    Raises:
        ValueError: If model_prob not in [0, 1] or decimal_odds <= 1.

    Example:
        >>> calculate_kelly_fraction(0.55, 1.91, fraction=0.25)
        0.0125  # 1.25% of bankroll
    """
    ...
```

**Class Docstrings:**

```python
class RAPMCalculator:
    """Regularized Adjusted Plus-Minus calculator using Ridge Regression.

    Calculates player impact metrics by solving the regression:
        Y = Xβ + ε
    where Y is point differential per 100 possessions, X is the sparse
    player-stint design matrix, and β are the RAPM coefficients.

    Attributes:
        lambda_: Ridge regularization strength (default 5000).
        min_minutes: Minimum player minutes for inclusion (default 100).
        time_decay_tau: Half-life for time decay weighting in days.

    Example:
        >>> calculator = RAPMCalculator(lambda_=5000)
        >>> coefficients = calculator.fit(stints_df)
        >>> print(coefficients[203507])  # LeBron's RAPM
        (2.1, 0.8, 2.9)  # (ORAPM, DRAPM, total)
    """
```

### 2.5 Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Modules | `snake_case` | `play_by_play.py`, `drift_detection.py` |
| Classes | `PascalCase` | `RAPMCalculator`, `GamePrediction` |
| Functions | `snake_case` | `calculate_hull_area`, `get_player_stats` |
| Constants | `SCREAMING_SNAKE` | `MAX_RETRIES`, `DEFAULT_LAMBDA` |
| Private | `_leading_underscore` | `_build_design_matrix`, `_validate_input` |
| Type Variables | `PascalCase + T` | `ModelT`, `DataFrameT` |

**Domain-Specific Naming:**

```python
# Use full metric names, not abbreviations
home_offensive_rating: float  # NOT home_ortg
defensive_rebound_percentage: float  # NOT drb_pct

# Exception: well-known abbreviations in variable names are acceptable
rapm_coefficients: dict[int, float]  # RAPM is universally known
efg_pct: float  # eFG% is standard

# Suffix indicators
_z: normalized value  # e.g., pace_z = z-score normalized pace
_pct: percentage value  # e.g., fg_pct
_prob: probability  # e.g., home_win_prob
_df: DataFrame  # e.g., stints_df
_arr: numpy array  # e.g., features_arr
```

### 2.6 Import Organization

Use `isort` with the following configuration:

```toml
[tool.isort]
profile = "black"
line_length = 88
known_first_party = ["nba_model"]
sections = ["FUTURE", "STDLIB", "THIRDPARTY", "FIRSTPARTY", "LOCALFOLDER"]
```

**Import Order:**

```python
# 1. Standard library
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Protocol, TypedDict

# 2. Third-party packages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix
from sqlalchemy.orm import Session

# 3. First-party (our package)
from nba_model.config import Settings
from nba_model.data.models import Game, Player
from nba_model.features.rapm import RAPMCalculator
```

---

## 3. Folder Structure

```
nba-model/
├── pyproject.toml              # Project metadata, dependencies, tool config
├── README.md                   # Project overview and quick start
├── DEVELOPMENT_GUIDELINES.md   # This document
├── .env.example                # Environment variable template (COMMIT THIS)
├── .env                        # Actual secrets (NEVER COMMIT)
├── .gitignore                  # Git ignore rules
│
├── nba_model/                  # Main package - ALL source code here
│   ├── __init__.py             # Package version and public API
│   ├── cli.py                  # Typer CLI entrypoint
│   ├── config.py               # Pydantic Settings configuration
│   │
│   ├── data/                   # Data layer (Phase 2)
│   │   ├── __init__.py
│   │   ├── api.py              # NBA API client wrapper with rate limiting
│   │   ├── models.py           # SQLAlchemy ORM models
│   │   ├── schema.py           # Database schema definitions (for reference)
│   │   ├── pipelines.py        # ETL orchestration and checkpointing
│   │   └── collectors/         # Individual data collectors
│   │       ├── __init__.py
│   │       ├── base.py         # Abstract base collector
│   │       ├── games.py        # Game data collection
│   │       ├── players.py      # Player/roster collection
│   │       ├── playbyplay.py   # Play-by-play collection
│   │       └── shots.py        # Shot chart collection
│   │
│   ├── features/               # Feature engineering (Phase 3)
│   │   ├── __init__.py
│   │   ├── rapm.py             # RAPM calculation with Ridge Regression
│   │   ├── spatial.py          # Convex hull spacing metrics
│   │   ├── fatigue.py          # Rest, travel, load metrics
│   │   ├── parsing.py          # Regex-based event extraction
│   │   └── normalization.py    # Z-score normalization by season
│   │
│   ├── models/                 # ML models (Phase 4)
│   │   ├── __init__.py
│   │   ├── transformer.py      # GameFlowTransformer sequence model
│   │   ├── gnn.py              # PlayerInteractionGNN (GATv2)
│   │   ├── fusion.py           # TwoTowerFusion architecture
│   │   ├── trainer.py          # Training loop and optimization
│   │   ├── dataset.py          # PyTorch Dataset implementations
│   │   └── registry.py         # Model versioning and storage
│   │
│   ├── backtest/               # Backtesting engine (Phase 5)
│   │   ├── __init__.py
│   │   ├── engine.py           # Walk-forward validation
│   │   ├── kelly.py            # Kelly criterion sizing
│   │   ├── devig.py            # Vig removal (power, Shin)
│   │   └── metrics.py          # Performance metrics (Sharpe, CLV, etc.)
│   │
│   ├── monitor/                # Self-improvement (Phase 6)
│   │   ├── __init__.py
│   │   ├── drift.py            # Covariate/concept drift detection
│   │   ├── triggers.py         # Retraining trigger logic
│   │   └── versioning.py       # Model version management
│   │
│   ├── predict/                # Inference pipeline (Phase 7)
│   │   ├── __init__.py
│   │   ├── inference.py        # Production prediction pipeline
│   │   ├── injuries.py         # Bayesian injury adjustments
│   │   └── signals.py          # Betting signal generation
│   │
│   └── output/                 # Output generation (Phase 8)
│       ├── __init__.py
│       ├── reports.py          # Report generation
│       └── dashboard.py        # GitHub Pages static site builder
│
├── data/                       # Data storage (GITIGNORED)
│   ├── nba.db                  # SQLite database
│   ├── models/                 # Saved model weights by version
│   │   ├── v1.0.0/
│   │   │   ├── transformer.pt
│   │   │   ├── gnn.pt
│   │   │   ├── fusion.pt
│   │   │   └── metadata.json
│   │   └── latest -> v1.0.0/   # Symlink to current production
│   └── cache/                  # API response cache
│
├── docs/                       # GitHub Pages source
│   ├── index.html
│   ├── predictions.html
│   ├── history.html
│   ├── model.html
│   ├── api/                    # JSON data for dashboard
│   │   ├── today.json
│   │   ├── signals.json
│   │   └── history/
│   └── assets/
│       ├── style.css
│       └── charts.js
│
├── templates/                  # Jinja2 templates for dashboard
│   ├── base.html
│   ├── predictions.html
│   └── components/
│
└── tests/                      # Test suite
    ├── conftest.py             # Shared fixtures
    ├── unit/                   # Unit tests (mirror nba_model structure)
    │   ├── data/
    │   ├── features/
    │   ├── models/
    │   ├── backtest/
    │   ├── monitor/
    │   ├── predict/
    │   └── output/
    ├── integration/            # Integration tests
    │   ├── test_data_pipeline.py
    │   ├── test_training_pipeline.py
    │   └── test_prediction_pipeline.py
    └── fixtures/               # Test data files
        ├── sample_games.json
        ├── sample_stints.parquet
        └── sample_shots.parquet
```

### 3.1 Directory Rules

| Directory | Contents | Gitignored? |
|-----------|----------|-------------|
| `nba_model/` | All production source code | No |
| `tests/` | All test code | No |
| `docs/` | GitHub Pages static site | No |
| `templates/` | Jinja2 templates | No |
| `data/` | Database, model weights, cache | **Yes** |
| `data/models/` | Trained model checkpoints | **Yes** |

---

## 4. Code Organization

### 4.1 Module Structure

Each module MUST follow this internal organization:

```python
"""Module docstring describing purpose.

This module handles X, Y, Z functionality.
"""
# 1. Future imports (if needed)
from __future__ import annotations

# 2. Standard library imports
import json
from dataclasses import dataclass
from typing import Protocol

# 3. Third-party imports
import numpy as np
import pandas as pd

# 4. Local imports
from nba_model.config import Settings

# 5. Module-level constants
DEFAULT_LAMBDA: float = 5000.0
MIN_STINTS: int = 100

# 6. Type definitions (TypedDict, Protocol, TypeAlias)
class CalculatorProtocol(Protocol):
    def calculate(self, data: pd.DataFrame) -> np.ndarray: ...

# 7. Exception classes
class InsufficientDataError(Exception):
    """Raised when not enough data for calculation."""

# 8. Dataclasses / Named tuples
@dataclass
class RAPMResult:
    player_id: int
    orapm: float
    drapm: float
    total: float

# 9. Main classes
class RAPMCalculator:
    """Main implementation."""
    ...

# 10. Public functions
def calculate_rapm_for_season(season: str) -> dict[int, RAPMResult]:
    """Public API function."""
    ...

# 11. Private/helper functions
def _build_sparse_matrix(stints: pd.DataFrame) -> csr_matrix:
    """Internal helper."""
    ...
```

### 4.2 Class Design Patterns

**Required Patterns:**

1. **Dependency Injection** - Never instantiate dependencies internally

```python
# CORRECT: Inject dependencies
class InferencePipeline:
    def __init__(
        self,
        model_registry: ModelRegistry,
        db_session: Session,
        injury_adjuster: InjuryAdjuster,
    ) -> None:
        self.registry = model_registry
        self.db = db_session
        self.adjuster = injury_adjuster

# FORBIDDEN: Creating dependencies internally
class InferencePipeline:
    def __init__(self) -> None:
        self.registry = ModelRegistry()  # FORBIDDEN
        self.db = create_session()  # FORBIDDEN
```

2. **Factory Methods** - For complex object creation

```python
class NBADataset:
    @classmethod
    def from_season(
        cls,
        season: str,
        db_session: Session,
        tokenizer: EventTokenizer,
    ) -> NBADataset:
        """Factory method for creating dataset from season."""
        games = db_session.query(Game).filter_by(season_id=season).all()
        return cls(games=games, tokenizer=tokenizer)
```

3. **Protocol for Interfaces** - Duck typing with static checking

```python
from typing import Protocol

class FeatureCalculator(Protocol):
    """Interface for feature calculators."""

    def fit(self, data: pd.DataFrame) -> None: ...
    def transform(self, data: pd.DataFrame) -> np.ndarray: ...
    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...

# Any class implementing these methods satisfies the Protocol
class RAPMCalculator:
    def fit(self, data: pd.DataFrame) -> None: ...
    def transform(self, data: pd.DataFrame) -> np.ndarray: ...
    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...
```

4. **Context Managers** - For resource management

```python
from contextlib import contextmanager

class NBAApiClient:
    @contextmanager
    def rate_limited(self) -> Iterator[None]:
        """Context manager ensuring rate limit compliance."""
        yield
        time.sleep(self.delay)

    def get_games(self, season: str) -> pd.DataFrame:
        with self.rate_limited():
            return leaguegamefinder.LeagueGameFinder(
                season_nullable=season
            ).get_data_frames()[0]
```

### 4.3 Separation of Concerns

| Layer | Responsibility | Examples |
|-------|----------------|----------|
| **Data** | Storage, retrieval, external APIs | `api.py`, `models.py`, `collectors/` |
| **Features** | Transformations, calculations | `rapm.py`, `spatial.py`, `fatigue.py` |
| **Models** | ML architectures, training | `transformer.py`, `gnn.py`, `trainer.py` |
| **Business Logic** | Betting rules, signals | `kelly.py`, `devig.py`, `signals.py` |
| **Presentation** | Output formatting, UI | `reports.py`, `dashboard.py` |
| **CLI** | User interface | `cli.py` |

**Cross-Layer Communication:**

```
CLI → Config → Business Logic → Features → Data
                    ↓
                 Models
                    ↓
               Presentation
```

---

## 5. Best Practices

### 5.1 Error Handling

**Raise specific exceptions with context:**

```python
# Define custom exceptions in each module
class DataCollectionError(Exception):
    """Base exception for data collection errors."""

class RateLimitExceeded(DataCollectionError):
    """Raised when API rate limit is hit."""

class GameNotFound(DataCollectionError):
    """Raised when requested game doesn't exist."""

# Always include context
def fetch_game(game_id: str) -> pd.DataFrame:
    try:
        response = self.client.get_boxscore(game_id)
    except requests.HTTPError as e:
        if e.response.status_code == 429:
            raise RateLimitExceeded(
                f"Rate limit exceeded while fetching game {game_id}. "
                f"Retry after {e.response.headers.get('Retry-After', 'unknown')} seconds."
            ) from e
        elif e.response.status_code == 404:
            raise GameNotFound(f"Game {game_id} not found in NBA API.") from e
        raise DataCollectionError(
            f"HTTP {e.response.status_code} fetching game {game_id}"
        ) from e
```

**Never silently swallow exceptions:**

```python
# FORBIDDEN
try:
    result = risky_operation()
except Exception:
    pass  # NEVER DO THIS

# CORRECT: Log and re-raise or handle explicitly
try:
    result = risky_operation()
except SpecificError as e:
    logger.warning(f"Operation failed: {e}, using fallback")
    result = fallback_value
```

### 5.2 Logging

Use **Loguru** with structured logging:

```python
from loguru import logger

# Configure in config.py
logger.add(
    "logs/nba_model_{time}.log",
    rotation="1 day",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
    serialize=True,  # JSON format for parsing
)

# Usage patterns
logger.info("Starting data collection for season {}", season)
logger.debug("Processing game {} ({}/{})", game_id, idx, total)
logger.warning("Missing play-by-play for game {}, skipping", game_id)
logger.error("Failed to calculate RAPM: {}", error)

# With context binding
with logger.contextualize(game_id=game_id, season=season):
    logger.info("Processing game")
    # All logs in this block include game_id and season
```

**Log Levels:**

| Level | Usage |
|-------|-------|
| `DEBUG` | Detailed diagnostic info (loop iterations, intermediate values) |
| `INFO` | Normal operations (starting/completing tasks, counts) |
| `WARNING` | Unexpected but handled situations (missing data, fallbacks) |
| `ERROR` | Failures that don't crash the system |
| `CRITICAL` | System failures requiring immediate attention |

### 5.3 Configuration Management

Use **Pydantic Settings** with environment variable support:

```python
# nba_model/config.py
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Database
    db_path: str = Field(default="data/nba.db", alias="NBA_DB_PATH")

    # API
    api_delay: float = Field(default=0.6, alias="NBA_API_DELAY")
    api_max_retries: int = Field(default=3, alias="NBA_API_MAX_RETRIES")

    # Model
    model_dir: str = Field(default="data/models", alias="NBA_MODEL_DIR")

    # Training
    learning_rate: float = Field(default=1e-4, alias="LEARNING_RATE")
    batch_size: int = Field(default=32, alias="BATCH_SIZE")

    # Kelly
    kelly_fraction: float = Field(default=0.25, alias="KELLY_FRACTION")
    max_bet_pct: float = Field(default=0.02, alias="MAX_BET_PCT")
    min_edge_pct: float = Field(default=0.02, alias="MIN_EDGE_PCT")

# Usage - singleton pattern
_settings: Settings | None = None

def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
```

**Example .env.example:**

```bash
# Database
NBA_DB_PATH=data/nba.db

# API Configuration
NBA_API_DELAY=0.6
NBA_API_MAX_RETRIES=3

# Model Training
LEARNING_RATE=0.0001
BATCH_SIZE=32

# Betting Parameters
KELLY_FRACTION=0.25
MAX_BET_PCT=0.02
MIN_EDGE_PCT=0.02
```

### 5.4 Secrets Handling

**NEVER commit secrets.** Use environment variables for all sensitive data.

```python
# .gitignore MUST include:
.env
*.pem
*credentials*
*secret*

# For CI/CD, use GitHub Secrets or similar
# Access via environment variables only

# FORBIDDEN patterns:
API_KEY = "sk-abc123..."  # NEVER hardcode
password = open("password.txt").read()  # NEVER store in files
```

---

## 6. Common Libraries

### 6.1 Required Dependencies

All versions are **minimum required versions**:

```toml
[project]
dependencies = [
    # CLI Framework
    "typer>=0.9.0",          # Modern CLI with type hints
    "rich>=13.0",            # Beautiful terminal output

    # Data Layer
    "nba_api>=1.4",          # NBA stats API wrapper
    "sqlalchemy>=2.0",       # ORM for SQLite
    "pandas>=2.0",           # Data manipulation
    "numpy>=1.24",           # Numerical computing

    # Machine Learning
    "torch>=2.0",            # Deep learning framework
    "torch-geometric>=2.4",  # Graph neural networks
    "scikit-learn>=1.3",     # Traditional ML, Ridge regression
    "scipy>=1.11",           # Scientific computing, sparse matrices

    # Feature Engineering
    "haversine>=2.8",        # Distance calculations for travel

    # Configuration
    "pydantic>=2.0",         # Data validation
    "pydantic-settings>=2.0", # Settings management
    "python-dotenv>=1.0",    # Environment file loading

    # Utilities
    "tqdm>=4.66",            # Progress bars
    "loguru>=0.7",           # Structured logging
    "jinja2>=3.1",           # Template engine for dashboard
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4",           # Testing framework
    "pytest-cov>=4.1",       # Coverage reporting
    "pytest-asyncio>=0.21",  # Async test support
    "black>=23.0",           # Code formatting
    "ruff>=0.1",             # Fast linting
    "mypy>=1.5",             # Static type checking
    "pandas-stubs>=2.0",     # Pandas type stubs
]
```

### 6.2 Library Selection Guidelines

| Task | Use This | Not This | Reason |
|------|----------|----------|--------|
| Ridge Regression (RAPM) | `sklearn.linear_model.Ridge` | PyTorch | Simpler, proven, supports sparse |
| Sparse Matrices | `scipy.sparse.csr_matrix` | Dense numpy | Memory efficient for RAPM design matrix |
| Neural Networks | PyTorch | TensorFlow | Better PyG integration, dynamic graphs |
| Graph Networks | `torch_geometric.nn.GATv2Conv` | Manual implementation | Battle-tested, optimized |
| Data Validation | Pydantic | Manual checks | Type safety, auto-documentation |
| CLI | Typer | argparse/Click | Modern, type-hinted, rich integration |
| Logging | Loguru | stdlib logging | Simpler API, structured output |
| Geospatial | `haversine` | Manual calculation | Tested, handles edge cases |
| Convex Hulls | `scipy.spatial.ConvexHull` | Manual | Efficient, handles degenerates |

### 6.3 PyTorch vs scikit-learn Decision Tree

```
Is this a neural network (Transformer, GNN, MLP with >2 layers)?
├── Yes → PyTorch
└── No → Is this a simple regression/classification?
    ├── Yes → scikit-learn
    └── No → Does it need GPU acceleration for large data?
        ├── Yes → PyTorch
        └── No → scikit-learn
```

**Specific Mappings:**

| Component | Library | Reason |
|-----------|---------|--------|
| RAPM Ridge Regression | sklearn | Sparse matrix support, simple fit |
| Transformer Encoder | PyTorch | Flexible attention implementation |
| GATv2 | PyTorch Geometric | Specialized GNN library |
| Two-Tower Fusion | PyTorch | Multi-task learning |
| Kelly Criterion | numpy/scipy | Simple math, no ML needed |
| Drift Detection (KS test) | scipy.stats | Statistical testing |
| Convex Hull | scipy.spatial | Computational geometry |

---

## 7. Testing Expectations

### 7.1 Coverage Requirements

| Category | Minimum Coverage | Target Coverage |
|----------|------------------|-----------------|
| Unit Tests | 80% | 90% |
| Integration Tests | 60% | 80% |
| Overall | 75% | 85% |

**Enforce with:**

```bash
pytest --cov=nba_model --cov-fail-under=75
```

### 7.2 Test Naming Convention

```python
# Pattern: test_<function_name>_<scenario>_<expected_outcome>

# Unit tests
def test_calculate_kelly_with_positive_edge_returns_bet_size():
    ...

def test_calculate_kelly_with_no_edge_returns_zero():
    ...

def test_calculate_kelly_with_invalid_prob_raises_value_error():
    ...

# Integration tests
def test_data_pipeline_processes_full_season():
    ...

def test_training_pipeline_improves_loss_over_epochs():
    ...
```

### 7.3 Test Structure

```python
# tests/unit/features/test_rapm.py

import pytest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from nba_model.features.rapm import RAPMCalculator, InsufficientDataError


class TestRAPMCalculator:
    """Tests for RAPMCalculator class."""

    @pytest.fixture
    def sample_stints(self) -> pd.DataFrame:
        """Create minimal stint data for testing."""
        return pd.DataFrame({
            'stint_id': [1, 2, 3],
            'home_lineup': [[1, 2, 3, 4, 5]] * 3,
            'away_lineup': [[6, 7, 8, 9, 10]] * 3,
            'point_diff_per_100': [5.0, -3.0, 2.0],
            'possessions': [10.0, 8.0, 12.0],
            'game_date': pd.to_datetime(['2024-01-01'] * 3),
        })

    @pytest.fixture
    def calculator(self) -> RAPMCalculator:
        """Create calculator with test parameters."""
        return RAPMCalculator(lambda_=100, min_minutes=0)

    def test_build_design_matrix_creates_sparse_matrix(
        self,
        calculator: RAPMCalculator,
        sample_stints: pd.DataFrame,
    ) -> None:
        """Design matrix should be sparse with correct dimensions."""
        X, y, w = calculator.build_stint_matrix(sample_stints, players=list(range(1, 11)))

        assert isinstance(X, csr_matrix)
        assert X.shape == (3, 10)  # 3 stints, 10 players
        assert len(y) == 3
        assert len(w) == 3

    def test_fit_returns_coefficients_for_all_players(
        self,
        calculator: RAPMCalculator,
        sample_stints: pd.DataFrame,
    ) -> None:
        """Fit should return RAPM tuple for each player."""
        result = calculator.fit(sample_stints)

        assert len(result) == 10
        for player_id, (orapm, drapm, total) in result.items():
            assert isinstance(orapm, float)
            assert isinstance(drapm, float)
            assert abs(total - (orapm + drapm)) < 0.01

    def test_fit_with_insufficient_data_raises_error(
        self,
        calculator: RAPMCalculator,
    ) -> None:
        """Should raise error when not enough stints."""
        tiny_data = pd.DataFrame({
            'stint_id': [1],
            'home_lineup': [[1, 2, 3, 4, 5]],
            'away_lineup': [[6, 7, 8, 9, 10]],
            'point_diff_per_100': [5.0],
            'possessions': [10.0],
            'game_date': pd.to_datetime(['2024-01-01']),
        })

        with pytest.raises(InsufficientDataError, match="at least 100 stints"):
            calculator.fit(tiny_data)
```

### 7.4 Mocking Guidelines

**What to Mock:**

- External APIs (NBA API, any HTTP calls)
- Database sessions (use in-memory SQLite for integration tests)
- Current time (`datetime.now()`, `time.time()`)
- Random number generators (seed for reproducibility)
- File system operations (use `tmp_path` fixture)

**What NOT to Mock:**

- Internal class methods (test actual implementation)
- Pure functions (test with real inputs/outputs)
- Data transformations (test actual logic)

```python
# Mocking examples
from unittest.mock import Mock, patch

class TestNBAApiClient:
    @patch('nba_model.data.api.leaguegamefinder.LeagueGameFinder')
    def test_get_games_with_rate_limiting(self, mock_finder: Mock) -> None:
        """Should delay between API calls."""
        mock_finder.return_value.get_data_frames.return_value = [pd.DataFrame()]

        client = NBAApiClient(delay=0.1)

        with patch('time.sleep') as mock_sleep:
            client.get_games("2023-24")
            mock_sleep.assert_called_with(0.1)


class TestInferencePipeline:
    @pytest.fixture
    def mock_registry(self) -> Mock:
        """Mock model registry."""
        registry = Mock(spec=ModelRegistry)
        registry.load_model.return_value = {
            'transformer': Mock(),
            'gnn': Mock(),
            'fusion': Mock(),
        }
        return registry

    def test_predict_game_uses_latest_model_by_default(
        self,
        mock_registry: Mock,
        db_session: Session,
    ) -> None:
        """Should load 'latest' version when not specified."""
        pipeline = InferencePipeline(mock_registry, db_session)

        mock_registry.load_model.assert_called_with('latest')
```

### 7.5 Fixtures Location

```
tests/
├── conftest.py              # Shared fixtures (db_session, sample data loaders)
├── unit/
│   ├── conftest.py          # Unit test specific fixtures
│   └── features/
│       └── conftest.py      # Feature test specific fixtures
└── fixtures/
    ├── sample_games.json    # Static test data files
    └── sample_stints.parquet
```

---

## 8. Git Workflow

### 8.1 Branch Naming

```
<type>/<ticket-id>-<short-description>
```

| Type | Usage | Example |
|------|-------|---------|
| `feature/` | New functionality | `feature/NBA-123-add-rapm-calculator` |
| `fix/` | Bug fixes | `fix/NBA-456-handle-missing-pbp-data` |
| `refactor/` | Code improvements | `refactor/NBA-789-optimize-stint-query` |
| `docs/` | Documentation only | `docs/NBA-101-update-api-reference` |
| `test/` | Test additions/fixes | `test/NBA-102-add-kelly-edge-cases` |
| `chore/` | Maintenance | `chore/NBA-103-upgrade-dependencies` |

### 8.2 Commit Message Format

Use **Conventional Commits** specification:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

**Types:**

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation changes |
| `style` | Formatting (no code change) |
| `refactor` | Code restructuring (no behavior change) |
| `perf` | Performance improvement |
| `test` | Test additions/modifications |
| `build` | Build system changes |
| `ci` | CI/CD changes |
| `chore` | Maintenance tasks |

**Scopes:**

| Scope | Usage |
|-------|-------|
| `data` | Data layer (collectors, schema) |
| `features` | Feature engineering |
| `models` | ML models |
| `backtest` | Backtesting engine |
| `monitor` | Drift detection, triggers |
| `predict` | Inference pipeline |
| `output` | Reports, dashboard |
| `cli` | CLI commands |
| `config` | Configuration |

**Examples:**

```bash
# Feature
feat(features): add convex hull spacing calculator

Implement SpacingCalculator class with hull area and centroid
calculation from shot location data.

Closes NBA-234

# Bug fix
fix(data): handle missing play-by-play gracefully

Previously, games without PBP data caused collector to crash.
Now logs warning and continues to next game.

Fixes NBA-567

# Breaking change
feat(models)!: change transformer output dimension

BREAKING CHANGE: Transformer output changed from 64 to 128 dims.
Existing model weights are incompatible.

# Multiple scopes
feat(features,data): add RAPM calculation with stint storage
```

### 8.3 Pull Request Requirements

**PR Title Format:**

```
[TYPE] Brief description of changes
```

Examples:
- `[FEATURE] Add Bayesian injury adjustment pipeline`
- `[FIX] Handle rate limiting in NBA API client`
- `[REFACTOR] Split collector into separate modules`

**PR Template:**

```markdown
## Summary
<!-- 2-3 sentence description of what this PR does -->

## Changes
- Added X
- Modified Y
- Removed Z

## Testing
<!-- How were these changes tested? -->
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing performed

## Screenshots
<!-- If applicable, add screenshots -->

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings introduced
- [ ] All tests pass (`pytest --cov=nba_model`)
- [ ] Type checking passes (`mypy nba_model/ --strict`)
- [ ] Linting passes (`ruff check .`)
```

### 8.4 Code Review Expectations

**Reviewer Responsibilities:**

1. Verify code follows all guidelines in this document
2. Check for logical correctness
3. Ensure adequate test coverage
4. Validate documentation accuracy
5. Approve or request changes within 24 hours (business days)

**Author Responsibilities:**

1. Self-review before requesting review
2. Respond to feedback within 24 hours
3. Keep PRs focused (<400 lines when possible)
4. Provide context for complex changes

**Approval Requirements:**

| Change Type | Required Approvals |
|-------------|-------------------|
| Documentation only | 1 |
| Bug fix | 1 |
| New feature | 2 |
| Architecture change | 2 + team lead |
| Security-related | 2 + security review |

---

## 9. Definition of Done

A feature is **DONE** only when ALL of the following are satisfied:

### 9.1 Code Quality

- [ ] Code follows all standards in this document
- [ ] All new code has type hints
- [ ] All public functions/classes have docstrings
- [ ] No `# type: ignore` without explanatory comment
- [ ] No `TODO` or `FIXME` comments left unaddressed
- [ ] No dead code or commented-out code
- [ ] Black formatting applied (`black .`)
- [ ] Ruff linting passes with zero errors (`ruff check .`)
- [ ] MyPy strict mode passes (`mypy nba_model/ --strict`)

### 9.2 Testing

- [ ] Unit tests written for all new functionality
- [ ] Unit test coverage ≥ 80% for new code
- [ ] Integration tests written if feature spans modules
- [ ] All existing tests pass (`pytest`)
- [ ] Edge cases and error paths tested
- [ ] No flaky tests introduced

### 9.3 Documentation

- [ ] Docstrings complete and accurate
- [ ] README updated if user-facing changes
- [ ] Glossary updated if new terms introduced
- [ ] CLI help text updated for new commands
- [ ] Architecture decisions documented (if applicable)

### 9.4 Review & Merge

- [ ] PR created with complete description
- [ ] Required approvals obtained
- [ ] All review comments addressed
- [ ] CI pipeline passes
- [ ] Merged to main via squash commit
- [ ] Branch deleted after merge

### 9.5 Deployment Readiness

- [ ] No hardcoded secrets or credentials
- [ ] Environment variables documented in `.env.example`
- [ ] Database migrations included (if schema changes)
- [ ] Model weights compatible with changes (or version bumped)
- [ ] Dashboard updated if output format changed

---

## Appendix: Quick Reference

### A.1 Common Commands

```bash
# Development setup
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Format and lint
black .
ruff check . --fix
mypy nba_model/ --strict

# Run tests
pytest                              # All tests
pytest --cov=nba_model              # With coverage
pytest -k "test_rapm"               # Filter by name
pytest tests/unit/                  # Unit tests only
pytest -x                           # Stop on first failure

# CLI commands
python -m nba_model --help          # Show all commands
python -m nba_model data collect    # Collect data
python -m nba_model train all       # Train models
python -m nba_model predict today   # Today's predictions
```

### A.2 File Templates

**New Module Template:**

```python
"""Brief module description.

Extended description if needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

# Constants
DEFAULT_VALUE: int = 100

# Types
@dataclass
class ResultType:
    """Container for results."""
    value: float
    metadata: dict[str, str]

# Main class
class Calculator:
    """Main implementation.

    Longer description of the class purpose and usage.

    Attributes:
        param1: Description of param1.
        param2: Description of param2.
    """

    def __init__(self, param1: int, param2: float = 1.0) -> None:
        self.param1 = param1
        self.param2 = param2

    def calculate(self, data: pd.DataFrame) -> ResultType:
        """Calculate something.

        Args:
            data: Input data with columns X, Y, Z.

        Returns:
            ResultType containing the calculation result.

        Raises:
            ValueError: If data is empty.
        """
        if data.empty:
            raise ValueError("Data cannot be empty")

        return ResultType(value=0.0, metadata={})
```

### A.3 Pre-commit Hook

Add to `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.0
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.8
    hooks:
      - id: ruff
        args: [--fix]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.1
    hooks:
      - id: mypy
        additional_dependencies: [pydantic, pandas-stubs]
        args: [--strict]
```

---

*This document is version-controlled. All changes require review and approval.*

*Last updated: 2026-01-31*
