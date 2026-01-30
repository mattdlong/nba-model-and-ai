# NBA Quantitative Trading Strategy - Implementation Plan

## Executive Summary

This document outlines the implementation plan for a "Gold Standard" NBA quantitative trading strategy built as a Python CLI application. The system integrates Transformer models, Graph Neural Networks (GNNs), and a Two-Tower fusion architecture to predict NBA game outcomes and generate betting signals using exclusively public data.

**Technology Stack:**
- **Language:** Python 3.11+
- **Data Source:** nba_api library
- **Database:** SQLite with SQLAlchemy ORM
- **CLI Framework:** Typer (modern, type-hinted)
- **ML Framework:** PyTorch + PyTorch Geometric
- **Dashboard:** GitHub Pages (static site generation)

---

## Phase 1: Application Structure

### Objectives
- Establish a clean, maintainable project structure
- Configure development environment and dependencies
- Build the CLI command interface
- Set up logging and configuration management

### Key Tasks

#### 1.1 Project Layout
```
nba-model/
├── pyproject.toml              # Project metadata and dependencies
├── README.md
├── .env.example                # Environment template
├── nba_model/
│   ├── __init__.py
│   ├── cli.py                  # Main CLI entrypoint (Typer app)
│   ├── config.py               # Configuration management (Pydantic)
│   │
│   ├── data/                   # Phase 2: Data layer
│   │   ├── __init__.py
│   │   ├── api.py              # NBA API client wrapper
│   │   ├── models.py           # SQLAlchemy ORM models
│   │   ├── schema.py           # Database schema definitions
│   │   ├── collectors/         # Data collection modules
│   │   │   ├── games.py
│   │   │   ├── players.py
│   │   │   ├── playbyplay.py
│   │   │   └── shots.py
│   │   └── pipelines.py        # ETL orchestration
│   │
│   ├── features/               # Phase 3: Feature engineering
│   │   ├── __init__.py
│   │   ├── rapm.py             # Regularized Adjusted Plus-Minus
│   │   ├── spatial.py          # Convex hull spacing metrics
│   │   ├── fatigue.py          # Rest/travel/load metrics
│   │   ├── parsing.py          # Regex event extraction
│   │   └── normalization.py    # Z-score by season
│   │
│   ├── models/                 # Phase 4: ML models
│   │   ├── __init__.py
│   │   ├── transformer.py      # Sequence model (game flow)
│   │   ├── gnn.py              # GATv2 player interaction model
│   │   ├── fusion.py           # Two-Tower architecture
│   │   ├── trainer.py          # Training loop
│   │   └── registry.py         # Model versioning/registry
│   │
│   ├── backtest/               # Phase 5: Backtesting
│   │   ├── __init__.py
│   │   ├── engine.py           # Walk-forward validation
│   │   ├── kelly.py            # Kelly criterion sizing
│   │   ├── devig.py            # Vig removal methods
│   │   └── metrics.py          # Performance metrics
│   │
│   ├── monitor/                # Phase 6: Self-improvement
│   │   ├── __init__.py
│   │   ├── drift.py            # Covariate drift detection
│   │   ├── triggers.py         # Retraining logic
│   │   └── versioning.py       # Model version management
│   │
│   ├── predict/                # Phase 7: Inference
│   │   ├── __init__.py
│   │   ├── inference.py        # Prediction pipeline
│   │   ├── injuries.py         # Bayesian injury adjustment
│   │   └── signals.py          # Betting signal generation
│   │
│   └── output/                 # Phase 8: Output/Dashboard
│       ├── __init__.py
│       ├── reports.py          # Report generation
│       └── dashboard.py        # GitHub Pages export
│
├── data/                       # Data storage (gitignored)
│   ├── nba.db                  # SQLite database
│   └── models/                 # Saved model weights
│
├── docs/                       # GitHub Pages source
│   ├── index.html
│   └── assets/
│
└── tests/
    ├── conftest.py
    ├── test_data/
    ├── test_features/
    └── test_models/
```

#### 1.2 Dependencies (pyproject.toml)
```toml
[project]
name = "nba-model"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
    # CLI
    "typer>=0.9.0",
    "rich>=13.0",

    # Data
    "nba_api>=1.4",
    "sqlalchemy>=2.0",
    "pandas>=2.0",
    "numpy>=1.24",

    # ML/DL
    "torch>=2.0",
    "torch-geometric>=2.4",
    "scikit-learn>=1.3",
    "scipy>=1.11",

    # Feature Engineering
    "haversine>=2.8",

    # Configuration
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "python-dotenv>=1.0",

    # Utilities
    "tqdm>=4.66",
    "loguru>=0.7",
    "jinja2>=3.1",  # Dashboard templates
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "black>=23.0",
    "ruff>=0.1",
    "mypy>=1.5",
]
```

#### 1.3 CLI Command Structure
```
nba-model
├── data
│   ├── collect        # Collect data from NBA API
│   ├── update         # Incremental data update
│   └── status         # Show database statistics
│
├── features
│   ├── build          # Build all feature tables
│   ├── rapm           # Calculate RAPM coefficients
│   └── spatial        # Calculate convex hull metrics
│
├── train
│   ├── transformer    # Train sequence model
│   ├── gnn            # Train graph model
│   ├── fusion         # Train fusion model
│   └── all            # Full training pipeline
│
├── backtest
│   ├── run            # Run walk-forward backtest
│   ├── report         # Generate backtest report
│   └── optimize       # Optimize Kelly fraction
│
├── monitor
│   ├── drift          # Check for covariate drift
│   ├── trigger        # Evaluate retraining triggers
│   └── versions       # List model versions
│
├── predict
│   ├── today          # Predictions for today's games
│   ├── game           # Single game prediction
│   └── signals        # Generate betting signals
│
└── dashboard
    ├── build          # Build GitHub Pages site
    └── deploy         # Deploy to GitHub Pages
```

### Technical Specifications
- **Configuration:** Pydantic Settings with `.env` file support
- **Logging:** Loguru with file rotation and structured JSON output
- **Database Path:** Configurable via `NBA_DB_PATH` environment variable
- **Rate Limiting:** Built-in delay between NBA API calls (default 0.6s)

### Dependencies on Other Phases
- None (foundation phase)

### Estimated Complexity
- **CLI Framework:** Low
- **Project Setup:** Low
- **Configuration System:** Medium

---

## Phase 2: Data Collection & Storage

### Objectives
- Design comprehensive SQLite schema for NBA data
- Implement robust NBA API data collectors
- Build ETL pipelines with error handling and resumability
- Enable incremental updates for daily operations

### Key Tasks

#### 2.1 SQLite Schema Design

```sql
-- Core Tables
CREATE TABLE seasons (
    season_id TEXT PRIMARY KEY,  -- e.g., "2023-24"
    start_date DATE,
    end_date DATE,
    games_count INTEGER
);

CREATE TABLE teams (
    team_id INTEGER PRIMARY KEY,
    abbreviation TEXT,
    full_name TEXT,
    city TEXT,
    arena_name TEXT,
    arena_lat REAL,
    arena_lon REAL
);

CREATE TABLE players (
    player_id INTEGER PRIMARY KEY,
    full_name TEXT,
    height_inches INTEGER,
    weight_lbs INTEGER,
    birth_date DATE,
    draft_year INTEGER,
    draft_round INTEGER,
    draft_number INTEGER
);

CREATE TABLE player_seasons (
    id INTEGER PRIMARY KEY,
    player_id INTEGER REFERENCES players,
    season_id TEXT REFERENCES seasons,
    team_id INTEGER REFERENCES teams,
    position TEXT,
    jersey_number TEXT,
    UNIQUE(player_id, season_id, team_id)
);

-- Game Tables
CREATE TABLE games (
    game_id TEXT PRIMARY KEY,  -- NBA's GAME_ID format
    season_id TEXT REFERENCES seasons,
    game_date DATE,
    home_team_id INTEGER REFERENCES teams,
    away_team_id INTEGER REFERENCES teams,
    home_score INTEGER,
    away_score INTEGER,
    status TEXT,  -- 'scheduled', 'completed', 'postponed'
    attendance INTEGER
);

CREATE TABLE game_stats (
    id INTEGER PRIMARY KEY,
    game_id TEXT REFERENCES games,
    team_id INTEGER REFERENCES teams,
    is_home BOOLEAN,
    -- Basic Stats
    points INTEGER,
    rebounds INTEGER,
    assists INTEGER,
    steals INTEGER,
    blocks INTEGER,
    turnovers INTEGER,
    -- Advanced Stats (from boxscoreadvancedv2)
    offensive_rating REAL,
    defensive_rating REAL,
    pace REAL,
    efg_pct REAL,
    tov_pct REAL,
    orb_pct REAL,
    ft_rate REAL
);

CREATE TABLE player_game_stats (
    id INTEGER PRIMARY KEY,
    game_id TEXT REFERENCES games,
    player_id INTEGER REFERENCES players,
    team_id INTEGER REFERENCES teams,
    minutes REAL,
    points INTEGER,
    rebounds INTEGER,
    assists INTEGER,
    steals INTEGER,
    blocks INTEGER,
    turnovers INTEGER,
    fgm INTEGER,
    fga INTEGER,
    fg3m INTEGER,
    fg3a INTEGER,
    ftm INTEGER,
    fta INTEGER,
    plus_minus INTEGER,
    -- Player tracking (from boxscoreplayertrackv2)
    distance_miles REAL,
    speed_avg REAL
);

-- Play-by-Play Tables
CREATE TABLE plays (
    id INTEGER PRIMARY KEY,
    game_id TEXT REFERENCES games,
    event_num INTEGER,
    period INTEGER,
    pc_time TEXT,  -- "MM:SS" format
    wc_time TEXT,  -- Wall clock time
    event_type INTEGER,  -- EVENTMSGTYPE
    event_action INTEGER,  -- EVENTMSGACTIONTYPE
    home_description TEXT,
    away_description TEXT,
    neutral_description TEXT,
    score_home INTEGER,
    score_away INTEGER,
    player1_id INTEGER REFERENCES players,
    player2_id INTEGER REFERENCES players,
    player3_id INTEGER REFERENCES players,
    team_id INTEGER REFERENCES teams,
    UNIQUE(game_id, event_num)
);

CREATE TABLE shots (
    id INTEGER PRIMARY KEY,
    game_id TEXT REFERENCES games,
    player_id INTEGER REFERENCES players,
    team_id INTEGER REFERENCES teams,
    period INTEGER,
    minutes_remaining INTEGER,
    seconds_remaining INTEGER,
    action_type TEXT,
    shot_type TEXT,  -- '2PT', '3PT'
    shot_zone_basic TEXT,
    shot_zone_area TEXT,
    shot_zone_range TEXT,
    shot_distance INTEGER,
    loc_x INTEGER,
    loc_y INTEGER,
    made BOOLEAN,
    UNIQUE(game_id, player_id, period, minutes_remaining, seconds_remaining, loc_x, loc_y)
);

-- Lineup Tracking (derived from play-by-play)
CREATE TABLE stints (
    id INTEGER PRIMARY KEY,
    game_id TEXT REFERENCES games,
    period INTEGER,
    start_time TEXT,
    end_time TEXT,
    duration_seconds INTEGER,
    -- Home lineup (5 player IDs stored as JSON array)
    home_lineup TEXT,
    -- Away lineup (5 player IDs stored as JSON array)
    away_lineup TEXT,
    -- Stint outcomes
    home_points INTEGER,
    away_points INTEGER,
    possessions REAL
);

-- Market Data
CREATE TABLE odds (
    id INTEGER PRIMARY KEY,
    game_id TEXT REFERENCES games,
    source TEXT,  -- 'pinnacle', 'draftkings', etc.
    timestamp DATETIME,
    home_ml REAL,  -- Money line decimal odds
    away_ml REAL,
    spread_home REAL,
    spread_home_odds REAL,
    spread_away_odds REAL,
    total REAL,
    over_odds REAL,
    under_odds REAL
);

-- Feature Tables (populated by Phase 3)
CREATE TABLE player_rapm (
    id INTEGER PRIMARY KEY,
    player_id INTEGER REFERENCES players,
    season_id TEXT REFERENCES seasons,
    calculation_date DATE,
    orapm REAL,  -- Offensive RAPM
    drapm REAL,  -- Defensive RAPM
    rapm REAL,   -- Total RAPM
    sample_stints INTEGER,
    UNIQUE(player_id, season_id, calculation_date)
);

CREATE TABLE lineup_spacing (
    id INTEGER PRIMARY KEY,
    season_id TEXT REFERENCES seasons,
    lineup_hash TEXT,  -- Hash of sorted 5 player IDs
    player_ids TEXT,   -- JSON array of 5 player IDs
    hull_area REAL,
    centroid_x REAL,
    centroid_y REAL,
    shot_count INTEGER,
    UNIQUE(season_id, lineup_hash)
);

-- Season Normalization Stats
CREATE TABLE season_stats (
    id INTEGER PRIMARY KEY,
    season_id TEXT REFERENCES seasons,
    metric_name TEXT,
    mean_value REAL,
    std_value REAL,
    min_value REAL,
    max_value REAL,
    UNIQUE(season_id, metric_name)
);

-- Indexes for common queries
CREATE INDEX idx_games_date ON games(game_date);
CREATE INDEX idx_games_season ON games(season_id);
CREATE INDEX idx_plays_game ON plays(game_id);
CREATE INDEX idx_shots_game ON shots(game_id);
CREATE INDEX idx_shots_player ON shots(player_id);
CREATE INDEX idx_stints_game ON stints(game_id);
CREATE INDEX idx_player_game_stats ON player_game_stats(game_id, player_id);
```

#### 2.2 NBA API Client Wrapper

```python
# nba_model/data/api.py - Design specification

class NBAApiClient:
    """
    Wrapper around nba_api with:
    - Automatic rate limiting (configurable delay)
    - Retry logic with exponential backoff
    - Response caching (optional)
    - Error handling and logging
    """

    def __init__(self, delay: float = 0.6, max_retries: int = 3):
        pass

    def get_league_game_finder(self, season: str, **kwargs) -> pd.DataFrame:
        """Fetch games for a season using leaguegamefinder"""
        pass

    def get_play_by_play(self, game_id: str) -> pd.DataFrame:
        """Fetch play-by-play using playbyplayv2"""
        pass

    def get_shot_chart(self, game_id: str = None, player_id: str = None,
                       season: str = None) -> pd.DataFrame:
        """Fetch shot chart using shotchartdetail"""
        pass

    def get_boxscore_advanced(self, game_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch advanced boxscore (team and player)"""
        pass

    def get_player_tracking(self, game_id: str) -> pd.DataFrame:
        """Fetch player tracking data using boxscoreplayertrackv2"""
        pass

    def get_team_roster(self, team_id: int, season: str) -> pd.DataFrame:
        """Fetch team roster using commonteamroster"""
        pass
```

#### 2.3 Data Collection Pipeline

**Collectors:**
1. **GamesCollector** - Fetches all games for specified seasons
2. **PlayersCollector** - Builds player master table from rosters
3. **PlayByPlayCollector** - Fetches play-by-play for each game
4. **ShotsCollector** - Fetches shot chart data
5. **BoxScoreCollector** - Fetches advanced box scores

**ETL Features:**
- Checkpointing: Track last successful game_id to enable resumption
- Batch processing: Commit every N games to avoid memory issues
- Validation: Schema validation before insert
- Deduplication: UNIQUE constraints handle re-runs gracefully

#### 2.4 Incremental Update Strategy

```python
# Daily update workflow
def daily_update():
    """
    1. Fetch games from yesterday that aren't in DB
    2. For each new completed game:
       - Fetch and store play-by-play
       - Fetch and store shot chart
       - Fetch and store box scores
    3. Update derived tables (stints)
    4. Log statistics
    """
    pass
```

### Technical Specifications

| Endpoint | Rate Limit | Estimated Rows/Season | Storage |
|----------|------------|----------------------|---------|
| leaguegamefinder | 0.6s | ~1,300 games | ~100KB |
| playbyplayv2 | 0.6s | ~400 events/game | ~50MB |
| shotchartdetail | 0.6s | ~80 shots/game | ~10MB |
| boxscoreadvancedv2 | 0.6s | ~30 players/game | ~5MB |

**Total estimated DB size:** ~200MB per season (5 seasons = ~1GB)

### Dependencies on Other Phases
- Phase 1: CLI structure and configuration

### Estimated Complexity
- **Schema Design:** Medium
- **API Client:** Medium
- **ETL Pipeline:** High (error handling, resumability)
- **Stint Derivation:** High (complex play-by-play parsing)

---

## Phase 3: Feature Engineering

### Objectives
- Implement RAPM calculation using Ridge Regression
- Build spatial analysis (Convex Hulls) for lineup spacing
- Calculate fatigue and travel metrics
- Create season-normalized features
- Parse play-by-play text for derived events

### Key Tasks

#### 3.1 Regularized Adjusted Plus-Minus (RAPM)

**Mathematical Implementation:**

```python
# nba_model/features/rapm.py - Design specification

class RAPMCalculator:
    """
    Calculates Regularized Adjusted Plus-Minus using Ridge Regression.

    Model: Y = Xβ + ε
    Where:
        Y = point differential per 100 possessions for each stint
        X = sparse design matrix (1 for home players, -1 for away, 0 otherwise)
        β = RAPM coefficients (the values we solve for)

    Ridge regression minimizes: ||Y - Xβ||² + λ||β||²
    """

    def __init__(self,
                 lambda_: float = 5000,  # Regularization strength
                 min_minutes: int = 100,  # Minimum player minutes
                 time_decay_tau: float = 180):  # Days for half-weight
        pass

    def build_stint_matrix(self,
                           stints: pd.DataFrame,
                           players: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build the design matrix X, target Y, and weights W.

        Returns:
            X: (n_stints, n_players) sparse matrix
            Y: (n_stints,) point differential per 100 possessions
            W: (n_stints,) time decay weights
        """
        pass

    def fit(self, stints: pd.DataFrame) -> dict[int, tuple[float, float, float]]:
        """
        Fit Ridge Regression and return RAPM coefficients.

        Returns:
            Dict mapping player_id -> (orapm, drapm, total_rapm)
        """
        pass

    def cross_validate_lambda(self, stints: pd.DataFrame,
                              lambdas: list[float]) -> float:
        """Find optimal lambda via cross-validation."""
        pass
```

**Implementation Notes:**
- Use `scipy.sparse` for the design matrix (very sparse, ~10/500 entries per row)
- Split into ORAPM/DRAPM by running separate regressions on points scored/allowed
- Apply time-decay weighting: `w_i = exp(-(today - game_date) / tau)`
- Typical λ range: 1000-10000, optimize via CV

#### 3.2 Convex Hull Spacing Analysis

```python
# nba_model/features/spatial.py - Design specification

from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde

class SpacingCalculator:
    """
    Calculates floor spacing metrics using Convex Hulls of shot locations.
    """

    def calculate_lineup_spacing(self,
                                 player_ids: list[int],
                                 shots_df: pd.DataFrame) -> dict:
        """
        Calculate spacing metrics for a 5-player lineup.

        Returns:
            {
                'hull_area': float,      # Area of convex hull (sq units)
                'centroid_x': float,     # X coordinate of centroid
                'centroid_y': float,     # Y coordinate of centroid
                'avg_distance': float,   # Avg distance from basket
                'corner_density': float, # Density of shots in corners
                'shot_count': int        # Number of shots used
            }
        """
        pass

    def calculate_player_shot_density(self,
                                      player_id: int,
                                      shots_df: pd.DataFrame) -> np.ndarray:
        """
        Create KDE-based shot density map for a player.

        Returns:
            2D numpy array (court grid) with density values
        """
        pass

    def calculate_lineup_gravity_overlap(self,
                                         player_ids: list[int],
                                         shots_df: pd.DataFrame) -> float:
        """
        Calculate overlapping gravity zones (indicates spacing issues).
        """
        pass
```

**Court Coordinates:**
- NBA court is 94x50 feet
- Shot chart LOC_X, LOC_Y are in tenths of feet from basket
- LOC_X range: roughly -250 to 250
- LOC_Y range: roughly -50 to 900

#### 3.3 Fatigue and Travel Metrics

```python
# nba_model/features/fatigue.py - Design specification

from haversine import haversine, Unit

class FatigueCalculator:
    """
    Calculate fatigue-related features for teams and players.
    """

    # NBA arena coordinates (lat, lon)
    ARENA_COORDS = {
        'ATL': (33.757, -84.396),
        'BOS': (42.366, -71.062),
        # ... all 30 teams
    }

    def calculate_rest_days(self,
                            team_id: int,
                            game_date: date,
                            games_df: pd.DataFrame) -> int:
        """Days since last game."""
        pass

    def calculate_travel_distance(self,
                                  team_id: int,
                                  game_date: date,
                                  lookback_days: int = 7,
                                  games_df: pd.DataFrame) -> float:
        """
        Total miles traveled in last N days using haversine formula.
        """
        pass

    def calculate_schedule_flags(self,
                                 team_id: int,
                                 game_date: date,
                                 games_df: pd.DataFrame) -> dict:
        """
        Returns:
            {
                'back_to_back': bool,
                '3_in_4': bool,      # 3rd game in 4 nights
                '4_in_5': bool,      # 4th game in 5 nights
                'home_stand': int,   # Consecutive home games
                'road_trip': int     # Consecutive away games
            }
        """
        pass

    def calculate_player_load(self,
                              player_id: int,
                              game_date: date,
                              lookback_games: int = 5,
                              player_stats_df: pd.DataFrame) -> dict:
        """
        Returns:
            {
                'avg_minutes': float,
                'total_distance': float,  # Miles run in last N games
                'minutes_trend': float    # Increasing/decreasing load
            }
        """
        pass
```

#### 3.4 Play-by-Play Event Parsing

```python
# nba_model/features/parsing.py - Design specification

import re

class EventParser:
    """
    Extract derived events from play-by-play text descriptions.
    """

    # Regex patterns
    PATTERNS = {
        'bad_pass': re.compile(r'Bad Pass', re.IGNORECASE),
        'lost_ball': re.compile(r'Lost Ball', re.IGNORECASE),
        'steal': re.compile(r'STEAL', re.IGNORECASE),
        'block': re.compile(r'BLOCK', re.IGNORECASE),
        'driving': re.compile(r'Driving', re.IGNORECASE),
        'pullup': re.compile(r'Pull(-|\s)?[Uu]p', re.IGNORECASE),
        'step_back': re.compile(r'Step Back', re.IGNORECASE),
        'catch_shoot': re.compile(r'Catch and Shoot', re.IGNORECASE),
        'transition': re.compile(r'Fast Break|Transition', re.IGNORECASE),
        'contested': re.compile(r'Contested', re.IGNORECASE),
    }

    def parse_turnover_type(self, description: str) -> str:
        """Classify turnover as 'unforced' (bad pass) or 'forced' (lost ball)."""
        pass

    def parse_shot_context(self, description: str) -> dict:
        """
        Extract shot context.

        Returns:
            {
                'shot_type': str,     # 'driving', 'pullup', 'catch_shoot', etc.
                'is_transition': bool,
                'is_contested': bool
            }
        """
        pass

    def calculate_shot_clock_usage(self,
                                   plays_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate time between events to estimate shot clock usage.
        Categorize possessions as 'early' (<8s), 'mid' (8-16s), 'late' (>16s).
        """
        pass
```

#### 3.5 Season Normalization

```python
# nba_model/features/normalization.py - Design specification

class SeasonNormalizer:
    """
    Z-score normalization by season to handle pace/space evolution.

    z = (x - μ_season) / σ_season
    """

    METRICS_TO_NORMALIZE = [
        'pace', 'offensive_rating', 'defensive_rating',
        'efg_pct', 'tov_pct', 'orb_pct', 'ft_rate',
        'fg3a_rate', 'points_per_game'
    ]

    def fit(self, games_df: pd.DataFrame) -> None:
        """Calculate mean/std for each metric by season."""
        pass

    def transform(self, df: pd.DataFrame, season: str) -> pd.DataFrame:
        """Apply z-score normalization using stored stats."""
        pass

    def save_stats(self, db_session) -> None:
        """Persist normalization stats to season_stats table."""
        pass

    def load_stats(self, db_session, season: str) -> None:
        """Load normalization stats from database."""
        pass
```

### Technical Specifications

| Feature | Computation Time | Storage | Update Frequency |
|---------|-----------------|---------|------------------|
| RAPM | ~30s per season | ~50KB | Weekly |
| Convex Hulls | ~5min per season | ~1MB | Weekly |
| Fatigue | ~1s per game | ~10KB | Daily |
| Event Parsing | ~10s per game | ~100KB | With data collection |

### Dependencies on Other Phases
- Phase 2: Populated stints table (for RAPM)
- Phase 2: Populated shots table (for Convex Hulls)
- Phase 2: Populated games/plays tables (for all features)

### Estimated Complexity
- **RAPM:** High (sparse matrix algebra, cross-validation)
- **Convex Hulls:** Medium
- **Fatigue:** Low
- **Event Parsing:** Medium (regex edge cases)
- **Normalization:** Low

---

## Phase 4: Model Training

### Objectives
- Implement Transformer encoder for sequence modeling
- Implement GATv2 for player interaction modeling
- Build Two-Tower fusion architecture
- Create multi-task training pipeline
- Establish model registry and versioning

### Key Tasks

#### 4.1 Transformer Sequence Model

```python
# nba_model/models/transformer.py - Design specification

import torch
import torch.nn as nn

class GameFlowTransformer(nn.Module):
    """
    Transformer Encoder for modeling game flow sequences.

    Input: Sequence of game events (tokenized)
    Output: Sequence representation for fusion

    Architecture (from research):
        - Embedding dimension: 128
        - Attention heads: 4
        - Encoder layers: 2
        - Sequence length: 50 events
        - Dropout: 0.1
    """

    def __init__(self,
                 vocab_size: int,        # Number of event types
                 d_model: int = 128,
                 nhead: int = 4,
                 num_layers: int = 2,
                 max_seq_len: int = 50,
                 dropout: float = 0.1):
        super().__init__()

        # Event type embedding
        self.event_embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        # Additional embeddings
        self.time_embedding = nn.Linear(1, d_model // 4)  # Time remaining
        self.score_embedding = nn.Linear(1, d_model // 4)  # Score differential
        self.lineup_embedding = nn.Linear(20, d_model // 2)  # 10 players x 2 (home/away)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, events, times, scores, lineups, mask=None):
        """
        Args:
            events: (batch, seq_len) event type indices
            times: (batch, seq_len, 1) time remaining normalized
            scores: (batch, seq_len, 1) score differential normalized
            lineups: (batch, seq_len, 20) one-hot lineup encoding
            mask: (batch, seq_len) attention mask

        Returns:
            (batch, d_model) sequence representation (CLS token or mean pooling)
        """
        pass


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    pass


class EventTokenizer:
    """
    Tokenize play-by-play events into model inputs.

    Event types (EVENTMSGTYPE):
        1: Made shot
        2: Missed shot
        3: Free throw
        4: Rebound
        5: Turnover
        6: Foul
        7: Violation
        8: Substitution
        9: Timeout
        10: Jump ball
        12: Period start
        13: Period end
    """

    def __init__(self):
        self.event_vocab = {...}  # Map event types to indices

    def tokenize_game(self, plays_df: pd.DataFrame,
                      lineups_df: pd.DataFrame) -> dict:
        """
        Convert game plays to model inputs.

        Returns:
            {
                'events': np.array,   # (seq_len,) event indices
                'times': np.array,    # (seq_len, 1) normalized time
                'scores': np.array,   # (seq_len, 1) normalized score diff
                'lineups': np.array   # (seq_len, 20) lineup encoding
            }
        """
        pass
```

#### 4.2 Graph Neural Network (GATv2)

```python
# nba_model/models/gnn.py - Design specification

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data

class PlayerInteractionGNN(nn.Module):
    """
    GATv2 for modeling player interactions within a game.

    Graph structure:
        - Nodes: 10 players on court
        - Edges:
            - Teammate edges (fully connected within team)
            - Opponent edges (positional matchups)

    Node features:
        - RAPM (o/d/total)
        - Height, weight (normalized)
        - Season shooting percentages
        - Position embedding
    """

    def __init__(self,
                 node_features: int = 16,
                 hidden_dim: int = 64,
                 output_dim: int = 128,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()

        self.input_proj = nn.Linear(node_features, hidden_dim)

        self.gat_layers = nn.ModuleList([
            GATv2Conv(
                in_channels=hidden_dim if i == 0 else hidden_dim * num_heads,
                out_channels=hidden_dim,
                heads=num_heads,
                dropout=dropout,
                concat=True if i < num_layers - 1 else False
            )
            for i in range(num_layers)
        ])

        self.output_proj = nn.Linear(hidden_dim * num_heads, output_dim)

    def forward(self, data: Data) -> torch.Tensor:
        """
        Args:
            data: PyG Data object with:
                - x: (10, node_features) node features
                - edge_index: (2, num_edges) edge connectivity
                - edge_attr: (num_edges, edge_features) edge weights

        Returns:
            (output_dim,) graph-level representation
        """
        pass


class LineupGraphBuilder:
    """
    Build PyG Data objects from lineup information.
    """

    def __init__(self, player_features_df: pd.DataFrame):
        """
        Args:
            player_features_df: DataFrame with player static features
                (rapm, height, weight, positions, shooting stats)
        """
        pass

    def build_graph(self,
                    home_lineup: list[int],
                    away_lineup: list[int],
                    chemistry_matrix: np.ndarray = None) -> Data:
        """
        Build graph for a specific lineup matchup.

        Args:
            home_lineup: 5 home player IDs
            away_lineup: 5 away player IDs
            chemistry_matrix: (5, 5) matrix of minutes played together

        Returns:
            PyG Data object
        """
        pass

    def _build_edges(self, home_lineup, away_lineup):
        """
        Build edge_index:
            - Fully connected within each team (20 teammate edges)
            - Positional matchups between teams (5 opponent edges)
        """
        pass
```

#### 4.3 Two-Tower Fusion Architecture

```python
# nba_model/models/fusion.py - Design specification

import torch
import torch.nn as nn

class TwoTowerFusion(nn.Module):
    """
    Two-Tower architecture combining context and dynamic features.

    Tower A (Context): Static features via MLP
        - Team season stats (normalized)
        - Rest days
        - Travel distance
        - RAPM sums for lineups
        - Fatigue flags

    Tower B (Dynamic): Sequence + Graph outputs
        - Transformer output
        - GNN output

    Fusion: Concatenate tower outputs, pass through final MLP

    Multi-task outputs:
        1. Home win probability (binary)
        2. Home margin (regression)
        3. Total points (regression)
    """

    def __init__(self,
                 context_dim: int = 32,
                 transformer_dim: int = 128,
                 gnn_dim: int = 128,
                 hidden_dim: int = 256,
                 dropout: float = 0.2):
        super().__init__()

        # Tower A: Context
        self.context_tower = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Tower B: Dynamic
        dynamic_input_dim = transformer_dim + gnn_dim
        self.dynamic_tower = nn.Sequential(
            nn.Linear(dynamic_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Fusion layers
        fusion_dim = hidden_dim * 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Multi-task heads
        self.win_head = nn.Linear(hidden_dim // 2, 1)  # Sigmoid for prob
        self.margin_head = nn.Linear(hidden_dim // 2, 1)  # Regression
        self.total_head = nn.Linear(hidden_dim // 2, 1)  # Regression

    def forward(self, context, transformer_out, gnn_out):
        """
        Args:
            context: (batch, context_dim) static features
            transformer_out: (batch, transformer_dim) from GameFlowTransformer
            gnn_out: (batch, gnn_dim) from PlayerInteractionGNN

        Returns:
            {
                'win_prob': (batch, 1) sigmoid probability
                'margin': (batch, 1) predicted margin
                'total': (batch, 1) predicted total
            }
        """
        pass


class ContextFeatureBuilder:
    """
    Build context feature vector for Tower A.
    """

    FEATURES = [
        # Team stats (home)
        'home_off_rating_z', 'home_def_rating_z', 'home_pace_z',
        'home_efg_z', 'home_tov_z', 'home_orb_z', 'home_ft_rate_z',
        # Team stats (away) - same 7 features
        # Rest/fatigue
        'home_rest_days', 'away_rest_days', 'rest_diff',
        'home_back_to_back', 'away_back_to_back',
        'home_travel_miles', 'away_travel_miles',
        # RAPM sums
        'home_rapm_sum', 'away_rapm_sum', 'rapm_diff',
        'home_orapm_sum', 'home_drapm_sum',
        'away_orapm_sum', 'away_drapm_sum',
        # Spacing
        'home_spacing_area', 'away_spacing_area',
    ]

    def build(self, game_id: str, db_session) -> np.ndarray:
        """Build context feature vector for a game."""
        pass
```

#### 4.4 Training Pipeline

```python
# nba_model/models/trainer.py - Design specification

class FusionTrainer:
    """
    Training loop for the full fusion model.
    """

    def __init__(self,
                 transformer: GameFlowTransformer,
                 gnn: PlayerInteractionGNN,
                 fusion: TwoTowerFusion,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5):
        pass

    def train_epoch(self, dataloader, optimizer) -> dict:
        """
        Train for one epoch.

        Loss = CrossEntropy(win_prob, win_label)
             + HuberLoss(margin, true_margin)
             + HuberLoss(total, true_total)

        Returns:
            {'loss': float, 'win_loss': float, 'margin_loss': float, ...}
        """
        pass

    def evaluate(self, dataloader) -> dict:
        """
        Evaluate on validation set.

        Returns:
            {
                'accuracy': float,      # Win prediction accuracy
                'margin_mae': float,    # Mean absolute error on margin
                'total_mae': float,     # Mean absolute error on total
                'brier_score': float,   # Calibration metric
                'log_loss': float       # Cross-entropy
            }
        """
        pass

    def fit(self,
            train_loader,
            val_loader,
            epochs: int = 50,
            patience: int = 10,
            checkpoint_dir: str = 'data/models') -> dict:
        """
        Full training loop with early stopping.
        """
        pass


class NBADataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for NBA games.

    Each sample contains:
        - Tokenized play-by-play sequence (for Transformer)
        - Lineup graph (for GNN)
        - Context features (for Tower A)
        - Labels (win, margin, total)
    """
    pass
```

#### 4.5 Model Registry

```python
# nba_model/models/registry.py - Design specification

class ModelRegistry:
    """
    Track and version trained models.

    Storage structure:
        data/models/
            v1.0.0/
                transformer.pt
                gnn.pt
                fusion.pt
                metadata.json
            v1.0.1/
                ...

    Metadata includes:
        - Training date
        - Training data range
        - Hyperparameters
        - Validation metrics
        - Git commit hash
    """

    def save_model(self,
                   version: str,
                   models: dict,
                   metrics: dict,
                   config: dict) -> None:
        """Save model weights and metadata."""
        pass

    def load_model(self, version: str = 'latest') -> dict:
        """Load model weights and metadata."""
        pass

    def list_versions(self) -> list[dict]:
        """List all model versions with metadata."""
        pass

    def compare_versions(self, v1: str, v2: str) -> dict:
        """Compare metrics between two versions."""
        pass
```

### Technical Specifications

| Component | Parameters | Training Time (est.) | Memory |
|-----------|-----------|---------------------|--------|
| Transformer | ~200K | 10 min/epoch | ~500MB |
| GNN | ~100K | 5 min/epoch | ~300MB |
| Fusion | ~150K | 5 min/epoch | ~200MB |
| **Total** | ~450K | ~1 hour (50 epochs) | ~1GB |

### Dependencies on Other Phases
- Phase 2: All data tables populated
- Phase 3: RAPM and spacing features calculated
- Phase 3: Normalization stats computed

### Estimated Complexity
- **Transformer:** Medium-High
- **GNN:** High (PyG integration)
- **Fusion:** Medium
- **Training Pipeline:** Medium
- **Model Registry:** Low

---

## Phase 5: Backtesting

### Objectives
- Implement walk-forward validation to prevent look-ahead bias
- Build devigging methods (Power Method, Shin's Method)
- Implement Kelly Criterion bet sizing
- Create comprehensive performance metrics and reports

### Key Tasks

#### 5.1 Walk-Forward Validation Engine

```python
# nba_model/backtest/engine.py - Design specification

class WalkForwardEngine:
    """
    Walk-forward validation for time-series betting strategy.

    Unlike k-fold CV, walk-forward respects temporal ordering:

    Fold 1: Train [Season 1-2] | Validate [Season 3 first half]
    Fold 2: Train [Season 1-3 first half] | Validate [Season 3 second half]
    Fold 3: Train [Season 1-3] | Validate [Season 4 first half]
    ...

    This prevents any future information leakage.
    """

    def __init__(self,
                 min_train_games: int = 500,
                 validation_window_games: int = 100,
                 step_size_games: int = 50):
        pass

    def generate_folds(self,
                       games_df: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate (train, validation) splits.

        Returns:
            List of (train_games, val_games) DataFrame tuples
        """
        pass

    def run_backtest(self,
                     trainer: FusionTrainer,
                     kelly_calculator: KellyCalculator,
                     devig_method: str = 'power',
                     initial_bankroll: float = 10000) -> BacktestResult:
        """
        Run full walk-forward backtest.

        For each fold:
            1. Train model on training set
            2. Generate predictions on validation set
            3. Apply Kelly sizing
            4. Simulate bets
            5. Track P&L

        Returns:
            BacktestResult with all metrics and bet history
        """
        pass


class BacktestResult:
    """Container for backtest results."""

    def __init__(self):
        self.bets: list[Bet] = []
        self.bankroll_history: list[float] = []
        self.metrics: dict = {}

    @property
    def total_return(self) -> float:
        pass

    @property
    def sharpe_ratio(self) -> float:
        pass

    @property
    def max_drawdown(self) -> float:
        pass

    @property
    def win_rate(self) -> float:
        pass

    @property
    def clv(self) -> float:
        """Average closing line value."""
        pass
```

#### 5.2 Devigging Methods

```python
# nba_model/backtest/devig.py - Design specification

class DevigCalculator:
    """
    Remove bookmaker vig to find true implied probabilities.
    """

    def multiplicative_devig(self,
                             odds_home: float,
                             odds_away: float) -> tuple[float, float]:
        """
        Simple multiplicative method (inferior but fast).

        p_fair = implied_p / sum(implied_p)
        """
        pass

    def power_method_devig(self,
                           odds_home: float,
                           odds_away: float) -> tuple[float, float]:
        """
        Power method - better for handling longshot bias.

        Solve for k such that: (1/O_home)^k + (1/O_away)^k = 1
        Fair prob = (1/O)^k
        """
        pass

    def shin_method_devig(self,
                          odds_home: float,
                          odds_away: float) -> tuple[float, float]:
        """
        Shin's method - models market with informed bettors.

        Iteratively solve for z (proportion of informed bettors)
        to derive true probabilities.

        This is the "gold standard" for liquid markets.
        """
        pass

    def calculate_edge(self,
                       model_prob: float,
                       market_prob: float) -> float:
        """
        Calculate betting edge.

        Edge = model_prob - market_prob
        """
        return model_prob - market_prob


def solve_power_k(odds: list[float], tol: float = 1e-6) -> float:
    """
    Binary search to find k for power method.

    Find k such that sum((1/o)^k for o in odds) = 1
    """
    pass
```

#### 5.3 Kelly Criterion Sizing

```python
# nba_model/backtest/kelly.py - Design specification

class KellyCalculator:
    """
    Kelly Criterion bet sizing with fractional Kelly and caps.

    Full Kelly: f* = (bp - q) / b
    Where:
        b = decimal odds - 1 (net odds)
        p = model probability of winning
        q = 1 - p

    We use Fractional Kelly (typically 0.25) to reduce variance.
    """

    def __init__(self,
                 fraction: float = 0.25,      # Quarter Kelly
                 max_bet_pct: float = 0.02,   # 2% max bet
                 min_edge_pct: float = 0.02): # 2% min edge to bet
        self.fraction = fraction
        self.max_bet_pct = max_bet_pct
        self.min_edge_pct = min_edge_pct

    def calculate_bet_size(self,
                           bankroll: float,
                           model_prob: float,
                           decimal_odds: float) -> float:
        """
        Calculate optimal bet size.

        Returns:
            Bet amount in dollars (0 if no edge or negative Kelly)
        """
        pass

    def calculate_full_kelly(self,
                             model_prob: float,
                             decimal_odds: float) -> float:
        """
        Calculate full Kelly fraction.

        Returns:
            Kelly fraction (can be negative if no edge)
        """
        b = decimal_odds - 1
        p = model_prob
        q = 1 - p
        return (b * p - q) / b

    def optimize_fraction(self,
                          historical_bets: list[Bet],
                          fractions: list[float] = None) -> float:
        """
        Find optimal Kelly fraction via historical simulation.

        Test different fractions and find the one that maximizes
        Sharpe ratio or geometric growth rate.
        """
        pass


@dataclass
class Bet:
    """Single bet record."""
    game_id: str
    timestamp: datetime
    bet_type: str  # 'moneyline', 'spread', 'total'
    side: str  # 'home', 'away', 'over', 'under'
    model_prob: float
    market_odds: float
    market_prob: float
    edge: float
    kelly_fraction: float
    bet_amount: float
    result: str  # 'win', 'loss', 'push'
    profit: float
```

#### 5.4 Performance Metrics

```python
# nba_model/backtest/metrics.py - Design specification

class BacktestMetrics:
    """
    Calculate comprehensive backtest performance metrics.
    """

    def calculate_all(self, result: BacktestResult) -> dict:
        """
        Calculate all metrics.

        Returns:
            {
                # Returns
                'total_return': float,      # Total % return
                'cagr': float,              # Compound annual growth rate
                'avg_bet_return': float,    # Average return per bet

                # Risk
                'volatility': float,        # Std dev of returns
                'sharpe_ratio': float,      # Risk-adjusted return
                'sortino_ratio': float,     # Downside risk-adjusted
                'max_drawdown': float,      # Largest peak-to-trough
                'max_drawdown_duration': int,  # Days in drawdown

                # Betting
                'total_bets': int,
                'win_rate': float,
                'avg_edge': float,
                'avg_odds': float,
                'roi': float,               # Return on investment

                # Calibration
                'brier_score': float,       # Probability calibration
                'log_loss': float,

                # CLV (Closing Line Value)
                'avg_clv': float,           # Key profitability indicator
                'clv_positive_rate': float, # % of bets with positive CLV

                # By bet type
                'metrics_by_type': dict,    # Same metrics split by ML/spread/total
            }
        """
        pass

    def calculate_clv(self, bet: Bet, closing_odds: float) -> float:
        """
        Calculate closing line value for a single bet.

        CLV = (closing_implied_prob - bet_implied_prob) / bet_implied_prob

        Positive CLV means you beat the closing line.
        """
        pass

    def generate_report(self, result: BacktestResult) -> str:
        """Generate human-readable backtest report."""
        pass
```

### Technical Specifications

| Metric | Target | Acceptable |
|--------|--------|------------|
| ROI | > 5% | > 2% |
| Win Rate | > 53% | > 51% |
| CLV | > 1% | > 0% |
| Max Drawdown | < 15% | < 25% |
| Sharpe Ratio | > 1.0 | > 0.5 |

### Dependencies on Other Phases
- Phase 2: Historical odds data (if available)
- Phase 4: Trained models

### Estimated Complexity
- **Walk-Forward Engine:** Medium-High
- **Devigging:** Medium (Shin's method is tricky)
- **Kelly Criterion:** Low
- **Metrics:** Low-Medium

---

## Phase 6: Self-Improvement Iterations

### Objectives
- Implement covariate drift detection
- Create automatic retraining triggers
- Build model versioning and A/B comparison
- Enable continuous learning pipeline

### Key Tasks

#### 6.1 Covariate Drift Detection

```python
# nba_model/monitor/drift.py - Design specification

from scipy.stats import ks_2samp
import numpy as np

class DriftDetector:
    """
    Detect when input feature distributions shift significantly.

    Methods:
        - Kolmogorov-Smirnov (KS) test
        - Maximum Mean Discrepancy (MMD)
        - Population Stability Index (PSI)
    """

    MONITORED_FEATURES = [
        'pace', 'offensive_rating', 'fg3a_rate',
        'rest_days', 'travel_distance', 'rapm_mean'
    ]

    def __init__(self,
                 reference_data: pd.DataFrame,
                 p_value_threshold: float = 0.05,
                 psi_threshold: float = 0.2):
        self.reference = reference_data
        self.p_threshold = p_value_threshold
        self.psi_threshold = psi_threshold

    def ks_test(self,
                feature: str,
                recent_data: pd.DataFrame) -> tuple[float, float]:
        """
        Kolmogorov-Smirnov test for distribution shift.

        Returns:
            (statistic, p_value)
        """
        pass

    def calculate_psi(self,
                      feature: str,
                      recent_data: pd.DataFrame,
                      n_bins: int = 10) -> float:
        """
        Population Stability Index.

        PSI = Σ (actual% - expected%) * ln(actual% / expected%)

        Interpretation:
            PSI < 0.1: No significant shift
            0.1 ≤ PSI < 0.2: Moderate shift
            PSI ≥ 0.2: Significant shift
        """
        pass

    def check_drift(self,
                    recent_data: pd.DataFrame,
                    window_days: int = 30) -> dict:
        """
        Check all monitored features for drift.

        Returns:
            {
                'has_drift': bool,
                'features_drifted': list[str],
                'details': dict[str, {'ks_stat', 'p_value', 'psi'}]
            }
        """
        pass


class ConceptDriftDetector:
    """
    Detect when model predictions diverge from actual outcomes.

    This indicates the relationship between features and target
    has changed (not just the feature distribution).
    """

    def __init__(self,
                 accuracy_threshold: float = 0.48,
                 calibration_threshold: float = 0.1):
        pass

    def check_prediction_drift(self,
                               predictions: list[float],
                               actuals: list[int],
                               window_size: int = 100) -> dict:
        """
        Check if recent predictions are significantly worse.

        Returns:
            {
                'accuracy_degraded': bool,
                'calibration_degraded': bool,
                'recent_accuracy': float,
                'recent_brier_score': float
            }
        """
        pass
```

#### 6.2 Retraining Triggers

```python
# nba_model/monitor/triggers.py - Design specification

class RetrainingTrigger:
    """
    Determine when to trigger model retraining.

    Triggers:
        1. Scheduled: Weekly/monthly refresh
        2. Drift-based: Covariate or concept drift detected
        3. Performance-based: ROI drops below threshold
        4. Data-based: Significant new data available
    """

    def __init__(self,
                 scheduled_interval_days: int = 7,
                 min_new_games: int = 50,
                 roi_threshold: float = -0.05,
                 accuracy_threshold: float = 0.48):
        pass

    def check_scheduled_trigger(self,
                                last_train_date: date) -> bool:
        """Check if scheduled retraining is due."""
        pass

    def check_drift_trigger(self,
                            drift_detector: DriftDetector,
                            recent_data: pd.DataFrame) -> bool:
        """Check if drift warrants retraining."""
        pass

    def check_performance_trigger(self,
                                  recent_bets: list[Bet]) -> bool:
        """Check if recent performance is unacceptable."""
        pass

    def check_data_trigger(self,
                           games_since_training: int) -> bool:
        """Check if enough new data to justify retraining."""
        pass

    def evaluate_all_triggers(self, context: dict) -> dict:
        """
        Evaluate all triggers.

        Returns:
            {
                'should_retrain': bool,
                'reason': str,
                'priority': str,  # 'high', 'medium', 'low'
                'trigger_details': dict
            }
        """
        pass
```

#### 6.3 Model Versioning and Comparison

```python
# nba_model/monitor/versioning.py - Design specification

class ModelVersionManager:
    """
    Manage model versions and enable A/B comparison.
    """

    def create_version(self,
                       models: dict,
                       config: dict,
                       metrics: dict,
                       parent_version: str = None) -> str:
        """
        Create new model version.

        Version format: v{major}.{minor}.{patch}
            - Major: Architecture change
            - Minor: Retraining with new data
            - Patch: Hyperparameter tuning

        Returns:
            Version string (e.g., 'v1.2.0')
        """
        pass

    def compare_versions(self,
                         version_a: str,
                         version_b: str,
                         test_data: pd.DataFrame) -> dict:
        """
        Compare two model versions on the same test data.

        Returns:
            {
                'version_a_metrics': dict,
                'version_b_metrics': dict,
                'winner': str,
                'improvement': dict  # Metric-by-metric comparison
            }
        """
        pass

    def promote_version(self, version: str) -> None:
        """Promote version to 'production' (latest)."""
        pass

    def rollback(self, to_version: str) -> None:
        """Rollback to previous version."""
        pass

    def get_lineage(self, version: str) -> list[str]:
        """Get version history/lineage."""
        pass
```

### Technical Specifications

| Component | Check Frequency | Thresholds |
|-----------|----------------|------------|
| Covariate Drift | Daily | KS p < 0.05, PSI > 0.2 |
| Concept Drift | Per 100 predictions | Accuracy < 48% |
| Scheduled Retrain | Weekly | N/A |
| Performance Trigger | Daily | ROI < -5% over 50 bets |

### Dependencies on Other Phases
- Phase 4: Model training infrastructure
- Phase 5: Backtest metrics for comparison

### Estimated Complexity
- **Drift Detection:** Medium
- **Retraining Triggers:** Low-Medium
- **Versioning:** Medium

---

## Phase 7: Predictions

### Objectives
- Build production inference pipeline
- Implement Bayesian injury probability adjustments
- Generate actionable betting signals
- Handle game-time decisions and late scratches

### Key Tasks

#### 7.1 Inference Pipeline

```python
# nba_model/predict/inference.py - Design specification

class InferencePipeline:
    """
    Production inference pipeline for generating predictions.
    """

    def __init__(self,
                 model_registry: ModelRegistry,
                 db_session,
                 model_version: str = 'latest'):
        self.models = model_registry.load_model(model_version)
        self.db = db_session

    def predict_game(self, game_id: str) -> GamePrediction:
        """
        Generate full prediction for a single game.

        Steps:
            1. Load game context (teams, venue, date)
            2. Build context features (Tower A)
            3. Get expected lineups
            4. Build lineup graphs (GNN input)
            5. Use recent game sequence for Transformer (or zeros for pre-game)
            6. Run fusion model
            7. Apply injury adjustments

        Returns:
            GamePrediction object
        """
        pass

    def predict_today(self) -> list[GamePrediction]:
        """Generate predictions for all games today."""
        pass

    def predict_date(self, date: date) -> list[GamePrediction]:
        """Generate predictions for all games on a specific date."""
        pass


@dataclass
class GamePrediction:
    """Container for game prediction."""
    game_id: str
    game_date: date
    home_team: str
    away_team: str

    # Raw model outputs
    home_win_prob: float
    predicted_margin: float
    predicted_total: float

    # Injury-adjusted
    home_win_prob_adjusted: float
    predicted_margin_adjusted: float
    predicted_total_adjusted: float

    # Uncertainty
    confidence: float  # Based on model uncertainty
    injury_uncertainty: float  # Based on GTD players

    # Feature importance (for explainability)
    top_factors: list[tuple[str, float]]

    # Metadata
    model_version: str
    prediction_timestamp: datetime
    lineup_home: list[str]
    lineup_away: list[str]
```

#### 7.2 Bayesian Injury Adjustments

```python
# nba_model/predict/injuries.py - Design specification

class InjuryAdjuster:
    """
    Bayesian adjustment for injury uncertainty.

    Prior probabilities (from research):
        - Probable: 93%
        - Questionable: 55%
        - Doubtful: 3%
        - Out: 0%

    Posterior is updated based on:
        - Player history (play-through-pain tendency)
        - Team context (tanking, back-to-back)
        - Injury type
    """

    PRIOR_PLAY_PROBS = {
        'probable': 0.93,
        'questionable': 0.55,
        'doubtful': 0.03,
        'out': 0.00,
        'available': 1.00
    }

    def __init__(self, db_session):
        self.db = db_session

    def get_play_probability(self,
                             player_id: int,
                             injury_status: str,
                             injury_type: str = None,
                             team_context: dict = None) -> float:
        """
        Calculate probability player will play.

        Uses Bayesian update:
            P(play|context) ∝ P(context|play) * P(play|status)

        Returns:
            Float probability in [0, 1]
        """
        pass

    def calculate_player_history_likelihood(self,
                                            player_id: int,
                                            injury_status: str) -> float:
        """
        Calculate likelihood based on player's historical patterns.

        Some players (e.g., AD) rest more; others play through pain.
        """
        pass

    def adjust_prediction(self,
                          base_prediction: GamePrediction,
                          injury_report: list[dict]) -> GamePrediction:
        """
        Adjust prediction based on injury report.

        For each questionable player, run two scenarios:
            1. Player plays
            2. Player doesn't play

        Expected value:
            P(win) = P(play) * P(win|plays) + P(not play) * P(win|not plays)

        Returns:
            Adjusted GamePrediction
        """
        pass

    def calculate_replacement_impact(self,
                                     player_id: int,
                                     replacement_id: int,
                                     team_id: int) -> float:
        """
        Calculate win probability impact of player replacement.

        Uses RAPM differential between starter and replacement.
        """
        pass


class InjuryReportFetcher:
    """
    Fetch current injury report from NBA API or scraping.
    """

    def get_current_injuries(self) -> pd.DataFrame:
        """
        Get current injury report.

        Returns DataFrame with:
            player_id, team_id, status, injury_description, report_date
        """
        pass
```

#### 7.3 Betting Signal Generation

```python
# nba_model/predict/signals.py - Design specification

class SignalGenerator:
    """
    Generate actionable betting signals from predictions.
    """

    def __init__(self,
                 devig_calculator: DevigCalculator,
                 kelly_calculator: KellyCalculator,
                 min_edge: float = 0.02):
        self.devig = devig_calculator
        self.kelly = kelly_calculator
        self.min_edge = min_edge

    def generate_signals(self,
                         predictions: list[GamePrediction],
                         current_odds: dict) -> list[BettingSignal]:
        """
        Generate betting signals for all predictions.

        Returns:
            List of BettingSignal objects with positive edge
        """
        pass

    def generate_game_signals(self,
                              prediction: GamePrediction,
                              odds: dict) -> list[BettingSignal]:
        """
        Generate all possible signals for a single game.

        Checks:
            - Moneyline (home/away)
            - Spread (home/away)
            - Total (over/under)

        Returns signals only where edge > min_edge
        """
        pass


@dataclass
class BettingSignal:
    """Actionable betting signal."""
    game_id: str
    game_date: date
    matchup: str  # "LAL @ BOS"

    bet_type: str  # 'moneyline', 'spread', 'total'
    side: str  # 'home', 'away', 'over', 'under'
    line: float  # Spread or total line (None for ML)

    model_prob: float
    market_prob: float
    edge: float

    recommended_odds: float  # Decimal
    kelly_fraction: float
    recommended_stake_pct: float

    confidence: str  # 'high', 'medium', 'low'

    # Context
    key_factors: list[str]
    injury_notes: list[str]
```

### Technical Specifications

| Operation | Latency Target | Notes |
|-----------|---------------|-------|
| Single game prediction | < 5s | Includes feature building |
| Full day predictions | < 2 min | ~15 games max |
| Injury adjustment | < 1s | Per scenario |
| Signal generation | < 1s | After prediction |

### Dependencies on Other Phases
- Phase 3: All feature calculators
- Phase 4: Trained models
- Phase 5: Devig and Kelly calculators

### Estimated Complexity
- **Inference Pipeline:** Medium
- **Injury Adjustments:** High (Bayesian logic)
- **Signal Generation:** Low-Medium

---

## Phase 8: Output Structure

### Objectives
- Generate prediction reports and dashboards
- Build static GitHub Pages site
- Create daily automated updates
- Provide model performance tracking

### Key Tasks

#### 8.1 Report Generation

```python
# nba_model/output/reports.py - Design specification

class ReportGenerator:
    """
    Generate various reports for predictions and performance.
    """

    def daily_predictions_report(self,
                                 predictions: list[GamePrediction],
                                 signals: list[BettingSignal]) -> dict:
        """
        Generate daily predictions report.

        Returns:
            {
                'date': str,
                'games': list[dict],  # Per-game predictions
                'signals': list[dict],  # Actionable bets
                'summary': dict  # Aggregate stats
            }
        """
        pass

    def performance_report(self,
                           period: str = 'week') -> dict:
        """
        Generate performance tracking report.

        Returns:
            {
                'period': str,
                'total_predictions': int,
                'accuracy': float,
                'roi': float,
                'clv': float,
                'calibration_curve': list,
                'by_bet_type': dict
            }
        """
        pass

    def model_health_report(self,
                            drift_results: dict,
                            recent_metrics: dict) -> dict:
        """
        Generate model health/monitoring report.
        """
        pass
```

#### 8.2 GitHub Pages Dashboard

```python
# nba_model/output/dashboard.py - Design specification

class DashboardBuilder:
    """
    Build static GitHub Pages dashboard.

    Structure:
        docs/
        ├── index.html           # Main dashboard
        ├── predictions.html     # Today's predictions
        ├── history.html         # Historical performance
        ├── model.html           # Model info and health
        ├── api/                  # JSON data files
        │   ├── today.json
        │   ├── signals.json
        │   ├── performance.json
        │   └── history/
        │       └── 2024-01-15.json
        └── assets/
            ├── style.css
            └── charts.js
    """

    def __init__(self,
                 output_dir: str = 'docs',
                 template_dir: str = 'templates'):
        pass

    def build_full_site(self) -> None:
        """Build complete static site."""
        pass

    def update_predictions(self,
                           predictions: list[GamePrediction],
                           signals: list[BettingSignal]) -> None:
        """Update today's predictions page and JSON."""
        pass

    def update_performance(self,
                           metrics: dict) -> None:
        """Update performance tracking page."""
        pass

    def archive_day(self, date: date) -> None:
        """Archive predictions to history."""
        pass


class ChartGenerator:
    """
    Generate chart data for dashboard visualization.
    """

    def bankroll_chart(self, history: list[float]) -> dict:
        """Generate bankroll growth chart data."""
        pass

    def calibration_chart(self,
                          predictions: list[float],
                          actuals: list[int]) -> dict:
        """Generate probability calibration chart."""
        pass

    def roi_by_month_chart(self, bets: list[Bet]) -> dict:
        """Generate monthly ROI bar chart."""
        pass
```

#### 8.3 Dashboard Content Structure

**index.html (Main Dashboard)**
- Current model version and health status
- Quick stats: Win rate, ROI, CLV
- Today's top signals
- Bankroll chart (if simulated)

**predictions.html**
- All games for today
- Model predictions vs market lines
- Recommended bets with confidence levels
- Key factors for each prediction
- Injury report summary

**history.html**
- Historical performance by period
- Calibration curves
- ROI over time
- Bet type breakdown
- Searchable bet history

**model.html**
- Model architecture summary
- Current version info
- Feature importance
- Drift detection status
- Retraining schedule

#### 8.4 Automation

```python
# CLI commands for automation

# Daily workflow (cron job)
nba-model data update           # Fetch yesterday's results
nba-model features build        # Update features
nba-model monitor drift         # Check for drift
nba-model predict today         # Generate predictions
nba-model dashboard build       # Update dashboard

# Weekly workflow
nba-model train all             # Retrain if triggered
nba-model backtest run          # Full backtest validation
```

### Technical Specifications

| Page | Data Source | Update Frequency |
|------|-------------|------------------|
| index.html | performance.json | Daily |
| predictions.html | today.json, signals.json | Daily (pre-game) |
| history.html | history/*.json | Daily |
| model.html | model_health.json | Daily |

### Dependencies on Other Phases
- Phase 5: Backtest results for performance
- Phase 6: Drift detection for model health
- Phase 7: Predictions and signals

### Estimated Complexity
- **Report Generation:** Low
- **Dashboard Builder:** Medium
- **Chart Generation:** Low-Medium
- **Automation:** Low

---

## Implementation Timeline Summary

| Phase | Description | Dependencies | Complexity |
|-------|-------------|--------------|------------|
| 1 | App Structure | None | Low |
| 2 | Data Collection & Storage | Phase 1 | High |
| 3 | Feature Engineering | Phase 2 | High |
| 4 | Model Training | Phase 2, 3 | High |
| 5 | Backtesting | Phase 4 | Medium |
| 6 | Self-Improvement | Phase 4, 5 | Medium |
| 7 | Predictions | Phase 3, 4, 5 | Medium |
| 8 | Output Structure | Phase 5, 6, 7 | Low-Medium |

### Critical Path
```
Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 7 → Phase 8
                                    ↓
                               Phase 6 (parallel)
```

### Risk Factors

1. **NBA API Rate Limits**: May need to implement aggressive caching or spread collection over multiple days
2. **Historical Odds Data**: May not be freely available; consider Kaggle datasets or paid APIs
3. **PyTorch Geometric Complexity**: GNN implementation requires careful attention to batch handling
4. **Drift Detection Sensitivity**: Tuning thresholds to avoid false positives
5. **Injury Report Timeliness**: Public injury data may lag behind reality

### Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Backtest ROI | > 3% | Walk-forward validation |
| Win Rate | > 52% | All bet types combined |
| CLV | > 0.5% | Versus closing lines |
| Model Accuracy | > 60% | Win prediction |
| Calibration | Brier < 0.24 | Well-calibrated probabilities |

---

## Appendix: Key Technical Decisions

### A1. Why SQLite over PostgreSQL?
- Single-file deployment simplicity
- No server management
- Sufficient for expected data volume (~5GB total)
- Easy backup and version control
- Can migrate to PostgreSQL later if needed

### A2. Why Typer over Click?
- Modern Python with type hints
- Automatic help generation from type annotations
- Better async support
- Cleaner syntax

### A3. Why PyTorch over TensorFlow?
- Better PyTorch Geometric integration for GNNs
- More intuitive debugging
- Stronger research community adoption
- Dynamic computation graphs suit our varying lineup sizes

### A4. Why Static GitHub Pages over Flask/FastAPI?
- Zero hosting costs
- No server maintenance
- Sufficient for daily prediction updates
- Can add API layer later if real-time needs arise

### A5. Transformer vs LSTM for Sequence Model
- Transformer handles long-range dependencies better (full game ~200 events)
- Attention mechanism provides interpretability (which events matter)
- Better parallelization during training
- State-of-the-art performance on sequence tasks
