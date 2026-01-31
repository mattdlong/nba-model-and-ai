# Phase 3: Feature Engineering - Implementation Summary

**Status:** ✅ Complete
**Date:** 2026-01-31
**Phase Duration:** Single session

## Overview

Phase 3 implements the feature engineering pipeline for the NBA prediction model. This includes player impact metrics (RAPM), spatial floor spacing analysis, fatigue/travel calculations, play-by-play event parsing, and season normalization.

## Implemented Components

### 3.1 RAPM Calculator (`nba_model/features/rapm.py`)

**Purpose:** Calculate Regularized Adjusted Plus-Minus coefficients using Ridge Regression.

**Key Features:**
- Sparse design matrix construction (+1 home, -1 away, 0 otherwise)
- Separate ORAPM/DRAPM regressions for offensive and defensive components
- Exponential time decay weighting (τ=180 days default)
- Minimum minutes filtering for player inclusion
- Lambda cross-validation for hyperparameter tuning
- Save/load functionality for fitted calculators

**Mathematical Model:**
```
Y = Xβ + ε
Minimize: ||Y - Xβ||² + λ||β||²
```

**Output:** Dict mapping player_id to `RAPMCoefficients` TypedDict with orapm, drapm, total_rapm, sample_stints.

### 3.2 Spacing Calculator (`nba_model/features/spatial.py`)

**Purpose:** Analyze floor spacing via convex hull geometry on shot distributions.

**Key Features:**
- Convex hull area calculation using scipy.spatial
- Shot distribution centroid calculation
- Average distance from basket
- Corner three-point density
- Deterministic lineup hash for deduplication
- KDE-based shot density maps for gravity overlap

**Edge Cases Handled:**
- Collinear points (returns 0 area)
- Insufficient shots (minimum thresholds)
- Degenerate hulls via QhullError handling

**Output:** `SpacingMetrics` TypedDict with hull_area, centroid_x, centroid_y, avg_distance, corner_density, shot_count.

### 3.3 Fatigue Calculator (`nba_model/features/fatigue.py`)

**Purpose:** Calculate rest, travel, and schedule fatigue indicators.

**Key Features:**
- Rest days calculation between games
- Haversine distance for travel calculations
- Schedule flags: back_to_back, three_in_four, four_in_five
- Home stand / road trip counting
- Player load metrics: avg_minutes, distance_miles, minutes_trend
- Complete arena coordinates for all 30 NBA teams

**Static Data:**
- `ARENA_COORDS`: Dict mapping team abbreviation to (lat, lon)
- `TEAM_ID_TO_ABBREV`: Dict mapping NBA team_id to abbreviation

**Output:** `FatigueIndicators` TypedDict with rest_days, back_to_back, three_in_four, four_in_five, travel_miles, home_stand, road_trip.

### 3.4 Event Parser (`nba_model/features/parsing.py`)

**Purpose:** Extract structured features from play-by-play text descriptions.

**Key Features:**
- Turnover classification: unforced (bad pass, traveling) vs forced (steal, lost ball)
- Shot type extraction: driving, pullup, stepback, catch_shoot, floating, fadeaway, etc.
- Shot context: is_transition, is_contested flags
- Shot clock usage categorization: early (<8s), mid (8-16s), late (>16s)
- Case-insensitive regex patterns

**Enums:**
- `TurnoverType`: UNFORCED, FORCED, UNKNOWN
- `ShotType`: DRIVING, PULLUP, STEPBACK, CATCH_SHOOT, etc.
- `ShotClockCategory`: EARLY, MID, LATE, UNKNOWN

**Output:** Parsed columns added to play-by-play DataFrames.

### 3.5 Season Normalizer (`nba_model/features/normalization.py`)

**Purpose:** Z-score normalization by season for cross-season comparability.

**Key Features:**
- Per-season mean/std calculation for each metric
- Transform: z = (x - μ_season) / σ_season
- Inverse transform for denormalization
- Database persistence to `season_stats` table
- JSON save/load for offline usage
- Zero-std handling (replaced with 1.0)

**Default Metrics:**
- pace, offensive_rating, defensive_rating
- efg_pct, tov_pct, orb_pct, ft_rate

**Output:** DataFrame with `*_z` columns for normalized values.

### 3.6 CLI Integration

**Implemented Commands:**

```bash
# Build all features
nba-model features build [--seasons 2023-24] [--force]

# RAPM only
nba-model features rapm --season 2023-24 [--lambda 5000] [--min-minutes 100] [--cv]

# Spacing only
nba-model features spatial --season 2023-24 [--min-shots 20]
```

**Build Command Workflow:**
1. Season Normalization - Calculate and save normalization stats
2. RAPM Calculation - Ridge regression on stints
3. Lineup Spacing - Convex hulls on shot distributions

## Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `nba_model/features/rapm.py` | ~400 | RAPM calculator |
| `nba_model/features/spatial.py` | ~280 | Spacing calculator |
| `nba_model/features/fatigue.py` | ~380 | Fatigue calculator |
| `nba_model/features/parsing.py` | ~320 | Event parser |
| `nba_model/features/normalization.py` | ~300 | Season normalizer |
| `nba_model/features/__init__.py` | ~115 | Public API exports |
| `tests/unit/features/test_rapm.py` | ~220 | RAPM unit tests |
| `tests/unit/features/test_spatial.py` | ~200 | Spatial unit tests |
| `tests/unit/features/test_fatigue.py` | ~250 | Fatigue unit tests |
| `tests/unit/features/test_parsing.py` | ~230 | Parsing unit tests |
| `tests/unit/features/test_normalization.py` | ~210 | Normalization unit tests |

## Database Tables Populated

| Table | Description |
|-------|-------------|
| `player_rapm` | RAPM coefficients per player-season |
| `lineup_spacing` | Hull area, centroid per lineup |
| `season_stats` | Normalization parameters per season-metric |

## Dependencies Used

| Library | Version | Purpose |
|---------|---------|---------|
| `scipy` | >=1.11 | Sparse matrices, ConvexHull, KDE, Ridge |
| `scikit-learn` | >=1.3 | Ridge regression, cross-validation |
| `haversine` | >=2.8 | Geodesic distance calculations |
| `numpy` | <2 | Numerical operations |
| `pandas` | >=2.0 | DataFrame operations |

## Testing

All tests pass:
- **test_rapm.py**: 12 tests covering matrix construction, fitting, time decay, filtering
- **test_spatial.py**: 18 tests covering hull area, centroid, edge cases, KDE
- **test_fatigue.py**: 17 tests covering haversine, schedule flags, travel
- **test_parsing.py**: 25 tests covering turnovers, shot types, context
- **test_normalization.py**: 20 tests covering fit, transform, persistence

## Success Criteria Met

| Criterion | Status |
|-----------|--------|
| `player_rapm` table contains valid RAPM coefficients | ✅ |
| RAPM values centered around zero with reasonable spread | ✅ |
| `lineup_spacing` populated for unique 5-player lineups | ✅ |
| Hull area values physically reasonable | ✅ |
| `season_stats` contains normalization parameters | ✅ |
| Event parser correctly classifies turnovers/shots | ✅ |
| Fatigue metrics correctly identify back-to-back games | ✅ |
| Travel distances match expected ranges | ✅ |

## Integration Points

**Upstream (Phase 2):**
- `Stint` model for RAPM design matrix
- `Shot` model for spatial analysis
- `Game` model for schedule/fatigue
- `Play` model for event parsing
- `GameStats` model for normalization

**Downstream (Phase 4+):**
- Feature tensors for Transformer/GNN models
- Normalized statistics for model training
- RAPM coefficients for player embeddings

## Next Steps (Phase 4)

1. Define model architectures (Transformer, GNN, Fusion)
2. Create PyTorch Dataset classes using Phase 3 features
3. Implement training loops with multi-task heads
4. Build feature tensor pipeline from RAPM + spacing + fatigue
