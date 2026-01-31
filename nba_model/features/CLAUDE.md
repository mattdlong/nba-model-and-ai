# Feature Engineering

## Responsibility

Transforms raw NBA data into ML-ready features. Handles RAPM calculation, spatial analytics, fatigue metrics, event parsing, and normalization.

## Status

✅ **Phase 3 - Complete**

## Structure

| File | Purpose | Key Outputs |
|------|---------|-------------|
| `__init__.py` | Public API exports | All feature classes and functions |
| `rapm.py` | RAPM calculation via Ridge Regression | ORAPM, DRAPM, total per player |
| `spatial.py` | Convex hull spacing analysis | Hull area, centroid, corner density |
| `fatigue.py` | Rest/travel/load metrics | B2B flags, travel distance, player load |
| `parsing.py` | Play-by-play text extraction | Turnover types, shot context |
| `normalization.py` | Z-score by season | Normalized feature vectors |

## Key Classes

| Class | Purpose |
|-------|---------|
| `RAPMCalculator` | Ridge regression on stint design matrix for player impact |
| `SpacingCalculator` | Convex hull and KDE analysis of shot distributions |
| `FatigueCalculator` | Haversine travel and schedule flag calculation |
| `EventParser` | Regex-based play description parsing |
| `SeasonNormalizer` | Z-score normalization with persistence |

## Key Algorithms

1. **RAPM:** Ridge regression on sparse design matrix (λ=5000)
   - Design matrix: +1 home players, -1 away players
   - Time decay: exp(-(today - game_date) / 180)
   - Separate ORAPM/DRAPM regressions

2. **Convex Hull:** scipy.spatial for shot distribution area
   - Handles degenerate cases (collinear points)
   - Corner density from corner zone detection

3. **Haversine:** Great-circle distance between arenas
   - Uses `haversine` library for accuracy
   - All 30 NBA arena coordinates included

4. **Z-Score:** (x - μ_season) / σ_season
   - Per-season statistics for cross-season comparability
   - Handles pace and space evolution

## Feature Categories

| Category | Count | Examples |
|----------|-------|----------|
| RAPM | 3 | ORAPM, DRAPM, total_rapm |
| Spatial | 5 | Hull area, centroid_x/y, corner_density, avg_distance |
| Fatigue | 7 | rest_days, back_to_back, travel_miles, home_stand |
| Contextual | 4+ | Shot type, is_transition, is_contested |

## CLI Commands

```bash
# Build all features
nba-model features build --seasons 2023-24

# RAPM only with options
nba-model features rapm --season 2023-24 --lambda 5000 --cv

# Spacing only
nba-model features spatial --season 2023-24 --min-shots 20
```

## Integration Points

- **Upstream:** `data/` provides ORM models (Stint, Shot, Game, etc.)
- **Downstream:** `models/` consumes feature tensors for training

## Usage Examples

```python
# RAPM Calculation
from nba_model.features import RAPMCalculator

calculator = RAPMCalculator(lambda_=5000, min_minutes=100)
coefficients = calculator.fit(stints_df)
# coefficients[player_id] = {'orapm': 2.1, 'drapm': 0.8, 'total_rapm': 2.9, ...}

# Spacing Analysis
from nba_model.features import SpacingCalculator

calculator = SpacingCalculator(min_shots=20)
metrics = calculator.calculate_lineup_spacing([1, 2, 3, 4, 5], shots_df)
# metrics = {'hull_area': 550.0, 'centroid_x': 0.0, ...}

# Fatigue Flags
from nba_model.features import FatigueCalculator

calculator = FatigueCalculator()
indicators = calculator.calculate_schedule_flags(team_id, game_date, games_df)
# indicators = {'back_to_back': True, 'travel_miles': 2500.0, ...}

# Season Normalization
from nba_model.features import SeasonNormalizer

normalizer = SeasonNormalizer()
normalizer.fit(game_stats_df)
normalized_df = normalizer.transform(df, season="2023-24")
```

## Anti-Patterns

- ❌ Never compute RAPM without time decay weighting
- ❌ Never use raw stats without z-score normalization
- ❌ Never hardcode feature dimensions (use config)
- ❌ Never skip minimum sample checks (min_minutes, min_shots)
- ❌ Never call NBA API for arena coordinates (use ARENA_COORDS constant)
