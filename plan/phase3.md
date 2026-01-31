# Phase 3: Feature Engineering

## Objectives

- Implement Regularized Adjusted Plus-Minus (RAPM) calculation using Ridge Regression
- Build spatial analysis via Convex Hulls for lineup floor spacing metrics
- Calculate fatigue and travel metrics using haversine distance
- Create season-normalized features via z-score transformation
- Parse play-by-play text descriptions for derived event classification

## Dependencies

- Phase 2 completed: `stints` table populated (required for RAPM)
- Phase 2 completed: `shots` table populated (required for Convex Hulls)
- Phase 2 completed: `games`, `plays`, `player_game_stats` tables populated

## Tasks

### 3.1 RAPM Calculator (`nba_model/features/rapm.py`)

Implement `RAPMCalculator` class with Ridge Regression for player impact estimation.

**Mathematical Model:**
- Solve `Y = Xβ + ε` where Y = point differential per 100 possessions per stint
- X = sparse design matrix (+1 for home players on court, -1 for away players, 0 otherwise)
- β = RAPM coefficients (offensive, defensive, total)
- Ridge minimizes: `||Y - Xβ||² + λ||β||²`

**Required Methods:**
- `build_stint_matrix()`: Construct sparse design matrix from stints table, target vector Y, and time-decay weights W
- `fit()`: Execute Ridge Regression, return dict mapping player_id to (orapm, drapm, total_rapm) tuple
- `cross_validate_lambda()`: Grid search over lambda values [1000, 2000, 5000, 10000] to find optimal regularization

**Implementation Constraints:**
- Use `scipy.sparse` for design matrix (expect ~10 non-zero entries per row of ~500 columns)
- Apply exponential time-decay weighting: `w_i = exp(-(today - game_date) / tau)` with tau=180 days default
- Filter players with < 100 minutes played
- Split ORAPM/DRAPM via separate regressions on points scored vs points allowed

**Output:** Populate `player_rapm` table with columns: player_id, season_id, calculation_date, orapm, drapm, rapm, sample_stints

### 3.2 Convex Hull Spacing Calculator (`nba_model/features/spatial.py`)

Implement `SpacingCalculator` class for lineup floor geometry analysis.

**Required Methods:**
- `calculate_lineup_spacing()`: For 5-player lineup, compute convex hull of aggregated shot locations
  - Return hull_area (sq units), centroid_x, centroid_y, avg_distance_from_basket, corner_density, shot_count
- `calculate_player_shot_density()`: Generate 2D KDE using `scipy.stats.gaussian_kde` over player shot locations
- `calculate_lineup_gravity_overlap()`: Quantify overlapping shooting zones between teammates (indicates spacing deficiencies)

**Court Coordinate System:**
- LOC_X range: approximately -250 to 250 (tenths of feet from basket center)
- LOC_Y range: approximately -50 to 900
- Use `scipy.spatial.ConvexHull` for hull computation

**Output:** Populate `lineup_spacing` table with lineup_hash (sorted player IDs hash), hull_area, centroid coordinates, shot_count

### 3.3 Fatigue Calculator (`nba_model/features/fatigue.py`)

Implement `FatigueCalculator` class for rest and travel impact metrics.

**Static Data Required:**
- `ARENA_COORDS`: Dict mapping team abbreviation to (latitude, longitude) for all 30 NBA arenas

**Required Methods:**
- `calculate_rest_days()`: Days since team's last game
- `calculate_travel_distance()`: Total miles traveled in lookback window using `haversine` library
- `calculate_schedule_flags()`: Return dict with boolean/int flags:
  - `back_to_back`: True if second game in consecutive nights
  - `3_in_4`: Third game in four nights
  - `4_in_5`: Fourth game in five nights
  - `home_stand`: Count of consecutive home games
  - `road_trip`: Count of consecutive away games
- `calculate_player_load()`: Per-player fatigue metrics:
  - avg_minutes over lookback window
  - total_distance_miles (from player tracking data)
  - minutes_trend (positive = increasing workload)

### 3.4 Event Parser (`nba_model/features/parsing.py`)

Implement `EventParser` class for extracting structured data from play-by-play text descriptions.

**Regex Pattern Dictionary:**
Define patterns for: bad_pass, lost_ball, steal, block, driving, pullup, step_back, catch_shoot, transition, contested

**Required Methods:**
- `parse_turnover_type()`: Classify as 'unforced' (bad pass patterns) or 'forced' (lost ball/steal patterns)
- `parse_shot_context()`: Extract shot_type enum (driving/pullup/catch_shoot/stepback/other), is_transition bool, is_contested bool
- `calculate_shot_clock_usage()`: Compute elapsed time between possession start and shot event, categorize as 'early' (<8s), 'mid' (8-16s), 'late' (>16s)

**Pattern Application:** Apply to home_description, away_description, neutral_description columns from plays table

### 3.5 Season Normalizer (`nba_model/features/normalization.py`)

Implement `SeasonNormalizer` class for cross-season statistical comparability.

**Target Metrics for Normalization:**
```
pace, offensive_rating, defensive_rating, efg_pct, tov_pct, orb_pct, ft_rate, fg3a_rate, points_per_game
```

**Required Methods:**
- `fit()`: Calculate mean and std for each metric per season from games/game_stats tables
- `transform()`: Apply z-score: `z = (x - μ_season) / σ_season`
- `save_stats()`: Persist normalization parameters to `season_stats` table
- `load_stats()`: Retrieve normalization parameters from database for a given season

**Output:** Populate `season_stats` table with: season_id, metric_name, mean_value, std_value, min_value, max_value

### 3.6 CLI Integration

Add feature subcommands to existing CLI structure:

- `nba-model features build`: Execute all feature calculators in dependency order (normalization → RAPM → spacing → fatigue)
- `nba-model features rapm`: Calculate RAPM coefficients for specified season(s)
- `nba-model features spatial`: Calculate convex hull metrics for specified season(s)

Include progress reporting via `tqdm` and structured logging via `loguru`.

## Success Criteria

- `player_rapm` table contains valid RAPM coefficients for all players meeting minimum minutes threshold
- RAPM values centered approximately around zero with reasonable spread (std ~1-3)
- `lineup_spacing` table populated for all unique 5-player lineups with sufficient shot sample (n >= 20)
- Hull area values physically reasonable (not exceeding half-court dimensions)
- `season_stats` table contains normalization parameters for all specified seasons
- Event parser correctly classifies >90% of turnovers and shot contexts based on manual spot-check
- Fatigue metrics correctly identify back-to-back games with 100% accuracy
- Travel distances match expected ranges (0 for home games, reasonable continental US distances for away)

## Testing Requirements

### Unit Tests (`tests/test_features/`)

**test_rapm.py:**
- Test sparse matrix construction with known lineup produces correct +1/-1/0 pattern
- Test ridge regression on synthetic data with known solution converges to expected coefficients
- Test time-decay weighting applies correct exponential decay
- Test minimum minutes filter excludes low-sample players

**test_spatial.py:**
- Test convex hull on square shot pattern returns expected area
- Test centroid calculation matches known geometric center
- Test edge case: collinear shots (degenerate hull) handled gracefully
- Test lineup hash determinism (same players different order = same hash)

**test_fatigue.py:**
- Test haversine calculation between known city pairs matches expected distances
- Test back_to_back detection with consecutive game dates
- Test 3_in_4 and 4_in_5 flag logic with various game schedules
- Test rest_days returns 0 for back-to-back, correct count otherwise

**test_parsing.py:**
- Test each regex pattern against known positive and negative examples from actual play descriptions
- Test turnover classification on sample "Bad Pass" vs "Lost Ball" descriptions
- Test shot context extraction identifies "Driving Layup" as driving=True
- Test shot clock categorization boundaries (7.9s = early, 8.0s = mid, 16.0s = mid, 16.1s = late)

**test_normalization.py:**
- Test fit() computes correct mean/std on known data
- Test transform() produces z-scores with mean ~0 and std ~1
- Test save/load round-trip preserves values to database
- Test transform with missing season raises appropriate error

### Integration Tests

**test_feature_pipeline.py:**
- Test `features build` CLI command executes without error on test database with minimal data
- Test RAPM calculation completes on real stints data subset
- Test spacing calculation handles lineups with varying shot counts
- Verify all output tables have expected schema and non-null values in required columns

### Validation Checks

- RAPM: Top/bottom 10 players by RAPM should include known high/low impact players (sanity check)
- Spacing: Teams known for three-point shooting should have larger hull areas
- Fatigue: Teams with compressed schedules should show elevated travel/b2b metrics
- Cross-reference calculated rest days against public schedule data

## Computational Specifications

| Feature | Estimated Time | Storage Impact | Update Frequency |
|---------|---------------|----------------|------------------|
| RAPM | ~30s per season | ~50KB | Weekly |
| Convex Hulls | ~5min per season | ~1MB | Weekly |
| Fatigue | ~1s per game | ~10KB | Daily (with data collection) |
| Event Parsing | ~10s per game | ~100KB | With data collection |
| Normalization | ~5s per season | ~5KB | Weekly |

---

Upon completion, commit all changes with message "feat: implement phase 3 - feature engineering" and push to origin.
