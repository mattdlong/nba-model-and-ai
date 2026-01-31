"""Feature engineering for NBA model.

This module provides feature calculation and transformation functionality
including RAPM, spatial analysis, fatigue metrics, event parsing, and
season normalization.

Submodules:
    rapm: Regularized Adjusted Plus-Minus calculation using Ridge Regression
    spatial: Convex hull floor spacing metrics
    fatigue: Rest, travel, and load indicators
    parsing: Play-by-play event text parsing via regex
    normalization: Z-score normalization by season

Example:
    >>> from nba_model.features import RAPMCalculator, SeasonNormalizer
    >>> calculator = RAPMCalculator(lambda_=5000)
    >>> coefficients = calculator.fit(stints_df)

    >>> normalizer = SeasonNormalizer()
    >>> normalizer.fit(game_stats_df)
    >>> normalized = normalizer.transform(df, season="2023-24")
"""

from __future__ import annotations

# Fatigue Calculator
from nba_model.features.fatigue import (
    ARENA_COORDS,
    DEFAULT_PLAYER_LOAD_LOOKBACK_GAMES,
    DEFAULT_TRAVEL_LOOKBACK_DAYS,
    TEAM_ID_TO_ABBREV,
    FatigueCalculator,
    PlayerLoadMetrics,
    calculate_haversine_distance,
)

# Season Normalizer
from nba_model.features.normalization import (
    ALL_NORMALIZABLE_METRICS,
    DEFAULT_METRICS_TO_NORMALIZE,
    MetricStats,
    SeasonNormalizer,
    normalize_by_season,
)

# Event Parser
from nba_model.features.parsing import (
    EventParser,
    EventPatterns,
    ShotClockCategory,
    ShotContext,
    ShotType,
    TurnoverContext,
    TurnoverType,
    parse_shot_context,
    parse_turnover_type,
)

# RAPM Calculator
from nba_model.features.rapm import (
    DEFAULT_LAMBDA,
    DEFAULT_MIN_MINUTES,
    DEFAULT_TIME_DECAY_TAU,
    RAPMCalculator,
    RAPMResult,
    calculate_rapm_for_season,
)

# Spacing Calculator
from nba_model.features.spatial import (
    MIN_SHOTS_FOR_HULL,
    MIN_SHOTS_FOR_LINEUP,
    LineupSpacingResult,
    SpacingCalculator,
    calculate_lineup_spacing,
)

__all__ = [
    "ALL_NORMALIZABLE_METRICS",
    "ARENA_COORDS",
    "DEFAULT_LAMBDA",
    "DEFAULT_METRICS_TO_NORMALIZE",
    "DEFAULT_MIN_MINUTES",
    "DEFAULT_PLAYER_LOAD_LOOKBACK_GAMES",
    "DEFAULT_TIME_DECAY_TAU",
    "DEFAULT_TRAVEL_LOOKBACK_DAYS",
    "MIN_SHOTS_FOR_HULL",
    "MIN_SHOTS_FOR_LINEUP",
    "TEAM_ID_TO_ABBREV",
    "EventParser",
    "EventPatterns",
    "FatigueCalculator",
    "LineupSpacingResult",
    "MetricStats",
    "PlayerLoadMetrics",
    "RAPMCalculator",
    "RAPMResult",
    "SeasonNormalizer",
    "ShotClockCategory",
    "ShotContext",
    "ShotType",
    "SpacingCalculator",
    "TurnoverContext",
    "TurnoverType",
    "calculate_haversine_distance",
    "calculate_lineup_spacing",
    "calculate_rapm_for_season",
    "normalize_by_season",
    "parse_shot_context",
    "parse_turnover_type",
]
