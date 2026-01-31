"""Feature engineering for NBA model.

This module provides feature calculation and transformation functionality
including RAPM, spatial analysis, fatigue metrics, and normalization.

Submodules:
    rapm: Regularized Adjusted Plus-Minus calculation
    spatial: Convex hull floor spacing metrics
    fatigue: Rest, travel, and load indicators
    parsing: Play-by-play event text parsing
    normalization: Z-score normalization by season

Example:
    >>> from nba_model.features import RAPMCalculator
    >>> calculator = RAPMCalculator(lambda_=5000)
    >>> coefficients = calculator.fit(stints_df)
"""
from __future__ import annotations

# Public API - will be populated in Phase 3
__all__: list[str] = []
