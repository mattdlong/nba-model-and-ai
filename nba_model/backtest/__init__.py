"""Backtesting engine for NBA betting strategy.

This module provides walk-forward validation, bet sizing, vig removal,
and performance metric calculation for strategy evaluation.

Submodules:
    engine: Walk-forward validation framework
    kelly: Kelly Criterion bet sizing
    devig: Vig removal methods (Power, Shin)
    metrics: Performance metrics (Sharpe, CLV, etc.)

Key concepts:
    - Walk-forward validation prevents look-ahead bias
    - Kelly Criterion optimizes bankroll growth
    - Devigging recovers true probabilities from odds
    - CLV (Closing Line Value) indicates long-term edge

Example:
    >>> from nba_model.backtest import WalkForwardEngine, KellyCalculator
    >>> engine = WalkForwardEngine(min_train_games=500)
    >>> result = engine.run_backtest(trainer, kelly_calc)
"""
from __future__ import annotations

# Public API - will be populated in Phase 5
__all__: list[str] = []
