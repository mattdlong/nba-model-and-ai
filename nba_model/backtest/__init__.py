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
    >>> kelly = KellyCalculator(fraction=0.25)
    >>> result = engine.run_backtest(games_df, trainer)
    >>> print(f"ROI: {result.metrics.roi:.2%}")
"""

from __future__ import annotations

# Devigging
from nba_model.backtest.devig import (
    ConvergenceError,
    DevigCalculator,
    DevigError,
    FairProbabilities,
    InvalidOddsError,
    calculate_overround,
    implied_probability,
    solve_power_k,
    solve_shin_z,
)

# Walk-forward engine
from nba_model.backtest.engine import (
    BacktestConfig,
    BacktestResult,
    FoldInfo,
    FoldResult,
    OddsProviderProtocol,
    TrainerProtocol,
    WalkForwardEngine,
    create_mock_trainer,
)

# Kelly criterion
from nba_model.backtest.kelly import (
    InvalidInputError,
    KellyCalculator,
    KellyError,
    KellyOptimizationResult,
    KellyResult,
    SimulationResult,
)

# Performance metrics
from nba_model.backtest.metrics import (
    BacktestMetricsCalculator,
    CLVResult,
    FullBacktestMetrics,
    calculate_calibration_curve,
)

__all__ = [
    "BacktestConfig",
    "BacktestMetricsCalculator",
    "BacktestResult",
    "CLVResult",
    "ConvergenceError",
    "DevigCalculator",
    "DevigError",
    "FairProbabilities",
    "FoldInfo",
    "FoldResult",
    "FullBacktestMetrics",
    "InvalidInputError",
    "InvalidOddsError",
    "KellyCalculator",
    "KellyError",
    "KellyOptimizationResult",
    "KellyResult",
    "OddsProviderProtocol",
    "SimulationResult",
    "TrainerProtocol",
    "WalkForwardEngine",
    "calculate_calibration_curve",
    "calculate_overround",
    "create_mock_trainer",
    "implied_probability",
    "solve_power_k",
    "solve_shin_z",
]
