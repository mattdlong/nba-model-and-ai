"""Prediction and inference for NBA model.

This module provides the production inference pipeline, injury adjustments,
and betting signal generation for live predictions.

Submodules:
    inference: Production prediction pipeline
    injuries: Bayesian injury probability adjustments
    signals: Betting signal generation

Key Concepts:
    - Inference Pipeline: End-to-end prediction flow
    - Injury Adjustment: Bayesian updates for GTD players
    - Betting Signals: Actionable bets with positive edge

Example:
    >>> from nba_model.predict import InferencePipeline, SignalGenerator
    >>> from nba_model.models import ModelRegistry
    >>> from nba_model.backtest import DevigCalculator, KellyCalculator
    >>> from nba_model.data import session_scope
    >>>
    >>> with session_scope() as session:
    ...     # Generate predictions
    ...     pipeline = InferencePipeline(ModelRegistry(), session)
    ...     predictions = pipeline.predict_today()
    ...
    ...     # Generate signals
    ...     generator = SignalGenerator(DevigCalculator(), KellyCalculator())
    ...     signals = generator.generate_signals(predictions, market_odds)
    ...
    ...     for signal in signals:
    ...         print(f"{signal.matchup}: {signal.bet_type} {signal.side}")
    ...         print(f"  Edge: {signal.edge:.1%}, Stake: {signal.recommended_stake_pct:.2%}")
"""

from __future__ import annotations

# Inference pipeline
from nba_model.predict.inference import (
    GamePrediction,
    InferencePipeline,
    PredictionBatch,
    InferenceError,
    ModelLoadError,
    GameNotFoundError,
    FeatureExtractionError,
    create_mock_pipeline,
)

# Injury adjustments
from nba_model.predict.injuries import (
    InjuryAdjuster,
    InjuryReport,
    InjuryReportFetcher,
    InjuryStatus,
    PlayerAvailability,
    InjuryAdjustmentResult,
    PRIOR_PLAY_PROBABILITIES,
    parse_injury_status,
)

# Signal generation
from nba_model.predict.signals import (
    SignalGenerator,
    BettingSignal,
    MarketOdds,
    BetType,
    Side,
    Confidence,
    create_market_odds,
    american_to_decimal,
    decimal_to_american,
)

__all__ = [
    # Inference
    "InferencePipeline",
    "GamePrediction",
    "PredictionBatch",
    "InferenceError",
    "ModelLoadError",
    "GameNotFoundError",
    "FeatureExtractionError",
    "create_mock_pipeline",
    # Injuries
    "InjuryAdjuster",
    "InjuryReport",
    "InjuryReportFetcher",
    "InjuryStatus",
    "PlayerAvailability",
    "InjuryAdjustmentResult",
    "PRIOR_PLAY_PROBABILITIES",
    "parse_injury_status",
    # Signals
    "SignalGenerator",
    "BettingSignal",
    "MarketOdds",
    "BetType",
    "Side",
    "Confidence",
    "create_market_odds",
    "american_to_decimal",
    "decimal_to_american",
]
