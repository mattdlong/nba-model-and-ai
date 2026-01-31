"""Prediction and inference for NBA model.

This module provides the production inference pipeline, injury adjustments,
and betting signal generation for live predictions.

Submodules:
    inference: Production prediction pipeline
    injuries: Bayesian injury probability adjustments
    signals: Betting signal generation

Key concepts:
    - Inference pipeline: End-to-end prediction flow
    - Injury adjustment: Bayesian updates for GTD players
    - Betting signals: Actionable bets with positive edge

Example:
    >>> from nba_model.predict import InferencePipeline
    >>> pipeline = InferencePipeline(model_registry, db_session)
    >>> predictions = pipeline.predict_today()
    >>> for p in predictions:
    ...     print(f"{p.matchup}: {p.home_win_prob:.1%}")
"""

from __future__ import annotations

# Public API - will be populated in Phase 7
__all__: list[str] = []
