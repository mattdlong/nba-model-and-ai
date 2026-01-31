"""Type definitions and protocols for the NBA model.

This module defines common types, protocols, and type aliases used throughout
the application. Using protocols enables duck typing with static type checking.

Example:
    >>> from nba_model.types import FeatureCalculator
    >>> def process(calculator: FeatureCalculator) -> None:
    ...     result = calculator.calculate(game_id="123")
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Protocol, TypedDict

import numpy as np
import pandas as pd

# =============================================================================
# Type Aliases
# =============================================================================

PlayerId = int
TeamId = int
GameId = str
SeasonId = str


# =============================================================================
# Protocols (Interfaces)
# =============================================================================


class FeatureCalculator(Protocol):
    """Protocol for feature calculator classes.

    Any class implementing these methods satisfies this protocol.
    """

    def fit(self, data: pd.DataFrame) -> None:
        """Fit the calculator to training data."""
        ...

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """Transform data into features."""
        ...

    def save(self, path: Path) -> None:
        """Save calculator state to disk."""
        ...

    def load(self, path: Path) -> None:
        """Load calculator state from disk."""
        ...


class DataCollector(Protocol):
    """Protocol for data collector classes."""

    def collect(self, season: SeasonId) -> pd.DataFrame:
        """Collect data for a season."""
        ...

    def update(self) -> pd.DataFrame:
        """Perform incremental update."""
        ...


class ModelPredictor(Protocol):
    """Protocol for model prediction classes."""

    def predict(self, game_id: GameId) -> dict[str, float]:
        """Generate prediction for a game."""
        ...

    def predict_batch(self, game_ids: list[GameId]) -> list[dict[str, float]]:
        """Generate predictions for multiple games."""
        ...


# =============================================================================
# TypedDicts for Structured Data
# =============================================================================


class RAPMCoefficients(TypedDict):
    """RAPM calculation results for a player."""

    player_id: PlayerId
    orapm: float
    drapm: float
    total_rapm: float
    sample_stints: int


class SpacingMetrics(TypedDict):
    """Convex hull spacing metrics for a lineup."""

    hull_area: float
    centroid_x: float
    centroid_y: float
    avg_distance: float
    corner_density: float
    shot_count: int


# FatigueIndicators uses functional form because keys start with digits (3_in_4, 4_in_5)
FatigueIndicators = TypedDict(
    "FatigueIndicators",
    {
        "rest_days": int,
        "back_to_back": bool,
        "3_in_4": bool,
        "4_in_5": bool,
        "travel_miles": float,
        "home_stand": int,
        "road_trip": int,
    },
)


class DriftResult(TypedDict):
    """Covariate drift detection results."""

    has_drift: bool
    features_drifted: list[str]
    details: dict[str, dict[str, float]]


class PredictionResult(TypedDict):
    """Game prediction output."""

    game_id: GameId
    home_win_prob: float
    predicted_margin: float
    predicted_total: float
    confidence: float
    model_version: str


class BettingSignal(TypedDict):
    """Actionable betting signal."""

    game_id: GameId
    bet_type: str
    side: str
    model_prob: float
    market_prob: float
    edge: float
    kelly_fraction: float
    recommended_stake_pct: float
    confidence: str


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass(frozen=True)
class GameInfo:
    """Immutable game information container."""

    game_id: GameId
    season_id: SeasonId
    game_date: date
    home_team_id: TeamId
    away_team_id: TeamId
    home_team: str
    away_team: str


@dataclass
class Bet:
    """Single bet record."""

    game_id: GameId
    timestamp: datetime
    bet_type: str
    side: str
    model_prob: float
    market_odds: float
    market_prob: float
    edge: float
    kelly_fraction: float
    bet_amount: float
    result: str | None = None
    profit: float | None = None


@dataclass
class BacktestMetrics:
    """Backtest performance metrics."""

    total_return: float
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    total_bets: int
    avg_edge: float
    avg_clv: float
    roi: float


@dataclass
class ModelMetadata:
    """Model version metadata."""

    version: str
    training_date: datetime
    training_data_start: date
    training_data_end: date
    hyperparameters: dict[str, Any]
    validation_metrics: dict[str, float]
    git_commit: str | None = None


# =============================================================================
# Exceptions
# =============================================================================


class NBAModelError(Exception):
    """Base exception for NBA model errors."""


class DataCollectionError(NBAModelError):
    """Error during data collection."""


class RateLimitExceeded(DataCollectionError):
    """API rate limit exceeded."""


class GameNotFound(DataCollectionError):
    """Requested game not found."""


class InsufficientDataError(NBAModelError):
    """Not enough data for calculation."""


class ModelNotFoundError(NBAModelError):
    """Requested model version not found."""


class DriftDetectedError(NBAModelError):
    """Significant drift detected in features."""
