"""Regularized Adjusted Plus-Minus (RAPM) calculator.

This module implements RAPM calculation using Ridge Regression on lineup stint data.
RAPM measures a player's impact on team point differential per 100 possessions,
controlling for teammates and opponents.

Mathematical Model:
    Solve Y = Xβ + ε where:
    - Y = point differential per 100 possessions per stint
    - X = sparse design matrix (+1 for home players, -1 for away players, 0 otherwise)
    - β = RAPM coefficients (offensive, defensive, total)

Ridge Regression minimizes: ||Y - Xβ||² + λ||β||²

Example:
    >>> from nba_model.features.rapm import RAPMCalculator
    >>> calculator = RAPMCalculator(lambda_=5000)
    >>> coefficients = calculator.fit(stints_df)
    >>> print(coefficients[203507])  # Player's RAPM
    {'orapm': 2.1, 'drapm': 0.8, 'total_rapm': 2.9, 'sample_stints': 1500}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge

from nba_model.logging import get_logger
from nba_model.types import InsufficientDataError, PlayerId, RAPMCoefficients

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)

# Constants
DEFAULT_LAMBDA: float = 5000.0
DEFAULT_MIN_MINUTES: int = 100
DEFAULT_TIME_DECAY_TAU: float = 180.0
MIN_STINTS_FOR_CALCULATION: int = 100
POSSESSIONS_PER_100: float = 100.0


@dataclass
class RAPMResult:
    """Container for RAPM calculation results."""

    player_id: PlayerId
    orapm: float
    drapm: float
    total_rapm: float
    sample_stints: int


class RAPMCalculator:
    """Regularized Adjusted Plus-Minus calculator using Ridge Regression.

    Calculates player impact metrics by solving the regression:
        Y = Xβ + ε
    where Y is point differential per 100 possessions, X is the sparse
    player-stint design matrix, and β are the RAPM coefficients.

    Attributes:
        lambda_: Ridge regularization strength (default 5000).
        min_minutes: Minimum player minutes for inclusion (default 100).
        time_decay_tau: Half-life for time decay weighting in days.
        player_mapping: Dict mapping player_id to matrix column index.
        fitted_: Whether the calculator has been fitted.

    Example:
        >>> calculator = RAPMCalculator(lambda_=5000)
        >>> coefficients = calculator.fit(stints_df)
        >>> print(coefficients[203507])  # LeBron's RAPM
    """

    def __init__(
        self,
        lambda_: float = DEFAULT_LAMBDA,
        min_minutes: int = DEFAULT_MIN_MINUTES,
        time_decay_tau: float = DEFAULT_TIME_DECAY_TAU,
    ) -> None:
        """Initialize RAPM calculator.

        Args:
            lambda_: Ridge regularization strength. Higher values shrink
                coefficients toward zero. Typical range: 1000-10000.
            min_minutes: Minimum minutes played for player to be included
                in calculation. Players below threshold are excluded.
            time_decay_tau: Half-life for exponential time decay weighting
                in days. Default 180 days = ~6 months.
        """
        self.lambda_ = lambda_
        self.min_minutes = min_minutes
        self.time_decay_tau = time_decay_tau
        self.player_mapping: dict[PlayerId, int] = {}
        self.fitted_ = False
        self._offensive_coefficients: np.ndarray | None = None
        self._defensive_coefficients: np.ndarray | None = None

    def _parse_lineup(self, lineup_json: str | list[int]) -> list[int]:
        """Parse lineup from JSON string or list.

        Args:
            lineup_json: JSON string array or list of player IDs.

        Returns:
            List of player IDs.
        """
        if isinstance(lineup_json, str):
            return json.loads(lineup_json)
        return list(lineup_json)

    def _get_eligible_players(
        self,
        stints_df: pd.DataFrame,
        player_minutes: dict[PlayerId, float],
    ) -> list[PlayerId]:
        """Get players meeting minimum minutes threshold.

        Args:
            stints_df: DataFrame with stint data.
            player_minutes: Dict mapping player_id to total minutes.

        Returns:
            List of eligible player IDs sorted for deterministic ordering.
        """
        eligible = [
            pid
            for pid, minutes in player_minutes.items()
            if minutes >= self.min_minutes
        ]
        return sorted(eligible)

    def _calculate_player_minutes(
        self,
        stints_df: pd.DataFrame,
    ) -> dict[PlayerId, float]:
        """Calculate total minutes per player from stints.

        Args:
            stints_df: DataFrame with columns home_lineup, away_lineup, duration_seconds.

        Returns:
            Dict mapping player_id to total minutes played.
        """
        player_minutes: dict[PlayerId, float] = {}

        for _, row in stints_df.iterrows():
            duration_minutes = row["duration_seconds"] / 60.0
            home_lineup = self._parse_lineup(row["home_lineup"])
            away_lineup = self._parse_lineup(row["away_lineup"])

            for player_id in home_lineup + away_lineup:
                if player_id not in player_minutes:
                    player_minutes[player_id] = 0.0
                player_minutes[player_id] += duration_minutes

        return player_minutes

    def _calculate_time_weights(
        self,
        game_dates: pd.Series,
        reference_date: date | None = None,
    ) -> np.ndarray:
        """Calculate exponential time decay weights.

        Weight formula: w = exp(-(reference_date - game_date) / tau)

        Args:
            game_dates: Series of game dates.
            reference_date: Reference date for decay. Defaults to max date in data.

        Returns:
            Array of weights in [0, 1].
        """
        import pandas as pd

        if reference_date is None:
            reference_date = pd.Timestamp(game_dates.max()).date()
        elif isinstance(reference_date, pd.Timestamp):
            reference_date = reference_date.date()

        # Convert game_dates to days ago
        days_ago = np.array(
            [(reference_date - pd.Timestamp(d).date()).days for d in game_dates],
            dtype=np.float64,
        )

        # Exponential decay
        weights = np.exp(-days_ago / self.time_decay_tau)

        return weights

    def build_stint_matrix(
        self,
        stints_df: pd.DataFrame,
        players: list[PlayerId],
        reference_date: date | None = None,
    ) -> tuple[csr_matrix, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build sparse design matrix from stints table.

        The design matrix X has:
        - Rows = stints
        - Columns = players
        - Values = +1 for home players, -1 for away players, 0 otherwise

        Args:
            stints_df: DataFrame with columns:
                - home_lineup: JSON array of 5 home player IDs
                - away_lineup: JSON array of 5 away player IDs
                - home_points: Points scored by home team
                - away_points: Points scored by away team
                - possessions: Number of possessions in stint
                - duration_seconds: Duration of stint
                - game_date: Date of game (for time decay)
            players: List of player IDs to include (defines column order).
            reference_date: Reference date for time decay calculation.

        Returns:
            Tuple of (X, y_offense, y_defense, weights, possessions) where:
                - X: Sparse design matrix (n_stints, n_players)
                - y_offense: Home points per 100 possessions
                - y_defense: Away points per 100 possessions
                - weights: Time decay weights
                - possessions: Possession counts per stint

        Raises:
            InsufficientDataError: If fewer than MIN_STINTS_FOR_CALCULATION stints.
        """
        n_stints = len(stints_df)
        n_players = len(players)

        if n_stints < MIN_STINTS_FOR_CALCULATION:
            raise InsufficientDataError(
                f"Need at least {MIN_STINTS_FOR_CALCULATION} stints for RAPM calculation, "
                f"got {n_stints}"
            )

        # Create player to column mapping
        self.player_mapping = {pid: idx for idx, pid in enumerate(players)}

        # Prepare sparse matrix data
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []

        # Target vectors
        y_offense = np.zeros(n_stints, dtype=np.float64)
        y_defense = np.zeros(n_stints, dtype=np.float64)
        possessions = np.zeros(n_stints, dtype=np.float64)

        for stint_idx, (_, row) in enumerate(stints_df.iterrows()):
            home_lineup = self._parse_lineup(row["home_lineup"])
            away_lineup = self._parse_lineup(row["away_lineup"])

            # Add home players with +1
            for player_id in home_lineup:
                if player_id in self.player_mapping:
                    rows.append(stint_idx)
                    cols.append(self.player_mapping[player_id])
                    data.append(1.0)

            # Add away players with -1
            for player_id in away_lineup:
                if player_id in self.player_mapping:
                    rows.append(stint_idx)
                    cols.append(self.player_mapping[player_id])
                    data.append(-1.0)

            # Calculate per-100 possession rates
            poss = (
                row["possessions"]
                if row["possessions"] and row["possessions"] > 0
                else 1.0
            )
            possessions[stint_idx] = poss

            # Home points scored per 100 possessions (offensive)
            y_offense[stint_idx] = row["home_points"] * POSSESSIONS_PER_100 / poss
            # Away points scored per 100 possessions (defensive - points allowed)
            y_defense[stint_idx] = row["away_points"] * POSSESSIONS_PER_100 / poss

        # Build sparse matrix
        X = csr_matrix((data, (rows, cols)), shape=(n_stints, n_players))

        # Calculate time weights
        weights = self._calculate_time_weights(stints_df["game_date"], reference_date)

        logger.debug(
            "Built design matrix: {} stints x {} players, density={:.4f}",
            n_stints,
            n_players,
            X.nnz / (n_stints * n_players),
        )

        return X, y_offense, y_defense, weights, possessions

    def fit(
        self,
        stints_df: pd.DataFrame,
        reference_date: date | None = None,
    ) -> dict[PlayerId, RAPMCoefficients]:
        """Fit Ridge Regression and return RAPM coefficients.

        Runs two separate regressions:
        1. Offensive RAPM: Home points scored per 100 possessions
        2. Defensive RAPM: Away points allowed per 100 possessions (negated)

        Total RAPM = ORAPM - DRAPM (offensive contribution minus defensive allowance)

        Args:
            stints_df: DataFrame with stint data (see build_stint_matrix for columns).
            reference_date: Reference date for time decay. Defaults to max date in data.

        Returns:
            Dict mapping player_id to RAPMCoefficients TypedDict with keys:
                - player_id: Player ID
                - orapm: Offensive RAPM
                - drapm: Defensive RAPM (positive = good defense)
                - total_rapm: Total RAPM
                - sample_stints: Number of stints used

        Raises:
            InsufficientDataError: If not enough data for calculation.
        """
        logger.info("Starting RAPM calculation with {} stints", len(stints_df))

        # Calculate player minutes and filter eligible players
        player_minutes = self._calculate_player_minutes(stints_df)
        eligible_players = self._get_eligible_players(stints_df, player_minutes)

        if len(eligible_players) < 10:
            raise InsufficientDataError(
                f"Need at least 10 eligible players, got {len(eligible_players)}"
            )

        logger.info(
            "Found {} eligible players (>= {} minutes)",
            len(eligible_players),
            self.min_minutes,
        )

        # Build design matrix
        X, y_offense, y_defense, weights, possessions = self.build_stint_matrix(
            stints_df, eligible_players, reference_date
        )

        # Apply sample weights (possession-weighted and time-weighted)
        sample_weights = weights * np.sqrt(possessions)

        # Fit offensive RAPM
        ridge_offense = Ridge(alpha=self.lambda_, fit_intercept=True)
        ridge_offense.fit(X, y_offense, sample_weight=sample_weights)
        self._offensive_coefficients = ridge_offense.coef_

        # Fit defensive RAPM
        # For defense, we want to minimize points allowed, so we negate y_defense
        # This makes positive DRAPM = good defense (allows fewer points)
        ridge_defense = Ridge(alpha=self.lambda_, fit_intercept=True)
        ridge_defense.fit(X, -y_defense, sample_weight=sample_weights)
        self._defensive_coefficients = ridge_defense.coef_

        # Calculate stint counts per player
        player_stint_counts = self._calculate_player_stint_counts(
            stints_df, eligible_players
        )

        # Build results dictionary
        results: dict[PlayerId, RAPMCoefficients] = {}
        for player_id, col_idx in self.player_mapping.items():
            orapm = float(self._offensive_coefficients[col_idx])
            drapm = float(self._defensive_coefficients[col_idx])
            total_rapm = orapm + drapm

            results[player_id] = RAPMCoefficients(
                player_id=player_id,
                orapm=round(orapm, 3),
                drapm=round(drapm, 3),
                total_rapm=round(total_rapm, 3),
                sample_stints=player_stint_counts.get(player_id, 0),
            )

        self.fitted_ = True
        logger.info(
            "RAPM calculation complete for {} players",
            len(results),
        )

        return results

    def _calculate_player_stint_counts(
        self,
        stints_df: pd.DataFrame,
        players: list[PlayerId],
    ) -> dict[PlayerId, int]:
        """Count stints each player appeared in.

        Args:
            stints_df: DataFrame with stint data.
            players: List of player IDs to count.

        Returns:
            Dict mapping player_id to stint count.
        """
        player_set = set(players)
        counts: dict[PlayerId, int] = dict.fromkeys(players, 0)

        for _, row in stints_df.iterrows():
            home_lineup = self._parse_lineup(row["home_lineup"])
            away_lineup = self._parse_lineup(row["away_lineup"])

            for player_id in home_lineup + away_lineup:
                if player_id in player_set:
                    counts[player_id] += 1

        return counts

    def cross_validate_lambda(
        self,
        stints_df: pd.DataFrame,
        lambdas: list[float] | None = None,
        cv_folds: int = 5,
    ) -> tuple[float, dict[float, float]]:
        """Find optimal lambda via cross-validation.

        Tests different regularization strengths and returns the one
        with best cross-validated R² score.

        Args:
            stints_df: DataFrame with stint data.
            lambdas: List of lambda values to test. Defaults to [1000, 2000, 5000, 10000].
            cv_folds: Number of cross-validation folds.

        Returns:
            Tuple of (best_lambda, scores_dict) where scores_dict maps
            lambda to mean CV score.

        Raises:
            InsufficientDataError: If not enough data for CV.
        """
        if lambdas is None:
            lambdas = [1000.0, 2000.0, 5000.0, 10000.0]

        logger.info("Cross-validating lambda with {} folds", cv_folds)

        # Calculate player minutes and get eligible players
        player_minutes = self._calculate_player_minutes(stints_df)
        eligible_players = self._get_eligible_players(stints_df, player_minutes)

        # Build design matrix
        X, y_offense, _, weights, possessions = self.build_stint_matrix(
            stints_df, eligible_players
        )

        sample_weights = weights * np.sqrt(possessions)

        # Use KFold for manual cross-validation with sample weights
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

        scores: dict[float, float] = {}
        for lambda_val in lambdas:
            fold_scores = []
            for train_idx, test_idx in kf.split(X):
                X_train = X[train_idx]
                X_test = X[test_idx]
                y_train = y_offense[train_idx]
                y_test = y_offense[test_idx]
                w_train = sample_weights[train_idx]

                ridge = Ridge(alpha=lambda_val, fit_intercept=True)
                ridge.fit(X_train, y_train, sample_weight=w_train)
                score = ridge.score(X_test, y_test)
                fold_scores.append(score)

            mean_score = float(np.mean(fold_scores))
            scores[lambda_val] = mean_score
            logger.debug("Lambda={}: CV R²={:.4f}", lambda_val, mean_score)

        best_lambda = max(scores, key=scores.get)  # type: ignore[arg-type]
        logger.info("Best lambda: {} (R²={:.4f})", best_lambda, scores[best_lambda])

        return best_lambda, scores

    def save(self, path: Path) -> None:
        """Save calculator state to disk.

        Args:
            path: Path to save file (JSON format).

        Raises:
            ValueError: If calculator has not been fitted.
        """
        if not self.fitted_:
            raise ValueError("Calculator has not been fitted")

        state = {
            "lambda_": self.lambda_,
            "min_minutes": self.min_minutes,
            "time_decay_tau": self.time_decay_tau,
            "player_mapping": {str(k): v for k, v in self.player_mapping.items()},
            "offensive_coefficients": (
                self._offensive_coefficients.tolist()  # type: ignore[union-attr]
                if self._offensive_coefficients is not None
                else None
            ),
            "defensive_coefficients": (
                self._defensive_coefficients.tolist()  # type: ignore[union-attr]
                if self._defensive_coefficients is not None
                else None
            ),
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

        logger.info("Saved RAPM calculator to {}", path)

    def load(self, path: Path) -> None:
        """Load calculator state from disk.

        Args:
            path: Path to saved state file.

        Raises:
            FileNotFoundError: If file doesn't exist.
        """
        with open(path) as f:
            state = json.load(f)

        self.lambda_ = state["lambda_"]
        self.min_minutes = state["min_minutes"]
        self.time_decay_tau = state["time_decay_tau"]
        self.player_mapping = {int(k): v for k, v in state["player_mapping"].items()}
        self._offensive_coefficients = (
            np.array(state["offensive_coefficients"])
            if state["offensive_coefficients"]
            else None
        )
        self._defensive_coefficients = (
            np.array(state["defensive_coefficients"])
            if state["defensive_coefficients"]
            else None
        )
        self.fitted_ = True

        logger.info("Loaded RAPM calculator from {}", path)


def calculate_rapm_for_season(
    stints_df: pd.DataFrame,
    lambda_: float = DEFAULT_LAMBDA,
    min_minutes: int = DEFAULT_MIN_MINUTES,
    reference_date: date | None = None,
) -> dict[PlayerId, RAPMCoefficients]:
    """Convenience function to calculate RAPM for a season.

    Args:
        stints_df: DataFrame with stint data.
        lambda_: Ridge regularization strength.
        min_minutes: Minimum minutes for player inclusion.
        reference_date: Reference date for time decay.

    Returns:
        Dict mapping player_id to RAPM coefficients.
    """
    calculator = RAPMCalculator(lambda_=lambda_, min_minutes=min_minutes)
    return calculator.fit(stints_df, reference_date)


__all__ = [
    "DEFAULT_LAMBDA",
    "DEFAULT_MIN_MINUTES",
    "DEFAULT_TIME_DECAY_TAU",
    "RAPMCalculator",
    "RAPMResult",
    "calculate_rapm_for_season",
]
