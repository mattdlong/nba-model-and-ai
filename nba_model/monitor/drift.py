"""Covariate and concept drift detection for NBA model monitoring.

This module implements statistical tests to detect when input feature
distributions shift (covariate drift) or when the model's predictions
diverge from actual outcomes (concept drift).

Methods:
    - Kolmogorov-Smirnov (KS) test: Non-parametric distribution comparison
    - Population Stability Index (PSI): Measures magnitude of distribution shift
    - Prediction accuracy/calibration monitoring: Detects concept drift

Example:
    >>> detector = DriftDetector(reference_data)
    >>> result = detector.check_drift(recent_data)
    >>> if result['has_drift']:
    ...     print(f"Drifted features: {result['features_drifted']}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import ks_2samp

from nba_model.logging import get_logger

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)

# =============================================================================
# Constants
# =============================================================================

# Features to monitor for covariate drift
MONITORED_FEATURES: list[str] = [
    "pace",
    "offensive_rating",
    "fg3a_rate",
    "rest_days",
    "travel_distance",
    "rapm_mean",
]

# PSI interpretation thresholds
PSI_STABLE: float = 0.1
PSI_MODERATE: float = 0.2

# Small epsilon to avoid log(0) in PSI calculation
PSI_EPSILON: float = 1e-10

# Default thresholds
DEFAULT_P_VALUE_THRESHOLD: float = 0.05
DEFAULT_PSI_THRESHOLD: float = 0.2
DEFAULT_ACCURACY_THRESHOLD: float = 0.48
DEFAULT_CALIBRATION_THRESHOLD: float = 0.1

# Baseline Brier score (random prediction at 0.5)
BASELINE_BRIER_SCORE: float = 0.25


# =============================================================================
# Exceptions
# =============================================================================


class DriftDetectionError(Exception):
    """Base exception for drift detection errors."""


class InsufficientDataError(DriftDetectionError):
    """Raised when there is insufficient data for drift detection."""


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class KSTestResult:
    """Result of a Kolmogorov-Smirnov test.

    Attributes:
        statistic: KS test statistic (max difference between CDFs).
        p_value: P-value for the null hypothesis (same distribution).
    """

    statistic: float
    p_value: float


@dataclass(frozen=True)
class FeatureDriftDetail:
    """Drift detection details for a single feature.

    Attributes:
        ks_stat: KS test statistic.
        p_value: KS test p-value.
        psi: Population Stability Index value.
        is_drifted: Whether drift was detected for this feature.
    """

    ks_stat: float
    p_value: float
    psi: float
    is_drifted: bool


@dataclass(frozen=True)
class DriftCheckResult:
    """Result of a complete drift check across all features.

    Attributes:
        has_drift: Whether any feature shows significant drift.
        features_drifted: List of feature names that drifted.
        details: Per-feature drift detection details.
    """

    has_drift: bool
    features_drifted: list[str]
    details: dict[str, FeatureDriftDetail]

    def to_dict(self) -> dict[str, bool | list[str] | dict[str, dict[str, float]]]:
        """Convert to dictionary format for JSON serialization."""
        return {
            "has_drift": self.has_drift,
            "features_drifted": self.features_drifted,
            "details": {
                name: {
                    "ks_stat": detail.ks_stat,
                    "p_value": detail.p_value,
                    "psi": detail.psi,
                }
                for name, detail in self.details.items()
            },
        }


@dataclass(frozen=True)
class ConceptDriftResult:
    """Result of concept drift detection.

    Attributes:
        accuracy_degraded: Whether accuracy dropped below threshold.
        calibration_degraded: Whether Brier score exceeds threshold.
        recent_accuracy: Accuracy over the recent window.
        recent_brier_score: Brier score over the recent window.
    """

    accuracy_degraded: bool
    calibration_degraded: bool
    recent_accuracy: float
    recent_brier_score: float

    def to_dict(self) -> dict[str, bool | float]:
        """Convert to dictionary format."""
        return {
            "accuracy_degraded": self.accuracy_degraded,
            "calibration_degraded": self.calibration_degraded,
            "recent_accuracy": self.recent_accuracy,
            "recent_brier_score": self.recent_brier_score,
        }


# =============================================================================
# Covariate Drift Detector
# =============================================================================


class DriftDetector:
    """Detects covariate drift in input feature distributions.

    Monitors feature distributions for significant shifts from the reference
    (training) distribution using KS tests and PSI.

    Attributes:
        reference_data: Reference DataFrame from training period.
        p_value_threshold: Significance level for KS test.
        psi_threshold: PSI threshold for significant drift.

    Example:
        >>> detector = DriftDetector(training_data, p_value_threshold=0.05)
        >>> result = detector.check_drift(recent_production_data)
        >>> if result.has_drift:
        ...     trigger_retraining()
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        p_value_threshold: float = DEFAULT_P_VALUE_THRESHOLD,
        psi_threshold: float = DEFAULT_PSI_THRESHOLD,
    ) -> None:
        """Initialize DriftDetector with reference data.

        Args:
            reference_data: DataFrame containing reference feature distributions.
            p_value_threshold: KS test significance threshold (default 0.05).
            psi_threshold: PSI threshold for significant drift (default 0.2).

        Raises:
            InsufficientDataError: If reference data is empty or too small.
        """
        if reference_data.empty or len(reference_data) < 10:
            raise InsufficientDataError(
                "Reference data must contain at least 10 samples"
            )

        self.reference_data = reference_data
        self.p_value_threshold = p_value_threshold
        self.psi_threshold = psi_threshold

        logger.debug(
            "Initialized DriftDetector with {} reference samples",
            len(reference_data),
        )

    def ks_test(
        self,
        feature: str,
        recent_data: pd.DataFrame,
    ) -> tuple[float, float]:
        """Execute two-sample Kolmogorov-Smirnov test.

        Compares the distribution of a feature between reference and recent data.

        Args:
            feature: Name of the feature column to test.
            recent_data: DataFrame with recent feature values.

        Returns:
            Tuple of (test_statistic, p_value).

        Raises:
            InsufficientDataError: If either dataset lacks the feature or is empty.
        """
        if feature not in self.reference_data.columns:
            raise InsufficientDataError(
                f"Feature '{feature}' not found in reference data"
            )
        if feature not in recent_data.columns:
            raise InsufficientDataError(
                f"Feature '{feature}' not found in recent data"
            )

        ref_values = self.reference_data[feature].dropna()
        recent_values = recent_data[feature].dropna()

        if len(ref_values) < 5 or len(recent_values) < 5:
            raise InsufficientDataError(
                f"Need at least 5 non-null values for feature '{feature}'"
            )

        statistic, p_value = ks_2samp(ref_values, recent_values)

        logger.debug(
            "KS test for {}: statistic={:.4f}, p_value={:.4f}",
            feature,
            statistic,
            p_value,
        )

        return float(statistic), float(p_value)

    def calculate_psi(
        self,
        feature: str,
        recent_data: pd.DataFrame,
        n_bins: int = 10,
    ) -> float:
        """Calculate Population Stability Index.

        PSI measures the magnitude of distribution shift by comparing
        the proportion of values in each bin between reference and recent data.

        Formula: PSI = sum((actual_pct - expected_pct) * ln(actual_pct / expected_pct))

        Interpretation:
            - PSI < 0.1: No significant shift
            - 0.1 <= PSI < 0.2: Moderate shift
            - PSI >= 0.2: Significant shift

        Args:
            feature: Name of the feature column.
            recent_data: DataFrame with recent feature values.
            n_bins: Number of quantile bins (default 10).

        Returns:
            PSI value as float.

        Raises:
            InsufficientDataError: If data is insufficient for PSI calculation.
        """
        if feature not in self.reference_data.columns:
            raise InsufficientDataError(
                f"Feature '{feature}' not found in reference data"
            )
        if feature not in recent_data.columns:
            raise InsufficientDataError(
                f"Feature '{feature}' not found in recent data"
            )

        ref_values = np.asarray(self.reference_data[feature].dropna().values)
        recent_values = np.asarray(recent_data[feature].dropna().values)

        if len(ref_values) < n_bins or len(recent_values) < n_bins:
            raise InsufficientDataError(
                f"Need at least {n_bins} non-null values for PSI calculation"
            )

        # Create bins based on reference data quantiles
        try:
            bin_edges = np.percentile(
                ref_values,
                np.linspace(0, 100, n_bins + 1),
            )
            # Ensure unique bin edges by adding small noise
            bin_edges = np.unique(bin_edges)
            if len(bin_edges) < 3:
                # Fall back to equal-width bins if quantiles don't work
                bin_edges = np.linspace(
                    float(min(ref_values.min(), recent_values.min())),
                    float(max(ref_values.max(), recent_values.max())),
                    n_bins + 1,
                )
        except Exception as e:
            raise InsufficientDataError(f"Failed to create bins: {e}") from e

        # Count values in each bin
        ref_counts, _ = np.histogram(ref_values, bins=bin_edges)
        recent_counts, _ = np.histogram(recent_values, bins=bin_edges)

        # Convert to proportions
        ref_pcts = ref_counts / len(ref_values)
        recent_pcts = recent_counts / len(recent_values)

        # Add epsilon to avoid log(0)
        ref_pcts = np.clip(ref_pcts, PSI_EPSILON, 1.0)
        recent_pcts = np.clip(recent_pcts, PSI_EPSILON, 1.0)

        # Calculate PSI
        psi = np.sum(
            (recent_pcts - ref_pcts) * np.log(recent_pcts / ref_pcts)
        )

        logger.debug("PSI for {}: {:.4f}", feature, psi)

        return float(psi)

    def check_drift(
        self,
        recent_data: pd.DataFrame,
        window_days: int = 30,
    ) -> DriftCheckResult:
        """Check all monitored features for drift.

        Runs both KS test and PSI calculation for each monitored feature
        and flags drift if either test indicates a significant shift.

        Args:
            recent_data: DataFrame with recent feature values.
            window_days: Number of days in the recent window (for logging).

        Returns:
            DriftCheckResult with comprehensive drift analysis.
        """
        features_drifted: list[str] = []
        details: dict[str, FeatureDriftDetail] = {}

        # Check each monitored feature
        for feature in MONITORED_FEATURES:
            if feature not in self.reference_data.columns:
                logger.warning("Monitored feature '{}' not in reference data", feature)
                continue
            if feature not in recent_data.columns:
                logger.warning("Monitored feature '{}' not in recent data", feature)
                continue

            try:
                # Run KS test
                ks_stat, p_value = self.ks_test(feature, recent_data)

                # Calculate PSI
                psi = self.calculate_psi(feature, recent_data)

                # Determine if drifted (either test fails)
                is_drifted = p_value < self.p_value_threshold or psi > self.psi_threshold

                details[feature] = FeatureDriftDetail(
                    ks_stat=ks_stat,
                    p_value=p_value,
                    psi=psi,
                    is_drifted=is_drifted,
                )

                if is_drifted:
                    features_drifted.append(feature)
                    logger.info(
                        "Drift detected for {}: KS p={:.4f}, PSI={:.4f}",
                        feature,
                        p_value,
                        psi,
                    )

            except InsufficientDataError as e:
                logger.warning("Skipping feature '{}': {}", feature, e)
                continue

        has_drift = len(features_drifted) > 0

        if has_drift:
            logger.warning(
                "Covariate drift detected in {} features: {}",
                len(features_drifted),
                features_drifted,
            )
        else:
            logger.info("No covariate drift detected")

        return DriftCheckResult(
            has_drift=has_drift,
            features_drifted=features_drifted,
            details=details,
        )


# =============================================================================
# Concept Drift Detector
# =============================================================================


class ConceptDriftDetector:
    """Detects concept drift by monitoring prediction-outcome divergence.

    Monitors when the model's predictions systematically diverge from
    actual outcomes, indicating the underlying relationship between
    features and targets has changed.

    Attributes:
        accuracy_threshold: Minimum acceptable accuracy (default 0.48).
        calibration_threshold: Maximum acceptable calibration error.

    Example:
        >>> detector = ConceptDriftDetector(accuracy_threshold=0.48)
        >>> result = detector.check_prediction_drift(predictions, actuals)
        >>> if result.accuracy_degraded:
        ...     retrain_model()
    """

    def __init__(
        self,
        accuracy_threshold: float = DEFAULT_ACCURACY_THRESHOLD,
        calibration_threshold: float = DEFAULT_CALIBRATION_THRESHOLD,
    ) -> None:
        """Initialize ConceptDriftDetector.

        Args:
            accuracy_threshold: Minimum acceptable win prediction accuracy.
            calibration_threshold: Maximum deviation from baseline Brier score.
        """
        if not 0 < accuracy_threshold < 1:
            raise ValueError(
                f"accuracy_threshold must be in (0, 1), got {accuracy_threshold}"
            )
        if calibration_threshold < 0:
            raise ValueError(
                f"calibration_threshold must be >= 0, got {calibration_threshold}"
            )

        self.accuracy_threshold = accuracy_threshold
        self.calibration_threshold = calibration_threshold

        logger.debug(
            "Initialized ConceptDriftDetector with accuracy_threshold={}, "
            "calibration_threshold={}",
            accuracy_threshold,
            calibration_threshold,
        )

    def check_prediction_drift(
        self,
        predictions: list[float],
        actuals: list[int],
        window_size: int = 100,
    ) -> ConceptDriftResult:
        """Check if recent predictions diverge from actual outcomes.

        Analyzes the most recent `window_size` predictions for accuracy
        degradation and calibration issues.

        Args:
            predictions: List of model probability predictions (0 to 1).
            actuals: List of actual outcomes (0 for loss, 1 for win).
            window_size: Number of recent predictions to analyze.

        Returns:
            ConceptDriftResult with drift analysis.

        Raises:
            InsufficientDataError: If fewer than 10 predictions provided.
        """
        if len(predictions) != len(actuals):
            raise ValueError(
                f"Length mismatch: {len(predictions)} predictions vs {len(actuals)} actuals"
            )

        if len(predictions) < 10:
            raise InsufficientDataError(
                "Need at least 10 predictions for drift detection"
            )

        # Use most recent window_size predictions
        recent_predictions = predictions[-window_size:]
        recent_actuals = actuals[-window_size:]

        preds_array = np.array(recent_predictions)
        actuals_array = np.array(recent_actuals)

        # Calculate accuracy (predicted winner vs actual)
        predicted_wins = preds_array >= 0.5
        actual_wins = actuals_array == 1
        recent_accuracy = np.mean(predicted_wins == actual_wins)

        # Calculate Brier score: mean((prob - outcome)^2)
        recent_brier_score = np.mean((preds_array - actuals_array) ** 2)

        # Check for degradation
        accuracy_degraded = recent_accuracy < self.accuracy_threshold
        calibration_degraded = (
            recent_brier_score > BASELINE_BRIER_SCORE + self.calibration_threshold
        )

        if accuracy_degraded:
            logger.warning(
                "Accuracy degradation detected: {:.2%} < {:.2%}",
                recent_accuracy,
                self.accuracy_threshold,
            )
        if calibration_degraded:
            logger.warning(
                "Calibration degradation detected: Brier={:.4f} > {:.4f}",
                recent_brier_score,
                BASELINE_BRIER_SCORE + self.calibration_threshold,
            )

        return ConceptDriftResult(
            accuracy_degraded=accuracy_degraded,
            calibration_degraded=calibration_degraded,
            recent_accuracy=float(recent_accuracy),
            recent_brier_score=float(recent_brier_score),
        )
