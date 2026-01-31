"""Unit tests for monitor.drift module.

Tests covariate and concept drift detection functionality including
KS tests, PSI calculations, and accuracy/calibration monitoring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_model.monitor.drift import (
    MONITORED_FEATURES,
    ConceptDriftDetector,
    ConceptDriftResult,
    DriftCheckResult,
    DriftDetector,
    InsufficientDataError,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_reference_data() -> pd.DataFrame:
    """Create sample reference data with all monitored features."""
    np.random.seed(42)
    n_samples = 100
    return pd.DataFrame({
        "pace": np.random.normal(100, 5, n_samples),
        "offensive_rating": np.random.normal(110, 8, n_samples),
        "fg3a_rate": np.random.normal(0.35, 0.05, n_samples),
        "rest_days": np.random.randint(1, 5, n_samples).astype(float),
        "travel_distance": np.random.exponential(500, n_samples),
        "rapm_mean": np.random.normal(0, 2, n_samples),
    })


@pytest.fixture
def sample_identical_data(sample_reference_data: pd.DataFrame) -> pd.DataFrame:
    """Create recent data identical to reference (no drift)."""
    np.random.seed(42)  # Same seed for identical distribution
    n_samples = 100
    return pd.DataFrame({
        "pace": np.random.normal(100, 5, n_samples),
        "offensive_rating": np.random.normal(110, 8, n_samples),
        "fg3a_rate": np.random.normal(0.35, 0.05, n_samples),
        "rest_days": np.random.randint(1, 5, n_samples).astype(float),
        "travel_distance": np.random.exponential(500, n_samples),
        "rapm_mean": np.random.normal(0, 2, n_samples),
    })


@pytest.fixture
def sample_drifted_data() -> pd.DataFrame:
    """Create recent data with significant drift in multiple features."""
    np.random.seed(43)  # Different seed
    n_samples = 100
    return pd.DataFrame({
        # Pace shifted significantly higher
        "pace": np.random.normal(115, 5, n_samples),
        # Offensive rating shifted
        "offensive_rating": np.random.normal(125, 8, n_samples),
        # 3PT rate increased significantly
        "fg3a_rate": np.random.normal(0.50, 0.05, n_samples),
        # Rest days same
        "rest_days": np.random.randint(1, 5, n_samples).astype(float),
        # Travel distance same
        "travel_distance": np.random.exponential(500, n_samples),
        # RAPM shifted
        "rapm_mean": np.random.normal(3, 2, n_samples),
    })


@pytest.fixture
def drift_detector(sample_reference_data: pd.DataFrame) -> DriftDetector:
    """Create a DriftDetector with sample reference data."""
    return DriftDetector(sample_reference_data)


# =============================================================================
# DriftDetector Tests
# =============================================================================


class TestDriftDetector:
    """Tests for DriftDetector class."""

    def test_init_with_valid_data(
        self,
        sample_reference_data: pd.DataFrame,
    ) -> None:
        """DriftDetector should initialize with valid reference data."""
        detector = DriftDetector(sample_reference_data)
        assert detector.p_value_threshold == 0.05
        assert detector.psi_threshold == 0.2
        assert len(detector.reference_data) == 100

    def test_init_with_custom_thresholds(
        self,
        sample_reference_data: pd.DataFrame,
    ) -> None:
        """DriftDetector should accept custom thresholds."""
        detector = DriftDetector(
            sample_reference_data,
            p_value_threshold=0.01,
            psi_threshold=0.3,
        )
        assert detector.p_value_threshold == 0.01
        assert detector.psi_threshold == 0.3

    def test_init_rejects_empty_data(self) -> None:
        """DriftDetector should reject empty reference data."""
        with pytest.raises(InsufficientDataError, match="at least 10 samples"):
            DriftDetector(pd.DataFrame())

    def test_init_rejects_small_data(self) -> None:
        """DriftDetector should reject data with < 10 samples."""
        small_data = pd.DataFrame({"pace": [1, 2, 3, 4, 5]})
        with pytest.raises(InsufficientDataError, match="at least 10 samples"):
            DriftDetector(small_data)


class TestKSTest:
    """Tests for KS test functionality."""

    def test_ks_test_identical_distributions_high_pvalue(
        self,
        drift_detector: DriftDetector,
        sample_identical_data: pd.DataFrame,
    ) -> None:
        """KS test should return high p-value for identical distributions."""
        stat, p_value = drift_detector.ks_test("pace", sample_identical_data)

        assert isinstance(stat, float)
        assert isinstance(p_value, float)
        assert 0 <= stat <= 1
        assert 0 <= p_value <= 1
        # For identical distributions, p-value should be relatively high
        assert p_value > 0.05

    def test_ks_test_shifted_distribution_low_pvalue(
        self,
        drift_detector: DriftDetector,
        sample_drifted_data: pd.DataFrame,
    ) -> None:
        """KS test should return low p-value for shifted distributions."""
        stat, p_value = drift_detector.ks_test("pace", sample_drifted_data)

        assert isinstance(stat, float)
        assert isinstance(p_value, float)
        # For shifted distribution, p-value should be low
        assert p_value < 0.05

    def test_ks_test_missing_feature_in_reference(
        self,
        drift_detector: DriftDetector,
    ) -> None:
        """KS test should raise error for missing feature in reference."""
        recent_data = pd.DataFrame({"unknown_feature": [1, 2, 3, 4, 5] * 10})

        with pytest.raises(InsufficientDataError, match="not found in reference"):
            drift_detector.ks_test("unknown_feature", recent_data)

    def test_ks_test_missing_feature_in_recent(
        self,
        drift_detector: DriftDetector,
    ) -> None:
        """KS test should raise error for missing feature in recent data."""
        recent_data = pd.DataFrame({"other_feature": [1, 2, 3, 4, 5] * 10})

        with pytest.raises(InsufficientDataError, match="not found in recent"):
            drift_detector.ks_test("pace", recent_data)


class TestPSICalculation:
    """Tests for PSI calculation."""

    def test_psi_identical_distributions_near_zero(
        self,
        drift_detector: DriftDetector,
        sample_identical_data: pd.DataFrame,
    ) -> None:
        """PSI should be near zero for identical distributions."""
        psi = drift_detector.calculate_psi("pace", sample_identical_data)

        assert isinstance(psi, float)
        assert psi >= 0
        # For identical distributions, PSI should be very low
        assert psi < 0.1

    def test_psi_heavily_shifted_above_threshold(
        self,
        drift_detector: DriftDetector,
        sample_drifted_data: pd.DataFrame,
    ) -> None:
        """PSI should exceed threshold for heavily shifted distributions."""
        psi = drift_detector.calculate_psi("pace", sample_drifted_data)

        assert isinstance(psi, float)
        # For heavily shifted distribution, PSI should exceed 0.2 threshold
        assert psi > 0.2  # Significant shift per Phase 6 spec

    def test_psi_custom_bin_count(
        self,
        drift_detector: DriftDetector,
        sample_identical_data: pd.DataFrame,
    ) -> None:
        """PSI should work with custom bin count."""
        psi_default = drift_detector.calculate_psi("pace", sample_identical_data)
        psi_custom = drift_detector.calculate_psi(
            "pace",
            sample_identical_data,
            n_bins=5,
        )

        assert isinstance(psi_custom, float)
        # Different bin counts may give different values
        assert abs(psi_default - psi_custom) < 0.5


class TestCheckDrift:
    """Tests for check_drift aggregation method."""

    def test_check_drift_no_drift_identical_data(
        self,
        drift_detector: DriftDetector,
        sample_identical_data: pd.DataFrame,
    ) -> None:
        """check_drift should report no drift for identical distributions."""
        result = drift_detector.check_drift(sample_identical_data)

        assert isinstance(result, DriftCheckResult)
        assert result.has_drift is False
        assert len(result.features_drifted) == 0

    def test_check_drift_detects_drift(
        self,
        drift_detector: DriftDetector,
        sample_drifted_data: pd.DataFrame,
    ) -> None:
        """check_drift should detect drift in shifted features."""
        result = drift_detector.check_drift(sample_drifted_data)

        assert isinstance(result, DriftCheckResult)
        assert result.has_drift is True
        assert len(result.features_drifted) > 0
        # Pace and fg3a_rate were significantly shifted
        assert "pace" in result.features_drifted or "fg3a_rate" in result.features_drifted

    def test_check_drift_returns_details_per_feature(
        self,
        drift_detector: DriftDetector,
        sample_identical_data: pd.DataFrame,
    ) -> None:
        """check_drift should return details for each monitored feature."""
        result = drift_detector.check_drift(sample_identical_data)

        # Should have details for features that exist in both datasets
        assert len(result.details) > 0
        for feature, detail in result.details.items():
            assert feature in MONITORED_FEATURES
            assert hasattr(detail, "ks_stat")
            assert hasattr(detail, "p_value")
            assert hasattr(detail, "psi")
            assert hasattr(detail, "is_drifted")

    def test_check_drift_handles_missing_features(
        self,
        sample_reference_data: pd.DataFrame,
    ) -> None:
        """check_drift should skip features not in both datasets."""
        # Reference has all features, recent missing some
        partial_recent = pd.DataFrame({
            "pace": np.random.normal(100, 5, 50),
            # Missing other features
        })

        detector = DriftDetector(sample_reference_data)
        result = detector.check_drift(partial_recent)

        # Should still work, checking only available features
        assert isinstance(result, DriftCheckResult)

    def test_check_drift_to_dict(
        self,
        drift_detector: DriftDetector,
        sample_identical_data: pd.DataFrame,
    ) -> None:
        """check_drift result should be convertible to dict."""
        result = drift_detector.check_drift(sample_identical_data)
        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "has_drift" in result_dict
        assert "features_drifted" in result_dict
        assert "details" in result_dict

    def test_check_drift_empty_recent_data(
        self,
        drift_detector: DriftDetector,
    ) -> None:
        """check_drift should handle empty recent DataFrame gracefully."""
        empty_df = pd.DataFrame()
        result = drift_detector.check_drift(empty_df)

        # Should return no drift with empty details when no features available
        assert isinstance(result, DriftCheckResult)
        assert result.has_drift is False
        assert len(result.features_drifted) == 0

    def test_check_drift_empty_feature_columns(
        self,
        drift_detector: DriftDetector,
    ) -> None:
        """check_drift should handle DataFrame with no matching features."""
        unrelated_df = pd.DataFrame({
            "unrelated_col": [1, 2, 3, 4, 5] * 10,
            "another_col": [10, 20, 30, 40, 50] * 10,
        })
        result = drift_detector.check_drift(unrelated_df)

        # Should return no drift with empty details
        assert isinstance(result, DriftCheckResult)
        assert result.has_drift is False
        assert len(result.details) == 0


class TestEmptyDataFrameHandling:
    """Tests for empty DataFrame edge cases."""

    def test_drift_detector_rejects_empty_reference(self) -> None:
        """DriftDetector should reject empty reference DataFrame."""
        empty_df = pd.DataFrame()
        with pytest.raises(InsufficientDataError, match="at least 10 samples"):
            DriftDetector(empty_df)

    def test_drift_detector_rejects_insufficient_reference(self) -> None:
        """DriftDetector should reject reference with insufficient rows."""
        small_df = pd.DataFrame({"pace": list(range(5))})
        with pytest.raises(InsufficientDataError, match="at least 10 samples"):
            DriftDetector(small_df)

    def test_ks_test_with_empty_recent_data(
        self,
        sample_reference_data: pd.DataFrame,
    ) -> None:
        """KS test should handle empty recent data appropriately."""
        detector = DriftDetector(sample_reference_data)
        empty_df = pd.DataFrame()

        with pytest.raises(InsufficientDataError, match="not found in recent"):
            detector.ks_test("pace", empty_df)

    def test_psi_with_empty_recent_data(
        self,
        sample_reference_data: pd.DataFrame,
    ) -> None:
        """PSI calculation should handle empty recent data appropriately."""
        detector = DriftDetector(sample_reference_data)
        empty_df = pd.DataFrame()

        with pytest.raises(InsufficientDataError, match="not found in recent"):
            detector.calculate_psi("pace", empty_df)


# =============================================================================
# ConceptDriftDetector Tests
# =============================================================================


class TestConceptDriftDetector:
    """Tests for ConceptDriftDetector class."""

    def test_init_with_defaults(self) -> None:
        """ConceptDriftDetector should initialize with default thresholds."""
        detector = ConceptDriftDetector()
        assert detector.accuracy_threshold == 0.48
        assert detector.calibration_threshold == 0.1

    def test_init_with_custom_thresholds(self) -> None:
        """ConceptDriftDetector should accept custom thresholds."""
        detector = ConceptDriftDetector(
            accuracy_threshold=0.52,
            calibration_threshold=0.15,
        )
        assert detector.accuracy_threshold == 0.52
        assert detector.calibration_threshold == 0.15

    def test_init_rejects_invalid_accuracy_threshold(self) -> None:
        """ConceptDriftDetector should reject invalid accuracy threshold."""
        with pytest.raises(ValueError, match="accuracy_threshold must be in"):
            ConceptDriftDetector(accuracy_threshold=1.5)

        with pytest.raises(ValueError, match="accuracy_threshold must be in"):
            ConceptDriftDetector(accuracy_threshold=-0.1)

    def test_init_rejects_negative_calibration_threshold(self) -> None:
        """ConceptDriftDetector should reject negative calibration threshold."""
        with pytest.raises(ValueError, match="calibration_threshold must be"):
            ConceptDriftDetector(calibration_threshold=-0.1)


class TestPredictionDrift:
    """Tests for check_prediction_drift method."""

    def test_accuracy_calculation_known_values(self) -> None:
        """Accuracy should match expected formula for known predictions."""
        detector = ConceptDriftDetector(accuracy_threshold=0.50)

        # 60% accuracy: 6 correct predictions out of 10
        # Pred >= 0.5 means predict win (1), pred < 0.5 means predict loss (0)
        # We want some wrong predictions mixed in
        predictions = [0.7, 0.8, 0.6, 0.4, 0.3, 0.2, 0.55, 0.45, 0.9, 0.1]
        actuals = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0]  # All correct predictions

        result = detector.check_prediction_drift(predictions, actuals)

        # Predictions >= 0.5 are win predictions (1), < 0.5 are loss predictions (0)
        # 0.7>=0.5->1 vs 1 = correct
        # 0.8>=0.5->1 vs 1 = correct
        # 0.6>=0.5->1 vs 1 = correct
        # 0.4<0.5->0 vs 0 = correct
        # 0.3<0.5->0 vs 0 = correct
        # 0.2<0.5->0 vs 0 = correct
        # 0.55>=0.5->1 vs 1 = correct
        # 0.45<0.5->0 vs 0 = correct
        # 0.9>=0.5->1 vs 1 = correct
        # 0.1<0.5->0 vs 0 = correct
        # = 10 correct out of 10 = 100%
        assert result.recent_accuracy == 1.0
        assert result.accuracy_degraded == False  # 100% > 50% threshold

    def test_brier_score_calculation(self) -> None:
        """Brier score should match expected formula."""
        detector = ConceptDriftDetector()

        # Perfect predictions: Brier = 0 (need 10+ samples)
        perfect_predictions = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        perfect_actuals = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

        result = detector.check_prediction_drift(perfect_predictions, perfect_actuals)
        assert abs(result.recent_brier_score) < 0.001

        # Worst predictions: Brier = 1 (need 10+ samples)
        worst_predictions = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        worst_actuals = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

        result = detector.check_prediction_drift(worst_predictions, worst_actuals)
        assert abs(result.recent_brier_score - 1.0) < 0.001

    def test_degradation_flags_at_threshold_boundaries(self) -> None:
        """Degradation flags should be set correctly at thresholds."""
        detector = ConceptDriftDetector(
            accuracy_threshold=0.50,
            calibration_threshold=0.10,
        )

        # 0% accuracy (all wrong) - should flag degradation
        # Predict win (>= 0.5) but actual is loss (0), predict loss (< 0.5) but actual is win (1)
        predictions = [0.6] * 5 + [0.4] * 5  # 5 predict win, 5 predict loss
        actuals = [0] * 5 + [1] * 5  # First 5 wrong (pred 1, actual 0), last 5 wrong (pred 0, actual 1)

        result = detector.check_prediction_drift(predictions, actuals)
        assert result.recent_accuracy == 0.0  # All predictions wrong
        assert result.accuracy_degraded == True

        # Good accuracy (above threshold) - should not flag
        good_predictions = [0.6] * 10
        good_actuals = [1] * 10  # All correct (pred >= 0.5 means predict 1, actual 1)

        result = detector.check_prediction_drift(good_predictions, good_actuals)
        assert result.accuracy_degraded == False

    def test_insufficient_data_raises_error(self) -> None:
        """check_prediction_drift should raise error with < 10 predictions."""
        detector = ConceptDriftDetector()

        with pytest.raises(InsufficientDataError, match="at least 10"):
            detector.check_prediction_drift([0.5] * 5, [1] * 5)

    def test_length_mismatch_raises_error(self) -> None:
        """check_prediction_drift should raise error for length mismatch."""
        detector = ConceptDriftDetector()

        with pytest.raises(ValueError, match="Length mismatch"):
            detector.check_prediction_drift([0.5] * 10, [1] * 12)

    def test_window_size_uses_recent(self) -> None:
        """check_prediction_drift should use only recent window_size samples."""
        detector = ConceptDriftDetector(accuracy_threshold=0.50)

        # First 50 are all wrong, last 50 are all correct
        predictions = [0.6] * 50 + [0.6] * 50
        actuals = [0] * 50 + [1] * 50

        # With window_size=50, should only see 100% accuracy from last 50
        result = detector.check_prediction_drift(predictions, actuals, window_size=50)
        assert result.recent_accuracy == 1.0

    def test_result_to_dict(self) -> None:
        """ConceptDriftResult should be convertible to dict."""
        detector = ConceptDriftDetector()
        predictions = [0.55] * 15
        actuals = [1] * 15

        result = detector.check_prediction_drift(predictions, actuals)
        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "accuracy_degraded" in result_dict
        assert "calibration_degraded" in result_dict
        assert "recent_accuracy" in result_dict
        assert "recent_brier_score" in result_dict
