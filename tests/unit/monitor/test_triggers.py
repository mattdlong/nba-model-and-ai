"""Unit tests for monitor.triggers module.

Tests retraining trigger evaluation including scheduled, drift,
performance, and data triggers.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from nba_model.monitor.drift import DriftDetector
from nba_model.monitor.triggers import (
    RetrainingTrigger,
    TriggerContext,
    TriggerResult,
)
from nba_model.types import Bet


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def trigger() -> RetrainingTrigger:
    """Create a RetrainingTrigger with default settings."""
    return RetrainingTrigger(
        scheduled_interval_days=7,
        min_new_games=50,
        roi_threshold=-0.05,
        accuracy_threshold=0.48,
    )


@pytest.fixture
def reference_data() -> pd.DataFrame:
    """Create reference data for drift detector."""
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
def drift_detector(reference_data: pd.DataFrame) -> DriftDetector:
    """Create a drift detector with reference data."""
    return DriftDetector(reference_data)


@pytest.fixture
def sample_winning_bets() -> list[Bet]:
    """Create sample bets with good performance."""
    bets = []
    for i in range(60):
        # 55% win rate, positive ROI
        is_win = i % 100 < 55
        bets.append(
            Bet(
                game_id=f"00220000{i:02d}",
                timestamp=datetime.now() - timedelta(days=60 - i),
                bet_type="moneyline",
                side="home",
                model_prob=0.55,
                market_odds=1.91,
                market_prob=0.52,
                edge=0.03,
                kelly_fraction=0.02,
                bet_amount=100.0,
                result="win" if is_win else "loss",
                profit=91.0 if is_win else -100.0,
            )
        )
    return bets


@pytest.fixture
def sample_losing_bets() -> list[Bet]:
    """Create sample bets with poor performance (40% win rate, negative ROI)."""
    bets = []
    for i in range(60):
        # 40% win rate: first 24 wins, rest losses (24/60 = 40%)
        is_win = i < 24
        bets.append(
            Bet(
                game_id=f"00220000{i:02d}",
                timestamp=datetime.now() - timedelta(days=60 - i),
                bet_type="moneyline",
                side="home",
                model_prob=0.55,
                market_odds=1.91,
                market_prob=0.52,
                edge=0.03,
                kelly_fraction=0.02,
                bet_amount=100.0,
                result="win" if is_win else "loss",
                profit=91.0 if is_win else -100.0,
            )
        )
    return bets


# =============================================================================
# RetrainingTrigger Initialization Tests
# =============================================================================


class TestRetrainingTriggerInit:
    """Tests for RetrainingTrigger initialization."""

    def test_init_with_defaults(self) -> None:
        """RetrainingTrigger should initialize with default values."""
        trigger = RetrainingTrigger()
        assert trigger.scheduled_interval_days == 7
        assert trigger.min_new_games == 50
        assert trigger.roi_threshold == -0.05
        assert trigger.accuracy_threshold == 0.48

    def test_init_with_custom_values(self) -> None:
        """RetrainingTrigger should accept custom values."""
        trigger = RetrainingTrigger(
            scheduled_interval_days=14,
            min_new_games=100,
            roi_threshold=-0.10,
            accuracy_threshold=0.50,
        )
        assert trigger.scheduled_interval_days == 14
        assert trigger.min_new_games == 100
        assert trigger.roi_threshold == -0.10
        assert trigger.accuracy_threshold == 0.50

    def test_init_rejects_invalid_interval(self) -> None:
        """RetrainingTrigger should reject interval < 1."""
        with pytest.raises(ValueError, match="scheduled_interval_days must be >= 1"):
            RetrainingTrigger(scheduled_interval_days=0)

    def test_init_rejects_invalid_min_games(self) -> None:
        """RetrainingTrigger should reject min_games < 1."""
        with pytest.raises(ValueError, match="min_new_games must be >= 1"):
            RetrainingTrigger(min_new_games=0)

    def test_init_rejects_invalid_accuracy_threshold(self) -> None:
        """RetrainingTrigger should reject accuracy threshold outside (0, 1)."""
        with pytest.raises(ValueError, match="accuracy_threshold must be in"):
            RetrainingTrigger(accuracy_threshold=1.5)


# =============================================================================
# Scheduled Trigger Tests
# =============================================================================


class TestScheduledTrigger:
    """Tests for scheduled retraining trigger."""

    def test_activates_at_exactly_n_days(self, trigger: RetrainingTrigger) -> None:
        """Scheduled trigger should activate at exactly interval days."""
        # At exactly 7 days
        last_train = date.today() - timedelta(days=7)
        assert trigger.check_scheduled_trigger(last_train) is True

    def test_activates_after_n_days(self, trigger: RetrainingTrigger) -> None:
        """Scheduled trigger should activate after interval days."""
        # More than 7 days
        last_train = date.today() - timedelta(days=10)
        assert trigger.check_scheduled_trigger(last_train) is True

    def test_does_not_activate_before_n_days(self, trigger: RetrainingTrigger) -> None:
        """Scheduled trigger should not activate before interval days."""
        # Less than 7 days
        last_train = date.today() - timedelta(days=5)
        assert trigger.check_scheduled_trigger(last_train) is False

    def test_does_not_activate_same_day(self, trigger: RetrainingTrigger) -> None:
        """Scheduled trigger should not activate on same day as training."""
        last_train = date.today()
        assert trigger.check_scheduled_trigger(last_train) is False


# =============================================================================
# Drift Trigger Tests
# =============================================================================


class TestDriftTrigger:
    """Tests for drift-based retraining trigger."""

    def test_passes_through_drift_detector_result(
        self,
        trigger: RetrainingTrigger,
        drift_detector: DriftDetector,
        reference_data: pd.DataFrame,
    ) -> None:
        """Drift trigger should return True when detector finds drift."""
        # Create drifted data
        np.random.seed(43)
        n_samples = 100
        drifted_data = pd.DataFrame({
            "pace": np.random.normal(120, 5, n_samples),  # Significant shift
            "offensive_rating": np.random.normal(130, 8, n_samples),
            "fg3a_rate": np.random.normal(0.50, 0.05, n_samples),
            "rest_days": np.random.randint(1, 5, n_samples).astype(float),
            "travel_distance": np.random.exponential(500, n_samples),
            "rapm_mean": np.random.normal(5, 2, n_samples),
        })

        result = trigger.check_drift_trigger(drift_detector, drifted_data)
        assert result is True

    def test_no_drift_returns_false(
        self,
        trigger: RetrainingTrigger,
        drift_detector: DriftDetector,
        reference_data: pd.DataFrame,
    ) -> None:
        """Drift trigger should return False when no drift detected."""
        # Use same seed as reference for identical distribution
        np.random.seed(42)
        n_samples = 100
        identical_data = pd.DataFrame({
            "pace": np.random.normal(100, 5, n_samples),
            "offensive_rating": np.random.normal(110, 8, n_samples),
            "fg3a_rate": np.random.normal(0.35, 0.05, n_samples),
            "rest_days": np.random.randint(1, 5, n_samples).astype(float),
            "travel_distance": np.random.exponential(500, n_samples),
            "rapm_mean": np.random.normal(0, 2, n_samples),
        })

        result = trigger.check_drift_trigger(drift_detector, identical_data)
        assert result is False


# =============================================================================
# Performance Trigger Tests
# =============================================================================


class TestPerformanceTrigger:
    """Tests for performance-based retraining trigger."""

    def test_activates_when_roi_below_threshold(
        self,
        trigger: RetrainingTrigger,
        sample_losing_bets: list[Bet],
    ) -> None:
        """Performance trigger should activate when ROI < threshold."""
        result = trigger.check_performance_trigger(sample_losing_bets)
        assert result is True

    def test_activates_when_accuracy_below_threshold(
        self,
        sample_losing_bets: list[Bet],
    ) -> None:
        """Performance trigger should activate when accuracy < threshold."""
        # Use higher accuracy threshold to ensure trigger
        trigger = RetrainingTrigger(accuracy_threshold=0.50)
        result = trigger.check_performance_trigger(sample_losing_bets)
        assert result is True

    def test_does_not_activate_with_good_performance(
        self,
        trigger: RetrainingTrigger,
        sample_winning_bets: list[Bet],
    ) -> None:
        """Performance trigger should not activate with good performance."""
        result = trigger.check_performance_trigger(sample_winning_bets)
        assert result is False

    def test_requires_minimum_bets(self, trigger: RetrainingTrigger) -> None:
        """Performance trigger should require minimum 50 bets."""
        # Only 30 bets
        few_bets = [
            Bet(
                game_id=f"00220000{i:02d}",
                timestamp=datetime.now(),
                bet_type="moneyline",
                side="home",
                model_prob=0.55,
                market_odds=1.91,
                market_prob=0.52,
                edge=0.03,
                kelly_fraction=0.02,
                bet_amount=100.0,
                result="loss",
                profit=-100.0,
            )
            for i in range(30)
        ]

        result = trigger.check_performance_trigger(few_bets)
        assert result is False  # Not enough bets to evaluate


# =============================================================================
# Data Trigger Tests
# =============================================================================


class TestDataTrigger:
    """Tests for data-based retraining trigger."""

    def test_activates_at_minimum_game_count(
        self,
        trigger: RetrainingTrigger,
    ) -> None:
        """Data trigger should activate at exactly min_new_games."""
        assert trigger.check_data_trigger(50) is True

    def test_activates_above_minimum(self, trigger: RetrainingTrigger) -> None:
        """Data trigger should activate above minimum."""
        assert trigger.check_data_trigger(100) is True

    def test_does_not_activate_below_minimum(
        self,
        trigger: RetrainingTrigger,
    ) -> None:
        """Data trigger should not activate below minimum."""
        assert trigger.check_data_trigger(30) is False

    def test_does_not_activate_at_zero(self, trigger: RetrainingTrigger) -> None:
        """Data trigger should not activate with no new games."""
        assert trigger.check_data_trigger(0) is False


# =============================================================================
# Evaluate All Triggers Tests
# =============================================================================


class TestEvaluateAllTriggers:
    """Tests for evaluate_all_triggers aggregation."""

    def test_high_priority_for_drift(
        self,
        drift_detector: DriftDetector,
    ) -> None:
        """Drift trigger should result in high priority."""
        trigger = RetrainingTrigger()

        # Create drifted data
        np.random.seed(43)
        drifted_data = pd.DataFrame({
            "pace": np.random.normal(120, 5, 100),
            "offensive_rating": np.random.normal(130, 8, 100),
            "fg3a_rate": np.random.normal(0.50, 0.05, 100),
            "rest_days": np.random.randint(1, 5, 100).astype(float),
            "travel_distance": np.random.exponential(500, 100),
            "rapm_mean": np.random.normal(5, 2, 100),
        })

        context = TriggerContext(
            last_train_date=date.today(),  # Not scheduled
            drift_detector=drift_detector,
            recent_data=drifted_data,
            recent_bets=[],
            games_since_training=0,
        )

        result = trigger.evaluate_all_triggers(context)

        assert isinstance(result, TriggerResult)
        assert result.should_retrain is True
        assert result.priority == "high"
        assert result.reason == "drift_detected"

    def test_high_priority_for_performance(
        self,
        trigger: RetrainingTrigger,
        sample_losing_bets: list[Bet],
    ) -> None:
        """Performance trigger should result in high priority."""
        context = TriggerContext(
            last_train_date=date.today(),  # Not scheduled
            recent_bets=sample_losing_bets,
            games_since_training=0,
        )

        result = trigger.evaluate_all_triggers(context)

        assert result.should_retrain is True
        assert result.priority == "high"
        assert result.reason == "performance_degraded"

    def test_medium_priority_for_scheduled(self, trigger: RetrainingTrigger) -> None:
        """Scheduled trigger should result in medium priority."""
        context = TriggerContext(
            last_train_date=date.today() - timedelta(days=10),  # Past due
            games_since_training=0,
        )

        result = trigger.evaluate_all_triggers(context)

        assert result.should_retrain is True
        assert result.priority == "medium"
        assert result.reason == "scheduled"

    def test_low_priority_for_data_only(self, trigger: RetrainingTrigger) -> None:
        """Data trigger alone should result in low priority."""
        context = TriggerContext(
            last_train_date=date.today(),  # Not scheduled
            games_since_training=100,  # Enough new games
        )

        result = trigger.evaluate_all_triggers(context)

        assert result.should_retrain is True
        assert result.priority == "low"
        assert result.reason == "new_data"

    def test_no_retrain_when_no_triggers(self, trigger: RetrainingTrigger) -> None:
        """Should not recommend retraining when no triggers active."""
        context = TriggerContext(
            last_train_date=date.today(),  # Not scheduled
            games_since_training=10,  # Not enough games
        )

        result = trigger.evaluate_all_triggers(context)

        assert result.should_retrain is False
        assert result.reason == ""
        assert result.priority == "low"

    def test_multiple_triggers_highest_priority_wins(
        self,
        drift_detector: DriftDetector,
        sample_losing_bets: list[Bet],
    ) -> None:
        """When multiple triggers active, highest priority should be reported."""
        trigger = RetrainingTrigger()

        # Create drifted data
        np.random.seed(43)
        drifted_data = pd.DataFrame({
            "pace": np.random.normal(120, 5, 100),
            "offensive_rating": np.random.normal(130, 8, 100),
            "fg3a_rate": np.random.normal(0.50, 0.05, 100),
            "rest_days": np.random.randint(1, 5, 100).astype(float),
            "travel_distance": np.random.exponential(500, 100),
            "rapm_mean": np.random.normal(5, 2, 100),
        })

        # All triggers should be active
        context = TriggerContext(
            last_train_date=date.today() - timedelta(days=10),  # Scheduled
            drift_detector=drift_detector,
            recent_data=drifted_data,  # Drift
            recent_bets=sample_losing_bets,  # Performance
            games_since_training=100,  # Data
        )

        result = trigger.evaluate_all_triggers(context)

        # Should get high priority (drift or performance)
        assert result.should_retrain is True
        assert result.priority == "high"
        # Drift comes first in priority check
        assert result.reason in ("drift_detected", "performance_degraded")

    def test_accepts_dict_context(self, trigger: RetrainingTrigger) -> None:
        """evaluate_all_triggers should accept dict context."""
        context = {
            "last_train_date": date.today() - timedelta(days=10),
            "games_since_training": 0,
        }

        result = trigger.evaluate_all_triggers(context)

        assert isinstance(result, TriggerResult)
        assert result.trigger_details["scheduled"] is True

    def test_trigger_details_all_returned(self, trigger: RetrainingTrigger) -> None:
        """trigger_details should include all trigger types."""
        context = TriggerContext(
            last_train_date=date.today(),
            games_since_training=0,
        )

        result = trigger.evaluate_all_triggers(context)

        assert "scheduled" in result.trigger_details
        assert "drift" in result.trigger_details
        assert "performance" in result.trigger_details
        assert "data" in result.trigger_details

    def test_result_to_dict(self, trigger: RetrainingTrigger) -> None:
        """TriggerResult should be convertible to dict."""
        context = TriggerContext(
            last_train_date=date.today(),
            games_since_training=0,
        )

        result = trigger.evaluate_all_triggers(context)
        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "should_retrain" in result_dict
        assert "reason" in result_dict
        assert "priority" in result_dict
        assert "trigger_details" in result_dict
