"""Integration tests for the Phase 6 monitoring pipeline.

Tests the complete flow: drift detection -> trigger evaluation -> version management.
Validates cross-module interactions for the monitoring system.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
import torch

from nba_model.monitor.drift import (
    ConceptDriftDetector,
    DriftDetector,
    InsufficientDataError,
    MONITORED_FEATURES,
)
from nba_model.monitor.triggers import RetrainingTrigger, TriggerContext
from nba_model.monitor.versioning import (
    ModelVersionManager,
    STATUS_ACTIVE,
    STATUS_PROMOTED,
)
from nba_model.types import Bet

if TYPE_CHECKING:
    pass


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def reference_feature_data() -> pd.DataFrame:
    """Generate reference feature data simulating training period."""
    np.random.seed(42)
    n_samples = 200

    return pd.DataFrame({
        "pace": np.random.normal(100, 5, n_samples),
        "offensive_rating": np.random.normal(110, 8, n_samples),
        "fg3a_rate": np.random.normal(0.35, 0.05, n_samples),
        "rest_days": np.random.randint(1, 5, n_samples).astype(float),
        "travel_distance": np.random.exponential(500, n_samples),
        "rapm_mean": np.random.normal(0, 2, n_samples),
    })


@pytest.fixture
def recent_no_drift_data() -> pd.DataFrame:
    """Generate recent data with no drift (same distribution as reference)."""
    np.random.seed(43)  # Different seed but same distribution
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
def recent_drifted_data() -> pd.DataFrame:
    """Generate recent data with significant drift in multiple features."""
    np.random.seed(44)
    n_samples = 100

    return pd.DataFrame({
        # Pace shifted significantly higher
        "pace": np.random.normal(115, 5, n_samples),
        # Offensive rating shifted
        "offensive_rating": np.random.normal(125, 8, n_samples),
        # 3PT rate increased
        "fg3a_rate": np.random.normal(0.50, 0.05, n_samples),
        # Rest days same
        "rest_days": np.random.randint(1, 5, n_samples).astype(float),
        # Travel same
        "travel_distance": np.random.exponential(500, n_samples),
        # RAPM shifted
        "rapm_mean": np.random.normal(3, 2, n_samples),
    })


@pytest.fixture
def synthetic_bets_good() -> list[Bet]:
    """Generate synthetic bet history with good performance."""
    np.random.seed(42)
    n_bets = 100

    bets = []
    for i in range(n_bets):
        model_prob = 0.55 + np.random.random() * 0.05
        market_odds = 1.90 + np.random.random() * 0.1
        won = np.random.random() < 0.56  # 56% win rate
        result = "win" if won else "loss"
        bet_amount = 100.0
        profit = bet_amount * (market_odds - 1) if won else -bet_amount

        bet = Bet(
            game_id=f"GAME{i:04d}",
            timestamp=datetime(2023, 1, 1) + timedelta(days=i),
            bet_type="moneyline",
            side="home",
            model_prob=model_prob,
            market_odds=market_odds,
            market_prob=1 / market_odds,
            edge=model_prob - (1 / market_odds),
            kelly_fraction=0.0,
            bet_amount=bet_amount,
            result=result,
            profit=profit,
        )
        bets.append(bet)

    return bets


@pytest.fixture
def synthetic_bets_bad() -> list[Bet]:
    """Generate synthetic bet history with bad performance (triggers retrain)."""
    np.random.seed(42)
    n_bets = 100

    bets = []
    for i in range(n_bets):
        model_prob = 0.52 + np.random.random() * 0.03
        market_odds = 1.90 + np.random.random() * 0.1
        won = np.random.random() < 0.42  # 42% win rate (below threshold)
        result = "win" if won else "loss"
        bet_amount = 100.0
        profit = bet_amount * (market_odds - 1) if won else -bet_amount

        bet = Bet(
            game_id=f"GAME{i:04d}",
            timestamp=datetime(2023, 1, 1) + timedelta(days=i),
            bet_type="moneyline",
            side="home",
            model_prob=model_prob,
            market_odds=market_odds,
            market_prob=1 / market_odds,
            edge=model_prob - (1 / market_odds),
            kelly_fraction=0.0,
            bet_amount=bet_amount,
            result=result,
            profit=profit,
        )
        bets.append(bet)

    return bets


@pytest.fixture
def temp_model_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for model storage."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir


# =============================================================================
# Drift Pipeline Integration Tests
# =============================================================================


@pytest.mark.integration
class TestDriftPipelineIntegration:
    """Integration tests for the drift detection pipeline."""

    def test_complete_drift_check_no_drift(
        self,
        reference_feature_data: pd.DataFrame,
        recent_no_drift_data: pd.DataFrame,
    ) -> None:
        """Full drift check should detect no drift for similar distributions."""
        detector = DriftDetector(
            reference_data=reference_feature_data,
            p_value_threshold=0.05,
            psi_threshold=0.2,
        )

        result = detector.check_drift(recent_no_drift_data)

        assert result.has_drift is False
        assert len(result.features_drifted) == 0
        # Should have checked available features
        assert len(result.details) > 0

    def test_complete_drift_check_with_drift(
        self,
        reference_feature_data: pd.DataFrame,
        recent_drifted_data: pd.DataFrame,
    ) -> None:
        """Full drift check should detect drift for shifted distributions."""
        detector = DriftDetector(
            reference_data=reference_feature_data,
            p_value_threshold=0.05,
            psi_threshold=0.2,
        )

        result = detector.check_drift(recent_drifted_data)

        assert result.has_drift is True
        assert len(result.features_drifted) > 0
        # Pace and fg3a_rate should be detected as drifted
        drifted_set = set(result.features_drifted)
        assert "pace" in drifted_set or "fg3a_rate" in drifted_set

    def test_covariate_and_concept_drift_together(
        self,
        reference_feature_data: pd.DataFrame,
        recent_drifted_data: pd.DataFrame,
    ) -> None:
        """Both covariate and concept drift can be checked in a single pipeline."""
        # Covariate drift check
        covariate_detector = DriftDetector(reference_data=reference_feature_data)
        covariate_result = covariate_detector.check_drift(recent_drifted_data)

        # Concept drift check (simulated predictions)
        concept_detector = ConceptDriftDetector(
            accuracy_threshold=0.48,
            calibration_threshold=0.1,
        )
        predictions = [0.55 + np.random.random() * 0.1 for _ in range(50)]
        actuals = [1 if np.random.random() < 0.42 else 0 for _ in range(50)]

        concept_result = concept_detector.check_prediction_drift(
            predictions=predictions,
            actuals=actuals,
        )

        # Both should run and return valid results
        assert isinstance(covariate_result.has_drift, bool)
        assert isinstance(concept_result.accuracy_degraded, bool)


# =============================================================================
# Trigger Pipeline Integration Tests
# =============================================================================


@pytest.mark.integration
class TestTriggerPipelineIntegration:
    """Integration tests for the retraining trigger pipeline."""

    def test_scheduled_trigger_activates_after_interval(self) -> None:
        """Scheduled trigger should activate when interval exceeded."""
        trigger = RetrainingTrigger(
            scheduled_interval_days=7,
            min_new_games=50,
        )

        # Last training 10 days ago
        context = TriggerContext(
            last_train_date=date.today() - timedelta(days=10),
            games_since_training=30,
        )

        result = trigger.evaluate_all_triggers(context)

        assert result.should_retrain is True
        assert result.trigger_details["scheduled"] is True
        assert result.reason == "scheduled"
        assert result.priority == "medium"

    def test_drift_trigger_activates_on_detected_drift(
        self,
        reference_feature_data: pd.DataFrame,
        recent_drifted_data: pd.DataFrame,
    ) -> None:
        """Drift trigger should activate when drift is detected."""
        trigger = RetrainingTrigger(
            scheduled_interval_days=30,  # Won't trigger
            min_new_games=1000,  # Won't trigger
        )

        drift_detector = DriftDetector(reference_data=reference_feature_data)

        context = TriggerContext(
            last_train_date=date.today() - timedelta(days=5),  # Within interval
            drift_detector=drift_detector,
            recent_data=recent_drifted_data,
            games_since_training=10,  # Below threshold
        )

        result = trigger.evaluate_all_triggers(context)

        assert result.should_retrain is True
        assert result.trigger_details["drift"] is True
        assert result.reason == "drift_detected"
        assert result.priority == "high"

    def test_performance_trigger_activates_on_poor_roi(
        self,
        synthetic_bets_bad: list[Bet],
    ) -> None:
        """Performance trigger should activate on poor betting performance."""
        trigger = RetrainingTrigger(
            scheduled_interval_days=30,
            min_new_games=1000,
            roi_threshold=-0.05,
            accuracy_threshold=0.48,
        )

        context = TriggerContext(
            last_train_date=date.today() - timedelta(days=5),
            recent_bets=synthetic_bets_bad,
            games_since_training=10,
        )

        result = trigger.evaluate_all_triggers(context)

        assert result.should_retrain is True
        assert result.trigger_details["performance"] is True
        assert result.reason == "performance_degraded"
        assert result.priority == "high"

    def test_data_trigger_activates_on_sufficient_games(self) -> None:
        """Data trigger should activate when enough new games available."""
        trigger = RetrainingTrigger(
            scheduled_interval_days=30,
            min_new_games=50,
        )

        context = TriggerContext(
            last_train_date=date.today() - timedelta(days=5),
            games_since_training=75,
        )

        result = trigger.evaluate_all_triggers(context)

        assert result.should_retrain is True
        assert result.trigger_details["data"] is True
        assert result.reason == "new_data"
        assert result.priority == "low"

    def test_no_triggers_when_all_conditions_pass(
        self,
        reference_feature_data: pd.DataFrame,
        recent_no_drift_data: pd.DataFrame,
        synthetic_bets_good: list[Bet],
    ) -> None:
        """No triggers should activate when model is performing well."""
        trigger = RetrainingTrigger(
            scheduled_interval_days=7,
            min_new_games=50,
            roi_threshold=-0.05,
            accuracy_threshold=0.48,
        )

        drift_detector = DriftDetector(reference_data=reference_feature_data)

        context = TriggerContext(
            last_train_date=date.today() - timedelta(days=3),  # Recent training
            drift_detector=drift_detector,
            recent_data=recent_no_drift_data,  # No drift
            recent_bets=synthetic_bets_good,  # Good performance
            games_since_training=20,  # Below threshold
        )

        result = trigger.evaluate_all_triggers(context)

        assert result.should_retrain is False
        assert all(not v for v in result.trigger_details.values())

    def test_dict_context_input(self) -> None:
        """Trigger evaluation should accept dict context as well as TriggerContext."""
        trigger = RetrainingTrigger(scheduled_interval_days=7)

        context_dict = {
            "last_train_date": date.today() - timedelta(days=10),
            "games_since_training": 100,
        }

        result = trigger.evaluate_all_triggers(context_dict)

        assert result.should_retrain is True
        assert result.trigger_details["scheduled"] is True


# =============================================================================
# Version Lifecycle Integration Tests
# =============================================================================


@pytest.mark.integration
class TestVersionLifecycleIntegration:
    """Integration tests for version management lifecycle."""

    def test_create_and_list_versions(self, temp_model_dir: Path) -> None:
        """Version creation should be reflected in version listing."""
        manager = ModelVersionManager(base_dir=temp_model_dir)

        # Create a simple model
        models = {
            "transformer": torch.nn.Linear(10, 5),
            "gnn": torch.nn.Linear(5, 3),
            "fusion": torch.nn.Linear(3, 1),
        }

        version = manager.create_version(
            models=models,
            config={"d_model": 128, "learning_rate": 1e-4},
            metrics={"accuracy": 0.58, "brier_score": 0.23},
        )

        assert version == "1.0.0"

        # List versions
        versions = manager.list_versions()
        assert len(versions) == 1
        assert versions[0]["version"] == "1.0.0"
        assert versions[0]["status"] == STATUS_ACTIVE

    def test_version_comparison_uses_metrics(self, temp_model_dir: Path) -> None:
        """Version comparison should use validation metrics."""
        manager = ModelVersionManager(base_dir=temp_model_dir)

        models = {
            "transformer": torch.nn.Linear(10, 5),
            "gnn": torch.nn.Linear(5, 3),
            "fusion": torch.nn.Linear(3, 1),
        }

        # Create version 1.0.0
        manager.create_version(
            models=models,
            config={"d_model": 128},
            metrics={"accuracy": 0.55, "brier_score": 0.25},
        )

        # Create version 1.1.0 with better accuracy
        manager.create_version(
            models=models,
            config={"d_model": 128},
            metrics={"accuracy": 0.60, "brier_score": 0.22},
            parent_version="1.0.0",
            bump="minor",
        )

        # Compare versions
        comparison = manager.compare_versions("1.0.0", "1.1.0")

        assert comparison.winner == "1.1.0"
        assert comparison.improvement["accuracy"] > 0  # 1.1.0 has higher accuracy

    def test_promote_and_rollback_workflow(self, temp_model_dir: Path) -> None:
        """Promotion and rollback should update version statuses correctly."""
        manager = ModelVersionManager(base_dir=temp_model_dir)

        models = {
            "transformer": torch.nn.Linear(10, 5),
            "gnn": torch.nn.Linear(5, 3),
            "fusion": torch.nn.Linear(3, 1),
        }

        # Create two versions
        manager.create_version(
            models=models,
            config={"d_model": 128},
            metrics={"accuracy": 0.55},
        )
        manager.create_version(
            models=models,
            config={"d_model": 256},
            metrics={"accuracy": 0.60},
            parent_version="1.0.0",
            bump="minor",
        )

        # Promote 1.1.0
        manager.promote_version("1.1.0")

        versions = manager.list_versions()
        v110 = next(v for v in versions if v["version"] == "1.1.0")
        assert v110["status"] == STATUS_PROMOTED

        # Rollback to 1.0.0
        manager.rollback("1.0.0")

        # 1.0.0 should now be promoted
        versions = manager.list_versions()
        v100 = next(v for v in versions if v["version"] == "1.0.0")
        assert v100["status"] == STATUS_PROMOTED

    def test_lineage_tracking(self, temp_model_dir: Path) -> None:
        """Version lineage should be correctly tracked."""
        manager = ModelVersionManager(base_dir=temp_model_dir)

        models = {
            "transformer": torch.nn.Linear(10, 5),
            "gnn": torch.nn.Linear(5, 3),
            "fusion": torch.nn.Linear(3, 1),
        }

        # Create chain: 1.0.0 -> 1.1.0 -> 1.2.0
        manager.create_version(
            models=models,
            config={},
            metrics={"accuracy": 0.55},
        )
        manager.create_version(
            models=models,
            config={},
            metrics={"accuracy": 0.57},
            parent_version="1.0.0",
            bump="minor",
        )
        manager.create_version(
            models=models,
            config={},
            metrics={"accuracy": 0.60},
            parent_version="1.1.0",
            bump="minor",
        )

        # Get lineage for 1.2.0
        lineage = manager.get_lineage("1.2.0")

        assert lineage == ["1.0.0", "1.1.0", "1.2.0"]

    def test_versions_sorted_by_creation_date(self, temp_model_dir: Path) -> None:
        """Version listing should be sorted by creation date descending."""
        import time

        manager = ModelVersionManager(base_dir=temp_model_dir)

        models = {
            "transformer": torch.nn.Linear(10, 5),
            "gnn": torch.nn.Linear(5, 3),
            "fusion": torch.nn.Linear(3, 1),
        }

        # Create versions with small delays to ensure different timestamps
        manager.create_version(
            models=models,
            config={},
            metrics={"accuracy": 0.55},
        )
        time.sleep(0.1)  # Small delay
        manager.create_version(
            models=models,
            config={},
            metrics={"accuracy": 0.57},
            parent_version="1.0.0",
            bump="minor",
        )
        time.sleep(0.1)
        manager.create_version(
            models=models,
            config={},
            metrics={"accuracy": 0.60},
            parent_version="1.1.0",
            bump="minor",
        )

        versions = manager.list_versions()

        # Should be sorted newest first
        assert versions[0]["version"] == "1.2.0"
        assert versions[1]["version"] == "1.1.0"
        assert versions[2]["version"] == "1.0.0"


# =============================================================================
# End-to-End Monitor Pipeline Test
# =============================================================================


@pytest.mark.integration
class TestEndToEndMonitorPipeline:
    """End-to-end integration test for the complete monitoring pipeline."""

    def test_complete_monitoring_cycle(
        self,
        temp_model_dir: Path,
        reference_feature_data: pd.DataFrame,
        recent_drifted_data: pd.DataFrame,
    ) -> None:
        """Test complete monitoring cycle: drift detection -> trigger -> version."""
        # Step 1: Initialize components
        manager = ModelVersionManager(base_dir=temp_model_dir)

        models = {
            "transformer": torch.nn.Linear(10, 5),
            "gnn": torch.nn.Linear(5, 3),
            "fusion": torch.nn.Linear(3, 1),
        }

        # Create initial version
        initial_version = manager.create_version(
            models=models,
            config={"d_model": 128},
            metrics={"accuracy": 0.55, "brier_score": 0.24},
        )

        # Step 2: Detect drift
        drift_detector = DriftDetector(reference_data=reference_feature_data)
        drift_result = drift_detector.check_drift(recent_drifted_data)

        assert drift_result.has_drift is True

        # Step 3: Evaluate triggers
        trigger = RetrainingTrigger(
            scheduled_interval_days=30,
            min_new_games=1000,
        )

        context = TriggerContext(
            last_train_date=date.today() - timedelta(days=5),
            drift_detector=drift_detector,
            recent_data=recent_drifted_data,
            games_since_training=10,
        )

        trigger_result = trigger.evaluate_all_triggers(context)

        assert trigger_result.should_retrain is True
        assert trigger_result.reason == "drift_detected"

        # Step 4: Retrain and create new version (simulated)
        new_version = manager.create_version(
            models=models,
            config={"d_model": 256},  # New config
            metrics={"accuracy": 0.60, "brier_score": 0.20},
            parent_version=initial_version,
            bump="minor",
        )

        assert new_version == "1.1.0"

        # Step 5: Compare versions
        comparison = manager.compare_versions(initial_version, new_version)
        assert comparison.winner == new_version

        # Step 6: Promote new version
        manager.promote_version(new_version)

        versions = manager.list_versions()
        promoted = next(v for v in versions if v["version"] == new_version)
        assert promoted["status"] == STATUS_PROMOTED
