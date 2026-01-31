"""Unit tests for monitor.versioning module.

Tests model version management including version creation, comparison,
promotion, rollback, and lineage tracking.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from nba_model.monitor.versioning import (
    ModelVersionManager,
    VersionComparisonResult,
    VersionMetadata,
)
from nba_model.types import ModelNotFoundError


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_model_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for model storage."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir


@pytest.fixture
def version_manager(temp_model_dir: Path) -> ModelVersionManager:
    """Create a ModelVersionManager with temp directory."""
    return ModelVersionManager(base_dir=temp_model_dir)


@pytest.fixture
def sample_model() -> nn.Module:
    """Create a simple model for testing."""
    return nn.Linear(10, 5)


@pytest.fixture
def sample_models() -> dict[str, nn.Module]:
    """Create sample model dictionary."""
    return {
        "transformer": nn.Linear(10, 5),
        "gnn": nn.Linear(10, 3),
        "fusion": nn.Linear(10, 2),
    }


@pytest.fixture
def sample_config() -> dict[str, int]:
    """Create sample hyperparameter config."""
    return {
        "d_model": 128,
        "num_heads": 4,
        "num_layers": 2,
    }


@pytest.fixture
def sample_metrics() -> dict[str, float]:
    """Create sample validation metrics."""
    return {
        "accuracy": 0.58,
        "brier_score": 0.22,
        "margin_mae": 6.5,
        "total_mae": 10.2,
    }


# =============================================================================
# VersionMetadata Tests
# =============================================================================


class TestVersionMetadata:
    """Tests for VersionMetadata dataclass."""

    def test_to_dict(self) -> None:
        """VersionMetadata should convert to dictionary."""
        meta = VersionMetadata(
            version="1.0.0",
            created_at=datetime(2024, 1, 15, 12, 0, 0),
            training_data_start=date(2023, 10, 1),
            training_data_end=date(2024, 1, 1),
            hyperparameters={"d_model": 128},
            validation_metrics={"accuracy": 0.58},
            git_commit="abc123",
            parent_version="0.9.0",
            status="active",
        )

        result = meta.to_dict()

        assert result["version"] == "1.0.0"
        assert result["created_at"] == "2024-01-15T12:00:00"
        assert result["training_data_start"] == "2023-10-01"
        assert result["training_data_end"] == "2024-01-01"
        assert result["hyperparameters"] == {"d_model": 128}
        assert result["validation_metrics"] == {"accuracy": 0.58}
        assert result["git_commit"] == "abc123"
        assert result["parent_version"] == "0.9.0"
        assert result["status"] == "active"

    def test_from_dict(self) -> None:
        """VersionMetadata should be constructable from dictionary."""
        data = {
            "version": "1.0.0",
            "created_at": "2024-01-15T12:00:00",
            "training_data_start": "2023-10-01",
            "training_data_end": "2024-01-01",
            "hyperparameters": {"d_model": 128},
            "validation_metrics": {"accuracy": 0.58},
            "git_commit": "abc123",
            "parent_version": "0.9.0",
            "status": "promoted",
        }

        meta = VersionMetadata.from_dict(data)

        assert meta.version == "1.0.0"
        assert meta.created_at == datetime(2024, 1, 15, 12, 0, 0)
        assert meta.training_data_start == date(2023, 10, 1)
        assert meta.training_data_end == date(2024, 1, 1)
        assert meta.status == "promoted"


# =============================================================================
# ModelVersionManager Tests
# =============================================================================


class TestModelVersionManager:
    """Tests for ModelVersionManager class."""

    def test_init_creates_directory(self, temp_model_dir: Path) -> None:
        """ModelVersionManager should work with existing directory."""
        manager = ModelVersionManager(base_dir=temp_model_dir)
        assert manager.base_dir == temp_model_dir
        assert temp_model_dir.exists()

    def test_init_with_new_directory(self, tmp_path: Path) -> None:
        """ModelVersionManager should create new directory if needed."""
        new_dir = tmp_path / "new_models"
        manager = ModelVersionManager(base_dir=new_dir)
        assert new_dir.exists()


class TestCreateVersion:
    """Tests for create_version method."""

    def test_creates_first_version(
        self,
        version_manager: ModelVersionManager,
        sample_models: dict[str, nn.Module],
        sample_config: dict[str, int],
        sample_metrics: dict[str, float],
    ) -> None:
        """create_version should create first version as 1.0.0."""
        version = version_manager.create_version(
            models=sample_models,
            config=sample_config,
            metrics=sample_metrics,
        )

        assert version == "1.0.0"
        version_dir = version_manager.base_dir / "v1.0.0"
        assert version_dir.exists()
        assert (version_dir / "metadata.json").exists()

    def test_saves_model_weights(
        self,
        version_manager: ModelVersionManager,
        sample_models: dict[str, nn.Module],
        sample_config: dict[str, int],
        sample_metrics: dict[str, float],
    ) -> None:
        """create_version should save model weight files."""
        version_manager.create_version(
            models=sample_models,
            config=sample_config,
            metrics=sample_metrics,
        )

        version_dir = version_manager.base_dir / "v1.0.0"
        assert (version_dir / "transformer.pt").exists()
        assert (version_dir / "gnn.pt").exists()
        assert (version_dir / "fusion.pt").exists()

    def test_bumps_version_with_parent(
        self,
        version_manager: ModelVersionManager,
        sample_models: dict[str, nn.Module],
        sample_config: dict[str, int],
        sample_metrics: dict[str, float],
    ) -> None:
        """create_version should bump version when parent specified."""
        # Create first version
        version_manager.create_version(
            models=sample_models,
            config=sample_config,
            metrics=sample_metrics,
        )

        # Create derived version
        version2 = version_manager.create_version(
            models=sample_models,
            config=sample_config,
            metrics=sample_metrics,
            parent_version="1.0.0",
            bump="minor",
        )

        assert version2 == "1.1.0"

    def test_major_bump(
        self,
        version_manager: ModelVersionManager,
        sample_models: dict[str, nn.Module],
        sample_config: dict[str, int],
        sample_metrics: dict[str, float],
    ) -> None:
        """create_version should support major version bump."""
        version_manager.create_version(
            models=sample_models,
            config=sample_config,
            metrics=sample_metrics,
        )

        version2 = version_manager.create_version(
            models=sample_models,
            config=sample_config,
            metrics=sample_metrics,
            parent_version="1.0.0",
            bump="major",
        )

        assert version2 == "2.0.0"


class TestCompareVersions:
    """Tests for compare_versions method."""

    def test_compares_using_stored_metrics(
        self,
        version_manager: ModelVersionManager,
        sample_models: dict[str, nn.Module],
        sample_config: dict[str, int],
    ) -> None:
        """compare_versions should use stored validation metrics."""
        # Create two versions with different metrics
        version_manager.create_version(
            models=sample_models,
            config=sample_config,
            metrics={"accuracy": 0.55, "brier_score": 0.25},
        )

        version_manager.create_version(
            models=sample_models,
            config=sample_config,
            metrics={"accuracy": 0.60, "brier_score": 0.20},
            parent_version="1.0.0",
        )

        result = version_manager.compare_versions("1.0.0", "1.1.0")

        assert isinstance(result, VersionComparisonResult)
        assert result.version_a == "1.0.0"
        assert result.version_b == "1.1.0"
        assert result.version_a_metrics["accuracy"] == 0.55
        assert result.version_b_metrics["accuracy"] == 0.60

    def test_computes_live_metrics_with_test_data(
        self,
        version_manager: ModelVersionManager,
        sample_models: dict[str, nn.Module],
        sample_config: dict[str, int],
    ) -> None:
        """compare_versions should compute metrics from test_data with win_prob."""
        import pandas as pd

        # Create two versions
        version_manager.create_version(
            models=sample_models,
            config=sample_config,
            metrics={"accuracy": 0.55, "brier_score": 0.25},
        )

        version_manager.create_version(
            models=sample_models,
            config=sample_config,
            metrics={"accuracy": 0.60, "brier_score": 0.20},
            parent_version="1.0.0",
        )

        # Create test data with pre-computed predictions (win_prob)
        # Version A: 60% accuracy (6/10 correct)
        # Version B: 80% accuracy (8/10 correct)
        test_data = pd.DataFrame({
            "home_win": [1, 1, 1, 0, 0, 1, 0, 1, 0, 0],
            "win_prob": [0.7, 0.8, 0.6, 0.3, 0.4, 0.6, 0.3, 0.7, 0.4, 0.2],
        })

        # Compare with test_data - should compute live metrics
        result = version_manager.compare_versions("1.0.0", "1.1.0", test_data=test_data)

        # Both versions should have the same computed metrics since they use same test_data
        # The predictions are: [1,1,1,0,0,1,0,1,0,0] (threshold 0.5)
        # Actuals are:         [1,1,1,0,0,1,0,1,0,0]
        # All 10 predictions are correct -> accuracy = 1.0
        assert result.version_a_metrics["accuracy"] == 1.0
        assert result.version_b_metrics["accuracy"] == 1.0
        assert "brier_score" in result.version_a_metrics

    def test_handles_empty_test_data(
        self,
        version_manager: ModelVersionManager,
        sample_models: dict[str, nn.Module],
        sample_config: dict[str, int],
    ) -> None:
        """compare_versions should fall back to stored metrics for empty test_data."""
        import pandas as pd

        version_manager.create_version(
            models=sample_models,
            config=sample_config,
            metrics={"accuracy": 0.55, "brier_score": 0.25},
        )

        version_manager.create_version(
            models=sample_models,
            config=sample_config,
            metrics={"accuracy": 0.60, "brier_score": 0.20},
            parent_version="1.0.0",
        )

        # Empty test data should fall back to stored metrics
        test_data = pd.DataFrame({"home_win": [], "win_prob": []})

        result = version_manager.compare_versions("1.0.0", "1.1.0", test_data=test_data)

        # Should use stored metrics for empty data
        assert result.version_a_metrics["accuracy"] == 0.55
        assert result.version_b_metrics["accuracy"] == 0.60

    def test_determines_winner_by_accuracy(
        self,
        version_manager: ModelVersionManager,
        sample_models: dict[str, nn.Module],
        sample_config: dict[str, int],
    ) -> None:
        """compare_versions should determine winner by accuracy first."""
        version_manager.create_version(
            models=sample_models,
            config=sample_config,
            metrics={"accuracy": 0.55},
        )

        version_manager.create_version(
            models=sample_models,
            config=sample_config,
            metrics={"accuracy": 0.60},
            parent_version="1.0.0",
        )

        result = version_manager.compare_versions("1.0.0", "1.1.0")

        assert result.winner == "1.1.0"

    def test_calculates_improvement(
        self,
        version_manager: ModelVersionManager,
        sample_models: dict[str, nn.Module],
        sample_config: dict[str, int],
    ) -> None:
        """compare_versions should calculate metric improvements."""
        version_manager.create_version(
            models=sample_models,
            config=sample_config,
            metrics={"accuracy": 0.55, "brier_score": 0.25},
        )

        version_manager.create_version(
            models=sample_models,
            config=sample_config,
            metrics={"accuracy": 0.60, "brier_score": 0.20},
            parent_version="1.0.0",
        )

        result = version_manager.compare_versions("1.0.0", "1.1.0")

        # Accuracy improvement (higher is better)
        assert abs(result.improvement["accuracy"] - 0.05) < 0.001
        # Brier score improvement (lower is better, so positive = B is better)
        assert abs(result.improvement["brier_score"] - 0.05) < 0.001

    def test_raises_on_missing_version(
        self,
        version_manager: ModelVersionManager,
        sample_models: dict[str, nn.Module],
        sample_config: dict[str, int],
    ) -> None:
        """compare_versions should raise error for missing version."""
        version_manager.create_version(
            models=sample_models,
            config=sample_config,
            metrics={"accuracy": 0.55},
        )

        with pytest.raises(ModelNotFoundError):
            version_manager.compare_versions("1.0.0", "2.0.0")


class TestPromoteVersion:
    """Tests for promote_version method."""

    def test_updates_latest_symlink(
        self,
        version_manager: ModelVersionManager,
        sample_models: dict[str, nn.Module],
        sample_config: dict[str, int],
        sample_metrics: dict[str, float],
    ) -> None:
        """promote_version should update latest pointer."""
        version_manager.create_version(
            models=sample_models,
            config=sample_config,
            metrics=sample_metrics,
        )

        version_manager.create_version(
            models=sample_models,
            config=sample_config,
            metrics=sample_metrics,
            parent_version="1.0.0",
        )

        version_manager.promote_version("1.1.0")

        # Latest should now point to 1.1.0
        latest = version_manager.registry.get_latest_version()
        assert latest == "1.1.0"

    def test_updates_status_to_promoted(
        self,
        version_manager: ModelVersionManager,
        sample_models: dict[str, nn.Module],
        sample_config: dict[str, int],
        sample_metrics: dict[str, float],
    ) -> None:
        """promote_version should set status to promoted."""
        version_manager.create_version(
            models=sample_models,
            config=sample_config,
            metrics=sample_metrics,
        )

        version_manager.promote_version("1.0.0")

        meta = version_manager._load_version_metadata("1.0.0")
        assert meta is not None
        assert meta.status == "promoted"

    def test_deprecates_previous_promoted(
        self,
        version_manager: ModelVersionManager,
        sample_models: dict[str, nn.Module],
        sample_config: dict[str, int],
        sample_metrics: dict[str, float],
    ) -> None:
        """promote_version should deprecate previously promoted version."""
        version_manager.create_version(
            models=sample_models,
            config=sample_config,
            metrics=sample_metrics,
        )
        version_manager.promote_version("1.0.0")

        version_manager.create_version(
            models=sample_models,
            config=sample_config,
            metrics=sample_metrics,
            parent_version="1.0.0",
        )
        version_manager.promote_version("1.1.0")

        meta_old = version_manager._load_version_metadata("1.0.0")
        assert meta_old is not None
        assert meta_old.status == "deprecated"

    def test_raises_on_missing_version(
        self,
        version_manager: ModelVersionManager,
    ) -> None:
        """promote_version should raise error for missing version."""
        with pytest.raises(ModelNotFoundError):
            version_manager.promote_version("9.9.9")


class TestRollback:
    """Tests for rollback method."""

    def test_restores_previous_version(
        self,
        version_manager: ModelVersionManager,
        sample_models: dict[str, nn.Module],
        sample_config: dict[str, int],
        sample_metrics: dict[str, float],
    ) -> None:
        """rollback should restore a previous version as latest."""
        version_manager.create_version(
            models=sample_models,
            config=sample_config,
            metrics=sample_metrics,
        )

        version_manager.create_version(
            models=sample_models,
            config=sample_config,
            metrics=sample_metrics,
            parent_version="1.0.0",
        )

        version_manager.rollback("1.0.0")

        latest = version_manager.registry.get_latest_version()
        assert latest == "1.0.0"

    def test_marks_rolled_back_version(
        self,
        version_manager: ModelVersionManager,
        sample_models: dict[str, nn.Module],
        sample_config: dict[str, int],
        sample_metrics: dict[str, float],
    ) -> None:
        """rollback should mark the rolled back version."""
        version_manager.create_version(
            models=sample_models,
            config=sample_config,
            metrics=sample_metrics,
        )

        version_manager.create_version(
            models=sample_models,
            config=sample_config,
            metrics=sample_metrics,
            parent_version="1.0.0",
        )

        version_manager.rollback("1.0.0")

        meta = version_manager._load_version_metadata("1.1.0")
        assert meta is not None
        assert meta.status == "rolled_back"

    def test_raises_on_missing_version(
        self,
        version_manager: ModelVersionManager,
    ) -> None:
        """rollback should raise error for missing version."""
        with pytest.raises(ModelNotFoundError):
            version_manager.rollback("9.9.9")


class TestGetLineage:
    """Tests for get_lineage method."""

    def test_returns_ancestry_chain(
        self,
        version_manager: ModelVersionManager,
        sample_models: dict[str, nn.Module],
        sample_config: dict[str, int],
        sample_metrics: dict[str, float],
    ) -> None:
        """get_lineage should return ordered ancestry chain."""
        version_manager.create_version(
            models=sample_models,
            config=sample_config,
            metrics=sample_metrics,
        )

        version_manager.create_version(
            models=sample_models,
            config=sample_config,
            metrics=sample_metrics,
            parent_version="1.0.0",
        )

        version_manager.create_version(
            models=sample_models,
            config=sample_config,
            metrics=sample_metrics,
            parent_version="1.1.0",
        )

        lineage = version_manager.get_lineage("1.2.0")

        assert lineage == ["1.0.0", "1.1.0", "1.2.0"]

    def test_returns_single_version_for_root(
        self,
        version_manager: ModelVersionManager,
        sample_models: dict[str, nn.Module],
        sample_config: dict[str, int],
        sample_metrics: dict[str, float],
    ) -> None:
        """get_lineage should return single version for root."""
        version_manager.create_version(
            models=sample_models,
            config=sample_config,
            metrics=sample_metrics,
        )

        lineage = version_manager.get_lineage("1.0.0")

        assert lineage == ["1.0.0"]


class TestListVersions:
    """Tests for list_versions method."""

    def test_returns_all_versions(
        self,
        version_manager: ModelVersionManager,
        sample_models: dict[str, nn.Module],
        sample_config: dict[str, int],
        sample_metrics: dict[str, float],
    ) -> None:
        """list_versions should return all versions."""
        version_manager.create_version(
            models=sample_models,
            config=sample_config,
            metrics=sample_metrics,
        )

        version_manager.create_version(
            models=sample_models,
            config=sample_config,
            metrics=sample_metrics,
            parent_version="1.0.0",
        )

        versions = version_manager.list_versions()

        assert len(versions) == 2
        version_numbers = [v.get("version") for v in versions]
        assert "1.0.0" in version_numbers
        assert "1.1.0" in version_numbers

    def test_sorted_by_creation_date(
        self,
        version_manager: ModelVersionManager,
        sample_models: dict[str, nn.Module],
        sample_config: dict[str, int],
        sample_metrics: dict[str, float],
    ) -> None:
        """list_versions should be sorted by creation date descending."""
        version_manager.create_version(
            models=sample_models,
            config=sample_config,
            metrics=sample_metrics,
        )

        version_manager.create_version(
            models=sample_models,
            config=sample_config,
            metrics=sample_metrics,
            parent_version="1.0.0",
        )

        versions = version_manager.list_versions()

        # Most recent first
        assert versions[0].get("version") == "1.1.0"
        assert versions[1].get("version") == "1.0.0"

    def test_empty_when_no_versions(
        self,
        version_manager: ModelVersionManager,
    ) -> None:
        """list_versions should return empty list when no versions."""
        versions = version_manager.list_versions()
        assert versions == []


class TestVersionComparisonResult:
    """Tests for VersionComparisonResult dataclass."""

    def test_to_dict(self) -> None:
        """VersionComparisonResult should convert to dictionary."""
        result = VersionComparisonResult(
            version_a="1.0.0",
            version_b="1.1.0",
            version_a_metrics={"accuracy": 0.55},
            version_b_metrics={"accuracy": 0.60},
            winner="1.1.0",
            improvement={"accuracy": 0.05},
        )

        result_dict = result.to_dict()

        assert result_dict["version_a"] == "1.0.0"
        assert result_dict["version_b"] == "1.1.0"
        assert result_dict["winner"] == "1.1.0"
        assert result_dict["improvement"]["accuracy"] == 0.05
