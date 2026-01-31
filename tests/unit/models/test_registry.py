"""Tests for ModelRegistry.

Tests cover:
- Version management
- Save/load cycles
- Metadata handling
- Version comparison
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from nba_model.models.registry import (
    ModelRegistry,
    VersionInfo,
    VersionComparison,
)
from nba_model.types import ModelNotFoundError


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 5)


class TestModelRegistry:
    """Tests for ModelRegistry class."""

    @pytest.fixture
    def registry(self, tmp_path: Path) -> ModelRegistry:
        """Create registry with temp directory."""
        return ModelRegistry(base_dir=tmp_path)

    @pytest.fixture
    def sample_models(self) -> dict[str, nn.Module]:
        """Create sample models for testing."""
        return {
            "transformer": SimpleModel(),
            "gnn": SimpleModel(),
            "fusion": SimpleModel(),
        }

    @pytest.fixture
    def sample_metrics(self) -> dict[str, float]:
        """Sample validation metrics."""
        return {
            "accuracy": 0.58,
            "brier_score": 0.23,
            "loss": 0.65,
        }

    @pytest.fixture
    def sample_config(self) -> dict[str, float]:
        """Sample training config."""
        return {
            "d_model": 128,
            "learning_rate": 1e-4,
            "batch_size": 32,
        }

    def test_initialization_creates_directory(self, tmp_path: Path) -> None:
        """Registry should create base directory."""
        registry = ModelRegistry(base_dir=tmp_path / "models")
        assert registry.base_dir.exists()

    def test_save_model_creates_version_dir(
        self,
        registry: ModelRegistry,
        sample_models: dict,
        sample_metrics: dict,
        sample_config: dict,
    ) -> None:
        """save_model should create versioned directory."""
        registry.save_model("1.0.0", sample_models, sample_metrics, sample_config)

        version_dir = registry.base_dir / "v1.0.0"
        assert version_dir.exists()
        assert (version_dir / "transformer.pt").exists()
        assert (version_dir / "gnn.pt").exists()
        assert (version_dir / "fusion.pt").exists()
        assert (version_dir / "metadata.json").exists()

    def test_save_model_metadata(
        self,
        registry: ModelRegistry,
        sample_models: dict,
        sample_metrics: dict,
        sample_config: dict,
    ) -> None:
        """Metadata should contain all required fields."""
        registry.save_model("1.0.0", sample_models, sample_metrics, sample_config)

        metadata_path = registry.base_dir / "v1.0.0" / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert metadata["version"] == "1.0.0"
        assert "training_date" in metadata
        assert metadata["hyperparameters"] == sample_config
        assert metadata["validation_metrics"] == sample_metrics

    def test_save_model_duplicate_raises(
        self,
        registry: ModelRegistry,
        sample_models: dict,
        sample_metrics: dict,
        sample_config: dict,
    ) -> None:
        """Should raise error when version exists."""
        registry.save_model("1.0.0", sample_models, sample_metrics, sample_config)

        with pytest.raises(ValueError, match="exists"):
            registry.save_model("1.0.0", sample_models, sample_metrics, sample_config)

    def test_save_model_invalid_version_raises(
        self,
        registry: ModelRegistry,
        sample_models: dict,
        sample_metrics: dict,
        sample_config: dict,
    ) -> None:
        """Should raise error for invalid version format."""
        with pytest.raises(ValueError, match="Invalid version"):
            registry.save_model("invalid", sample_models, sample_metrics, sample_config)

    def test_load_model_returns_state_dicts(
        self,
        registry: ModelRegistry,
        sample_models: dict,
        sample_metrics: dict,
        sample_config: dict,
    ) -> None:
        """load_model should return state dictionaries."""
        registry.save_model("1.0.0", sample_models, sample_metrics, sample_config)
        loaded = registry.load_model("1.0.0")

        assert "transformer" in loaded
        assert "gnn" in loaded
        assert "fusion" in loaded
        assert isinstance(loaded["transformer"], dict)

    def test_load_model_weights_match(
        self,
        registry: ModelRegistry,
        sample_metrics: dict,
        sample_config: dict,
    ) -> None:
        """Loaded weights should match saved weights."""
        # Create model with known weights
        model = SimpleModel()
        with torch.no_grad():
            model.linear.weight.fill_(0.5)

        models = {"transformer": model, "gnn": model, "fusion": model}
        registry.save_model("1.0.0", models, sample_metrics, sample_config)

        loaded = registry.load_model("1.0.0")
        assert torch.allclose(
            loaded["transformer"]["linear.weight"],
            torch.full((5, 10), 0.5),
        )

    def test_load_model_latest(
        self,
        registry: ModelRegistry,
        sample_models: dict,
        sample_metrics: dict,
        sample_config: dict,
    ) -> None:
        """Should load latest version with 'latest' argument."""
        registry.save_model("1.0.0", sample_models, sample_metrics, sample_config)
        registry.save_model("1.0.1", sample_models, sample_metrics, sample_config)

        loaded = registry.load_model("latest")
        # Should succeed if latest symlink works
        assert "transformer" in loaded

    def test_load_model_not_found_raises(
        self,
        registry: ModelRegistry,
    ) -> None:
        """Should raise ModelNotFoundError for missing version."""
        with pytest.raises(ModelNotFoundError):
            registry.load_model("9.9.9")

    def test_load_metadata(
        self,
        registry: ModelRegistry,
        sample_models: dict,
        sample_metrics: dict,
        sample_config: dict,
    ) -> None:
        """Should load metadata correctly."""
        registry.save_model(
            "1.0.0",
            sample_models,
            sample_metrics,
            sample_config,
            training_data_start=date(2023, 10, 1),
            training_data_end=date(2024, 4, 1),
        )

        metadata = registry.load_metadata("1.0.0")
        assert metadata is not None
        assert metadata.version == "1.0.0"
        assert metadata.training_data_start == date(2023, 10, 1)
        assert metadata.validation_metrics == sample_metrics

    def test_list_versions(
        self,
        registry: ModelRegistry,
        sample_models: dict,
        sample_metrics: dict,
        sample_config: dict,
    ) -> None:
        """Should list all versions."""
        registry.save_model("1.0.0", sample_models, sample_metrics, sample_config)
        registry.save_model("1.1.0", sample_models, sample_metrics, sample_config)
        registry.save_model("2.0.0", sample_models, sample_metrics, sample_config)

        versions = registry.list_versions()

        assert len(versions) == 3
        # Should be sorted descending
        assert versions[0].version == "2.0.0"
        assert versions[1].version == "1.1.0"
        assert versions[2].version == "1.0.0"

    def test_list_versions_marks_latest(
        self,
        registry: ModelRegistry,
        sample_models: dict,
        sample_metrics: dict,
        sample_config: dict,
    ) -> None:
        """Should mark latest version."""
        registry.save_model("1.0.0", sample_models, sample_metrics, sample_config)
        registry.save_model("1.1.0", sample_models, sample_metrics, sample_config)

        versions = registry.list_versions()
        latest = [v for v in versions if v.is_latest]

        assert len(latest) == 1
        assert latest[0].version == "1.1.0"

    def test_compare_versions(
        self,
        registry: ModelRegistry,
        sample_models: dict,
        sample_config: dict,
    ) -> None:
        """Should compare two versions."""
        metrics_v1 = {"accuracy": 0.55, "loss": 0.70}
        metrics_v2 = {"accuracy": 0.58, "loss": 0.65}

        registry.save_model("1.0.0", sample_models, metrics_v1, sample_config)
        registry.save_model("1.1.0", sample_models, metrics_v2, sample_config)

        comparison = registry.compare_versions("1.0.0", "1.1.0")

        assert comparison.version_a == "1.0.0"
        assert comparison.version_b == "1.1.0"
        assert comparison.winner == "1.1.0"  # Lower loss
        assert comparison.improvements["accuracy"] == pytest.approx(0.03, rel=1e-6)

    def test_delete_version(
        self,
        registry: ModelRegistry,
        sample_models: dict,
        sample_metrics: dict,
        sample_config: dict,
    ) -> None:
        """Should delete version directory."""
        registry.save_model("1.0.0", sample_models, sample_metrics, sample_config)
        assert registry.delete_version("1.0.0")
        assert not (registry.base_dir / "v1.0.0").exists()

    def test_delete_nonexistent_returns_false(
        self,
        registry: ModelRegistry,
    ) -> None:
        """Should return False for nonexistent version."""
        assert not registry.delete_version("9.9.9")

    def test_next_version_patch(
        self,
        registry: ModelRegistry,
        sample_models: dict,
        sample_metrics: dict,
        sample_config: dict,
    ) -> None:
        """next_version should bump patch number."""
        registry.save_model("1.2.3", sample_models, sample_metrics, sample_config)
        assert registry.next_version("patch") == "1.2.4"

    def test_next_version_minor(
        self,
        registry: ModelRegistry,
        sample_models: dict,
        sample_metrics: dict,
        sample_config: dict,
    ) -> None:
        """next_version should bump minor and reset patch."""
        registry.save_model("1.2.3", sample_models, sample_metrics, sample_config)
        assert registry.next_version("minor") == "1.3.0"

    def test_next_version_major(
        self,
        registry: ModelRegistry,
        sample_models: dict,
        sample_metrics: dict,
        sample_config: dict,
    ) -> None:
        """next_version should bump major and reset minor/patch."""
        registry.save_model("1.2.3", sample_models, sample_metrics, sample_config)
        assert registry.next_version("major") == "2.0.0"

    def test_next_version_no_versions(
        self,
        registry: ModelRegistry,
    ) -> None:
        """next_version should return 1.0.0 if no versions exist."""
        assert registry.next_version() == "1.0.0"

    def test_version_normalization(
        self,
        registry: ModelRegistry,
        sample_models: dict,
        sample_metrics: dict,
        sample_config: dict,
    ) -> None:
        """Should handle versions with 'v' prefix."""
        registry.save_model("v1.0.0", sample_models, sample_metrics, sample_config)
        # Should load with or without 'v'
        loaded = registry.load_model("1.0.0")
        assert "transformer" in loaded


class TestVersionInfo:
    """Tests for VersionInfo dataclass."""

    def test_all_fields(self, tmp_path: Path) -> None:
        """VersionInfo should have all fields."""
        info = VersionInfo(
            version="1.0.0",
            path=tmp_path,
            metadata=None,
            is_latest=True,
        )

        assert info.version == "1.0.0"
        assert info.path == tmp_path
        assert info.is_latest is True
