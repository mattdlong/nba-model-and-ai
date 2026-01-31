"""Model versioning and storage registry.

This module implements versioned model storage and retrieval with comprehensive
metadata tracking. Models are saved in a structured directory hierarchy with
semantic versioning.

Storage Structure:
    data/models/
    ├── v1.0.0/
    │   ├── transformer.pt
    │   ├── gnn.pt
    │   ├── fusion.pt
    │   └── metadata.json
    ├── v1.1.0/
    │   └── ...
    └── latest -> v1.1.0 (symlink)

Example:
    >>> from nba_model.models.registry import ModelRegistry
    >>> registry = ModelRegistry()
    >>> registry.save_model("1.0.0", models, metrics, config)
    >>> models = registry.load_model("latest")
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

import torch

from nba_model.config import get_settings
from nba_model.logging import get_logger
from nba_model.types import ModelMetadata, ModelNotFoundError

logger = get_logger(__name__)

# =============================================================================
# Constants
# =============================================================================

MODEL_FILES = ["transformer.pt", "gnn.pt", "fusion.pt"]
METADATA_FILE = "metadata.json"
LATEST_LINK = "latest"

VERSION_PATTERN = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")


# =============================================================================
# Data Containers
# =============================================================================


@dataclass
class VersionInfo:
    """Information about a model version.

    Attributes:
        version: Version string (e.g., "1.0.0").
        path: Path to version directory.
        metadata: Model metadata.
        is_latest: Whether this is the latest version.
    """

    version: str
    path: Path
    metadata: ModelMetadata | None = None
    is_latest: bool = False


@dataclass
class VersionComparison:
    """Comparison between two model versions.

    Attributes:
        version_a: First version string.
        version_b: Second version string.
        metrics_a: Metrics for version A.
        metrics_b: Metrics for version B.
        winner: Version with better metrics.
        improvements: Metric-by-metric comparison.
    """

    version_a: str
    version_b: str
    metrics_a: dict[str, float]
    metrics_b: dict[str, float]
    winner: str
    improvements: dict[str, float] = field(default_factory=dict)


# =============================================================================
# Registry
# =============================================================================


class ModelRegistry:
    """Model version management and storage.

    Manages versioned model checkpoints with metadata tracking, including
    training dates, hyperparameters, validation metrics, and git commit hashes.

    Attributes:
        base_dir: Base directory for model storage.
        device: Device to load models onto.

    Example:
        >>> registry = ModelRegistry("data/models")
        >>> registry.save_model("1.0.0", models, metrics, config)
        >>> loaded = registry.load_model("latest")
        >>> transformer, gnn, fusion = loaded["transformer"], loaded["gnn"], loaded["fusion"]
    """

    def __init__(
        self,
        base_dir: str | Path | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Initialize ModelRegistry.

        Args:
            base_dir: Directory for model storage. If None, uses config default.
            device: Device to load models onto. If None, auto-detects.
        """
        if base_dir is None:
            settings = get_settings()
            self.base_dir = settings.model_dir_obj
        else:
            self.base_dir = Path(base_dir)

        self.base_dir.mkdir(parents=True, exist_ok=True)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        logger.debug("Initialized ModelRegistry at {}", self.base_dir)

    def save_model(
        self,
        version: str,
        models: dict[str, torch.nn.Module],
        metrics: dict[str, float],
        config: dict[str, Any],
        training_data_start: date | None = None,
        training_data_end: date | None = None,
        parent_version: str | None = None,
    ) -> Path:
        """Save model weights and metadata to versioned directory.

        Args:
            version: Version string (e.g., "1.0.0").
            models: Dictionary with keys "transformer", "gnn", "fusion"
                mapping to model instances.
            metrics: Validation metrics dictionary.
            config: Hyperparameter configuration.
            training_data_start: Start date of training data.
            training_data_end: End date of training data.
            parent_version: Parent version for lineage tracking.

        Returns:
            Path to the saved version directory.

        Raises:
            ValueError: If version format is invalid or version exists.
        """
        # Validate version format
        version = self._normalize_version(version)
        if not self._is_valid_version(version):
            raise ValueError(f"Invalid version format: {version}")

        version_dir = self.base_dir / f"v{version}"
        if version_dir.exists():
            raise ValueError(f"Version {version} already exists")

        # Create version directory
        version_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights
        for name, model in models.items():
            model_path = version_dir / f"{name}.pt"
            if hasattr(model, "state_dict"):
                torch.save(model.state_dict(), model_path)
            else:
                torch.save(model, model_path)
            logger.debug("Saved {} to {}", name, model_path)

        # Create metadata
        metadata = ModelMetadata(
            version=version,
            training_date=datetime.now(),
            training_data_start=training_data_start or date.today(),
            training_data_end=training_data_end or date.today(),
            hyperparameters=config,
            validation_metrics=metrics,
            git_commit=self._get_git_commit(),
        )

        # Add parent version if provided
        metadata_dict = {
            "version": metadata.version,
            "training_date": metadata.training_date.isoformat(),
            "training_data_start": metadata.training_data_start.isoformat(),
            "training_data_end": metadata.training_data_end.isoformat(),
            "hyperparameters": metadata.hyperparameters,
            "validation_metrics": metadata.validation_metrics,
            "git_commit": metadata.git_commit,
            "parent_version": parent_version,
        }

        # Save metadata
        metadata_path = version_dir / METADATA_FILE
        with open(metadata_path, "w") as f:
            json.dump(metadata_dict, f, indent=2, default=str)

        # Update latest symlink
        self._update_latest_link(version)

        logger.info("Saved model version {} to {}", version, version_dir)
        return version_dir

    def load_model(
        self,
        version: str = "latest",
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Load model weights by version.

        Args:
            version: Version to load ("latest" or specific like "1.0.0").

        Returns:
            Dictionary with keys "transformer", "gnn", "fusion" mapping
            to state_dict dictionaries.

        Raises:
            ModelNotFoundError: If version doesn't exist.
        """
        version_dir = self._resolve_version(version)

        if not version_dir.exists():
            raise ModelNotFoundError(f"Version {version} not found at {version_dir}")

        models = {}
        for model_file in MODEL_FILES:
            model_path = version_dir / model_file
            if model_path.exists():
                name = model_file.replace(".pt", "")
                models[name] = torch.load(model_path, map_location=self.device)
                logger.debug("Loaded {} from {}", name, model_path)
            else:
                logger.warning("Model file {} not found in {}", model_file, version_dir)

        logger.info("Loaded model version {} from {}", version, version_dir)
        return models

    def load_metadata(self, version: str = "latest") -> ModelMetadata | None:
        """Load metadata for a version.

        Args:
            version: Version to load metadata for.

        Returns:
            ModelMetadata object or None if not found.
        """
        version_dir = self._resolve_version(version)
        metadata_path = version_dir / METADATA_FILE

        if not metadata_path.exists():
            return None

        with open(metadata_path) as f:
            data = json.load(f)

        return ModelMetadata(
            version=data["version"],
            training_date=datetime.fromisoformat(data["training_date"]),
            training_data_start=date.fromisoformat(data["training_data_start"]),
            training_data_end=date.fromisoformat(data["training_data_end"]),
            hyperparameters=data.get("hyperparameters", {}),
            validation_metrics=data.get("validation_metrics", {}),
            git_commit=data.get("git_commit"),
        )

    def list_versions(self) -> list[VersionInfo]:
        """List all model versions with metadata.

        Returns:
            List of VersionInfo objects, sorted by version descending.
        """
        versions = []
        latest_version = self._get_latest_version()

        for item in self.base_dir.iterdir():
            if item.is_dir() and item.name.startswith("v"):
                version_str = item.name[1:]  # Remove "v" prefix
                if self._is_valid_version(version_str):
                    metadata = self.load_metadata(version_str)
                    versions.append(
                        VersionInfo(
                            version=version_str,
                            path=item,
                            metadata=metadata,
                            is_latest=(version_str == latest_version),
                        )
                    )

        # Sort by version descending
        versions.sort(key=lambda v: self._parse_version(v.version), reverse=True)
        return versions

    def compare_versions(
        self,
        version_a: str,
        version_b: str,
    ) -> VersionComparison:
        """Compare metrics between two versions.

        Args:
            version_a: First version to compare.
            version_b: Second version to compare.

        Returns:
            VersionComparison with detailed metric comparison.

        Raises:
            ModelNotFoundError: If either version doesn't exist.
        """
        meta_a = self.load_metadata(version_a)
        meta_b = self.load_metadata(version_b)

        if meta_a is None:
            raise ModelNotFoundError(f"Version {version_a} not found")
        if meta_b is None:
            raise ModelNotFoundError(f"Version {version_b} not found")

        metrics_a = meta_a.validation_metrics
        metrics_b = meta_b.validation_metrics

        # Calculate improvements (positive = B is better for most metrics)
        improvements = {}
        for key in set(metrics_a.keys()) | set(metrics_b.keys()):
            val_a = metrics_a.get(key, 0.0)
            val_b = metrics_b.get(key, 0.0)

            # For accuracy, higher is better
            # For loss/MAE/brier, lower is better
            if "accuracy" in key or "roi" in key:
                improvements[key] = val_b - val_a  # Positive if B is better
            else:
                improvements[key] = val_a - val_b  # Positive if B is better (lower)

        # Determine winner based on validation loss
        loss_a = metrics_a.get("loss", float("inf"))
        loss_b = metrics_b.get("loss", float("inf"))
        winner = version_a if loss_a < loss_b else version_b

        return VersionComparison(
            version_a=version_a,
            version_b=version_b,
            metrics_a=metrics_a,
            metrics_b=metrics_b,
            winner=winner,
            improvements=improvements,
        )

    def delete_version(self, version: str) -> bool:
        """Delete a model version.

        Args:
            version: Version to delete.

        Returns:
            True if deleted, False if not found.
        """
        import shutil

        version = self._normalize_version(version)
        version_dir = self.base_dir / f"v{version}"

        if not version_dir.exists():
            return False

        shutil.rmtree(version_dir)
        logger.info("Deleted model version {}", version)

        # Update latest if needed
        if self._get_latest_version() == version:
            versions = self.list_versions()
            if versions:
                self._update_latest_link(versions[0].version)

        return True

    def get_latest_version(self) -> str | None:
        """Get the latest version string.

        Returns:
            Latest version string or None if no versions exist.
        """
        return self._get_latest_version()

    def _resolve_version(self, version: str) -> Path:
        """Resolve version string to directory path.

        Args:
            version: Version string or "latest".

        Returns:
            Path to version directory.
        """
        if version == "latest":
            latest_path = self.base_dir / LATEST_LINK
            if latest_path.is_symlink():
                return latest_path.resolve()
            elif latest_path.exists():
                return latest_path
            else:
                # Find latest by version number
                latest = self._get_latest_version()
                if latest:
                    return self.base_dir / f"v{latest}"
                raise ModelNotFoundError("No model versions found")
        else:
            version = self._normalize_version(version)
            return self.base_dir / f"v{version}"

    def _update_latest_link(self, version: str) -> None:
        """Update the 'latest' symlink to point to a version.

        Args:
            version: Version to link to.
        """
        version = self._normalize_version(version)
        latest_path = self.base_dir / LATEST_LINK
        target_path = self.base_dir / f"v{version}"

        # Remove existing link/file
        if latest_path.is_symlink() or latest_path.exists():
            latest_path.unlink()

        # Create new symlink (relative for portability)
        try:
            latest_path.symlink_to(f"v{version}")
            logger.debug("Updated latest symlink to v{}", version)
        except OSError as e:
            # Symlinks may fail on some Windows systems
            logger.warning("Could not create symlink: {}", e)

    def _get_latest_version(self) -> str | None:
        """Get the latest version from symlink or version numbers.

        Returns:
            Latest version string or None.
        """
        latest_path = self.base_dir / LATEST_LINK

        # Try symlink first
        if latest_path.is_symlink():
            target = latest_path.resolve()
            if target.exists() and target.name.startswith("v"):
                return target.name[1:]

        # Fall back to finding highest version
        versions = []
        for item in self.base_dir.iterdir():
            if item.is_dir() and item.name.startswith("v"):
                version_str = item.name[1:]
                if self._is_valid_version(version_str):
                    versions.append(version_str)

        if not versions:
            return None

        versions.sort(key=self._parse_version, reverse=True)
        return versions[0]

    def _normalize_version(self, version: str) -> str:
        """Normalize version string (remove 'v' prefix if present).

        Args:
            version: Version string.

        Returns:
            Normalized version without 'v' prefix.
        """
        if version.startswith("v"):
            return version[1:]
        return version

    def _is_valid_version(self, version: str) -> bool:
        """Check if version string is valid semantic version.

        Args:
            version: Version string to validate.

        Returns:
            True if valid.
        """
        return VERSION_PATTERN.match(version) is not None

    def _parse_version(self, version: str) -> tuple[int, int, int]:
        """Parse version string into tuple for comparison.

        Args:
            version: Version string (e.g., "1.2.3").

        Returns:
            Tuple of (major, minor, patch).
        """
        match = VERSION_PATTERN.match(version)
        if match:
            return int(match.group(1)), int(match.group(2)), int(match.group(3))
        return 0, 0, 0

    def _get_git_commit(self) -> str | None:
        """Get current git commit hash.

        Returns:
            Short git commit hash or None if not in git repo.
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        return None

    def next_version(
        self,
        bump: str = "patch",
    ) -> str:
        """Calculate the next version number.

        Args:
            bump: Version component to bump ("major", "minor", "patch").

        Returns:
            Next version string.
        """
        latest = self._get_latest_version()

        if latest is None:
            return "1.0.0"

        major, minor, patch = self._parse_version(latest)

        if bump == "major":
            return f"{major + 1}.0.0"
        elif bump == "minor":
            return f"{major}.{minor + 1}.0"
        else:  # patch
            return f"{major}.{minor}.{patch + 1}"
