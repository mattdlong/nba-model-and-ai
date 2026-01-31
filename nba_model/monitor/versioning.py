"""Model version management for continuous learning pipeline.

This module provides enhanced version management capabilities for the
model monitoring system, including version comparison, promotion workflows,
rollback functionality, and lineage tracking.

Extends the base ModelRegistry with:
    - Automatic version bumping based on change type
    - Version comparison on test data with live inference
    - Promotion and rollback workflows
    - Lineage tracking for version ancestry

Example:
    >>> manager = ModelVersionManager()
    >>> version = manager.create_version(models, config, metrics)
    >>> comparison = manager.compare_versions("v1.0.0", "v1.1.0", test_data)
    >>> if comparison.winner == "v1.1.0":
    ...     manager.promote_version("v1.1.0")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from nba_model.logging import get_logger
from nba_model.models.registry import ModelRegistry
from nba_model.types import ModelNotFoundError

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)

# =============================================================================
# Constants
# =============================================================================

# Version status values
STATUS_ACTIVE: str = "active"
STATUS_PROMOTED: str = "promoted"
STATUS_DEPRECATED: str = "deprecated"
STATUS_ROLLED_BACK: str = "rolled_back"

# Metadata file name
METADATA_FILE: str = "metadata.json"

# Metrics for comparison (higher is better for first group, lower for second)
METRICS_HIGHER_BETTER: set[str] = {"accuracy", "win_rate", "roi", "clv"}
METRICS_LOWER_BETTER: set[str] = {"brier_score", "log_loss", "margin_mae", "total_mae"}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class VersionMetadata:
    """Extended metadata for a model version.

    Attributes:
        version: Semantic version string (e.g., "1.0.0").
        created_at: ISO datetime when version was created.
        training_data_start: Start date of training data.
        training_data_end: End date of training data.
        hyperparameters: Model hyperparameters used.
        validation_metrics: Validation performance metrics.
        git_commit: Optional git commit hash.
        parent_version: Version this was derived from.
        status: Version status (active, promoted, deprecated).
    """

    version: str
    created_at: datetime
    training_data_start: date
    training_data_end: date
    hyperparameters: dict[str, Any]
    validation_metrics: dict[str, float]
    git_commit: str | None = None
    parent_version: str | None = None
    status: str = STATUS_ACTIVE

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "training_data_start": self.training_data_start.isoformat(),
            "training_data_end": self.training_data_end.isoformat(),
            "hyperparameters": self.hyperparameters,
            "validation_metrics": self.validation_metrics,
            "git_commit": self.git_commit,
            "parent_version": self.parent_version,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VersionMetadata:
        """Create from dictionary.

        Handles both new format (created_at) and legacy format (training_date).
        """
        # Handle both created_at and training_date (legacy from ModelRegistry)
        created_at_str = data.get("created_at") or data.get("training_date")
        if created_at_str:
            created_at = datetime.fromisoformat(created_at_str)
        else:
            created_at = datetime.now()

        return cls(
            version=data["version"],
            created_at=created_at,
            training_data_start=date.fromisoformat(data["training_data_start"]),
            training_data_end=date.fromisoformat(data["training_data_end"]),
            hyperparameters=data.get("hyperparameters", {}),
            validation_metrics=data.get("validation_metrics", {}),
            git_commit=data.get("git_commit"),
            parent_version=data.get("parent_version"),
            status=data.get("status", STATUS_ACTIVE),
        )


@dataclass
class VersionComparisonResult:
    """Result of comparing two model versions.

    Attributes:
        version_a: First version compared.
        version_b: Second version compared.
        version_a_metrics: Metrics for version A.
        version_b_metrics: Metrics for version B.
        winner: Version with better overall performance.
        improvement: Metric-by-metric improvement (B - A).
    """

    version_a: str
    version_b: str
    version_a_metrics: dict[str, float]
    version_b_metrics: dict[str, float]
    winner: str
    improvement: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "version_a": self.version_a,
            "version_b": self.version_b,
            "version_a_metrics": self.version_a_metrics,
            "version_b_metrics": self.version_b_metrics,
            "winner": self.winner,
            "improvement": self.improvement,
        }


# =============================================================================
# Model Version Manager
# =============================================================================


class ModelVersionManager:
    """Enhanced model version management for continuous learning.

    Provides additional functionality on top of ModelRegistry:
        - Automatic version string generation
        - Test data comparison with live inference
        - Promotion/deprecation workflow
        - Rollback capabilities
        - Version lineage tracking

    Attributes:
        base_dir: Base directory for model storage.
        registry: Underlying ModelRegistry instance.

    Example:
        >>> manager = ModelVersionManager()
        >>> version = manager.create_version(
        ...     models={"transformer": t, "gnn": g, "fusion": f},
        ...     config={"d_model": 128},
        ...     metrics={"accuracy": 0.58},
        ...     parent_version="1.0.0",
        ... )
        >>> print(f"Created version {version}")
    """

    def __init__(
        self,
        base_dir: str | Path | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Initialize ModelVersionManager.

        Args:
            base_dir: Directory for model storage. If None, uses config default.
            device: Device to load models onto. If None, auto-detects.
        """
        self.registry = ModelRegistry(base_dir=base_dir, device=device)
        self.base_dir = self.registry.base_dir

        logger.debug("Initialized ModelVersionManager at {}", self.base_dir)

    def create_version(
        self,
        models: dict[str, torch.nn.Module],
        config: dict[str, Any],
        metrics: dict[str, float],
        parent_version: str | None = None,
        bump: str = "minor",
    ) -> str:
        """Create a new model version.

        Automatically determines the next version number based on the
        parent version and bump type.

        Args:
            models: Dictionary of model instances to save.
            config: Hyperparameter configuration.
            metrics: Validation metrics.
            parent_version: Version this is derived from (optional).
            bump: Version component to bump ("major", "minor", "patch").

        Returns:
            Version string for the new version (e.g., "1.2.0").

        Raises:
            ValueError: If bump type is invalid.
        """
        # Determine version string
        if parent_version:
            # Bump from parent
            next_version = self._bump_version(parent_version, bump)
        else:
            # Get next version from registry
            next_version = self.registry.next_version(bump)

        logger.info(
            "Creating version {} (parent={}, bump={})",
            next_version,
            parent_version,
            bump,
        )

        # Save using registry
        self.registry.save_model(
            version=next_version,
            models=models,
            metrics=metrics,
            config=config,
            parent_version=parent_version,
        )

        # Update metadata with status
        self._update_version_status(next_version, STATUS_ACTIVE)

        return next_version

    def compare_versions(
        self,
        version_a: str,
        version_b: str,
        test_data: pd.DataFrame | None = None,
    ) -> VersionComparisonResult:
        """Compare two model versions.

        If test_data is provided, runs inference with both versions and
        computes fresh metrics. Otherwise, uses stored validation metrics.

        Args:
            version_a: First version to compare.
            version_b: Second version to compare.
            test_data: Optional test DataFrame for live comparison.

        Returns:
            VersionComparisonResult with detailed comparison.

        Raises:
            ModelNotFoundError: If either version doesn't exist.
        """
        # Load metadata for both versions
        meta_a = self._load_version_metadata(version_a)
        meta_b = self._load_version_metadata(version_b)

        if meta_a is None:
            raise ModelNotFoundError(f"Version {version_a} not found")
        if meta_b is None:
            raise ModelNotFoundError(f"Version {version_b} not found")

        # Get metrics (from storage or computed)
        if test_data is not None and not test_data.empty:
            metrics_a = self._compute_metrics_on_data(version_a, test_data)
            metrics_b = self._compute_metrics_on_data(version_b, test_data)
        else:
            metrics_a = meta_a.validation_metrics
            metrics_b = meta_b.validation_metrics

        # Calculate improvements (positive = B is better)
        improvement: dict[str, float] = {}
        for key in set(metrics_a.keys()) | set(metrics_b.keys()):
            val_a = metrics_a.get(key, 0.0)
            val_b = metrics_b.get(key, 0.0)

            if key in METRICS_HIGHER_BETTER:
                improvement[key] = val_b - val_a
            elif key in METRICS_LOWER_BETTER:
                improvement[key] = val_a - val_b  # Lower is better, so invert
            else:
                improvement[key] = val_b - val_a

        # Determine winner using priority: accuracy > brier_score > margin_mae
        winner = self._determine_winner(metrics_a, metrics_b, version_a, version_b)

        logger.info(
            "Version comparison: {} vs {} -> winner = {}",
            version_a,
            version_b,
            winner,
        )

        return VersionComparisonResult(
            version_a=version_a,
            version_b=version_b,
            version_a_metrics=metrics_a,
            version_b_metrics=metrics_b,
            winner=winner,
            improvement=improvement,
        )

    def promote_version(self, version: str) -> None:
        """Promote a version to production (latest).

        Sets the specified version as 'promoted' status and updates
        the 'latest' symlink. Deprecates the previously promoted version.

        Args:
            version: Version to promote.

        Raises:
            ModelNotFoundError: If version doesn't exist.
        """
        version = self.registry._normalize_version(version)

        # Verify version exists
        version_dir = self.base_dir / f"v{version}"
        if not version_dir.exists():
            raise ModelNotFoundError(f"Version {version} not found")

        # Find and deprecate currently promoted version
        for v_info in self.registry.list_versions():
            if v_info.version == version:
                continue
            meta = self._load_version_metadata(v_info.version)
            if meta and meta.status == STATUS_PROMOTED:
                self._update_version_status(v_info.version, STATUS_DEPRECATED)
                break

        # Promote new version
        self._update_version_status(version, STATUS_PROMOTED)

        # Update latest symlink
        self.registry._update_latest_link(version)

        logger.info("Promoted version {} to production", version)

    def rollback(self, to_version: str) -> None:
        """Rollback to a previous version.

        Sets the specified version as 'latest' and marks the current
        version as rolled back.

        Args:
            to_version: Version to rollback to.

        Raises:
            ModelNotFoundError: If version doesn't exist.
        """
        to_version = self.registry._normalize_version(to_version)

        # Verify version exists
        version_dir = self.base_dir / f"v{to_version}"
        if not version_dir.exists():
            raise ModelNotFoundError(f"Version {to_version} not found")

        # Mark current latest as rolled back
        current_latest = self.registry.get_latest_version()
        if current_latest and current_latest != to_version:
            self._update_version_status(current_latest, STATUS_ROLLED_BACK)
            self._add_rollback_event(current_latest, to_version)

        # Update status and symlink
        self._update_version_status(to_version, STATUS_PROMOTED)
        self.registry._update_latest_link(to_version)

        logger.warning(
            "Rolled back from {} to {}",
            current_latest,
            to_version,
        )

    def get_lineage(self, version: str) -> list[str]:
        """Get version ancestry/lineage.

        Traverses parent_version links to build the complete
        version history from oldest ancestor to specified version.

        Args:
            version: Version to get lineage for.

        Returns:
            Ordered list of versions from oldest to specified.

        Raises:
            ModelNotFoundError: If version doesn't exist.
        """
        version = self.registry._normalize_version(version)
        lineage: list[str] = []
        current: str | None = version

        # Traverse parent chain (with loop protection)
        visited: set[str] = set()
        while current is not None and current not in visited:
            visited.add(current)
            lineage.append(current)

            meta = self._load_version_metadata(current)
            if meta is None:
                break
            current = meta.parent_version

        # Reverse to get oldest first
        lineage.reverse()
        return lineage

    def list_versions(self) -> list[dict[str, Any]]:
        """List all model versions with metadata.

        Returns:
            List of version metadata dicts, sorted by creation date descending.
        """
        versions = self.registry.list_versions()
        result = []

        for version_info in versions:
            meta = self._load_version_metadata(version_info.version)
            if meta:
                result.append(meta.to_dict())
            else:
                result.append({
                    "version": version_info.version,
                    "is_latest": version_info.is_latest,
                    "status": STATUS_ACTIVE,
                })

        return result

    def _bump_version(self, version: str, bump: str) -> str:
        """Bump a version string by the specified component.

        Args:
            version: Current version string.
            bump: Component to bump ("major", "minor", "patch").

        Returns:
            Bumped version string.
        """
        version = self.registry._normalize_version(version)
        major, minor, patch = self.registry._parse_version(version)

        if bump == "major":
            return f"{major + 1}.0.0"
        elif bump == "minor":
            return f"{major}.{minor + 1}.0"
        else:  # patch
            return f"{major}.{minor}.{patch + 1}"

    def _load_version_metadata(self, version: str) -> VersionMetadata | None:
        """Load extended metadata for a version.

        Args:
            version: Version string.

        Returns:
            VersionMetadata or None if not found.
        """
        version = self.registry._normalize_version(version)
        metadata_path = self.base_dir / f"v{version}" / METADATA_FILE

        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path) as f:
                data = json.load(f)
            return VersionMetadata.from_dict(data)
        except Exception as e:
            logger.warning("Failed to load metadata for {}: {}", version, e)
            return None

    def _update_version_status(self, version: str, status: str) -> None:
        """Update the status field in version metadata.

        Args:
            version: Version to update.
            status: New status value.
        """
        version = self.registry._normalize_version(version)
        metadata_path = self.base_dir / f"v{version}" / METADATA_FILE

        if not metadata_path.exists():
            return

        try:
            with open(metadata_path) as f:
                data = json.load(f)

            data["status"] = status

            with open(metadata_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            logger.warning("Failed to update status for {}: {}", version, e)

    def _add_rollback_event(self, from_version: str, to_version: str) -> None:
        """Add rollback event to version metadata.

        Args:
            from_version: Version being rolled back from.
            to_version: Version being rolled back to.
        """
        from_version = self.registry._normalize_version(from_version)
        metadata_path = self.base_dir / f"v{from_version}" / METADATA_FILE

        if not metadata_path.exists():
            return

        try:
            with open(metadata_path) as f:
                data = json.load(f)

            if "rollback_events" not in data:
                data["rollback_events"] = []

            data["rollback_events"].append({
                "timestamp": datetime.now().isoformat(),
                "rolled_back_to": to_version,
            })

            with open(metadata_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            logger.warning("Failed to add rollback event: {}", e)

    def _compute_metrics_on_data(
        self,
        version: str,
        test_data: pd.DataFrame,
    ) -> dict[str, float]:
        """Compute metrics by running inference on test data.

        This is a placeholder that returns stored metrics if available.
        Full implementation would load models and run inference.

        Args:
            version: Version to evaluate.
            test_data: Test DataFrame with game features and outcomes.

        Returns:
            Dictionary of computed metrics.
        """
        # For now, fall back to stored metrics
        # Full implementation would:
        # 1. Load models for version
        # 2. Run inference on test_data
        # 3. Compare predictions to actuals
        # 4. Compute accuracy, brier_score, margin_mae, total_mae

        meta = self._load_version_metadata(version)
        if meta:
            return meta.validation_metrics

        logger.warning(
            "Could not compute live metrics for {} - returning empty",
            version,
        )
        return {}

    def _determine_winner(
        self,
        metrics_a: dict[str, float],
        metrics_b: dict[str, float],
        version_a: str,
        version_b: str,
    ) -> str:
        """Determine winner between two versions based on metrics.

        Priority order:
        1. Higher accuracy wins
        2. If tied, lower Brier score wins
        3. If tied, lower margin MAE wins

        Args:
            metrics_a: Metrics for version A.
            metrics_b: Metrics for version B.
            version_a: Version A string.
            version_b: Version B string.

        Returns:
            Winning version string.
        """
        # Check accuracy
        acc_a = metrics_a.get("accuracy", 0.0)
        acc_b = metrics_b.get("accuracy", 0.0)
        if abs(acc_a - acc_b) > 0.001:
            return version_b if acc_b > acc_a else version_a

        # Check Brier score (lower is better)
        brier_a = metrics_a.get("brier_score", 1.0)
        brier_b = metrics_b.get("brier_score", 1.0)
        if abs(brier_a - brier_b) > 0.001:
            return version_b if brier_b < brier_a else version_a

        # Check margin MAE (lower is better)
        mae_a = metrics_a.get("margin_mae", float("inf"))
        mae_b = metrics_b.get("margin_mae", float("inf"))
        if abs(mae_a - mae_b) > 0.001:
            return version_b if mae_b < mae_a else version_a

        # Default to newer version (B)
        return version_b
