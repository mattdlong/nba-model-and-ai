"""Checkpoint management for resumable data collection pipelines.

This module provides checkpoint persistence for data collection pipelines,
enabling resumable operations after interruption.

Example:
    >>> from nba_model.data.checkpoint import CheckpointManager, Checkpoint
    >>> manager = CheckpointManager()
    >>> checkpoint = Checkpoint(
    ...     pipeline_name="historical_load",
    ...     last_game_id="0022300100",
    ...     last_season="2023-24",
    ...     total_processed=100,
    ...     status="running",
    ... )
    >>> manager.save(checkpoint)
    >>> loaded = manager.load("historical_load")
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from nba_model.config import get_settings

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


@dataclass
class Checkpoint:
    """Represents pipeline progress checkpoint.

    Attributes:
        pipeline_name: Unique identifier for the pipeline.
        last_game_id: Last successfully processed game ID.
        last_season: Last season being processed.
        total_processed: Total number of games processed.
        status: Current status ('running', 'completed', 'failed', 'paused').
        last_updated: Timestamp of last update.
        error_message: Error message if status is 'failed'.
        metadata: Additional metadata dict.
    """

    pipeline_name: str
    last_game_id: str | None = None
    last_season: str | None = None
    total_processed: int = 0
    status: str = "running"
    last_updated: datetime = field(default_factory=datetime.now)
    error_message: str | None = None
    metadata: dict | None = None

    def to_dict(self) -> dict:
        """Convert checkpoint to dictionary for serialization."""
        data = asdict(self)
        # Convert datetime to ISO format string
        data["last_updated"] = self.last_updated.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Checkpoint:
        """Create checkpoint from dictionary.

        Args:
            data: Dictionary with checkpoint fields.

        Returns:
            Checkpoint instance.
        """
        # Convert ISO string back to datetime
        if isinstance(data.get("last_updated"), str):
            data["last_updated"] = datetime.fromisoformat(data["last_updated"])
        return cls(**data)


class CheckpointManager:
    """Manages pipeline checkpoints for resumable data collection.

    Stores checkpoints as JSON files. Each pipeline has its own checkpoint
    file, enabling multiple pipelines to run independently.

    Attributes:
        storage_path: Path to checkpoint storage directory.
    """

    def __init__(
        self,
        storage_path: Path | None = None,
        session: Session | None = None,
    ) -> None:
        """Initialize checkpoint manager.

        Args:
            storage_path: Path for file-based storage.
                Defaults to data/checkpoints/ from settings.
            session: Database session (reserved for future DB-based storage).
        """
        if storage_path is None:
            settings = get_settings()
            storage_path = settings.data_dir / "checkpoints"

        self.storage_path = Path(storage_path)
        self._session = session  # Reserved for future use

        # Ensure directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)

        logger.debug(f"CheckpointManager initialized: {self.storage_path}")

    def _get_checkpoint_path(self, pipeline_name: str) -> Path:
        """Get path to checkpoint file for a pipeline.

        Args:
            pipeline_name: Name of the pipeline.

        Returns:
            Path to the checkpoint JSON file.
        """
        # Sanitize pipeline name for filesystem
        safe_name = pipeline_name.replace("/", "_").replace("\\", "_")
        return self.storage_path / f"{safe_name}.json"

    def save(self, checkpoint: Checkpoint) -> None:
        """Save checkpoint state to file.

        Args:
            checkpoint: Checkpoint to save.
        """
        checkpoint.last_updated = datetime.now()
        path = self._get_checkpoint_path(checkpoint.pipeline_name)

        try:
            with open(path, "w") as f:
                json.dump(checkpoint.to_dict(), f, indent=2)
            logger.debug(
                f"Saved checkpoint for {checkpoint.pipeline_name}: "
                f"{checkpoint.total_processed} processed"
            )
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise

    def load(self, pipeline_name: str) -> Checkpoint | None:
        """Load checkpoint for a pipeline.

        Args:
            pipeline_name: Name of the pipeline.

        Returns:
            Checkpoint if found, None otherwise.
        """
        path = self._get_checkpoint_path(pipeline_name)

        if not path.exists():
            logger.debug(f"No checkpoint found for {pipeline_name}")
            return None

        try:
            with open(path) as f:
                data = json.load(f)
            checkpoint = Checkpoint.from_dict(data)
            logger.debug(
                f"Loaded checkpoint for {pipeline_name}: "
                f"{checkpoint.total_processed} processed, status={checkpoint.status}"
            )
            return checkpoint
        except Exception as e:
            logger.error(f"Failed to load checkpoint for {pipeline_name}: {e}")
            return None

    def clear(self, pipeline_name: str) -> None:
        """Clear checkpoint for a pipeline (for fresh start).

        Args:
            pipeline_name: Name of the pipeline.
        """
        path = self._get_checkpoint_path(pipeline_name)

        if path.exists():
            try:
                path.unlink()
                logger.info(f"Cleared checkpoint for {pipeline_name}")
            except Exception as e:
                logger.error(f"Failed to clear checkpoint for {pipeline_name}: {e}")
                raise
        else:
            logger.debug(f"No checkpoint to clear for {pipeline_name}")

    def list_all(self) -> list[Checkpoint]:
        """List all checkpoints.

        Returns:
            List of all checkpoints.
        """
        checkpoints = []

        if not self.storage_path.exists():
            return checkpoints

        for path in self.storage_path.glob("*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
                checkpoints.append(Checkpoint.from_dict(data))
            except Exception as e:
                logger.warning(f"Failed to load checkpoint from {path}: {e}")

        # Sort by last_updated descending
        checkpoints.sort(key=lambda c: c.last_updated, reverse=True)

        return checkpoints

    def update_status(
        self,
        pipeline_name: str,
        status: str,
        error_message: str | None = None,
    ) -> None:
        """Update checkpoint status.

        Args:
            pipeline_name: Name of the pipeline.
            status: New status.
            error_message: Optional error message for failed status.
        """
        checkpoint = self.load(pipeline_name)
        if checkpoint is None:
            checkpoint = Checkpoint(pipeline_name=pipeline_name)

        checkpoint.status = status
        checkpoint.error_message = error_message
        self.save(checkpoint)
