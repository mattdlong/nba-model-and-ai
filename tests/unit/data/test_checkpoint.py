"""Tests for checkpoint management.

Tests the CheckpointManager class for save/load/clear operations.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from nba_model.data.checkpoint import Checkpoint, CheckpointManager


@pytest.fixture
def temp_checkpoint_dir(tmp_path: Path) -> Path:
    """Create a temporary checkpoint directory."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir


@pytest.fixture
def checkpoint_manager(temp_checkpoint_dir: Path) -> CheckpointManager:
    """Create a checkpoint manager with temp storage."""
    return CheckpointManager(storage_path=temp_checkpoint_dir)


class TestCheckpointDataclass:
    """Tests for Checkpoint dataclass."""

    def test_create_checkpoint_minimal(self) -> None:
        """Should create checkpoint with minimal fields."""
        cp = Checkpoint(pipeline_name="test_pipeline")

        assert cp.pipeline_name == "test_pipeline"
        assert cp.last_game_id is None
        assert cp.last_season is None
        assert cp.total_processed == 0
        assert cp.status == "running"
        assert cp.error_message is None

    def test_create_checkpoint_full(self) -> None:
        """Should create checkpoint with all fields."""
        now = datetime.now()
        cp = Checkpoint(
            pipeline_name="test_pipeline",
            last_game_id="0022300001",
            last_season="2023-24",
            total_processed=100,
            status="completed",
            last_updated=now,
            error_message=None,
            metadata={"key": "value"},
        )

        assert cp.pipeline_name == "test_pipeline"
        assert cp.last_game_id == "0022300001"
        assert cp.last_season == "2023-24"
        assert cp.total_processed == 100
        assert cp.status == "completed"
        assert cp.metadata == {"key": "value"}

    def test_checkpoint_to_dict(self) -> None:
        """Should convert checkpoint to dict."""
        cp = Checkpoint(
            pipeline_name="test_pipeline",
            total_processed=50,
        )
        data = cp.to_dict()

        assert data["pipeline_name"] == "test_pipeline"
        assert data["total_processed"] == 50
        assert isinstance(data["last_updated"], str)

    def test_checkpoint_from_dict(self) -> None:
        """Should create checkpoint from dict."""
        data = {
            "pipeline_name": "test_pipeline",
            "last_game_id": "0022300001",
            "last_season": "2023-24",
            "total_processed": 100,
            "status": "completed",
            "last_updated": "2024-01-15T10:30:00",
            "error_message": None,
            "metadata": None,
        }

        cp = Checkpoint.from_dict(data)

        assert cp.pipeline_name == "test_pipeline"
        assert cp.last_game_id == "0022300001"
        assert cp.total_processed == 100
        assert isinstance(cp.last_updated, datetime)


class TestCheckpointManager:
    """Tests for CheckpointManager."""

    def test_init_creates_directory(self, tmp_path: Path) -> None:
        """Should create storage directory if it doesn't exist."""
        storage = tmp_path / "new_checkpoints"
        assert not storage.exists()

        CheckpointManager(storage_path=storage)

        assert storage.exists()

    def test_save_and_load_checkpoint(
        self, checkpoint_manager: CheckpointManager
    ) -> None:
        """Should save and load checkpoint."""
        cp = Checkpoint(
            pipeline_name="test_pipeline",
            last_game_id="0022300001",
            total_processed=100,
            status="running",
        )

        checkpoint_manager.save(cp)
        loaded = checkpoint_manager.load("test_pipeline")

        assert loaded is not None
        assert loaded.pipeline_name == "test_pipeline"
        assert loaded.last_game_id == "0022300001"
        assert loaded.total_processed == 100

    def test_load_nonexistent_returns_none(
        self, checkpoint_manager: CheckpointManager
    ) -> None:
        """Should return None for nonexistent checkpoint."""
        loaded = checkpoint_manager.load("nonexistent_pipeline")
        assert loaded is None

    def test_clear_checkpoint(self, checkpoint_manager: CheckpointManager) -> None:
        """Should clear checkpoint."""
        cp = Checkpoint(pipeline_name="test_pipeline")
        checkpoint_manager.save(cp)

        # Verify it exists
        assert checkpoint_manager.load("test_pipeline") is not None

        # Clear it
        checkpoint_manager.clear("test_pipeline")

        # Verify it's gone
        assert checkpoint_manager.load("test_pipeline") is None

    def test_clear_nonexistent_no_error(
        self, checkpoint_manager: CheckpointManager
    ) -> None:
        """Should not error when clearing nonexistent checkpoint."""
        # Should not raise
        checkpoint_manager.clear("nonexistent_pipeline")

    def test_list_all_checkpoints(
        self, checkpoint_manager: CheckpointManager
    ) -> None:
        """Should list all checkpoints."""
        cp1 = Checkpoint(pipeline_name="pipeline_1", total_processed=100)
        cp2 = Checkpoint(pipeline_name="pipeline_2", total_processed=200)

        checkpoint_manager.save(cp1)
        checkpoint_manager.save(cp2)

        all_checkpoints = checkpoint_manager.list_all()

        assert len(all_checkpoints) == 2
        names = {cp.pipeline_name for cp in all_checkpoints}
        assert "pipeline_1" in names
        assert "pipeline_2" in names

    def test_list_all_empty(self, checkpoint_manager: CheckpointManager) -> None:
        """Should return empty list when no checkpoints."""
        all_checkpoints = checkpoint_manager.list_all()
        assert all_checkpoints == []

    def test_update_status(self, checkpoint_manager: CheckpointManager) -> None:
        """Should update checkpoint status."""
        cp = Checkpoint(pipeline_name="test_pipeline", status="running")
        checkpoint_manager.save(cp)

        checkpoint_manager.update_status("test_pipeline", "completed")

        loaded = checkpoint_manager.load("test_pipeline")
        assert loaded is not None
        assert loaded.status == "completed"

    def test_update_status_with_error(
        self, checkpoint_manager: CheckpointManager
    ) -> None:
        """Should update status with error message."""
        cp = Checkpoint(pipeline_name="test_pipeline", status="running")
        checkpoint_manager.save(cp)

        checkpoint_manager.update_status(
            "test_pipeline", "failed", error_message="API timeout"
        )

        loaded = checkpoint_manager.load("test_pipeline")
        assert loaded is not None
        assert loaded.status == "failed"
        assert loaded.error_message == "API timeout"

    def test_save_updates_timestamp(
        self, checkpoint_manager: CheckpointManager
    ) -> None:
        """Should update last_updated on save."""
        cp = Checkpoint(pipeline_name="test_pipeline")
        original_time = cp.last_updated

        checkpoint_manager.save(cp)

        loaded = checkpoint_manager.load("test_pipeline")
        assert loaded is not None
        assert loaded.last_updated >= original_time

    def test_pipeline_name_sanitization(
        self, checkpoint_manager: CheckpointManager
    ) -> None:
        """Should sanitize pipeline names for filesystem."""
        # Names with special characters
        cp = Checkpoint(pipeline_name="test/pipeline\\name")
        checkpoint_manager.save(cp)

        loaded = checkpoint_manager.load("test/pipeline\\name")
        assert loaded is not None
        assert loaded.pipeline_name == "test/pipeline\\name"
