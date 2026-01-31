"""Tests for logging module."""
from __future__ import annotations

from pathlib import Path

import pytest
from loguru import logger

from nba_model.logging import get_logger, setup_logging


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_logging_default_parameters(self, tmp_path: Path) -> None:
        """setup_logging should work with default parameters."""
        log_dir = tmp_path / "logs"
        setup_logging(log_dir=str(log_dir))

        assert log_dir.exists()

    def test_setup_logging_creates_directory(self, tmp_path: Path) -> None:
        """setup_logging should create log directory if it doesn't exist."""
        log_dir = tmp_path / "nested" / "logs"
        setup_logging(log_dir=str(log_dir))

        assert log_dir.exists()

    def test_setup_logging_accepts_debug_level(self, tmp_path: Path) -> None:
        """setup_logging should accept DEBUG level."""
        log_dir = tmp_path / "logs"
        setup_logging(level="DEBUG", log_dir=str(log_dir))

        assert log_dir.exists()

    def test_setup_logging_accepts_custom_rotation(self, tmp_path: Path) -> None:
        """setup_logging should accept custom rotation setting."""
        log_dir = tmp_path / "logs"
        setup_logging(log_dir=str(log_dir), rotation="100 MB")

        assert log_dir.exists()

    def test_setup_logging_accepts_custom_retention(self, tmp_path: Path) -> None:
        """setup_logging should accept custom retention setting."""
        log_dir = tmp_path / "logs"
        setup_logging(log_dir=str(log_dir), retention="7 days")

        assert log_dir.exists()

    def test_setup_logging_accepts_serialize_false(self, tmp_path: Path) -> None:
        """setup_logging should accept serialize=False."""
        log_dir = tmp_path / "logs"
        setup_logging(log_dir=str(log_dir), serialize=False)

        assert log_dir.exists()

    def test_setup_logging_all_parameters(self, tmp_path: Path) -> None:
        """setup_logging should accept all custom parameters."""
        log_dir = tmp_path / "logs"
        setup_logging(
            level="WARNING",
            log_dir=str(log_dir),
            rotation="500 MB",
            retention="14 days",
            serialize=False,
        )

        assert log_dir.exists()


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_returns_logger(self) -> None:
        """get_logger should return a logger instance."""
        log = get_logger(__name__)

        assert log is not None

    def test_get_logger_with_module_name(self) -> None:
        """get_logger should accept module name."""
        log = get_logger("nba_model.test_module")

        assert log is not None

    def test_get_logger_can_log_info(self, tmp_path: Path) -> None:
        """Logger from get_logger should be able to log."""
        log_dir = tmp_path / "logs"
        setup_logging(log_dir=str(log_dir))
        log = get_logger("test")

        # Should not raise
        log.info("Test message")

    def test_get_logger_can_log_with_formatting(self, tmp_path: Path) -> None:
        """Logger should support message formatting."""
        log_dir = tmp_path / "logs"
        setup_logging(log_dir=str(log_dir))
        log = get_logger("test")

        # Should not raise
        log.info("Processing game {}", "0022300001")


class TestLoggerExports:
    """Tests for module exports."""

    def test_logger_is_exported(self) -> None:
        """Base logger should be exported."""
        from nba_model.logging import logger as exported_logger

        assert exported_logger is logger

    def test_all_exports_available(self) -> None:
        """All expected exports should be available."""
        from nba_model.logging import __all__

        assert "setup_logging" in __all__
        assert "get_logger" in __all__
        assert "logger" in __all__
