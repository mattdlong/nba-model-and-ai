"""Logging configuration using Loguru.

This module provides centralized logging setup for the NBA model application.
Supports both console and file output with rotation and structured JSON format.

Example:
    >>> from nba_model.logging import setup_logging, get_logger
    >>> setup_logging(level="DEBUG")
    >>> logger = get_logger(__name__)
    >>> logger.info("Processing game {}", game_id)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from loguru import logger


def setup_logging(
    level: str = "INFO",
    log_dir: str = "logs",
    rotation: str = "1 day",
    retention: str = "30 days",
    serialize: bool = True,
) -> None:
    """Configure application logging.

    Sets up Loguru with console output and rotating file handlers.
    File logs are written in JSON format for easy parsing.

    Args:
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_dir: Directory for log files.
        rotation: When to rotate log files (e.g., "1 day", "100 MB").
        retention: How long to keep old log files.
        serialize: Whether to write JSON-formatted logs to file.

    Example:
        >>> setup_logging(level="DEBUG", log_dir="logs")
    """
    # Remove default handler
    logger.remove()

    # Console handler with colored output
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File handler with rotation
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_path / "nba_model_{time:YYYY-MM-DD}.log",
        level=level,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        ),
        rotation=rotation,
        retention=retention,
        serialize=serialize,
        enqueue=True,  # Thread-safe
    )


def get_logger(name: str) -> Any:
    """Get a logger instance bound with the given name.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Loguru logger bound with the given name.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Starting process")
    """
    return logger.bind(name=name)


# Export the base logger for direct use
__all__ = ["get_logger", "logger", "setup_logging"]
