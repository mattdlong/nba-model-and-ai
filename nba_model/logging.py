"""Logging configuration using Loguru.

This module provides centralized logging setup for the NBA model application.
Supports both console and file output with rotation and structured JSON format.

Example:
    >>> from nba_model.logging import setup_logging, get_logger
    >>> setup_logging(level="DEBUG")
    >>> logger = get_logger(__name__)
    >>> logger.info("Processing game {}", game_id)

Status Tags:
    >>> from nba_model.logging import SUCCESS, FAIL, WARN
    >>> logger.info(f"{SUCCESS} Game 0022300001 collected")
    >>> logger.warning(f"{WARN} Missing shots data for game 0022300001")
    >>> logger.error(f"{FAIL} Game 0022300001 failed to collect")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from loguru import logger

# Color-coded status tags for terminal output
# These use ANSI escape codes that work with loguru's colorize=True
SUCCESS = "\033[92m[SUCCESS]\033[0m"  # Green
FAIL = "\033[91m[FAIL]\033[0m"        # Red
WARN = "\033[93m[WARN]\033[0m"        # Yellow


class InterceptHandler(logging.Handler):
    """Handler to intercept stdlib logging and redirect to loguru.

    This ensures that all logging (both loguru and stdlib) goes through
    loguru's formatting and output handlers.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record by forwarding to loguru."""
        # Get corresponding loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


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
    Also intercepts stdlib logging to route through loguru.

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

    # Intercept stdlib logging and route to loguru
    # This ensures all logging (both loguru and stdlib) uses the same output
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)


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
__all__ = ["get_logger", "logger", "setup_logging", "SUCCESS", "FAIL", "WARN"]
