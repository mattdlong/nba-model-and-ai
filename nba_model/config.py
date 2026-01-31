"""Configuration management using Pydantic Settings.

This module provides centralized configuration for the NBA model application,
supporting environment variables and .env file loading.

Example:
    >>> from nba_model.config import get_settings
    >>> settings = get_settings()
    >>> print(settings.db_path)
    'data/nba.db'
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables.

    All settings can be overridden via environment variables or .env file.
    Environment variables take precedence over .env file values.

    Attributes:
        db_path: Path to SQLite database file.
        api_delay: Delay between NBA API calls in seconds.
        api_max_retries: Maximum retry attempts for failed API calls.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_dir: Directory for log files.
        model_dir: Directory for saved model weights.
        learning_rate: Training learning rate.
        batch_size: Training batch size.
        kelly_fraction: Fractional Kelly multiplier for bet sizing.
        max_bet_pct: Maximum bet as percentage of bankroll.
        min_edge_pct: Minimum edge required to place bet.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Database
    db_path: str = Field(
        default="data/nba.db",
        alias="NBA_DB_PATH",
        description="Path to SQLite database file",
    )

    # NBA API Configuration
    api_delay: float = Field(
        default=0.6,
        alias="NBA_API_DELAY",
        ge=0.0,
        description="Delay between API calls in seconds",
    )
    api_max_retries: int = Field(
        default=3,
        alias="NBA_API_MAX_RETRIES",
        ge=1,
        le=10,
        description="Maximum retry attempts for failed API calls",
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        alias="LOG_LEVEL",
        description="Logging level",
    )
    log_dir: str = Field(
        default="logs",
        alias="LOG_DIR",
        description="Directory for log files",
    )

    # Model Configuration
    model_dir: str = Field(
        default="data/models",
        alias="MODEL_DIR",
        description="Directory for saved model weights",
    )

    # Training Hyperparameters
    learning_rate: float = Field(
        default=1e-4,
        alias="LEARNING_RATE",
        gt=0.0,
        description="Training learning rate",
    )
    batch_size: int = Field(
        default=32,
        alias="BATCH_SIZE",
        ge=1,
        description="Training batch size",
    )

    # Betting Parameters
    kelly_fraction: float = Field(
        default=0.25,
        alias="KELLY_FRACTION",
        ge=0.0,
        le=1.0,
        description="Fractional Kelly multiplier (0.25 = quarter Kelly)",
    )
    max_bet_pct: float = Field(
        default=0.02,
        alias="MAX_BET_PCT",
        ge=0.0,
        le=1.0,
        description="Maximum bet as percentage of bankroll",
    )
    min_edge_pct: float = Field(
        default=0.02,
        alias="MIN_EDGE_PCT",
        ge=0.0,
        le=1.0,
        description="Minimum edge required to place bet",
    )

    @field_validator("db_path", "log_dir", "model_dir")
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Ensure path strings are valid."""
        if not v or v.isspace():
            raise ValueError("Path cannot be empty or whitespace")
        return v

    @property
    def db_path_obj(self) -> Path:
        """Return database path as Path object."""
        return Path(self.db_path)

    @property
    def model_dir_obj(self) -> Path:
        """Return model directory as Path object."""
        return Path(self.model_dir)

    @property
    def log_dir_obj(self) -> Path:
        """Return log directory as Path object."""
        return Path(self.log_dir)

    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        self.db_path_obj.parent.mkdir(parents=True, exist_ok=True)
        self.model_dir_obj.mkdir(parents=True, exist_ok=True)
        self.log_dir_obj.mkdir(parents=True, exist_ok=True)


# Singleton pattern for settings
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the application settings singleton.

    Returns:
        Settings instance loaded from environment.

    Example:
        >>> settings = get_settings()
        >>> print(settings.api_delay)
        0.6
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset the settings singleton (useful for testing)."""
    global _settings
    _settings = None
