"""Tests for configuration module."""

from __future__ import annotations

from pathlib import Path

import pytest

from nba_model.config import Settings, get_settings, reset_settings


class TestSettings:
    """Tests for Settings class."""

    def test_default_values(self) -> None:
        """Settings should have sensible defaults."""
        settings = Settings()

        assert settings.db_path == "data/nba.db"
        assert settings.api_delay == 0.6
        assert settings.api_max_retries == 3
        assert settings.log_level == "INFO"
        assert settings.kelly_fraction == 0.25

    def test_path_properties(self) -> None:
        """Path properties should return Path objects."""
        settings = Settings()

        assert isinstance(settings.db_path_obj, Path)
        assert isinstance(settings.model_dir_obj, Path)
        assert isinstance(settings.log_dir_obj, Path)

    def test_validation_rejects_empty_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Empty paths should be rejected."""
        # pydantic_settings uses aliases as env var names
        monkeypatch.setenv("NBA_DB_PATH", "   ")
        with pytest.raises(ValueError, match="cannot be empty"):
            Settings()

    def test_validation_rejects_invalid_kelly_fraction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Kelly fraction must be between 0 and 1."""
        monkeypatch.setenv("KELLY_FRACTION", "1.5")
        with pytest.raises(ValueError):
            Settings()

    def test_ensure_directories_creates_dirs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ensure_directories should create required directories."""
        # Set env vars for pydantic_settings
        monkeypatch.setenv("NBA_DB_PATH", str(tmp_path / "data" / "test.db"))
        monkeypatch.setenv("MODEL_DIR", str(tmp_path / "models"))
        monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))

        settings = Settings()
        settings.ensure_directories()

        assert (tmp_path / "data").exists()
        assert (tmp_path / "models").exists()
        assert (tmp_path / "logs").exists()


class TestGetSettings:
    """Tests for get_settings function."""

    def setup_method(self) -> None:
        """Reset settings before each test."""
        reset_settings()

    def teardown_method(self) -> None:
        """Reset settings after each test."""
        reset_settings()

    def test_returns_singleton(self) -> None:
        """get_settings should return the same instance."""
        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2

    def test_loads_from_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Settings should load from environment variables."""
        monkeypatch.setenv("NBA_DB_PATH", "custom/path/db.sqlite")
        monkeypatch.setenv("NBA_API_DELAY", "1.5")

        reset_settings()
        settings = get_settings()

        assert settings.db_path == "custom/path/db.sqlite"
        assert settings.api_delay == 1.5


class TestResetSettings:
    """Tests for reset_settings function."""

    def test_reset_clears_singleton(self) -> None:
        """reset_settings should clear the cached instance."""
        settings1 = get_settings()
        reset_settings()
        settings2 = get_settings()

        # They should be equal but not the same object
        assert settings1 is not settings2
        assert settings1.db_path == settings2.db_path
