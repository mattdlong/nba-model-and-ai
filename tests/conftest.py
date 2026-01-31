"""Shared pytest fixtures for NBA model tests.

This module contains fixtures used across multiple test modules:
- Database session fixtures (in-memory SQLite)
- Sample data fixtures (games, players, stints)
- Configuration fixtures (test settings)
- Mock fixtures (API client, model registry)

Example:
    def test_something(db_session, sample_games):
        # db_session is an in-memory SQLite session
        # sample_games is a DataFrame of test game data
        pass
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Generator

import pandas as pd
import pytest

from nba_model.config import Settings, reset_settings


# =============================================================================
# Paths
# =============================================================================


@pytest.fixture
def fixtures_dir() -> Path:
    """Return path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Create and return temporary data directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "models").mkdir()
    return data_dir


# =============================================================================
# Configuration
# =============================================================================


@pytest.fixture
def test_settings(tmp_data_dir: Path) -> Generator[Settings, None, None]:
    """Provide test settings with temporary directories.

    Automatically resets settings singleton after test.
    """
    import os

    # Set environment variables for test
    os.environ["NBA_DB_PATH"] = str(tmp_data_dir / "test.db")
    os.environ["MODEL_DIR"] = str(tmp_data_dir / "models")
    os.environ["LOG_DIR"] = str(tmp_data_dir / "logs")
    os.environ["LOG_LEVEL"] = "DEBUG"

    reset_settings()
    from nba_model.config import get_settings

    settings = get_settings()
    settings.ensure_directories()

    yield settings

    # Cleanup
    reset_settings()
    for key in ["NBA_DB_PATH", "MODEL_DIR", "LOG_DIR", "LOG_LEVEL"]:
        os.environ.pop(key, None)


# =============================================================================
# Sample Data
# =============================================================================


@pytest.fixture
def sample_game_ids() -> list[str]:
    """Return sample NBA game IDs."""
    return [
        "0022300001",
        "0022300002",
        "0022300003",
    ]


@pytest.fixture
def sample_player_ids() -> list[int]:
    """Return sample NBA player IDs."""
    return [
        203507,  # Giannis Antetokounmpo
        201566,  # Russell Westbrook
        203954,  # Joel Embiid
        1628369,  # Jayson Tatum
        201142,  # Kevin Durant
    ]


@pytest.fixture
def sample_team_ids() -> list[int]:
    """Return sample NBA team IDs."""
    return [
        1610612738,  # Boston Celtics
        1610612747,  # Los Angeles Lakers
        1610612744,  # Golden State Warriors
        1610612748,  # Miami Heat
        1610612749,  # Milwaukee Bucks
    ]


@pytest.fixture
def sample_games_df() -> pd.DataFrame:
    """Return sample games DataFrame."""
    return pd.DataFrame(
        {
            "game_id": ["0022300001", "0022300002", "0022300003"],
            "season_id": ["2023-24", "2023-24", "2023-24"],
            "game_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "home_team_id": [1610612738, 1610612747, 1610612744],
            "away_team_id": [1610612749, 1610612748, 1610612738],
            "home_score": [112, 105, 118],
            "away_score": [108, 110, 115],
            "status": ["completed", "completed", "completed"],
        }
    )


@pytest.fixture
def sample_stints_df() -> pd.DataFrame:
    """Return sample stints DataFrame for RAPM testing."""
    return pd.DataFrame(
        {
            "stint_id": list(range(1, 101)),
            "game_id": ["0022300001"] * 100,
            "home_lineup": [[1, 2, 3, 4, 5]] * 100,
            "away_lineup": [[6, 7, 8, 9, 10]] * 100,
            "point_diff_per_100": [float(i % 10 - 5) for i in range(100)],
            "possessions": [10.0] * 100,
            "game_date": pd.to_datetime(["2024-01-01"] * 100),
        }
    )


@pytest.fixture
def sample_shots_df() -> pd.DataFrame:
    """Return sample shots DataFrame."""
    import numpy as np

    rng = np.random.default_rng(42)
    n_shots = 100

    return pd.DataFrame(
        {
            "game_id": ["0022300001"] * n_shots,
            "player_id": rng.choice([1, 2, 3, 4, 5], n_shots),
            "team_id": [1610612738] * n_shots,
            "loc_x": rng.integers(-250, 250, n_shots),
            "loc_y": rng.integers(-50, 300, n_shots),
            "shot_made": rng.choice([True, False], n_shots),
            "shot_type": rng.choice(["2PT", "3PT"], n_shots),
        }
    )


@pytest.fixture
def sample_prediction_result() -> dict[str, Any]:
    """Return sample prediction result."""
    return {
        "game_id": "0022300001",
        "home_win_prob": 0.58,
        "predicted_margin": 4.5,
        "predicted_total": 218.5,
        "confidence": 0.72,
        "model_version": "v1.0.0",
    }


@pytest.fixture
def sample_betting_signal() -> dict[str, Any]:
    """Return sample betting signal."""
    return {
        "game_id": "0022300001",
        "bet_type": "moneyline",
        "side": "home",
        "model_prob": 0.58,
        "market_prob": 0.52,
        "edge": 0.06,
        "kelly_fraction": 0.025,
        "recommended_stake_pct": 0.00625,
        "confidence": "medium",
    }


# =============================================================================
# Time Fixtures
# =============================================================================


@pytest.fixture
def frozen_now() -> datetime:
    """Return a fixed datetime for deterministic tests."""
    return datetime(2024, 1, 15, 12, 0, 0)


@pytest.fixture
def frozen_today() -> date:
    """Return a fixed date for deterministic tests."""
    return date(2024, 1, 15)


# =============================================================================
# Marker Helpers
# =============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
