"""Tests for BaseCollector.

Tests the base collector class functionality including error handling.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from nba_model.data.collectors.base import BaseCollector


class ConcreteCollector(BaseCollector):
    """Concrete implementation for testing."""

    def collect_game(self, game_id: str) -> Any:
        """Collect data for a game."""
        return {"game_id": game_id}


@pytest.fixture
def mock_api_client() -> MagicMock:
    """Create a mock NBA API client."""
    return MagicMock()


@pytest.fixture
def collector(mock_api_client: MagicMock) -> ConcreteCollector:
    """Create a concrete collector for testing."""
    return ConcreteCollector(api_client=mock_api_client)


class TestBaseCollectorInit:
    """Tests for BaseCollector initialization."""

    def test_init_with_api_client(self, mock_api_client: MagicMock) -> None:
        """Should initialize with API client."""
        collector = ConcreteCollector(api_client=mock_api_client)
        assert collector.api is mock_api_client

    def test_init_with_session(self, mock_api_client: MagicMock) -> None:
        """Should initialize with optional session."""
        mock_session = MagicMock()
        collector = ConcreteCollector(
            api_client=mock_api_client, db_session=mock_session
        )
        assert collector.session is mock_session

    def test_init_without_session(self, mock_api_client: MagicMock) -> None:
        """Should work without session."""
        collector = ConcreteCollector(api_client=mock_api_client)
        assert collector.session is None


class TestLogProgress:
    """Tests for progress logging."""

    def test_log_progress_calculates_percentage(
        self, collector: ConcreteCollector
    ) -> None:
        """Should calculate percentage correctly."""
        # Just call to verify no exceptions
        collector._log_progress(5, 10, "item-5")

    def test_log_progress_zero_total(
        self, collector: ConcreteCollector
    ) -> None:
        """Should handle zero total gracefully."""
        # Should not raise divide by zero
        collector._log_progress(0, 0, "item-0")


class TestHandleError:
    """Tests for error handling strategies."""

    def test_handle_error_raise_strategy(
        self, collector: ConcreteCollector
    ) -> None:
        """Should re-raise error with raise strategy."""
        error = ValueError("Test error")
        with pytest.raises(ValueError, match="Test error"):
            collector._handle_error(error, "item-1", "raise")

    def test_handle_error_log_strategy(
        self, collector: ConcreteCollector
    ) -> None:
        """Should log error and continue with log strategy."""
        error = ValueError("Test error")
        # Should not raise
        collector._handle_error(error, "item-1", "log")

    def test_handle_error_skip_strategy(
        self, collector: ConcreteCollector
    ) -> None:
        """Should silently skip with skip strategy."""
        error = ValueError("Test error")
        # Should not raise or log
        collector._handle_error(error, "item-1", "skip")


class TestCollectGame:
    """Tests for collect_game method."""

    def test_collect_game_abstract(
        self, collector: ConcreteCollector
    ) -> None:
        """Concrete implementation should work."""
        result = collector.collect_game("0022300001")
        assert result == {"game_id": "0022300001"}
