"""Tests for data validation utilities.

Tests the DataValidator class for validating game completeness,
stint data, season completeness, and referential integrity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from nba_model.data.validation import DataValidator, ValidationResult


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_default_is_valid(self) -> None:
        """Should be valid by default."""
        result = ValidationResult()
        assert result.valid is True
        assert result.errors == []
        assert result.warnings == []

    def test_add_error_marks_invalid(self) -> None:
        """Should mark as invalid when adding error."""
        result = ValidationResult()
        result.add_error("Test error")

        assert result.valid is False
        assert "Test error" in result.errors

    def test_add_warning_stays_valid(self) -> None:
        """Should stay valid when adding warning."""
        result = ValidationResult()
        result.add_warning("Test warning")

        assert result.valid is True
        assert "Test warning" in result.warnings

    def test_add_multiple_errors(self) -> None:
        """Should accumulate multiple errors."""
        result = ValidationResult()
        result.add_error("Error 1")
        result.add_error("Error 2")

        assert len(result.errors) == 2
        assert result.valid is False

    def test_merge_results(self) -> None:
        """Should merge results correctly."""
        result1 = ValidationResult()
        result1.add_error("Error 1")
        result1.add_warning("Warning 1")

        result2 = ValidationResult()
        result2.add_warning("Warning 2")

        result1.merge(result2)

        assert result1.valid is False
        assert len(result1.errors) == 1
        assert len(result1.warnings) == 2

    def test_merge_invalid_marks_invalid(self) -> None:
        """Should mark as invalid when merging invalid result."""
        result1 = ValidationResult()  # Valid
        result2 = ValidationResult()
        result2.add_error("Error")

        result1.merge(result2)

        assert result1.valid is False
        assert "Error" in result1.errors


class TestDataValidator:
    """Tests for DataValidator class."""

    @pytest.fixture
    def validator(self) -> DataValidator:
        """Create a validator instance."""
        return DataValidator()

    def test_init_creates_logger(self, validator: DataValidator) -> None:
        """Should initialize with a logger."""
        assert validator.logger is not None


class TestValidateBatch:
    """Tests for validate_batch method."""

    @pytest.fixture
    def validator(self) -> DataValidator:
        """Create a validator instance."""
        return DataValidator()

    def test_empty_batch_is_valid(self, validator: DataValidator) -> None:
        """Should validate empty batch."""
        result = validator.validate_batch(
            plays=[],
            shots=[],
            game_stats=[],
            player_game_stats=[],
        )

        assert result.valid is True

    def test_play_missing_game_id(self, validator: DataValidator) -> None:
        """Should error on play without game_id."""

        @dataclass
        class MockPlay:
            game_id: str | None
            event_num: int | None

        play = MockPlay(game_id=None, event_num=1)
        result = validator.validate_batch(
            plays=[play],
            shots=[],
            game_stats=[],
            player_game_stats=[],
        )

        assert result.valid is False
        assert any("Play missing game_id" in e for e in result.errors)

    def test_play_missing_event_num(self, validator: DataValidator) -> None:
        """Should error on play without event_num."""

        @dataclass
        class MockPlay:
            game_id: str | None
            event_num: int | None

        play = MockPlay(game_id="0022300001", event_num=None)
        result = validator.validate_batch(
            plays=[play],
            shots=[],
            game_stats=[],
            player_game_stats=[],
        )

        assert result.valid is False
        assert any("missing event_num" in e for e in result.errors)

    def test_shot_missing_game_id(self, validator: DataValidator) -> None:
        """Should error on shot without game_id."""

        @dataclass
        class MockShot:
            game_id: str | None
            player_id: int | None

        shot = MockShot(game_id=None, player_id=123)
        result = validator.validate_batch(
            plays=[],
            shots=[shot],
            game_stats=[],
            player_game_stats=[],
        )

        assert result.valid is False
        assert any("Shot missing game_id" in e for e in result.errors)

    def test_shot_missing_player_id(self, validator: DataValidator) -> None:
        """Should error on shot without player_id."""

        @dataclass
        class MockShot:
            game_id: str | None
            player_id: int | None

        shot = MockShot(game_id="0022300001", player_id=None)
        result = validator.validate_batch(
            plays=[],
            shots=[shot],
            game_stats=[],
            player_game_stats=[],
        )

        assert result.valid is False
        assert any("missing player_id" in e for e in result.errors)

    def test_game_stats_missing_game_id(self, validator: DataValidator) -> None:
        """Should error on game stats without game_id."""

        @dataclass
        class MockGameStats:
            game_id: str | None
            team_id: int | None

        gs = MockGameStats(game_id=None, team_id=123)
        result = validator.validate_batch(
            plays=[],
            shots=[],
            game_stats=[gs],
            player_game_stats=[],
        )

        assert result.valid is False
        assert any("GameStats missing game_id" in e for e in result.errors)

    def test_game_stats_missing_team_id(self, validator: DataValidator) -> None:
        """Should error on game stats without team_id."""

        @dataclass
        class MockGameStats:
            game_id: str | None
            team_id: int | None

        gs = MockGameStats(game_id="0022300001", team_id=None)
        result = validator.validate_batch(
            plays=[],
            shots=[],
            game_stats=[gs],
            player_game_stats=[],
        )

        assert result.valid is False
        assert any("missing team_id" in e for e in result.errors)

    def test_player_game_stats_missing_game_id(self, validator: DataValidator) -> None:
        """Should error on player game stats without game_id."""

        @dataclass
        class MockPlayerGameStats:
            game_id: str | None
            player_id: int | None

        pgs = MockPlayerGameStats(game_id=None, player_id=123)
        result = validator.validate_batch(
            plays=[],
            shots=[],
            game_stats=[],
            player_game_stats=[pgs],
        )

        assert result.valid is False
        assert any("PlayerGameStats missing game_id" in e for e in result.errors)

    def test_player_game_stats_missing_player_id(self, validator: DataValidator) -> None:
        """Should error on player game stats without player_id."""

        @dataclass
        class MockPlayerGameStats:
            game_id: str | None
            player_id: int | None

        pgs = MockPlayerGameStats(game_id="0022300001", player_id=None)
        result = validator.validate_batch(
            plays=[],
            shots=[],
            game_stats=[],
            player_game_stats=[pgs],
        )

        assert result.valid is False
        assert any("missing player_id" in e for e in result.errors)

    def test_valid_batch(self, validator: DataValidator) -> None:
        """Should pass validation for valid batch."""

        @dataclass
        class MockPlay:
            game_id: str
            event_num: int

        @dataclass
        class MockShot:
            game_id: str
            player_id: int

        @dataclass
        class MockGameStats:
            game_id: str
            team_id: int

        @dataclass
        class MockPlayerGameStats:
            game_id: str
            player_id: int

        result = validator.validate_batch(
            plays=[MockPlay(game_id="0022300001", event_num=1)],
            shots=[MockShot(game_id="0022300001", player_id=123)],
            game_stats=[MockGameStats(game_id="0022300001", team_id=1)],
            player_game_stats=[MockPlayerGameStats(game_id="0022300001", player_id=123)],
        )

        assert result.valid is True
        assert result.errors == []


class TestValidateGameCompleteness:
    """Tests for validate_game_completeness method using mocks."""

    @pytest.fixture
    def validator(self) -> DataValidator:
        """Create a validator instance."""
        return DataValidator()

    def test_missing_game_returns_error(self, validator: DataValidator) -> None:
        """Should error when game is not found."""
        from unittest.mock import MagicMock

        session = MagicMock()
        session.query.return_value.filter.return_value.first.return_value = None

        result = validator.validate_game_completeness(session, "0022300001")

        assert result.valid is False
        assert any("not found" in e for e in result.errors)

    def test_no_plays_returns_error(self, validator: DataValidator) -> None:
        """Should error when game has no plays."""
        from unittest.mock import MagicMock

        session = MagicMock()
        # Game exists
        mock_game = MagicMock()
        session.query.return_value.filter.return_value.first.return_value = mock_game
        # Play count = 0
        session.query.return_value.filter.return_value.scalar.return_value = 0

        result = validator.validate_game_completeness(session, "0022300001")

        assert result.valid is False
        assert any("no play-by-play" in e for e in result.errors)

    def test_low_play_count_returns_warning(self, validator: DataValidator) -> None:
        """Should warn when play count is low."""
        from unittest.mock import MagicMock

        session = MagicMock()
        # Game exists
        mock_game = MagicMock()
        session.query.return_value.filter.return_value.first.return_value = mock_game
        # Low play count
        session.query.return_value.filter.return_value.scalar.side_effect = [50, 2, 10, 100]

        result = validator.validate_game_completeness(session, "0022300001")

        assert any("low play count" in w for w in result.warnings)


class TestValidateSeasonCompleteness:
    """Tests for validate_season_completeness method using mocks."""

    @pytest.fixture
    def validator(self) -> DataValidator:
        """Create a validator instance."""
        return DataValidator()

    def test_no_games_returns_error(self, validator: DataValidator) -> None:
        """Should error when season has no games."""
        from unittest.mock import MagicMock

        session = MagicMock()
        session.query.return_value.filter.return_value.scalar.return_value = 0

        result = validator.validate_season_completeness(session, "2023-24")

        assert result.valid is False
        assert any("no games" in e for e in result.errors)


class TestValidateStints:
    """Tests for validate_stints method using mocks."""

    @pytest.fixture
    def validator(self) -> DataValidator:
        """Create a validator instance."""
        return DataValidator()

    def test_no_stints_returns_warning(self, validator: DataValidator) -> None:
        """Should warn when game has no stints."""
        from unittest.mock import MagicMock

        session = MagicMock()
        session.query.return_value.filter.return_value.all.return_value = []

        result = validator.validate_stints(session, "0022300001")

        assert any("no stint data" in w for w in result.warnings)
