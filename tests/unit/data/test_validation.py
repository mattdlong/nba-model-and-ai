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

    def test_wrong_team_count_returns_error(self, validator: DataValidator) -> None:
        """Should error when stints have wrong team count."""
        from unittest.mock import MagicMock

        session = MagicMock()

        # Create mock stint for only one team
        mock_stint = MagicMock()
        mock_stint.team_id = 1
        mock_stint.start_time = 0
        mock_stint.end_time = 100
        mock_stint.lineup_json = '["1", "2", "3", "4", "5"]'

        session.query.return_value.filter.return_value.all.return_value = [mock_stint]

        result = validator.validate_stints(session, "0022300001")

        assert result.valid is False
        assert any("teams" in e for e in result.errors)

    def test_overlapping_stints_returns_error(self, validator: DataValidator) -> None:
        """Should error when stints overlap."""
        from unittest.mock import MagicMock

        session = MagicMock()

        # Create overlapping stints for same team
        stint1 = MagicMock()
        stint1.team_id = 1
        stint1.start_time = 0
        stint1.end_time = 200  # Overlaps with stint2
        stint1.lineup_json = '["1", "2", "3", "4", "5"]'

        stint2 = MagicMock()
        stint2.team_id = 1
        stint2.start_time = 100  # Starts before stint1 ends
        stint2.end_time = 300
        stint2.lineup_json = '["1", "2", "3", "4", "5"]'

        # Need two teams
        stint3 = MagicMock()
        stint3.team_id = 2
        stint3.start_time = 0
        stint3.end_time = 100
        stint3.lineup_json = '["6", "7", "8", "9", "10"]'

        session.query.return_value.filter.return_value.all.return_value = [
            stint1,
            stint2,
            stint3,
        ]

        result = validator.validate_stints(session, "0022300001")

        assert result.valid is False
        assert any("overlapping" in e for e in result.errors)

    def test_invalid_lineup_size_returns_error(self, validator: DataValidator) -> None:
        """Should error when lineup has wrong number of players."""
        from unittest.mock import MagicMock

        session = MagicMock()

        # Create stint with wrong lineup size
        stint1 = MagicMock()
        stint1.team_id = 1
        stint1.start_time = 0
        stint1.end_time = 100
        stint1.lineup_json = '["1", "2", "3", "4"]'  # Only 4 players

        stint2 = MagicMock()
        stint2.team_id = 2
        stint2.start_time = 0
        stint2.end_time = 100
        stint2.lineup_json = '["6", "7", "8", "9", "10"]'

        session.query.return_value.filter.return_value.all.return_value = [stint1, stint2]

        result = validator.validate_stints(session, "0022300001")

        assert result.valid is False
        assert any("players" in e for e in result.errors)

    def test_invalid_lineup_json_returns_error(self, validator: DataValidator) -> None:
        """Should error when lineup JSON is invalid."""
        from unittest.mock import MagicMock

        session = MagicMock()

        # Create stint with invalid JSON
        stint1 = MagicMock()
        stint1.team_id = 1
        stint1.start_time = 0
        stint1.end_time = 100
        stint1.lineup_json = "not valid json"

        stint2 = MagicMock()
        stint2.team_id = 2
        stint2.start_time = 0
        stint2.end_time = 100
        stint2.lineup_json = '["6", "7", "8", "9", "10"]'

        session.query.return_value.filter.return_value.all.return_value = [stint1, stint2]

        result = validator.validate_stints(session, "0022300001")

        assert result.valid is False
        assert any("invalid lineup JSON" in e for e in result.errors)


class TestValidateReferentialIntegrity:
    """Tests for validate_referential_integrity method."""

    @pytest.fixture
    def validator(self) -> DataValidator:
        """Create a validator instance."""
        return DataValidator()

    def test_valid_data_returns_valid(self, validator: DataValidator) -> None:
        """Should return valid for consistent data."""
        from unittest.mock import MagicMock

        session = MagicMock()
        # All queries return 0 orphans
        session.query.return_value.outerjoin.return_value.filter.return_value.count.return_value = (
            0
        )

        result = validator.validate_referential_integrity(session)

        assert result.valid is True

    def test_orphan_home_teams_returns_error(self, validator: DataValidator) -> None:
        """Should error when games have invalid home_team_id."""
        from unittest.mock import MagicMock

        session = MagicMock()
        # First query returns orphans, rest return 0
        session.query.return_value.outerjoin.return_value.filter.return_value.count.side_effect = [
            5,  # orphan home teams
            0,  # orphan away teams
            0,  # orphan player stats
            0,  # orphan shots
        ]

        result = validator.validate_referential_integrity(session)

        assert result.valid is False
        assert any("home_team_id" in e for e in result.errors)

    def test_orphan_away_teams_returns_error(self, validator: DataValidator) -> None:
        """Should error when games have invalid away_team_id."""
        from unittest.mock import MagicMock

        session = MagicMock()
        session.query.return_value.outerjoin.return_value.filter.return_value.count.side_effect = [
            0,  # orphan home teams
            3,  # orphan away teams
            0,  # orphan player stats
            0,  # orphan shots
        ]

        result = validator.validate_referential_integrity(session)

        assert result.valid is False
        assert any("away_team_id" in e for e in result.errors)

    def test_orphan_player_stats_returns_error(self, validator: DataValidator) -> None:
        """Should error when player stats have invalid player_id."""
        from unittest.mock import MagicMock

        session = MagicMock()
        session.query.return_value.outerjoin.return_value.filter.return_value.count.side_effect = [
            0,  # orphan home teams
            0,  # orphan away teams
            10,  # orphan player stats
            0,  # orphan shots
        ]

        result = validator.validate_referential_integrity(session)

        assert result.valid is False
        assert any("player_game_stats" in e for e in result.errors)

    def test_orphan_shots_returns_error(self, validator: DataValidator) -> None:
        """Should error when shots have invalid player_id."""
        from unittest.mock import MagicMock

        session = MagicMock()
        session.query.return_value.outerjoin.return_value.filter.return_value.count.side_effect = [
            0,  # orphan home teams
            0,  # orphan away teams
            0,  # orphan player stats
            25,  # orphan shots
        ]

        result = validator.validate_referential_integrity(session)

        assert result.valid is False
        assert any("shots" in e for e in result.errors)


class TestValidateSeasonCompletenessExtended:
    """Extended tests for validate_season_completeness method."""

    @pytest.fixture
    def validator(self) -> DataValidator:
        """Create a validator instance."""
        return DataValidator()

    def test_low_game_count_returns_warning(self, validator: DataValidator) -> None:
        """Should warn when season has few games."""
        from unittest.mock import MagicMock

        session = MagicMock()
        # 500 games (low), all with pbp and boxscores
        session.query.return_value.filter.return_value.scalar.side_effect = [500]
        session.query.return_value.join.return_value.filter.return_value.scalar.side_effect = [
            500,
            500,
        ]

        result = validator.validate_season_completeness(session, "2023-24")

        assert any("only" in w for w in result.warnings)

    def test_missing_pbp_returns_warning(self, validator: DataValidator) -> None:
        """Should warn when games are missing play-by-play."""
        from unittest.mock import MagicMock

        session = MagicMock()
        # 1200 games, but only 1000 with pbp
        session.query.return_value.filter.return_value.scalar.return_value = 1200
        session.query.return_value.join.return_value.filter.return_value.scalar.side_effect = [
            1000,  # games with pbp
            1200,  # games with boxscores
        ]

        result = validator.validate_season_completeness(session, "2023-24")

        assert any("missing play-by-play" in w for w in result.warnings)

    def test_missing_boxscores_returns_warning(self, validator: DataValidator) -> None:
        """Should warn when games are missing box scores."""
        from unittest.mock import MagicMock

        session = MagicMock()
        # 1200 games, all with pbp, but only 1150 with boxscores
        session.query.return_value.filter.return_value.scalar.return_value = 1200
        session.query.return_value.join.return_value.filter.return_value.scalar.side_effect = [
            1200,  # games with pbp
            1150,  # games with boxscores
        ]

        result = validator.validate_season_completeness(session, "2023-24")

        assert any("missing box scores" in w for w in result.warnings)


class TestValidateGameCompletenessExtended:
    """Extended tests for validate_game_completeness method."""

    @pytest.fixture
    def validator(self) -> DataValidator:
        """Create a validator instance."""
        return DataValidator()

    def test_high_play_count_returns_warning(self, validator: DataValidator) -> None:
        """Should warn when play count is unusually high."""
        from unittest.mock import MagicMock

        session = MagicMock()
        # Game exists
        mock_game = MagicMock()
        session.query.return_value.filter.return_value.first.return_value = mock_game
        # Very high play count (overtime games)
        session.query.return_value.filter.return_value.scalar.side_effect = [
            800,  # play count (very high)
            2,  # game stats count
            15,  # player stats count
            100,  # shot count
        ]

        result = validator.validate_game_completeness(session, "0022300001")

        assert any("high play count" in w for w in result.warnings)

    def test_missing_box_scores_returns_error(self, validator: DataValidator) -> None:
        """Should error when game is missing team box scores."""
        from unittest.mock import MagicMock

        session = MagicMock()
        mock_game = MagicMock()
        session.query.return_value.filter.return_value.first.return_value = mock_game
        session.query.return_value.filter.return_value.scalar.side_effect = [
            300,  # play count
            1,  # game stats count (only 1 team)
            15,  # player stats count
            100,  # shot count
        ]

        result = validator.validate_game_completeness(session, "0022300001")

        assert result.valid is False
        assert any("missing team box scores" in e for e in result.errors)

    def test_low_player_stats_returns_warning(self, validator: DataValidator) -> None:
        """Should warn when player stats count is low."""
        from unittest.mock import MagicMock

        session = MagicMock()
        mock_game = MagicMock()
        session.query.return_value.filter.return_value.first.return_value = mock_game
        session.query.return_value.filter.return_value.scalar.side_effect = [
            300,  # play count
            2,  # game stats count
            5,  # player stats count (too low)
            100,  # shot count
        ]

        result = validator.validate_game_completeness(session, "0022300001")

        assert any("low player stats count" in w for w in result.warnings)

    def test_no_shots_returns_warning(self, validator: DataValidator) -> None:
        """Should warn when game has no shot data."""
        from unittest.mock import MagicMock

        session = MagicMock()
        mock_game = MagicMock()
        session.query.return_value.filter.return_value.first.return_value = mock_game
        session.query.return_value.filter.return_value.scalar.side_effect = [
            300,  # play count
            2,  # game stats count
            20,  # player stats count
            0,  # shot count (none)
        ]

        result = validator.validate_game_completeness(session, "0022300001")

        assert any("no shot data" in w for w in result.warnings)
