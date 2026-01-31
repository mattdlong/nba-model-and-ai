"""Unit tests for event parser.

Tests cover:
- Regex pattern matching for turnovers
- Shot type classification
- Shot context extraction
- Shot clock categorization
"""

from __future__ import annotations

import pandas as pd
import pytest

from nba_model.features.parsing import (
    EventParser,
    ShotClockCategory,
    ShotContext,
    ShotType,
    TurnoverType,
    parse_shot_context,
    parse_turnover_type,
)


class TestEventParser:
    """Tests for EventParser class."""

    @pytest.fixture
    def parser(self) -> EventParser:
        """Create event parser."""
        return EventParser()

    # Turnover Type Tests

    def test_bad_pass_is_unforced(self, parser: EventParser) -> None:
        """Bad pass should be classified as unforced."""
        result = parser.parse_turnover_type("Bad Pass Turnover")
        assert result == TurnoverType.UNFORCED

    def test_traveling_is_unforced(self, parser: EventParser) -> None:
        """Traveling should be classified as unforced."""
        result = parser.parse_turnover_type("Traveling Violation")
        assert result == TurnoverType.UNFORCED

    def test_offensive_foul_is_unforced(self, parser: EventParser) -> None:
        """Offensive foul should be classified as unforced."""
        result = parser.parse_turnover_type("Offensive Foul Turnover")
        assert result == TurnoverType.UNFORCED

    def test_lost_ball_is_forced(self, parser: EventParser) -> None:
        """Lost ball should be classified as forced."""
        result = parser.parse_turnover_type("Lost Ball Turnover")
        assert result == TurnoverType.FORCED

    def test_steal_is_forced(self, parser: EventParser) -> None:
        """Steal should be classified as forced."""
        result = parser.parse_turnover_type("STEAL by Player")
        assert result == TurnoverType.FORCED

    def test_shot_clock_violation_is_unforced(self, parser: EventParser) -> None:
        """Shot clock violation should be unforced."""
        result = parser.parse_turnover_type("Shot Clock Violation")
        assert result == TurnoverType.UNFORCED

    def test_empty_description_returns_unknown(self, parser: EventParser) -> None:
        """Empty description should return unknown."""
        result = parser.parse_turnover_type("")
        assert result == TurnoverType.UNKNOWN

        result = parser.parse_turnover_type(None)
        assert result == TurnoverType.UNKNOWN

    # Shot Type Tests

    def test_driving_layup_classified_as_driving(self, parser: EventParser) -> None:
        """Driving Layup should be classified as driving."""
        result = parser.parse_shot_type("LeBron James Driving Layup")
        assert result == ShotType.DRIVING

    def test_pullup_jumper_classified_correctly(self, parser: EventParser) -> None:
        """Pull-up jumper should be classified as pullup."""
        result = parser.parse_shot_type("Stephen Curry Pull-Up Jump Shot")
        assert result == ShotType.PULLUP

    def test_stepback_classified_correctly(self, parser: EventParser) -> None:
        """Step-back should be classified correctly."""
        result = parser.parse_shot_type("James Harden Step Back Jump Shot")
        assert result == ShotType.STEPBACK

    def test_dunk_classified_correctly(self, parser: EventParser) -> None:
        """Dunk should be classified correctly."""
        result = parser.parse_shot_type("Giannis Antetokounmpo Dunk")
        assert result == ShotType.DUNK

    def test_floating_shot_classified_correctly(self, parser: EventParser) -> None:
        """Floater should be classified as floating."""
        result = parser.parse_shot_type("Trae Young Floating Jump Shot")
        assert result == ShotType.FLOATING

    def test_fadeaway_classified_correctly(self, parser: EventParser) -> None:
        """Fadeaway should be classified correctly."""
        result = parser.parse_shot_type("Kevin Durant Fadeaway Jump Shot")
        assert result == ShotType.FADEAWAY

    def test_hook_shot_classified_correctly(self, parser: EventParser) -> None:
        """Hook shot should be classified correctly."""
        result = parser.parse_shot_type("Joel Embiid Hook Shot")
        assert result == ShotType.HOOK

    def test_turnaround_classified_correctly(self, parser: EventParser) -> None:
        """Turnaround should be classified correctly."""
        result = parser.parse_shot_type("DeMar DeRozan Turnaround Jump Shot")
        assert result == ShotType.TURNAROUND

    def test_tip_in_classified_correctly(self, parser: EventParser) -> None:
        """Tip-in should be classified correctly."""
        result = parser.parse_shot_type("Tip Shot Made")
        assert result == ShotType.TIP

    def test_layup_classified_correctly(self, parser: EventParser) -> None:
        """Regular layup should be classified correctly."""
        result = parser.parse_shot_type("Anthony Davis Layup")
        assert result == ShotType.LAYUP

    def test_unknown_shot_returns_other(self, parser: EventParser) -> None:
        """Unknown shot type should return other."""
        result = parser.parse_shot_type("Some Random Shot")
        assert result == ShotType.OTHER

    # Shot Context Tests

    def test_transition_detected(self, parser: EventParser) -> None:
        """Fast break should be detected as transition."""
        context = parser.parse_shot_context("Fast Break Layup")
        assert context.is_transition is True

    def test_contested_shot_detected(self, parser: EventParser) -> None:
        """Contested shot should be detected."""
        context = parser.parse_shot_context("Contested Jump Shot")
        assert context.is_contested is True

    def test_full_shot_context(self, parser: EventParser) -> None:
        """Full shot context should capture all attributes."""
        context = parser.parse_shot_context("Fast Break Contested Driving Layup")

        assert context.shot_type == ShotType.DRIVING
        assert context.is_transition is True
        assert context.is_contested is True

    # Turnover Context Tests

    def test_turnover_context_with_steal(self, parser: EventParser) -> None:
        """Turnover context should include steal information."""
        context = parser.parse_turnover_context(
            "Lost Ball Turnover - STEAL by Marcus Smart"
        )

        assert context.turnover_type == TurnoverType.FORCED
        assert context.is_steal is True
        assert context.handler_error is True

    def test_turnover_context_bad_pass(self, parser: EventParser) -> None:
        """Bad pass context should indicate handler error."""
        context = parser.parse_turnover_context("Bad Pass Turnover")

        assert context.turnover_type == TurnoverType.UNFORCED
        assert context.is_steal is False
        assert context.handler_error is True

    # Shot Clock Category Tests

    def test_shot_clock_early_category(self, parser: EventParser) -> None:
        """Early shot clock (< 8s) should be categorized correctly."""
        # 7.9s used = early
        assert ShotClockCategory.EARLY.value == "early"

    def test_shot_clock_mid_category(self, parser: EventParser) -> None:
        """Mid shot clock (8-16s) should be categorized correctly."""
        # 8.0s used = mid
        # 16.0s used = mid
        assert ShotClockCategory.MID.value == "mid"

    def test_shot_clock_late_category(self, parser: EventParser) -> None:
        """Late shot clock (> 16s) should be categorized correctly."""
        # 16.1s used = late
        assert ShotClockCategory.LATE.value == "late"

    def test_calculate_shot_clock_usage(self, parser: EventParser) -> None:
        """Shot clock usage should be calculated from play-by-play."""
        plays_df = pd.DataFrame(
            {
                "event_type": [4, 1, 4, 2],  # rebound, made shot, rebound, missed shot
                "pc_time": ["10:00", "9:55", "9:50", "9:40"],  # 5s, 10s used
                "period": [1, 1, 1, 1],
            }
        )

        result_df = parser.calculate_shot_clock_usage(plays_df)

        assert "shot_clock_category" in result_df.columns

    # Parsing All Descriptions Tests

    def test_parse_all_descriptions(self, parser: EventParser) -> None:
        """Should add parsed columns to DataFrame."""
        plays_df = pd.DataFrame(
            {
                "home_description": ["Bad Pass Turnover", None],
                "away_description": [None, "Driving Layup"],
                "neutral_description": [None, None],
            }
        )

        result_df = parser.parse_all_descriptions(plays_df)

        assert "turnover_type" in result_df.columns
        assert "shot_type" in result_df.columns
        assert "is_transition" in result_df.columns
        assert "is_contested" in result_df.columns
        assert result_df.loc[0, "turnover_type"] == TurnoverType.UNFORCED.value
        assert result_df.loc[1, "shot_type"] == ShotType.DRIVING.value

    # Convenience Function Tests

    def test_parse_turnover_type_function(self) -> None:
        """Convenience function should work correctly."""
        result = parse_turnover_type("Bad Pass Turnover")
        assert result == TurnoverType.UNFORCED

    def test_parse_shot_context_function(self) -> None:
        """Convenience function should work correctly."""
        context = parse_shot_context("Driving Layup")
        assert isinstance(context, ShotContext)
        assert context.shot_type == ShotType.DRIVING

    # Edge Case Tests

    def test_case_insensitivity(self, parser: EventParser) -> None:
        """Parsing should be case insensitive."""
        assert parser.parse_turnover_type("BAD PASS") == TurnoverType.UNFORCED
        assert parser.parse_turnover_type("bad pass") == TurnoverType.UNFORCED
        assert parser.parse_shot_type("DRIVING LAYUP") == ShotType.DRIVING
        assert parser.parse_shot_type("driving layup") == ShotType.DRIVING

    def test_hyphenated_variants(self, parser: EventParser) -> None:
        """Should handle hyphenated variants."""
        assert parser.parse_shot_type("Pull-Up Jump Shot") == ShotType.PULLUP
        assert parser.parse_shot_type("Pullup Jump Shot") == ShotType.PULLUP
        assert parser.parse_shot_type("Step-Back Three") == ShotType.STEPBACK
        assert parser.parse_shot_type("Stepback Three") == ShotType.STEPBACK
