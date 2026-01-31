"""Tests for stint derivation logic.

Tests the StintDeriver class for deriving lineup stints
from play-by-play substitution events.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from nba_model.data.stints import (
    EVENT_FIELD_GOAL_MADE,
    EVENT_FIELD_GOAL_MISSED,
    EVENT_FREE_THROW,
    EVENT_JUMP_BALL,
    EVENT_PERIOD_END,
    EVENT_PERIOD_START,
    EVENT_REBOUND,
    EVENT_SUBSTITUTION,
    EVENT_TURNOVER,
    LineupChange,
    PERIOD_MINUTES,
    OT_MINUTES,
    StintData,
    StintDeriver,
)


class TestLineupChange:
    """Tests for LineupChange dataclass."""

    def test_create_lineup_change(self) -> None:
        """Should create a lineup change record."""
        change = LineupChange(
            event_num=100,
            period=1,
            pc_time="6:30",
            team_id=1610612744,
            player_in=201939,
            player_out=201566,
            game_seconds=330,
        )

        assert change.event_num == 100
        assert change.period == 1
        assert change.pc_time == "6:30"
        assert change.team_id == 1610612744
        assert change.player_in == 201939
        assert change.player_out == 201566
        assert change.game_seconds == 330

    def test_default_game_seconds(self) -> None:
        """Should default game_seconds to 0."""
        change = LineupChange(
            event_num=1,
            period=1,
            pc_time="12:00",
            team_id=1,
            player_in=1,
            player_out=2,
        )

        assert change.game_seconds == 0


class TestStintData:
    """Tests for StintData dataclass."""

    def test_create_stint_data(self) -> None:
        """Should create stint data with all fields."""
        stint = StintData(
            game_id="0022300001",
            team_id=1610612744,
            lineup=[1, 2, 3, 4, 5],
            start_event_num=1,
            end_event_num=50,
            start_time=0,
            end_time=300,
            home_points=10,
            away_points=8,
            possessions=12.5,
        )

        assert stint.game_id == "0022300001"
        assert stint.team_id == 1610612744
        assert stint.lineup == [1, 2, 3, 4, 5]
        assert stint.start_time == 0
        assert stint.end_time == 300
        assert stint.home_points == 10
        assert stint.away_points == 8
        assert stint.possessions == 12.5

    def test_default_values(self) -> None:
        """Should have default values for points and possessions."""
        stint = StintData(
            game_id="0022300001",
            team_id=1,
            lineup=[1, 2, 3, 4, 5],
            start_event_num=1,
            end_event_num=10,
            start_time=0,
            end_time=100,
        )

        assert stint.home_points == 0
        assert stint.away_points == 0
        assert stint.possessions == 0.0


class TestStintDeriver:
    """Tests for StintDeriver class."""

    @pytest.fixture
    def deriver(self) -> StintDeriver:
        """Create a stint deriver instance."""
        return StintDeriver()

    def test_init_creates_logger(self, deriver: StintDeriver) -> None:
        """Should initialize with a logger."""
        assert deriver.logger is not None


class TestParseTimeToSeconds:
    """Tests for _parse_time_to_seconds method."""

    @pytest.fixture
    def deriver(self) -> StintDeriver:
        """Create a stint deriver instance."""
        return StintDeriver()

    def test_period_1_start(self, deriver: StintDeriver) -> None:
        """Should return 0 seconds at period 1 start."""
        seconds = deriver._parse_time_to_seconds(1, "12:00")
        assert seconds == 0

    def test_period_1_end(self, deriver: StintDeriver) -> None:
        """Should return 720 seconds at period 1 end."""
        seconds = deriver._parse_time_to_seconds(1, "0:00")
        assert seconds == 720  # 12 minutes

    def test_period_2_start(self, deriver: StintDeriver) -> None:
        """Should return 720 seconds at period 2 start."""
        seconds = deriver._parse_time_to_seconds(2, "12:00")
        assert seconds == 720  # After period 1

    def test_halftime(self, deriver: StintDeriver) -> None:
        """Should return 1440 seconds at halftime."""
        seconds = deriver._parse_time_to_seconds(2, "0:00")
        assert seconds == 1440  # 24 minutes

    def test_period_3_midway(self, deriver: StintDeriver) -> None:
        """Should calculate mid-period time correctly."""
        seconds = deriver._parse_time_to_seconds(3, "6:00")
        # After periods 1-2 (1440s) + 6 minutes into period 3 (360s)
        assert seconds == 1440 + 360

    def test_overtime_period(self, deriver: StintDeriver) -> None:
        """Should calculate OT time correctly."""
        # OT starts at period 5
        seconds = deriver._parse_time_to_seconds(5, "5:00")
        # After 4 regulation periods (4 * 720 = 2880s)
        assert seconds == 2880  # Start of OT

    def test_overtime_midway(self, deriver: StintDeriver) -> None:
        """Should calculate OT midway correctly."""
        seconds = deriver._parse_time_to_seconds(5, "2:30")
        # After 4 periods (2880s) + 2.5 minutes into OT (150s)
        assert seconds == 2880 + 150

    def test_double_overtime(self, deriver: StintDeriver) -> None:
        """Should calculate double OT correctly."""
        seconds = deriver._parse_time_to_seconds(6, "5:00")
        # After 4 periods + 1 OT
        assert seconds == 2880 + 300  # 3180

    def test_invalid_time_format(self, deriver: StintDeriver) -> None:
        """Should handle invalid time format gracefully."""
        seconds = deriver._parse_time_to_seconds(1, "invalid")
        # Should treat as 0:00 (end of period)
        assert seconds == 720

    def test_empty_time(self, deriver: StintDeriver) -> None:
        """Should handle empty time gracefully."""
        seconds = deriver._parse_time_to_seconds(1, "")
        assert seconds == 720


class TestInferTeamIds:
    """Tests for _infer_team_ids method."""

    @pytest.fixture
    def deriver(self) -> StintDeriver:
        """Create a stint deriver instance."""
        return StintDeriver()

    def test_infer_from_plays(self, deriver: StintDeriver) -> None:
        """Should infer team IDs from plays."""

        @dataclass
        class MockPlay:
            player1_team_id: int | None

        plays = [
            MockPlay(player1_team_id=1610612744),  # GSW
            MockPlay(player1_team_id=1610612747),  # LAL
            MockPlay(player1_team_id=1610612744),
        ]

        home_id, away_id = deriver._infer_team_ids(plays)

        assert home_id is not None
        assert away_id is not None
        assert {home_id, away_id} == {1610612744, 1610612747}

    def test_no_teams_found(self, deriver: StintDeriver) -> None:
        """Should return None when no teams found."""

        @dataclass
        class MockPlay:
            player1_team_id: int | None

        plays = [
            MockPlay(player1_team_id=None),
            MockPlay(player1_team_id=0),
        ]

        home_id, away_id = deriver._infer_team_ids(plays)

        assert home_id is None
        assert away_id is None

    def test_only_one_team(self, deriver: StintDeriver) -> None:
        """Should return None when only one team found."""

        @dataclass
        class MockPlay:
            player1_team_id: int | None

        plays = [
            MockPlay(player1_team_id=1610612744),
            MockPlay(player1_team_id=1610612744),
        ]

        home_id, away_id = deriver._infer_team_ids(plays)

        assert home_id is None
        assert away_id is None


class TestEstimatePossessions:
    """Tests for _estimate_possessions method."""

    @pytest.fixture
    def deriver(self) -> StintDeriver:
        """Create a stint deriver instance."""
        return StintDeriver()

    def test_empty_plays(self, deriver: StintDeriver) -> None:
        """Should return 0 for empty plays."""
        possessions = deriver._estimate_possessions([], 1, 100)
        assert possessions == 0.0

    def test_field_goal_counts(self, deriver: StintDeriver) -> None:
        """Should count field goals toward possessions."""

        @dataclass
        class MockPlay:
            event_num: int
            event_type: int
            home_description: str | None = None
            visitor_description: str | None = None

        plays = [
            MockPlay(event_num=1, event_type=EVENT_FIELD_GOAL_MADE),
            MockPlay(event_num=2, event_type=EVENT_FIELD_GOAL_MISSED),
        ]

        possessions = deriver._estimate_possessions(plays, 0, 10)
        assert possessions == 2.0  # 2 FGA

    def test_free_throws_weighted(self, deriver: StintDeriver) -> None:
        """Should weight free throws at 0.44."""

        @dataclass
        class MockPlay:
            event_num: int
            event_type: int
            home_description: str | None = None
            visitor_description: str | None = None

        plays = [
            MockPlay(event_num=1, event_type=EVENT_FREE_THROW),
            MockPlay(event_num=2, event_type=EVENT_FREE_THROW),
        ]

        possessions = deriver._estimate_possessions(plays, 0, 10)
        assert possessions == pytest.approx(0.88)  # 2 * 0.44

    def test_turnovers_counted(self, deriver: StintDeriver) -> None:
        """Should count turnovers toward possessions."""

        @dataclass
        class MockPlay:
            event_num: int
            event_type: int
            home_description: str | None = None
            visitor_description: str | None = None

        plays = [
            MockPlay(event_num=1, event_type=EVENT_TURNOVER),
        ]

        possessions = deriver._estimate_possessions(plays, 0, 10)
        assert possessions == 1.0

    def test_offensive_rebounds_subtracted(self, deriver: StintDeriver) -> None:
        """Should subtract offensive rebounds from possessions."""

        @dataclass
        class MockPlay:
            event_num: int
            event_type: int
            home_description: str | None = None
            visitor_description: str | None = None

        plays = [
            MockPlay(event_num=1, event_type=EVENT_FIELD_GOAL_MISSED),
            MockPlay(event_num=2, event_type=EVENT_REBOUND, home_description="OFFENSIVE REBOUND"),
        ]

        possessions = deriver._estimate_possessions(plays, 0, 10)
        assert possessions == 0.0  # 1 FGA - 1 OREB

    def test_filters_by_event_num_range(self, deriver: StintDeriver) -> None:
        """Should only count plays in range."""

        @dataclass
        class MockPlay:
            event_num: int
            event_type: int
            home_description: str | None = None
            visitor_description: str | None = None

        plays = [
            MockPlay(event_num=1, event_type=EVENT_FIELD_GOAL_MADE),  # Outside
            MockPlay(event_num=5, event_type=EVENT_FIELD_GOAL_MADE),  # Inside
            MockPlay(event_num=10, event_type=EVENT_FIELD_GOAL_MADE),  # Outside
        ]

        possessions = deriver._estimate_possessions(plays, 3, 8)
        assert possessions == 1.0

    def test_minimum_zero_possessions(self, deriver: StintDeriver) -> None:
        """Should not return negative possessions."""

        @dataclass
        class MockPlay:
            event_num: int
            event_type: int
            home_description: str | None = None
            visitor_description: str | None = None

        # Many offensive rebounds, no shots
        plays = [
            MockPlay(event_num=1, event_type=EVENT_REBOUND, home_description="OFF"),
            MockPlay(event_num=2, event_type=EVENT_REBOUND, home_description="OFF"),
        ]

        possessions = deriver._estimate_possessions(plays, 0, 10)
        assert possessions == 0.0  # max(0, -2)


class TestDeriveStints:
    """Tests for derive_stints method."""

    @pytest.fixture
    def deriver(self) -> StintDeriver:
        """Create a stint deriver instance."""
        return StintDeriver()

    def test_empty_plays(self, deriver: StintDeriver) -> None:
        """Should return empty list for no plays."""
        stints = deriver.derive_stints([], "0022300001")
        assert stints == []

    def test_no_team_ids_found(self, deriver: StintDeriver) -> None:
        """Should return empty when team IDs cannot be inferred."""

        @dataclass
        class MockPlay:
            event_num: int
            period: int
            event_type: int
            player1_id: int | None
            player1_team_id: int | None
            player2_id: int | None = None
            pc_time_string: str | None = "12:00"
            home_description: str | None = None
            visitor_description: str | None = None
            score_home: int | None = None
            score_away: int | None = None

        plays = [
            MockPlay(
                event_num=1,
                period=1,
                event_type=EVENT_PERIOD_START,
                player1_id=None,
                player1_team_id=None,
            ),
        ]

        stints = deriver.derive_stints(plays, "0022300001")
        assert stints == []


class TestCalculateStintOutcomes:
    """Tests for _calculate_stint_outcomes method."""

    @pytest.fixture
    def deriver(self) -> StintDeriver:
        """Create a stint deriver instance."""
        return StintDeriver()

    def test_empty_plays_returns_zeros(self, deriver: StintDeriver) -> None:
        """Should return zero values for empty plays."""
        home_pts, away_pts, poss = deriver._calculate_stint_outcomes([], 0, 100)
        assert home_pts == 0
        assert away_pts == 0
        assert poss == 0.0

    def test_counts_two_point_field_goals(self, deriver: StintDeriver) -> None:
        """Should count 2-point field goals."""

        @dataclass
        class MockPlay:
            event_num: int
            event_type: int
            home_description: str | None = None
            visitor_description: str | None = None
            score_home: int | None = None
            score_away: int | None = None

        plays = [
            MockPlay(
                event_num=5,
                event_type=EVENT_FIELD_GOAL_MADE,
                home_description="Player makes layup",
            ),
        ]

        home_pts, away_pts, _ = deriver._calculate_stint_outcomes(plays, 0, 10)
        assert home_pts == 2
        assert away_pts == 0

    def test_counts_three_point_field_goals(self, deriver: StintDeriver) -> None:
        """Should count 3-point field goals."""

        @dataclass
        class MockPlay:
            event_num: int
            event_type: int
            home_description: str | None = None
            visitor_description: str | None = None
            score_home: int | None = None
            score_away: int | None = None

        plays = [
            MockPlay(
                event_num=5,
                event_type=EVENT_FIELD_GOAL_MADE,
                home_description="Player makes 3PT shot",
            ),
        ]

        home_pts, away_pts, _ = deriver._calculate_stint_outcomes(plays, 0, 10)
        assert home_pts == 3
        assert away_pts == 0

    def test_counts_visitor_points(self, deriver: StintDeriver) -> None:
        """Should count visitor team points."""

        @dataclass
        class MockPlay:
            event_num: int
            event_type: int
            home_description: str | None = None
            visitor_description: str | None = None
            score_home: int | None = None
            score_away: int | None = None

        plays = [
            MockPlay(
                event_num=5,
                event_type=EVENT_FIELD_GOAL_MADE,
                visitor_description="Player makes 3PT shot",
            ),
        ]

        home_pts, away_pts, _ = deriver._calculate_stint_outcomes(plays, 0, 10)
        assert home_pts == 0
        assert away_pts == 3

    def test_counts_free_throws(self, deriver: StintDeriver) -> None:
        """Should count made free throws."""

        @dataclass
        class MockPlay:
            event_num: int
            event_type: int
            home_description: str | None = None
            visitor_description: str | None = None
            score_home: int | None = None
            score_away: int | None = None

        plays = [
            MockPlay(
                event_num=5,
                event_type=EVENT_FREE_THROW,
                home_description="Player free throw 1 of 2",
            ),
        ]

        home_pts, away_pts, _ = deriver._calculate_stint_outcomes(plays, 0, 10)
        assert home_pts == 1
        assert away_pts == 0

    def test_does_not_count_missed_free_throws(self, deriver: StintDeriver) -> None:
        """Should not count missed free throws."""

        @dataclass
        class MockPlay:
            event_num: int
            event_type: int
            home_description: str | None = None
            visitor_description: str | None = None
            score_home: int | None = None
            score_away: int | None = None

        plays = [
            MockPlay(
                event_num=5,
                event_type=EVENT_FREE_THROW,
                home_description="Player MISS free throw",
            ),
        ]

        home_pts, away_pts, _ = deriver._calculate_stint_outcomes(plays, 0, 10)
        assert home_pts == 0
        assert away_pts == 0


class TestGetStartingLineups:
    """Tests for _get_starting_lineups method."""

    @pytest.fixture
    def deriver(self) -> StintDeriver:
        """Create a stint deriver instance."""
        return StintDeriver()

    def test_extracts_lineups_from_period_1(self, deriver: StintDeriver) -> None:
        """Should extract starting lineups from period 1 plays."""

        @dataclass
        class MockPlay:
            event_num: int
            period: int
            event_type: int
            player1_id: int | None
            player1_team_id: int | None
            player2_id: int | None = None

        # Create plays with 5 players per team
        plays = [
            MockPlay(event_num=1, period=1, event_type=1, player1_id=101, player1_team_id=1),
            MockPlay(event_num=2, period=1, event_type=1, player1_id=102, player1_team_id=1),
            MockPlay(event_num=3, period=1, event_type=1, player1_id=103, player1_team_id=1),
            MockPlay(event_num=4, period=1, event_type=1, player1_id=104, player1_team_id=1),
            MockPlay(event_num=5, period=1, event_type=1, player1_id=105, player1_team_id=1),
            MockPlay(event_num=6, period=1, event_type=1, player1_id=201, player1_team_id=2),
            MockPlay(event_num=7, period=1, event_type=1, player1_id=202, player1_team_id=2),
            MockPlay(event_num=8, period=1, event_type=1, player1_id=203, player1_team_id=2),
            MockPlay(event_num=9, period=1, event_type=1, player1_id=204, player1_team_id=2),
            MockPlay(event_num=10, period=1, event_type=1, player1_id=205, player1_team_id=2),
        ]

        home_lineup, away_lineup = deriver._get_starting_lineups(plays, 1, 2)

        assert len(home_lineup) == 5
        assert len(away_lineup) == 5
        assert home_lineup == [101, 102, 103, 104, 105]
        assert away_lineup == [201, 202, 203, 204, 205]


class TestTrackSubstitutions:
    """Tests for _track_substitutions method."""

    @pytest.fixture
    def deriver(self) -> StintDeriver:
        """Create a stint deriver instance."""
        return StintDeriver()

    def test_tracks_substitution_events(self, deriver: StintDeriver) -> None:
        """Should track substitution events."""

        @dataclass
        class MockPlay:
            event_num: int
            period: int
            event_type: int
            player1_id: int | None
            player1_team_id: int | None
            player2_id: int | None = None
            pc_time_string: str | None = "6:00"

        plays = [
            MockPlay(
                event_num=100,
                period=1,
                event_type=EVENT_SUBSTITUTION,
                player1_id=106,  # player entering
                player1_team_id=1,
                player2_id=101,  # player leaving
            ),
        ]

        changes = deriver._track_substitutions(
            plays,
            [101, 102, 103, 104, 105],
            [201, 202, 203, 204, 205],
            1,
            2,
        )

        assert len(changes) == 1
        assert changes[0].player_in == 106
        assert changes[0].player_out == 101
        assert changes[0].team_id == 1


class TestLineupToJson:
    """Tests for _lineup_to_json method."""

    @pytest.fixture
    def deriver(self) -> StintDeriver:
        """Create a stint deriver instance."""
        return StintDeriver()

    def test_sorted_output(self, deriver: StintDeriver) -> None:
        """Should return sorted JSON array."""
        result = deriver._lineup_to_json([5, 3, 1, 4, 2])
        assert result == "[1, 2, 3, 4, 5]"

    def test_empty_lineup(self, deriver: StintDeriver) -> None:
        """Should handle empty lineup."""
        result = deriver._lineup_to_json([])
        assert result == "[]"


class TestEventConstants:
    """Tests for event type constants."""

    def test_substitution_is_8(self) -> None:
        """Substitution event type should be 8."""
        assert EVENT_SUBSTITUTION == 8

    def test_period_start_is_12(self) -> None:
        """Period start event type should be 12."""
        assert EVENT_PERIOD_START == 12

    def test_period_end_is_13(self) -> None:
        """Period end event type should be 13."""
        assert EVENT_PERIOD_END == 13

    def test_period_minutes(self) -> None:
        """Regulation period should be 12 minutes."""
        assert PERIOD_MINUTES == 12

    def test_ot_minutes(self) -> None:
        """Overtime period should be 5 minutes."""
        assert OT_MINUTES == 5
