"""Unit tests for fatigue calculator.

Tests cover:
- Haversine distance calculations
- Back-to-back detection
- 3-in-4 and 4-in-5 flag logic
- Rest days calculation
- Home stand / road trip counting
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from nba_model.features.fatigue import (
    ARENA_COORDS,
    TEAM_ID_TO_ABBREV,
    FatigueCalculator,
    PlayerLoadMetrics,
    calculate_haversine_distance,
)


class TestFatigueCalculator:
    """Tests for FatigueCalculator class."""

    @pytest.fixture
    def calculator(self) -> FatigueCalculator:
        """Create fatigue calculator."""
        return FatigueCalculator()

    @pytest.fixture
    def sample_games_df(self) -> pd.DataFrame:
        """Create sample games DataFrame."""
        today = date.today()
        return pd.DataFrame(
            {
                "game_id": ["001", "002", "003", "004", "005"],
                "game_date": pd.to_datetime(
                    [
                        today - timedelta(days=5),
                        today - timedelta(days=3),
                        today - timedelta(days=2),
                        today - timedelta(days=1),
                        today,
                    ]
                ),
                "home_team_id": [
                    1610612738,
                    1610612747,
                    1610612738,
                    1610612744,
                    1610612738,
                ],
                "away_team_id": [
                    1610612749,
                    1610612738,
                    1610612744,
                    1610612738,
                    1610612749,
                ],
            }
        )

    def test_haversine_between_known_cities(self) -> None:
        """Haversine distance should match expected for known city pairs."""
        # Boston to Los Angeles ~ 2600 miles
        boston = ARENA_COORDS["BOS"]
        la = ARENA_COORDS["LAL"]

        distance = calculate_haversine_distance(boston, la)

        assert distance == pytest.approx(2600, rel=0.05)

    def test_haversine_same_location_is_zero(self) -> None:
        """Distance to same location should be zero."""
        boston = ARENA_COORDS["BOS"]

        distance = calculate_haversine_distance(boston, boston)

        assert distance == pytest.approx(0, abs=0.1)

    def test_rest_days_for_back_to_back(
        self,
        calculator: FatigueCalculator,
        sample_games_df: pd.DataFrame,
    ) -> None:
        """Rest days should be 0 for back-to-back game."""
        today = date.today()
        # Team 1610612738 (BOS) played yesterday and today
        rest = calculator.calculate_rest_days(1610612738, today, sample_games_df)

        # Yesterday was game 4, today is game 5 - back to back
        assert rest == 0

    def test_rest_days_calculation(
        self,
        calculator: FatigueCalculator,
        sample_games_df: pd.DataFrame,
    ) -> None:
        """Rest days should be calculated correctly."""
        today = date.today()
        # Team 1610612749 (MIL) played 5 days ago and today
        # Their games are on day -5 and day 0
        # Last game before today was 5 days ago, so rest = 5 - 1 = 4 days
        rest = calculator.calculate_rest_days(1610612749, today, sample_games_df)

        assert rest == 4  # 5 days ago - 1 (game day doesn't count)

    def test_back_to_back_flag(
        self,
        calculator: FatigueCalculator,
        sample_games_df: pd.DataFrame,
    ) -> None:
        """Back-to-back flag should be True when played yesterday."""
        today = date.today()
        indicators = calculator.calculate_schedule_flags(
            1610612738, today, sample_games_df
        )

        assert indicators["back_to_back"] is True

    def test_three_in_four_detection(
        self,
        calculator: FatigueCalculator,
    ) -> None:
        """3-in-4 should be True when third game in four nights."""
        today = date.today()
        games_df = pd.DataFrame(
            {
                "game_id": ["001", "002", "003"],
                "game_date": pd.to_datetime(
                    [
                        today - timedelta(days=3),
                        today - timedelta(days=1),
                        today,
                    ]
                ),
                "home_team_id": [1610612738, 1610612738, 1610612738],
                "away_team_id": [1610612749, 1610612749, 1610612749],
            }
        )

        indicators = calculator.calculate_schedule_flags(1610612738, today, games_df)

        assert indicators["three_in_four"] is True

    def test_four_in_five_detection(
        self,
        calculator: FatigueCalculator,
    ) -> None:
        """4-in-5 should be True when fourth game in five nights."""
        today = date.today()
        games_df = pd.DataFrame(
            {
                "game_id": ["001", "002", "003", "004"],
                "game_date": pd.to_datetime(
                    [
                        today - timedelta(days=4),
                        today - timedelta(days=2),
                        today - timedelta(days=1),
                        today,
                    ]
                ),
                "home_team_id": [1610612738, 1610612738, 1610612738, 1610612738],
                "away_team_id": [1610612749, 1610612749, 1610612749, 1610612749],
            }
        )

        indicators = calculator.calculate_schedule_flags(1610612738, today, games_df)

        assert indicators["four_in_five"] is True

    def test_not_three_in_four_when_only_two_games(
        self,
        calculator: FatigueCalculator,
    ) -> None:
        """3-in-4 should be False when only two games in window."""
        today = date.today()
        games_df = pd.DataFrame(
            {
                "game_id": ["001", "002"],
                "game_date": pd.to_datetime(
                    [
                        today - timedelta(days=2),
                        today,
                    ]
                ),
                "home_team_id": [1610612738, 1610612738],
                "away_team_id": [1610612749, 1610612749],
            }
        )

        indicators = calculator.calculate_schedule_flags(1610612738, today, games_df)

        assert indicators["three_in_four"] is False

    def test_home_stand_count(
        self,
        calculator: FatigueCalculator,
    ) -> None:
        """Home stand should count consecutive home games."""
        today = date.today()
        games_df = pd.DataFrame(
            {
                "game_id": ["001", "002", "003"],
                "game_date": pd.to_datetime(
                    [
                        today - timedelta(days=2),
                        today - timedelta(days=1),
                        today,
                    ]
                ),
                # Team 1610612738 is home in all three
                "home_team_id": [1610612738, 1610612738, 1610612738],
                "away_team_id": [1610612749, 1610612747, 1610612744],
            }
        )

        indicators = calculator.calculate_schedule_flags(1610612738, today, games_df)

        assert indicators["home_stand"] == 3
        assert indicators["road_trip"] == 0

    def test_road_trip_count(
        self,
        calculator: FatigueCalculator,
    ) -> None:
        """Road trip should count consecutive away games."""
        today = date.today()
        games_df = pd.DataFrame(
            {
                "game_id": ["001", "002", "003"],
                "game_date": pd.to_datetime(
                    [
                        today - timedelta(days=2),
                        today - timedelta(days=1),
                        today,
                    ]
                ),
                # Team 1610612738 is away in all three
                "home_team_id": [1610612749, 1610612747, 1610612744],
                "away_team_id": [1610612738, 1610612738, 1610612738],
            }
        )

        indicators = calculator.calculate_schedule_flags(1610612738, today, games_df)

        assert indicators["road_trip"] == 3
        assert indicators["home_stand"] == 0

    def test_travel_distance_home_game_is_zero(
        self,
        calculator: FatigueCalculator,
    ) -> None:
        """Travel for home games only should be zero."""
        today = date.today()
        games_df = pd.DataFrame(
            {
                "game_id": ["001", "002"],
                "game_date": pd.to_datetime(
                    [
                        today - timedelta(days=2),
                        today,
                    ]
                ),
                "home_team_id": [1610612738, 1610612738],  # BOS at home
                "away_team_id": [1610612749, 1610612744],
            }
        )

        distance = calculator.calculate_travel_distance(1610612738, today, games_df)

        assert distance == 0.0

    def test_travel_distance_road_trip(
        self,
        calculator: FatigueCalculator,
    ) -> None:
        """Travel distance should accumulate on road trips."""
        today = date.today()
        # BOS plays at LAL then at GSW
        games_df = pd.DataFrame(
            {
                "game_id": ["001", "002"],
                "game_date": pd.to_datetime(
                    [
                        today - timedelta(days=2),
                        today,
                    ]
                ),
                "home_team_id": [1610612747, 1610612744],  # LAL, GSW
                "away_team_id": [1610612738, 1610612738],  # BOS away
            }
        )

        distance = calculator.calculate_travel_distance(
            1610612738, today, games_df, lookback_days=7
        )

        # LA to SF ~ 350 miles
        assert distance > 300  # Should have some travel

    def test_player_load_metrics(
        self,
        calculator: FatigueCalculator,
    ) -> None:
        """Player load should calculate correctly from stats."""
        today = date.today()
        # Games sorted ascending by date (oldest to newest)
        # After sorting descending (most recent first) and fitting:
        # [38, 36, 34, 32, 30] with x = [0, 1, 2, 3, 4]
        # This gives a negative slope (decreasing minutes)
        player_stats_df = pd.DataFrame(
            {
                "player_id": [101] * 5,
                "game_date": pd.to_datetime(
                    [today - timedelta(days=i) for i in range(5, 0, -1)]
                ),
                "minutes": [30, 32, 34, 36, 38],
                "distance_miles": [2.0, 2.1, 2.2, 2.3, 2.4],
            }
        )

        metrics = calculator.calculate_player_load(
            101, today, player_stats_df, lookback_games=5
        )

        assert isinstance(metrics, PlayerLoadMetrics)
        assert metrics.avg_minutes == pytest.approx(34.0, rel=0.01)
        assert metrics.total_distance_miles == pytest.approx(11.0, rel=0.01)
        # Minutes trend is computed on descending-sorted data (most recent first)
        # So [38, 36, 34, 32, 30] = negative slope (decreasing trend)
        assert metrics.minutes_trend < 0

    def test_player_load_no_games_returns_zeros(
        self,
        calculator: FatigueCalculator,
    ) -> None:
        """Player load with no prior games should return zeros."""
        today = date.today()
        player_stats_df = pd.DataFrame(
            {
                "player_id": [102],  # Different player
                "game_date": pd.to_datetime([today - timedelta(days=1)]),
                "minutes": [30],
                "distance_miles": [2.0],
            }
        )

        metrics = calculator.calculate_player_load(101, today, player_stats_df)

        assert metrics.avg_minutes == 0.0
        assert metrics.total_distance_miles == 0.0
        assert metrics.minutes_trend == 0.0

    def test_arena_coords_completeness(self) -> None:
        """All 30 NBA teams should have arena coordinates."""
        assert len(ARENA_COORDS) == 30

        for abbrev, (lat, lon) in ARENA_COORDS.items():
            # Verify coordinates are in valid ranges
            assert -90 <= lat <= 90, f"Invalid latitude for {abbrev}"
            assert -180 <= lon <= 180, f"Invalid longitude for {abbrev}"

    def test_team_id_mapping_completeness(self) -> None:
        """All 30 NBA teams should be in ID mapping."""
        assert len(TEAM_ID_TO_ABBREV) == 30

        # Verify all mapped abbreviations exist in arena coords
        for _team_id, abbrev in TEAM_ID_TO_ABBREV.items():
            assert abbrev in ARENA_COORDS, f"Missing coords for {abbrev}"
