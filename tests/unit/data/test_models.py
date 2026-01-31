"""Tests for SQLAlchemy ORM model classes.

Tests the model classes including relationships, data integrity,
and JSON array storage patterns.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from typing import TYPE_CHECKING, Generator

import pytest
from sqlalchemy.exc import IntegrityError

from nba_model.config import reset_settings
from nba_model.data import (
    Game,
    GameStats,
    LineupSpacing,
    Odds,
    Play,
    Player,
    PlayerGameStats,
    PlayerRAPM,
    PlayerSeason,
    Season,
    SeasonStats,
    Shot,
    Stint,
    Team,
    init_db,
    reset_engine,
    session_scope,
)

if TYPE_CHECKING:
    from nba_model.config import Settings


@pytest.fixture(autouse=True)
def reset_db_between_tests(test_settings: "Settings") -> Generator[None, None, None]:
    """Reset database engine and initialize before each test."""
    reset_engine()
    init_db()
    yield
    reset_engine()


@pytest.fixture
def sample_data(test_settings: "Settings") -> dict[str, int | str]:
    """Create sample reference data for relationship tests."""
    with session_scope() as session:
        season = Season(
            season_id="2023-24",
            start_date=date(2023, 10, 24),
            end_date=date(2024, 4, 14),
            games_count=1230,
        )
        team_home = Team(
            team_id=1610612738,
            abbreviation="BOS",
            full_name="Boston Celtics",
            city="Boston",
            arena_name="TD Garden",
            arena_lat=42.3662,
            arena_lon=-71.0621,
        )
        team_away = Team(
            team_id=1610612747,
            abbreviation="LAL",
            full_name="Los Angeles Lakers",
            city="Los Angeles",
            arena_name="Crypto.com Arena",
            arena_lat=34.0430,
            arena_lon=-118.2673,
        )
        player1 = Player(
            player_id=1628369,
            full_name="Jayson Tatum",
            height_inches=80,
            weight_lbs=210,
            birth_date=date(1998, 3, 3),
            draft_year=2017,
            draft_round=1,
            draft_number=3,
        )
        player2 = Player(
            player_id=201566,
            full_name="Russell Westbrook",
            height_inches=75,
            weight_lbs=200,
            birth_date=date(1988, 11, 12),
            draft_year=2008,
            draft_round=1,
            draft_number=4,
        )
        session.add_all([season, team_home, team_away, player1, player2])

    return {
        "season_id": "2023-24",
        "home_team_id": 1610612738,
        "away_team_id": 1610612747,
        "player1_id": 1628369,
        "player2_id": 201566,
    }


class TestSeasonModel:
    """Tests for Season model."""

    def test_create_season(self, test_settings: "Settings") -> None:
        """Should create a valid season record."""
        with session_scope() as session:
            season = Season(
                season_id="2023-24",
                start_date=date(2023, 10, 24),
                end_date=date(2024, 4, 14),
                games_count=1230,
            )
            session.add(season)

        with session_scope() as session:
            found = session.query(Season).filter_by(season_id="2023-24").first()
            assert found is not None
            assert found.start_date == date(2023, 10, 24)
            assert found.games_count == 1230

    def test_season_repr(self, test_settings: "Settings") -> None:
        """Season __repr__ should be informative."""
        season = Season(
            season_id="2023-24",
            start_date=date(2023, 10, 24),
            end_date=date(2024, 4, 14),
        )
        assert "2023-24" in repr(season)


class TestTeamModel:
    """Tests for Team model."""

    def test_create_team_with_arena(self, test_settings: "Settings") -> None:
        """Should create team with arena location."""
        with session_scope() as session:
            team = Team(
                team_id=1610612738,
                abbreviation="BOS",
                full_name="Boston Celtics",
                city="Boston",
                arena_name="TD Garden",
                arena_lat=42.3662,
                arena_lon=-71.0621,
            )
            session.add(team)

        with session_scope() as session:
            found = session.query(Team).filter_by(team_id=1610612738).first()
            assert found is not None
            assert found.abbreviation == "BOS"
            assert found.arena_lat == pytest.approx(42.3662)

    def test_team_without_arena_coords(self, test_settings: "Settings") -> None:
        """Should create team without optional arena coordinates."""
        with session_scope() as session:
            team = Team(
                team_id=1,
                abbreviation="TST",
                full_name="Test Team",
                city="Test City",
            )
            session.add(team)

        with session_scope() as session:
            found = session.query(Team).filter_by(team_id=1).first()
            assert found is not None
            assert found.arena_lat is None
            assert found.arena_lon is None


class TestPlayerModel:
    """Tests for Player model."""

    def test_create_player_full_info(self, test_settings: "Settings") -> None:
        """Should create player with all biographical info."""
        with session_scope() as session:
            player = Player(
                player_id=1628369,
                full_name="Jayson Tatum",
                height_inches=80,
                weight_lbs=210,
                birth_date=date(1998, 3, 3),
                draft_year=2017,
                draft_round=1,
                draft_number=3,
            )
            session.add(player)

        with session_scope() as session:
            found = session.query(Player).filter_by(player_id=1628369).first()
            assert found is not None
            assert found.full_name == "Jayson Tatum"
            assert found.draft_number == 3

    def test_create_undrafted_player(self, test_settings: "Settings") -> None:
        """Should create player without draft info (undrafted)."""
        with session_scope() as session:
            player = Player(
                player_id=12345,
                full_name="Undrafted Player",
            )
            session.add(player)

        with session_scope() as session:
            found = session.query(Player).filter_by(player_id=12345).first()
            assert found is not None
            assert found.draft_year is None
            assert found.draft_round is None


class TestGameModel:
    """Tests for Game model and relationships."""

    def test_create_game(
        self, test_settings: "Settings", sample_data: dict[str, int | str]
    ) -> None:
        """Should create a game with all required fields."""
        with session_scope() as session:
            game = Game(
                game_id="0022300001",
                season_id=sample_data["season_id"],
                game_date=date(2024, 1, 1),
                home_team_id=sample_data["home_team_id"],
                away_team_id=sample_data["away_team_id"],
                home_score=115,
                away_score=108,
                status="completed",
                attendance=19156,
            )
            session.add(game)

        with session_scope() as session:
            found = session.query(Game).filter_by(game_id="0022300001").first()
            assert found is not None
            assert found.home_score == 115
            assert found.status == "completed"

    def test_game_season_relationship(
        self, test_settings: "Settings", sample_data: dict[str, int | str]
    ) -> None:
        """Game.season should navigate to Season."""
        with session_scope() as session:
            game = Game(
                game_id="0022300001",
                season_id=sample_data["season_id"],
                game_date=date(2024, 1, 1),
                home_team_id=sample_data["home_team_id"],
                away_team_id=sample_data["away_team_id"],
                status="completed",
            )
            session.add(game)

        with session_scope() as session:
            game = session.query(Game).filter_by(game_id="0022300001").first()
            assert game is not None
            assert game.season is not None
            assert game.season.season_id == "2023-24"

    def test_game_team_relationships(
        self, test_settings: "Settings", sample_data: dict[str, int | str]
    ) -> None:
        """Game should navigate to home and away teams."""
        with session_scope() as session:
            game = Game(
                game_id="0022300001",
                season_id=sample_data["season_id"],
                game_date=date(2024, 1, 1),
                home_team_id=sample_data["home_team_id"],
                away_team_id=sample_data["away_team_id"],
                status="completed",
            )
            session.add(game)

        with session_scope() as session:
            game = session.query(Game).filter_by(game_id="0022300001").first()
            assert game is not None
            assert game.home_team.abbreviation == "BOS"
            assert game.away_team.abbreviation == "LAL"

    def test_season_games_relationship(
        self, test_settings: "Settings", sample_data: dict[str, int | str]
    ) -> None:
        """Season.games should return list of games."""
        with session_scope() as session:
            game1 = Game(
                game_id="0022300001",
                season_id=sample_data["season_id"],
                game_date=date(2024, 1, 1),
                home_team_id=sample_data["home_team_id"],
                away_team_id=sample_data["away_team_id"],
                status="completed",
            )
            game2 = Game(
                game_id="0022300002",
                season_id=sample_data["season_id"],
                game_date=date(2024, 1, 2),
                home_team_id=sample_data["away_team_id"],
                away_team_id=sample_data["home_team_id"],
                status="completed",
            )
            session.add_all([game1, game2])

        with session_scope() as session:
            season = session.query(Season).filter_by(season_id="2023-24").first()
            assert season is not None
            assert len(season.games) == 2


class TestPlayModel:
    """Tests for Play model and relationships."""

    def test_create_play(
        self, test_settings: "Settings", sample_data: dict[str, int | str]
    ) -> None:
        """Should create a play-by-play event."""
        with session_scope() as session:
            game = Game(
                game_id="0022300001",
                season_id=sample_data["season_id"],
                game_date=date(2024, 1, 1),
                home_team_id=sample_data["home_team_id"],
                away_team_id=sample_data["away_team_id"],
                status="completed",
            )
            session.add(game)

        with session_scope() as session:
            play = Play(
                game_id="0022300001",
                event_num=1,
                period=1,
                pc_time="12:00",
                event_type=12,  # Start period
                home_description="Period Start",
            )
            session.add(play)

        with session_scope() as session:
            found = session.query(Play).filter_by(game_id="0022300001").first()
            assert found is not None
            assert found.period == 1
            assert found.pc_time == "12:00"

    def test_game_plays_relationship(
        self, test_settings: "Settings", sample_data: dict[str, int | str]
    ) -> None:
        """Game.plays should return list of plays."""
        with session_scope() as session:
            game = Game(
                game_id="0022300001",
                season_id=sample_data["season_id"],
                game_date=date(2024, 1, 1),
                home_team_id=sample_data["home_team_id"],
                away_team_id=sample_data["away_team_id"],
                status="completed",
            )
            session.add(game)

        with session_scope() as session:
            play1 = Play(
                game_id="0022300001", event_num=1, period=1, event_type=12
            )
            play2 = Play(
                game_id="0022300001", event_num=2, period=1, event_type=10
            )
            session.add_all([play1, play2])

        with session_scope() as session:
            game = session.query(Game).filter_by(game_id="0022300001").first()
            assert game is not None
            assert len(game.plays) == 2


class TestShotModel:
    """Tests for Shot model."""

    def test_create_shot(
        self, test_settings: "Settings", sample_data: dict[str, int | str]
    ) -> None:
        """Should create a shot record with coordinates."""
        with session_scope() as session:
            game = Game(
                game_id="0022300001",
                season_id=sample_data["season_id"],
                game_date=date(2024, 1, 1),
                home_team_id=sample_data["home_team_id"],
                away_team_id=sample_data["away_team_id"],
                status="completed",
            )
            session.add(game)

        with session_scope() as session:
            shot = Shot(
                game_id="0022300001",
                player_id=sample_data["player1_id"],
                team_id=sample_data["home_team_id"],
                period=1,
                minutes_remaining=10,
                seconds_remaining=30,
                action_type="Jump Shot",
                shot_type="2PT",
                shot_zone_basic="Mid-Range",
                shot_zone_area="Center(C)",
                shot_zone_range="16-24 ft.",
                shot_distance=18,
                loc_x=100,
                loc_y=150,
                made=True,
            )
            session.add(shot)

        with session_scope() as session:
            found = session.query(Shot).filter_by(game_id="0022300001").first()
            assert found is not None
            assert found.loc_x == 100
            assert found.made is True


class TestStintModel:
    """Tests for Stint model with JSON lineup storage."""

    def test_create_stint_with_json_lineups(
        self, test_settings: "Settings", sample_data: dict[str, int | str]
    ) -> None:
        """Should store lineup arrays as JSON strings."""
        with session_scope() as session:
            game = Game(
                game_id="0022300001",
                season_id=sample_data["season_id"],
                game_date=date(2024, 1, 1),
                home_team_id=sample_data["home_team_id"],
                away_team_id=sample_data["away_team_id"],
                status="completed",
            )
            session.add(game)

        home_lineup = [1, 2, 3, 4, 5]
        away_lineup = [6, 7, 8, 9, 10]

        with session_scope() as session:
            stint = Stint(
                game_id="0022300001",
                period=1,
                start_time="12:00",
                end_time="10:30",
                duration_seconds=90,
                home_lineup=json.dumps(home_lineup),
                away_lineup=json.dumps(away_lineup),
                home_points=5,
                away_points=3,
                possessions=4.5,
            )
            session.add(stint)

        with session_scope() as session:
            found = session.query(Stint).filter_by(game_id="0022300001").first()
            assert found is not None
            assert json.loads(found.home_lineup) == [1, 2, 3, 4, 5]
            assert json.loads(found.away_lineup) == [6, 7, 8, 9, 10]
            assert found.possessions == pytest.approx(4.5)


class TestOddsModel:
    """Tests for Odds model."""

    def test_create_odds(
        self, test_settings: "Settings", sample_data: dict[str, int | str]
    ) -> None:
        """Should create odds record with all market types."""
        with session_scope() as session:
            game = Game(
                game_id="0022300001",
                season_id=sample_data["season_id"],
                game_date=date(2024, 1, 1),
                home_team_id=sample_data["home_team_id"],
                away_team_id=sample_data["away_team_id"],
                status="scheduled",
            )
            session.add(game)

        with session_scope() as session:
            odds = Odds(
                game_id="0022300001",
                source="pinnacle",
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                home_ml=1.91,
                away_ml=1.95,
                spread_home=-3.5,
                spread_home_odds=1.91,
                spread_away_odds=1.91,
                total=218.5,
                over_odds=1.91,
                under_odds=1.91,
            )
            session.add(odds)

        with session_scope() as session:
            found = session.query(Odds).filter_by(game_id="0022300001").first()
            assert found is not None
            assert found.source == "pinnacle"
            assert found.home_ml == pytest.approx(1.91)
            assert found.total == pytest.approx(218.5)


class TestPlayerRAPMModel:
    """Tests for PlayerRAPM model."""

    def test_create_rapm(
        self, test_settings: "Settings", sample_data: dict[str, int | str]
    ) -> None:
        """Should create RAPM calculation record."""
        with session_scope() as session:
            rapm = PlayerRAPM(
                player_id=sample_data["player1_id"],
                season_id=sample_data["season_id"],
                calculation_date=date(2024, 1, 15),
                orapm=2.5,
                drapm=1.2,
                rapm=3.7,
                sample_stints=500,
            )
            session.add(rapm)

        with session_scope() as session:
            found = (
                session.query(PlayerRAPM)
                .filter_by(player_id=sample_data["player1_id"])
                .first()
            )
            assert found is not None
            assert found.orapm == pytest.approx(2.5)
            assert found.rapm == pytest.approx(3.7)


class TestLineupSpacingModel:
    """Tests for LineupSpacing model with JSON lineup storage."""

    def test_create_lineup_spacing(
        self, test_settings: "Settings", sample_data: dict[str, int | str]
    ) -> None:
        """Should create lineup spacing record."""
        player_ids = [1, 2, 3, 4, 5]
        lineup_hash = "abc123hash"

        with session_scope() as session:
            spacing = LineupSpacing(
                season_id=sample_data["season_id"],
                lineup_hash=lineup_hash,
                player_ids=json.dumps(player_ids),
                hull_area=1500.5,
                centroid_x=50.0,
                centroid_y=100.0,
                shot_count=150,
            )
            session.add(spacing)

        with session_scope() as session:
            found = (
                session.query(LineupSpacing)
                .filter_by(lineup_hash=lineup_hash)
                .first()
            )
            assert found is not None
            assert json.loads(found.player_ids) == [1, 2, 3, 4, 5]
            assert found.hull_area == pytest.approx(1500.5)


class TestSeasonStatsModel:
    """Tests for SeasonStats model."""

    def test_create_season_stats(
        self, test_settings: "Settings", sample_data: dict[str, int | str]
    ) -> None:
        """Should create season normalization stats."""
        with session_scope() as session:
            stats = SeasonStats(
                season_id=sample_data["season_id"],
                metric_name="offensive_rating",
                mean_value=110.5,
                std_value=5.2,
                min_value=95.0,
                max_value=125.0,
            )
            session.add(stats)

        with session_scope() as session:
            found = (
                session.query(SeasonStats)
                .filter_by(metric_name="offensive_rating")
                .first()
            )
            assert found is not None
            assert found.mean_value == pytest.approx(110.5)


class TestDataIntegrity:
    """Tests for data integrity across models."""

    def test_valid_record_insert(
        self, test_settings: "Settings", sample_data: dict[str, int | str]
    ) -> None:
        """Should insert valid records without errors."""
        with session_scope() as session:
            game = Game(
                game_id="0022300001",
                season_id=sample_data["season_id"],
                game_date=date(2024, 1, 1),
                home_team_id=sample_data["home_team_id"],
                away_team_id=sample_data["away_team_id"],
                status="completed",
            )
            session.add(game)

        with session_scope() as session:
            found = session.query(Game).count()
            assert found == 1

    def test_constraint_violations_raise_errors(
        self, test_settings: "Settings", sample_data: dict[str, int | str]
    ) -> None:
        """Constraint violations should raise IntegrityError."""
        # Try to insert with non-existent foreign key
        with pytest.raises(IntegrityError):
            with session_scope() as session:
                game = Game(
                    game_id="0022300001",
                    season_id="INVALID_SEASON",
                    game_date=date(2024, 1, 1),
                    home_team_id=sample_data["home_team_id"],
                    away_team_id=sample_data["away_team_id"],
                    status="completed",
                )
                session.add(game)


class TestModelRepresentations:
    """Tests for model __repr__ methods."""

    def test_all_models_have_repr(
        self, test_settings: "Settings", sample_data: dict[str, int | str]
    ) -> None:
        """All models should have informative __repr__."""
        season = Season(
            season_id="2023-24",
            start_date=date(2023, 10, 24),
            end_date=date(2024, 4, 14),
        )
        assert "Season" in repr(season)
        assert "2023-24" in repr(season)

        team = Team(
            team_id=1,
            abbreviation="TST",
            full_name="Test Team",
            city="Test City",
        )
        assert "Team" in repr(team)
        assert "TST" in repr(team)

        player = Player(player_id=1, full_name="Test Player")
        assert "Player" in repr(player)
        assert "Test Player" in repr(player)
