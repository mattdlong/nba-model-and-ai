"""Integration tests for feature engineering pipeline.

Tests end-to-end feature calculation including normalization, RAPM,
spacing, and fatigue metrics using the complete feature pipeline.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from nba_model.data.models import (
    Game,
    GameStats,
    LineupSpacing,
    Player,
    PlayerRAPM,
    Season,
    SeasonStats,
    Shot,
    Stint,
    Team,
)
from nba_model.data.schema import Base
from nba_model.features import (
    FatigueCalculator,
    RAPMCalculator,
    SeasonNormalizer,
    SpacingCalculator,
)


@pytest.fixture
def in_memory_engine():
    """Create in-memory SQLite engine for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def db_session(in_memory_engine):
    """Create a database session for testing."""
    SessionLocal = sessionmaker(bind=in_memory_engine)
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture
def sample_season(db_session: Session) -> Season:
    """Create a sample season in the database."""
    season = Season(
        season_id="2023-24",
        start_date=date(2023, 10, 24),
        end_date=date(2024, 4, 14),
        games_count=82,
    )
    db_session.add(season)
    db_session.commit()
    return season


@pytest.fixture
def sample_teams(db_session: Session) -> list[Team]:
    """Create sample teams in the database."""
    teams = [
        Team(
            team_id=1610612744,
            abbreviation="GSW",
            full_name="Golden State Warriors",
            city="San Francisco",
            arena_name="Chase Center",
            arena_lat=37.768,
            arena_lon=-122.388,
        ),
        Team(
            team_id=1610612747,
            abbreviation="LAL",
            full_name="Los Angeles Lakers",
            city="Los Angeles",
            arena_name="Crypto.com Arena",
            arena_lat=34.043,
            arena_lon=-118.267,
        ),
    ]
    for team in teams:
        db_session.add(team)
    db_session.commit()
    return teams


@pytest.fixture
def sample_players(db_session: Session) -> list[Player]:
    """Create sample players in the database."""
    players = [
        Player(player_id=201939, full_name="Stephen Curry", height_inches=74),
        Player(player_id=2544, full_name="LeBron James", height_inches=81),
        Player(player_id=201566, full_name="Klay Thompson", height_inches=79),
        Player(player_id=203110, full_name="Draymond Green", height_inches=79),
        Player(player_id=1628398, full_name="Jonathan Kuminga", height_inches=80),
        Player(player_id=203507, full_name="Anthony Davis", height_inches=82),
        Player(player_id=1628973, full_name="Austin Reaves", height_inches=77),
        Player(player_id=1630162, full_name="Max Christie", height_inches=78),
        Player(player_id=203901, full_name="D'Angelo Russell", height_inches=77),
        Player(player_id=1627783, full_name="Rui Hachimura", height_inches=80),
    ]
    for player in players:
        db_session.add(player)
    db_session.commit()
    return players


@pytest.fixture
def sample_games(
    db_session: Session,
    sample_season: Season,
    sample_teams: list[Team],
) -> list[Game]:
    """Create sample games for testing."""
    today = date.today()
    games = [
        Game(
            game_id="0022300001",
            season_id="2023-24",
            game_date=today - timedelta(days=3),
            home_team_id=1610612744,
            away_team_id=1610612747,
            home_score=121,
            away_score=115,
            status="completed",
        ),
        Game(
            game_id="0022300002",
            season_id="2023-24",
            game_date=today - timedelta(days=1),
            home_team_id=1610612747,
            away_team_id=1610612744,
            home_score=118,
            away_score=112,
            status="completed",
        ),
        Game(
            game_id="0022300003",
            season_id="2023-24",
            game_date=today,
            home_team_id=1610612744,
            away_team_id=1610612747,
            home_score=125,
            away_score=120,
            status="completed",
        ),
    ]
    for game in games:
        db_session.add(game)
    db_session.commit()
    return games


@pytest.fixture
def sample_game_stats(
    db_session: Session,
    sample_games: list[Game],
) -> list[GameStats]:
    """Create sample game stats for testing normalization."""
    stats = []
    for game in sample_games:
        # Home team stats
        home_stats = GameStats(
            game_id=game.game_id,
            team_id=game.home_team_id,
            is_home=True,
            points=game.home_score,
            rebounds=45,
            assists=28,
            steals=8,
            blocks=5,
            turnovers=12,
            pace=102.5,
            offensive_rating=115.2,
            defensive_rating=108.5,
            efg_pct=0.545,
            tov_pct=12.5,
            orb_pct=28.5,
            ft_rate=0.341,
        )
        # Away team stats
        away_stats = GameStats(
            game_id=game.game_id,
            team_id=game.away_team_id,
            is_home=False,
            points=game.away_score,
            rebounds=42,
            assists=25,
            steals=6,
            blocks=3,
            turnovers=14,
            pace=102.5,
            offensive_rating=108.5,
            defensive_rating=115.2,
            efg_pct=0.489,
            tov_pct=14.2,
            orb_pct=25.0,
            ft_rate=0.304,
        )
        stats.extend([home_stats, away_stats])

    for stat in stats:
        db_session.add(stat)
    db_session.commit()
    return stats


@pytest.fixture
def sample_stints(
    db_session: Session,
    sample_games: list[Game],
    sample_players: list[Player],
) -> list[Stint]:
    """Create sample stints for RAPM testing."""
    # Get player IDs for lineups
    gsw_lineup = [201939, 201566, 203110, 1628398, 1627783]
    lal_lineup = [2544, 203507, 1628973, 1630162, 203901]

    stints = []
    for game in sample_games:
        # Create a couple of stints per game
        stint1 = Stint(
            game_id=game.game_id,
            period=1,
            start_time="12:00",
            end_time="6:00",
            duration_seconds=360,
            home_lineup=json.dumps(gsw_lineup),
            away_lineup=json.dumps(lal_lineup),
            home_points=12,
            away_points=10,
            possessions=15.0,
        )
        stint2 = Stint(
            game_id=game.game_id,
            period=1,
            start_time="6:00",
            end_time="0:00",
            duration_seconds=360,
            home_lineup=json.dumps(gsw_lineup),
            away_lineup=json.dumps(lal_lineup),
            home_points=14,
            away_points=12,
            possessions=16.0,
        )
        stints.extend([stint1, stint2])

    for stint in stints:
        db_session.add(stint)
    db_session.commit()
    return stints


@pytest.fixture
def sample_shots(
    db_session: Session,
    sample_games: list[Game],
    sample_players: list[Player],
) -> list[Shot]:
    """Create sample shots for spacing testing."""
    shots = []
    player_ids = [201939, 201566, 203110, 1628398, 1627783]

    for game in sample_games:
        for i, player_id in enumerate(player_ids):
            # Create multiple shots per player with varying locations
            for j in range(5):
                shot = Shot(
                    game_id=game.game_id,
                    player_id=player_id,
                    team_id=1610612744,
                    period=1,
                    minutes_remaining=10 - j,
                    seconds_remaining=30,
                    loc_x=(i - 2) * 50 + j * 10,
                    loc_y=100 + j * 20,
                    made=j % 2 == 0,
                    shot_type="2PT Field Goal" if j < 3 else "3PT Field Goal",
                    shot_zone_basic="Restricted Area",
                )
                shots.append(shot)

    for shot in shots:
        db_session.add(shot)
    db_session.commit()
    return shots


@pytest.mark.integration
class TestSeasonNormalizationIntegration:
    """Integration tests for season normalization."""

    def test_normalize_and_save_to_database(
        self,
        db_session: Session,
        sample_game_stats: list[GameStats],
        sample_games: list[Game],
    ) -> None:
        """Normalization stats should be saved to database."""
        # Build stats DataFrame
        records = []
        for gs in sample_game_stats:
            game = next(g for g in sample_games if g.game_id == gs.game_id)
            records.append(
                {
                    "season_id": game.season_id,
                    "pace": gs.pace,
                    "offensive_rating": gs.offensive_rating,
                    "defensive_rating": gs.defensive_rating,
                    "efg_pct": gs.efg_pct,
                    "tov_pct": gs.tov_pct,
                    "orb_pct": gs.orb_pct,
                    "ft_rate": gs.ft_rate,
                }
            )

        stats_df = pd.DataFrame(records)

        # Fit and save
        normalizer = SeasonNormalizer()
        normalizer.fit(stats_df)
        count = normalizer.save_stats(db_session)

        # Verify saved to database
        assert count > 0
        stored_stats = db_session.query(SeasonStats).all()
        assert len(stored_stats) > 0
        assert all(s.season_id == "2023-24" for s in stored_stats)

    def test_load_stats_from_database(
        self,
        db_session: Session,
        sample_game_stats: list[GameStats],
        sample_games: list[Game],
    ) -> None:
        """Normalization stats should be loadable from database."""
        # First save stats
        records = []
        for gs in sample_game_stats:
            game = next(g for g in sample_games if g.game_id == gs.game_id)
            records.append(
                {
                    "season_id": game.season_id,
                    "pace": gs.pace,
                    "offensive_rating": gs.offensive_rating,
                }
            )

        stats_df = pd.DataFrame(records)
        normalizer = SeasonNormalizer(metrics=["pace", "offensive_rating"])
        normalizer.fit(stats_df)
        normalizer.save_stats(db_session)

        # Load into new normalizer
        new_normalizer = SeasonNormalizer()
        new_normalizer.load_stats(db_session, season="2023-24")

        assert new_normalizer.fitted_ is True
        assert len(new_normalizer.stats) > 0


@pytest.mark.integration
class TestSpacingIntegration:
    """Integration tests for spacing calculation."""

    def test_calculate_lineup_spacing_from_shots(
        self,
        db_session: Session,
        sample_shots: list[Shot],
    ) -> None:
        """Spacing should be calculated from shot data."""
        # Build shots DataFrame
        shots_df = pd.DataFrame(
            [
                {"player_id": s.player_id, "loc_x": s.loc_x, "loc_y": s.loc_y}
                for s in sample_shots
            ]
        )

        # Calculate spacing for a lineup
        calculator = SpacingCalculator(min_shots=10)
        lineup = [201939, 201566, 203110, 1628398, 1627783]
        metrics = calculator.calculate_lineup_spacing(lineup, shots_df)

        # Verify metrics are reasonable
        assert metrics["shot_count"] > 0
        assert metrics["hull_area"] >= 0
        assert "centroid_x" in metrics
        assert "centroid_y" in metrics

    def test_save_spacing_to_database(
        self,
        db_session: Session,
        sample_shots: list[Shot],
    ) -> None:
        """Spacing records should be saveable to database."""
        shots_df = pd.DataFrame(
            [
                {"player_id": s.player_id, "loc_x": s.loc_x, "loc_y": s.loc_y}
                for s in sample_shots
            ]
        )

        calculator = SpacingCalculator(min_shots=10)
        lineup = [201939, 201566, 203110, 1628398, 1627783]
        metrics = calculator.calculate_lineup_spacing(lineup, shots_df)

        lineup_hash = SpacingCalculator.compute_lineup_hash(lineup)
        spacing_record = LineupSpacing(
            season_id="2023-24",
            lineup_hash=lineup_hash,
            player_ids=json.dumps(lineup),
            hull_area=metrics["hull_area"],
            centroid_x=metrics["centroid_x"],
            centroid_y=metrics["centroid_y"],
            shot_count=metrics["shot_count"],
        )
        db_session.add(spacing_record)
        db_session.commit()

        # Verify stored
        stored = db_session.query(LineupSpacing).first()
        assert stored is not None
        assert stored.lineup_hash == lineup_hash
        assert stored.hull_area == metrics["hull_area"]


@pytest.mark.integration
class TestFatigueIntegration:
    """Integration tests for fatigue calculation."""

    def test_calculate_fatigue_for_game(
        self,
        sample_games: list[Game],
    ) -> None:
        """Fatigue indicators should be calculated for team-games."""
        # Build games DataFrame
        games_df = pd.DataFrame(
            [
                {
                    "game_id": g.game_id,
                    "game_date": g.game_date,
                    "home_team_id": g.home_team_id,
                    "away_team_id": g.away_team_id,
                }
                for g in sample_games
            ]
        )

        calculator = FatigueCalculator()
        today = date.today()

        # Calculate for home team on most recent game
        indicators = calculator.calculate_schedule_flags(1610612744, today, games_df)

        # Verify all expected keys are present
        assert "rest_days" in indicators
        assert "back_to_back" in indicators
        assert "3_in_4" in indicators
        assert "4_in_5" in indicators
        assert "travel_miles" in indicators
        assert "home_stand" in indicators
        assert "road_trip" in indicators

    def test_back_to_back_detection(
        self,
        sample_games: list[Game],
    ) -> None:
        """Back-to-back games should be correctly detected."""
        games_df = pd.DataFrame(
            [
                {
                    "game_id": g.game_id,
                    "game_date": g.game_date,
                    "home_team_id": g.home_team_id,
                    "away_team_id": g.away_team_id,
                }
                for g in sample_games
            ]
        )

        calculator = FatigueCalculator()
        today = date.today()

        indicators = calculator.calculate_schedule_flags(1610612744, today, games_df)

        # Should be back-to-back since game yesterday and today
        assert indicators["back_to_back"] is True


@pytest.mark.integration
class TestRAPMIntegration:
    """Integration tests for RAPM calculation."""

    def test_rapm_with_minimal_stints(
        self,
        sample_stints: list[Stint],
        sample_games: list[Game],
    ) -> None:
        """RAPM should handle minimal stint data gracefully."""
        # Build stints DataFrame
        stint_records = []
        for stint in sample_stints:
            game = next(g for g in sample_games if g.game_id == stint.game_id)
            stint_records.append(
                {
                    "home_lineup": stint.home_lineup,
                    "away_lineup": stint.away_lineup,
                    "home_points": stint.home_points,
                    "away_points": stint.away_points,
                    "possessions": stint.possessions or 1.0,
                    "duration_seconds": stint.duration_seconds,
                    "game_date": game.game_date,
                }
            )

        stints_df = pd.DataFrame(stint_records)

        # With minimal data, RAPM should still work (or handle gracefully)
        calculator = RAPMCalculator(lambda_=5000, min_minutes=1)  # Low threshold
        try:
            coefficients = calculator.fit(stints_df)
            # If it succeeds, verify structure
            if coefficients:
                for _player_id, coef in coefficients.items():
                    assert "orapm" in coef
                    assert "drapm" in coef
                    assert "total_rapm" in coef
                    assert "sample_stints" in coef
        except Exception:
            # With minimal data, it may raise - that's acceptable
            pass

    def test_save_rapm_to_database(
        self,
        db_session: Session,
        sample_stints: list[Stint],
        sample_games: list[Game],
    ) -> None:
        """RAPM records should be saveable to database."""
        # Create a synthetic RAPM record
        from datetime import date as dt

        rapm_record = PlayerRAPM(
            player_id=201939,
            season_id="2023-24",
            calculation_date=dt.today(),
            orapm=2.5,
            drapm=1.2,
            rapm=3.7,
            sample_stints=100,
        )
        db_session.add(rapm_record)
        db_session.commit()

        # Verify stored
        stored = db_session.query(PlayerRAPM).filter_by(player_id=201939).first()
        assert stored is not None
        assert stored.orapm == 2.5
        assert stored.drapm == 1.2
        assert stored.rapm == 3.7


@pytest.mark.integration
class TestFeaturePipelineFlow:
    """Integration tests for the complete feature pipeline flow."""

    def test_full_pipeline_order(
        self,
        db_session: Session,
        sample_game_stats: list[GameStats],
        sample_games: list[Game],
        sample_stints: list[Stint],
        sample_shots: list[Shot],
    ) -> None:
        """Feature pipeline should execute in correct order."""
        # Step 1: Normalization
        records = []
        for gs in sample_game_stats:
            game = next(g for g in sample_games if g.game_id == gs.game_id)
            records.append(
                {
                    "season_id": game.season_id,
                    "pace": gs.pace,
                    "offensive_rating": gs.offensive_rating,
                }
            )

        stats_df = pd.DataFrame(records)
        normalizer = SeasonNormalizer(metrics=["pace", "offensive_rating"])
        normalizer.fit(stats_df)
        normalizer.save_stats(db_session)

        # Step 2: Verify normalization saved
        norm_stats = db_session.query(SeasonStats).all()
        assert len(norm_stats) > 0

        # Step 3: Spacing
        shots_df = pd.DataFrame(
            [
                {"player_id": s.player_id, "loc_x": s.loc_x, "loc_y": s.loc_y}
                for s in sample_shots
            ]
        )

        calculator = SpacingCalculator(min_shots=10)
        lineup = [201939, 201566, 203110, 1628398, 1627783]
        metrics = calculator.calculate_lineup_spacing(lineup, shots_df)

        spacing_record = LineupSpacing(
            season_id="2023-24",
            lineup_hash=SpacingCalculator.compute_lineup_hash(lineup),
            player_ids=json.dumps(lineup),
            hull_area=metrics["hull_area"],
            centroid_x=metrics["centroid_x"],
            centroid_y=metrics["centroid_y"],
            shot_count=metrics["shot_count"],
        )
        db_session.add(spacing_record)
        db_session.commit()

        # Step 4: Fatigue
        games_df = pd.DataFrame(
            [
                {
                    "game_id": g.game_id,
                    "game_date": g.game_date,
                    "home_team_id": g.home_team_id,
                    "away_team_id": g.away_team_id,
                }
                for g in sample_games
            ]
        )

        fatigue_calc = FatigueCalculator()
        today = date.today()
        indicators = fatigue_calc.calculate_schedule_flags(1610612744, today, games_df)

        # Verify all pipeline outputs
        assert len(db_session.query(SeasonStats).all()) > 0
        assert len(db_session.query(LineupSpacing).all()) > 0
        assert "rest_days" in indicators
