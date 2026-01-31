"""Integration tests for data collection pipeline.

Tests end-to-end data collection, transformation, and storage
using the complete data pipeline with ORM models.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from nba_model.data.collectors.boxscores import BoxScoreCollector
from nba_model.data.collectors.games import GamesCollector
from nba_model.data.collectors.playbyplay import PlayByPlayCollector
from nba_model.data.collectors.players import PlayersCollector
from nba_model.data.collectors.shots import ShotsCollector
from nba_model.data.models import (
    Game,
    GameStats,
    Play,
    Player,
    PlayerGameStats,
    PlayerSeason,
    Season,
    Shot,
    Stint,
    Team,
)
from nba_model.data.schema import Base
from nba_model.data.stints import StintDeriver


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
def mock_api_client():
    """Create a mock NBA API client."""
    return MagicMock()


@pytest.fixture
def sample_season(db_session: Session) -> Season:
    """Create a sample season in the database."""
    season = Season(
        season_id="2023-24",
        start_date=date(2023, 10, 24),
        end_date=date(2024, 4, 14),
        games_count=1230,
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


@pytest.mark.integration
class TestGameCollectionIntegration:
    """Integration tests for game collection."""

    def test_collect_and_store_games(
        self,
        mock_api_client: MagicMock,
        db_session: Session,
        sample_season: Season,
        sample_teams: list[Team],
    ) -> None:
        """Collected games should be stored in database with relationships."""
        # Setup mock response
        mock_api_client.get_league_game_finder.return_value = pd.DataFrame(
            {
                "GAME_ID": ["0022300001", "0022300001"],
                "GAME_DATE": ["2023-10-24", "2023-10-24"],
                "TEAM_ID": [1610612744, 1610612747],
                "MATCHUP": ["GSW vs. LAL", "LAL @ GSW"],
                "PTS": [121, 115],
                "WL": ["W", "L"],
                "PLUS_MINUS": [6, -6],
            }
        )

        collector = GamesCollector(api_client=mock_api_client, db_session=db_session)
        games = collector.collect_season("2023-24")

        # Verify games collected
        assert len(games) == 1
        assert games[0].game_id == "0022300001"
        assert games[0].home_team_id == 1610612744
        assert games[0].away_team_id == 1610612747
        assert games[0].home_score == 121
        assert games[0].away_score == 115
        assert games[0].status == "completed"

        # Store and verify
        for game in games:
            db_session.add(game)
        db_session.commit()

        stored_games = db_session.query(Game).all()
        assert len(stored_games) == 1
        assert stored_games[0].season_id == "2023-24"


@pytest.mark.integration
class TestPlayByPlayIntegration:
    """Integration tests for play-by-play collection."""

    def test_collect_and_store_plays(
        self,
        mock_api_client: MagicMock,
        db_session: Session,
        sample_season: Season,
        sample_teams: list[Team],
        sample_players: list[Player],
    ) -> None:
        """Collected plays should be stored with correct relationships."""
        # Create a game first
        game = Game(
            game_id="0022300001",
            season_id="2023-24",
            game_date=date(2023, 10, 24),
            home_team_id=1610612744,
            away_team_id=1610612747,
            home_score=121,
            away_score=115,
            status="completed",
        )
        db_session.add(game)
        db_session.commit()

        # Setup mock response
        mock_api_client.get_play_by_play.return_value = pd.DataFrame(
            {
                "EVENTNUM": [1, 2, 3],
                "PERIOD": [1, 1, 1],
                "PCTIMESTRING": ["12:00", "11:45", "11:30"],
                "WCTIMESTRING": ["7:00 PM", "7:00 PM", "7:00 PM"],
                "EVENTMSGTYPE": [12, 10, 1],  # Period start, jump ball, FG made
                "EVENTMSGACTIONTYPE": [0, 0, 1],
                "HOMEDESCRIPTION": [None, None, "Curry 25' 3PT"],
                "VISITORDESCRIPTION": [None, None, None],
                "NEUTRALDESCRIPTION": ["Period Start", "Jump Ball", None],
                "SCORE": [None, None, "3 - 0"],
                "PLAYER1_ID": [None, 201939, 201939],
                "PLAYER2_ID": [None, 2544, None],
                "PLAYER3_ID": [None, None, None],
                "PLAYER1_TEAM_ID": [None, 1610612744, 1610612744],
            }
        )

        collector = PlayByPlayCollector(
            api_client=mock_api_client, db_session=db_session
        )
        plays = collector.collect_game("0022300001")

        # Verify plays collected
        assert len(plays) == 3
        assert plays[0].event_type == 12  # Period start
        assert plays[2].home_description == "Curry 25' 3PT"

        # Store and verify
        for play in plays:
            db_session.add(play)
        db_session.commit()

        stored_plays = db_session.query(Play).all()
        assert len(stored_plays) == 3
        assert stored_plays[0].game_id == "0022300001"


@pytest.mark.integration
class TestStintDerivationIntegration:
    """Integration tests for stint derivation from play-by-play."""

    def test_derive_stints_from_plays(
        self,
        db_session: Session,
        sample_season: Season,
        sample_teams: list[Team],
        sample_players: list[Player],
    ) -> None:
        """Stints should be derived from play-by-play with correct ORM fields."""
        # Create a game
        game = Game(
            game_id="0022300001",
            season_id="2023-24",
            game_date=date(2023, 10, 24),
            home_team_id=1610612744,
            away_team_id=1610612747,
            home_score=121,
            away_score=115,
            status="completed",
        )
        db_session.add(game)
        db_session.commit()

        # Create play objects with correct ORM field names
        plays = [
            Play(
                game_id="0022300001",
                event_num=1,
                period=1,
                pc_time="12:00",
                event_type=12,  # Period start
                team_id=None,
                player1_id=None,
            ),
        ]

        # Add plays for starting players (need 5 per team)
        home_players = [201939, 201566, 203110, 1628398, 1627783]
        away_players = [2544, 203507, 1628973, 1630162, 203901]

        event_num = 2
        for pid in home_players:
            plays.append(
                Play(
                    game_id="0022300001",
                    event_num=event_num,
                    period=1,
                    pc_time="11:50",
                    event_type=1,  # FG made
                    team_id=1610612744,
                    player1_id=pid,
                    home_description="Made shot",
                )
            )
            event_num += 1

        for pid in away_players:
            plays.append(
                Play(
                    game_id="0022300001",
                    event_num=event_num,
                    period=1,
                    pc_time="11:40",
                    event_type=1,  # FG made
                    team_id=1610612747,
                    player1_id=pid,
                    away_description="Made shot",
                )
            )
            event_num += 1

        # Add substitution event
        plays.append(
            Play(
                game_id="0022300001",
                event_num=event_num,
                period=1,
                pc_time="6:00",
                event_type=8,  # Substitution
                team_id=1610612744,
                player1_id=1629743,  # Player entering
                player2_id=1628398,  # Player leaving
            )
        )
        event_num += 1

        # Add period end
        plays.append(
            Play(
                game_id="0022300001",
                event_num=event_num,
                period=1,
                pc_time="0:00",
                event_type=13,  # Period end
                team_id=None,
                player1_id=None,
            )
        )

        # Store plays
        for play in plays:
            db_session.add(play)
        db_session.commit()

        # Derive stints
        deriver = StintDeriver()
        stints = deriver.derive_stints(
            plays, "0022300001", home_team_id=1610612744, away_team_id=1610612747
        )

        # Verify stints have correct ORM fields
        assert len(stints) >= 1
        for stint in stints:
            assert hasattr(stint, "period")
            assert hasattr(stint, "start_time")
            assert hasattr(stint, "end_time")
            assert hasattr(stint, "duration_seconds")
            assert hasattr(stint, "home_lineup")
            assert hasattr(stint, "away_lineup")
            assert hasattr(stint, "home_points")
            assert hasattr(stint, "away_points")
            assert hasattr(stint, "possessions")

            # Verify ORM field types
            assert isinstance(stint.period, int)
            assert isinstance(stint.start_time, str)
            assert isinstance(stint.end_time, str)
            assert isinstance(stint.duration_seconds, int)
            assert isinstance(stint.home_lineup, str)  # JSON string
            assert isinstance(stint.away_lineup, str)  # JSON string

        # Store stints and verify
        for stint in stints:
            db_session.add(stint)
        db_session.commit()

        stored_stints = db_session.query(Stint).all()
        assert len(stored_stints) == len(stints)


@pytest.mark.integration
class TestBoxScoreIntegration:
    """Integration tests for box score collection."""

    def test_collect_and_store_box_scores(
        self,
        mock_api_client: MagicMock,
        db_session: Session,
        sample_season: Season,
        sample_teams: list[Team],
        sample_players: list[Player],
    ) -> None:
        """Box scores should be collected and stored with relationships."""
        # Create a game first
        game = Game(
            game_id="0022300001",
            season_id="2023-24",
            game_date=date(2023, 10, 24),
            home_team_id=1610612744,
            away_team_id=1610612747,
            home_score=121,
            away_score=115,
            status="completed",
        )
        db_session.add(game)
        db_session.commit()

        # Setup mock responses
        mock_api_client.get_boxscore_traditional.return_value = (
            pd.DataFrame(
                {
                    "TEAM_ID": [1610612744, 1610612747],
                    "MATCHUP": ["GSW vs. LAL", "LAL @ GSW"],
                    "PTS": [121, 115],
                    "REB": [45, 42],
                    "AST": [28, 25],
                    "STL": [8, 6],
                    "BLK": [5, 3],
                    "TO": [12, 14],
                }
            ),
            pd.DataFrame(
                {
                    "PLAYER_ID": [201939, 2544],
                    "TEAM_ID": [1610612744, 1610612747],
                    "MIN": [35.5, 38.2],
                    "PTS": [32, 28],
                    "REB": [6, 8],
                    "AST": [11, 7],
                    "STL": [2, 1],
                    "BLK": [0, 1],
                    "TO": [3, 4],
                    "FGM": [11, 10],
                    "FGA": [22, 24],
                    "FG3M": [6, 2],
                    "FG3A": [12, 6],
                    "FTM": [4, 6],
                    "FTA": [5, 8],
                    "PLUS_MINUS": [12, -8],
                }
            ),
        )

        mock_api_client.get_boxscore_advanced.return_value = (
            pd.DataFrame(),
            pd.DataFrame(),
        )
        mock_api_client.get_player_tracking.return_value = pd.DataFrame()

        collector = BoxScoreCollector(
            api_client=mock_api_client, db_session=db_session
        )
        game_stats, player_stats = collector.collect_game("0022300001")

        # Verify game stats
        assert len(game_stats) == 2

        # Store and verify
        for gs in game_stats:
            db_session.add(gs)
        for ps in player_stats:
            db_session.add(ps)
        db_session.commit()

        stored_game_stats = db_session.query(GameStats).all()
        stored_player_stats = db_session.query(PlayerGameStats).all()

        assert len(stored_game_stats) == 2
        assert len(stored_player_stats) == 2


@pytest.mark.integration
class TestShotCollectionIntegration:
    """Integration tests for shot collection."""

    def test_collect_and_store_shots(
        self,
        mock_api_client: MagicMock,
        db_session: Session,
        sample_season: Season,
        sample_teams: list[Team],
        sample_players: list[Player],
    ) -> None:
        """Shots should be collected and stored with court coordinates."""
        # Create a game first
        game = Game(
            game_id="0022300001",
            season_id="2023-24",
            game_date=date(2023, 10, 24),
            home_team_id=1610612744,
            away_team_id=1610612747,
            home_score=121,
            away_score=115,
            status="completed",
        )
        db_session.add(game)
        db_session.commit()

        # Setup mock response
        mock_api_client.get_shot_chart.return_value = pd.DataFrame(
            {
                "GAME_ID": ["0022300001", "0022300001"],
                "PLAYER_ID": [201939, 201939],
                "TEAM_ID": [1610612744, 1610612744],
                "PERIOD": [1, 1],
                "MINUTES_REMAINING": [10, 8],
                "SECONDS_REMAINING": [30, 15],
                "LOC_X": [0, 150],
                "LOC_Y": [50, 200],
                "SHOT_MADE_FLAG": [1, 0],
                "SHOT_TYPE": ["2PT Field Goal", "3PT Field Goal"],
                "ACTION_TYPE": ["Layup", "Jump Shot"],
                "SHOT_ZONE_BASIC": ["Restricted Area", "Above the Break 3"],
                "SHOT_ZONE_AREA": ["Center(C)", "Right Side Center(RC)"],
                "SHOT_ZONE_RANGE": ["Less Than 8 ft.", "24+ ft."],
                "SHOT_DISTANCE": [2, 26],
            }
        )

        collector = ShotsCollector(api_client=mock_api_client, db_session=db_session)
        shots = collector.collect_game("0022300001")

        # Verify shots collected
        assert len(shots) == 2
        assert shots[0].made is True
        assert shots[1].made is False
        assert shots[0].loc_x == 0
        assert shots[0].loc_y == 50

        # Store and verify
        for shot in shots:
            db_session.add(shot)
        db_session.commit()

        stored_shots = db_session.query(Shot).all()
        assert len(stored_shots) == 2


@pytest.mark.integration
class TestPlayersCollectionIntegration:
    """Integration tests for player collection."""

    def test_collect_teams(
        self,
        mock_api_client: MagicMock,
        db_session: Session,
    ) -> None:
        """Teams should be collected from static data."""
        collector = PlayersCollector(
            api_client=mock_api_client, db_session=db_session
        )
        teams = collector.collect_teams()

        # Verify all 30 teams
        assert len(teams) == 30

        # Verify team has arena coordinates
        gsw = next(t for t in teams if t.team_id == 1610612744)
        assert gsw.abbreviation == "GSW"
        assert gsw.arena_lat is not None
        assert gsw.arena_lon is not None

        # Store and verify
        for team in teams:
            db_session.add(team)
        db_session.commit()

        stored_teams = db_session.query(Team).all()
        assert len(stored_teams) == 30
