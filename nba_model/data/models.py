"""SQLAlchemy ORM models for NBA data.

This module defines all database models for storing NBA game data,
player statistics, play-by-play events, and derived features.

Models are organized into categories:
- Core Reference: Season, Team, Player, PlayerSeason
- Game Data: Game, GameStats, PlayerGameStats
- Play-by-Play: Play, Shot
- Derived/Features: Stint, Odds, PlayerRAPM, LineupSpacing, SeasonStats

Example:
    >>> from nba_model.data.models import Game, Team
    >>> from nba_model.data.db import session_scope
    >>> with session_scope() as session:
    ...     game = session.query(Game).first()
    ...     print(game.home_team.full_name)
"""
from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Index, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from nba_model.data.schema import Base, TimestampMixin

if TYPE_CHECKING:
    pass


# =============================================================================
# Core Reference Models
# =============================================================================


class Season(Base):
    """NBA Season reference table.

    Stores season metadata including date ranges and game counts.
    Season ID follows NBA format (e.g., "2023-24").

    Attributes:
        season_id: Primary key in "YYYY-YY" format.
        start_date: First game date of season.
        end_date: Last game date of season.
        games_count: Total number of games in season.
        games: Relationship to all games in this season.
    """

    __tablename__ = "seasons"

    season_id: Mapped[str] = mapped_column(String(7), primary_key=True)
    start_date: Mapped[date] = mapped_column(nullable=False)
    end_date: Mapped[date] = mapped_column(nullable=False)
    games_count: Mapped[int | None] = mapped_column(nullable=True)

    # Relationships
    games: Mapped[list[Game]] = relationship(back_populates="season")

    def __repr__(self) -> str:
        return f"<Season(season_id={self.season_id!r})>"


class Team(Base):
    """NBA Team reference table.

    Stores team information including arena location for travel calculations.

    Attributes:
        team_id: NBA team ID (primary key).
        abbreviation: Three-letter team code (e.g., "LAL", "BOS").
        full_name: Full team name.
        city: Team's city.
        arena_name: Home arena name.
        arena_lat: Arena latitude for travel distance calculations.
        arena_lon: Arena longitude for travel distance calculations.
    """

    __tablename__ = "teams"

    team_id: Mapped[int] = mapped_column(primary_key=True)
    abbreviation: Mapped[str] = mapped_column(String(3), nullable=False)
    full_name: Mapped[str] = mapped_column(String(100), nullable=False)
    city: Mapped[str] = mapped_column(String(50), nullable=False)
    arena_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    arena_lat: Mapped[float | None] = mapped_column(nullable=True)
    arena_lon: Mapped[float | None] = mapped_column(nullable=True)

    def __repr__(self) -> str:
        return f"<Team(team_id={self.team_id}, abbreviation={self.abbreviation!r})>"


class Player(Base):
    """NBA Player reference table.

    Stores player biographical information.

    Attributes:
        player_id: NBA player ID (primary key).
        full_name: Player's full name.
        height_inches: Height in inches.
        weight_lbs: Weight in pounds.
        birth_date: Date of birth.
        draft_year: Year drafted.
        draft_round: Draft round (1 or 2, or None if undrafted).
        draft_number: Overall pick number.
    """

    __tablename__ = "players"

    player_id: Mapped[int] = mapped_column(primary_key=True)
    full_name: Mapped[str] = mapped_column(String(100), nullable=False)
    height_inches: Mapped[int | None] = mapped_column(nullable=True)
    weight_lbs: Mapped[int | None] = mapped_column(nullable=True)
    birth_date: Mapped[date | None] = mapped_column(nullable=True)
    draft_year: Mapped[int | None] = mapped_column(nullable=True)
    draft_round: Mapped[int | None] = mapped_column(nullable=True)
    draft_number: Mapped[int | None] = mapped_column(nullable=True)

    def __repr__(self) -> str:
        return f"<Player(player_id={self.player_id}, full_name={self.full_name!r})>"


class PlayerSeason(Base):
    """Player-Season-Team association table.

    Tracks which team(s) a player played for in each season.
    A player can have multiple entries per season if traded.

    Attributes:
        id: Auto-increment primary key.
        player_id: Foreign key to players.
        season_id: Foreign key to seasons.
        team_id: Foreign key to teams.
        position: Player's position (e.g., "PG", "SF").
        jersey_number: Player's jersey number.
    """

    __tablename__ = "player_seasons"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(
        ForeignKey("players.player_id"), nullable=False
    )
    season_id: Mapped[str] = mapped_column(
        ForeignKey("seasons.season_id"), nullable=False
    )
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.team_id"), nullable=False)
    position: Mapped[str | None] = mapped_column(String(10), nullable=True)
    jersey_number: Mapped[str | None] = mapped_column(String(3), nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "player_id", "season_id", "team_id", name="uq_player_season_team"
        ),
    )

    def __repr__(self) -> str:
        return f"<PlayerSeason(player_id={self.player_id}, season_id={self.season_id!r}, team_id={self.team_id})>"


# =============================================================================
# Game Models
# =============================================================================


class Game(Base):
    """NBA Game table.

    Stores game metadata and final scores.

    Attributes:
        game_id: NBA GAME_ID format (primary key).
        season_id: Foreign key to seasons.
        game_date: Date of the game.
        home_team_id: Foreign key to home team.
        away_team_id: Foreign key to away team.
        home_score: Home team final score.
        away_score: Away team final score.
        status: Game status ('scheduled', 'completed', 'postponed').
        attendance: Game attendance.
        season: Relationship to Season.
        home_team: Relationship to home Team.
        away_team: Relationship to away Team.
        plays: Relationship to Play events.
        shots: Relationship to Shot records.
        stints: Relationship to Stint records.
    """

    __tablename__ = "games"

    game_id: Mapped[str] = mapped_column(String(10), primary_key=True)
    season_id: Mapped[str] = mapped_column(
        ForeignKey("seasons.season_id"), nullable=False
    )
    game_date: Mapped[date] = mapped_column(nullable=False)
    home_team_id: Mapped[int] = mapped_column(
        ForeignKey("teams.team_id"), nullable=False
    )
    away_team_id: Mapped[int] = mapped_column(
        ForeignKey("teams.team_id"), nullable=False
    )
    home_score: Mapped[int | None] = mapped_column(nullable=True)
    away_score: Mapped[int | None] = mapped_column(nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="scheduled")
    attendance: Mapped[int | None] = mapped_column(nullable=True)

    # Relationships
    season: Mapped[Season] = relationship(back_populates="games")
    home_team: Mapped[Team] = relationship(foreign_keys=[home_team_id])
    away_team: Mapped[Team] = relationship(foreign_keys=[away_team_id])
    plays: Mapped[list[Play]] = relationship(back_populates="game")
    shots: Mapped[list[Shot]] = relationship(back_populates="game")
    stints: Mapped[list[Stint]] = relationship(back_populates="game")

    __table_args__ = (
        Index("idx_games_date", "game_date"),
        Index("idx_games_season", "season_id"),
    )

    def __repr__(self) -> str:
        return f"<Game(game_id={self.game_id!r}, game_date={self.game_date})>"


class GameStats(Base, TimestampMixin):
    """Team-level game statistics.

    Stores both basic and advanced statistics for each team in a game.

    Attributes:
        id: Auto-increment primary key.
        game_id: Foreign key to games.
        team_id: Foreign key to teams.
        is_home: True if this is the home team.
        points: Total points scored.
        rebounds: Total rebounds.
        assists: Total assists.
        steals: Total steals.
        blocks: Total blocks.
        turnovers: Total turnovers.
        offensive_rating: Offensive rating (from boxscoreadvancedv2).
        defensive_rating: Defensive rating.
        pace: Game pace.
        efg_pct: Effective field goal percentage.
        tov_pct: Turnover percentage.
        orb_pct: Offensive rebound percentage.
        ft_rate: Free throw rate.
    """

    __tablename__ = "game_stats"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    game_id: Mapped[str] = mapped_column(ForeignKey("games.game_id"), nullable=False)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.team_id"), nullable=False)
    is_home: Mapped[bool] = mapped_column(nullable=False)

    # Basic stats
    points: Mapped[int | None] = mapped_column(nullable=True)
    rebounds: Mapped[int | None] = mapped_column(nullable=True)
    assists: Mapped[int | None] = mapped_column(nullable=True)
    steals: Mapped[int | None] = mapped_column(nullable=True)
    blocks: Mapped[int | None] = mapped_column(nullable=True)
    turnovers: Mapped[int | None] = mapped_column(nullable=True)

    # Advanced stats (from boxscoreadvancedv2)
    offensive_rating: Mapped[float | None] = mapped_column(nullable=True)
    defensive_rating: Mapped[float | None] = mapped_column(nullable=True)
    pace: Mapped[float | None] = mapped_column(nullable=True)
    efg_pct: Mapped[float | None] = mapped_column(nullable=True)
    tov_pct: Mapped[float | None] = mapped_column(nullable=True)
    orb_pct: Mapped[float | None] = mapped_column(nullable=True)
    ft_rate: Mapped[float | None] = mapped_column(nullable=True)

    __table_args__ = (
        UniqueConstraint("game_id", "team_id", name="uq_game_team"),
        Index("idx_game_stats_game", "game_id"),
    )

    def __repr__(self) -> str:
        return f"<GameStats(game_id={self.game_id!r}, team_id={self.team_id}, is_home={self.is_home})>"


class PlayerGameStats(Base, TimestampMixin):
    """Player-level game statistics.

    Stores box score and player tracking data for each player in a game.

    Attributes:
        id: Auto-increment primary key.
        game_id: Foreign key to games.
        player_id: Foreign key to players.
        team_id: Foreign key to teams.
        minutes: Minutes played.
        points: Points scored.
        rebounds: Total rebounds.
        assists: Assists.
        steals: Steals.
        blocks: Blocks.
        turnovers: Turnovers.
        fgm: Field goals made.
        fga: Field goals attempted.
        fg3m: Three-pointers made.
        fg3a: Three-pointers attempted.
        ftm: Free throws made.
        fta: Free throws attempted.
        plus_minus: Plus/minus for the game.
        distance_miles: Total distance traveled (from player tracking).
        speed_avg: Average speed (from player tracking).
    """

    __tablename__ = "player_game_stats"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    game_id: Mapped[str] = mapped_column(ForeignKey("games.game_id"), nullable=False)
    player_id: Mapped[int] = mapped_column(
        ForeignKey("players.player_id"), nullable=False
    )
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.team_id"), nullable=False)

    # Box score stats
    minutes: Mapped[float | None] = mapped_column(nullable=True)
    points: Mapped[int | None] = mapped_column(nullable=True)
    rebounds: Mapped[int | None] = mapped_column(nullable=True)
    assists: Mapped[int | None] = mapped_column(nullable=True)
    steals: Mapped[int | None] = mapped_column(nullable=True)
    blocks: Mapped[int | None] = mapped_column(nullable=True)
    turnovers: Mapped[int | None] = mapped_column(nullable=True)
    fgm: Mapped[int | None] = mapped_column(nullable=True)
    fga: Mapped[int | None] = mapped_column(nullable=True)
    fg3m: Mapped[int | None] = mapped_column(nullable=True)
    fg3a: Mapped[int | None] = mapped_column(nullable=True)
    ftm: Mapped[int | None] = mapped_column(nullable=True)
    fta: Mapped[int | None] = mapped_column(nullable=True)
    plus_minus: Mapped[int | None] = mapped_column(nullable=True)

    # Player tracking (from boxscoreplayertrackv2)
    distance_miles: Mapped[float | None] = mapped_column(nullable=True)
    speed_avg: Mapped[float | None] = mapped_column(nullable=True)

    __table_args__ = (
        UniqueConstraint("game_id", "player_id", name="uq_player_game"),
        Index("idx_player_game_stats", "game_id", "player_id"),
    )

    def __repr__(self) -> str:
        return f"<PlayerGameStats(game_id={self.game_id!r}, player_id={self.player_id})>"


# =============================================================================
# Play-by-Play Models
# =============================================================================


class Play(Base):
    """Play-by-play event table.

    Stores individual play events from games.

    Attributes:
        id: Auto-increment primary key.
        game_id: Foreign key to games.
        event_num: Event number within the game.
        period: Game period (1-4 for regulation, 5+ for overtime).
        pc_time: Game clock time in "MM:SS" format.
        wc_time: Wall clock time of the event.
        event_type: EVENTMSGTYPE from NBA API.
        event_action: EVENTMSGACTIONTYPE from NBA API.
        home_description: Home team event description.
        away_description: Away team event description.
        neutral_description: Neutral event description.
        score_home: Home team score after event.
        score_away: Away team score after event.
        player1_id: Primary player involved.
        player2_id: Secondary player involved.
        player3_id: Tertiary player involved.
        team_id: Team associated with event.
        game: Relationship to Game.
    """

    __tablename__ = "plays"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    game_id: Mapped[str] = mapped_column(ForeignKey("games.game_id"), nullable=False)
    event_num: Mapped[int] = mapped_column(nullable=False)
    period: Mapped[int] = mapped_column(nullable=False)
    pc_time: Mapped[str | None] = mapped_column(String(10), nullable=True)
    wc_time: Mapped[str | None] = mapped_column(String(20), nullable=True)
    event_type: Mapped[int] = mapped_column(nullable=False)
    event_action: Mapped[int | None] = mapped_column(nullable=True)
    home_description: Mapped[str | None] = mapped_column(String(500), nullable=True)
    away_description: Mapped[str | None] = mapped_column(String(500), nullable=True)
    neutral_description: Mapped[str | None] = mapped_column(String(500), nullable=True)
    score_home: Mapped[int | None] = mapped_column(nullable=True)
    score_away: Mapped[int | None] = mapped_column(nullable=True)
    player1_id: Mapped[int | None] = mapped_column(
        ForeignKey("players.player_id"), nullable=True
    )
    player2_id: Mapped[int | None] = mapped_column(
        ForeignKey("players.player_id"), nullable=True
    )
    player3_id: Mapped[int | None] = mapped_column(
        ForeignKey("players.player_id"), nullable=True
    )
    team_id: Mapped[int | None] = mapped_column(
        ForeignKey("teams.team_id"), nullable=True
    )

    # Relationships
    game: Mapped[Game] = relationship(back_populates="plays")

    __table_args__ = (
        UniqueConstraint("game_id", "event_num", name="uq_play_game_event"),
        Index("idx_plays_game", "game_id"),
    )

    def __repr__(self) -> str:
        return f"<Play(game_id={self.game_id!r}, event_num={self.event_num})>"


class Shot(Base):
    """Shot chart table.

    Stores individual shot attempts with court coordinates.

    Attributes:
        id: Auto-increment primary key.
        game_id: Foreign key to games.
        player_id: Foreign key to players.
        team_id: Foreign key to teams.
        period: Game period.
        minutes_remaining: Minutes remaining in period.
        seconds_remaining: Seconds remaining in period.
        action_type: Type of shot action.
        shot_type: '2PT' or '3PT'.
        shot_zone_basic: Basic zone classification.
        shot_zone_area: Area classification.
        shot_zone_range: Range classification.
        shot_distance: Distance from basket in feet.
        loc_x: X coordinate on court.
        loc_y: Y coordinate on court.
        made: True if shot was made.
        game: Relationship to Game.
    """

    __tablename__ = "shots"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    game_id: Mapped[str] = mapped_column(ForeignKey("games.game_id"), nullable=False)
    player_id: Mapped[int] = mapped_column(
        ForeignKey("players.player_id"), nullable=False
    )
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.team_id"), nullable=False)
    period: Mapped[int] = mapped_column(nullable=False)
    minutes_remaining: Mapped[int] = mapped_column(nullable=False)
    seconds_remaining: Mapped[int] = mapped_column(nullable=False)
    action_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    shot_type: Mapped[str | None] = mapped_column(String(10), nullable=True)
    shot_zone_basic: Mapped[str | None] = mapped_column(String(30), nullable=True)
    shot_zone_area: Mapped[str | None] = mapped_column(String(30), nullable=True)
    shot_zone_range: Mapped[str | None] = mapped_column(String(30), nullable=True)
    shot_distance: Mapped[int | None] = mapped_column(nullable=True)
    loc_x: Mapped[int] = mapped_column(nullable=False)
    loc_y: Mapped[int] = mapped_column(nullable=False)
    made: Mapped[bool] = mapped_column(nullable=False)

    # Relationships
    game: Mapped[Game] = relationship(back_populates="shots")

    __table_args__ = (
        UniqueConstraint(
            "game_id",
            "player_id",
            "period",
            "minutes_remaining",
            "seconds_remaining",
            "loc_x",
            "loc_y",
            name="uq_shot_unique",
        ),
        Index("idx_shots_game", "game_id"),
        Index("idx_shots_player", "player_id"),
    )

    def __repr__(self) -> str:
        return f"<Shot(game_id={self.game_id!r}, player_id={self.player_id}, made={self.made})>"


# =============================================================================
# Derived and Feature Models
# =============================================================================


class Stint(Base):
    """Lineup stint table.

    Stores continuous periods where both teams have the same 5 players on court.
    Used for RAPM calculations.

    Attributes:
        id: Auto-increment primary key.
        game_id: Foreign key to games.
        period: Game period.
        start_time: Start time in "MM:SS" format.
        end_time: End time in "MM:SS" format.
        duration_seconds: Duration in seconds.
        home_lineup: JSON array of 5 home player IDs.
        away_lineup: JSON array of 5 away player IDs.
        home_points: Points scored by home team.
        away_points: Points scored by away team.
        possessions: Estimated possessions in stint.
        game: Relationship to Game.
    """

    __tablename__ = "stints"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    game_id: Mapped[str] = mapped_column(ForeignKey("games.game_id"), nullable=False)
    period: Mapped[int] = mapped_column(nullable=False)
    start_time: Mapped[str] = mapped_column(String(10), nullable=False)
    end_time: Mapped[str] = mapped_column(String(10), nullable=False)
    duration_seconds: Mapped[int] = mapped_column(nullable=False)
    home_lineup: Mapped[str] = mapped_column(Text, nullable=False)  # JSON array
    away_lineup: Mapped[str] = mapped_column(Text, nullable=False)  # JSON array
    home_points: Mapped[int] = mapped_column(nullable=False)
    away_points: Mapped[int] = mapped_column(nullable=False)
    possessions: Mapped[float | None] = mapped_column(nullable=True)

    # Relationships
    game: Mapped[Game] = relationship(back_populates="stints")

    __table_args__ = (Index("idx_stints_game", "game_id"),)

    def __repr__(self) -> str:
        return f"<Stint(game_id={self.game_id!r}, period={self.period}, start={self.start_time})>"


class Odds(Base, TimestampMixin):
    """Betting odds table.

    Stores historical betting lines from various sportsbooks.

    Attributes:
        id: Auto-increment primary key.
        game_id: Foreign key to games.
        source: Odds source (e.g., 'pinnacle', 'draftkings').
        timestamp: When the odds were captured.
        home_ml: Home team moneyline (decimal odds).
        away_ml: Away team moneyline (decimal odds).
        spread_home: Home team spread.
        spread_home_odds: Home spread odds (decimal).
        spread_away_odds: Away spread odds (decimal).
        total: Over/under total points.
        over_odds: Over odds (decimal).
        under_odds: Under odds (decimal).
    """

    __tablename__ = "odds"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    game_id: Mapped[str] = mapped_column(ForeignKey("games.game_id"), nullable=False)
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(nullable=False)
    home_ml: Mapped[float | None] = mapped_column(nullable=True)
    away_ml: Mapped[float | None] = mapped_column(nullable=True)
    spread_home: Mapped[float | None] = mapped_column(nullable=True)
    spread_home_odds: Mapped[float | None] = mapped_column(nullable=True)
    spread_away_odds: Mapped[float | None] = mapped_column(nullable=True)
    total: Mapped[float | None] = mapped_column(nullable=True)
    over_odds: Mapped[float | None] = mapped_column(nullable=True)
    under_odds: Mapped[float | None] = mapped_column(nullable=True)

    __table_args__ = (Index("idx_odds_game", "game_id"),)

    def __repr__(self) -> str:
        return f"<Odds(game_id={self.game_id!r}, source={self.source!r})>"


class PlayerRAPM(Base, TimestampMixin):
    """Player RAPM (Regularized Adjusted Plus-Minus) table.

    Stores calculated RAPM values for each player-season combination.
    Used by the feature engineering pipeline in Phase 3.

    Attributes:
        id: Auto-increment primary key.
        player_id: Foreign key to players.
        season_id: Foreign key to seasons.
        calculation_date: Date when RAPM was calculated.
        orapm: Offensive RAPM.
        drapm: Defensive RAPM.
        rapm: Total RAPM (orapm + drapm).
        sample_stints: Number of stints used in calculation.
    """

    __tablename__ = "player_rapm"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(
        ForeignKey("players.player_id"), nullable=False
    )
    season_id: Mapped[str] = mapped_column(
        ForeignKey("seasons.season_id"), nullable=False
    )
    calculation_date: Mapped[date] = mapped_column(nullable=False)
    orapm: Mapped[float] = mapped_column(nullable=False)
    drapm: Mapped[float] = mapped_column(nullable=False)
    rapm: Mapped[float] = mapped_column(nullable=False)
    sample_stints: Mapped[int] = mapped_column(nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "player_id", "season_id", "calculation_date", name="uq_player_rapm"
        ),
        Index("idx_player_rapm_player", "player_id"),
        Index("idx_player_rapm_season", "season_id"),
    )

    def __repr__(self) -> str:
        return f"<PlayerRAPM(player_id={self.player_id}, season_id={self.season_id!r}, rapm={self.rapm:.2f})>"


class LineupSpacing(Base, TimestampMixin):
    """Lineup spacing metrics table.

    Stores spatial distribution metrics for 5-player lineups.
    Used by the feature engineering pipeline in Phase 3.

    Attributes:
        id: Auto-increment primary key.
        season_id: Foreign key to seasons.
        lineup_hash: Hash of sorted 5 player IDs for fast lookup.
        player_ids: JSON array of 5 player IDs.
        hull_area: Convex hull area of lineup shot distribution.
        centroid_x: X coordinate of shot distribution centroid.
        centroid_y: Y coordinate of shot distribution centroid.
        shot_count: Number of shots used in calculation.
    """

    __tablename__ = "lineup_spacing"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    season_id: Mapped[str] = mapped_column(
        ForeignKey("seasons.season_id"), nullable=False
    )
    lineup_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    player_ids: Mapped[str] = mapped_column(String(100), nullable=False)  # JSON array
    hull_area: Mapped[float] = mapped_column(nullable=False)
    centroid_x: Mapped[float] = mapped_column(nullable=False)
    centroid_y: Mapped[float] = mapped_column(nullable=False)
    shot_count: Mapped[int] = mapped_column(nullable=False)

    __table_args__ = (
        UniqueConstraint("season_id", "lineup_hash", name="uq_lineup_spacing"),
        Index("idx_lineup_spacing_season", "season_id"),
        Index("idx_lineup_spacing_hash", "lineup_hash"),
    )

    def __repr__(self) -> str:
        return f"<LineupSpacing(season_id={self.season_id!r}, lineup_hash={self.lineup_hash[:8]}...)>"


class SeasonStats(Base, TimestampMixin):
    """Season-level statistics for normalization.

    Stores mean/std/min/max for various metrics to enable z-score normalization.
    Used by the feature engineering pipeline in Phase 3.

    Attributes:
        id: Auto-increment primary key.
        season_id: Foreign key to seasons.
        metric_name: Name of the metric.
        mean_value: Mean value across season.
        std_value: Standard deviation.
        min_value: Minimum value.
        max_value: Maximum value.
    """

    __tablename__ = "season_stats"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    season_id: Mapped[str] = mapped_column(
        ForeignKey("seasons.season_id"), nullable=False
    )
    metric_name: Mapped[str] = mapped_column(String(50), nullable=False)
    mean_value: Mapped[float] = mapped_column(nullable=False)
    std_value: Mapped[float] = mapped_column(nullable=False)
    min_value: Mapped[float] = mapped_column(nullable=False)
    max_value: Mapped[float] = mapped_column(nullable=False)

    __table_args__ = (
        UniqueConstraint("season_id", "metric_name", name="uq_season_metric"),
        Index("idx_season_stats_season", "season_id"),
    )

    def __repr__(self) -> str:
        return f"<SeasonStats(season_id={self.season_id!r}, metric_name={self.metric_name!r})>"


# List of all models for explicit imports
__all__ = [
    "Game",
    "GameStats",
    "LineupSpacing",
    "Odds",
    "Play",
    "Player",
    "PlayerGameStats",
    "PlayerRAPM",
    "PlayerSeason",
    "Season",
    "SeasonStats",
    "Shot",
    "Stint",
    "Team",
]
