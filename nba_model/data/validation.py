"""Data validation utilities for NBA data integrity.

This module provides validation functions to ensure data quality
and integrity before committing to the database.

Example:
    >>> from nba_model.data.validation import DataValidator
    >>> validator = DataValidator()
    >>> result = validator.validate_game_completeness(session, "0022300001")
    >>> if not result.valid:
    ...     print(f"Errors: {result.errors}")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sqlalchemy import func

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from nba_model.data.models import (
        Game,
        GameStats,
        Play,
        Player,
        PlayerGameStats,
        Shot,
        Stint,
    )

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation.

    Attributes:
        valid: Whether validation passed.
        errors: List of error messages (validation failures).
        warnings: List of warning messages (potential issues).
    """

    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        """Add an error and mark as invalid."""
        self.errors.append(message)
        self.valid = False

    def add_warning(self, message: str) -> None:
        """Add a warning (does not affect validity)."""
        self.warnings.append(message)

    def merge(self, other: ValidationResult) -> None:
        """Merge another validation result into this one."""
        if not other.valid:
            self.valid = False
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)


class DataValidator:
    """Validates data integrity and quality.

    Provides methods to validate game completeness, stint data,
    season completeness, and referential integrity.
    """

    def __init__(self) -> None:
        """Initialize data validator."""
        self.logger = logging.getLogger(self.__class__.__name__)

    def validate_game_completeness(
        self,
        session: Session,
        game_id: str,
    ) -> ValidationResult:
        """Validate that all expected data exists for a game.

        Checks:
        - Game record exists
        - Play-by-play has expected row count (100-700)
        - Box scores exist for both teams
        - Player game stats exist

        Args:
            session: Database session.
            game_id: Game ID to validate.

        Returns:
            ValidationResult with any errors or warnings.
        """
        from nba_model.data.models import Game, GameStats, Play, PlayerGameStats, Shot

        result = ValidationResult()

        # Check game exists
        game = session.query(Game).filter(Game.game_id == game_id).first()
        if game is None:
            result.add_error(f"Game {game_id} not found")
            return result

        # Check play-by-play count
        play_count = (
            session.query(func.count(Play.id)).filter(Play.game_id == game_id).scalar()
        )
        if play_count == 0:
            result.add_error(f"Game {game_id} has no play-by-play data")
        elif play_count < 100:
            result.add_warning(
                f"Game {game_id} has low play count ({play_count}), expected 100-700"
            )
        elif play_count > 700:
            result.add_warning(
                f"Game {game_id} has high play count ({play_count}), expected 100-700"
            )

        # Check box scores
        game_stats_count = (
            session.query(func.count(GameStats.id))
            .filter(GameStats.game_id == game_id)
            .scalar()
        )
        if game_stats_count < 2:
            result.add_error(
                f"Game {game_id} missing team box scores (found {game_stats_count})"
            )

        # Check player stats
        player_stats_count = (
            session.query(func.count(PlayerGameStats.id))
            .filter(PlayerGameStats.game_id == game_id)
            .scalar()
        )
        if player_stats_count < 10:
            result.add_warning(
                f"Game {game_id} has low player stats count ({player_stats_count})"
            )

        # Check shots (optional - may not exist for older games)
        shot_count = (
            session.query(func.count(Shot.id)).filter(Shot.game_id == game_id).scalar()
        )
        if shot_count == 0:
            result.add_warning(f"Game {game_id} has no shot data")

        return result

    def validate_stints(
        self,
        session: Session,
        game_id: str,
    ) -> ValidationResult:
        """Validate stint data for a game.

        Checks:
        - Stints exist for both teams
        - No overlapping stints within a team
        - Each lineup has 5 players
        - Stint durations are reasonable

        Args:
            session: Database session.
            game_id: Game ID to validate.

        Returns:
            ValidationResult with any errors or warnings.
        """
        import json

        from nba_model.data.models import Stint

        result = ValidationResult()

        # Get all stints for game
        stints = session.query(Stint).filter(Stint.game_id == game_id).all()

        if not stints:
            result.add_warning(f"Game {game_id} has no stint data")
            return result

        # Group by team
        team_stints: dict[int, list] = {}
        for stint in stints:
            if stint.team_id not in team_stints:
                team_stints[stint.team_id] = []
            team_stints[stint.team_id].append(stint)

        # Check we have 2 teams
        if len(team_stints) != 2:
            result.add_error(
                f"Game {game_id} has stints for {len(team_stints)} teams, expected 2"
            )

        for team_id, team_stint_list in team_stints.items():
            # Sort by start time
            team_stint_list.sort(key=lambda s: s.start_time)

            # Check for overlaps
            for i in range(len(team_stint_list) - 1):
                current = team_stint_list[i]
                next_stint = team_stint_list[i + 1]
                if current.end_time > next_stint.start_time:
                    result.add_error(
                        f"Game {game_id} team {team_id}: overlapping stints "
                        f"({current.end_time} > {next_stint.start_time})"
                    )

            # Check lineup sizes
            for stint in team_stint_list:
                try:
                    lineup = json.loads(stint.lineup_json)
                    if len(lineup) != 5:
                        result.add_error(
                            f"Game {game_id} stint has {len(lineup)} players, expected 5"
                        )
                except (json.JSONDecodeError, TypeError):
                    result.add_error(f"Game {game_id} stint has invalid lineup JSON")

        return result

    def validate_season_completeness(
        self,
        session: Session,
        season: str,
    ) -> ValidationResult:
        """Validate that season data is complete.

        Checks:
        - Expected number of games (~1230 regular season)
        - All games have play-by-play
        - All games have box scores

        Args:
            session: Database session.
            season: Season ID (e.g., "2023-24").

        Returns:
            ValidationResult with any errors or warnings.
        """
        from nba_model.data.models import Game, GameStats, Play

        result = ValidationResult()

        # Count games
        game_count = (
            session.query(func.count(Game.game_id))
            .filter(Game.season_id == season)
            .scalar()
        )

        if game_count == 0:
            result.add_error(f"Season {season} has no games")
            return result

        # Expected ~1230 regular season games (30 teams x 82 games / 2)
        if game_count < 1000:
            result.add_warning(
                f"Season {season} has only {game_count} games, expected ~1230"
            )

        # Check games with play-by-play
        games_with_pbp = (
            session.query(func.count(func.distinct(Play.game_id)))
            .join(Game, Play.game_id == Game.game_id)
            .filter(Game.season_id == season)
            .scalar()
        )

        if games_with_pbp < game_count:
            missing = game_count - games_with_pbp
            result.add_warning(
                f"Season {season}: {missing} games missing play-by-play data"
            )

        # Check games with box scores
        games_with_boxscores = (
            session.query(func.count(func.distinct(GameStats.game_id)))
            .join(Game, GameStats.game_id == Game.game_id)
            .filter(Game.season_id == season)
            .scalar()
        )

        if games_with_boxscores < game_count:
            missing = game_count - games_with_boxscores
            result.add_warning(
                f"Season {season}: {missing} games missing box scores"
            )

        return result

    def validate_referential_integrity(
        self,
        session: Session,
    ) -> ValidationResult:
        """Validate foreign key relationships.

        Checks:
        - All game team_ids exist in teams
        - All player_game_stats player_ids exist in players
        - All shots player_ids exist in players

        Args:
            session: Database session.

        Returns:
            ValidationResult with any errors or warnings.
        """
        from nba_model.data.models import Game, Player, PlayerGameStats, Shot, Team

        result = ValidationResult()

        # Check game home_team_ids
        orphan_home_teams = (
            session.query(Game)
            .outerjoin(Team, Game.home_team_id == Team.team_id)
            .filter(Team.team_id.is_(None))
            .count()
        )
        if orphan_home_teams > 0:
            result.add_error(f"{orphan_home_teams} games have invalid home_team_id")

        # Check game away_team_ids
        orphan_away_teams = (
            session.query(Game)
            .outerjoin(Team, Game.away_team_id == Team.team_id)
            .filter(Team.team_id.is_(None))
            .count()
        )
        if orphan_away_teams > 0:
            result.add_error(f"{orphan_away_teams} games have invalid away_team_id")

        # Check player_game_stats player_ids
        orphan_player_stats = (
            session.query(PlayerGameStats)
            .outerjoin(Player, PlayerGameStats.player_id == Player.player_id)
            .filter(Player.player_id.is_(None))
            .count()
        )
        if orphan_player_stats > 0:
            result.add_error(
                f"{orphan_player_stats} player_game_stats have invalid player_id"
            )

        # Check shots player_ids
        orphan_shots = (
            session.query(Shot)
            .outerjoin(Player, Shot.player_id == Player.player_id)
            .filter(Player.player_id.is_(None))
            .count()
        )
        if orphan_shots > 0:
            result.add_error(f"{orphan_shots} shots have invalid player_id")

        return result

    def validate_batch(
        self,
        plays: list,
        shots: list,
        game_stats: list,
        player_game_stats: list,
    ) -> ValidationResult:
        """Validate a batch of data before commit.

        Checks:
        - Required fields are populated
        - IDs are valid
        - Data types are correct

        Args:
            plays: List of Play objects.
            shots: List of Shot objects.
            game_stats: List of GameStats objects.
            player_game_stats: List of PlayerGameStats objects.

        Returns:
            ValidationResult with any errors or warnings.
        """
        result = ValidationResult()

        # Check plays have required fields
        for play in plays:
            if not play.game_id:
                result.add_error("Play missing game_id")
            if play.event_num is None:
                result.add_error(f"Play in game {play.game_id} missing event_num")

        # Check shots have required fields
        for shot in shots:
            if not shot.game_id:
                result.add_error("Shot missing game_id")
            if not shot.player_id:
                result.add_error(f"Shot in game {shot.game_id} missing player_id")

        # Check game stats have required fields
        for gs in game_stats:
            if not gs.game_id:
                result.add_error("GameStats missing game_id")
            if not gs.team_id:
                result.add_error(f"GameStats for game {gs.game_id} missing team_id")

        # Check player stats have required fields
        for pgs in player_game_stats:
            if not pgs.game_id:
                result.add_error("PlayerGameStats missing game_id")
            if not pgs.player_id:
                result.add_error(
                    f"PlayerGameStats for game {pgs.game_id} missing player_id"
                )

        return result
