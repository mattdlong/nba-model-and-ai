"""Comprehensive data quality review for NBA database.

This module provides tools to analyze database completeness, consistency,
validity, and referential integrity. It produces detailed reports identifying
issues and games that need repair.

Example:
    >>> from nba_model.data.quality import DataQualityReviewer
    >>> from nba_model.data import init_db, session_scope
    >>> init_db()
    >>> with session_scope() as session:
    ...     reviewer = DataQualityReviewer()
    ...     report = reviewer.run_full_review(session)
    ...     print(report.summary)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from sqlalchemy import func, and_, distinct, case

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


@dataclass
class QualityIssue:
    """A single quality issue found during review.

    Attributes:
        severity: Issue severity level (error, warning, info).
        dimension: Quality dimension (completeness, consistency, validity, referential).
        entity: Entity type affected (Game, Play, Stint, etc.).
        entity_id: ID of the specific entity if applicable.
        message: Human-readable description of the issue.
    """

    severity: Literal["error", "warning", "info"]
    dimension: str
    entity: str
    entity_id: str | None
    message: str

    def __str__(self) -> str:
        entity_str = f" ({self.entity_id})" if self.entity_id else ""
        return f"[{self.severity.upper()}] {self.entity}{entity_str}: {self.message}"


@dataclass
class QualityReport:
    """Result of a comprehensive data quality review.

    Attributes:
        generated_at: Timestamp when the report was generated.
        issues: List of all quality issues found.
        summary: Count of issues by severity.
        games_needing_repair: List of game IDs with errors that need fixing.
        season_filter: Season filter used for the review (None = all seasons).
    """

    generated_at: datetime
    issues: list[QualityIssue] = field(default_factory=list)
    summary: dict[str, int] = field(default_factory=dict)
    games_needing_repair: list[str] = field(default_factory=list)
    season_filter: str | None = None

    def add_issue(self, issue: QualityIssue) -> None:
        """Add an issue to the report."""
        self.issues.append(issue)

        # Update summary
        key = f"{issue.dimension}_{issue.severity}"
        self.summary[key] = self.summary.get(key, 0) + 1

        # Track games needing repair (errors only)
        if issue.severity == "error" and issue.entity_id:
            if issue.entity_id not in self.games_needing_repair:
                self.games_needing_repair.append(issue.entity_id)

    def get_by_dimension(self, dimension: str) -> list[QualityIssue]:
        """Get all issues for a specific dimension."""
        return [i for i in self.issues if i.dimension == dimension]

    def get_by_severity(self, severity: str) -> list[QualityIssue]:
        """Get all issues for a specific severity."""
        return [i for i in self.issues if i.severity == severity]

    @property
    def error_count(self) -> int:
        """Total number of errors."""
        return sum(v for k, v in self.summary.items() if k.endswith("_error"))

    @property
    def warning_count(self) -> int:
        """Total number of warnings."""
        return sum(v for k, v in self.summary.items() if k.endswith("_warning"))

    @property
    def info_count(self) -> int:
        """Total number of info messages."""
        return sum(v for k, v in self.summary.items() if k.endswith("_info"))


class DataQualityReviewer:
    """Reviews data quality across the entire NBA database.

    Performs checks across four dimensions:
    - Completeness: Are all expected records present?
    - Consistency: Do related records agree?
    - Validity: Are values within expected ranges?
    - Referential Integrity: Do all FKs reference valid records?
    """

    def __init__(self) -> None:
        """Initialize the data quality reviewer."""
        self.logger = logging.getLogger(self.__class__.__name__)

    def run_full_review(
        self,
        session: Session,
        season: str | None = None,
    ) -> QualityReport:
        """Run a comprehensive data quality review.

        Args:
            session: Database session.
            season: Optional season filter (e.g., "2023-24").

        Returns:
            QualityReport with all issues found.
        """
        report = QualityReport(
            generated_at=datetime.now(),
            season_filter=season,
        )

        self.logger.info("Starting data quality review...")

        # Run all checks
        completeness_issues = self.check_completeness(session, season)
        for issue in completeness_issues:
            report.add_issue(issue)

        consistency_issues = self.check_consistency(session, season)
        for issue in consistency_issues:
            report.add_issue(issue)

        validity_issues = self.check_validity(session, season)
        for issue in validity_issues:
            report.add_issue(issue)

        referential_issues = self.check_referential_integrity(session, season)
        for issue in referential_issues:
            report.add_issue(issue)

        self.logger.info(
            f"Review complete: {report.error_count} errors, "
            f"{report.warning_count} warnings, {report.info_count} info"
        )

        return report

    def check_completeness(
        self,
        session: Session,
        season: str | None = None,
    ) -> list[QualityIssue]:
        """Check for missing or incomplete data.

        Checks:
        - Games without plays
        - Games with <100 plays
        - Games missing TeamBoxScore (Total)
        - Games with <6 TeamBoxScore rows
        - Games missing PlayerBoxScore
        - Games with <10 player stats
        - Games without stints
        - Seasons with <1000 games

        Args:
            session: Database session.
            season: Optional season filter.

        Returns:
            List of completeness issues.
        """
        from nba_model.data.models import (
            Game,
            Play,
            PlayerBoxScore,
            Season,
            Stint,
            TeamBoxScore,
        )

        issues: list[QualityIssue] = []

        # Build base game query
        game_query = session.query(Game)
        if season:
            game_query = game_query.filter(Game.season_id == season)

        # Get all game IDs for the scope
        game_ids = [g.game_id for g in game_query.all()]

        if not game_ids:
            issues.append(
                QualityIssue(
                    severity="warning",
                    dimension="completeness",
                    entity="Season",
                    entity_id=season,
                    message="No games found in database" + (f" for season {season}" if season else ""),
                )
            )
            return issues

        # Games without plays
        games_with_plays = (
            session.query(distinct(Play.game_id))
            .filter(Play.game_id.in_(game_ids))
            .all()
        )
        games_with_plays_set = {g[0] for g in games_with_plays}
        games_without_plays = set(game_ids) - games_with_plays_set

        for game_id in games_without_plays:
            issues.append(
                QualityIssue(
                    severity="error",
                    dimension="completeness",
                    entity="Game",
                    entity_id=game_id,
                    message="Missing play-by-play data",
                )
            )

        # Games with low play count (<100)
        play_counts = (
            session.query(Play.game_id, func.count(Play.action_id))
            .filter(Play.game_id.in_(game_ids))
            .group_by(Play.game_id)
            .all()
        )
        for game_id, count in play_counts:
            if count < 100:
                issues.append(
                    QualityIssue(
                        severity="warning",
                        dimension="completeness",
                        entity="Game",
                        entity_id=game_id,
                        message=f"Low play count ({count}), expected >100",
                    )
                )

        # Games missing TeamBoxScore (Total)
        games_with_total = (
            session.query(distinct(TeamBoxScore.game_id))
            .filter(
                TeamBoxScore.game_id.in_(game_ids),
                TeamBoxScore.stat_type == "Total",
            )
            .all()
        )
        games_with_total_set = {g[0] for g in games_with_total}
        games_missing_total = set(game_ids) - games_with_total_set

        for game_id in games_missing_total:
            issues.append(
                QualityIssue(
                    severity="error",
                    dimension="completeness",
                    entity="Game",
                    entity_id=game_id,
                    message="Missing TeamBoxScore (Total)",
                )
            )

        # Games with <6 TeamBoxScore rows (need 2 teams x 3 types)
        box_counts = (
            session.query(TeamBoxScore.game_id, func.count())
            .filter(TeamBoxScore.game_id.in_(game_ids))
            .group_by(TeamBoxScore.game_id)
            .all()
        )
        for game_id, count in box_counts:
            if count < 6:
                issues.append(
                    QualityIssue(
                        severity="warning",
                        dimension="completeness",
                        entity="Game",
                        entity_id=game_id,
                        message=f"Low TeamBoxScore count ({count}), expected 6",
                    )
                )

        # Games missing PlayerBoxScore
        games_with_player_stats = (
            session.query(distinct(PlayerBoxScore.game_id))
            .filter(PlayerBoxScore.game_id.in_(game_ids))
            .all()
        )
        games_with_player_stats_set = {g[0] for g in games_with_player_stats}
        games_missing_player_stats = set(game_ids) - games_with_player_stats_set

        for game_id in games_missing_player_stats:
            issues.append(
                QualityIssue(
                    severity="error",
                    dimension="completeness",
                    entity="Game",
                    entity_id=game_id,
                    message="Missing player box scores",
                )
            )

        # Games with <10 player stats
        player_stat_counts = (
            session.query(PlayerBoxScore.game_id, func.count(PlayerBoxScore.person_id))
            .filter(PlayerBoxScore.game_id.in_(game_ids))
            .group_by(PlayerBoxScore.game_id)
            .all()
        )
        for game_id, count in player_stat_counts:
            if count < 10:
                issues.append(
                    QualityIssue(
                        severity="warning",
                        dimension="completeness",
                        entity="Game",
                        entity_id=game_id,
                        message=f"Low player box score count ({count}), expected >10",
                    )
                )

        # Games without stints
        games_with_stints = (
            session.query(distinct(Stint.game_id))
            .filter(Stint.game_id.in_(game_ids))
            .all()
        )
        games_with_stints_set = {g[0] for g in games_with_stints}
        games_without_stints = set(game_ids) - games_with_stints_set

        for game_id in games_without_stints:
            issues.append(
                QualityIssue(
                    severity="warning",
                    dimension="completeness",
                    entity="Game",
                    entity_id=game_id,
                    message="Missing stint data",
                )
            )

        # Seasons with <1000 games (if not filtering by specific season)
        if not season:
            season_counts = (
                session.query(Season.season_id, func.count(Game.game_id))
                .outerjoin(Game, Season.season_id == Game.season_id)
                .group_by(Season.season_id)
                .all()
            )
            for season_id, count in season_counts:
                if count < 1000:
                    issues.append(
                        QualityIssue(
                            severity="warning",
                            dimension="completeness",
                            entity="Season",
                            entity_id=season_id,
                            message=f"Low game count ({count}), expected >1000",
                        )
                    )

        return issues

    def check_consistency(
        self,
        session: Session,
        season: str | None = None,
    ) -> list[QualityIssue]:
        """Check for inconsistencies between related data.

        Checks:
        - Game.home_score != TeamBoxScore.points (Total)
        - Starters + Bench != Total in TeamBoxScore
        - SUM(PlayerBoxScore.points) != TeamBoxScore.points
        - Stint total points vs game score mismatch

        Args:
            session: Database session.
            season: Optional season filter.

        Returns:
            List of consistency issues.
        """
        from nba_model.data.models import (
            Game,
            PlayerBoxScore,
            Stint,
            TeamBoxScore,
        )

        issues: list[QualityIssue] = []

        # Build base game query
        game_query = session.query(Game)
        if season:
            game_query = game_query.filter(Game.season_id == season)

        games = game_query.all()

        for game in games:
            # Check Game score vs TeamBoxScore Total
            total_box_scores = (
                session.query(TeamBoxScore)
                .filter(
                    TeamBoxScore.game_id == game.game_id,
                    TeamBoxScore.stat_type == "Total",
                )
                .all()
            )

            for box in total_box_scores:
                if box.team_id == game.home_team_id:
                    if game.home_score and box.points and game.home_score != box.points:
                        issues.append(
                            QualityIssue(
                                severity="error",
                                dimension="consistency",
                                entity="Game",
                                entity_id=game.game_id,
                                message=f"Home score mismatch: Game={game.home_score}, BoxScore={box.points}",
                            )
                        )
                elif box.team_id == game.away_team_id:
                    if game.away_score and box.points and game.away_score != box.points:
                        issues.append(
                            QualityIssue(
                                severity="error",
                                dimension="consistency",
                                entity="Game",
                                entity_id=game.game_id,
                                message=f"Away score mismatch: Game={game.away_score}, BoxScore={box.points}",
                            )
                        )

            # Check Starters + Bench = Total
            team_boxes = (
                session.query(TeamBoxScore)
                .filter(TeamBoxScore.game_id == game.game_id)
                .all()
            )

            # Group by team
            teams: dict[int, dict[str, TeamBoxScore]] = {}
            for box in team_boxes:
                if box.team_id not in teams:
                    teams[box.team_id] = {}
                teams[box.team_id][box.stat_type] = box

            for team_id, stat_types in teams.items():
                if "Starters" in stat_types and "Bench" in stat_types and "Total" in stat_types:
                    starters = stat_types["Starters"]
                    bench = stat_types["Bench"]
                    total = stat_types["Total"]

                    if (
                        starters.points is not None
                        and bench.points is not None
                        and total.points is not None
                    ):
                        expected = starters.points + bench.points
                        if expected != total.points:
                            issues.append(
                                QualityIssue(
                                    severity="warning",
                                    dimension="consistency",
                                    entity="Game",
                                    entity_id=game.game_id,
                                    message=f"Team {team_id}: Starters({starters.points}) + Bench({bench.points}) != Total({total.points})",
                                )
                            )

            # Check SUM(PlayerBoxScore.points) vs TeamBoxScore Total
            player_points = (
                session.query(
                    PlayerBoxScore.team_id,
                    func.sum(PlayerBoxScore.points),
                )
                .filter(PlayerBoxScore.game_id == game.game_id)
                .group_by(PlayerBoxScore.team_id)
                .all()
            )

            for team_id, player_total in player_points:
                if team_id in teams and "Total" in teams[team_id]:
                    box_total = teams[team_id]["Total"].points
                    if (
                        player_total is not None
                        and box_total is not None
                        and player_total != box_total
                    ):
                        issues.append(
                            QualityIssue(
                                severity="warning",
                                dimension="consistency",
                                entity="Game",
                                entity_id=game.game_id,
                                message=f"Team {team_id}: SUM(player points)={player_total} != TeamBoxScore={box_total}",
                            )
                        )

        # Check stint totals vs game scores (sample check - expensive for full DB)
        stint_query = session.query(
            Stint.game_id,
            func.sum(Stint.home_points).label("total_home"),
            func.sum(Stint.away_points).label("total_away"),
        ).group_by(Stint.game_id)

        if season:
            stint_query = stint_query.join(Game).filter(Game.season_id == season)

        stint_totals = stint_query.limit(1000).all()  # Limit for performance

        for game_id, stint_home, stint_away in stint_totals:
            game = session.query(Game).filter(Game.game_id == game_id).first()
            if game and game.home_score and game.away_score:
                # Allow some margin for stints not covering full game
                if stint_home and abs(stint_home - game.home_score) > 20:
                    issues.append(
                        QualityIssue(
                            severity="info",
                            dimension="consistency",
                            entity="Game",
                            entity_id=game_id,
                            message=f"Stint home points ({stint_home}) differs from game score ({game.home_score})",
                        )
                    )
                if stint_away and abs(stint_away - game.away_score) > 20:
                    issues.append(
                        QualityIssue(
                            severity="info",
                            dimension="consistency",
                            entity="Game",
                            entity_id=game_id,
                            message=f"Stint away points ({stint_away}) differs from game score ({game.away_score})",
                        )
                    )

        return issues

    def check_validity(
        self,
        session: Session,
        season: str | None = None,
    ) -> list[QualityIssue]:
        """Check for invalid or unreasonable values.

        Checks:
        - Invalid stat_type (not Starters/Bench/Total)
        - Negative statistics
        - Unrealistic scores (>200 or <50)
        - Invalid periods (<1 or >8)
        - Stint lineups != 5 players
        - Zero-duration stints

        Args:
            session: Database session.
            season: Optional season filter.

        Returns:
            List of validity issues.
        """
        from nba_model.data.models import (
            Game,
            Play,
            PlayerBoxScore,
            Stint,
            TeamBoxScore,
        )

        issues: list[QualityIssue] = []

        # Build base game query for filtering
        game_query = session.query(Game.game_id)
        if season:
            game_query = game_query.filter(Game.season_id == season)
        game_ids = [g[0] for g in game_query.all()]

        if not game_ids:
            return issues

        # Invalid stat_type values
        invalid_stat_types = (
            session.query(TeamBoxScore.game_id, TeamBoxScore.stat_type)
            .filter(
                TeamBoxScore.game_id.in_(game_ids),
                ~TeamBoxScore.stat_type.in_(["Starters", "Bench", "Total"]),
            )
            .all()
        )
        for game_id, stat_type in invalid_stat_types:
            issues.append(
                QualityIssue(
                    severity="error",
                    dimension="validity",
                    entity="TeamBoxScore",
                    entity_id=game_id,
                    message=f"Invalid stat_type: {stat_type}",
                )
            )

        # Negative points in box scores
        negative_points = (
            session.query(TeamBoxScore.game_id, TeamBoxScore.team_id, TeamBoxScore.points)
            .filter(
                TeamBoxScore.game_id.in_(game_ids),
                TeamBoxScore.points < 0,
            )
            .all()
        )
        for game_id, team_id, points in negative_points:
            issues.append(
                QualityIssue(
                    severity="error",
                    dimension="validity",
                    entity="TeamBoxScore",
                    entity_id=game_id,
                    message=f"Negative points ({points}) for team {team_id}",
                )
            )

        # Negative player points
        negative_player_points = (
            session.query(PlayerBoxScore.game_id, PlayerBoxScore.person_id, PlayerBoxScore.points)
            .filter(
                PlayerBoxScore.game_id.in_(game_ids),
                PlayerBoxScore.points < 0,
            )
            .all()
        )
        for game_id, person_id, points in negative_player_points:
            issues.append(
                QualityIssue(
                    severity="error",
                    dimension="validity",
                    entity="PlayerBoxScore",
                    entity_id=game_id,
                    message=f"Negative points ({points}) for player {person_id}",
                )
            )

        # Unrealistic game scores (>200 or <50 for completed games)
        games = (
            session.query(Game)
            .filter(
                Game.game_id.in_(game_ids),
                Game.status == "completed",
            )
            .all()
        )
        for game in games:
            if game.home_score and (game.home_score > 200 or game.home_score < 50):
                issues.append(
                    QualityIssue(
                        severity="warning",
                        dimension="validity",
                        entity="Game",
                        entity_id=game.game_id,
                        message=f"Unusual home score: {game.home_score}",
                    )
                )
            if game.away_score and (game.away_score > 200 or game.away_score < 50):
                issues.append(
                    QualityIssue(
                        severity="warning",
                        dimension="validity",
                        entity="Game",
                        entity_id=game.game_id,
                        message=f"Unusual away score: {game.away_score}",
                    )
                )

        # Invalid periods (<1 or >8)
        invalid_periods = (
            session.query(distinct(Play.game_id), Play.period)
            .filter(
                Play.game_id.in_(game_ids),
                (Play.period < 1) | (Play.period > 8),
            )
            .all()
        )
        for game_id, period in invalid_periods:
            issues.append(
                QualityIssue(
                    severity="error",
                    dimension="validity",
                    entity="Play",
                    entity_id=game_id,
                    message=f"Invalid period: {period}",
                )
            )

        # Stint lineup validation
        stints = (
            session.query(Stint)
            .filter(Stint.game_id.in_(game_ids))
            .all()
        )

        for stint in stints:
            # Check lineup sizes
            try:
                home_lineup = json.loads(stint.home_lineup) if isinstance(stint.home_lineup, str) else stint.home_lineup
                away_lineup = json.loads(stint.away_lineup) if isinstance(stint.away_lineup, str) else stint.away_lineup

                if len(home_lineup) != 5:
                    issues.append(
                        QualityIssue(
                            severity="error",
                            dimension="validity",
                            entity="Stint",
                            entity_id=stint.game_id,
                            message=f"Home lineup has {len(home_lineup)} players, expected 5",
                        )
                    )
                if len(away_lineup) != 5:
                    issues.append(
                        QualityIssue(
                            severity="error",
                            dimension="validity",
                            entity="Stint",
                            entity_id=stint.game_id,
                            message=f"Away lineup has {len(away_lineup)} players, expected 5",
                        )
                    )
            except (json.JSONDecodeError, TypeError) as e:
                issues.append(
                    QualityIssue(
                        severity="error",
                        dimension="validity",
                        entity="Stint",
                        entity_id=stint.game_id,
                        message=f"Invalid lineup JSON: {e}",
                    )
                )

            # Check for zero-duration stints
            if stint.duration_seconds == 0:
                issues.append(
                    QualityIssue(
                        severity="warning",
                        dimension="validity",
                        entity="Stint",
                        entity_id=stint.game_id,
                        message="Zero-duration stint",
                    )
                )

        return issues

    def check_referential_integrity(
        self,
        session: Session,
        season: str | None = None,
    ) -> list[QualityIssue]:
        """Check foreign key relationships.

        Checks:
        - Games with invalid team_id FK
        - Games with invalid season_id FK
        - PlayerBoxScore with invalid person_id
        - Shots with invalid player_id
        - Stint lineup player_ids not in Players

        Args:
            session: Database session.
            season: Optional season filter.

        Returns:
            List of referential integrity issues.
        """
        from nba_model.data.models import (
            Game,
            Player,
            PlayerBoxScore,
            Season,
            Shot,
            Stint,
            Team,
        )

        issues: list[QualityIssue] = []

        # Build base game query for filtering
        if season:
            game_ids = [
                g[0]
                for g in session.query(Game.game_id).filter(Game.season_id == season).all()
            ]
        else:
            game_ids = None  # Check all

        # Games with invalid home_team_id
        invalid_home_query = (
            session.query(Game.game_id, Game.home_team_id)
            .outerjoin(Team, Game.home_team_id == Team.team_id)
            .filter(Team.team_id.is_(None))
        )
        if game_ids:
            invalid_home_query = invalid_home_query.filter(Game.game_id.in_(game_ids))

        for game_id, team_id in invalid_home_query.all():
            issues.append(
                QualityIssue(
                    severity="error",
                    dimension="referential",
                    entity="Game",
                    entity_id=game_id,
                    message=f"Invalid home_team_id: {team_id}",
                )
            )

        # Games with invalid away_team_id
        invalid_away_query = (
            session.query(Game.game_id, Game.away_team_id)
            .outerjoin(Team, Game.away_team_id == Team.team_id)
            .filter(Team.team_id.is_(None))
        )
        if game_ids:
            invalid_away_query = invalid_away_query.filter(Game.game_id.in_(game_ids))

        for game_id, team_id in invalid_away_query.all():
            issues.append(
                QualityIssue(
                    severity="error",
                    dimension="referential",
                    entity="Game",
                    entity_id=game_id,
                    message=f"Invalid away_team_id: {team_id}",
                )
            )

        # Games with invalid season_id
        invalid_season_query = (
            session.query(Game.game_id, Game.season_id)
            .outerjoin(Season, Game.season_id == Season.season_id)
            .filter(Season.season_id.is_(None))
        )
        if game_ids:
            invalid_season_query = invalid_season_query.filter(Game.game_id.in_(game_ids))

        for game_id, season_id in invalid_season_query.all():
            issues.append(
                QualityIssue(
                    severity="error",
                    dimension="referential",
                    entity="Game",
                    entity_id=game_id,
                    message=f"Invalid season_id: {season_id}",
                )
            )

        # PlayerBoxScore with invalid person_id
        invalid_person_query = (
            session.query(PlayerBoxScore.game_id, PlayerBoxScore.person_id)
            .outerjoin(Player, PlayerBoxScore.person_id == Player.player_id)
            .filter(Player.player_id.is_(None))
        )
        if game_ids:
            invalid_person_query = invalid_person_query.filter(
                PlayerBoxScore.game_id.in_(game_ids)
            )

        orphan_count = 0
        for game_id, person_id in invalid_person_query.limit(100).all():
            orphan_count += 1
            if orphan_count <= 10:  # Only report first 10 individually
                issues.append(
                    QualityIssue(
                        severity="error",
                        dimension="referential",
                        entity="PlayerBoxScore",
                        entity_id=game_id,
                        message=f"Invalid person_id: {person_id}",
                    )
                )

        if orphan_count > 10:
            issues.append(
                QualityIssue(
                    severity="error",
                    dimension="referential",
                    entity="PlayerBoxScore",
                    entity_id=None,
                    message=f"Found {orphan_count}+ player box scores with invalid person_id",
                )
            )

        # Shots with invalid player_id
        invalid_shot_query = (
            session.query(Shot.game_id, Shot.player_id)
            .outerjoin(Player, Shot.player_id == Player.player_id)
            .filter(Player.player_id.is_(None))
        )
        if game_ids:
            invalid_shot_query = invalid_shot_query.filter(Shot.game_id.in_(game_ids))

        shot_orphan_count = 0
        for game_id, player_id in invalid_shot_query.limit(100).all():
            shot_orphan_count += 1
            if shot_orphan_count <= 10:
                issues.append(
                    QualityIssue(
                        severity="error",
                        dimension="referential",
                        entity="Shot",
                        entity_id=game_id,
                        message=f"Invalid player_id: {player_id}",
                    )
                )

        if shot_orphan_count > 10:
            issues.append(
                QualityIssue(
                    severity="error",
                    dimension="referential",
                    entity="Shot",
                    entity_id=None,
                    message=f"Found {shot_orphan_count}+ shots with invalid player_id",
                )
            )

        # Stint lineup player validation
        stint_query = session.query(Stint)
        if game_ids:
            stint_query = stint_query.filter(Stint.game_id.in_(game_ids))

        # Get all valid player IDs
        valid_player_ids = {p[0] for p in session.query(Player.player_id).all()}

        invalid_lineup_count = 0
        for stint in stint_query.limit(1000).all():  # Limit for performance
            try:
                home_lineup = (
                    json.loads(stint.home_lineup)
                    if isinstance(stint.home_lineup, str)
                    else stint.home_lineup
                )
                away_lineup = (
                    json.loads(stint.away_lineup)
                    if isinstance(stint.away_lineup, str)
                    else stint.away_lineup
                )

                for player_id in home_lineup + away_lineup:
                    if player_id not in valid_player_ids:
                        invalid_lineup_count += 1
                        if invalid_lineup_count <= 10:
                            issues.append(
                                QualityIssue(
                                    severity="warning",
                                    dimension="referential",
                                    entity="Stint",
                                    entity_id=stint.game_id,
                                    message=f"Lineup contains unknown player_id: {player_id}",
                                )
                            )

            except (json.JSONDecodeError, TypeError):
                pass  # Already caught in validity checks

        if invalid_lineup_count > 10:
            issues.append(
                QualityIssue(
                    severity="warning",
                    dimension="referential",
                    entity="Stint",
                    entity_id=None,
                    message=f"Found {invalid_lineup_count}+ stint lineup entries with unknown player_id",
                )
            )

        return issues
