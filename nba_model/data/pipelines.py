"""ETL pipeline orchestration for NBA data collection.

This module provides the CollectionPipeline class for orchestrating
full data collection with checkpointing and batch processing.

Example:
    >>> from nba_model.data.pipelines import CollectionPipeline
    >>> from nba_model.data import NBAApiClient, session_scope
    >>> from nba_model.data.checkpoint import CheckpointManager
    >>> with session_scope() as session:
    ...     pipeline = CollectionPipeline(
    ...         session=session,
    ...         api_client=NBAApiClient(),
    ...         checkpoint_manager=CheckpointManager(),
    ...     )
    ...     result = pipeline.full_historical_load(["2023-24"])
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import TYPE_CHECKING

from nba_model.logging import FAIL, SUCCESS, WARN

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from nba_model.data.api import NBAApiClient
    from nba_model.data.checkpoint import CheckpointManager
    from nba_model.data.models import (
        Game,
        GameStats,
        Play,
        PlayerGameStats,
        Shot,
        Stint,
    )

logger = logging.getLogger(__name__)


class PipelineStatus(Enum):
    """Status of a pipeline execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class PipelineResult:
    """Results from pipeline execution.

    Attributes:
        status: Final pipeline status.
        seasons_processed: List of seasons that were processed.
        games_processed: Total number of games processed.
        plays_collected: Total number of plays collected.
        shots_collected: Total number of shots collected.
        stints_derived: Total number of stints derived.
        errors: List of error messages.
        duration_seconds: Total execution time in seconds.
    """

    status: PipelineStatus
    seasons_processed: list[str] = field(default_factory=list)
    games_processed: int = 0
    plays_collected: int = 0
    shots_collected: int = 0
    stints_derived: int = 0
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0


@dataclass
class BatchResult:
    """Results from single batch collection.

    Attributes:
        game_ids: List of game IDs in this batch.
        plays: List of collected Play objects.
        shots: List of collected Shot objects.
        game_stats: List of collected GameStats objects.
        player_game_stats: List of collected PlayerGameStats objects.
        stints: List of derived Stint objects.
        errors: List of (game_id, error_message) tuples.
    """

    game_ids: list[str] = field(default_factory=list)
    plays: list = field(default_factory=list)
    shots: list = field(default_factory=list)
    game_stats: list = field(default_factory=list)
    player_game_stats: list = field(default_factory=list)
    stints: list = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)


class CollectionPipeline:
    """Orchestrates full data collection with checkpointing and batch processing.

    Provides methods for:
    - Full historical data load across multiple seasons
    - Incremental updates for new games
    - Repair of specific games

    All operations support checkpointing for resumability.
    """

    def __init__(
        self,
        session: Session,
        api_client: NBAApiClient,
        checkpoint_manager: CheckpointManager,
        batch_size: int = 50,
    ) -> None:
        """Initialize pipeline.

        Args:
            session: Database session.
            api_client: NBA API client.
            checkpoint_manager: Checkpoint storage.
            batch_size: Games per batch commit.
        """
        self.session = session
        self.api = api_client
        self.checkpoint = checkpoint_manager
        self.batch_size = batch_size
        self.logger = logging.getLogger(self.__class__.__name__)

        # Import collectors here to avoid circular imports
        from nba_model.data.collectors import (
            BoxScoreCollector,
            GamesCollector,
            PlayByPlayCollector,
            PlayersCollector,
            ShotsCollector,
        )
        from nba_model.data.stints import StintDeriver
        from nba_model.data.validation import DataValidator

        self.games_collector = GamesCollector(api_client, session)
        self.players_collector = PlayersCollector(api_client, session)
        self.pbp_collector = PlayByPlayCollector(api_client, session)
        self.shots_collector = ShotsCollector(api_client, session)
        self.boxscore_collector = BoxScoreCollector(api_client, session)
        self.stint_deriver = StintDeriver()
        self.validator = DataValidator()

    def full_historical_load(
        self,
        seasons: list[str],
        resume: bool = True,
    ) -> PipelineResult:
        """Complete historical data collection for specified seasons.

        Order of operations per season:
        1. Collect all games for season
        2. Collect team rosters (populates players)
        3. For each game batch:
           a. Collect play-by-play
           b. Collect shots
           c. Collect box scores
           d. Derive stints
           e. Commit batch
           f. Save checkpoint

        Args:
            seasons: List of season strings (e.g., ["2019-20", "2020-21"]).
            resume: Whether to resume from checkpoint.

        Returns:
            PipelineResult with statistics.
        """
        start_time = time.time()
        result = PipelineResult(status=PipelineStatus.RUNNING)
        pipeline_name = "full_historical_load"

        # Check for existing checkpoint
        checkpoint = None
        if resume:
            checkpoint = self.checkpoint.load(pipeline_name)
            if checkpoint and checkpoint.status == "completed":
                self.logger.info("Previous run completed, starting fresh")
                checkpoint = None

        try:
            # Initialize checkpoint if needed
            if checkpoint is None:
                from nba_model.data.checkpoint import Checkpoint

                checkpoint = Checkpoint(
                    pipeline_name=pipeline_name,
                    status="running",
                    total_processed=0,
                )
                self.checkpoint.save(checkpoint)

            # Process each season
            start_season_idx = 0
            if checkpoint.last_season:
                try:
                    start_season_idx = seasons.index(checkpoint.last_season)
                    self.logger.info(f"Resuming from season {checkpoint.last_season}")
                except ValueError:
                    start_season_idx = 0

            for season in seasons[start_season_idx:]:
                self.logger.info(f"Processing season {season}")
                checkpoint.last_season = season
                self.checkpoint.save(checkpoint)

                # Collect teams and rosters first
                self._collect_teams_and_rosters(season)

                # Get games for season
                games = self._collect_games_for_season(season)
                if not games:
                    self.logger.warning(f"No games found for season {season}")
                    continue

                # Determine starting point within season
                start_idx = 0
                if checkpoint.last_game_id:
                    game_ids = [g.game_id for g in games]
                    try:
                        start_idx = game_ids.index(checkpoint.last_game_id) + 1
                        self.logger.info(
                            f"Resuming from game {checkpoint.last_game_id} "
                            f"(index {start_idx})"
                        )
                    except ValueError:
                        start_idx = 0

                # Process games in batches
                remaining_games = games[start_idx:]
                total_games = len(remaining_games)

                for batch_start in range(0, total_games, self.batch_size):
                    batch_end = min(batch_start + self.batch_size, total_games)
                    batch_games = remaining_games[batch_start:batch_end]

                    self.logger.info(
                        f"Processing batch {batch_start // self.batch_size + 1}: "
                        f"games {batch_start + 1}-{batch_end} of {total_games}"
                    )

                    # Collect batch
                    batch_result = self._collect_game_batch(
                        [g.game_id for g in batch_games]
                    )

                    # Validate batch
                    validation = self._validate_batch(batch_result)
                    if not validation.valid:
                        self.logger.warning(
                            f"Batch validation warnings: {validation.errors}"
                        )
                        # Continue despite warnings, just log errors

                    # Commit batch
                    self._commit_batch(batch_result)

                    # Update statistics
                    result.plays_collected += len(batch_result.plays)
                    result.shots_collected += len(batch_result.shots)
                    result.stints_derived += len(batch_result.stints)
                    result.games_processed += len(batch_result.game_ids) - len(
                        batch_result.errors
                    )

                    for game_id, error in batch_result.errors:
                        result.errors.append(f"Game {game_id}: {error}")

                    # Update checkpoint
                    checkpoint.last_game_id = batch_games[-1].game_id
                    checkpoint.total_processed = result.games_processed
                    self.checkpoint.save(checkpoint)

                result.seasons_processed.append(season)
                checkpoint.last_game_id = None  # Reset for next season

            # Mark complete
            result.status = PipelineStatus.COMPLETED
            checkpoint.status = "completed"
            self.checkpoint.save(checkpoint)

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            result.status = PipelineStatus.FAILED
            result.errors.append(str(e))

            if checkpoint:
                checkpoint.status = "failed"
                checkpoint.error_message = str(e)
                self.checkpoint.save(checkpoint)

        result.duration_seconds = time.time() - start_time
        return result

    def incremental_update(self) -> PipelineResult:
        """Daily update for new completed games.

        1. Find games completed since last update
        2. Collect all data for new games
        3. Derive stints
        4. Update checkpoint

        Returns:
            PipelineResult with statistics.
        """
        from nba_model.data.models import Game

        start_time = time.time()
        result = PipelineResult(status=PipelineStatus.RUNNING)
        pipeline_name = "incremental_update"

        try:
            # Find most recent game in database
            last_game = (
                self.session.query(Game)
                .filter(Game.status == "completed")
                .order_by(Game.game_date.desc())
                .first()
            )

            if last_game:
                last_date = last_game.game_date
                self.logger.info(f"Last game date in DB: {last_date}")
            else:
                last_date = date(2024, 1, 1)  # Default if no games
                self.logger.info("No games in DB, using default start date")

            # Get current season
            current_season = self._get_current_season()

            # Fetch new games from API
            new_games = self.games_collector.collect_date_range(
                start_date=last_date,
                end_date=date.today(),
            )

            # Filter to only completed games not in DB
            existing_ids = set(
                row[0]
                for row in self.session.query(Game.game_id)
                .filter(Game.game_date >= last_date)
                .all()
            )

            games_to_process = [
                g for g in new_games if g.game_id not in existing_ids and g.status == "completed"
            ]

            self.logger.info(f"Found {len(games_to_process)} new games to process")

            if not games_to_process:
                result.status = PipelineStatus.COMPLETED
                result.duration_seconds = time.time() - start_time
                return result

            # Save games first
            for game in games_to_process:
                self.session.merge(game)
            self.session.commit()

            # Process in batches
            for batch_start in range(0, len(games_to_process), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(games_to_process))
                batch_games = games_to_process[batch_start:batch_end]

                batch_result = self._collect_game_batch(
                    [g.game_id for g in batch_games]
                )

                self._commit_batch(batch_result)

                result.plays_collected += len(batch_result.plays)
                result.shots_collected += len(batch_result.shots)
                result.stints_derived += len(batch_result.stints)
                result.games_processed += len(batch_result.game_ids) - len(
                    batch_result.errors
                )

            result.status = PipelineStatus.COMPLETED

        except Exception as e:
            self.logger.error(f"Incremental update failed: {e}")
            result.status = PipelineStatus.FAILED
            result.errors.append(str(e))

        result.duration_seconds = time.time() - start_time
        return result

    def repair_games(self, game_ids: list[str]) -> PipelineResult:
        """Re-fetch specific games (for data repair).

        Deletes existing data for the specified games and re-collects.

        Args:
            game_ids: List of game IDs to re-fetch.

        Returns:
            PipelineResult.
        """
        from nba_model.data.models import GameStats, Play, PlayerGameStats, Shot, Stint

        start_time = time.time()
        result = PipelineResult(status=PipelineStatus.RUNNING)

        try:
            self.logger.info(f"Repairing {len(game_ids)} games")

            # Delete existing data for these games
            for game_id in game_ids:
                self.session.query(Play).filter(Play.game_id == game_id).delete()
                self.session.query(Shot).filter(Shot.game_id == game_id).delete()
                self.session.query(Stint).filter(Stint.game_id == game_id).delete()
                self.session.query(GameStats).filter(
                    GameStats.game_id == game_id
                ).delete()
                self.session.query(PlayerGameStats).filter(
                    PlayerGameStats.game_id == game_id
                ).delete()

            self.session.commit()

            # Re-collect
            batch_result = self._collect_game_batch(game_ids)
            self._commit_batch(batch_result)

            result.plays_collected = len(batch_result.plays)
            result.shots_collected = len(batch_result.shots)
            result.stints_derived = len(batch_result.stints)
            result.games_processed = len(game_ids) - len(batch_result.errors)

            for game_id, error in batch_result.errors:
                result.errors.append(f"Game {game_id}: {error}")

            result.status = PipelineStatus.COMPLETED

        except Exception as e:
            self.logger.error(f"Repair failed: {e}")
            result.status = PipelineStatus.FAILED
            result.errors.append(str(e))

        result.duration_seconds = time.time() - start_time
        return result

    def _collect_teams_and_rosters(self, season: str) -> None:
        """Collect teams and rosters for a season.

        Args:
            season: Season string.
        """
        from nba_model.data.collectors.players import TEAM_DATA
        from nba_model.data.models import Season, Team

        self.logger.info(f"Collecting teams and rosters for {season}")

        # Ensure teams exist
        teams = self.players_collector.collect_teams()
        for team in teams:
            self.session.merge(team)

        # Ensure season exists
        season_obj = self.session.query(Season).filter(Season.season_id == season).first()
        if not season_obj:
            # Create minimal season record
            year = int(season[:4])
            season_obj = Season(
                season_id=season,
                start_date=date(year, 10, 1),
                end_date=date(year + 1, 6, 30),
            )
            self.session.add(season_obj)

        self.session.commit()

        # Collect rosters
        team_ids = list(TEAM_DATA.keys())
        try:
            players, player_seasons = self.players_collector.collect_rosters(
                season=season,
                team_ids=team_ids,
            )

            for player in players:
                self.session.merge(player)
            for ps in player_seasons:
                self.session.merge(ps)

            self.session.commit()
            self.logger.info(
                f"Collected {len(players)} players, {len(player_seasons)} player-seasons"
            )
        except Exception as e:
            self.logger.warning(f"Error collecting rosters: {e}")
            self.session.rollback()

    def _collect_games_for_season(self, season: str) -> list:
        """Collect games for a season.

        Args:
            season: Season string.

        Returns:
            List of Game objects.
        """
        self.logger.info(f"Collecting games for season {season}")

        try:
            games = self.games_collector.collect_season(season)

            # Save games to database
            for game in games:
                self.session.merge(game)
            self.session.commit()

            self.logger.info(f"Collected {len(games)} games for {season}")
            return games

        except Exception as e:
            self.logger.error(f"Error collecting games for {season}: {e}")
            self.session.rollback()
            return []

    def _collect_game_batch(self, game_ids: list[str]) -> BatchResult:
        """Collect all data for a batch of games.

        Collects: plays, shots, box scores, stints.
        Logs per-game status with color-coded tags.

        Args:
            game_ids: List of game IDs.

        Returns:
            BatchResult with collected data.
        """
        from nba_model.data.models import Game

        batch = BatchResult(game_ids=game_ids.copy())

        for game_id in game_ids:
            # Track what was collected for this game
            game_plays = []
            game_shots = []
            game_stats_list = []
            game_player_stats = []
            game_stints = []
            warnings = []
            has_error = False

            try:
                # Get game info for team IDs
                game = (
                    self.session.query(Game).filter(Game.game_id == game_id).first()
                )

                # Collect play-by-play
                game_plays = self.pbp_collector.collect_game(game_id)
                batch.plays.extend(game_plays)

                # Collect shots
                try:
                    game_shots = self.shots_collector.collect_game(game_id)
                    batch.shots.extend(game_shots)
                except Exception as e:
                    warnings.append(f"shots: {e}")

                # Collect box scores
                try:
                    game_stats_list, game_player_stats = (
                        self.boxscore_collector.collect_game(game_id)
                    )
                    batch.game_stats.extend(game_stats_list)
                    batch.player_game_stats.extend(game_player_stats)
                except Exception as e:
                    warnings.append(f"boxscores: {e}")

                # Derive stints from plays
                if game_plays and game:
                    try:
                        game_stints = self.stint_deriver.derive_stints(
                            game_plays,
                            game_id,
                            home_team_id=game.home_team_id,
                            away_team_id=game.away_team_id,
                        )
                        batch.stints.extend(game_stints)
                    except Exception as e:
                        warnings.append(f"stints: {e}")

            except Exception as e:
                has_error = True
                batch.errors.append((game_id, str(e)))
                self.logger.error(
                    f"{FAIL} Game {game_id}: {e}"
                )
                continue

            # Log per-game status with color-coded tags
            if has_error:
                # Already logged above
                pass
            elif warnings:
                # Partial success - some data missing
                self.logger.warning(
                    f"{WARN} Game {game_id}: "
                    f"{len(game_plays)} plays, {len(game_shots)} shots, "
                    f"{len(game_player_stats)} player stats, {len(game_stints)} stints "
                    f"(missing: {', '.join(warnings)})"
                )
            else:
                # Full success
                self.logger.info(
                    f"{SUCCESS} Game {game_id}: "
                    f"{len(game_plays)} plays, {len(game_shots)} shots, "
                    f"{len(game_player_stats)} player stats, {len(game_stints)} stints"
                )

        return batch

    def _validate_batch(self, batch: BatchResult) -> "ValidationResult":
        """Validate batch data before commit.

        Args:
            batch: BatchResult to validate.

        Returns:
            ValidationResult.
        """
        return self.validator.validate_batch(
            plays=batch.plays,
            shots=batch.shots,
            game_stats=batch.game_stats,
            player_game_stats=batch.player_game_stats,
        )

    def _commit_batch(self, batch: BatchResult) -> bool:
        """Commit batch to database.

        Args:
            batch: BatchResult to commit.

        Returns:
            True if commit succeeded, False otherwise.
        """
        try:
            # Add plays
            for play in batch.plays:
                self.session.merge(play)

            # Add shots
            for shot in batch.shots:
                self.session.merge(shot)

            # Add game stats
            for gs in batch.game_stats:
                self.session.merge(gs)

            # Add player game stats
            for pgs in batch.player_game_stats:
                self.session.merge(pgs)

            # Add stints
            for stint in batch.stints:
                self.session.merge(stint)

            self.session.commit()

            self.logger.info(
                f"{SUCCESS} Batch persisted: "
                f"{len(batch.plays)} plays, {len(batch.shots)} shots, "
                f"{len(batch.game_stats)} game stats, "
                f"{len(batch.player_game_stats)} player stats, "
                f"{len(batch.stints)} stints"
            )
            return True

        except Exception as e:
            self.logger.error(
                f"{FAIL} Batch persistence failed: {e}"
            )
            self.session.rollback()
            return False
            self.session.rollback()
            raise

    def _get_current_season(self) -> str:
        """Determine current NBA season.

        Returns:
            Season string (e.g., "2023-24").
        """
        today = date.today()
        year = today.year

        # NBA season starts in October
        if today.month >= 10:
            return f"{year}-{str(year + 1)[-2:]}"
        else:
            return f"{year - 1}-{str(year)[-2:]}"
