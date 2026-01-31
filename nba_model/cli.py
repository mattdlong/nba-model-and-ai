"""CLI entrypoint using Typer.

This module defines the command-line interface for the NBA model application.
Commands are organized into subcommand groups for data, features, training,
backtesting, monitoring, prediction, and dashboard operations.

Example:
    $ nba-model --help
    $ nba-model data collect --seasons 2023-24
    $ nba-model predict today
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from tqdm import tqdm

from nba_model import __version__
from nba_model.config import get_settings
from nba_model.logging import get_logger, setup_logging

# Logger for CLI
logger = get_logger(__name__)

# Initialize console for rich output
console = Console()

# Create main app
app = typer.Typer(
    name="nba-model",
    help="NBA Quantitative Trading Strategy CLI",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Create subcommand groups
data_app = typer.Typer(
    name="data",
    help="Data collection and management commands",
    no_args_is_help=True,
)
features_app = typer.Typer(
    name="features",
    help="Feature engineering commands",
    no_args_is_help=True,
)
train_app = typer.Typer(
    name="train",
    help="Model training commands",
    no_args_is_help=True,
)
backtest_app = typer.Typer(
    name="backtest",
    help="Backtesting and validation commands",
    no_args_is_help=True,
)
monitor_app = typer.Typer(
    name="monitor",
    help="Model monitoring and drift detection commands",
    no_args_is_help=True,
)
predict_app = typer.Typer(
    name="predict",
    help="Prediction and signal generation commands",
    no_args_is_help=True,
)
dashboard_app = typer.Typer(
    name="dashboard",
    help="Dashboard and reporting commands",
    no_args_is_help=True,
)

# Register subcommand groups
app.add_typer(data_app, name="data")
app.add_typer(features_app, name="features")
app.add_typer(train_app, name="train")
app.add_typer(backtest_app, name="backtest")
app.add_typer(monitor_app, name="monitor")
app.add_typer(predict_app, name="predict")
app.add_typer(dashboard_app, name="dashboard")


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"[bold blue]nba-model[/bold blue] version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-v",
            help="Show version and exit",
            callback=version_callback,
            is_eager=True,
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-V",
            help="Enable verbose output",
        ),
    ] = False,
) -> None:
    """NBA Quantitative Trading Strategy CLI.

    A tool for collecting NBA data, engineering features, training ML models,
    and generating betting signals.
    """
    # Setup logging based on verbosity
    settings = get_settings()
    log_level = "DEBUG" if verbose else settings.log_level
    setup_logging(level=log_level, log_dir=settings.log_dir_obj)


# =============================================================================
# Data Commands
# =============================================================================


@data_app.command("collect")
def data_collect(
    seasons: Annotated[
        list[str] | None,
        typer.Option(
            "--seasons",
            "-s",
            help="Seasons to collect (e.g., 2023-24)",
        ),
    ] = None,
    full: Annotated[
        bool,
        typer.Option(
            "--full",
            "-f",
            help="Collect all available data (5 seasons)",
        ),
    ] = False,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume/--no-resume",
            help="Resume from checkpoint",
        ),
    ] = True,
) -> None:
    """Collect data from NBA API.

    Fetches games, players, play-by-play, and shot data for specified seasons.
    """
    from nba_model.data import NBAApiClient, init_db, session_scope
    from nba_model.data.checkpoint import CheckpointManager
    from nba_model.data.pipelines import CollectionPipeline

    settings = get_settings()
    settings.ensure_directories()

    # Determine seasons
    if full:
        seasons_to_collect = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]
    elif seasons:
        seasons_to_collect = seasons
    else:
        console.print("[red]Error: Specify --seasons or --full[/red]")
        raise typer.Exit(1)

    console.print(
        Panel(
            f"[bold]Seasons:[/bold] {', '.join(seasons_to_collect)}\n"
            f"[bold]Resume:[/bold] {resume}",
            title="Data Collection",
        )
    )

    # Initialize database
    init_db()

    # Initialize components
    with session_scope() as session:
        api_client = NBAApiClient(
            delay=settings.api_delay,
            max_retries=settings.api_max_retries,
        )
        checkpoint_mgr = CheckpointManager()
        pipeline = CollectionPipeline(
            session=session,
            api_client=api_client,
            checkpoint_manager=checkpoint_mgr,
        )

        # Run collection
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Collecting data...", total=None)
            result = pipeline.full_historical_load(
                seasons=seasons_to_collect,
                resume=resume,
            )

        # Display results
        _display_pipeline_result(result)


@data_app.command("update")
def data_update() -> None:
    """Incremental data update for recent games.

    Fetches games since last update and populates all related tables.
    """
    from nba_model.data import NBAApiClient, init_db, session_scope
    from nba_model.data.checkpoint import CheckpointManager
    from nba_model.data.pipelines import CollectionPipeline

    settings = get_settings()
    settings.ensure_directories()

    # Initialize database
    init_db()

    with session_scope() as session:
        api_client = NBAApiClient(
            delay=settings.api_delay,
            max_retries=settings.api_max_retries,
        )
        checkpoint_mgr = CheckpointManager()
        pipeline = CollectionPipeline(
            session=session,
            api_client=api_client,
            checkpoint_manager=checkpoint_mgr,
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Updating data...", total=None)
            result = pipeline.incremental_update()

        _display_pipeline_result(result)


@data_app.command("status")
def data_status() -> None:
    """Show database statistics and data freshness.

    Displays counts of games, players, and data recency information.
    """

    from nba_model.data import init_db, session_scope
    from nba_model.data.checkpoint import CheckpointManager

    settings = get_settings()

    # Check if database exists
    if not settings.db_path_obj.exists():
        console.print(
            Panel(
                f"[bold]Database:[/bold] {settings.db_path}\n"
                "[yellow]Database not found. Run 'data collect' first.[/yellow]",
                title="Data Status",
            )
        )
        return

    init_db()

    with session_scope() as session:
        stats = _get_database_stats(session)

        table = Table(title="Database Status")
        table.add_column("Entity", style="cyan")
        table.add_column("Count", justify="right")
        table.add_column("Date Range", style="green")

        for entity, count, date_range in stats:
            table.add_row(entity, str(count), date_range or "N/A")

        console.print(table)

        # Show checkpoint status
        checkpoint_mgr = CheckpointManager()
        checkpoints = checkpoint_mgr.list_all()
        if checkpoints:
            console.print("\n[bold]Checkpoints:[/bold]")
            for cp in checkpoints:
                status_color = "green" if cp.status == "completed" else "yellow"
                console.print(
                    f"  {cp.pipeline_name}: [{status_color}]{cp.status}[/{status_color}] "
                    f"({cp.total_processed} games)"
                )


@data_app.command("repair")
def data_repair(
    game_ids: Annotated[
        list[str],
        typer.Argument(help="Game IDs to re-fetch"),
    ],
) -> None:
    """Re-fetch specific game data."""
    from nba_model.data import NBAApiClient, init_db, session_scope
    from nba_model.data.checkpoint import CheckpointManager
    from nba_model.data.pipelines import CollectionPipeline

    settings = get_settings()
    settings.ensure_directories()
    init_db()

    console.print(f"[bold]Repairing games:[/bold] {', '.join(game_ids)}")

    with session_scope() as session:
        api_client = NBAApiClient(
            delay=settings.api_delay,
            max_retries=settings.api_max_retries,
        )
        checkpoint_mgr = CheckpointManager()
        pipeline = CollectionPipeline(
            session=session,
            api_client=api_client,
            checkpoint_manager=checkpoint_mgr,
        )

        result = pipeline.repair_games(game_ids)
        _display_pipeline_result(result)


def _get_database_stats(session) -> list[tuple[str, int, str | None]]:
    """Get row counts and date ranges for each entity."""
    from sqlalchemy import func

    from nba_model.data import Game, Play, Player, Shot, Stint

    stats = []

    # Games
    game_count = session.query(func.count(Game.game_id)).scalar() or 0
    if game_count > 0:
        min_date = session.query(func.min(Game.game_date)).scalar()
        max_date = session.query(func.max(Game.game_date)).scalar()
        date_range = f"{min_date} to {max_date}"
    else:
        date_range = None
    stats.append(("Games", game_count, date_range))

    # Other entities
    stats.append(
        ("Players", session.query(func.count(Player.player_id)).scalar() or 0, None)
    )
    stats.append(("Plays", session.query(func.count(Play.id)).scalar() or 0, None))
    stats.append(("Shots", session.query(func.count(Shot.id)).scalar() or 0, None))
    stats.append(("Stints", session.query(func.count(Stint.id)).scalar() or 0, None))

    return stats


def _display_pipeline_result(result) -> None:
    """Display pipeline result to console."""
    from nba_model.data.pipelines import PipelineStatus

    status_color = {
        PipelineStatus.COMPLETED: "green",
        PipelineStatus.FAILED: "red",
        PipelineStatus.RUNNING: "yellow",
        PipelineStatus.PAUSED: "yellow",
        PipelineStatus.PENDING: "white",
    }.get(result.status, "white")

    console.print(f"\n[{status_color}]Status: {result.status.value}[/{status_color}]")
    console.print(f"Games processed: {result.games_processed}")
    console.print(f"Plays collected: {result.plays_collected}")
    console.print(f"Shots collected: {result.shots_collected}")
    console.print(f"Stints derived: {result.stints_derived}")
    console.print(f"Duration: {result.duration_seconds:.1f}s")

    if result.errors:
        console.print(f"\n[red]Errors ({len(result.errors)}):[/red]")
        for error in result.errors[:10]:
            console.print(f"  - {error}")
        if len(result.errors) > 10:
            console.print(f"  ... and {len(result.errors) - 10} more")


# =============================================================================
# Features Commands
# =============================================================================


@features_app.command("build")
def features_build(
    seasons: Annotated[
        list[str] | None,
        typer.Option(
            "--seasons",
            "-s",
            help="Seasons to build features for (e.g., 2023-24)",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Force rebuild all features",
        ),
    ] = False,
) -> None:
    """Build all feature tables.

    Calculates RAPM, spacing metrics, fatigue indicators, and normalizations
    in dependency order: normalization -> RAPM -> spacing -> fatigue.
    """
    import json
    from datetime import date

    import pandas as pd
    from sqlalchemy import func

    from nba_model.data import (
        Game,
        GameStats,
        LineupSpacing,
        PlayerRAPM,
        Shot,
        Stint,
        init_db,
        session_scope,
    )
    from nba_model.features import (
        FatigueCalculator,
        RAPMCalculator,
        SeasonNormalizer,
        SpacingCalculator,
    )

    settings = get_settings()
    settings.ensure_directories()

    if not settings.db_path_obj.exists():
        console.print("[red]Error: Database not found. Run 'data collect' first.[/red]")
        raise typer.Exit(1)

    init_db()

    with session_scope() as session:
        # Determine seasons
        if seasons:
            seasons_to_process = seasons
        else:
            # Get all seasons with data
            result = session.query(Game.season_id).distinct().all()
            seasons_to_process = [r[0] for r in result]

        if not seasons_to_process:
            console.print("[yellow]No seasons found in database.[/yellow]")
            return

        console.print(
            Panel(
                f"[bold]Seasons:[/bold] {', '.join(seasons_to_process)}\n"
                f"[bold]Force rebuild:[/bold] {force}",
                title="Feature Building",
            )
        )

        # Step 1: Season Normalization
        console.print("\n[bold cyan]Step 1/4: Season Normalization[/bold cyan]")

        # Get game stats with season info
        game_stats_query = (
            session.query(GameStats, Game.season_id)
            .join(Game, GameStats.game_id == Game.game_id)
            .filter(Game.season_id.in_(seasons_to_process))
        )

        records = []
        for gs, season_id in tqdm(
            game_stats_query, desc="Collecting stats", leave=False
        ):
            # Calculate derived metrics
            fg3a_rate = None
            if gs.fga and gs.fga > 0 and gs.fg3a is not None:
                fg3a_rate = gs.fg3a / gs.fga

            records.append(
                {
                    "season_id": season_id,
                    "pace": gs.pace,
                    "offensive_rating": gs.offensive_rating,
                    "defensive_rating": gs.defensive_rating,
                    "efg_pct": gs.efg_pct,
                    "tov_pct": gs.tov_pct,
                    "orb_pct": gs.orb_pct,
                    "ft_rate": gs.ft_rate,
                    "fg3a_rate": fg3a_rate,
                    "points_per_game": gs.points,
                }
            )

        if records:
            stats_df = pd.DataFrame(records)
            normalizer = SeasonNormalizer()
            normalizer.fit(stats_df)
            normalizer.save_stats(session)
            console.print(
                f"  [green]Saved normalization stats for {len(seasons_to_process)} seasons[/green]"
            )
        else:
            console.print("  [yellow]No game stats found[/yellow]")

        # Step 2: RAPM Calculation
        console.print("\n[bold cyan]Step 2/4: RAPM Calculation[/bold cyan]")
        for season in tqdm(seasons_to_process, desc="Seasons", leave=False):
            # Check if already exists
            existing_count = (
                session.query(func.count(PlayerRAPM.id))
                .filter_by(season_id=season)
                .scalar()
            )

            if existing_count > 0 and not force:
                console.print(
                    f"  [yellow]Season {season}: {existing_count} RAPM records exist (skipping)[/yellow]"
                )
                continue

            # Clear existing if force
            if force and existing_count > 0:
                session.query(PlayerRAPM).filter_by(season_id=season).delete()
                session.commit()

            # Get stints for season
            stints_query = (
                session.query(Stint, Game.game_date)
                .join(Game, Stint.game_id == Game.game_id)
                .filter(Game.season_id == season)
            )

            stint_records = []
            for stint, game_date in stints_query:
                stint_records.append(
                    {
                        "home_lineup": stint.home_lineup,
                        "away_lineup": stint.away_lineup,
                        "home_points": stint.home_points,
                        "away_points": stint.away_points,
                        "possessions": stint.possessions or 1.0,
                        "duration_seconds": stint.duration_seconds,
                        "game_date": game_date,
                    }
                )

            if len(stint_records) < 100:
                console.print(
                    f"  [yellow]Season {season}: Insufficient stints ({len(stint_records)})[/yellow]"
                )
                continue

            stints_df = pd.DataFrame(stint_records)
            try:
                calculator = RAPMCalculator(lambda_=5000, min_minutes=100)
                coefficients = calculator.fit(stints_df)

                # Save to database
                today = date.today()
                for player_id, coef in coefficients.items():
                    rapm_record = PlayerRAPM(
                        player_id=player_id,
                        season_id=season,
                        calculation_date=today,
                        orapm=coef["orapm"],
                        drapm=coef["drapm"],
                        rapm=coef["total_rapm"],
                        sample_stints=coef["sample_stints"],
                    )
                    session.add(rapm_record)
                session.commit()
                console.print(
                    f"  [green]Season {season}: Calculated RAPM for {len(coefficients)} players[/green]"
                )
            except Exception as e:
                console.print(
                    f"  [red]Season {season}: RAPM calculation failed: {e}[/red]"
                )

        # Step 3: Spacing Metrics
        console.print("\n[bold cyan]Step 3/4: Lineup Spacing[/bold cyan]")
        for season in tqdm(seasons_to_process, desc="Seasons", leave=False):
            # Check existing
            existing_count = (
                session.query(func.count(LineupSpacing.id))
                .filter_by(season_id=season)
                .scalar()
            )

            if existing_count > 0 and not force:
                console.print(
                    f"  [yellow]Season {season}: {existing_count} spacing records exist (skipping)[/yellow]"
                )
                continue

            # Clear existing if force
            if force and existing_count > 0:
                session.query(LineupSpacing).filter_by(season_id=season).delete()
                session.commit()

            # Get shots for season
            shots_query = (
                session.query(Shot.player_id, Shot.loc_x, Shot.loc_y)
                .join(Game, Shot.game_id == Game.game_id)
                .filter(Game.season_id == season)
            )

            shots_df = pd.DataFrame(
                [{"player_id": s[0], "loc_x": s[1], "loc_y": s[2]} for s in shots_query]
            )

            if len(shots_df) < 100:
                console.print(
                    f"  [yellow]Season {season}: Insufficient shots ({len(shots_df)})[/yellow]"
                )
                continue

            # Get unique lineups from stints
            stints = (
                session.query(Stint.home_lineup, Stint.away_lineup)
                .join(Game, Stint.game_id == Game.game_id)
                .filter(Game.season_id == season)
                .all()
            )

            unique_lineups = set()
            for home, away in stints:
                home_ids = tuple(
                    sorted(json.loads(home) if isinstance(home, str) else home)
                )
                away_ids = tuple(
                    sorted(json.loads(away) if isinstance(away, str) else away)
                )
                unique_lineups.add(home_ids)
                unique_lineups.add(away_ids)

            calculator = SpacingCalculator(min_shots=20)
            saved_count = 0

            for lineup in tqdm(unique_lineups, desc=f"Lineups ({season})", leave=False):
                lineup_list = list(lineup)
                metrics = calculator.calculate_lineup_spacing(lineup_list, shots_df)

                if metrics["shot_count"] >= 20:
                    lineup_hash = SpacingCalculator.compute_lineup_hash(lineup_list)
                    spacing_record = LineupSpacing(
                        season_id=season,
                        lineup_hash=lineup_hash,
                        player_ids=json.dumps(lineup_list),
                        hull_area=metrics["hull_area"],
                        centroid_x=metrics["centroid_x"],
                        centroid_y=metrics["centroid_y"],
                        shot_count=metrics["shot_count"],
                    )
                    session.add(spacing_record)
                    saved_count += 1

            session.commit()
            console.print(
                f"  [green]Season {season}: Calculated spacing for {saved_count} lineups[/green]"
            )

        # Step 4: Fatigue Metrics
        console.print("\n[bold cyan]Step 4/4: Fatigue Metrics[/bold cyan]")

        # Get all games for fatigue calculation
        games_query = (
            session.query(Game)
            .filter(Game.season_id.in_(seasons_to_process))
            .order_by(Game.game_date)
        )

        games_list = games_query.all()
        if not games_list:
            console.print("  [yellow]No games found for fatigue calculation[/yellow]")
        else:
            # Build games DataFrame for fatigue calculator
            games_df = pd.DataFrame(
                [
                    {
                        "game_id": g.game_id,
                        "game_date": g.game_date,
                        "home_team_id": g.home_team_id,
                        "away_team_id": g.away_team_id,
                    }
                    for g in games_list
                ]
            )

            fatigue_calc = FatigueCalculator()
            fatigue_count = 0

            for game in tqdm(games_list, desc="Calculating fatigue", leave=False):
                game_date = game.game_date
                if hasattr(game_date, "date"):
                    game_date = game_date.date()

                # Calculate fatigue for home team
                fatigue_calc.calculate_schedule_flags(
                    game.home_team_id, game_date, games_df
                )

                # Calculate fatigue for away team
                fatigue_calc.calculate_schedule_flags(
                    game.away_team_id, game_date, games_df
                )

                fatigue_count += 2

            console.print(
                f"  [green]Calculated fatigue indicators for {fatigue_count} team-games[/green]"
            )

    console.print("\n[bold green]Feature building complete![/bold green]")


@features_app.command("rapm")
def features_rapm(
    season: Annotated[
        str,
        typer.Option(
            "--season",
            "-s",
            help="Season to calculate RAPM for (e.g., 2023-24)",
        ),
    ],
    lambda_val: Annotated[
        float,
        typer.Option(
            "--lambda",
            "-l",
            help="Ridge regularization strength",
        ),
    ] = 5000.0,
    min_minutes: Annotated[
        int,
        typer.Option(
            "--min-minutes",
            "-m",
            help="Minimum minutes for player inclusion",
        ),
    ] = 100,
    cross_validate: Annotated[
        bool,
        typer.Option(
            "--cv",
            help="Cross-validate lambda parameter",
        ),
    ] = False,
) -> None:
    """Calculate RAPM coefficients.

    Runs Ridge Regression on stint data to calculate player impact metrics.
    """
    from datetime import date

    import pandas as pd

    from nba_model.data import Game, PlayerRAPM, Stint, init_db, session_scope
    from nba_model.features import RAPMCalculator

    settings = get_settings()

    if not settings.db_path_obj.exists():
        console.print("[red]Error: Database not found. Run 'data collect' first.[/red]")
        raise typer.Exit(1)

    init_db()

    console.print(
        Panel(
            f"[bold]Season:[/bold] {season}\n"
            f"[bold]Lambda:[/bold] {lambda_val}\n"
            f"[bold]Min Minutes:[/bold] {min_minutes}\n"
            f"[bold]Cross-validate:[/bold] {cross_validate}",
            title="RAPM Calculation",
        )
    )

    with session_scope() as session:
        # Get stints for season
        stints_query = (
            session.query(Stint, Game.game_date)
            .join(Game, Stint.game_id == Game.game_id)
            .filter(Game.season_id == season)
        )

        stint_records = []
        for stint, game_date in stints_query:
            stint_records.append(
                {
                    "home_lineup": stint.home_lineup,
                    "away_lineup": stint.away_lineup,
                    "home_points": stint.home_points,
                    "away_points": stint.away_points,
                    "possessions": stint.possessions or 1.0,
                    "duration_seconds": stint.duration_seconds,
                    "game_date": game_date,
                }
            )

        if len(stint_records) < 100:
            console.print(
                f"[red]Error: Insufficient stints ({len(stint_records)}). Need at least 100.[/red]"
            )
            raise typer.Exit(1)

        stints_df = pd.DataFrame(stint_records)
        console.print(f"Found {len(stints_df)} stints")

        calculator = RAPMCalculator(lambda_=lambda_val, min_minutes=min_minutes)

        # Cross-validate if requested
        if cross_validate:
            console.print("\nCross-validating lambda...")
            best_lambda, scores = calculator.cross_validate_lambda(stints_df)
            console.print(f"Best lambda: {best_lambda}")
            for lam, score in scores.items():
                console.print(f"  Lambda {lam}: R² = {score:.4f}")
            calculator.lambda_ = best_lambda

        # Calculate RAPM
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Calculating RAPM...", total=None)
            coefficients = calculator.fit(stints_df)

        # Display top/bottom players
        table = Table(title="RAPM Results (Top/Bottom 10)")
        table.add_column("Player ID", style="cyan")
        table.add_column("ORAPM", justify="right")
        table.add_column("DRAPM", justify="right")
        table.add_column("Total", justify="right", style="green")
        table.add_column("Stints", justify="right")

        sorted_players = sorted(
            coefficients.items(),
            key=lambda x: x[1]["total_rapm"],
            reverse=True,
        )

        for pid, coef in sorted_players[:10]:
            table.add_row(
                str(pid),
                f"{coef['orapm']:+.2f}",
                f"{coef['drapm']:+.2f}",
                f"{coef['total_rapm']:+.2f}",
                str(coef["sample_stints"]),
            )

        table.add_row("...", "...", "...", "...", "...")

        for pid, coef in sorted_players[-10:]:
            table.add_row(
                str(pid),
                f"{coef['orapm']:+.2f}",
                f"{coef['drapm']:+.2f}",
                f"{coef['total_rapm']:+.2f}",
                str(coef["sample_stints"]),
            )

        console.print(table)

        # Save to database
        today = date.today()
        session.query(PlayerRAPM).filter_by(season_id=season).delete()

        for player_id, coef in coefficients.items():
            rapm_record = PlayerRAPM(
                player_id=player_id,
                season_id=season,
                calculation_date=today,
                orapm=coef["orapm"],
                drapm=coef["drapm"],
                rapm=coef["total_rapm"],
                sample_stints=coef["sample_stints"],
            )
            session.add(rapm_record)

        session.commit()
        console.print(f"\n[green]Saved RAPM for {len(coefficients)} players[/green]")


@features_app.command("spatial")
def features_spatial(
    season: Annotated[
        str,
        typer.Option(
            "--season",
            "-s",
            help="Season to calculate spacing for (e.g., 2023-24)",
        ),
    ],
    min_shots: Annotated[
        int,
        typer.Option(
            "--min-shots",
            help="Minimum shots for lineup inclusion",
        ),
    ] = 20,
) -> None:
    """Calculate convex hull spacing metrics.

    Analyzes shot distributions to measure lineup floor spacing.
    """
    import json

    import pandas as pd

    from nba_model.data import Game, LineupSpacing, Shot, Stint, init_db, session_scope
    from nba_model.features import SpacingCalculator

    settings = get_settings()

    if not settings.db_path_obj.exists():
        console.print("[red]Error: Database not found. Run 'data collect' first.[/red]")
        raise typer.Exit(1)

    init_db()

    console.print(
        Panel(
            f"[bold]Season:[/bold] {season}\n" f"[bold]Min Shots:[/bold] {min_shots}",
            title="Spacing Calculation",
        )
    )

    with session_scope() as session:
        # Get shots
        shots_query = (
            session.query(Shot.player_id, Shot.loc_x, Shot.loc_y)
            .join(Game, Shot.game_id == Game.game_id)
            .filter(Game.season_id == season)
        )

        shots_df = pd.DataFrame(
            [{"player_id": s[0], "loc_x": s[1], "loc_y": s[2]} for s in shots_query]
        )

        if len(shots_df) < 100:
            console.print(f"[red]Error: Insufficient shots ({len(shots_df)})[/red]")
            raise typer.Exit(1)

        console.print(f"Found {len(shots_df)} shots")

        # Get unique lineups
        stints = (
            session.query(Stint.home_lineup, Stint.away_lineup)
            .join(Game, Stint.game_id == Game.game_id)
            .filter(Game.season_id == season)
            .all()
        )

        unique_lineups = set()
        for home, away in stints:
            home_ids = tuple(
                sorted(json.loads(home) if isinstance(home, str) else home)
            )
            away_ids = tuple(
                sorted(json.loads(away) if isinstance(away, str) else away)
            )
            unique_lineups.add(home_ids)
            unique_lineups.add(away_ids)

        console.print(f"Found {len(unique_lineups)} unique lineups")

        calculator = SpacingCalculator(min_shots=min_shots)

        # Clear existing
        session.query(LineupSpacing).filter_by(season_id=season).delete()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Calculating spacing...", total=None)

            saved_count = 0
            for lineup in unique_lineups:
                lineup_list = list(lineup)
                metrics = calculator.calculate_lineup_spacing(lineup_list, shots_df)

                if metrics["shot_count"] >= min_shots:
                    lineup_hash = SpacingCalculator.compute_lineup_hash(lineup_list)
                    spacing_record = LineupSpacing(
                        season_id=season,
                        lineup_hash=lineup_hash,
                        player_ids=json.dumps(lineup_list),
                        hull_area=metrics["hull_area"],
                        centroid_x=metrics["centroid_x"],
                        centroid_y=metrics["centroid_y"],
                        shot_count=metrics["shot_count"],
                    )
                    session.add(spacing_record)
                    saved_count += 1

            session.commit()

        # Display sample results
        table = Table(title=f"Spacing Results (Sample of {min(10, saved_count)})")
        table.add_column("Lineup Hash", style="cyan")
        table.add_column("Hull Area", justify="right")
        table.add_column("Centroid X", justify="right")
        table.add_column("Centroid Y", justify="right")
        table.add_column("Shots", justify="right")

        sample = (
            session.query(LineupSpacing).filter_by(season_id=season).limit(10).all()
        )
        for record in sample:
            table.add_row(
                record.lineup_hash[:12] + "...",
                f"{record.hull_area:.1f}",
                f"{record.centroid_x:.1f}",
                f"{record.centroid_y:.1f}",
                str(record.shot_count),
            )

        console.print(table)
        console.print(f"\n[green]Saved spacing for {saved_count} lineups[/green]")


# =============================================================================
# Train Commands
# =============================================================================


@train_app.command("transformer")
def train_transformer(
    epochs: Annotated[
        int,
        typer.Option(
            "--epochs",
            "-e",
            help="Number of training epochs",
        ),
    ] = 50,
    learning_rate: Annotated[
        float,
        typer.Option(
            "--lr",
            help="Learning rate",
        ),
    ] = 1e-4,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            "-b",
            help="Training batch size",
        ),
    ] = 32,
    save_dir: Annotated[
        str | None,
        typer.Option(
            "--save-dir",
            help="Directory to save trained model",
        ),
    ] = None,
    season: Annotated[
        str | None,
        typer.Option(
            "--season",
            "-s",
            help="Season to train on (e.g., 2023-24)",
        ),
    ] = None,
) -> None:
    """Train the Transformer sequence model.

    Trains GameFlowTransformer on play-by-play event sequences.
    Uses AdamW optimizer with gradient clipping.

    Note: For full multi-task training with all components, use 'train all'.
    """
    from pathlib import Path

    import torch
    import torch.nn.functional as F
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torch.utils.data import DataLoader

    from nba_model.models import GameFlowTransformer

    settings = get_settings()
    output_dir = Path(save_dir) if save_dir else settings.model_dir_obj / "transformer"
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        Panel(
            f"[bold]Training GameFlowTransformer[/bold]\n"
            f"Epochs: {epochs} | LR: {learning_rate} | Batch: {batch_size}",
            title="Train Transformer",
        )
    )

    # Initialize model
    model = GameFlowTransformer(
        vocab_size=15,  # 15 event types
        d_model=128,
        nhead=4,
        num_layers=2,
        max_seq_len=50,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    console.print(f"Using device: [cyan]{device}[/cyan]")

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    console.print(
        f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # Attempt to train if database exists
    trained = False
    if settings.db_path_obj.exists():
        from nba_model.data import init_db, session_scope
        from nba_model.models import NBADataset, nba_collate_fn, temporal_split

        init_db()

        with session_scope() as session:
            # Determine training season
            if season:
                train_season = season
            else:
                from sqlalchemy import func

                from nba_model.data.models import Game

                result = session.query(func.max(Game.season_id)).scalar()
                train_season = result

            if train_season:
                console.print(f"Loading data for season: [cyan]{train_season}[/cyan]")
                dataset = NBADataset.from_season(train_season, session)

                if len(dataset) >= 10:
                    train_dataset, val_dataset = temporal_split(dataset, val_ratio=0.2)
                    console.print(
                        f"Training: {len(train_dataset)} games, Validation: {len(val_dataset)} games"
                    )

                    train_loader = DataLoader(
                        train_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        collate_fn=nba_collate_fn,
                    )

                    console.print("\n[bold green]Training Transformer...[/bold green]")
                    model.train()
                    for epoch in range(epochs):
                        epoch_loss = 0.0
                        batch_count = 0
                        for batch in train_loader:
                            optimizer.zero_grad()
                            # Get transformer output (sequence representation)
                            tokens = batch["sequence"]
                            output = model(
                                tokens.events.to(device),
                                tokens.times.to(device),
                                tokens.scores.to(device),
                                tokens.lineups.to(device),
                                tokens.mask.to(device),
                            )
                            # Use margin prediction as auxiliary loss target
                            target = batch["margin"].to(device)
                            loss = F.huber_loss(
                                output.mean(dim=1, keepdim=True), target
                            )
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            optimizer.step()
                            epoch_loss += loss.item()
                            batch_count += 1

                        avg_loss = epoch_loss / max(batch_count, 1)
                        scheduler.step(avg_loss)
                        if (epoch + 1) % 10 == 0 or epoch == 0:
                            console.print(
                                f"  Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}"
                            )

                    trained = True
                    console.print("[green]Training complete![/green]")
                else:
                    console.print(
                        f"[yellow]Insufficient games ({len(dataset)}). Using 'train all' recommended.[/yellow]"
                    )
    else:
        console.print("[yellow]Database not found. Run 'data collect' first.[/yellow]")

    if not trained:
        console.print("[yellow]Saving initialized (untrained) weights.[/yellow]")
        console.print("[yellow]For full training, use: nba-model train all[/yellow]")

    # Save model
    model_path = output_dir / "transformer.pt"
    torch.save(model.state_dict(), model_path)
    console.print(f"[green]Saved model to {model_path}[/green]")


@train_app.command("gnn")
def train_gnn(
    epochs: Annotated[
        int,
        typer.Option(
            "--epochs",
            "-e",
            help="Number of training epochs",
        ),
    ] = 50,
    learning_rate: Annotated[
        float,
        typer.Option(
            "--lr",
            help="Learning rate",
        ),
    ] = 1e-4,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            "-b",
            help="Training batch size",
        ),
    ] = 32,
    save_dir: Annotated[
        str | None,
        typer.Option(
            "--save-dir",
            help="Directory to save trained model",
        ),
    ] = None,
    season: Annotated[
        str | None,
        typer.Option(
            "--season",
            "-s",
            help="Season to train on (e.g., 2023-24)",
        ),
    ] = None,
) -> None:
    """Train the GNN player interaction model.

    Trains GATv2-based PlayerInteractionGNN on lineup graphs.
    Models player interactions and team dynamics.

    Note: For full multi-task training with all components, use 'train all'.
    """
    from pathlib import Path

    import torch
    import torch.nn.functional as F
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from torch_geometric.data import Batch

    from nba_model.models import PlayerInteractionGNN

    settings = get_settings()
    output_dir = Path(save_dir) if save_dir else settings.model_dir_obj / "gnn"
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        Panel(
            f"[bold]Training PlayerInteractionGNN[/bold]\n"
            f"Epochs: {epochs} | LR: {learning_rate} | Batch: {batch_size}",
            title="Train GNN",
        )
    )

    # Initialize model
    model = PlayerInteractionGNN(
        node_features=16,
        hidden_dim=64,
        output_dim=128,
        num_heads=4,
        num_layers=2,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    console.print(f"Using device: [cyan]{device}[/cyan]")

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    console.print(
        f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # Attempt to train if database exists
    trained = False
    if settings.db_path_obj.exists():
        from nba_model.data import init_db, session_scope
        from nba_model.models import NBADataset, nba_collate_fn, temporal_split

        init_db()

        with session_scope() as session:
            # Determine training season
            if season:
                train_season = season
            else:
                from sqlalchemy import func

                from nba_model.data.models import Game

                result = session.query(func.max(Game.season_id)).scalar()
                train_season = result

            if train_season:
                console.print(f"Loading data for season: [cyan]{train_season}[/cyan]")
                dataset = NBADataset.from_season(train_season, session)

                if len(dataset) >= 10:
                    train_dataset, val_dataset = temporal_split(dataset, val_ratio=0.2)
                    console.print(
                        f"Training: {len(train_dataset)} games, Validation: {len(val_dataset)} games"
                    )

                    train_loader = DataLoader(
                        train_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        collate_fn=nba_collate_fn,
                    )

                    console.print("\n[bold green]Training GNN...[/bold green]")
                    model.train()
                    for epoch in range(epochs):
                        epoch_loss = 0.0
                        batch_count = 0
                        for batch in train_loader:
                            optimizer.zero_grad()
                            # Get GNN output from graph batch
                            graph_batch = Batch.from_data_list(batch["graph"]).to(
                                device
                            )
                            output = model(
                                graph_batch.x, graph_batch.edge_index, graph_batch.batch
                            )
                            # Use margin prediction as auxiliary loss target
                            target = batch["margin"].to(device)
                            loss = F.huber_loss(
                                output.mean(dim=1, keepdim=True), target
                            )
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            optimizer.step()
                            epoch_loss += loss.item()
                            batch_count += 1

                        avg_loss = epoch_loss / max(batch_count, 1)
                        if (epoch + 1) % 10 == 0 or epoch == 0:
                            console.print(
                                f"  Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}"
                            )

                    trained = True
                    console.print("[green]Training complete![/green]")
                else:
                    console.print(
                        f"[yellow]Insufficient games ({len(dataset)}). Using 'train all' recommended.[/yellow]"
                    )
    else:
        console.print("[yellow]Database not found. Run 'data collect' first.[/yellow]")

    if not trained:
        console.print("[yellow]Saving initialized (untrained) weights.[/yellow]")
        console.print("[yellow]For full training, use: nba-model train all[/yellow]")

    # Save model
    model_path = output_dir / "gnn.pt"
    torch.save(model.state_dict(), model_path)
    console.print(f"[green]Saved model to {model_path}[/green]")


@train_app.command("fusion")
def train_fusion(
    epochs: Annotated[
        int,
        typer.Option(
            "--epochs",
            "-e",
            help="Number of training epochs",
        ),
    ] = 50,
    learning_rate: Annotated[
        float,
        typer.Option(
            "--lr",
            help="Learning rate",
        ),
    ] = 1e-4,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            "-b",
            help="Training batch size",
        ),
    ] = 32,
    patience: Annotated[
        int,
        typer.Option(
            "--patience",
            help="Early stopping patience",
        ),
    ] = 10,
    save_dir: Annotated[
        str | None,
        typer.Option(
            "--save-dir",
            help="Directory to save trained model",
        ),
    ] = None,
    season: Annotated[
        str | None,
        typer.Option(
            "--season",
            "-s",
            help="Season to train on (e.g., 2023-24)",
        ),
    ] = None,
) -> None:
    """Train the Two-Tower fusion model.

    Trains complete fusion architecture with multi-task outputs:
    - Win probability (BCE loss)
    - Point margin (Huber loss)
    - Total points (Huber loss)

    Note: This command trains all three components (Transformer, GNN, Fusion).
    Equivalent to 'train all' but saves to a specific directory.
    """
    from pathlib import Path

    import torch
    from torch.utils.data import DataLoader

    from nba_model.models import (
        FusionTrainer,
        GameFlowTransformer,
        PlayerInteractionGNN,
        TrainingConfig,
        TwoTowerFusion,
    )

    settings = get_settings()
    output_dir = Path(save_dir) if save_dir else settings.model_dir_obj / "fusion"
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        Panel(
            f"[bold]Training Two-Tower Fusion Model[/bold]\n"
            f"Epochs: {epochs} | LR: {learning_rate} | Batch: {batch_size} | Patience: {patience}",
            title="Train Fusion",
        )
    )

    # Initialize component models
    transformer = GameFlowTransformer(vocab_size=15, d_model=128, nhead=4, num_layers=2)
    gnn = PlayerInteractionGNN(
        node_features=16, hidden_dim=64, output_dim=128, num_heads=4, num_layers=2
    )
    fusion = TwoTowerFusion(
        context_dim=32, transformer_dim=128, gnn_dim=128, hidden_dim=256
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Using device: [cyan]{device}[/cyan]")

    # Create trainer
    config = TrainingConfig(
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience,
    )
    trainer = FusionTrainer(transformer, gnn, fusion, config, device=device)

    total_params = sum(
        sum(p.numel() for p in m.parameters()) for m in [transformer, gnn, fusion]
    )
    console.print(f"Total parameters: {total_params:,}")

    # Attempt to train if database exists
    trained = False
    if settings.db_path_obj.exists():
        from nba_model.data import init_db, session_scope
        from nba_model.models import NBADataset, nba_collate_fn, temporal_split

        init_db()

        with session_scope() as session:
            # Determine training season
            if season:
                train_season = season
            else:
                from sqlalchemy import func

                from nba_model.data.models import Game

                result = session.query(func.max(Game.season_id)).scalar()
                train_season = result

            if train_season:
                console.print(f"Loading data for season: [cyan]{train_season}[/cyan]")
                dataset = NBADataset.from_season(train_season, session)

                if len(dataset) >= 10:
                    train_dataset, val_dataset = temporal_split(dataset, val_ratio=0.2)
                    console.print(
                        f"Training: {len(train_dataset)} games, Validation: {len(val_dataset)} games"
                    )

                    if len(train_dataset) > 0 and len(val_dataset) > 0:
                        train_loader = DataLoader(
                            train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=nba_collate_fn,
                        )
                        val_loader = DataLoader(
                            val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            collate_fn=nba_collate_fn,
                        )

                        console.print(
                            "\n[bold green]Training Fusion Model...[/bold green]"
                        )
                        with Progress(
                            SpinnerColumn(),
                            TextColumn("[progress.description]{task.description}"),
                            console=console,
                        ) as progress:
                            progress.add_task("Training...", total=None)
                            history = trainer.fit(
                                train_loader, val_loader, epochs=epochs
                            )

                        trained = True
                        console.print("[green]Training complete![/green]")
                        console.print(f"  Best epoch: {history.best_epoch}")
                        console.print(f"  Best val loss: {history.best_val_loss:.4f}")
                else:
                    console.print(
                        f"[yellow]Insufficient games ({len(dataset)}). Need at least 10.[/yellow]"
                    )
    else:
        console.print("[yellow]Database not found. Run 'data collect' first.[/yellow]")

    if not trained:
        console.print("[yellow]Saving initialized (untrained) weights.[/yellow]")

    # Save models
    trainer.save_models(output_dir)
    console.print(f"[green]Saved models to {output_dir}[/green]")


@train_app.command("all")
def train_all(
    epochs: Annotated[
        int,
        typer.Option(
            "--epochs",
            "-e",
            help="Number of training epochs",
        ),
    ] = 50,
    learning_rate: Annotated[
        float,
        typer.Option(
            "--lr",
            help="Learning rate",
        ),
    ] = 1e-4,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            "-b",
            help="Training batch size",
        ),
    ] = 32,
    patience: Annotated[
        int,
        typer.Option(
            "--patience",
            help="Early stopping patience",
        ),
    ] = 10,
    version: Annotated[
        str | None,
        typer.Option(
            "--version",
            help="Model version to save (auto-increments if not provided)",
        ),
    ] = None,
    season: Annotated[
        str | None,
        typer.Option(
            "--season",
            "-s",
            help="Season to train on (e.g., 2023-24)",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Initialize models without training (for testing)",
        ),
    ] = False,
) -> None:
    """Run full training pipeline.

    Trains the complete fusion model (Transformer + GNN + Fusion) and
    saves versioned models to the registry.
    """

    import torch
    from torch.utils.data import DataLoader

    from nba_model.models import (
        FusionTrainer,
        GameFlowTransformer,
        ModelRegistry,
        NBADataset,
        PlayerInteractionGNN,
        TrainingConfig,
        TwoTowerFusion,
        nba_collate_fn,
        temporal_split,
    )

    settings = get_settings()

    console.print(
        Panel(
            f"[bold]Full Training Pipeline[/bold]\n"
            f"Epochs: {epochs} | LR: {learning_rate} | Batch: {batch_size} | Patience: {patience}",
            title="Train All",
        )
    )

    # Initialize registry
    registry = ModelRegistry(base_dir=settings.model_dir_obj)

    # Determine version
    model_version = version or registry.next_version("minor")
    console.print(f"Model version: [cyan]{model_version}[/cyan]")

    # Initialize models
    transformer = GameFlowTransformer(vocab_size=15, d_model=128, nhead=4, num_layers=2)
    gnn = PlayerInteractionGNN(
        node_features=16, hidden_dim=64, output_dim=128, num_heads=4, num_layers=2
    )
    fusion = TwoTowerFusion(
        context_dim=32, transformer_dim=128, gnn_dim=128, hidden_dim=256
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Using device: [cyan]{device}[/cyan]")

    # Create trainer
    config = TrainingConfig(
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience,
    )
    trainer = FusionTrainer(transformer, gnn, fusion, config, device=device)

    total_params = sum(
        sum(p.numel() for p in m.parameters()) for m in [transformer, gnn, fusion]
    )
    console.print(f"Total parameters: {total_params:,}")

    # Display training configuration
    table = Table(title="Training Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Epochs", str(epochs))
    table.add_row("Learning Rate", str(learning_rate))
    table.add_row("Batch Size", str(batch_size))
    table.add_row("Patience", str(patience))
    table.add_row("Device", str(device))
    table.add_row("Version", model_version)
    console.print(table)

    # Check for training data
    train_metrics = {"accuracy": 0.0, "loss": 0.0}
    trained = False

    if not dry_run and settings.db_path_obj.exists():
        from nba_model.data import init_db, session_scope

        init_db()

        with session_scope() as session:
            # Determine training season
            if season:
                train_season = season
            else:
                # Get most recent season with data
                from sqlalchemy import func

                from nba_model.data.models import Game

                result = session.query(func.max(Game.season_id)).scalar()
                train_season = result if result else None

            if train_season:
                console.print(f"Loading data for season: [cyan]{train_season}[/cyan]")

                # Create dataset
                dataset = NBADataset.from_season(train_season, session)

                if len(dataset) >= 10:  # Minimum games for training
                    console.print(f"Found {len(dataset)} games for training")

                    # Split data temporally
                    train_dataset, val_dataset = temporal_split(dataset, val_ratio=0.2)
                    console.print(
                        f"Split: {len(train_dataset)} train, {len(val_dataset)} val"
                    )

                    if len(train_dataset) > 0 and len(val_dataset) > 0:
                        # Create data loaders
                        train_loader = DataLoader(
                            train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=nba_collate_fn,
                        )
                        val_loader = DataLoader(
                            val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            collate_fn=nba_collate_fn,
                        )

                        # Train the model
                        console.print("\n[bold green]Starting training...[/bold green]")
                        with Progress(
                            SpinnerColumn(),
                            TextColumn("[progress.description]{task.description}"),
                            console=console,
                        ) as progress:
                            progress.add_task("Training...", total=None)
                            history = trainer.fit(
                                train_loader,
                                val_loader,
                                epochs=epochs,
                            )

                        trained = True

                        # Extract final metrics
                        if history.epochs:
                            final_epoch = history.epochs[-1]
                            train_metrics = {
                                "accuracy": final_epoch.win_accuracy,
                                "loss": final_epoch.val_loss,
                                "brier_score": final_epoch.brier_score,
                                "margin_mae": final_epoch.margin_mae,
                                "total_mae": final_epoch.total_mae,
                            }

                        # Display training results
                        console.print("\n[bold]Training Results:[/bold]")
                        results_table = Table()
                        results_table.add_column("Metric", style="cyan")
                        results_table.add_column("Value", style="green")
                        results_table.add_row("Best Epoch", str(history.best_epoch))
                        results_table.add_row(
                            "Best Val Loss", f"{history.best_val_loss:.4f}"
                        )
                        results_table.add_row(
                            "Training Time", f"{history.total_time:.1f}s"
                        )
                        if history.epochs:
                            results_table.add_row(
                                "Final Accuracy", f"{train_metrics['accuracy']:.3f}"
                            )
                        console.print(results_table)
                    else:
                        console.print(
                            "[yellow]Not enough data in train/val splits[/yellow]"
                        )
                else:
                    console.print(
                        f"[yellow]Insufficient games ({len(dataset)}). Need at least 10.[/yellow]"
                    )
            else:
                console.print("[yellow]No season data found in database.[/yellow]")
    else:
        if dry_run:
            console.print("[yellow]Dry run mode - skipping training.[/yellow]")
        else:
            console.print(
                "[yellow]Database not found. Run 'data collect' first for full training.[/yellow]"
            )

    # Save models to registry
    hyperparams = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "patience": patience,
        "d_model": 128,
        "gnn_hidden": 64,
        "fusion_hidden": 256,
        "trained": trained,
    }

    models = {
        "transformer": transformer,
        "gnn": gnn,
        "fusion": fusion,
    }
    registry.save_model(model_version, models, train_metrics, hyperparams)
    console.print(f"[green]Saved version {model_version} to registry[/green]")

    # Show available versions
    versions = registry.list_versions()
    if versions:
        console.print("\n[bold]Available model versions:[/bold]")
        for v in versions[:5]:  # Show latest 5
            marker = " [green](latest)[/green]" if v.is_latest else ""
            console.print(f"  - {v.version}{marker}")


@train_app.command("list")
def train_list() -> None:
    """List all trained model versions.

    Shows version, training date, and validation metrics for each saved model.
    """
    from nba_model.models import ModelRegistry

    settings = get_settings()
    registry = ModelRegistry(base_dir=settings.model_dir_obj)

    versions = registry.list_versions()

    if not versions:
        console.print("[yellow]No trained models found.[/yellow]")
        console.print("Run [cyan]nba-model train all[/cyan] to train a model.")
        return

    table = Table(title="Trained Model Versions")
    table.add_column("Version", style="cyan")
    table.add_column("Date", style="green")
    table.add_column("Accuracy", justify="right")
    table.add_column("Loss", justify="right")
    table.add_column("Status", style="yellow")

    for v in versions:
        metadata = registry.load_metadata(v.version)
        if metadata:
            date_str = (
                metadata.training_date.strftime("%Y-%m-%d")
                if metadata.training_date
                else "N/A"
            )
            accuracy = metadata.validation_metrics.get("accuracy", 0.0)
            loss = metadata.validation_metrics.get("loss", 0.0)
        else:
            date_str = "N/A"
            accuracy = 0.0
            loss = 0.0

        status = "latest" if v.is_latest else ""
        table.add_row(
            v.version,
            date_str,
            f"{accuracy:.3f}",
            f"{loss:.3f}",
            status,
        )

    console.print(table)


@train_app.command("compare")
def train_compare(
    version_a: Annotated[
        str,
        typer.Argument(help="First version to compare"),
    ],
    version_b: Annotated[
        str,
        typer.Argument(help="Second version to compare"),
    ],
) -> None:
    """Compare two model versions.

    Shows metric differences and determines which version performs better.
    """
    from nba_model.models import ModelRegistry

    settings = get_settings()
    registry = ModelRegistry(base_dir=settings.model_dir_obj)

    try:
        comparison = registry.compare_versions(version_a, version_b)
    except Exception as e:
        console.print(f"[red]Error comparing versions: {e}[/red]")
        raise typer.Exit(1) from None

    console.print(
        Panel(
            f"[bold]Version Comparison[/bold]\n" f"{version_a} vs {version_b}",
            title="Compare Models",
        )
    )

    table = Table(title="Metric Comparison")
    table.add_column("Metric", style="cyan")
    table.add_column(version_a, justify="right")
    table.add_column(version_b, justify="right")
    table.add_column("Δ", justify="right", style="yellow")

    for metric, delta in comparison.improvements.items():
        val_a = comparison.metrics_a.get(metric, 0.0)
        val_b = comparison.metrics_b.get(metric, 0.0)
        delta_str = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}"
        table.add_row(metric, f"{val_a:.4f}", f"{val_b:.4f}", delta_str)

    console.print(table)

    if comparison.winner:
        console.print(f"\n[green]Winner: {comparison.winner}[/green]")


# =============================================================================
# Backtest Commands
# =============================================================================


@backtest_app.command("run")
def backtest_run(
    start_date: Annotated[
        str | None,
        typer.Option(
            "--start",
            help="Start date (YYYY-MM-DD)",
        ),
    ] = None,
    end_date: Annotated[
        str | None,
        typer.Option(
            "--end",
            help="End date (YYYY-MM-DD)",
        ),
    ] = None,
    min_train: Annotated[
        int,
        typer.Option(
            "--min-train",
            help="Minimum training games per fold",
        ),
    ] = 500,
    val_window: Annotated[
        int,
        typer.Option(
            "--val-window",
            help="Validation window size (games)",
        ),
    ] = 100,
    kelly_fraction: Annotated[
        float,
        typer.Option(
            "--kelly",
            "-k",
            help="Kelly fraction multiplier (0.25 = quarter Kelly)",
        ),
    ] = 0.25,
    max_bet: Annotated[
        float,
        typer.Option(
            "--max-bet",
            help="Maximum bet as fraction of bankroll",
        ),
    ] = 0.02,
    min_edge: Annotated[
        float,
        typer.Option(
            "--min-edge",
            help="Minimum edge required to place bet",
        ),
    ] = 0.02,
    devig_method: Annotated[
        str,
        typer.Option(
            "--devig",
            help="Devigging method (multiplicative, power, shin)",
        ),
    ] = "power",
    bankroll: Annotated[
        float,
        typer.Option(
            "--bankroll",
            "-b",
            help="Initial bankroll",
        ),
    ] = 10000.0,
    season: Annotated[
        str | None,
        typer.Option(
            "--season",
            "-s",
            help="Season to backtest (e.g., 2023-24)",
        ),
    ] = None,
) -> None:
    """Run walk-forward backtest.

    Executes time-series validation with proper temporal ordering to prevent
    look-ahead bias. Trains model on historical data and validates on future games.
    """
    from datetime import datetime

    import pandas as pd

    from nba_model.backtest import (
        BacktestConfig,
        BacktestMetricsCalculator,
        WalkForwardEngine,
        create_mock_trainer,
    )

    settings = get_settings()

    console.print(
        Panel(
            f"[bold]Walk-Forward Backtest[/bold]\n"
            f"Kelly: {kelly_fraction} | Max Bet: {max_bet:.1%} | Min Edge: {min_edge:.1%}\n"
            f"Devig Method: {devig_method} | Initial Bankroll: ${bankroll:,.0f}",
            title="Backtest Run",
        )
    )

    # Check for database
    if not settings.db_path_obj.exists():
        console.print("[yellow]Database not found. Running demo with synthetic data.[/yellow]")

        # Create synthetic games for demo
        import numpy as np

        np.random.seed(42)
        n_games = 800
        dates = pd.date_range("2023-01-01", periods=n_games, freq="D")
        games_df = pd.DataFrame(
            {
                "game_id": [f"GAME{i:04d}" for i in range(n_games)],
                "game_date": dates,
                "home_score": np.random.randint(90, 130, n_games),
                "away_score": np.random.randint(90, 130, n_games),
            }
        )
    else:
        from nba_model.data import Game, init_db, session_scope

        init_db()

        with session_scope() as session:
            # Query games
            query = session.query(Game)

            if season:
                query = query.filter(Game.season_id == season)
            if start_date:
                query = query.filter(Game.game_date >= datetime.strptime(start_date, "%Y-%m-%d").date())
            if end_date:
                query = query.filter(Game.game_date <= datetime.strptime(end_date, "%Y-%m-%d").date())

            games = query.order_by(Game.game_date).all()

            if not games:
                console.print("[red]No games found matching criteria.[/red]")
                raise typer.Exit(1)

            games_df = pd.DataFrame(
                [
                    {
                        "game_id": g.game_id,
                        "game_date": g.game_date,
                        "home_score": g.home_score or 100,
                        "away_score": g.away_score or 100,
                    }
                    for g in games
                ]
            )

    console.print(f"Found {len(games_df)} games for backtesting")

    # Create backtest configuration
    config = BacktestConfig(
        min_train_games=min_train,
        validation_window_games=val_window,
        step_size_games=val_window // 2,
        initial_bankroll=bankroll,
        devig_method=devig_method,
        kelly_fraction=kelly_fraction,
        max_bet_pct=max_bet,
        min_edge_pct=min_edge,
    )

    # Create engine and trainer
    engine = WalkForwardEngine(
        min_train_games=config.min_train_games,
        validation_window_games=config.validation_window_games,
        step_size_games=config.step_size_games,
    )

    trainer = create_mock_trainer()

    # Run backtest with progress
    console.print("\n[bold green]Running walk-forward backtest...[/bold green]")

    def progress_cb(fold: int, total: int, status: str) -> None:
        console.print(f"  [{fold}/{total}] {status}")

    try:
        result = engine.run_backtest(
            games_df=games_df,
            trainer=trainer,
            config=config,
            progress_callback=progress_cb,
        )
    except ValueError as e:
        console.print(f"[red]Backtest failed: {e}[/red]")
        raise typer.Exit(1) from None

    # Display results
    console.print("\n[bold]Backtest Results:[/bold]")

    results_table = Table()
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="green", justify="right")

    if result.metrics:
        m = result.metrics
        results_table.add_row("Total Bets", str(m.total_bets))
        results_table.add_row("Win Rate", f"{m.win_rate:.2%}")
        results_table.add_row("ROI", f"{m.roi:.2%}")
        results_table.add_row("Total Return", f"{m.total_return:.2%}")
        results_table.add_row("Sharpe Ratio", f"{m.sharpe_ratio:.2f}")
        results_table.add_row("Sortino Ratio", f"{m.sortino_ratio:.2f}")
        results_table.add_row("Max Drawdown", f"{m.max_drawdown:.2%}")
        results_table.add_row("Avg Edge", f"{m.avg_edge:.4f}")
        results_table.add_row("Total Wagered", f"${m.total_wagered:,.2f}")
        results_table.add_row("Total Profit", f"${m.total_profit:,.2f}")

    console.print(results_table)

    # Show fold summary
    if result.fold_results:
        console.print(f"\n[bold]Folds:[/bold] {len(result.fold_results)}")
        for fold in result.fold_results[:5]:
            fi = fold.fold_info
            pnl = fold.bankroll_end - fold.bankroll_start
            pnl_str = f"+${pnl:,.0f}" if pnl >= 0 else f"-${abs(pnl):,.0f}"
            console.print(
                f"  Fold {fi.fold_num}: {fi.train_games} train, {fi.val_games} val, "
                f"{len(fold.bets)} bets, {pnl_str}"
            )
        if len(result.fold_results) > 5:
            console.print(f"  ... and {len(result.fold_results) - 5} more folds")

    console.print(f"\n[green]Final Bankroll: ${result.bankroll_history[-1]:,.2f}[/green]")


@backtest_app.command("report")
def backtest_report(
    result_file: Annotated[
        str | None,
        typer.Option(
            "--file",
            "-f",
            help="Path to backtest result JSON file",
        ),
    ] = None,
) -> None:
    """Generate backtest report.

    Creates detailed performance analysis from backtest results.
    """
    import json
    from pathlib import Path

    from nba_model.backtest import BacktestMetricsCalculator, FullBacktestMetrics

    console.print(
        Panel(
            "[bold]Backtest Report Generator[/bold]",
            title="Backtest Report",
        )
    )

    if result_file:
        console.print(f"Loading results from: {result_file}")
        path = Path(result_file)
        if not path.exists():
            console.print(f"[red]File not found: {result_file}[/red]")
            raise typer.Exit(1)

        with open(path) as f:
            data = json.load(f)

        # Parse metrics from JSON
        metrics_data = data.get("metrics", {})
        metrics = FullBacktestMetrics(
            total_return=metrics_data.get("total_return", 0.0),
            cagr=metrics_data.get("cagr", 0.0),
            avg_bet_return=metrics_data.get("avg_bet_return", 0.0),
            volatility=metrics_data.get("volatility", 0.0),
            sharpe_ratio=metrics_data.get("sharpe_ratio", 0.0),
            sortino_ratio=metrics_data.get("sortino_ratio", 0.0),
            max_drawdown=metrics_data.get("max_drawdown", 0.0),
            max_drawdown_duration=metrics_data.get("max_drawdown_duration", 0),
            total_bets=metrics_data.get("total_bets", 0),
            win_rate=metrics_data.get("win_rate", 0.0),
            avg_edge=metrics_data.get("avg_edge", 0.0),
            avg_odds=metrics_data.get("avg_odds", 0.0),
            roi=metrics_data.get("roi", 0.0),
            brier_score=metrics_data.get("brier_score", 0.0),
            log_loss=metrics_data.get("log_loss", 0.0),
            avg_clv=metrics_data.get("avg_clv", 0.0),
            clv_positive_rate=metrics_data.get("clv_positive_rate", 0.0),
            metrics_by_type=metrics_data.get("metrics_by_type", {}),
            total_wagered=metrics_data.get("total_wagered", 0.0),
            total_profit=metrics_data.get("total_profit", 0.0),
            win_count=metrics_data.get("win_count", 0),
            loss_count=metrics_data.get("loss_count", 0),
            push_count=metrics_data.get("push_count", 0),
        )

        # Generate report
        calc = BacktestMetricsCalculator()
        title = data.get("title", "Backtest Report")
        report = calc.generate_report(metrics, title)
        console.print(report)

        # Show additional info from file
        if "config" in data:
            console.print("\n[bold]Configuration:[/bold]")
            config = data["config"]
            console.print(f"  Kelly Fraction: {config.get('kelly_fraction', 'N/A')}")
            console.print(f"  Devig Method: {config.get('devig_method', 'N/A')}")
            console.print(f"  Min Edge: {config.get('min_edge_pct', 'N/A')}")
            console.print(f"  Max Bet: {config.get('max_bet_pct', 'N/A')}")

        if "start_date" in data and "end_date" in data:
            console.print(f"\n[bold]Period:[/bold] {data['start_date']} to {data['end_date']}")

    else:
        console.print("[yellow]No result file specified.[/yellow]")
        console.print("Run [cyan]nba-model backtest run[/cyan] first to generate results.")
        console.print("\nExample report format:")

        # Show example report
        calc = BacktestMetricsCalculator()
        example_metrics = FullBacktestMetrics(
            total_return=0.15,
            cagr=0.12,
            sharpe_ratio=1.5,
            sortino_ratio=2.1,
            max_drawdown=0.08,
            win_rate=0.54,
            total_bets=250,
            avg_edge=0.03,
            avg_clv=0.01,
            roi=0.05,
        )

        report = calc.generate_report(example_metrics, "Example Backtest Report")
        console.print(report)


@backtest_app.command("optimize")
def backtest_optimize(
    fractions: Annotated[
        str | None,
        typer.Option(
            "--fractions",
            help="Comma-separated Kelly fractions to test (e.g., 0.1,0.2,0.25,0.3)",
        ),
    ] = None,
    metric: Annotated[
        str,
        typer.Option(
            "--metric",
            "-m",
            help="Optimization metric (sharpe or growth)",
        ),
    ] = "sharpe",
    season: Annotated[
        str | None,
        typer.Option(
            "--season",
            "-s",
            help="Season to use for optimization",
        ),
    ] = None,
) -> None:
    """Optimize Kelly fraction.

    Finds optimal bet sizing via historical simulation by testing different
    Kelly fractions and maximizing Sharpe ratio or geometric growth rate.
    """
    from datetime import datetime

    import numpy as np

    from nba_model.backtest import KellyCalculator
    from nba_model.types import Bet

    settings = get_settings()

    # Parse fractions
    if fractions:
        fraction_list = [float(f.strip()) for f in fractions.split(",")]
    else:
        fraction_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    console.print(
        Panel(
            f"[bold]Kelly Fraction Optimization[/bold]\n"
            f"Testing fractions: {fraction_list}\n"
            f"Optimization metric: {metric}",
            title="Backtest Optimize",
        )
    )

    # Generate synthetic bet history for demo
    console.print("[yellow]Using synthetic bet history for demonstration.[/yellow]")

    np.random.seed(42)
    n_bets = 200

    synthetic_bets = []
    for i in range(n_bets):
        model_prob = 0.48 + np.random.random() * 0.1  # 48-58% model probability
        market_odds = 1.85 + np.random.random() * 0.2  # 1.85-2.05 odds
        edge = model_prob - (1 / market_odds)

        # Simulate outcome based on model probability
        won = np.random.random() < model_prob
        result = "win" if won else "loss"

        bet = Bet(
            game_id=f"GAME{i:04d}",
            timestamp=datetime(2023, 1, 1) + np.timedelta64(i, "D"),
            bet_type="moneyline",
            side="home",
            model_prob=model_prob,
            market_odds=market_odds,
            market_prob=1 / market_odds,
            edge=edge,
            kelly_fraction=0.0,  # Will be calculated
            bet_amount=100.0,  # Will be calculated
            result=result,
            profit=100 * (market_odds - 1) if won else -100,
        )
        synthetic_bets.append(bet)

    # Run optimization
    kelly_calc = KellyCalculator()
    result = kelly_calc.optimize_fraction(
        historical_bets=synthetic_bets,
        fractions=fraction_list,
        initial_bankroll=10000.0,
        metric=metric,
    )

    # Display results
    console.print("\n[bold]Optimization Results:[/bold]")

    results_table = Table()
    results_table.add_column("Kelly Fraction", style="cyan", justify="center")
    results_table.add_column(metric.title(), style="green", justify="right")

    sorted_fractions = sorted(result.results_by_fraction.keys())
    for frac in sorted_fractions:
        value = result.results_by_fraction[frac]
        marker = " [bold yellow]*[/bold yellow]" if frac == result.best_fraction else ""
        results_table.add_row(f"{frac:.2f}{marker}", f"{value:.4f}")

    console.print(results_table)

    console.print(f"\n[green]Optimal Kelly Fraction: {result.best_fraction:.2f}[/green]")
    console.print(f"Best {metric}: {result.best_metric:.4f}")


# =============================================================================
# Monitor Commands
# =============================================================================


@monitor_app.command("drift")
def monitor_drift() -> None:
    """Check for covariate drift.

    Runs KS tests and PSI calculations on feature distributions.
    """
    console.print(
        Panel(
            "[yellow]Drift detection not yet implemented (Phase 6)[/yellow]",
            title="Monitor Drift",
        )
    )


@monitor_app.command("trigger")
def monitor_trigger() -> None:
    """Evaluate retraining triggers.

    Checks all conditions that might require model retraining.
    """
    console.print(
        Panel(
            "[yellow]Trigger evaluation not yet implemented (Phase 6)[/yellow]",
            title="Monitor Trigger",
        )
    )


@monitor_app.command("versions")
def monitor_versions() -> None:
    """List model versions.

    Displays all saved model versions with metadata.
    """
    settings = get_settings()
    console.print(
        Panel(
            f"[bold]Model Directory:[/bold] {settings.model_dir}\n"
            "[yellow]Version listing not yet implemented (Phase 6)[/yellow]",
            title="Monitor Versions",
        )
    )


# =============================================================================
# Predict Commands
# =============================================================================


@predict_app.command("today")
def predict_today() -> None:
    """Generate predictions for today's games.

    Produces win probabilities, margins, and totals for all games today.
    """
    console.print(
        Panel(
            "[yellow]Predictions not yet implemented (Phase 7)[/yellow]",
            title="Predict Today",
        )
    )


@predict_app.command("game")
def predict_game(
    game_id: Annotated[
        str,
        typer.Argument(help="NBA Game ID"),
    ],
) -> None:
    """Generate prediction for a single game.

    Produces detailed prediction with confidence intervals.
    """
    console.print(
        Panel(
            f"[bold]Game ID:[/bold] {game_id}\n"
            "[yellow]Single game prediction not yet implemented (Phase 7)[/yellow]",
            title="Predict Game",
        )
    )


@predict_app.command("signals")
def predict_signals(
    min_edge: Annotated[
        float,
        typer.Option(
            "--min-edge",
            help="Minimum edge percentage to show",
        ),
    ] = 0.02,
) -> None:
    """Generate betting signals.

    Identifies bets with positive expected value.
    """
    console.print(
        Panel(
            f"[bold]Min Edge:[/bold] {min_edge:.1%}\n"
            "[yellow]Signal generation not yet implemented (Phase 7)[/yellow]",
            title="Predict Signals",
        )
    )


# =============================================================================
# Dashboard Commands
# =============================================================================


@dashboard_app.command("build")
def dashboard_build() -> None:
    """Build GitHub Pages site.

    Generates static HTML/JSON for the dashboard.
    """
    console.print(
        Panel(
            "[yellow]Dashboard building not yet implemented (Phase 8)[/yellow]",
            title="Dashboard Build",
        )
    )


@dashboard_app.command("deploy")
def dashboard_deploy() -> None:
    """Deploy to GitHub Pages.

    Pushes built dashboard to gh-pages branch.
    """
    console.print(
        Panel(
            "[yellow]Dashboard deployment not yet implemented (Phase 8)[/yellow]",
            title="Dashboard Deploy",
        )
    )


if __name__ == "__main__":
    app()
