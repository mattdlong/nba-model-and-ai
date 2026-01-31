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

from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from nba_model import __version__
from nba_model.config import get_settings
from nba_model.logging import setup_logging

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
        Optional[bool],
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
        Optional[list[str]],
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
    from sqlalchemy import func

    from nba_model.data import Game, Play, Player, Shot, Stint, init_db, session_scope
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
    stats.append(("Players", session.query(func.count(Player.player_id)).scalar() or 0, None))
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

    Calculates RAPM, spacing metrics, fatigue indicators, and normalizations.
    """
    console.print(
        Panel(
            "[yellow]Feature building not yet implemented (Phase 3)[/yellow]",
            title="Features Build",
        )
    )
    if force:
        console.print("Force rebuild enabled")


@features_app.command("rapm")
def features_rapm(
    season: Annotated[
        Optional[str],
        typer.Option(
            "--season",
            "-s",
            help="Season to calculate RAPM for",
        ),
    ] = None,
) -> None:
    """Calculate RAPM coefficients.

    Runs Ridge Regression on stint data to calculate player impact metrics.
    """
    console.print(
        Panel(
            "[yellow]RAPM calculation not yet implemented (Phase 3)[/yellow]",
            title="Features RAPM",
        )
    )
    if season:
        console.print(f"Season: {season}")


@features_app.command("spatial")
def features_spatial(
    season: Annotated[
        Optional[str],
        typer.Option(
            "--season",
            "-s",
            help="Season to calculate spacing for",
        ),
    ] = None,
) -> None:
    """Calculate convex hull spacing metrics.

    Analyzes shot distributions to measure lineup floor spacing.
    """
    console.print(
        Panel(
            "[yellow]Spatial analysis not yet implemented (Phase 3)[/yellow]",
            title="Features Spatial",
        )
    )
    if season:
        console.print(f"Season: {season}")


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
) -> None:
    """Train the Transformer sequence model.

    Trains GameFlowTransformer on play-by-play event sequences.
    """
    console.print(
        Panel(
            "[yellow]Transformer training not yet implemented (Phase 4)[/yellow]",
            title="Train Transformer",
        )
    )
    console.print(f"Epochs: {epochs}")


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
) -> None:
    """Train the GNN player interaction model.

    Trains GATv2-based PlayerInteractionGNN on lineup graphs.
    """
    console.print(
        Panel(
            "[yellow]GNN training not yet implemented (Phase 4)[/yellow]",
            title="Train GNN",
        )
    )
    console.print(f"Epochs: {epochs}")


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
) -> None:
    """Train the Two-Tower fusion model.

    Trains complete fusion architecture with multi-task outputs.
    """
    console.print(
        Panel(
            "[yellow]Fusion training not yet implemented (Phase 4)[/yellow]",
            title="Train Fusion",
        )
    )
    console.print(f"Epochs: {epochs}")


@train_app.command("all")
def train_all(
    epochs: Annotated[
        int,
        typer.Option(
            "--epochs",
            "-e",
            help="Number of training epochs per model",
        ),
    ] = 50,
) -> None:
    """Run full training pipeline.

    Trains all models sequentially: Transformer, GNN, then Fusion.
    """
    console.print(
        Panel(
            "[yellow]Full training pipeline not yet implemented (Phase 4)[/yellow]",
            title="Train All",
        )
    )
    console.print(f"Epochs per model: {epochs}")


# =============================================================================
# Backtest Commands
# =============================================================================


@backtest_app.command("run")
def backtest_run(
    start_date: Annotated[
        Optional[str],
        typer.Option(
            "--start",
            help="Start date (YYYY-MM-DD)",
        ),
    ] = None,
    end_date: Annotated[
        Optional[str],
        typer.Option(
            "--end",
            help="End date (YYYY-MM-DD)",
        ),
    ] = None,
) -> None:
    """Run walk-forward backtest.

    Executes time-series validation with proper temporal ordering.
    """
    console.print(
        Panel(
            "[yellow]Backtesting not yet implemented (Phase 5)[/yellow]",
            title="Backtest Run",
        )
    )
    if start_date:
        console.print(f"Start: {start_date}")
    if end_date:
        console.print(f"End: {end_date}")


@backtest_app.command("report")
def backtest_report() -> None:
    """Generate backtest report.

    Creates detailed performance analysis from backtest results.
    """
    console.print(
        Panel(
            "[yellow]Backtest reporting not yet implemented (Phase 5)[/yellow]",
            title="Backtest Report",
        )
    )


@backtest_app.command("optimize")
def backtest_optimize() -> None:
    """Optimize Kelly fraction.

    Finds optimal bet sizing via historical simulation.
    """
    console.print(
        Panel(
            "[yellow]Kelly optimization not yet implemented (Phase 5)[/yellow]",
            title="Backtest Optimize",
        )
    )


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
