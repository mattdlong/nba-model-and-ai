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
    setup_logging(level=log_level, log_dir=settings.log_dir)


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
) -> None:
    """Collect data from NBA API.

    Fetches games, players, play-by-play, and shot data for specified seasons.
    """
    console.print(
        Panel(
            "[yellow]Data collection not yet implemented (Phase 2)[/yellow]",
            title="Data Collect",
        )
    )
    if seasons:
        console.print(f"Seasons requested: {', '.join(seasons)}")
    if full:
        console.print("Full collection mode enabled")


@data_app.command("update")
def data_update() -> None:
    """Incremental data update for recent games.

    Fetches games since last update and populates all related tables.
    """
    console.print(
        Panel(
            "[yellow]Incremental update not yet implemented (Phase 2)[/yellow]",
            title="Data Update",
        )
    )


@data_app.command("status")
def data_status() -> None:
    """Show database statistics and data freshness.

    Displays counts of games, players, and data recency information.
    """
    settings = get_settings()
    console.print(
        Panel(
            f"[bold]Database:[/bold] {settings.db_path}\n"
            "[yellow]Status check not yet implemented (Phase 2)[/yellow]",
            title="Data Status",
        )
    )


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
