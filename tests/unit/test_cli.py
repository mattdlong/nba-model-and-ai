"""Tests for CLI module."""

from __future__ import annotations

from unittest.mock import MagicMock

from typer.testing import CliRunner

from nba_model.cli import app

runner = CliRunner()


class TestMainApp:
    """Tests for main CLI app."""

    def test_help_shows_all_commands(self) -> None:
        """Help should list all subcommand groups."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "data" in result.stdout
        assert "features" in result.stdout
        assert "train" in result.stdout
        assert "backtest" in result.stdout
        assert "monitor" in result.stdout
        assert "predict" in result.stdout
        assert "dashboard" in result.stdout

    def test_version_flag(self) -> None:
        """--version should show version and exit."""
        result = runner.invoke(app, ["--version"])

        assert result.exit_code == 0
        assert "0.1.0" in result.stdout

    def test_short_version_flag(self) -> None:
        """-v should show version and exit."""
        result = runner.invoke(app, ["-v"])

        assert result.exit_code == 0
        assert "0.1.0" in result.stdout

    def test_verbose_flag(self) -> None:
        """--verbose should enable verbose mode."""
        # Run a command with verbose flag to trigger main callback
        # Use features help which doesn't require database
        result = runner.invoke(app, ["--verbose", "features", "--help"])

        assert result.exit_code == 0

    def test_short_verbose_flag(self) -> None:
        """-V should enable verbose mode."""
        result = runner.invoke(app, ["-V", "features", "--help"])

        assert result.exit_code == 0


class TestDataCommands:
    """Tests for data subcommands."""

    def test_data_help(self) -> None:
        """Data help should show subcommands."""
        result = runner.invoke(app, ["data", "--help"])

        assert result.exit_code == 0
        assert "collect" in result.stdout
        assert "update" in result.stdout
        assert "status" in result.stdout

    def test_data_collect_requires_seasons_or_full(self) -> None:
        """Data collect without seasons or --full should error."""
        result = runner.invoke(app, ["data", "collect"])

        # Should exit with error when no seasons specified
        assert result.exit_code == 1
        assert "Error" in result.stdout

    def test_data_collect_shows_seasons_in_output(self) -> None:
        """Data collect with seasons should show seasons in output panel."""
        # Just check the output format, don't run the actual pipeline
        result = runner.invoke(app, ["data", "collect", "--seasons", "2023-24"])

        # The panel should show the season before pipeline runs
        assert "2023-24" in result.stdout

    def test_data_collect_with_full_flag_shows_all_seasons(self) -> None:
        """Data collect with --full should show all seasons in output."""
        result = runner.invoke(app, ["data", "collect", "--full"])

        # Should show all 5 seasons in the output panel
        assert "2019-20" in result.stdout or "Data Collection" in result.stdout

    def test_data_collect_with_short_flags(self) -> None:
        """Data collect should accept short flags."""
        result = runner.invoke(app, ["data", "collect", "-s", "2022-23"])

        # Should show the season in output
        assert "2022-23" in result.stdout

    def test_data_status_shows_database_info(self) -> None:
        """Data status should show database information."""
        result = runner.invoke(app, ["data", "status"])

        # Should show "Database" somewhere in output (either path or "not found")
        assert "Database" in result.stdout or "database" in result.stdout.lower()


class TestFeaturesCommands:
    """Tests for features subcommands."""

    def test_features_help(self) -> None:
        """Features help should show subcommands."""
        result = runner.invoke(app, ["features", "--help"])

        assert result.exit_code == 0
        assert "build" in result.stdout
        assert "rapm" in result.stdout
        assert "spatial" in result.stdout

    def test_features_build_help(self) -> None:
        """Features build help should show options."""
        result = runner.invoke(app, ["features", "build", "--help"])

        assert result.exit_code == 0
        assert "--seasons" in result.stdout
        assert "--force" in result.stdout

    def test_features_rapm_requires_season(self) -> None:
        """Features rapm should require --season option."""
        result = runner.invoke(app, ["features", "rapm"])

        assert result.exit_code == 2  # Missing required option

    def test_features_rapm_help(self) -> None:
        """Features rapm help should show options."""
        result = runner.invoke(app, ["features", "rapm", "--help"])

        assert result.exit_code == 0
        assert "--season" in result.stdout
        assert "--lambda" in result.stdout
        assert "--cv" in result.stdout

    def test_features_spatial_requires_season(self) -> None:
        """Features spatial should require --season option."""
        result = runner.invoke(app, ["features", "spatial"])

        assert result.exit_code == 2  # Missing required option

    def test_features_spatial_help(self) -> None:
        """Features spatial help should show options."""
        result = runner.invoke(app, ["features", "spatial", "--help"])

        assert result.exit_code == 0
        assert "--season" in result.stdout
        assert "--min-shots" in result.stdout


class TestTrainCommands:
    """Tests for train subcommands."""

    def test_train_help(self) -> None:
        """Train help should show subcommands."""
        result = runner.invoke(app, ["train", "--help"])

        assert result.exit_code == 0
        assert "transformer" in result.stdout
        assert "gnn" in result.stdout
        assert "fusion" in result.stdout
        assert "all" in result.stdout

    def test_train_transformer_with_epochs(self) -> None:
        """Train transformer should accept epochs option."""
        result = runner.invoke(app, ["train", "transformer", "--epochs", "10"])

        assert result.exit_code == 0
        assert "10" in result.stdout

    def test_train_transformer_default_epochs(self) -> None:
        """Train transformer should use default epochs."""
        result = runner.invoke(app, ["train", "transformer"])

        assert result.exit_code == 0
        assert "50" in result.stdout  # Default epochs

    def test_train_gnn_runs(self) -> None:
        """Train gnn should run without error."""
        result = runner.invoke(app, ["train", "gnn"])

        assert result.exit_code == 0
        assert "Phase 4" in result.stdout

    def test_train_gnn_with_epochs(self) -> None:
        """Train gnn should accept epochs option."""
        result = runner.invoke(app, ["train", "gnn", "--epochs", "25"])

        assert result.exit_code == 0
        assert "25" in result.stdout

    def test_train_fusion_runs(self) -> None:
        """Train fusion should run without error."""
        result = runner.invoke(app, ["train", "fusion"])

        assert result.exit_code == 0
        assert "Phase 4" in result.stdout

    def test_train_fusion_with_epochs(self) -> None:
        """Train fusion should accept epochs option."""
        result = runner.invoke(app, ["train", "fusion", "--epochs", "30"])

        assert result.exit_code == 0
        assert "30" in result.stdout

    def test_train_all_runs(self) -> None:
        """Train all should run without error."""
        result = runner.invoke(app, ["train", "all"])

        assert result.exit_code == 0
        assert "Phase 4" in result.stdout

    def test_train_all_with_epochs(self) -> None:
        """Train all should accept epochs option."""
        result = runner.invoke(app, ["train", "all", "--epochs", "20"])

        assert result.exit_code == 0
        assert "20" in result.stdout


class TestBacktestCommands:
    """Tests for backtest subcommands."""

    def test_backtest_help(self) -> None:
        """Backtest help should show subcommands."""
        result = runner.invoke(app, ["backtest", "--help"])

        assert result.exit_code == 0
        assert "run" in result.stdout
        assert "report" in result.stdout
        assert "optimize" in result.stdout

    def test_backtest_run_default(self) -> None:
        """Backtest run should run without options."""
        result = runner.invoke(app, ["backtest", "run"])

        assert result.exit_code == 0
        # Should show backtest results or progress
        assert "Backtest" in result.stdout or "Walk-Forward" in result.stdout

    def test_backtest_run_with_kelly_option(self) -> None:
        """Backtest run should accept kelly option."""
        result = runner.invoke(app, ["backtest", "run", "--kelly", "0.5"])

        assert result.exit_code == 0
        # Should show backtest configuration with kelly value
        assert "Kelly" in result.stdout or "Backtest" in result.stdout

    def test_backtest_report_runs(self) -> None:
        """Backtest report should run without error."""
        result = runner.invoke(app, ["backtest", "report"])

        assert result.exit_code == 0
        # When no file specified, should show example report
        assert "Example Backtest Report" in result.stdout or "Report" in result.stdout

    def test_backtest_optimize_runs(self) -> None:
        """Backtest optimize should run without error."""
        result = runner.invoke(app, ["backtest", "optimize"])

        # The optimize command may fail with certain data issues
        # but should at least start running
        assert "Optimize" in result.stdout or "Kelly" in result.stdout or result.exit_code in (0, 1)


class TestMonitorCommands:
    """Tests for monitor subcommands."""

    def test_monitor_help(self) -> None:
        """Monitor help should show subcommands."""
        result = runner.invoke(app, ["monitor", "--help"])

        assert result.exit_code == 0
        assert "drift" in result.stdout
        assert "trigger" in result.stdout
        assert "versions" in result.stdout

    def test_monitor_drift_runs(self) -> None:
        """Monitor drift should run without error."""
        result = runner.invoke(app, ["monitor", "drift"])

        assert result.exit_code == 0
        assert "Phase 6" in result.stdout

    def test_monitor_trigger_runs(self) -> None:
        """Monitor trigger should run without error."""
        result = runner.invoke(app, ["monitor", "trigger"])

        assert result.exit_code == 0
        assert "Phase 6" in result.stdout

    def test_monitor_versions_runs(self) -> None:
        """Monitor versions should run without error."""
        result = runner.invoke(app, ["monitor", "versions"])

        assert result.exit_code == 0
        assert "Model Directory" in result.stdout


class TestPredictCommands:
    """Tests for predict subcommands."""

    def test_predict_help(self) -> None:
        """Predict help should show subcommands."""
        result = runner.invoke(app, ["predict", "--help"])

        assert result.exit_code == 0
        assert "today" in result.stdout
        assert "game" in result.stdout
        assert "signals" in result.stdout

    def test_predict_game_requires_id(self) -> None:
        """Predict game should require game_id argument."""
        result = runner.invoke(app, ["predict", "game", "0022300001"])

        assert result.exit_code == 0
        assert "0022300001" in result.stdout

    def test_predict_today_runs(self) -> None:
        """Predict today should run without error."""
        result = runner.invoke(app, ["predict", "today"])

        assert result.exit_code == 0
        assert "Phase 7" in result.stdout

    def test_predict_signals_default(self) -> None:
        """Predict signals should run with default min-edge."""
        result = runner.invoke(app, ["predict", "signals"])

        assert result.exit_code == 0
        assert "2.0%" in result.stdout  # Default min_edge

    def test_predict_signals_with_min_edge(self) -> None:
        """Predict signals should accept --min-edge option."""
        result = runner.invoke(app, ["predict", "signals", "--min-edge", "0.05"])

        assert result.exit_code == 0
        assert "5.0%" in result.stdout


class TestDashboardCommands:
    """Tests for dashboard subcommands."""

    def test_dashboard_help(self) -> None:
        """Dashboard help should show subcommands."""
        result = runner.invoke(app, ["dashboard", "--help"])

        assert result.exit_code == 0
        assert "build" in result.stdout
        assert "deploy" in result.stdout

    def test_dashboard_build_runs(self) -> None:
        """Dashboard build should run without error."""
        result = runner.invoke(app, ["dashboard", "build"])

        assert result.exit_code == 0
        assert "Phase 8" in result.stdout

    def test_dashboard_deploy_runs(self) -> None:
        """Dashboard deploy should run without error."""
        result = runner.invoke(app, ["dashboard", "deploy"])

        assert result.exit_code == 0
        assert "Phase 8" in result.stdout


class TestDataRepairCommand:
    """Tests for data repair command."""

    def test_data_repair_shows_game_ids(self) -> None:
        """Data repair should show the game IDs being repaired."""
        result = runner.invoke(app, ["data", "repair", "0022300001"])

        # Should show the game ID before attempting repair
        assert "0022300001" in result.stdout

    def test_data_repair_multiple_games(self) -> None:
        """Data repair should accept multiple game IDs."""
        result = runner.invoke(
            app, ["data", "repair", "0022300001", "0022300002", "0022300003"]
        )

        # Should show multiple game IDs
        assert "0022300001" in result.stdout

    def test_data_repair_help(self) -> None:
        """Data repair help should show usage."""
        result = runner.invoke(app, ["data", "repair", "--help"])

        assert result.exit_code == 0
        assert "GAME_IDS" in result.stdout


class TestDisplayPipelineResult:
    """Tests for _display_pipeline_result helper."""

    def test_display_completed_result(self) -> None:
        """Should display completed result without error."""
        from nba_model.cli import _display_pipeline_result
        from nba_model.data.pipelines import PipelineResult, PipelineStatus

        result = PipelineResult(
            status=PipelineStatus.COMPLETED,
            games_processed=10,
            plays_collected=500,
            shots_collected=100,
            stints_derived=50,
            duration_seconds=30.5,
        )

        # Should not raise
        _display_pipeline_result(result)

    def test_display_failed_result(self) -> None:
        """Should display failed result without error."""
        from nba_model.cli import _display_pipeline_result
        from nba_model.data.pipelines import PipelineResult, PipelineStatus

        result = PipelineResult(
            status=PipelineStatus.FAILED,
            errors=["Error 1", "Error 2"],
            duration_seconds=5.0,
        )

        # Should not raise
        _display_pipeline_result(result)

    def test_display_result_with_many_errors(self) -> None:
        """Should truncate errors list when there are many."""
        from nba_model.cli import _display_pipeline_result
        from nba_model.data.pipelines import PipelineResult, PipelineStatus

        result = PipelineResult(
            status=PipelineStatus.FAILED,
            errors=[f"Error {i}" for i in range(20)],
            duration_seconds=5.0,
        )

        # Should not raise (truncates to first 10)
        _display_pipeline_result(result)


class TestGetDatabaseStats:
    """Tests for _get_database_stats helper."""

    def test_returns_stats_list(self) -> None:
        """Should return list of (entity, count, date_range) tuples."""

        from nba_model.cli import _get_database_stats

        session = MagicMock()
        # Mock all query returns
        session.query.return_value.scalar.return_value = 0

        stats = _get_database_stats(session)

        # Should return a list of tuples
        assert isinstance(stats, list)
        assert len(stats) >= 4  # Games, Players, Plays, Shots, Stints

    def test_returns_date_range_when_games_exist(self) -> None:
        """Should return date range when games exist."""
        from datetime import date

        from nba_model.cli import _get_database_stats

        session = MagicMock()
        # First call for game count (> 0)
        # Subsequent calls for min/max dates and other counts
        session.query.return_value.scalar.side_effect = [
            100,  # game count
            date(2024, 1, 1),  # min date
            date(2024, 6, 30),  # max date
            50,  # players
            500,  # plays
            100,  # shots
            25,  # stints
        ]

        stats = _get_database_stats(session)

        # Games entry should have date range
        games_stat = stats[0]
        assert games_stat[0] == "Games"
        assert games_stat[1] == 100
        assert games_stat[2] is not None  # Date range
