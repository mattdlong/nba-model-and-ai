"""Tests for CLI module."""
from __future__ import annotations

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
        result = runner.invoke(app, ["--verbose", "data", "status"])

        assert result.exit_code == 0

    def test_short_verbose_flag(self) -> None:
        """-V should enable verbose mode."""
        result = runner.invoke(app, ["-V", "data", "status"])

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

    def test_data_collect_runs(self) -> None:
        """Data collect should run without error."""
        result = runner.invoke(app, ["data", "collect"])

        assert result.exit_code == 0
        assert "Phase 2" in result.stdout  # Shows not implemented message

    def test_data_collect_with_seasons(self) -> None:
        """Data collect should accept seasons option."""
        result = runner.invoke(app, ["data", "collect", "--seasons", "2023-24"])

        assert result.exit_code == 0
        assert "2023-24" in result.stdout

    def test_data_collect_with_full_flag(self) -> None:
        """Data collect should accept --full flag."""
        result = runner.invoke(app, ["data", "collect", "--full"])

        assert result.exit_code == 0
        assert "Full collection mode enabled" in result.stdout

    def test_data_collect_with_short_flags(self) -> None:
        """Data collect should accept short flags."""
        result = runner.invoke(app, ["data", "collect", "-s", "2022-23", "-f"])

        assert result.exit_code == 0
        assert "2022-23" in result.stdout
        assert "Full collection mode enabled" in result.stdout

    def test_data_update_runs(self) -> None:
        """Data update should run without error."""
        result = runner.invoke(app, ["data", "update"])

        assert result.exit_code == 0
        assert "Phase 2" in result.stdout

    def test_data_status_runs(self) -> None:
        """Data status should run without error."""
        result = runner.invoke(app, ["data", "status"])

        assert result.exit_code == 0
        assert "Database" in result.stdout


class TestFeaturesCommands:
    """Tests for features subcommands."""

    def test_features_help(self) -> None:
        """Features help should show subcommands."""
        result = runner.invoke(app, ["features", "--help"])

        assert result.exit_code == 0
        assert "build" in result.stdout
        assert "rapm" in result.stdout
        assert "spatial" in result.stdout

    def test_features_build_runs(self) -> None:
        """Features build should run without error."""
        result = runner.invoke(app, ["features", "build"])

        assert result.exit_code == 0

    def test_features_build_with_force(self) -> None:
        """Features build should accept --force flag."""
        result = runner.invoke(app, ["features", "build", "--force"])

        assert result.exit_code == 0
        assert "Force rebuild enabled" in result.stdout

    def test_features_rapm_runs(self) -> None:
        """Features rapm should run without error."""
        result = runner.invoke(app, ["features", "rapm"])

        assert result.exit_code == 0
        assert "Phase 3" in result.stdout

    def test_features_rapm_with_season(self) -> None:
        """Features rapm should accept --season option."""
        result = runner.invoke(app, ["features", "rapm", "--season", "2023-24"])

        assert result.exit_code == 0
        assert "2023-24" in result.stdout

    def test_features_spatial_runs(self) -> None:
        """Features spatial should run without error."""
        result = runner.invoke(app, ["features", "spatial"])

        assert result.exit_code == 0
        assert "Phase 3" in result.stdout

    def test_features_spatial_with_season(self) -> None:
        """Features spatial should accept --season option."""
        result = runner.invoke(app, ["features", "spatial", "--season", "2023-24"])

        assert result.exit_code == 0
        assert "2023-24" in result.stdout


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
        assert "Phase 5" in result.stdout

    def test_backtest_run_with_dates(self) -> None:
        """Backtest run should accept date options."""
        result = runner.invoke(
            app, ["backtest", "run", "--start", "2023-01-01", "--end", "2023-12-31"]
        )

        assert result.exit_code == 0
        assert "2023-01-01" in result.stdout
        assert "2023-12-31" in result.stdout

    def test_backtest_report_runs(self) -> None:
        """Backtest report should run without error."""
        result = runner.invoke(app, ["backtest", "report"])

        assert result.exit_code == 0
        assert "Phase 5" in result.stdout

    def test_backtest_optimize_runs(self) -> None:
        """Backtest optimize should run without error."""
        result = runner.invoke(app, ["backtest", "optimize"])

        assert result.exit_code == 0
        assert "Phase 5" in result.stdout


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
