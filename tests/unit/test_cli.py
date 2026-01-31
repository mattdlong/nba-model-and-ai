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


class TestBacktestCommands:
    """Tests for backtest subcommands."""

    def test_backtest_help(self) -> None:
        """Backtest help should show subcommands."""
        result = runner.invoke(app, ["backtest", "--help"])

        assert result.exit_code == 0
        assert "run" in result.stdout
        assert "report" in result.stdout
        assert "optimize" in result.stdout


class TestMonitorCommands:
    """Tests for monitor subcommands."""

    def test_monitor_help(self) -> None:
        """Monitor help should show subcommands."""
        result = runner.invoke(app, ["monitor", "--help"])

        assert result.exit_code == 0
        assert "drift" in result.stdout
        assert "trigger" in result.stdout
        assert "versions" in result.stdout


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


class TestDashboardCommands:
    """Tests for dashboard subcommands."""

    def test_dashboard_help(self) -> None:
        """Dashboard help should show subcommands."""
        result = runner.invoke(app, ["dashboard", "--help"])

        assert result.exit_code == 0
        assert "build" in result.stdout
        assert "deploy" in result.stdout
