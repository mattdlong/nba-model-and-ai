"""Unit tests for DashboardBuilder class."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from nba_model.output.dashboard import (
    DashboardBuilder,
    OutputWriteError,
    build_dashboard,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# Mock Data Classes
# =============================================================================


@dataclass
class MockGamePrediction:
    """Mock GamePrediction for testing."""

    game_id: str = "0022400001"
    game_date: date = field(default_factory=date.today)
    home_team: str = "BOS"
    away_team: str = "LAL"
    matchup: str = "LAL @ BOS"
    home_win_prob: float = 0.65
    predicted_margin: float = 5.5
    predicted_total: float = 220.5
    home_win_prob_adjusted: float = 0.62
    predicted_margin_adjusted: float = 4.8
    predicted_total_adjusted: float = 218.0
    confidence: float = 0.75
    injury_uncertainty: float = 0.15
    top_factors: list = field(default_factory=lambda: [("home_court", 0.12)])
    model_version: str = "v1.0.0"
    prediction_timestamp: datetime = field(default_factory=datetime.now)
    home_lineup: list = field(default_factory=list)
    away_lineup: list = field(default_factory=list)
    inference_time_ms: float = 150.0


@dataclass
class MockBettingSignal:
    """Mock BettingSignal for testing."""

    game_id: str = "0022400001"
    game_date: date = field(default_factory=date.today)
    matchup: str = "LAL @ BOS"
    bet_type: str = "moneyline"
    side: str = "home"
    line: float | None = None
    model_prob: float = 0.62
    market_prob: float = 0.55
    edge: float = 0.07
    recommended_odds: float = 1.82
    kelly_fraction: float = 0.08
    recommended_stake_pct: float = 0.02
    confidence: str = "high"
    key_factors: list = field(default_factory=lambda: ["home_court"])
    injury_notes: list = field(default_factory=list)
    model_confidence: float = 0.75
    injury_uncertainty: float = 0.15


# =============================================================================
# Test DashboardBuilder
# =============================================================================


class TestDashboardBuilder:
    """Tests for DashboardBuilder class."""

    @pytest.fixture
    def output_dir(self, tmp_path: Path) -> Path:
        """Create temporary output directory."""
        return tmp_path / "docs"

    @pytest.fixture
    def template_dir(self, tmp_path: Path) -> Path:
        """Create temporary template directory with minimal templates."""
        templates = tmp_path / "templates"
        templates.mkdir()

        # Create base template
        base = templates / "base.html"
        base.write_text("""
<!DOCTYPE html>
<html>
<head><title>{% block title %}{% endblock %}</title></head>
<body>{% block content %}{% endblock %}</body>
</html>
""")

        # Create index template
        index = templates / "index.html"
        index.write_text("""
{% extends "base.html" %}
{% block title %}Dashboard{% endblock %}
{% block content %}
<h1>Dashboard</h1>
<p>Generated: {{ generated_at }}</p>
{% endblock %}
""")

        # Create predictions template
        preds = templates / "predictions.html"
        preds.write_text("""
{% extends "base.html" %}
{% block content %}
<h1>Predictions</h1>
{% for game in predictions %}
<div>{{ game.matchup }}</div>
{% endfor %}
{% endblock %}
""")

        # Create history template
        history = templates / "history.html"
        history.write_text("""
{% extends "base.html" %}
{% block content %}
<h1>History</h1>
{% endblock %}
""")

        # Create model template
        model = templates / "model.html"
        model.write_text("""
{% extends "base.html" %}
{% block content %}
<h1>Model Health</h1>
<p>Status: {{ health.status }}</p>
{% endblock %}
""")

        return templates

    @pytest.fixture
    def builder(
        self,
        output_dir: Path,
        template_dir: Path,
    ) -> DashboardBuilder:
        """Create DashboardBuilder with temp directories."""
        return DashboardBuilder(
            output_dir=output_dir,
            template_dir=template_dir,
        )

    @pytest.fixture
    def sample_predictions(self) -> list[MockGamePrediction]:
        """Create sample predictions."""
        return [
            MockGamePrediction(game_id="001", matchup="LAL @ BOS"),
            MockGamePrediction(game_id="002", matchup="GSW @ PHX"),
        ]

    @pytest.fixture
    def sample_signals(self) -> list[MockBettingSignal]:
        """Create sample signals."""
        return [
            MockBettingSignal(game_id="001", edge=0.07),
            MockBettingSignal(game_id="002", edge=0.04),
        ]

    # -------------------------------------------------------------------------
    # Build Full Site Tests
    # -------------------------------------------------------------------------

    def test_build_full_site_creates_directories(
        self,
        builder: DashboardBuilder,
        output_dir: Path,
    ) -> None:
        """Build should create output directory structure."""
        builder.build_full_site()

        assert output_dir.exists()
        assert (output_dir / "api").exists()
        assert (output_dir / "api" / "history").exists()
        assert (output_dir / "assets").exists()

    def test_build_full_site_creates_json_files(
        self,
        builder: DashboardBuilder,
        output_dir: Path,
    ) -> None:
        """Build should create JSON data files."""
        builder.build_full_site()

        assert (output_dir / "api" / "today.json").exists()
        assert (output_dir / "api" / "signals.json").exists()
        assert (output_dir / "api" / "performance.json").exists()

    def test_build_full_site_renders_html(
        self,
        builder: DashboardBuilder,
        output_dir: Path,
    ) -> None:
        """Build should render HTML pages from templates."""
        builder.build_full_site()

        assert (output_dir / "index.html").exists()
        assert (output_dir / "predictions.html").exists()
        assert (output_dir / "history.html").exists()
        assert (output_dir / "model.html").exists()

    def test_build_full_site_returns_file_count(
        self,
        builder: DashboardBuilder,
    ) -> None:
        """Build should return count of created files."""
        count = builder.build_full_site()

        assert count > 0

    def test_build_full_site_valid_json(
        self,
        builder: DashboardBuilder,
        output_dir: Path,
    ) -> None:
        """Build should create valid JSON files."""
        builder.build_full_site()

        today_json = output_dir / "api" / "today.json"
        data = json.loads(today_json.read_text())

        assert "date" in data
        assert "games" in data
        assert "signals" in data

    def test_build_full_site_valid_html(
        self,
        builder: DashboardBuilder,
        output_dir: Path,
    ) -> None:
        """Build should create valid HTML files."""
        builder.build_full_site()

        index_html = output_dir / "index.html"
        content = index_html.read_text()

        assert "<!DOCTYPE html>" in content
        assert "<html>" in content
        assert "</html>" in content

    # -------------------------------------------------------------------------
    # Update Predictions Tests
    # -------------------------------------------------------------------------

    def test_update_predictions_writes_today_json(
        self,
        builder: DashboardBuilder,
        output_dir: Path,
        sample_predictions: list[MockGamePrediction],
        sample_signals: list[MockBettingSignal],
    ) -> None:
        """Update predictions should write today.json."""
        builder.update_predictions(sample_predictions, sample_signals)

        today_json = output_dir / "api" / "today.json"
        assert today_json.exists()

        data = json.loads(today_json.read_text())
        assert len(data["games"]) == 2
        assert len(data["signals"]) == 2

    def test_update_predictions_writes_signals_json(
        self,
        builder: DashboardBuilder,
        output_dir: Path,
        sample_predictions: list[MockGamePrediction],
        sample_signals: list[MockBettingSignal],
    ) -> None:
        """Update predictions should write signals.json."""
        builder.update_predictions(sample_predictions, sample_signals)

        signals_json = output_dir / "api" / "signals.json"
        assert signals_json.exists()

        data = json.loads(signals_json.read_text())
        assert "signals" in data
        assert "generated_at" in data

    def test_update_predictions_renders_predictions_page(
        self,
        builder: DashboardBuilder,
        output_dir: Path,
        sample_predictions: list[MockGamePrediction],
        sample_signals: list[MockBettingSignal],
    ) -> None:
        """Update predictions should render predictions.html."""
        builder.update_predictions(sample_predictions, sample_signals)

        preds_html = output_dir / "predictions.html"
        assert preds_html.exists()

        content = preds_html.read_text()
        assert "LAL @ BOS" in content

    def test_update_predictions_empty_lists(
        self,
        builder: DashboardBuilder,
        output_dir: Path,
    ) -> None:
        """Update predictions should handle empty lists."""
        builder.update_predictions([], [])

        today_json = output_dir / "api" / "today.json"
        assert today_json.exists()

        data = json.loads(today_json.read_text())
        assert data["games"] == []
        assert data["signals"] == []

    # -------------------------------------------------------------------------
    # Update Performance Tests
    # -------------------------------------------------------------------------

    def test_update_performance_writes_json(
        self,
        builder: DashboardBuilder,
        output_dir: Path,
    ) -> None:
        """Update performance should write performance.json."""
        builder.update_performance(metrics=None, bets=None, bankroll_history=None)

        perf_json = output_dir / "api" / "performance.json"
        assert perf_json.exists()

    def test_update_performance_includes_charts(
        self,
        builder: DashboardBuilder,
        output_dir: Path,
    ) -> None:
        """Update performance should include chart data."""
        history = [10000.0, 10500.0, 10300.0]
        builder.update_performance(bankroll_history=history)

        perf_json = output_dir / "api" / "performance.json"
        data = json.loads(perf_json.read_text())

        assert "charts" in data
        assert "bankroll" in data["charts"]

    def test_update_performance_renders_history_page(
        self,
        builder: DashboardBuilder,
        output_dir: Path,
    ) -> None:
        """Update performance should render history.html."""
        builder.update_performance()

        history_html = output_dir / "history.html"
        assert history_html.exists()

    # -------------------------------------------------------------------------
    # Update Model Health Tests
    # -------------------------------------------------------------------------

    def test_update_model_health_renders_model_page(
        self,
        builder: DashboardBuilder,
        output_dir: Path,
    ) -> None:
        """Update model health should render model.html."""
        drift_result = {
            "has_drift": False,
            "features_drifted": [],
            "details": {},
        }
        builder.update_model_health(drift_results=drift_result)

        model_html = output_dir / "model.html"
        assert model_html.exists()

        content = model_html.read_text()
        assert "Model Health" in content

    def test_update_model_health_includes_status(
        self,
        builder: DashboardBuilder,
        output_dir: Path,
    ) -> None:
        """Update model health should show status in page."""
        drift_result = {
            "has_drift": False,
            "features_drifted": [],
            "details": {},
        }
        builder.update_model_health(drift_results=drift_result)

        model_html = output_dir / "model.html"
        content = model_html.read_text()

        assert "healthy" in content

    # -------------------------------------------------------------------------
    # Archive Day Tests
    # -------------------------------------------------------------------------

    def test_archive_day_creates_history_file(
        self,
        builder: DashboardBuilder,
        output_dir: Path,
        sample_predictions: list[MockGamePrediction],
        sample_signals: list[MockBettingSignal],
    ) -> None:
        """Archive day should create dated file in history."""
        # First create today's predictions
        builder.update_predictions(sample_predictions, sample_signals)

        # Archive
        archive_date = date(2024, 1, 15)
        archive_path = builder.archive_day(archive_date)

        assert archive_path.exists()
        assert archive_path.name == "2024-01-15.json"

    def test_archive_day_copies_today_content(
        self,
        builder: DashboardBuilder,
        output_dir: Path,
        sample_predictions: list[MockGamePrediction],
        sample_signals: list[MockBettingSignal],
    ) -> None:
        """Archive day should copy today's content to archive."""
        builder.update_predictions(sample_predictions, sample_signals)

        archive_date = date(2024, 1, 15)
        archive_path = builder.archive_day(archive_date)

        data = json.loads(archive_path.read_text())
        assert len(data["games"]) == 2

    def test_archive_day_creates_empty_if_no_today(
        self,
        builder: DashboardBuilder,
        output_dir: Path,
    ) -> None:
        """Archive day should create empty archive if today.json doesn't exist."""
        archive_date = date(2024, 1, 15)
        archive_path = builder.archive_day(archive_date)

        assert archive_path.exists()
        data = json.loads(archive_path.read_text())
        assert data["games"] == []

    def test_archive_day_default_date(
        self,
        builder: DashboardBuilder,
        output_dir: Path,
    ) -> None:
        """Archive day should use today's date by default."""
        builder.build_full_site()
        archive_path = builder.archive_day()

        assert archive_path.name == f"{date.today().isoformat()}.json"

    # -------------------------------------------------------------------------
    # No Template Directory Tests
    # -------------------------------------------------------------------------

    def test_build_without_templates(
        self,
        output_dir: Path,
    ) -> None:
        """Build should work without templates (JSON only)."""
        builder = DashboardBuilder(
            output_dir=output_dir,
            template_dir=Path("/nonexistent/templates"),
        )

        count = builder.build_full_site()

        # Should still create JSON files
        assert count >= 3
        assert (output_dir / "api" / "today.json").exists()


# =============================================================================
# Test Convenience Function
# =============================================================================


class TestBuildDashboard:
    """Tests for build_dashboard convenience function."""

    def test_build_dashboard_returns_file_count(
        self,
        tmp_path: Path,
    ) -> None:
        """Convenience function should return file count."""
        output_dir = tmp_path / "docs"

        count = build_dashboard(
            output_dir=output_dir,
            template_dir=Path("/nonexistent"),
        )

        assert count > 0

    def test_build_dashboard_creates_structure(
        self,
        tmp_path: Path,
    ) -> None:
        """Convenience function should create directory structure."""
        output_dir = tmp_path / "docs"

        build_dashboard(
            output_dir=output_dir,
            template_dir=Path("/nonexistent"),
        )

        assert output_dir.exists()
        assert (output_dir / "api").exists()
