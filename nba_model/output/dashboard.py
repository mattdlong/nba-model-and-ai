"""GitHub Pages static dashboard builder for NBA model.

This module generates a complete static site for GitHub Pages hosting,
including HTML pages rendered from Jinja2 templates and JSON data files
for client-side consumption.

Site Structure:
    docs/
    ├── index.html           # Main dashboard
    ├── predictions.html     # Today's predictions
    ├── history.html         # Historical performance
    ├── model.html           # Model info and health
    ├── api/                 # JSON data files
    │   ├── today.json
    │   ├── signals.json
    │   ├── performance.json
    │   └── history/{YYYY-MM-DD}.json
    └── assets/
        ├── style.css
        └── charts.js

Example:
    >>> from nba_model.output import DashboardBuilder
    >>> builder = DashboardBuilder()
    >>> builder.update_predictions(predictions, signals)
    >>> builder.build_full_site()
"""

from __future__ import annotations

import json
import shutil
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jinja2 import Environment, FileSystemLoader, TemplateNotFound

from nba_model.logging import get_logger
from nba_model.output.charts import ChartGenerator
from nba_model.output.reports import ReportGenerator

if TYPE_CHECKING:
    from nba_model.backtest.metrics import FullBacktestMetrics
    from nba_model.monitor.drift import DriftCheckResult
    from nba_model.predict.inference import GamePrediction
    from nba_model.predict.signals import BettingSignal
    from nba_model.types import Bet

logger = get_logger(__name__)

# =============================================================================
# Constants
# =============================================================================

DEFAULT_OUTPUT_DIR: str = "docs"
DEFAULT_TEMPLATE_DIR: str = "templates"

# Template file names
TEMPLATE_FILES: list[str] = [
    "base.html",
    "index.html",
    "predictions.html",
    "history.html",
    "model.html",
]

# Static asset files
ASSET_FILES: list[str] = [
    "style.css",
    "charts.js",
]

# JSON data files
JSON_FILES: list[str] = [
    "today.json",
    "signals.json",
    "performance.json",
]


# =============================================================================
# Exceptions
# =============================================================================


class DashboardBuildError(Exception):
    """Base exception for dashboard building errors."""


class TemplateLoadError(DashboardBuildError):
    """Failed to load template file."""


class OutputWriteError(DashboardBuildError):
    """Failed to write output file."""


# =============================================================================
# Dashboard Builder
# =============================================================================


class DashboardBuilder:
    """Build static GitHub Pages dashboard site.

    Generates a complete static site with HTML pages and JSON data files
    suitable for hosting on GitHub Pages.

    Attributes:
        output_dir: Directory for generated files (default: 'docs').
        template_dir: Directory containing Jinja2 templates (default: 'templates').

    Example:
        >>> builder = DashboardBuilder()
        >>> builder.update_predictions(predictions, signals)
        >>> builder.build_full_site()
        >>> # Files generated in docs/ directory
    """

    def __init__(
        self,
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
        template_dir: str | Path = DEFAULT_TEMPLATE_DIR,
    ) -> None:
        """Initialize DashboardBuilder.

        Args:
            output_dir: Output directory for generated site.
            template_dir: Directory containing Jinja2 templates.
        """
        self.output_dir = Path(output_dir)
        self.template_dir = Path(template_dir)

        self._report_generator = ReportGenerator()
        self._chart_generator = ChartGenerator()
        self._jinja_env: Environment | None = None

    @property
    def jinja_env(self) -> Environment:
        """Get Jinja2 environment, creating if necessary.

        Returns:
            Configured Jinja2 Environment.

        Raises:
            TemplateLoadError: If template directory not found.
        """
        if self._jinja_env is None:
            if not self.template_dir.exists():
                logger.warning(
                    "Template directory '{}' not found, creating empty environment",
                    self.template_dir
                )
                self._jinja_env = Environment()
            else:
                self._jinja_env = Environment(
                    loader=FileSystemLoader(str(self.template_dir)),
                    autoescape=True,
                )
        return self._jinja_env

    def build_full_site(self) -> int:
        """Generate complete static site from templates.

        Creates all HTML pages, copies static assets, and initializes
        JSON data directory structure.

        Returns:
            Number of files created.

        Raises:
            DashboardBuildError: If site generation fails.

        Example:
            >>> file_count = builder.build_full_site()
            >>> print(f"Created {file_count} files")
        """
        logger.info("Building full dashboard site to {}", self.output_dir)

        file_count = 0

        # Create directory structure
        self._ensure_directories()

        # Render HTML pages
        pages_rendered = self._render_all_pages()
        file_count += pages_rendered

        # Copy static assets
        assets_copied = self._copy_static_assets()
        file_count += assets_copied

        # Initialize JSON data files with empty/placeholder data
        json_created = self._initialize_json_files()
        file_count += json_created

        logger.info("Dashboard build complete: {} files created", file_count)
        return file_count

    def update_predictions(
        self,
        predictions: list[GamePrediction],
        signals: list[BettingSignal],
    ) -> None:
        """Update today's predictions page and JSON data.

        Renders predictions.html template and writes today.json and
        signals.json to the api directory.

        Args:
            predictions: List of GamePrediction objects.
            signals: List of BettingSignal objects.

        Example:
            >>> builder.update_predictions(predictions, signals)
            >>> # Updates docs/predictions.html, docs/api/today.json, docs/api/signals.json
        """
        logger.info(
            "Updating predictions with {} games, {} signals",
            len(predictions), len(signals)
        )

        # Generate report data
        report = self._report_generator.daily_predictions_report(predictions, signals)

        # Write JSON files
        api_dir = self.output_dir / "api"
        api_dir.mkdir(parents=True, exist_ok=True)

        self._write_json(api_dir / "today.json", report)
        self._write_json(
            api_dir / "signals.json",
            {"signals": report["signals"], "generated_at": report["generated_at"]},
        )

        # Render predictions page
        self._render_page(
            "predictions.html",
            predictions=report["games"],
            signals=report["signals"],
            summary=report["summary"],
            generated_at=report["generated_at"],
        )

        # Also update index with latest data
        self._render_index_page(report)

    def update_performance(
        self,
        metrics: FullBacktestMetrics | None = None,
        bets: list[Bet] | None = None,
        bankroll_history: list[float] | None = None,
    ) -> None:
        """Update performance tracking page and data.

        Renders history.html and writes performance.json with updated
        performance metrics and chart data.

        Args:
            metrics: Pre-computed backtest metrics.
            bets: Historical bet list for calculations.
            bankroll_history: Bankroll values for chart.

        Example:
            >>> builder.update_performance(metrics=metrics, bankroll_history=history)
        """
        logger.info("Updating performance data")

        # Generate performance report
        perf_report = self._report_generator.performance_report(
            "month", metrics=metrics, bets=bets
        )

        # Generate chart data
        charts = {}
        if bankroll_history:
            charts["bankroll"] = self._chart_generator.bankroll_chart(bankroll_history)

        if bets:
            charts["roi_by_month"] = self._chart_generator.roi_by_month_chart(bets)
            charts["win_rate_trend"] = self._chart_generator.win_rate_trend_chart(bets)

        # Write JSON
        api_dir = self.output_dir / "api"
        api_dir.mkdir(parents=True, exist_ok=True)

        self._write_json(
            api_dir / "performance.json",
            {"metrics": perf_report, "charts": charts},
        )

        # Render history page
        self._render_page(
            "history.html",
            metrics=perf_report,
            charts=charts,
            generated_at=datetime.now().isoformat(),
        )

    def update_model_health(
        self,
        drift_results: DriftCheckResult | dict[str, Any] | None = None,
        recent_metrics: dict[str, float] | None = None,
        model_info: dict[str, Any] | None = None,
    ) -> None:
        """Update model health page and data.

        Renders model.html with drift detection status, feature stability,
        and retraining recommendations.

        Args:
            drift_results: Drift detection check result.
            recent_metrics: Recent performance metrics.
            model_info: Model metadata (version, training date, etc.).

        Example:
            >>> builder.update_model_health(drift_result, metrics, model_info)
        """
        logger.info("Updating model health data")

        # Generate health report
        health_report = self._report_generator.model_health_report(
            drift_results=drift_results,
            recent_metrics=recent_metrics,
        )

        model_info = model_info or {}

        # Render model page
        self._render_page(
            "model.html",
            health=health_report,
            model_info=model_info,
            generated_at=datetime.now().isoformat(),
        )

    def archive_day(self, archive_date: date | None = None) -> Path:
        """Archive current day predictions to history.

        Moves current today.json to history directory with date-stamped
        filename.

        Args:
            archive_date: Date to archive (default: today).

        Returns:
            Path to archived file.

        Example:
            >>> archive_path = builder.archive_day()
            >>> print(f"Archived to {archive_path}")
        """
        archive_date = archive_date or date.today()
        logger.info("Archiving predictions for {}", archive_date)

        history_dir = self.output_dir / "api" / "history"
        history_dir.mkdir(parents=True, exist_ok=True)

        today_file = self.output_dir / "api" / "today.json"
        archive_file = history_dir / f"{archive_date.isoformat()}.json"

        if today_file.exists():
            shutil.copy2(today_file, archive_file)
            logger.debug("Archived {} to {}", today_file, archive_file)
        else:
            # Create empty archive file
            self._write_json(
                archive_file,
                {"date": archive_date.isoformat(), "games": [], "signals": []},
            )

        return archive_file

    def _ensure_directories(self) -> None:
        """Create output directory structure.

        Creates:
        - docs/
        - docs/api/
        - docs/api/history/
        - docs/assets/
        """
        directories = [
            self.output_dir,
            self.output_dir / "api",
            self.output_dir / "api" / "history",
            self.output_dir / "assets",
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug("Ensured directory: {}", directory)

    def _render_all_pages(self) -> int:
        """Render all HTML pages from templates.

        Returns:
            Number of pages rendered.
        """
        pages_rendered = 0

        # Check if templates exist
        if not self.template_dir.exists():
            logger.warning("Template directory not found, skipping HTML rendering")
            return 0

        pages = ["index.html", "predictions.html", "history.html", "model.html"]

        for page in pages:
            try:
                self._render_page(page)
                pages_rendered += 1
            except TemplateNotFound:
                logger.debug("Template {} not found, skipping", page)

        return pages_rendered

    def _render_page(self, template_name: str, **context: Any) -> None:
        """Render a single page template.

        Args:
            template_name: Template filename.
            **context: Template context variables.
        """
        try:
            template = self.jinja_env.get_template(template_name)
        except TemplateNotFound:
            logger.debug("Template {} not found", template_name)
            return

        # Add common context
        context.setdefault("generated_at", datetime.now().isoformat())
        context.setdefault("page_name", template_name.replace(".html", ""))

        html = template.render(**context)

        output_path = self.output_dir / template_name
        output_path.write_text(html, encoding="utf-8")
        logger.debug("Rendered {} to {}", template_name, output_path)

    def _render_index_page(self, daily_report: dict[str, Any]) -> None:
        """Render index page with latest data.

        Args:
            daily_report: Daily predictions report data.
        """
        # Get top 3 signals for preview
        top_signals = daily_report.get("signals", [])[:3]

        self._render_page(
            "index.html",
            summary=daily_report.get("summary", {}),
            top_signals=top_signals,
            prediction_date=daily_report.get("date", date.today().isoformat()),
        )

    def _copy_static_assets(self) -> int:
        """Copy static assets to output directory.

        Assets are located in docs/assets (the default output location).
        When building to a custom output directory, we copy from docs/assets.

        Returns:
            Number of assets copied.
        """
        assets_copied = 0
        dest_dir = self.output_dir / "assets"
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Assets live in docs/assets (the canonical location)
        # Check multiple potential source locations in order of preference
        potential_sources = [
            Path("docs") / "assets",  # Default canonical location
            self.template_dir / "assets",  # Template-relative (backwards compat)
        ]

        source_dir = None
        for candidate in potential_sources:
            if candidate.exists() and candidate.is_dir():
                source_dir = candidate
                break

        if source_dir:
            for asset_file in source_dir.iterdir():
                if asset_file.is_file():
                    shutil.copy2(asset_file, dest_dir / asset_file.name)
                    assets_copied += 1
                    logger.debug("Copied asset: {}", asset_file.name)

        return assets_copied

    def _initialize_json_files(self) -> int:
        """Initialize JSON data files with placeholder data.

        Returns:
            Number of files created.
        """
        api_dir = self.output_dir / "api"
        files_created = 0

        # today.json
        today_file = api_dir / "today.json"
        if not today_file.exists():
            self._write_json(
                today_file,
                {
                    "date": date.today().isoformat(),
                    "generated_at": datetime.now().isoformat(),
                    "games": [],
                    "signals": [],
                    "summary": {},
                },
            )
            files_created += 1

        # signals.json
        signals_file = api_dir / "signals.json"
        if not signals_file.exists():
            self._write_json(
                signals_file,
                {
                    "signals": [],
                    "generated_at": datetime.now().isoformat(),
                },
            )
            files_created += 1

        # performance.json
        perf_file = api_dir / "performance.json"
        if not perf_file.exists():
            self._write_json(
                perf_file,
                {
                    "metrics": {},
                    "charts": {},
                },
            )
            files_created += 1

        return files_created

    def _write_json(self, path: Path, data: dict[str, Any]) -> None:
        """Write data to JSON file.

        Args:
            path: Output file path.
            data: Dictionary to serialize.

        Raises:
            OutputWriteError: If write fails.
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(data, indent=2, default=str),
                encoding="utf-8",
            )
            logger.debug("Wrote JSON to {}", path)
        except OSError as e:
            raise OutputWriteError(f"Failed to write {path}: {e}") from e


# =============================================================================
# Convenience Functions
# =============================================================================


def build_dashboard(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    template_dir: str | Path = DEFAULT_TEMPLATE_DIR,
) -> int:
    """Convenience function to build full dashboard.

    Args:
        output_dir: Output directory.
        template_dir: Template directory.

    Returns:
        Number of files created.

    Example:
        >>> file_count = build_dashboard()
        >>> print(f"Dashboard built with {file_count} files")
    """
    builder = DashboardBuilder(output_dir, template_dir)
    return builder.build_full_site()
