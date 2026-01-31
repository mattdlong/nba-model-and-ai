"""Output generation for NBA model.

This module provides report generation and dashboard building functionality
for presenting predictions and performance metrics.

Submodules:
    reports: Report generation (daily, performance, health)
    charts: Chart data generation for visualizations
    dashboard: GitHub Pages static site builder

Key concepts:
    - Static site generation for GitHub Pages
    - JSON data files for frontend consumption
    - Jinja2 templates for HTML rendering

Example:
    >>> from nba_model.output import DashboardBuilder, ReportGenerator, ChartGenerator
    >>> builder = DashboardBuilder(output_dir="docs")
    >>> builder.update_predictions(predictions, signals)
    >>> builder.build_full_site()
"""

from __future__ import annotations

from nba_model.output.charts import (
    ChartGenerationError,
    ChartGenerator,
    InsufficientDataError as ChartInsufficientDataError,
)
from nba_model.output.dashboard import (
    DashboardBuildError,
    DashboardBuilder,
    OutputWriteError,
    TemplateLoadError,
    build_dashboard,
)
from nba_model.output.reports import (
    InvalidPeriodError,
    ReportGenerationError,
    ReportGenerator,
)

__all__: list[str] = [
    # Reports
    "ReportGenerator",
    "ReportGenerationError",
    "InvalidPeriodError",
    # Charts
    "ChartGenerator",
    "ChartGenerationError",
    "ChartInsufficientDataError",
    # Dashboard
    "DashboardBuilder",
    "DashboardBuildError",
    "TemplateLoadError",
    "OutputWriteError",
    "build_dashboard",
]
