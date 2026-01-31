"""Output generation for NBA model.

This module provides report generation and dashboard building functionality
for presenting predictions and performance metrics.

Submodules:
    reports: Report generation (daily, performance, health)
    dashboard: GitHub Pages static site builder

Key concepts:
    - Static site generation for GitHub Pages
    - JSON data files for frontend consumption
    - Jinja2 templates for HTML rendering

Example:
    >>> from nba_model.output import DashboardBuilder
    >>> builder = DashboardBuilder(output_dir="docs")
    >>> builder.update_predictions(predictions, signals)
    >>> builder.build_full_site()
"""
from __future__ import annotations

# Public API - will be populated in Phase 8
__all__: list[str] = []
