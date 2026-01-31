"""Chart data generation for NBA model dashboard.

This module generates JavaScript-compatible chart data structures for
Chart.js visualization in the GitHub Pages dashboard.

Example:
    >>> from nba_model.output import ChartGenerator
    >>> generator = ChartGenerator()
    >>> chart_data = generator.bankroll_chart(history)
    >>> # Returns Chart.js compatible data structure
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from nba_model.logging import get_logger

if TYPE_CHECKING:
    from nba_model.types import Bet

logger = get_logger(__name__)

# =============================================================================
# Constants
# =============================================================================

# Calibration chart bins
CALIBRATION_BINS: list[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
NUM_CALIBRATION_BINS: int = 10

# Chart color scheme
CHART_COLORS = {
    "primary": "rgb(59, 130, 246)",  # Blue
    "secondary": "rgb(16, 185, 129)",  # Green
    "danger": "rgb(239, 68, 68)",  # Red
    "warning": "rgb(245, 158, 11)",  # Amber
    "neutral": "rgb(156, 163, 175)",  # Gray
    "perfect": "rgb(107, 114, 128)",  # Darker gray for perfect calibration line
}


# =============================================================================
# Exceptions
# =============================================================================


class ChartGenerationError(Exception):
    """Base exception for chart generation errors."""


class InsufficientDataError(ChartGenerationError):
    """Raised when there is insufficient data to generate a chart."""


# =============================================================================
# Chart Generator
# =============================================================================


class ChartGenerator:
    """Generate chart data for dashboard visualizations.

    Produces Chart.js-compatible data structures for:
    - Bankroll growth line chart
    - Probability calibration plot
    - ROI by month bar chart

    All methods return dictionaries that can be serialized to JSON
    and consumed directly by Chart.js in the dashboard.

    Example:
        >>> generator = ChartGenerator()
        >>> bankroll_data = generator.bankroll_chart([10000, 10250, 10180, 10500])
        >>> # Can be used directly with Chart.js
    """

    def __init__(self) -> None:
        """Initialize ChartGenerator."""
        pass

    def bankroll_chart(
        self,
        bankroll_history: list[float],
        dates: list[date | str] | None = None,
    ) -> dict[str, Any]:
        """Generate bankroll growth chart data.

        Creates a line chart data structure showing bankroll progression
        over time.

        Args:
            bankroll_history: List of bankroll values over time.
            dates: Optional list of dates for x-axis labels.
                If not provided, uses indices.

        Returns:
            Dictionary with Chart.js-compatible structure:
            - labels: X-axis labels (dates or indices)
            - datasets: List with single dataset for bankroll line

        Example:
            >>> data = generator.bankroll_chart([10000, 10500, 10300])
            >>> data['labels']
            ['1', '2', '3']
            >>> data['datasets'][0]['data']
            [10000, 10500, 10300]
        """
        if not bankroll_history:
            return self._empty_line_chart()

        logger.debug("Generating bankroll chart with {} data points", len(bankroll_history))

        # Generate labels
        if dates:
            labels = [
                d.isoformat() if isinstance(d, date) else str(d)
                for d in dates
            ]
        else:
            labels = [str(i + 1) for i in range(len(bankroll_history))]

        # Determine color based on performance
        initial = bankroll_history[0] if bankroll_history else 0
        final = bankroll_history[-1] if bankroll_history else 0
        color = CHART_COLORS["secondary"] if final >= initial else CHART_COLORS["danger"]

        return {
            "labels": labels,
            "datasets": [
                {
                    "label": "Bankroll",
                    "data": [round(v, 2) for v in bankroll_history],
                    "borderColor": color,
                    "backgroundColor": color.replace("rgb", "rgba").replace(")", ", 0.1)"),
                    "fill": True,
                    "tension": 0.1,
                }
            ],
        }

    def calibration_chart(
        self,
        predictions: list[float],
        actuals: list[int],
    ) -> dict[str, Any]:
        """Generate probability calibration chart data.

        Creates a calibration plot comparing predicted probabilities to
        actual outcomes. Perfect calibration appears as a diagonal line.

        Args:
            predictions: List of predicted probabilities (0.0 to 1.0).
            actuals: List of actual outcomes (0 or 1).

        Returns:
            Dictionary with Chart.js-compatible structure containing:
            - labels: Probability bin centers
            - datasets: Actual rates and perfect calibration line

        Raises:
            ValueError: If predictions and actuals have different lengths.

        Example:
            >>> data = generator.calibration_chart([0.6, 0.7, 0.3], [1, 1, 0])
            >>> # Shows how well predictions match actual outcomes
        """
        if len(predictions) != len(actuals):
            raise ValueError(
                f"predictions and actuals must have same length: "
                f"{len(predictions)} vs {len(actuals)}"
            )

        if not predictions:
            return self._empty_calibration_chart()

        logger.debug("Generating calibration chart with {} predictions", len(predictions))

        # Bin predictions and calculate actual rates
        bin_centers, actual_rates, counts = self._calculate_calibration_bins(
            predictions, actuals
        )

        return {
            "labels": [f"{c:.0%}" for c in bin_centers],
            "datasets": [
                {
                    "label": "Actual Rate",
                    "data": [round(r, 4) if r is not None else None for r in actual_rates],
                    "borderColor": CHART_COLORS["primary"],
                    "backgroundColor": CHART_COLORS["primary"],
                    "pointRadius": [max(3, min(10, c // 5)) for c in counts],
                    "fill": False,
                    "tension": 0,
                },
                {
                    "label": "Perfect Calibration",
                    "data": bin_centers,
                    "borderColor": CHART_COLORS["perfect"],
                    "borderDash": [5, 5],
                    "pointRadius": 0,
                    "fill": False,
                    "tension": 0,
                },
            ],
            "metadata": {
                "bin_counts": counts,
                "total_predictions": len(predictions),
            },
        }

    def roi_by_month_chart(
        self,
        bets: list[Bet],
    ) -> dict[str, Any]:
        """Generate ROI by month bar chart data.

        Aggregates betting results by calendar month and shows ROI
        for each month as a bar chart.

        Args:
            bets: List of Bet objects with profit and bet_amount.

        Returns:
            Dictionary with Chart.js-compatible structure:
            - labels: Month labels (YYYY-MM format)
            - datasets: ROI values for each month with color coding

        Example:
            >>> data = generator.roi_by_month_chart(bets)
            >>> # Shows monthly performance as bar chart
        """
        if not bets:
            return self._empty_bar_chart()

        logger.debug("Generating ROI by month chart with {} bets", len(bets))

        # Aggregate by month
        monthly_data = self._aggregate_by_month(bets)

        # Sort by month
        sorted_months = sorted(monthly_data.keys())

        labels = sorted_months
        roi_values = [monthly_data[m]["roi"] for m in sorted_months]

        # Color code by positive/negative
        colors = [
            CHART_COLORS["secondary"] if roi >= 0 else CHART_COLORS["danger"]
            for roi in roi_values
        ]

        return {
            "labels": labels,
            "datasets": [
                {
                    "label": "ROI %",
                    "data": [round(roi * 100, 2) for roi in roi_values],
                    "backgroundColor": colors,
                    "borderColor": colors,
                    "borderWidth": 1,
                }
            ],
            "metadata": {
                "monthly_stats": {
                    m: {
                        "bets": monthly_data[m]["count"],
                        "wagered": round(monthly_data[m]["wagered"], 2),
                        "profit": round(monthly_data[m]["profit"], 2),
                    }
                    for m in sorted_months
                },
            },
        }

    def win_rate_trend_chart(
        self,
        bets: list[Bet],
        window: int = 50,
    ) -> dict[str, Any]:
        """Generate rolling win rate trend chart.

        Shows win rate over a rolling window of bets to visualize
        performance trends.

        Args:
            bets: List of Bet objects with results.
            window: Rolling window size (default: 50).

        Returns:
            Dictionary with Chart.js-compatible line chart structure.
        """
        if not bets or len(bets) < window:
            return self._empty_line_chart()

        logger.debug(
            "Generating win rate trend chart with {} bets, window {}",
            len(bets), window
        )

        # Calculate rolling win rate
        wins = [1 if b.result == "win" else 0 for b in bets]
        rolling_rates = []

        for i in range(window - 1, len(wins)):
            window_wins = wins[i - window + 1 : i + 1]
            rate = sum(window_wins) / len(window_wins)
            rolling_rates.append(rate)

        labels = list(range(window, len(bets) + 1))

        return {
            "labels": [str(i) for i in labels],
            "datasets": [
                {
                    "label": f"Win Rate ({window}-bet rolling)",
                    "data": [round(r, 4) for r in rolling_rates],
                    "borderColor": CHART_COLORS["primary"],
                    "fill": False,
                    "tension": 0.1,
                },
                {
                    "label": "Break-even (52.4%)",
                    "data": [0.524] * len(rolling_rates),
                    "borderColor": CHART_COLORS["neutral"],
                    "borderDash": [5, 5],
                    "pointRadius": 0,
                    "fill": False,
                },
            ],
        }

    def edge_distribution_chart(
        self,
        bets: list[Bet],
    ) -> dict[str, Any]:
        """Generate edge distribution histogram.

        Shows the distribution of betting edges across all bets.

        Args:
            bets: List of Bet objects with edge values.

        Returns:
            Dictionary with Chart.js-compatible bar chart structure.
        """
        if not bets:
            return self._empty_bar_chart()

        edges = [b.edge for b in bets]

        # Create histogram bins
        bins = np.linspace(0, 0.15, 16)  # 0% to 15% in 1% increments
        hist, bin_edges = np.histogram(edges, bins=bins)

        labels = [f"{bin_edges[i]*100:.0f}-{bin_edges[i+1]*100:.0f}%" for i in range(len(hist))]

        return {
            "labels": labels,
            "datasets": [
                {
                    "label": "Bet Count",
                    "data": hist.tolist(),
                    "backgroundColor": CHART_COLORS["primary"],
                }
            ],
        }

    def _calculate_calibration_bins(
        self,
        predictions: list[float],
        actuals: list[int],
    ) -> tuple[list[float], list[float | None], list[int]]:
        """Calculate calibration bin statistics.

        Args:
            predictions: Predicted probabilities.
            actuals: Actual outcomes.

        Returns:
            Tuple of (bin_centers, actual_rates, counts).
        """
        # Initialize bins
        bins: dict[int, list[int]] = {i: [] for i in range(NUM_CALIBRATION_BINS)}

        # Assign predictions to bins
        for pred, actual in zip(predictions, actuals):
            bin_idx = min(int(pred * NUM_CALIBRATION_BINS), NUM_CALIBRATION_BINS - 1)
            bins[bin_idx].append(actual)

        # Calculate statistics per bin
        bin_centers = []
        actual_rates: list[float | None] = []
        counts = []

        for i in range(NUM_CALIBRATION_BINS):
            center = (i + 0.5) / NUM_CALIBRATION_BINS
            bin_centers.append(center)

            bin_actuals = bins[i]
            counts.append(len(bin_actuals))

            if bin_actuals:
                actual_rates.append(sum(bin_actuals) / len(bin_actuals))
            else:
                actual_rates.append(None)

        return bin_centers, actual_rates, counts

    def _aggregate_by_month(
        self, bets: list[Bet]
    ) -> dict[str, dict[str, float | int]]:
        """Aggregate bet statistics by month.

        Args:
            bets: List of bets.

        Returns:
            Dictionary mapping month strings to statistics.
        """
        monthly: dict[str, dict[str, float | int]] = defaultdict(
            lambda: {"wagered": 0.0, "profit": 0.0, "count": 0}
        )

        for bet in bets:
            month_key = bet.timestamp.strftime("%Y-%m")
            monthly[month_key]["wagered"] += bet.bet_amount
            monthly[month_key]["profit"] += bet.profit or 0.0
            monthly[month_key]["count"] += 1

        # Calculate ROI for each month
        result = {}
        for month, stats in monthly.items():
            wagered = stats["wagered"]
            roi = stats["profit"] / wagered if wagered > 0 else 0.0
            result[month] = {
                "wagered": stats["wagered"],
                "profit": stats["profit"],
                "count": stats["count"],
                "roi": roi,
            }

        return result

    def _empty_line_chart(self) -> dict[str, Any]:
        """Return empty line chart structure.

        Returns:
            Empty Chart.js line chart data.
        """
        return {
            "labels": [],
            "datasets": [
                {
                    "label": "No Data",
                    "data": [],
                    "borderColor": CHART_COLORS["neutral"],
                    "fill": False,
                }
            ],
        }

    def _empty_bar_chart(self) -> dict[str, Any]:
        """Return empty bar chart structure.

        Returns:
            Empty Chart.js bar chart data.
        """
        return {
            "labels": [],
            "datasets": [
                {
                    "label": "No Data",
                    "data": [],
                    "backgroundColor": CHART_COLORS["neutral"],
                }
            ],
        }

    def _empty_calibration_chart(self) -> dict[str, Any]:
        """Return empty calibration chart structure.

        Returns:
            Empty calibration chart data.
        """
        bin_centers = [(i + 0.5) / NUM_CALIBRATION_BINS for i in range(NUM_CALIBRATION_BINS)]

        return {
            "labels": [f"{c:.0%}" for c in bin_centers],
            "datasets": [
                {
                    "label": "Actual Rate",
                    "data": [],
                    "borderColor": CHART_COLORS["primary"],
                    "fill": False,
                },
                {
                    "label": "Perfect Calibration",
                    "data": bin_centers,
                    "borderColor": CHART_COLORS["perfect"],
                    "borderDash": [5, 5],
                    "pointRadius": 0,
                    "fill": False,
                },
            ],
            "metadata": {"total_predictions": 0},
        }
