"""Report generation for NBA model predictions and performance.

This module provides comprehensive report generation functionality including
daily prediction reports, performance tracking, and model health summaries.

Example:
    >>> from nba_model.output import ReportGenerator
    >>> generator = ReportGenerator()
    >>> report = generator.daily_predictions_report(predictions, signals)
    >>> print(f"Generated report for {report['date']} with {len(report['games'])} games")
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

from nba_model.logging import get_logger

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

# Confidence thresholds for signal categorization
HIGH_CONFIDENCE_THRESHOLD: float = 0.05
MEDIUM_CONFIDENCE_THRESHOLD: float = 0.03

# Report periods
VALID_PERIODS: set[str] = {"week", "month", "season"}

# Default empty report structures
EMPTY_SUMMARY_STATS: dict[str, int | float] = {
    "total_games": 0,
    "total_signals": 0,
    "high_confidence_signals": 0,
    "medium_confidence_signals": 0,
    "avg_edge": 0.0,
    "max_edge": 0.0,
}


# =============================================================================
# Exceptions
# =============================================================================


class ReportGenerationError(Exception):
    """Base exception for report generation errors."""


class InvalidPeriodError(ReportGenerationError):
    """Invalid report period specified."""


# =============================================================================
# Report Generator
# =============================================================================


class ReportGenerator:
    """Generate various reports for predictions and performance.

    Provides three primary report types:
    - Daily predictions: Today's game predictions and betting signals
    - Performance: Historical accuracy, ROI, CLV metrics
    - Model health: Drift detection and retraining recommendations

    All reports return dictionaries suitable for JSON serialization
    and Jinja2 template rendering.

    Example:
        >>> generator = ReportGenerator()
        >>> daily = generator.daily_predictions_report(predictions, signals)
        >>> perf = generator.performance_report("week", metrics, bets)
        >>> health = generator.model_health_report(drift_result, recent_metrics)
    """

    def __init__(self) -> None:
        """Initialize ReportGenerator."""
        self._generated_at = datetime.now()

    def daily_predictions_report(
        self,
        predictions: list[GamePrediction],
        signals: list[BettingSignal],
    ) -> dict[str, Any]:
        """Generate daily predictions report.

        Creates a comprehensive report of today's game predictions and
        actionable betting signals, suitable for dashboard display.

        Args:
            predictions: List of GamePrediction objects for today's games.
            signals: List of BettingSignal objects with positive edge.

        Returns:
            Dictionary containing:
            - date: Report date as ISO string
            - generated_at: Timestamp of report generation
            - games: List of per-game prediction dictionaries
            - signals: List of actionable signal dictionaries
            - summary: Aggregate statistics

        Example:
            >>> report = generator.daily_predictions_report(predictions, signals)
            >>> for game in report['games']:
            ...     print(f"{game['matchup']}: {game['home_win_prob']:.1%}")
        """
        logger.info(
            "Generating daily predictions report for {} games", len(predictions)
        )

        games = [self._format_prediction(p) for p in predictions]
        formatted_signals = [self._format_signal(s) for s in signals]
        summary = self._calculate_summary(predictions, signals)

        report_date = (
            predictions[0].game_date if predictions else date.today()
        )

        return {
            "date": report_date.isoformat(),
            "generated_at": datetime.now().isoformat(),
            "games": games,
            "signals": formatted_signals,
            "summary": summary,
        }

    def performance_report(
        self,
        period: str,
        metrics: FullBacktestMetrics | None = None,
        bets: list[Bet] | None = None,
    ) -> dict[str, Any]:
        """Generate performance tracking report.

        Creates a report of model performance over a specified period,
        including accuracy, ROI, calibration, and CLV metrics.

        Args:
            period: Time period ("week", "month", or "season").
            metrics: Pre-computed backtest metrics, if available.
            bets: List of historical bets for calculation.

        Returns:
            Dictionary containing:
            - period: Report period
            - generated_at: Timestamp
            - total_predictions: Count of predictions
            - accuracy: Win rate as decimal
            - roi: Return on investment
            - clv: Average closing line value
            - calibration_curve: Data points for calibration plot
            - by_bet_type: Metrics broken down by bet type

        Raises:
            InvalidPeriodError: If period not in valid options.

        Example:
            >>> report = generator.performance_report("week", metrics=metrics)
            >>> print(f"Weekly ROI: {report['roi']:.1%}")
        """
        if period not in VALID_PERIODS:
            raise InvalidPeriodError(
                f"Invalid period '{period}'. Must be one of: {VALID_PERIODS}"
            )

        logger.info("Generating performance report for period: {}", period)

        if metrics is not None:
            return self._format_metrics_report(period, metrics)

        if bets:
            return self._calculate_performance_from_bets(period, bets)

        return self._empty_performance_report(period)

    def model_health_report(
        self,
        drift_results: DriftCheckResult | dict[str, Any] | None = None,
        recent_metrics: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Generate model health/monitoring report.

        Creates a report on model health status including drift detection
        results, feature stability, and retraining recommendations.

        Args:
            drift_results: Drift detection check result or dict.
            recent_metrics: Recent performance metrics dictionary.

        Returns:
            Dictionary containing:
            - status: Overall health status ("healthy", "warning", "critical")
            - generated_at: Timestamp
            - drift_detected: Whether covariate drift detected
            - features_drifted: List of drifted feature names
            - feature_stability: Per-feature stability metrics
            - recent_performance: Recent accuracy/Brier metrics
            - retraining_recommended: Boolean recommendation
            - recommendation_reason: Explanation if retraining recommended

        Example:
            >>> report = generator.model_health_report(drift_result, metrics)
            >>> if report['retraining_recommended']:
            ...     print(f"Retraining needed: {report['recommendation_reason']}")
        """
        logger.info("Generating model health report")

        # Handle DriftCheckResult dataclass or dict
        if drift_results is not None:
            if hasattr(drift_results, "to_dict"):
                drift_dict = drift_results.to_dict()
            else:
                drift_dict = drift_results
        else:
            drift_dict = {"has_drift": False, "features_drifted": [], "details": {}}

        recent_metrics = recent_metrics or {}

        # Determine status
        status = self._determine_health_status(drift_dict, recent_metrics)

        # Build feature stability from drift details
        feature_stability = self._build_feature_stability(drift_dict)

        # Determine retraining recommendation
        retraining_recommended, reason = self._should_retrain(
            drift_dict, recent_metrics
        )

        return {
            "status": status,
            "generated_at": datetime.now().isoformat(),
            "drift_detected": drift_dict.get("has_drift", False),
            "features_drifted": drift_dict.get("features_drifted", []),
            "feature_stability": feature_stability,
            "recent_performance": recent_metrics,
            "retraining_recommended": retraining_recommended,
            "recommendation_reason": reason,
        }

    def _format_prediction(self, prediction: GamePrediction) -> dict[str, Any]:
        """Format a GamePrediction into a dictionary.

        Args:
            prediction: GamePrediction object.

        Returns:
            Dictionary with formatted prediction data.
        """
        return {
            "game_id": prediction.game_id,
            "game_date": prediction.game_date.isoformat(),
            "matchup": prediction.matchup,
            "home_team": prediction.home_team,
            "away_team": prediction.away_team,
            "home_win_prob": round(prediction.home_win_prob, 4),
            "predicted_margin": round(prediction.predicted_margin, 1),
            "predicted_total": round(prediction.predicted_total, 1),
            "home_win_prob_adjusted": round(prediction.home_win_prob_adjusted, 4),
            "predicted_margin_adjusted": round(prediction.predicted_margin_adjusted, 1),
            "predicted_total_adjusted": round(prediction.predicted_total_adjusted, 1),
            "confidence": round(prediction.confidence, 3),
            "injury_uncertainty": round(prediction.injury_uncertainty, 3),
            "top_factors": prediction.top_factors[:5],
            "model_version": prediction.model_version,
            "home_lineup": prediction.home_lineup,
            "away_lineup": prediction.away_lineup,
        }

    def _format_signal(self, signal: BettingSignal) -> dict[str, Any]:
        """Format a BettingSignal into a dictionary.

        Args:
            signal: BettingSignal object.

        Returns:
            Dictionary with formatted signal data.
        """
        return {
            "game_id": signal.game_id,
            "game_date": signal.game_date.isoformat(),
            "matchup": signal.matchup,
            "bet_type": signal.bet_type,
            "side": signal.side,
            "line": signal.line,
            "model_prob": round(signal.model_prob, 4),
            "market_prob": round(signal.market_prob, 4),
            "edge": round(signal.edge, 4),
            "recommended_odds": round(signal.recommended_odds, 3),
            "kelly_fraction": round(signal.kelly_fraction, 4),
            "recommended_stake_pct": round(signal.recommended_stake_pct, 4),
            "confidence": signal.confidence,
            "key_factors": signal.key_factors[:3],
            "injury_notes": signal.injury_notes[:3],
        }

    def _calculate_summary(
        self,
        predictions: list[GamePrediction],
        signals: list[BettingSignal],
    ) -> dict[str, int | float]:
        """Calculate summary statistics for daily report.

        Args:
            predictions: List of predictions.
            signals: List of signals.

        Returns:
            Dictionary with summary statistics.
        """
        if not predictions:
            return EMPTY_SUMMARY_STATS.copy()

        high_conf = sum(1 for s in signals if s.confidence == "high")
        medium_conf = sum(1 for s in signals if s.confidence == "medium")

        edges = [s.edge for s in signals] if signals else [0.0]

        return {
            "total_games": len(predictions),
            "total_signals": len(signals),
            "high_confidence_signals": high_conf,
            "medium_confidence_signals": medium_conf,
            "avg_edge": round(sum(edges) / len(edges), 4) if edges else 0.0,
            "max_edge": round(max(edges), 4) if edges else 0.0,
        }

    def _format_metrics_report(
        self,
        period: str,
        metrics: FullBacktestMetrics,
    ) -> dict[str, Any]:
        """Format backtest metrics into a performance report.

        Args:
            period: Report period.
            metrics: FullBacktestMetrics object.

        Returns:
            Formatted performance report dictionary.
        """
        metrics_dict = metrics.to_dict()

        # Build calibration curve data points (placeholder bins)
        calibration_curve = self._build_calibration_curve_placeholder()

        return {
            "period": period,
            "generated_at": datetime.now().isoformat(),
            "total_predictions": metrics.total_bets,
            "accuracy": round(metrics.win_rate, 4),
            "roi": round(metrics.roi, 4),
            "clv": round(metrics.avg_clv, 4),
            "sharpe_ratio": round(metrics.sharpe_ratio, 3),
            "max_drawdown": round(metrics.max_drawdown, 4),
            "calibration_curve": calibration_curve,
            "by_bet_type": metrics.metrics_by_type,
            "total_wagered": round(metrics.total_wagered, 2),
            "total_profit": round(metrics.total_profit, 2),
            "win_count": metrics.win_count,
            "loss_count": metrics.loss_count,
            "brier_score": round(metrics.brier_score, 4),
        }

    def _calculate_performance_from_bets(
        self,
        period: str,
        bets: list[Bet],
    ) -> dict[str, Any]:
        """Calculate performance metrics from bet history.

        Args:
            period: Report period.
            bets: List of Bet objects.

        Returns:
            Performance report dictionary.
        """
        if not bets:
            return self._empty_performance_report(period)

        total_wagered = sum(b.bet_amount for b in bets)
        profits = [b.profit or 0.0 for b in bets]
        total_profit = sum(profits)

        wins = sum(1 for b in bets if b.result == "win")
        losses = sum(1 for b in bets if b.result == "loss")
        pushes = sum(1 for b in bets if b.result == "push")

        win_rate = wins / len(bets) if bets else 0.0
        roi = total_profit / total_wagered if total_wagered > 0 else 0.0

        edges = [b.edge for b in bets]
        avg_edge = sum(edges) / len(edges) if edges else 0.0

        return {
            "period": period,
            "generated_at": datetime.now().isoformat(),
            "total_predictions": len(bets),
            "accuracy": round(win_rate, 4),
            "roi": round(roi, 4),
            "clv": round(avg_edge, 4),
            "sharpe_ratio": 0.0,  # Would need daily returns to calculate
            "max_drawdown": 0.0,  # Would need bankroll history
            "calibration_curve": self._build_calibration_curve_placeholder(),
            "by_bet_type": self._segment_by_bet_type(bets),
            "total_wagered": round(total_wagered, 2),
            "total_profit": round(total_profit, 2),
            "win_count": wins,
            "loss_count": losses,
            "push_count": pushes,
            "brier_score": 0.0,
        }

    def _segment_by_bet_type(
        self, bets: list[Bet]
    ) -> dict[str, dict[str, float]]:
        """Segment bet statistics by bet type.

        Args:
            bets: List of bets.

        Returns:
            Dictionary with metrics per bet type.
        """
        by_type: dict[str, list[Bet]] = {}
        for bet in bets:
            if bet.bet_type not in by_type:
                by_type[bet.bet_type] = []
            by_type[bet.bet_type].append(bet)

        result = {}
        for bet_type, type_bets in by_type.items():
            wins = sum(1 for b in type_bets if b.result == "win")
            total_wagered = sum(b.bet_amount for b in type_bets)
            total_profit = sum(b.profit or 0.0 for b in type_bets)

            result[bet_type] = {
                "count": len(type_bets),
                "win_rate": wins / len(type_bets) if type_bets else 0.0,
                "roi": total_profit / total_wagered if total_wagered > 0 else 0.0,
            }

        return result

    def _empty_performance_report(self, period: str) -> dict[str, Any]:
        """Generate empty performance report structure.

        Args:
            period: Report period.

        Returns:
            Empty performance report dictionary.
        """
        return {
            "period": period,
            "generated_at": datetime.now().isoformat(),
            "total_predictions": 0,
            "accuracy": 0.0,
            "roi": 0.0,
            "clv": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "calibration_curve": [],
            "by_bet_type": {},
            "total_wagered": 0.0,
            "total_profit": 0.0,
            "win_count": 0,
            "loss_count": 0,
            "brier_score": 0.0,
        }

    def _build_calibration_curve_placeholder(self) -> list[dict[str, float]]:
        """Build placeholder calibration curve data points.

        Returns:
            List of calibration curve data points.
        """
        # Placeholder bins from 0.1 to 0.9
        bins = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        return [
            {"predicted": b, "actual": b, "count": 0}
            for b in bins
        ]

    def _determine_health_status(
        self,
        drift_dict: dict[str, Any],
        recent_metrics: dict[str, float],
    ) -> str:
        """Determine overall model health status.

        Args:
            drift_dict: Drift detection results.
            recent_metrics: Recent performance metrics.

        Returns:
            Status string: "healthy", "warning", or "critical".
        """
        has_drift = drift_dict.get("has_drift", False)
        features_drifted = drift_dict.get("features_drifted", [])

        # Check performance degradation
        accuracy = recent_metrics.get("accuracy", 0.55)
        brier_score = recent_metrics.get("brier_score", 0.22)

        if has_drift and len(features_drifted) >= 3:
            return "critical"

        if accuracy < 0.48 or brier_score > 0.30:
            return "critical"

        if has_drift or accuracy < 0.52 or brier_score > 0.26:
            return "warning"

        return "healthy"

    def _build_feature_stability(
        self, drift_dict: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        """Build feature stability metrics from drift details.

        Args:
            drift_dict: Drift detection results.

        Returns:
            Dictionary with per-feature stability info.
        """
        details = drift_dict.get("details", {})
        features_drifted = set(drift_dict.get("features_drifted", []))

        stability = {}
        for feature, detail in details.items():
            if isinstance(detail, dict):
                stability[feature] = {
                    "ks_stat": detail.get("ks_stat", 0.0),
                    "p_value": detail.get("p_value", 1.0),
                    "psi": detail.get("psi", 0.0),
                    "status": "drifted" if feature in features_drifted else "stable",
                }

        return stability

    def _should_retrain(
        self,
        drift_dict: dict[str, Any],
        recent_metrics: dict[str, float],
    ) -> tuple[bool, str]:
        """Determine if retraining is recommended.

        Args:
            drift_dict: Drift detection results.
            recent_metrics: Recent performance metrics.

        Returns:
            Tuple of (should_retrain, reason).
        """
        reasons = []

        # Check drift
        if drift_dict.get("has_drift", False):
            features = drift_dict.get("features_drifted", [])
            if len(features) >= 2:
                reasons.append(f"Significant covariate drift in {len(features)} features")
            elif features:
                reasons.append(f"Covariate drift detected in: {', '.join(features)}")

        # Check performance
        accuracy = recent_metrics.get("accuracy", 0.55)
        if accuracy < 0.48:
            reasons.append(f"Accuracy degraded to {accuracy:.1%}")

        brier_score = recent_metrics.get("brier_score", 0.22)
        if brier_score > 0.28:
            reasons.append(f"Calibration degraded (Brier: {brier_score:.3f})")

        roi = recent_metrics.get("roi", 0.0)
        if roi < -0.05:
            reasons.append(f"Negative ROI: {roi:.1%}")

        if reasons:
            return True, "; ".join(reasons)

        return False, "Model performance within acceptable thresholds"
