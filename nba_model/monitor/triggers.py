"""Retraining trigger evaluation for automated model refresh.

This module implements multiple trigger conditions that determine when
model retraining should occur. Triggers include scheduled intervals,
drift detection, performance degradation, and data availability.

Trigger Types:
    - Scheduled: Time-based periodic retraining
    - Drift: Statistical distribution shift detected
    - Performance: Model predictions degrading
    - Data: Sufficient new training data available

Example:
    >>> trigger = RetrainingTrigger(scheduled_interval_days=7)
    >>> result = trigger.evaluate_all_triggers(context)
    >>> if result.should_retrain:
    ...     print(f"Retraining recommended: {result.reason}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any

import pandas as pd

from nba_model.logging import get_logger
from nba_model.monitor.drift import DriftDetector

if TYPE_CHECKING:
    from nba_model.types import Bet

logger = get_logger(__name__)

# =============================================================================
# Constants
# =============================================================================

# Default configuration
DEFAULT_SCHEDULED_INTERVAL_DAYS: int = 7
DEFAULT_MIN_NEW_GAMES: int = 50
DEFAULT_ROI_THRESHOLD: float = -0.05  # -5% ROI triggers retrain
DEFAULT_ACCURACY_THRESHOLD: float = 0.48
MIN_BETS_FOR_PERFORMANCE: int = 50

# Priority levels
PRIORITY_HIGH: str = "high"
PRIORITY_MEDIUM: str = "medium"
PRIORITY_LOW: str = "low"

# Trigger reason strings
REASON_DRIFT: str = "drift_detected"
REASON_PERFORMANCE: str = "performance_degraded"
REASON_SCHEDULED: str = "scheduled"
REASON_DATA: str = "new_data"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class TriggerResult:
    """Result of retraining trigger evaluation.

    Attributes:
        should_retrain: Whether retraining is recommended.
        reason: Primary reason for retraining recommendation.
        priority: Urgency level ('high', 'medium', 'low').
        trigger_details: Per-trigger activation status.
    """

    should_retrain: bool
    reason: str
    priority: str
    trigger_details: dict[str, bool]

    def to_dict(self) -> dict[str, bool | str | dict[str, bool]]:
        """Convert to dictionary format."""
        return {
            "should_retrain": self.should_retrain,
            "reason": self.reason,
            "priority": self.priority,
            "trigger_details": self.trigger_details,
        }


@dataclass
class TriggerContext:
    """Context data required for trigger evaluation.

    Attributes:
        last_train_date: Date of last model training.
        drift_detector: Optional DriftDetector with reference data.
        recent_data: Optional DataFrame with recent feature data.
        recent_bets: Optional list of recent Bet objects.
        games_since_training: Number of games since last training.
    """

    last_train_date: date
    drift_detector: DriftDetector | None = None
    recent_data: pd.DataFrame | None = None
    recent_bets: list[Bet] = field(default_factory=list)
    games_since_training: int = 0


# =============================================================================
# Retraining Trigger Class
# =============================================================================


class RetrainingTrigger:
    """Evaluates multiple conditions to determine if model retraining is needed.

    Supports four trigger types:
        1. Scheduled: Periodic retraining based on time interval
        2. Drift: Triggered by covariate drift detection
        3. Performance: Triggered by degraded prediction performance
        4. Data: Triggered by sufficient new training data

    Priority assignment:
        - High: Drift or performance triggers activated
        - Medium: Scheduled trigger activated
        - Low: Only data trigger activated

    Attributes:
        scheduled_interval_days: Days between scheduled retrains.
        min_new_games: Minimum games to justify data-based retrain.
        roi_threshold: ROI below this triggers performance retrain.
        accuracy_threshold: Accuracy below this triggers retrain.

    Example:
        >>> trigger = RetrainingTrigger(
        ...     scheduled_interval_days=7,
        ...     min_new_games=50,
        ...     roi_threshold=-0.05,
        ... )
        >>> context = TriggerContext(
        ...     last_train_date=date(2024, 1, 1),
        ...     recent_bets=bets,
        ...     games_since_training=100,
        ... )
        >>> result = trigger.evaluate_all_triggers(context)
    """

    def __init__(
        self,
        scheduled_interval_days: int = DEFAULT_SCHEDULED_INTERVAL_DAYS,
        min_new_games: int = DEFAULT_MIN_NEW_GAMES,
        roi_threshold: float = DEFAULT_ROI_THRESHOLD,
        accuracy_threshold: float = DEFAULT_ACCURACY_THRESHOLD,
    ) -> None:
        """Initialize RetrainingTrigger with configuration.

        Args:
            scheduled_interval_days: Days between scheduled retrains.
            min_new_games: Minimum games to justify data-based retrain.
            roi_threshold: ROI below this triggers performance retrain.
            accuracy_threshold: Win rate below this triggers retrain.

        Raises:
            ValueError: If parameters are out of valid ranges.
        """
        if scheduled_interval_days < 1:
            raise ValueError(
                f"scheduled_interval_days must be >= 1, got {scheduled_interval_days}"
            )
        if min_new_games < 1:
            raise ValueError(f"min_new_games must be >= 1, got {min_new_games}")
        if not 0 < accuracy_threshold < 1:
            raise ValueError(
                f"accuracy_threshold must be in (0, 1), got {accuracy_threshold}"
            )

        self.scheduled_interval_days = scheduled_interval_days
        self.min_new_games = min_new_games
        self.roi_threshold = roi_threshold
        self.accuracy_threshold = accuracy_threshold

        logger.debug(
            "Initialized RetrainingTrigger: interval={}d, min_games={}, "
            "roi_threshold={:.2%}, accuracy_threshold={:.2%}",
            scheduled_interval_days,
            min_new_games,
            roi_threshold,
            accuracy_threshold,
        )

    def check_scheduled_trigger(self, last_train_date: date) -> bool:
        """Check if scheduled retraining is due.

        Args:
            last_train_date: Date of the last model training.

        Returns:
            True if current_date - last_train_date >= scheduled_interval_days.
        """
        days_since = (date.today() - last_train_date).days

        triggered = days_since >= self.scheduled_interval_days

        logger.debug(
            "Scheduled trigger: {} days since training, threshold={}d, triggered={}",
            days_since,
            self.scheduled_interval_days,
            triggered,
        )

        return triggered

    def check_drift_trigger(
        self,
        drift_detector: DriftDetector,
        recent_data: pd.DataFrame,
    ) -> bool:
        """Check if drift warrants retraining.

        Args:
            drift_detector: Configured DriftDetector with reference data.
            recent_data: Recent feature data to check for drift.

        Returns:
            True if drift is detected in any monitored feature.
        """
        try:
            result = drift_detector.check_drift(recent_data)
            triggered = result.has_drift

            if triggered:
                logger.info(
                    "Drift trigger activated: drifted features = {}",
                    result.features_drifted,
                )

            return triggered

        except Exception as e:
            logger.warning("Drift check failed: {}", e)
            return False

    def check_performance_trigger(self, recent_bets: list[Bet]) -> bool:
        """Check if recent performance warrants retraining.

        Evaluates ROI and win rate from recent bets to detect
        performance degradation.

        Args:
            recent_bets: List of recent Bet objects with results.

        Returns:
            True if ROI < roi_threshold OR win_rate < accuracy_threshold.
        """
        if len(recent_bets) < MIN_BETS_FOR_PERFORMANCE:
            logger.debug(
                "Insufficient bets for performance check: {} < {}",
                len(recent_bets),
                MIN_BETS_FOR_PERFORMANCE,
            )
            return False

        # Calculate ROI
        total_wagered = sum(bet.bet_amount for bet in recent_bets)
        if total_wagered <= 0:
            return False

        total_profit = sum(bet.profit or 0.0 for bet in recent_bets)
        roi = total_profit / total_wagered

        # Calculate win rate
        decided_bets = [b for b in recent_bets if b.result in ("win", "loss")]
        if not decided_bets:
            return False

        wins = sum(1 for b in decided_bets if b.result == "win")
        win_rate = wins / len(decided_bets)

        # Check thresholds
        roi_failed = roi < self.roi_threshold
        accuracy_failed = win_rate < self.accuracy_threshold

        triggered = roi_failed or accuracy_failed

        if triggered:
            logger.info(
                "Performance trigger activated: ROI={:.2%} (threshold={:.2%}), "
                "win_rate={:.2%} (threshold={:.2%})",
                roi,
                self.roi_threshold,
                win_rate,
                self.accuracy_threshold,
            )

        return triggered

    def check_data_trigger(self, games_since_training: int) -> bool:
        """Check if sufficient new data justifies retraining.

        Args:
            games_since_training: Number of new games since last training.

        Returns:
            True if games_since_training >= min_new_games.
        """
        triggered = games_since_training >= self.min_new_games

        logger.debug(
            "Data trigger: {} games since training, threshold={}, triggered={}",
            games_since_training,
            self.min_new_games,
            triggered,
        )

        return triggered

    def evaluate_all_triggers(
        self,
        context: TriggerContext | dict[str, Any],
    ) -> TriggerResult:
        """Evaluate all trigger conditions and determine if retraining is needed.

        Checks all four trigger types and returns a comprehensive result
        with the primary reason and priority level.

        Priority assignment:
            - High: Drift or performance triggers activated
            - Medium: Scheduled trigger activated
            - Low: Only data trigger activated

        Args:
            context: TriggerContext object or dict with evaluation context.
                Required keys: last_train_date
                Optional keys: drift_detector, recent_data, recent_bets,
                              games_since_training

        Returns:
            TriggerResult with retraining recommendation.
        """
        # Convert dict to TriggerContext if needed
        if isinstance(context, dict):
            context = TriggerContext(
                last_train_date=context["last_train_date"],
                drift_detector=context.get("drift_detector"),
                recent_data=context.get("recent_data"),
                recent_bets=context.get("recent_bets", []),
                games_since_training=context.get("games_since_training", 0),
            )

        trigger_details: dict[str, bool] = {
            "scheduled": False,
            "drift": False,
            "performance": False,
            "data": False,
        }

        # Check scheduled trigger
        trigger_details["scheduled"] = self.check_scheduled_trigger(
            context.last_train_date
        )

        # Check drift trigger (if detector and data provided)
        if context.drift_detector is not None and context.recent_data is not None:
            trigger_details["drift"] = self.check_drift_trigger(
                context.drift_detector,
                context.recent_data,
            )

        # Check performance trigger (if bets provided)
        if context.recent_bets:
            trigger_details["performance"] = self.check_performance_trigger(
                context.recent_bets
            )

        # Check data trigger
        trigger_details["data"] = self.check_data_trigger(
            context.games_since_training
        )

        # Determine if retraining is needed
        should_retrain = any(trigger_details.values())

        # Determine reason and priority
        reason = ""
        priority = PRIORITY_LOW

        if trigger_details["drift"]:
            reason = REASON_DRIFT
            priority = PRIORITY_HIGH
        elif trigger_details["performance"]:
            reason = REASON_PERFORMANCE
            priority = PRIORITY_HIGH
        elif trigger_details["scheduled"]:
            reason = REASON_SCHEDULED
            priority = PRIORITY_MEDIUM
        elif trigger_details["data"]:
            reason = REASON_DATA
            priority = PRIORITY_LOW

        if should_retrain:
            logger.info(
                "Retraining recommended: reason={}, priority={}, triggers={}",
                reason,
                priority,
                trigger_details,
            )
        else:
            logger.debug("No retraining needed: triggers={}", trigger_details)

        return TriggerResult(
            should_retrain=should_retrain,
            reason=reason,
            priority=priority,
            trigger_details=trigger_details,
        )
