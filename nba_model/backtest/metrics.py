"""Performance metrics for backtesting results.

This module provides comprehensive performance metrics calculation for
backtest results including returns, risk metrics, betting statistics,
calibration measures, and closing line value (CLV) analysis.

Example:
    >>> metrics_calc = BacktestMetricsCalculator()
    >>> metrics = metrics_calc.calculate_all(backtest_result)
    >>> print(f"ROI: {metrics.roi:.2%}, Sharpe: {metrics.sharpe_ratio:.2f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from nba_model.backtest.engine import BacktestResult
    from nba_model.types import Bet

# =============================================================================
# Constants
# =============================================================================

TRADING_DAYS_PER_YEAR: int = 252  # For annualization
MIN_BETS_FOR_STATS: int = 10


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class FullBacktestMetrics:
    """Comprehensive backtest performance metrics.

    Attributes:
        total_return: Total percentage return over period.
        cagr: Compound annual growth rate.
        avg_bet_return: Average return per bet.
        volatility: Standard deviation of returns.
        sharpe_ratio: Risk-adjusted return.
        sortino_ratio: Downside risk-adjusted return.
        max_drawdown: Largest peak-to-trough decline.
        max_drawdown_duration: Days in maximum drawdown.
        total_bets: Count of all bets placed.
        win_rate: Percentage of winning bets.
        avg_edge: Average edge across all bets.
        avg_odds: Average decimal odds of bets.
        roi: Return on investment (profit / total wagered).
        brier_score: Calibration metric for probabilities.
        log_loss: Cross-entropy loss of predictions.
        avg_clv: Average closing line value.
        clv_positive_rate: Percentage of bets with positive CLV.
        metrics_by_type: Metrics segmented by bet type.
        total_wagered: Total amount wagered.
        total_profit: Total profit/loss.
        win_count: Number of winning bets.
        loss_count: Number of losing bets.
        push_count: Number of pushed bets.
    """

    # Returns
    total_return: float = 0.0
    cagr: float = 0.0
    avg_bet_return: float = 0.0

    # Risk
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0

    # Betting
    total_bets: int = 0
    win_rate: float = 0.0
    avg_edge: float = 0.0
    avg_odds: float = 0.0
    roi: float = 0.0

    # Calibration
    brier_score: float = 0.0
    log_loss: float = 0.0

    # CLV
    avg_clv: float = 0.0
    clv_positive_rate: float = 0.0

    # Segmentation
    metrics_by_type: dict[str, dict[str, float]] = field(default_factory=dict)

    # Totals
    total_wagered: float = 0.0
    total_profit: float = 0.0
    win_count: int = 0
    loss_count: int = 0
    push_count: int = 0

    def to_dict(self) -> dict[str, float | int | dict]:
        """Convert metrics to dictionary format.

        Returns:
            Dictionary with all metric values.
        """
        return {
            "total_return": self.total_return,
            "cagr": self.cagr,
            "avg_bet_return": self.avg_bet_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration": self.max_drawdown_duration,
            "total_bets": self.total_bets,
            "win_rate": self.win_rate,
            "avg_edge": self.avg_edge,
            "avg_odds": self.avg_odds,
            "roi": self.roi,
            "brier_score": self.brier_score,
            "log_loss": self.log_loss,
            "avg_clv": self.avg_clv,
            "clv_positive_rate": self.clv_positive_rate,
            "metrics_by_type": self.metrics_by_type,
            "total_wagered": self.total_wagered,
            "total_profit": self.total_profit,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "push_count": self.push_count,
        }


@dataclass(frozen=True)
class CLVResult:
    """Closing Line Value calculation result.

    Attributes:
        clv: Closing line value as decimal.
        clv_pct: CLV as percentage.
        bet_implied_prob: Implied probability at bet time.
        closing_implied_prob: Implied probability at close.
    """

    clv: float
    clv_pct: float
    bet_implied_prob: float
    closing_implied_prob: float


# =============================================================================
# Main Calculator Class
# =============================================================================


class BacktestMetricsCalculator:
    """Calculator for comprehensive backtest performance metrics.

    Computes all relevant metrics for evaluating a betting strategy
    including returns, risk measures, betting statistics, and calibration.

    Example:
        >>> calc = BacktestMetricsCalculator()
        >>> metrics = calc.calculate_all(result)
        >>> report = calc.generate_report(result)
    """

    def calculate_from_bets(
        self,
        bets: list[Bet],
        bankroll_history: list[float] | None = None,
        initial_bankroll: float = 10000.0,
        closing_odds: dict[str, float] | None = None,
        closing_odds_map: dict[tuple[str, str, str], float] | None = None,
    ) -> FullBacktestMetrics:
        """Calculate all performance metrics from bet list.

        Args:
            bets: List of Bet objects with outcomes.
            bankroll_history: Optional list of bankroll values over time.
            initial_bankroll: Initial bankroll for calculations.
            closing_odds: Optional dict mapping game_id to closing decimal odds.
                         Deprecated: use closing_odds_map for proper CLV by bet type.
            closing_odds_map: Optional dict mapping (game_id, bet_type, side) to
                             closing decimal odds. Supports all bet types.

        Returns:
            FullBacktestMetrics with all computed metrics.
        """
        if not bets:
            return FullBacktestMetrics()

        metrics = FullBacktestMetrics()
        metrics.total_bets = len(bets)

        # Calculate basic betting stats
        self._calculate_betting_stats(bets, metrics)

        # Calculate returns
        self._calculate_returns(bets, metrics, initial_bankroll, bankroll_history)

        # Calculate risk metrics
        self._calculate_risk_metrics(bets, metrics, bankroll_history)

        # Calculate calibration
        self._calculate_calibration(bets, metrics)

        # Calculate CLV - prefer closing_odds_map if available
        if closing_odds_map:
            self._calculate_clv_metrics_by_type(bets, closing_odds_map, metrics)
        elif closing_odds:
            # Legacy support: convert to map format (assumes home ML)
            legacy_map: dict[tuple[str, str, str], float] = {}
            for game_id, odds in closing_odds.items():
                legacy_map[(game_id, "moneyline", "home")] = odds
            self._calculate_clv_metrics_by_type(bets, legacy_map, metrics)

        # Calculate metrics by bet type
        metrics.metrics_by_type = self._calculate_metrics_by_type(bets)

        return metrics

    def _calculate_betting_stats(
        self,
        bets: list[Bet],
        metrics: FullBacktestMetrics,
    ) -> None:
        """Calculate basic betting statistics."""
        wins = [b for b in bets if b.result == "win"]
        losses = [b for b in bets if b.result == "loss"]
        pushes = [b for b in bets if b.result == "push"]

        metrics.win_count = len(wins)
        metrics.loss_count = len(losses)
        metrics.push_count = len(pushes)

        decided_bets = len(wins) + len(losses)
        metrics.win_rate = len(wins) / decided_bets if decided_bets > 0 else 0.0

        # Calculate totals
        metrics.total_wagered = sum(b.bet_amount for b in bets)
        metrics.total_profit = sum(b.profit or 0.0 for b in bets)

        # Calculate averages
        metrics.avg_edge = np.mean([b.edge for b in bets]) if bets else 0.0
        metrics.avg_odds = np.mean([b.market_odds for b in bets]) if bets else 0.0

        # ROI
        metrics.roi = (
            metrics.total_profit / metrics.total_wagered
            if metrics.total_wagered > 0
            else 0.0
        )

    def _calculate_returns(
        self,
        bets: list[Bet],
        metrics: FullBacktestMetrics,
        initial_bankroll: float,
        bankroll_history: list[float] | None,
    ) -> None:
        """Calculate return metrics."""
        # Total return
        final_bankroll = initial_bankroll + metrics.total_profit
        metrics.total_return = (final_bankroll - initial_bankroll) / initial_bankroll

        # Average bet return
        if metrics.total_bets > 0:
            bet_returns = [
                (b.profit or 0.0) / b.bet_amount if b.bet_amount > 0 else 0.0
                for b in bets
            ]
            metrics.avg_bet_return = np.mean(bet_returns)

        # CAGR - need to know time period
        if bets:
            start_date = min(b.timestamp for b in bets).date()
            end_date = max(b.timestamp for b in bets).date()
            days = (end_date - start_date).days
            years = days / 365.25 if days > 0 else 1.0

            if final_bankroll > 0 and initial_bankroll > 0:
                metrics.cagr = (final_bankroll / initial_bankroll) ** (1 / years) - 1

    def _calculate_risk_metrics(
        self,
        bets: list[Bet],
        metrics: FullBacktestMetrics,
        bankroll_history: list[float] | None,
    ) -> None:
        """Calculate risk metrics."""
        # Calculate per-bet returns
        bet_returns = [
            (b.profit or 0.0) / b.bet_amount if b.bet_amount > 0 else 0.0 for b in bets
        ]

        if len(bet_returns) >= MIN_BETS_FOR_STATS:
            returns_array = np.array(bet_returns)

            # Volatility (std dev of returns)
            metrics.volatility = float(np.std(returns_array))

            # Sharpe ratio (simplified - using 0 as risk-free rate)
            mean_return = np.mean(returns_array)
            if metrics.volatility > 0:
                # Annualized Sharpe (assuming ~1000 bets/year)
                metrics.sharpe_ratio = (
                    mean_return
                    / metrics.volatility
                    * np.sqrt(min(len(bets), TRADING_DAYS_PER_YEAR))
                )

            # Sortino ratio (downside volatility)
            negative_returns = returns_array[returns_array < 0]
            if len(negative_returns) > 0:
                downside_std = float(np.std(negative_returns))
                if downside_std > 0:
                    metrics.sortino_ratio = (
                        mean_return
                        / downside_std
                        * np.sqrt(min(len(bets), TRADING_DAYS_PER_YEAR))
                    )

        # Max drawdown from bankroll history
        if bankroll_history and len(bankroll_history) > 1:
            metrics.max_drawdown, metrics.max_drawdown_duration = (
                self._calculate_drawdown_metrics(bankroll_history)
            )

    def _calculate_drawdown_metrics(
        self,
        bankroll_history: list[float],
    ) -> tuple[float, int]:
        """Calculate max drawdown and duration.

        Args:
            bankroll_history: List of bankroll values.

        Returns:
            Tuple of (max_drawdown, max_drawdown_duration_days).
        """
        if not bankroll_history or len(bankroll_history) < 2:
            return 0.0, 0

        peak = bankroll_history[0]
        max_drawdown = 0.0
        max_drawdown_duration = 0
        current_drawdown_start = 0

        for i, value in enumerate(bankroll_history):
            if value >= peak:
                # New peak - calculate duration of previous drawdown
                if peak > 0 and max_drawdown > 0:
                    duration = i - current_drawdown_start
                    max_drawdown_duration = max(max_drawdown_duration, duration)
                peak = value
                current_drawdown_start = i
            else:
                # In drawdown
                drawdown = (peak - value) / peak if peak > 0 else 0.0
                max_drawdown = max(max_drawdown, drawdown)

        # Check if still in drawdown at end
        if bankroll_history[-1] < peak:
            duration = len(bankroll_history) - current_drawdown_start
            max_drawdown_duration = max(max_drawdown_duration, duration)

        return max_drawdown, max_drawdown_duration

    def _calculate_calibration(
        self,
        bets: list[Bet],
        metrics: FullBacktestMetrics,
    ) -> None:
        """Calculate probability calibration metrics."""
        # Only use bets with decided outcomes
        decided_bets = [b for b in bets if b.result in ("win", "loss")]

        if not decided_bets:
            return

        probs = np.array([b.model_prob for b in decided_bets])
        outcomes = np.array([1.0 if b.result == "win" else 0.0 for b in decided_bets])

        # Brier score: mean((prob - outcome)^2)
        metrics.brier_score = float(np.mean((probs - outcomes) ** 2))

        # Log loss: -mean(outcome * log(prob) + (1-outcome) * log(1-prob))
        # Clip probabilities to avoid log(0)
        eps = 1e-15
        probs_clipped = np.clip(probs, eps, 1 - eps)
        log_loss = -np.mean(
            outcomes * np.log(probs_clipped)
            + (1 - outcomes) * np.log(1 - probs_clipped)
        )
        metrics.log_loss = float(log_loss)

    def _calculate_clv_metrics(
        self,
        bets: list[Bet],
        closing_odds: dict[str, float],
        metrics: FullBacktestMetrics,
    ) -> None:
        """Calculate closing line value metrics (legacy version)."""
        clv_values = []

        for bet in bets:
            if bet.game_id in closing_odds:
                closing_odd = closing_odds[bet.game_id]
                result = self.calculate_clv(bet, closing_odd)
                if result is not None:
                    clv_values.append(result.clv)

        if clv_values:
            metrics.avg_clv = float(np.mean(clv_values))
            metrics.clv_positive_rate = sum(1 for c in clv_values if c > 0) / len(
                clv_values
            )

    def _calculate_clv_metrics_by_type(
        self,
        bets: list[Bet],
        closing_odds_map: dict[tuple[str, str, str], float],
        metrics: FullBacktestMetrics,
    ) -> None:
        """Calculate closing line value metrics for all bet types.

        Args:
            bets: List of bet objects.
            closing_odds_map: Dict mapping (game_id, bet_type, side) to closing odds.
            metrics: Metrics object to update.
        """
        clv_values = []

        for bet in bets:
            key = (bet.game_id, bet.bet_type, bet.side)
            if key in closing_odds_map:
                closing_odd = closing_odds_map[key]
                result = self.calculate_clv(bet, closing_odd)
                if result is not None:
                    clv_values.append(result.clv)

        if clv_values:
            metrics.avg_clv = float(np.mean(clv_values))
            metrics.clv_positive_rate = sum(1 for c in clv_values if c > 0) / len(
                clv_values
            )

    def calculate_clv(
        self,
        bet: Bet,
        closing_odds: float,
    ) -> CLVResult | None:
        """Calculate closing line value for a single bet.

        CLV = (closing_implied_prob - bet_implied_prob) / bet_implied_prob

        Positive CLV means you beat the closing line.

        Args:
            bet: The bet placed.
            closing_odds: Closing decimal odds for the same market.

        Returns:
            CLVResult with CLV details, or None if calculation not possible.
        """
        if closing_odds <= 1 or bet.market_odds <= 1:
            return None

        bet_implied = 1.0 / bet.market_odds
        closing_implied = 1.0 / closing_odds

        # CLV is positive if closing line moved in your favor
        # (i.e., closing implied prob is higher than when you bet)
        clv = (closing_implied - bet_implied) / bet_implied

        return CLVResult(
            clv=clv,
            clv_pct=clv * 100,
            bet_implied_prob=bet_implied,
            closing_implied_prob=closing_implied,
        )

    def _calculate_metrics_by_type(
        self,
        bets: list[Bet],
    ) -> dict[str, dict[str, float]]:
        """Calculate metrics segmented by bet type."""
        bet_types = {b.bet_type for b in bets}
        metrics_by_type: dict[str, dict[str, float]] = {}

        for bet_type in bet_types:
            type_bets = [b for b in bets if b.bet_type == bet_type]
            if not type_bets:
                continue

            wins = [b for b in type_bets if b.result == "win"]
            losses = [b for b in type_bets if b.result == "loss"]
            decided = len(wins) + len(losses)

            total_wagered = sum(b.bet_amount for b in type_bets)
            total_profit = sum(b.profit or 0.0 for b in type_bets)

            metrics_by_type[bet_type] = {
                "total_bets": len(type_bets),
                "win_rate": len(wins) / decided if decided > 0 else 0.0,
                "roi": total_profit / total_wagered if total_wagered > 0 else 0.0,
                "avg_edge": float(np.mean([b.edge for b in type_bets])),
                "avg_odds": float(np.mean([b.market_odds for b in type_bets])),
                "total_wagered": total_wagered,
                "total_profit": total_profit,
            }

        return metrics_by_type

    def calculate_all(
        self,
        result: BacktestResult,
        closing_odds_map: dict[tuple[str, str, str], float] | None = None,
    ) -> dict[str, float | int | dict]:
        """Calculate all metrics from a BacktestResult object.

        This is the spec-compliant API (Phase 5 requirement) that takes a
        BacktestResult directly and returns a dictionary of metrics.

        Args:
            result: BacktestResult object from walk-forward backtest.
            closing_odds_map: Optional closing odds for CLV calculation.

        Returns:
            Dictionary containing all computed metrics.
        """
        initial_bankroll = (
            result.config.initial_bankroll if result.config else 10000.0
        )

        metrics = self.calculate_from_bets(
            bets=result.bets,
            bankroll_history=result.bankroll_history,
            initial_bankroll=initial_bankroll,
            closing_odds_map=closing_odds_map,
        )

        return metrics.to_dict()

    def generate_report(
        self,
        metrics: FullBacktestMetrics,
        title: str = "Backtest Report",
    ) -> str:
        """Generate human-readable backtest report.

        Args:
            metrics: Computed backtest metrics.
            title: Report title.

        Returns:
            Formatted text report.
        """
        lines = [
            f"{'=' * 60}",
            f"{title:^60}",
            f"{'=' * 60}",
            "",
            "RETURNS",
            "-" * 40,
            f"Total Return:     {metrics.total_return:>12.2%}",
            f"CAGR:             {metrics.cagr:>12.2%}",
            f"Avg Bet Return:   {metrics.avg_bet_return:>12.2%}",
            "",
            "RISK METRICS",
            "-" * 40,
            f"Volatility:       {metrics.volatility:>12.4f}",
            f"Sharpe Ratio:     {metrics.sharpe_ratio:>12.2f}",
            f"Sortino Ratio:    {metrics.sortino_ratio:>12.2f}",
            f"Max Drawdown:     {metrics.max_drawdown:>12.2%}",
            f"DD Duration:      {metrics.max_drawdown_duration:>12} days",
            "",
            "BETTING STATS",
            "-" * 40,
            f"Total Bets:       {metrics.total_bets:>12}",
            f"Wins:             {metrics.win_count:>12}",
            f"Losses:           {metrics.loss_count:>12}",
            f"Pushes:           {metrics.push_count:>12}",
            f"Win Rate:         {metrics.win_rate:>12.2%}",
            f"ROI:              {metrics.roi:>12.2%}",
            f"Avg Edge:         {metrics.avg_edge:>12.4f}",
            f"Avg Odds:         {metrics.avg_odds:>12.3f}",
            "",
            "CALIBRATION",
            "-" * 40,
            f"Brier Score:      {metrics.brier_score:>12.4f}",
            f"Log Loss:         {metrics.log_loss:>12.4f}",
            "",
            "CLOSING LINE VALUE",
            "-" * 40,
            f"Avg CLV:          {metrics.avg_clv:>12.4f}",
            f"CLV+ Rate:        {metrics.clv_positive_rate:>12.2%}",
            "",
            "TOTALS",
            "-" * 40,
            f"Total Wagered:    ${metrics.total_wagered:>11,.2f}",
            f"Total Profit:     ${metrics.total_profit:>11,.2f}",
        ]

        # Add bet type breakdown
        if metrics.metrics_by_type:
            lines.extend(["", "BY BET TYPE", "-" * 40])
            for bet_type, type_metrics in metrics.metrics_by_type.items():
                lines.extend(
                    [
                        f"\n{bet_type.upper()}:",
                        f"  Bets:     {type_metrics['total_bets']:>8.0f}",
                        f"  Win Rate: {type_metrics['win_rate']:>8.2%}",
                        f"  ROI:      {type_metrics['roi']:>8.2%}",
                        f"  Avg Edge: {type_metrics['avg_edge']:>8.4f}",
                    ]
                )

        lines.extend(["", "=" * 60])

        return "\n".join(lines)


def calculate_calibration_curve(
    bets: list[Bet],
    n_bins: int = 10,
) -> tuple[list[float], list[float], list[int]]:
    """Calculate probability calibration curve data.

    Groups bets by predicted probability and calculates actual win rate
    in each bin.

    Args:
        bets: List of decided bets.
        n_bins: Number of probability bins.

    Returns:
        Tuple of (bin_centers, actual_rates, bin_counts).
    """
    decided = [b for b in bets if b.result in ("win", "loss")]
    if not decided:
        return [], [], []

    probs = np.array([b.model_prob for b in decided])
    outcomes = np.array([1.0 if b.result == "win" else 0.0 for b in decided])

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    actual_rates = []
    bin_counts = []

    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            actual_rates.append(float(outcomes[mask].mean()))
            bin_counts.append(int(mask.sum()))

    return bin_centers, actual_rates, bin_counts
