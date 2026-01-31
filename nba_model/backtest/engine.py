"""Walk-forward validation engine for backtesting betting strategies.

This module implements walk-forward validation that respects temporal ordering
to prevent look-ahead bias. Unlike k-fold cross-validation, walk-forward
maintains strict chronological ordering of training and validation data.

Fold Generation:
    Fold 1: Train [Season 1-2], Validate [Season 3 first half]
    Fold 2: Train [Season 1-3 first half], Validate [Season 3 second half]
    ...

Example:
    >>> engine = WalkForwardEngine(min_train_games=500)
    >>> folds = engine.generate_folds(games_df)
    >>> result = engine.run_backtest(trainer, kelly_calc, devig_method='power')
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING, Protocol

import numpy as np
import pandas as pd

from nba_model.backtest.devig import DevigCalculator
from nba_model.backtest.kelly import KellyCalculator
from nba_model.backtest.metrics import BacktestMetricsCalculator, FullBacktestMetrics
from nba_model.types import Bet

if TYPE_CHECKING:
    pass

# =============================================================================
# Constants
# =============================================================================

DEFAULT_MIN_TRAIN_GAMES: int = 500
DEFAULT_VALIDATION_WINDOW: int = 100
DEFAULT_STEP_SIZE: int = 50
DEFAULT_INITIAL_BANKROLL: float = 10000.0


# =============================================================================
# Protocols
# =============================================================================


class TrainerProtocol(Protocol):
    """Protocol for model trainers used in backtesting."""

    def train(self, train_df: pd.DataFrame) -> None:
        """Train model on training data."""
        ...

    def predict(self, game_df: pd.DataFrame) -> dict[str, dict[str, float]]:
        """Generate predictions for games.

        Returns:
            Dict mapping game_id to prediction dict with keys:
            - 'home_win_prob': float
            - 'predicted_margin': float
            - 'predicted_total': float
        """
        ...


class OddsProviderProtocol(Protocol):
    """Protocol for odds data providers."""

    def get_odds(self, game_id: str, timestamp: datetime) -> dict[str, float] | None:
        """Get odds for a game at a specific time.

        Returns:
            Dict with keys: 'home_ml', 'away_ml', 'spread', 'total'
            or None if not available.
        """
        ...

    def get_closing_odds(self, game_id: str) -> dict[str, float] | None:
        """Get closing odds for a game."""
        ...


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class FoldInfo:
    """Information about a single walk-forward fold.

    Attributes:
        fold_num: Fold number (1-indexed).
        train_start: Start date of training period.
        train_end: End date of training period.
        val_start: Start date of validation period.
        val_end: End date of validation period.
        train_games: Number of games in training set.
        val_games: Number of games in validation set.
    """

    fold_num: int
    train_start: date
    train_end: date
    val_start: date
    val_end: date
    train_games: int
    val_games: int


@dataclass
class BacktestResult:
    """Container for aggregated backtest outcomes.

    Attributes:
        bets: List of all Bet objects placed during backtest.
        bankroll_history: List of bankroll values over time.
        fold_results: Results for each individual fold.
        metrics: Computed performance metrics.
        config: Configuration used for backtest.
        start_date: First date in backtest.
        end_date: Last date in backtest.
    """

    bets: list[Bet] = field(default_factory=list)
    bankroll_history: list[float] = field(default_factory=list)
    fold_results: list[FoldResult] = field(default_factory=list)
    metrics: FullBacktestMetrics | None = None
    config: BacktestConfig | None = None
    start_date: date | None = None
    end_date: date | None = None

    @property
    def total_return(self) -> float:
        """Calculate total return from bankroll history."""
        if not self.bankroll_history or len(self.bankroll_history) < 2:
            return 0.0
        initial = self.bankroll_history[0]
        final = self.bankroll_history[-1]
        return (final - initial) / initial if initial > 0 else 0.0

    @property
    def sharpe_ratio(self) -> float:
        """Get Sharpe ratio from metrics."""
        return self.metrics.sharpe_ratio if self.metrics else 0.0

    @property
    def max_drawdown(self) -> float:
        """Get max drawdown from metrics."""
        return self.metrics.max_drawdown if self.metrics else 0.0

    @property
    def win_rate(self) -> float:
        """Get win rate from metrics."""
        return self.metrics.win_rate if self.metrics else 0.0

    @property
    def avg_clv(self) -> float:
        """Get average CLV from metrics."""
        return self.metrics.avg_clv if self.metrics else 0.0


@dataclass
class FoldResult:
    """Results from a single walk-forward fold.

    Attributes:
        fold_info: Information about the fold.
        bets: Bets placed during this fold.
        train_metrics: Training metrics (if available).
        val_metrics: Validation metrics.
        bankroll_start: Bankroll at start of fold.
        bankroll_end: Bankroll at end of fold.
    """

    fold_info: FoldInfo
    bets: list[Bet] = field(default_factory=list)
    train_metrics: dict[str, float] = field(default_factory=dict)
    val_metrics: dict[str, float] = field(default_factory=dict)
    bankroll_start: float = 0.0
    bankroll_end: float = 0.0


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for walk-forward backtest.

    Attributes:
        min_train_games: Minimum games in training set.
        validation_window_games: Games per validation window.
        step_size_games: Step size between folds.
        initial_bankroll: Starting bankroll.
        devig_method: Devigging method ('multiplicative', 'power', 'shin').
        kelly_fraction: Fractional Kelly multiplier.
        max_bet_pct: Maximum bet as percentage of bankroll.
        min_edge_pct: Minimum edge to place bet.
        bet_types: Bet types to consider ('moneyline', 'spread', 'total').
    """

    min_train_games: int = DEFAULT_MIN_TRAIN_GAMES
    validation_window_games: int = DEFAULT_VALIDATION_WINDOW
    step_size_games: int = DEFAULT_STEP_SIZE
    initial_bankroll: float = DEFAULT_INITIAL_BANKROLL
    devig_method: str = "power"
    kelly_fraction: float = 0.25
    max_bet_pct: float = 0.02
    min_edge_pct: float = 0.02
    bet_types: tuple[str, ...] = ("moneyline",)


# =============================================================================
# Walk-Forward Engine
# =============================================================================


class WalkForwardEngine:
    """Walk-forward validation engine for backtesting.

    Implements temporal cross-validation that prevents future information
    leakage by always training on past data and validating on future data.

    Attributes:
        min_train_games: Minimum games required in training set.
        validation_window_games: Number of games per validation fold.
        step_size_games: Number of games to advance between folds.

    Example:
        >>> engine = WalkForwardEngine(min_train_games=500)
        >>> folds = engine.generate_folds(games_df)
        >>> for train_df, val_df in folds:
        ...     # Train and validate
    """

    def __init__(
        self,
        min_train_games: int = DEFAULT_MIN_TRAIN_GAMES,
        validation_window_games: int = DEFAULT_VALIDATION_WINDOW,
        step_size_games: int = DEFAULT_STEP_SIZE,
    ) -> None:
        """Initialize WalkForwardEngine.

        Args:
            min_train_games: Minimum games in training set (default 500).
            validation_window_games: Games per validation fold (default 100).
            step_size_games: Step size between folds (default 50).
        """
        if min_train_games < 1:
            raise ValueError("min_train_games must be >= 1")
        if validation_window_games < 1:
            raise ValueError("validation_window_games must be >= 1")
        if step_size_games < 1:
            raise ValueError("step_size_games must be >= 1")

        self.min_train_games = min_train_games
        self.validation_window_games = validation_window_games
        self.step_size_games = step_size_games

    def generate_folds(
        self,
        games_df: pd.DataFrame,
        date_column: str = "game_date",
    ) -> list[tuple[pd.DataFrame, pd.DataFrame, FoldInfo]]:
        """Generate walk-forward train/validation folds.

        Args:
            games_df: DataFrame with game data, must have date column.
            date_column: Name of the date column.

        Returns:
            List of (train_df, val_df, fold_info) tuples.

        Raises:
            ValueError: If insufficient data for even one fold.
        """
        if date_column not in games_df.columns:
            raise ValueError(f"Date column '{date_column}' not found in DataFrame")

        # Sort by date
        df = games_df.sort_values(date_column).reset_index(drop=True)

        total_games = len(df)
        min_required = self.min_train_games + self.validation_window_games

        if total_games < min_required:
            raise ValueError(
                f"Insufficient data: {total_games} games, need {min_required}"
            )

        folds: list[tuple[pd.DataFrame, pd.DataFrame, FoldInfo]] = []
        fold_num = 0
        train_end_idx = self.min_train_games

        while train_end_idx + self.validation_window_games <= total_games:
            fold_num += 1

            # Training set: all games before train_end_idx
            train_df = df.iloc[:train_end_idx].copy()

            # Validation set: next validation_window_games games
            val_start_idx = train_end_idx
            val_end_idx = min(val_start_idx + self.validation_window_games, total_games)
            val_df = df.iloc[val_start_idx:val_end_idx].copy()

            # Create fold info
            fold_info = FoldInfo(
                fold_num=fold_num,
                train_start=df.iloc[0][date_column],
                train_end=df.iloc[train_end_idx - 1][date_column],
                val_start=df.iloc[val_start_idx][date_column],
                val_end=df.iloc[val_end_idx - 1][date_column],
                train_games=len(train_df),
                val_games=len(val_df),
            )

            folds.append((train_df, val_df, fold_info))

            # Advance training window
            train_end_idx += self.step_size_games

        return folds

    def run_backtest(
        self,
        games_df: pd.DataFrame,
        trainer: TrainerProtocol,
        odds_provider: OddsProviderProtocol | None = None,
        config: BacktestConfig | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> BacktestResult:
        """Execute full walk-forward backtest.

        For each fold:
        1. Train model on training set
        2. Generate predictions on validation set
        3. Fetch market odds (if provider available)
        4. Apply devigging to get fair probabilities
        5. Calculate edge (model_prob - market_prob)
        6. Apply Kelly sizing for bets with positive edge
        7. Simulate bet outcomes
        8. Track profit/loss and bankroll

        Args:
            games_df: DataFrame with game data and outcomes.
            trainer: Model trainer implementing TrainerProtocol.
            odds_provider: Optional odds data provider.
            config: Backtest configuration (uses defaults if None).
            progress_callback: Optional callback(fold_num, total_folds, status).

        Returns:
            BacktestResult with all metrics and bet history.
        """
        cfg = config or BacktestConfig()

        # Initialize calculators
        devig_calc = DevigCalculator()
        kelly_calc = KellyCalculator(
            fraction=cfg.kelly_fraction,
            max_bet_pct=cfg.max_bet_pct,
            min_edge_pct=cfg.min_edge_pct,
        )

        # Generate folds
        folds = self.generate_folds(games_df)

        # Initialize result
        result = BacktestResult(
            bankroll_history=[cfg.initial_bankroll],
            config=cfg,
        )
        bankroll = cfg.initial_bankroll

        # Process each fold
        for train_df, val_df, fold_info in folds:
            if progress_callback:
                progress_callback(
                    fold_info.fold_num,
                    len(folds),
                    f"Training fold {fold_info.fold_num}",
                )

            fold_result = FoldResult(
                fold_info=fold_info,
                bankroll_start=bankroll,
            )

            # Train model
            try:
                trainer.train(train_df)
            except Exception as e:
                # Log and skip fold on training failure
                fold_result.train_metrics["error"] = str(e)
                result.fold_results.append(fold_result)
                continue

            if progress_callback:
                progress_callback(
                    fold_info.fold_num,
                    len(folds),
                    f"Validating fold {fold_info.fold_num}",
                )

            # Generate predictions and place bets
            try:
                predictions = trainer.predict(val_df)
            except Exception:
                # Skip fold on prediction failure
                result.fold_results.append(fold_result)
                continue

            # Process each game in validation set
            for _, game_row in val_df.iterrows():
                game_id = game_row["game_id"]

                if game_id not in predictions:
                    continue

                pred = predictions[game_id]
                game_bets = self._process_game(
                    game_row=game_row,
                    prediction=pred,
                    odds_provider=odds_provider,
                    devig_calc=devig_calc,
                    kelly_calc=kelly_calc,
                    bankroll=bankroll,
                    config=cfg,
                )

                # Add bets and update bankroll
                for bet in game_bets:
                    result.bets.append(bet)
                    fold_result.bets.append(bet)
                    bankroll += bet.profit or 0.0
                    result.bankroll_history.append(bankroll)

            fold_result.bankroll_end = bankroll
            result.fold_results.append(fold_result)

        # Calculate metrics
        metrics_calc = BacktestMetricsCalculator()

        # Get closing odds if available
        closing_odds: dict[str, float] = {}
        if odds_provider:
            for bet in result.bets:
                close = odds_provider.get_closing_odds(bet.game_id)
                if close and "home_ml" in close:
                    closing_odds[bet.game_id] = close["home_ml"]

        result.metrics = metrics_calc.calculate_all(
            bets=result.bets,
            bankroll_history=result.bankroll_history,
            initial_bankroll=cfg.initial_bankroll,
            closing_odds=closing_odds if closing_odds else None,
        )

        # Set date range
        if result.bets:
            result.start_date = min(b.timestamp.date() for b in result.bets)
            result.end_date = max(b.timestamp.date() for b in result.bets)

        return result

    def _process_game(
        self,
        game_row: pd.Series,
        prediction: dict[str, float],
        odds_provider: OddsProviderProtocol | None,
        devig_calc: DevigCalculator,
        kelly_calc: KellyCalculator,
        bankroll: float,
        config: BacktestConfig,
    ) -> list[Bet]:
        """Process a single game and generate bets.

        Args:
            game_row: Series with game data.
            prediction: Model predictions for this game.
            odds_provider: Optional odds provider.
            devig_calc: Devig calculator.
            kelly_calc: Kelly calculator.
            bankroll: Current bankroll.
            config: Backtest configuration.

        Returns:
            List of Bet objects for this game.
        """
        bets: list[Bet] = []
        game_id = game_row["game_id"]
        game_date = game_row["game_date"]
        timestamp = datetime.combine(game_date, datetime.min.time())

        # Determine game outcome
        home_score = game_row.get("home_score", 0)
        away_score = game_row.get("away_score", 0)
        home_won = home_score > away_score

        # Get odds (use defaults if no provider)
        odds = odds_provider.get_odds(game_id, timestamp) if odds_provider else None

        # Default to -110 lines if no odds available
        if odds is None:
            odds = {
                "home_ml": 1.91,
                "away_ml": 1.91,
                "spread": -3.5,  # Home favored
                "total": 220.0,
            }

        # Process moneyline if in bet_types
        if "moneyline" in config.bet_types:
            ml_bet = self._create_moneyline_bet(
                game_id=game_id,
                timestamp=timestamp,
                prediction=prediction,
                odds=odds,
                home_won=home_won,
                devig_calc=devig_calc,
                kelly_calc=kelly_calc,
                bankroll=bankroll,
                config=config,
            )
            if ml_bet:
                bets.append(ml_bet)

        # Process spread if in bet_types
        if "spread" in config.bet_types:
            spread_bet = self._create_spread_bet(
                game_id=game_id,
                timestamp=timestamp,
                prediction=prediction,
                odds=odds,
                margin=home_score - away_score,
                devig_calc=devig_calc,
                kelly_calc=kelly_calc,
                bankroll=bankroll,
                config=config,
            )
            if spread_bet:
                bets.append(spread_bet)

        # Process total if in bet_types
        if "total" in config.bet_types:
            total_bet = self._create_total_bet(
                game_id=game_id,
                timestamp=timestamp,
                prediction=prediction,
                odds=odds,
                actual_total=home_score + away_score,
                devig_calc=devig_calc,
                kelly_calc=kelly_calc,
                bankroll=bankroll,
                config=config,
            )
            if total_bet:
                bets.append(total_bet)

        return bets

    def _create_moneyline_bet(
        self,
        game_id: str,
        timestamp: datetime,
        prediction: dict[str, float],
        odds: dict[str, float],
        home_won: bool,
        devig_calc: DevigCalculator,
        kelly_calc: KellyCalculator,
        bankroll: float,
        config: BacktestConfig,
    ) -> Bet | None:
        """Create a moneyline bet if edge exists."""
        home_ml = odds.get("home_ml", 1.91)
        away_ml = odds.get("away_ml", 1.91)

        # Devig to get fair probabilities
        try:
            fair_probs = devig_calc.devig(home_ml, away_ml, config.devig_method)
        except Exception:
            return None

        model_home_prob = prediction.get("home_win_prob", 0.5)

        # Check home side
        home_edge = model_home_prob - fair_probs.home
        away_edge = (1 - model_home_prob) - fair_probs.away

        # Bet on side with best edge
        if home_edge >= config.min_edge_pct and home_edge >= away_edge:
            kelly_result = kelly_calc.calculate(bankroll, model_home_prob, home_ml)
            if kelly_result.has_edge:
                result_str = "win" if home_won else "loss"
                profit = (
                    kelly_result.bet_amount * (home_ml - 1)
                    if home_won
                    else -kelly_result.bet_amount
                )
                return Bet(
                    game_id=game_id,
                    timestamp=timestamp,
                    bet_type="moneyline",
                    side="home",
                    model_prob=model_home_prob,
                    market_odds=home_ml,
                    market_prob=fair_probs.home,
                    edge=home_edge,
                    kelly_fraction=kelly_result.full_kelly,
                    bet_amount=kelly_result.bet_amount,
                    result=result_str,
                    profit=profit,
                )

        elif away_edge >= config.min_edge_pct:
            model_away_prob = 1 - model_home_prob
            kelly_result = kelly_calc.calculate(bankroll, model_away_prob, away_ml)
            if kelly_result.has_edge:
                result_str = "win" if not home_won else "loss"
                profit = (
                    kelly_result.bet_amount * (away_ml - 1)
                    if not home_won
                    else -kelly_result.bet_amount
                )
                return Bet(
                    game_id=game_id,
                    timestamp=timestamp,
                    bet_type="moneyline",
                    side="away",
                    model_prob=model_away_prob,
                    market_odds=away_ml,
                    market_prob=fair_probs.away,
                    edge=away_edge,
                    kelly_fraction=kelly_result.full_kelly,
                    bet_amount=kelly_result.bet_amount,
                    result=result_str,
                    profit=profit,
                )

        return None

    def _create_spread_bet(
        self,
        game_id: str,
        timestamp: datetime,
        prediction: dict[str, float],
        odds: dict[str, float],
        margin: int,
        devig_calc: DevigCalculator,
        kelly_calc: KellyCalculator,
        bankroll: float,
        config: BacktestConfig,
    ) -> Bet | None:
        """Create a spread bet if edge exists.

        This is a simplified implementation - a full version would need
        point spread probabilities and line values.
        """
        predicted_margin = prediction.get("predicted_margin", 0.0)
        spread = odds.get("spread", -3.5)  # Home spread

        # Estimate probability home covers (simplified)
        # Using a normal distribution centered on predicted margin
        from scipy.stats import norm

        std_dev = 12.0  # Typical NBA game std dev
        home_cover_prob = float(norm.cdf((predicted_margin - spread) / std_dev))

        # Assume -110 lines for spread
        spread_odds = 1.91

        home_edge = home_cover_prob - 0.5  # Against -110 line

        if home_edge >= config.min_edge_pct:
            kelly_result = kelly_calc.calculate(bankroll, home_cover_prob, spread_odds)
            if kelly_result.has_edge:
                home_covered = margin > spread
                result_str = "win" if home_covered else "loss"
                profit = (
                    kelly_result.bet_amount * (spread_odds - 1)
                    if home_covered
                    else -kelly_result.bet_amount
                )
                return Bet(
                    game_id=game_id,
                    timestamp=timestamp,
                    bet_type="spread",
                    side="home",
                    model_prob=home_cover_prob,
                    market_odds=spread_odds,
                    market_prob=0.5,
                    edge=home_edge,
                    kelly_fraction=kelly_result.full_kelly,
                    bet_amount=kelly_result.bet_amount,
                    result=result_str,
                    profit=profit,
                )

        return None

    def _create_total_bet(
        self,
        game_id: str,
        timestamp: datetime,
        prediction: dict[str, float],
        odds: dict[str, float],
        actual_total: int,
        devig_calc: DevigCalculator,
        kelly_calc: KellyCalculator,
        bankroll: float,
        config: BacktestConfig,
    ) -> Bet | None:
        """Create a total (over/under) bet if edge exists."""
        predicted_total = prediction.get("predicted_total", 220.0)
        total_line = odds.get("total", 220.0)

        # Estimate over probability (simplified)
        from scipy.stats import norm

        std_dev = 15.0  # Typical NBA total std dev
        over_prob = float(1 - norm.cdf((total_line - predicted_total) / std_dev))

        # Assume -110 lines
        total_odds = 1.91

        over_edge = over_prob - 0.5
        under_edge = (1 - over_prob) - 0.5

        if over_edge >= config.min_edge_pct and over_edge >= under_edge:
            kelly_result = kelly_calc.calculate(bankroll, over_prob, total_odds)
            if kelly_result.has_edge:
                went_over = actual_total > total_line
                result_str = "win" if went_over else "loss"
                profit = (
                    kelly_result.bet_amount * (total_odds - 1)
                    if went_over
                    else -kelly_result.bet_amount
                )
                return Bet(
                    game_id=game_id,
                    timestamp=timestamp,
                    bet_type="total",
                    side="over",
                    model_prob=over_prob,
                    market_odds=total_odds,
                    market_prob=0.5,
                    edge=over_edge,
                    kelly_fraction=kelly_result.full_kelly,
                    bet_amount=kelly_result.bet_amount,
                    result=result_str,
                    profit=profit,
                )

        elif under_edge >= config.min_edge_pct:
            under_prob = 1 - over_prob
            kelly_result = kelly_calc.calculate(bankroll, under_prob, total_odds)
            if kelly_result.has_edge:
                went_under = actual_total < total_line
                result_str = "win" if went_under else "loss"
                profit = (
                    kelly_result.bet_amount * (total_odds - 1)
                    if went_under
                    else -kelly_result.bet_amount
                )
                return Bet(
                    game_id=game_id,
                    timestamp=timestamp,
                    bet_type="total",
                    side="under",
                    model_prob=under_prob,
                    market_odds=total_odds,
                    market_prob=0.5,
                    edge=under_edge,
                    kelly_fraction=kelly_result.full_kelly,
                    bet_amount=kelly_result.bet_amount,
                    result=result_str,
                    profit=profit,
                )

        return None


def create_mock_trainer() -> TrainerProtocol:
    """Create a mock trainer for testing.

    Returns:
        A trainer that returns random predictions.
    """

    class MockTrainer:
        def train(self, train_df: pd.DataFrame) -> None:
            pass

        def predict(self, game_df: pd.DataFrame) -> dict[str, dict[str, float]]:
            predictions = {}
            for _, row in game_df.iterrows():
                game_id = row["game_id"]
                # Random-ish predictions
                np.random.seed(hash(game_id) % (2**32))
                predictions[game_id] = {
                    "home_win_prob": 0.45 + np.random.random() * 0.1,
                    "predicted_margin": np.random.normal(0, 5),
                    "predicted_total": 215 + np.random.normal(0, 10),
                }
            return predictions

    return MockTrainer()
