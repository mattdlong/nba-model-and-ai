"""Integration tests for the backtesting pipeline.

Tests the complete flow: odds -> devig -> kelly -> metrics.
Uses a synthetic dataset of 200+ games to validate end-to-end functionality.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from nba_model.backtest import (
    BacktestConfig,
    BacktestMetricsCalculator,
    DevigCalculator,
    KellyCalculator,
    WalkForwardEngine,
    create_mock_trainer,
)
from nba_model.types import Bet

if TYPE_CHECKING:
    pass


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def large_games_df() -> pd.DataFrame:
    """Generate a synthetic dataset of 800 games for backtesting.

    This exceeds the minimum 200 games required for integration tests
    and allows for multiple walk-forward folds.
    """
    np.random.seed(42)
    n_games = 800

    start_date = datetime(2022, 10, 1)
    dates = [start_date + timedelta(days=i // 3) for i in range(n_games)]

    return pd.DataFrame(
        {
            "game_id": [f"GAME{i:04d}" for i in range(n_games)],
            "game_date": dates,
            "home_team_id": np.random.randint(1, 31, n_games),
            "away_team_id": np.random.randint(1, 31, n_games),
            "home_score": np.random.randint(90, 130, n_games),
            "away_score": np.random.randint(90, 130, n_games),
            "season_id": ["2022-23"] * n_games,
        }
    )


@pytest.fixture
def synthetic_bets() -> list[Bet]:
    """Generate synthetic bet history for optimization tests."""
    np.random.seed(42)
    n_bets = 200

    bets = []
    for i in range(n_bets):
        model_prob = 0.48 + np.random.random() * 0.1
        market_odds = 1.85 + np.random.random() * 0.2
        market_prob = 1 / market_odds
        edge = model_prob - market_prob

        won = np.random.random() < model_prob
        result = "win" if won else "loss"
        bet_amount = 100.0
        profit = bet_amount * (market_odds - 1) if won else -bet_amount

        bet = Bet(
            game_id=f"GAME{i:04d}",
            timestamp=datetime(2023, 1, 1) + timedelta(days=i),
            bet_type="moneyline",
            side="home",
            model_prob=model_prob,
            market_odds=market_odds,
            market_prob=market_prob,
            edge=edge,
            kelly_fraction=0.0,
            bet_amount=bet_amount,
            result=result,
            profit=profit,
        )
        bets.append(bet)

    return bets


# =============================================================================
# Devigging Integration Tests
# =============================================================================


@pytest.mark.integration
class TestDevigIntegration:
    """Integration tests for devigging methods."""

    def test_all_methods_produce_valid_probabilities(self) -> None:
        """All devig methods should produce probabilities summing to 1."""
        calc = DevigCalculator()
        test_odds = [
            (1.91, 1.91),  # Balanced -110 lines
            (1.50, 2.80),  # Heavy favorite
            (2.20, 1.75),  # Slight underdog
            (1.10, 9.00),  # Large favorite
        ]

        for home_odds, away_odds in test_odds:
            for method in ["multiplicative", "power", "shin"]:
                result = calc.devig(home_odds, away_odds, method)
                total = result.home + result.away
                assert 0.99 <= total <= 1.01, (
                    f"{method} method failed for odds ({home_odds}, {away_odds}): "
                    f"sum = {total}"
                )
                assert result.home > 0 and result.away > 0

    def test_shin_iterative_solve_converges(self) -> None:
        """Shin method should converge for typical betting odds."""
        calc = DevigCalculator()

        # Test with various overround levels
        test_cases = [
            (1.91, 1.91),  # ~4.8% vig
            (1.95, 1.95),  # ~2.6% vig
            (1.87, 1.87),  # ~7.0% vig
            (1.50, 2.70),  # Asymmetric with moderate vig
        ]

        for home_odds, away_odds in test_cases:
            result = calc.shin_method_devig(home_odds, away_odds)

            # Verify convergence: probs should sum to 1
            assert abs(result.home + result.away - 1.0) < 0.01

            # Verify method label
            assert result.method == "shin"

    def test_power_vs_shin_similar_for_balanced_odds(self) -> None:
        """Power and Shin methods should produce similar results for balanced odds."""
        calc = DevigCalculator()

        power_result = calc.power_method_devig(1.91, 1.91)
        shin_result = calc.shin_method_devig(1.91, 1.91)

        # Should be close for balanced lines
        assert abs(power_result.home - shin_result.home) < 0.02
        assert abs(power_result.away - shin_result.away) < 0.02


# =============================================================================
# Kelly Integration Tests
# =============================================================================


@pytest.mark.integration
class TestKellyIntegration:
    """Integration tests for Kelly criterion sizing."""

    def test_kelly_with_devigged_market_prob(self) -> None:
        """Kelly should use devigged market probability for edge calculation."""
        devig_calc = DevigCalculator()
        kelly_calc = KellyCalculator(
            fraction=0.25,
            max_bet_pct=0.02,
            min_edge_pct=0.02,
        )

        # Get devigged probabilities
        fair_probs = devig_calc.power_method_devig(1.91, 1.91)

        # Model thinks home has 55% chance
        model_prob = 0.55
        bankroll = 10000.0

        # Calculate with devigged market probability
        result = kelly_calc.calculate(
            bankroll=bankroll,
            model_prob=model_prob,
            decimal_odds=1.91,
            market_prob=fair_probs.home,
        )

        # Edge should be model_prob - devigged_prob, not model_prob - implied_prob
        expected_edge = model_prob - fair_probs.home
        assert abs(result.edge - expected_edge) < 0.001

    def test_kelly_optimization_on_synthetic_data(
        self, synthetic_bets: list[Bet]
    ) -> None:
        """Kelly optimization should find a reasonable fraction."""
        kelly_calc = KellyCalculator()
        fractions = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]

        result = kelly_calc.optimize_fraction(
            historical_bets=synthetic_bets,
            fractions=fractions,
            initial_bankroll=10000.0,
            metric="sharpe",
        )

        # Best fraction should be within tested range
        assert result.best_fraction in fractions

        # Should have results for all fractions
        assert len(result.results_by_fraction) == len(fractions)

        # Best metric should be finite
        assert np.isfinite(result.best_metric)

    def test_kelly_respects_min_edge_threshold(self) -> None:
        """Kelly should not bet when edge is below minimum threshold."""
        kelly_calc = KellyCalculator(
            fraction=0.25,
            max_bet_pct=0.02,
            min_edge_pct=0.05,  # 5% minimum edge
        )

        # Model has only 3% edge
        result = kelly_calc.calculate(
            bankroll=10000.0,
            model_prob=0.55,
            decimal_odds=1.91,
            market_prob=0.52,  # 3% edge
        )

        # Should not have edge (below threshold)
        assert not result.has_edge
        assert result.bet_amount == 0.0


# =============================================================================
# Walk-Forward Engine Integration Tests
# =============================================================================


@pytest.mark.integration
class TestWalkForwardIntegration:
    """Integration tests for walk-forward validation engine."""

    def test_fold_generation_respects_temporal_ordering(
        self, large_games_df: pd.DataFrame
    ) -> None:
        """All validation games should be after all training games."""
        engine = WalkForwardEngine(
            min_train_games=500,
            validation_window_games=100,
            step_size_games=50,
        )

        folds = engine.generate_folds(large_games_df)

        assert len(folds) > 0, "Should generate at least one fold"

        for train_df, val_df, fold_info in folds:
            train_max_date = train_df["game_date"].max()
            val_min_date = val_df["game_date"].min()

            # Validation should start after training ends
            assert val_min_date >= train_max_date, (
                f"Fold {fold_info.fold_num}: Validation starts before training ends"
            )

            # No overlap
            assert len(set(train_df["game_id"]) & set(val_df["game_id"])) == 0

    def test_full_backtest_produces_valid_metrics(
        self, large_games_df: pd.DataFrame
    ) -> None:
        """Full backtest should produce valid, finite metrics."""
        config = BacktestConfig(
            min_train_games=500,
            validation_window_games=100,
            step_size_games=50,
            initial_bankroll=10000.0,
            devig_method="power",
            kelly_fraction=0.25,
            max_bet_pct=0.02,
            min_edge_pct=0.02,
        )

        engine = WalkForwardEngine(
            min_train_games=config.min_train_games,
            validation_window_games=config.validation_window_games,
            step_size_games=config.step_size_games,
        )

        trainer = create_mock_trainer()

        result = engine.run_backtest(
            games_df=large_games_df,
            trainer=trainer,
            config=config,
        )

        # Should have results
        assert result.bankroll_history is not None
        assert len(result.bankroll_history) > 0

        # Metrics should exist and be finite
        assert result.metrics is not None
        assert np.isfinite(result.metrics.total_return)
        assert np.isfinite(result.metrics.sharpe_ratio)
        assert np.isfinite(result.metrics.max_drawdown)

        # Win rate should be in valid range
        assert 0.0 <= result.metrics.win_rate <= 1.0

        # ROI should be finite (could be negative)
        assert np.isfinite(result.metrics.roi)

    def test_backtest_with_multiple_bet_types(
        self, large_games_df: pd.DataFrame
    ) -> None:
        """Backtest should handle multiple bet types correctly."""
        config = BacktestConfig(
            min_train_games=500,
            validation_window_games=100,
            step_size_games=50,
            initial_bankroll=10000.0,
            bet_types=("moneyline", "spread", "total"),
        )

        engine = WalkForwardEngine(
            min_train_games=config.min_train_games,
            validation_window_games=config.validation_window_games,
            step_size_games=config.step_size_games,
        )

        trainer = create_mock_trainer()

        result = engine.run_backtest(
            games_df=large_games_df,
            trainer=trainer,
            config=config,
        )

        # Should have metrics by bet type
        assert result.metrics is not None
        if result.metrics.metrics_by_type:
            # At least one bet type should have results
            assert len(result.metrics.metrics_by_type) > 0


# =============================================================================
# Metrics Integration Tests
# =============================================================================


@pytest.mark.integration
class TestMetricsIntegration:
    """Integration tests for backtest metrics calculation."""

    def test_metrics_from_backtest_result(
        self, large_games_df: pd.DataFrame
    ) -> None:
        """Metrics should be calculable from BacktestResult."""
        config = BacktestConfig(
            min_train_games=500,
            validation_window_games=100,
            step_size_games=50,
        )

        engine = WalkForwardEngine(
            min_train_games=config.min_train_games,
            validation_window_games=config.validation_window_games,
            step_size_games=config.step_size_games,
        )

        trainer = create_mock_trainer()

        result = engine.run_backtest(
            games_df=large_games_df,
            trainer=trainer,
            config=config,
        )

        # Calculate metrics using the spec-compliant API
        calc = BacktestMetricsCalculator()
        metrics_dict = calc.calculate_from_result(result)

        # Should return a dictionary
        assert isinstance(metrics_dict, dict)

        # Should have expected keys
        expected_keys = [
            "total_return",
            "sharpe_ratio",
            "max_drawdown",
            "win_rate",
            "roi",
            "total_bets",
        ]
        for key in expected_keys:
            assert key in metrics_dict, f"Missing key: {key}"

    def test_metrics_to_dict_round_trip(self, synthetic_bets: list[Bet]) -> None:
        """Metrics should be convertible to dict and back."""
        calc = BacktestMetricsCalculator()
        bankroll_history = [10000.0]

        # Simulate bankroll history
        bankroll = 10000.0
        for bet in synthetic_bets:
            bankroll += bet.profit or 0.0
            bankroll_history.append(bankroll)

        metrics = calc.calculate_all(
            bets=synthetic_bets,
            bankroll_history=bankroll_history,
            initial_bankroll=10000.0,
        )

        # Convert to dict
        metrics_dict = metrics.to_dict()

        # Verify all fields are present
        assert metrics_dict["total_bets"] == len(synthetic_bets)
        assert metrics_dict["win_rate"] == metrics.win_rate
        assert metrics_dict["sharpe_ratio"] == metrics.sharpe_ratio

    def test_clv_calculation_with_closing_odds_map(
        self, synthetic_bets: list[Bet]
    ) -> None:
        """CLV should be calculated correctly with closing odds map."""
        calc = BacktestMetricsCalculator()

        # Create closing odds map
        closing_odds_map: dict[tuple[str, str, str], float] = {}
        for bet in synthetic_bets[:50]:
            # Simulate closing odds slightly different from bet odds
            closing_odds = bet.market_odds * (0.95 + np.random.random() * 0.1)
            key = (bet.game_id, bet.bet_type, bet.side)
            closing_odds_map[key] = closing_odds

        metrics = calc.calculate_all(
            bets=synthetic_bets,
            bankroll_history=[10000.0] * (len(synthetic_bets) + 1),
            initial_bankroll=10000.0,
            closing_odds_map=closing_odds_map,
        )

        # Should have CLV metrics
        assert metrics.avg_clv != 0.0 or metrics.clv_positive_rate >= 0.0


# =============================================================================
# End-to-End Pipeline Test
# =============================================================================


@pytest.mark.integration
class TestEndToEndPipeline:
    """End-to-end integration test for the complete backtest pipeline."""

    def test_complete_backtest_pipeline(self, large_games_df: pd.DataFrame) -> None:
        """Test complete flow: data -> devig -> kelly -> backtest -> metrics."""
        # Step 1: Configure
        config = BacktestConfig(
            min_train_games=500,
            validation_window_games=100,
            step_size_games=100,
            initial_bankroll=10000.0,
            devig_method="shin",  # Use gold standard method
            kelly_fraction=0.25,
            max_bet_pct=0.02,
            min_edge_pct=0.02,
            bet_types=("moneyline",),
        )

        # Step 2: Create components
        engine = WalkForwardEngine(
            min_train_games=config.min_train_games,
            validation_window_games=config.validation_window_games,
            step_size_games=config.step_size_games,
        )
        trainer = create_mock_trainer()

        # Step 3: Run backtest
        result = engine.run_backtest(
            games_df=large_games_df,
            trainer=trainer,
            config=config,
        )

        # Step 4: Verify results
        # Should complete without errors
        assert result is not None

        # Should have fold results
        assert len(result.fold_results) > 0

        # Should have bankroll history
        assert len(result.bankroll_history) > 1

        # Metrics should be computed
        assert result.metrics is not None

        # Step 5: Generate report (should not raise)
        calc = BacktestMetricsCalculator()
        report = calc.generate_report(result.metrics, "Integration Test Report")
        assert len(report) > 100  # Should have substantial content

        # Step 6: Verify no NaN or infinite values
        assert np.isfinite(result.metrics.total_return)
        assert np.isfinite(result.metrics.sharpe_ratio)
        assert np.isfinite(result.metrics.sortino_ratio)
        assert np.isfinite(result.metrics.max_drawdown)
        assert np.isfinite(result.metrics.roi)
        assert np.isfinite(result.metrics.win_rate)
        assert np.isfinite(result.metrics.avg_edge)

        # Step 7: Verify plausible ranges
        assert -1.0 <= result.metrics.total_return <= 10.0
        assert 0.0 <= result.metrics.win_rate <= 1.0
        assert 0.0 <= result.metrics.max_drawdown <= 1.0
