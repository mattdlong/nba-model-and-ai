"""Tests for walk-forward validation engine."""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from nba_model.backtest.engine import (
    BacktestConfig,
    BacktestResult,
    FoldInfo,
    FoldResult,
    WalkForwardEngine,
    create_mock_trainer,
)


class TestFoldInfo:
    """Tests for FoldInfo dataclass."""

    def test_fold_info_creation(self) -> None:
        """Test FoldInfo can be created with all fields."""
        info = FoldInfo(
            fold_num=1,
            train_start=date(2023, 1, 1),
            train_end=date(2023, 6, 30),
            val_start=date(2023, 7, 1),
            val_end=date(2023, 7, 31),
            train_games=500,
            val_games=100,
        )
        assert info.fold_num == 1
        assert info.train_games == 500
        assert info.val_games == 100


class TestBacktestConfig:
    """Tests for BacktestConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = BacktestConfig()
        assert config.min_train_games == 500
        assert config.validation_window_games == 100
        assert config.step_size_games == 50
        assert config.initial_bankroll == 10000.0
        assert config.kelly_fraction == 0.25
        assert config.max_bet_pct == 0.02
        assert config.min_edge_pct == 0.02
        assert config.devig_method == "power"

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = BacktestConfig(
            min_train_games=200,
            kelly_fraction=0.5,
            bet_types=("moneyline", "spread"),
        )
        assert config.min_train_games == 200
        assert config.kelly_fraction == 0.5
        assert "spread" in config.bet_types


class TestBacktestResult:
    """Tests for BacktestResult class."""

    def test_empty_result(self) -> None:
        """Test empty result properties."""
        result = BacktestResult()
        assert result.total_return == 0.0
        assert result.sharpe_ratio == 0.0
        assert result.max_drawdown == 0.0
        assert result.win_rate == 0.0

    def test_total_return_calculation(self) -> None:
        """Test total return from bankroll history."""
        result = BacktestResult(bankroll_history=[10000, 11000, 11500])
        assert result.total_return == 0.15  # (11500 - 10000) / 10000


class TestWalkForwardEngine:
    """Tests for WalkForwardEngine class."""

    @pytest.fixture
    def engine(self) -> WalkForwardEngine:
        """Create engine with small values for testing."""
        return WalkForwardEngine(
            min_train_games=50,
            validation_window_games=20,
            step_size_games=10,
        )

    @pytest.fixture
    def sample_games_df(self) -> pd.DataFrame:
        """Create sample games DataFrame."""
        n_games = 200
        dates = pd.date_range("2023-01-01", periods=n_games, freq="D")
        return pd.DataFrame(
            {
                "game_id": [f"GAME{i:04d}" for i in range(n_games)],
                "game_date": dates,
                "home_score": np.random.randint(90, 130, n_games),
                "away_score": np.random.randint(90, 130, n_games),
                "season_id": "2022-23",
            }
        )

    def test_invalid_parameters_raise(self) -> None:
        """Test invalid engine parameters raise errors."""
        with pytest.raises(ValueError):
            WalkForwardEngine(min_train_games=0)
        with pytest.raises(ValueError):
            WalkForwardEngine(validation_window_games=0)
        with pytest.raises(ValueError):
            WalkForwardEngine(step_size_games=0)

    def test_generate_folds_respects_min_train(
        self, engine: WalkForwardEngine, sample_games_df: pd.DataFrame
    ) -> None:
        """Test fold generation respects minimum training size."""
        folds = engine.generate_folds(sample_games_df)

        for train_df, val_df, fold_info in folds:
            assert len(train_df) >= engine.min_train_games

    def test_generate_folds_chronological(
        self, engine: WalkForwardEngine, sample_games_df: pd.DataFrame
    ) -> None:
        """Test folds are chronologically ordered (no leakage)."""
        folds = engine.generate_folds(sample_games_df)

        for train_df, val_df, fold_info in folds:
            # All training dates should be before all validation dates
            max_train_date = train_df["game_date"].max()
            min_val_date = val_df["game_date"].min()
            assert max_train_date < min_val_date

    def test_generate_folds_disjoint(
        self, engine: WalkForwardEngine, sample_games_df: pd.DataFrame
    ) -> None:
        """Test train/val sets are disjoint within each fold."""
        folds = engine.generate_folds(sample_games_df)

        for train_df, val_df, fold_info in folds:
            train_ids = set(train_df["game_id"])
            val_ids = set(val_df["game_id"])
            assert len(train_ids & val_ids) == 0

    def test_generate_folds_validation_window(
        self, engine: WalkForwardEngine, sample_games_df: pd.DataFrame
    ) -> None:
        """Test validation window size is respected."""
        folds = engine.generate_folds(sample_games_df)

        for train_df, val_df, fold_info in folds:
            assert len(val_df) <= engine.validation_window_games

    def test_generate_folds_insufficient_data(self) -> None:
        """Test insufficient data raises error."""
        engine = WalkForwardEngine(
            min_train_games=100,
            validation_window_games=50,
        )
        small_df = pd.DataFrame(
            {
                "game_id": ["G1", "G2"],
                "game_date": [date(2023, 1, 1), date(2023, 1, 2)],
            }
        )

        with pytest.raises(ValueError, match="Insufficient data"):
            engine.generate_folds(small_df)

    def test_generate_folds_missing_date_column(
        self, engine: WalkForwardEngine
    ) -> None:
        """Test missing date column raises error."""
        df = pd.DataFrame({"game_id": ["G1", "G2"], "score": [100, 110]})

        with pytest.raises(ValueError, match="Date column"):
            engine.generate_folds(df)

    def test_generate_folds_returns_fold_info(
        self, engine: WalkForwardEngine, sample_games_df: pd.DataFrame
    ) -> None:
        """Test fold info is populated correctly."""
        folds = engine.generate_folds(sample_games_df)

        for i, (train_df, val_df, fold_info) in enumerate(folds, 1):
            assert fold_info.fold_num == i
            assert fold_info.train_games == len(train_df)
            assert fold_info.val_games == len(val_df)

    def test_run_backtest_with_mock_trainer(
        self, engine: WalkForwardEngine, sample_games_df: pd.DataFrame
    ) -> None:
        """Test backtest runs with mock trainer."""
        trainer = create_mock_trainer()
        config = BacktestConfig(
            min_train_games=50,
            validation_window_games=20,
            step_size_games=10,
        )

        result = engine.run_backtest(
            games_df=sample_games_df,
            trainer=trainer,
            config=config,
        )

        assert isinstance(result, BacktestResult)
        assert len(result.fold_results) > 0
        assert len(result.bankroll_history) > 0

    def test_run_backtest_tracks_bankroll(
        self, engine: WalkForwardEngine, sample_games_df: pd.DataFrame
    ) -> None:
        """Test backtest tracks bankroll history."""
        trainer = create_mock_trainer()
        config = BacktestConfig(
            min_train_games=50,
            validation_window_games=20,
            initial_bankroll=10000.0,
        )

        result = engine.run_backtest(
            games_df=sample_games_df,
            trainer=trainer,
            config=config,
        )

        # Bankroll history should start with initial value
        assert result.bankroll_history[0] == 10000.0

    def test_run_backtest_computes_metrics(
        self, engine: WalkForwardEngine, sample_games_df: pd.DataFrame
    ) -> None:
        """Test backtest computes metrics."""
        trainer = create_mock_trainer()
        config = BacktestConfig(
            min_train_games=50,
            validation_window_games=20,
        )

        result = engine.run_backtest(
            games_df=sample_games_df,
            trainer=trainer,
            config=config,
        )

        assert result.metrics is not None
        assert hasattr(result.metrics, "total_bets")
        assert hasattr(result.metrics, "win_rate")
        assert hasattr(result.metrics, "roi")

    def test_run_backtest_with_progress_callback(
        self, engine: WalkForwardEngine, sample_games_df: pd.DataFrame
    ) -> None:
        """Test backtest calls progress callback."""
        trainer = create_mock_trainer()
        config = BacktestConfig(
            min_train_games=50,
            validation_window_games=20,
        )

        progress_calls = []

        def callback(fold: int, total: int, status: str) -> None:
            progress_calls.append((fold, total, status))

        result = engine.run_backtest(
            games_df=sample_games_df,
            trainer=trainer,
            config=config,
            progress_callback=callback,
        )

        assert len(progress_calls) > 0


class TestMockTrainer:
    """Tests for mock trainer."""

    def test_mock_trainer_train(self) -> None:
        """Test mock trainer train method."""
        trainer = create_mock_trainer()
        df = pd.DataFrame({"game_id": ["G1"], "game_date": [date(2023, 1, 1)]})
        # Should not raise
        trainer.train(df)

    def test_mock_trainer_predict(self) -> None:
        """Test mock trainer predict method."""
        trainer = create_mock_trainer()
        df = pd.DataFrame(
            {
                "game_id": ["G1", "G2"],
                "game_date": [date(2023, 1, 1), date(2023, 1, 2)],
            }
        )

        predictions = trainer.predict(df)

        assert "G1" in predictions
        assert "G2" in predictions
        assert "home_win_prob" in predictions["G1"]
        assert "predicted_margin" in predictions["G1"]
        assert "predicted_total" in predictions["G1"]
        assert 0 <= predictions["G1"]["home_win_prob"] <= 1


class TestIntegration:
    """Integration tests for the backtest engine."""

    def test_full_backtest_pipeline(self) -> None:
        """Test complete backtest pipeline."""
        # Create sample data
        np.random.seed(42)
        n_games = 300
        dates = pd.date_range("2023-01-01", periods=n_games, freq="D")
        games_df = pd.DataFrame(
            {
                "game_id": [f"GAME{i:04d}" for i in range(n_games)],
                "game_date": dates,
                "home_score": np.random.randint(90, 130, n_games),
                "away_score": np.random.randint(90, 130, n_games),
            }
        )

        # Configure backtest
        config = BacktestConfig(
            min_train_games=100,
            validation_window_games=50,
            step_size_games=25,
            initial_bankroll=10000.0,
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

        # Run backtest
        result = engine.run_backtest(
            games_df=games_df,
            trainer=trainer,
            config=config,
        )

        # Verify results
        assert result.metrics is not None
        assert len(result.fold_results) >= 2
        assert len(result.bankroll_history) >= 1
        assert result.config == config

        # Check no look-ahead bias
        for fold in result.fold_results:
            assert fold.fold_info.train_end < fold.fold_info.val_start
