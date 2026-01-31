"""Unit tests for RAPM calculator.

Tests cover:
- Sparse matrix construction with correct +1/-1/0 pattern
- Ridge regression convergence on synthetic data
- Time decay weighting application
- Minimum minutes filtering
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from nba_model.features.rapm import (
    RAPMCalculator,
)
from nba_model.types import InsufficientDataError


class TestRAPMCalculator:
    """Tests for RAPMCalculator class."""

    @pytest.fixture
    def sample_stints_df(self) -> pd.DataFrame:
        """Create sample stint data for testing."""
        n_stints = 150
        return pd.DataFrame(
            {
                "home_lineup": [[1, 2, 3, 4, 5]] * n_stints,
                "away_lineup": [[6, 7, 8, 9, 10]] * n_stints,
                "home_points": [10 + (i % 5) for i in range(n_stints)],
                "away_points": [8 + (i % 4) for i in range(n_stints)],
                "possessions": [10.0] * n_stints,
                "duration_seconds": [120] * n_stints,
                "game_date": pd.to_datetime(
                    [date.today() - timedelta(days=i % 30) for i in range(n_stints)]
                ),
            }
        )

    @pytest.fixture
    def calculator(self) -> RAPMCalculator:
        """Create calculator with test parameters."""
        return RAPMCalculator(lambda_=100, min_minutes=0)

    def test_build_design_matrix_creates_sparse_matrix(
        self,
        calculator: RAPMCalculator,
        sample_stints_df: pd.DataFrame,
    ) -> None:
        """Design matrix should be sparse with correct dimensions."""
        players = list(range(1, 11))  # 10 players
        X, y_off, y_def, weights, poss = calculator.build_stint_matrix(
            sample_stints_df, players
        )

        assert isinstance(X, csr_matrix)
        assert X.shape == (150, 10)  # 150 stints, 10 players
        assert len(y_off) == 150
        assert len(y_def) == 150
        assert len(weights) == 150
        assert len(poss) == 150

    def test_design_matrix_has_correct_values(
        self,
        calculator: RAPMCalculator,
    ) -> None:
        """Design matrix should have +1 for home, -1 for away players."""
        stints_df = pd.DataFrame(
            {
                "home_lineup": [[1, 2, 3, 4, 5]] * 100,
                "away_lineup": [[6, 7, 8, 9, 10]] * 100,
                "home_points": [10] * 100,
                "away_points": [8] * 100,
                "possessions": [10.0] * 100,
                "duration_seconds": [120] * 100,
                "game_date": pd.to_datetime([date.today()] * 100),
            }
        )

        players = list(range(1, 11))
        X, _, _, _, _ = calculator.build_stint_matrix(stints_df, players)

        # Check first row
        row = X.getrow(0).toarray().flatten()

        # Players 1-5 should be +1 (home)
        for i in range(5):
            assert row[i] == 1.0, f"Home player {i+1} should be +1"

        # Players 6-10 should be -1 (away)
        for i in range(5, 10):
            assert row[i] == -1.0, f"Away player {i+6} should be -1"

    def test_fit_returns_coefficients_for_all_players(
        self,
        calculator: RAPMCalculator,
        sample_stints_df: pd.DataFrame,
    ) -> None:
        """Fit should return RAPM tuple for each player."""
        result = calculator.fit(sample_stints_df)

        assert len(result) == 10  # 10 unique players
        for _player_id, coef in result.items():
            assert "orapm" in coef
            assert "drapm" in coef
            assert "total_rapm" in coef
            assert "sample_stints" in coef
            assert isinstance(coef["orapm"], float)
            assert isinstance(coef["drapm"], float)
            assert isinstance(coef["total_rapm"], float)
            assert isinstance(coef["sample_stints"], int)

    def test_fit_with_insufficient_stints_raises_error(
        self,
        calculator: RAPMCalculator,
    ) -> None:
        """Should raise error when not enough stints."""
        tiny_data = pd.DataFrame(
            {
                "home_lineup": [[1, 2, 3, 4, 5]] * 10,
                "away_lineup": [[6, 7, 8, 9, 10]] * 10,
                "home_points": [10] * 10,
                "away_points": [8] * 10,
                "possessions": [10.0] * 10,
                "duration_seconds": [120] * 10,
                "game_date": pd.to_datetime([date.today()] * 10),
            }
        )

        with pytest.raises(InsufficientDataError, match="at least"):
            calculator.fit(tiny_data)

    def test_time_decay_weighting_applies_exponential_decay(
        self,
        calculator: RAPMCalculator,
    ) -> None:
        """Time decay should apply exponential weighting."""
        today = date.today()
        dates = pd.Series(
            [
                today,
                today - timedelta(days=90),
                today - timedelta(days=180),
            ]
        )

        weights = calculator._calculate_time_weights(dates, reference_date=today)

        # Today should have weight ~1.0
        assert weights[0] == pytest.approx(1.0, rel=0.01)

        # 90 days ago should be e^(-90/180) ~ 0.607
        assert weights[1] == pytest.approx(0.607, rel=0.05)

        # 180 days ago should be e^(-180/180) ~ 0.368
        assert weights[2] == pytest.approx(0.368, rel=0.05)

    def test_minimum_minutes_filter_excludes_low_sample_players(self) -> None:
        """Players below minimum minutes should be excluded."""
        # Create data where one player (99) has fewer minutes than threshold
        n_stints = 200
        stints_df = pd.DataFrame(
            {
                # Player 99 replaces player 5 in only 10% of stints
                "home_lineup": [[1, 2, 3, 4, 5]] * 180 + [[1, 2, 3, 4, 99]] * 20,
                "away_lineup": [[6, 7, 8, 9, 10]] * n_stints,
                "home_points": [10] * n_stints,
                "away_points": [8] * n_stints,
                "possessions": [10.0] * n_stints,
                "duration_seconds": [120] * n_stints,  # 2 minutes each
                "game_date": pd.to_datetime([date.today()] * n_stints),
            }
        )

        # Player 99 appears in 20 stints * 2 min = 40 min
        # Player 5 appears in 180 stints * 2 min = 360 min
        # All other players appear in 200 stints * 2 min = 400 min
        # With min_minutes=50, player 99 should be excluded (40 < 50)

        calculator = RAPMCalculator(lambda_=100, min_minutes=50)
        result = calculator.fit(stints_df)

        # Player 99 has 40 min, below threshold - should be excluded
        assert 99 not in result
        # All other players should be included (10 players total)
        assert 1 in result
        assert 5 in result
        assert 6 in result
        assert 10 in result

    def test_rapm_coefficients_sum_correctly(
        self,
        calculator: RAPMCalculator,
        sample_stints_df: pd.DataFrame,
    ) -> None:
        """Total RAPM should approximately equal ORAPM + DRAPM."""
        result = calculator.fit(sample_stints_df)

        for _player_id, coef in result.items():
            expected_total = coef["orapm"] + coef["drapm"]
            assert coef["total_rapm"] == pytest.approx(expected_total, rel=0.01)

    def test_cross_validate_lambda_returns_best_score(
        self,
        calculator: RAPMCalculator,
        sample_stints_df: pd.DataFrame,
    ) -> None:
        """Cross-validation should return best lambda and scores."""
        best_lambda, scores = calculator.cross_validate_lambda(
            sample_stints_df,
            lambdas=[100.0, 1000.0, 5000.0],
            cv_folds=3,
        )

        assert best_lambda in [100.0, 1000.0, 5000.0]
        assert len(scores) == 3
        assert all(isinstance(v, float) for v in scores.values())

    def test_save_and_load_preserves_state(
        self,
        calculator: RAPMCalculator,
        sample_stints_df: pd.DataFrame,
        tmp_path,
    ) -> None:
        """Save and load should preserve calculator state."""
        # Fit calculator
        calculator.fit(sample_stints_df)

        # Save
        save_path = tmp_path / "rapm_state.json"
        calculator.save(save_path)

        # Load into new calculator
        new_calculator = RAPMCalculator()
        new_calculator.load(save_path)

        assert new_calculator.lambda_ == calculator.lambda_
        assert new_calculator.min_minutes == calculator.min_minutes
        assert new_calculator.player_mapping == calculator.player_mapping
        assert new_calculator.fitted_

    def test_json_lineup_parsing(
        self,
        calculator: RAPMCalculator,
    ) -> None:
        """Should handle JSON string lineups correctly."""
        import json

        stints_df = pd.DataFrame(
            {
                "home_lineup": [json.dumps([1, 2, 3, 4, 5])] * 100,
                "away_lineup": [json.dumps([6, 7, 8, 9, 10])] * 100,
                "home_points": [10] * 100,
                "away_points": [8] * 100,
                "possessions": [10.0] * 100,
                "duration_seconds": [120] * 100,
                "game_date": pd.to_datetime([date.today()] * 100),
            }
        )

        # Should not raise
        players = list(range(1, 11))
        design_matrix, _, _, _, _ = calculator.build_stint_matrix(stints_df, players)
        assert design_matrix.shape == (100, 10)
