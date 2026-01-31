"""Unit tests for spatial spacing calculator.

Tests cover:
- Convex hull area calculation
- Centroid calculation accuracy
- Edge cases (collinear points, insufficient shots)
- Lineup hash determinism
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_model.features.spatial import (
    SpacingCalculator,
    calculate_lineup_spacing,
)


class TestSpacingCalculator:
    """Tests for SpacingCalculator class."""

    @pytest.fixture
    def calculator(self) -> SpacingCalculator:
        """Create calculator with test parameters."""
        return SpacingCalculator(min_shots=5)

    @pytest.fixture
    def sample_shots_df(self) -> pd.DataFrame:
        """Create sample shot data forming a square pattern."""
        # Create shots in a square pattern for predictable hull area
        return pd.DataFrame(
            {
                "player_id": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
                "loc_x": [-100, 100, 100, -100, -50, 50, 50, -50, 0, 0, -25, 25],
                "loc_y": [0, 0, 200, 200, 50, 50, 150, 150, 100, 100, 75, 75],
            }
        )

    def test_hull_area_on_square_pattern(
        self,
        calculator: SpacingCalculator,
    ) -> None:
        """Convex hull of square should have expected area."""
        # Create a 200x200 square
        shots_df = pd.DataFrame(
            {
                "player_id": [1, 1, 1, 1],
                "loc_x": [-100, 100, 100, -100],
                "loc_y": [0, 0, 200, 200],
            }
        )

        metrics = calculator.calculate_lineup_spacing([1], shots_df)

        # Area should be 200 * 200 = 40000
        assert metrics["hull_area"] == pytest.approx(40000, rel=0.01)

    def test_hull_area_on_triangle_pattern(
        self,
        calculator: SpacingCalculator,
    ) -> None:
        """Convex hull of triangle should have expected area."""
        # Create a triangle with base 200 and height 200
        # Area = 0.5 * base * height = 0.5 * 200 * 200 = 20000
        shots_df = pd.DataFrame(
            {
                "player_id": [1, 1, 1],
                "loc_x": [0, 100, -100],
                "loc_y": [200, 0, 0],
            }
        )

        # Need at least MIN_SHOTS_FOR_HULL (4) for hull calculation
        # Add more shots to meet minimum
        shots_df = pd.concat([shots_df, shots_df], ignore_index=True)

        metrics = calculator.calculate_lineup_spacing([1], shots_df)

        # Area should be approximately 20000
        assert metrics["hull_area"] == pytest.approx(20000, rel=0.01)

    def test_centroid_calculation_matches_geometric_center(
        self,
        calculator: SpacingCalculator,
    ) -> None:
        """Centroid should be at geometric center of points."""
        # Create symmetric pattern around (0, 100)
        shots_df = pd.DataFrame(
            {
                "player_id": [1, 1, 1, 1],
                "loc_x": [-100, 100, -100, 100],
                "loc_y": [0, 0, 200, 200],
            }
        )

        metrics = calculator.calculate_lineup_spacing([1], shots_df)

        assert metrics["centroid_x"] == pytest.approx(0, abs=1)
        assert metrics["centroid_y"] == pytest.approx(100, abs=1)

    def test_collinear_points_return_zero_hull(
        self,
        calculator: SpacingCalculator,
    ) -> None:
        """Collinear shots should return zero hull area."""
        # All shots on a line
        shots_df = pd.DataFrame(
            {
                "player_id": [1, 1, 1, 1, 1],
                "loc_x": [0, 0, 0, 0, 0],
                "loc_y": [0, 50, 100, 150, 200],
            }
        )

        metrics = calculator.calculate_lineup_spacing([1], shots_df)

        assert metrics["hull_area"] == 0.0

    def test_insufficient_shots_returns_zeros(
        self,
        calculator: SpacingCalculator,
    ) -> None:
        """With fewer than minimum shots, return zero metrics."""
        shots_df = pd.DataFrame(
            {
                "player_id": [1, 1],
                "loc_x": [0, 100],
                "loc_y": [0, 100],
            }
        )

        metrics = calculator.calculate_lineup_spacing([1], shots_df)

        assert metrics["hull_area"] == 0.0
        assert metrics["centroid_x"] == 0.0
        assert metrics["centroid_y"] == 0.0
        assert metrics["shot_count"] == 2

    def test_lineup_hash_is_deterministic(self) -> None:
        """Same players in different order should produce same hash."""
        lineup1 = [5, 3, 1, 4, 2]
        lineup2 = [1, 2, 3, 4, 5]
        lineup3 = [4, 5, 2, 3, 1]

        hash1 = SpacingCalculator.compute_lineup_hash(lineup1)
        hash2 = SpacingCalculator.compute_lineup_hash(lineup2)
        hash3 = SpacingCalculator.compute_lineup_hash(lineup3)

        assert hash1 == hash2 == hash3

    def test_different_lineups_have_different_hashes(self) -> None:
        """Different lineups should have different hashes."""
        lineup1 = [1, 2, 3, 4, 5]
        lineup2 = [1, 2, 3, 4, 6]

        hash1 = SpacingCalculator.compute_lineup_hash(lineup1)
        hash2 = SpacingCalculator.compute_lineup_hash(lineup2)

        assert hash1 != hash2

    def test_corner_density_calculation(
        self,
        calculator: SpacingCalculator,
    ) -> None:
        """Corner density should reflect proportion of corner shots."""
        # Create shots: 5 in corners, 5 elsewhere
        shots_df = pd.DataFrame(
            {
                "player_id": [1] * 10,
                "loc_x": [-240, -240, 240, 240, 240, 0, 0, 0, 0, 0],
                "loc_y": [50, 60, 50, 60, 70, 100, 150, 200, 250, 300],
            }
        )

        metrics = calculator.calculate_lineup_spacing([1], shots_df)

        # 5 corner shots out of 10 = 0.5
        assert metrics["corner_density"] == pytest.approx(0.5, rel=0.1)

    def test_avg_distance_calculation(
        self,
        calculator: SpacingCalculator,
    ) -> None:
        """Average distance should be mean distance from basket."""
        # Create shots at known distances from (0, 0)
        # Distance from (100, 0) = 100
        # Distance from (0, 100) = 100
        # Distance from (30, 40) = 50 (3-4-5 triangle)
        # Distance from (0, 0) = 0
        shots_df = pd.DataFrame(
            {
                "player_id": [1, 1, 1, 1],
                "loc_x": [100, 0, 30, 0],
                "loc_y": [0, 100, 40, 0],
            }
        )

        metrics = calculator.calculate_lineup_spacing([1], shots_df)

        # Average = (100 + 100 + 50 + 0) / 4 = 62.5
        assert metrics["avg_distance"] == pytest.approx(62.5, rel=0.01)

    def test_multi_player_lineup_aggregates_shots(
        self,
        calculator: SpacingCalculator,
        sample_shots_df: pd.DataFrame,
    ) -> None:
        """Lineup spacing should aggregate shots from all players."""
        metrics = calculator.calculate_lineup_spacing([1, 2, 3], sample_shots_df)

        # Should include all 12 shots
        assert metrics["shot_count"] == 12

    def test_calculate_all_lineup_spacing(
        self,
        calculator: SpacingCalculator,
        sample_shots_df: pd.DataFrame,
    ) -> None:
        """Should calculate spacing for multiple lineups."""
        lineups = [[1, 2], [2, 3], [1, 3]]
        results = calculator.calculate_all_lineup_spacing(lineups, sample_shots_df)

        assert len(results) == 3
        for lineup_hash, metrics in results.items():
            assert len(lineup_hash) == 16  # SHA256 truncated to 16 chars
            assert "hull_area" in metrics
            assert "shot_count" in metrics

    def test_player_shot_density_returns_grid(
        self,
        calculator: SpacingCalculator,
        sample_shots_df: pd.DataFrame,
    ) -> None:
        """KDE density should return properly shaped grid."""
        density = calculator.calculate_player_shot_density(1, sample_shots_df)

        assert density.shape == (calculator.grid_resolution, calculator.grid_resolution)
        assert np.all(density >= 0)  # Densities are non-negative

    def test_player_shot_density_insufficient_shots_returns_zeros(
        self,
        calculator: SpacingCalculator,
    ) -> None:
        """With too few shots, density should return zeros."""
        shots_df = pd.DataFrame(
            {
                "player_id": [1, 1],
                "loc_x": [0, 100],
                "loc_y": [0, 100],
            }
        )

        density = calculator.calculate_player_shot_density(1, shots_df)

        assert np.all(density == 0)

    def test_convenience_function_works(self) -> None:
        """calculate_lineup_spacing function should work correctly."""
        shots_df = pd.DataFrame(
            {
                "player_id": [1, 1, 1, 1, 1],
                "loc_x": [-100, 100, 100, -100, 0],
                "loc_y": [0, 0, 200, 200, 100],
            }
        )

        metrics = calculate_lineup_spacing([1], shots_df, min_shots=5)

        assert "hull_area" in metrics
        assert metrics["shot_count"] == 5
