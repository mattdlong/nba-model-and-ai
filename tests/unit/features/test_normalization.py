"""Unit tests for season normalizer.

Tests cover:
- fit() computes correct mean/std
- transform() produces correct z-scores
- save/load round-trip
- error handling for missing seasons
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_model.features.normalization import (
    MetricStats,
    SeasonNormalizer,
    normalize_by_season,
)


class TestSeasonNormalizer:
    """Tests for SeasonNormalizer class."""

    @pytest.fixture
    def normalizer(self) -> SeasonNormalizer:
        """Create season normalizer."""
        return SeasonNormalizer()

    @pytest.fixture
    def sample_stats_df(self) -> pd.DataFrame:
        """Create sample game stats DataFrame."""
        np.random.seed(42)
        n = 100

        return pd.DataFrame(
            {
                "season_id": ["2023-24"] * n,
                "pace": np.random.normal(100, 5, n),
                "offensive_rating": np.random.normal(110, 8, n),
                "defensive_rating": np.random.normal(108, 7, n),
                "efg_pct": np.random.normal(0.52, 0.05, n),
            }
        )

    @pytest.fixture
    def multi_season_df(self) -> pd.DataFrame:
        """Create multi-season stats DataFrame."""
        np.random.seed(42)
        n = 50

        season1 = pd.DataFrame(
            {
                "season_id": ["2022-23"] * n,
                "pace": np.random.normal(98, 4, n),
                "offensive_rating": np.random.normal(108, 7, n),
            }
        )

        season2 = pd.DataFrame(
            {
                "season_id": ["2023-24"] * n,
                "pace": np.random.normal(102, 5, n),
                "offensive_rating": np.random.normal(112, 8, n),
            }
        )

        return pd.concat([season1, season2], ignore_index=True)

    def test_fit_computes_correct_mean(
        self,
        normalizer: SeasonNormalizer,
        sample_stats_df: pd.DataFrame,
    ) -> None:
        """fit() should compute correct mean for each metric."""
        normalizer.fit(sample_stats_df)

        pace_stats = normalizer.stats[("2023-24", "pace")]
        expected_mean = sample_stats_df["pace"].mean()

        assert pace_stats.mean_value == pytest.approx(expected_mean, rel=0.01)

    def test_fit_computes_correct_std(
        self,
        normalizer: SeasonNormalizer,
        sample_stats_df: pd.DataFrame,
    ) -> None:
        """fit() should compute correct std for each metric."""
        normalizer.fit(sample_stats_df)

        pace_stats = normalizer.stats[("2023-24", "pace")]
        expected_std = sample_stats_df["pace"].std()

        assert pace_stats.std_value == pytest.approx(expected_std, rel=0.01)

    def test_transform_produces_z_scores_with_mean_zero(
        self,
        normalizer: SeasonNormalizer,
        sample_stats_df: pd.DataFrame,
    ) -> None:
        """transform() should produce z-scores with mean ~0."""
        normalizer.fit(sample_stats_df)
        result_df = normalizer.transform(sample_stats_df, season="2023-24")

        # Z-scores should have mean ~0
        assert result_df["pace_z"].mean() == pytest.approx(0, abs=0.01)

    def test_transform_produces_z_scores_with_std_one(
        self,
        normalizer: SeasonNormalizer,
        sample_stats_df: pd.DataFrame,
    ) -> None:
        """transform() should produce z-scores with std ~1."""
        normalizer.fit(sample_stats_df)
        result_df = normalizer.transform(sample_stats_df, season="2023-24")

        # Z-scores should have std ~1
        assert result_df["pace_z"].std() == pytest.approx(1, rel=0.05)

    def test_transform_single_value(
        self,
        normalizer: SeasonNormalizer,
        sample_stats_df: pd.DataFrame,
    ) -> None:
        """transform_value() should work for single values."""
        normalizer.fit(sample_stats_df)

        mean = sample_stats_df["pace"].mean()
        std = sample_stats_df["pace"].std()

        # Value at mean should give z-score of 0
        z = normalizer.transform_value(mean, "pace", "2023-24")
        assert z == pytest.approx(0, abs=0.01)

        # Value one std above mean should give z-score of 1
        z = normalizer.transform_value(mean + std, "pace", "2023-24")
        assert z == pytest.approx(1, rel=0.05)

    def test_inverse_transform_value(
        self,
        normalizer: SeasonNormalizer,
        sample_stats_df: pd.DataFrame,
    ) -> None:
        """inverse_transform_value() should recover original value."""
        normalizer.fit(sample_stats_df)

        original = 105.0
        z = normalizer.transform_value(original, "pace", "2023-24")
        recovered = normalizer.inverse_transform_value(z, "pace", "2023-24")

        assert recovered == pytest.approx(original, rel=0.01)

    def test_transform_without_fit_raises_error(
        self,
        normalizer: SeasonNormalizer,
        sample_stats_df: pd.DataFrame,
    ) -> None:
        """transform() without fit() should raise error."""
        with pytest.raises(ValueError, match="not been fitted"):
            normalizer.transform(sample_stats_df, season="2023-24")

    def test_transform_missing_season_warns(
        self,
        normalizer: SeasonNormalizer,
        sample_stats_df: pd.DataFrame,
    ) -> None:
        """transform() with missing season should warn and skip."""
        normalizer.fit(sample_stats_df)

        # Should not raise, but should skip unknown season
        result_df = normalizer.transform(sample_stats_df, season="2019-20")

        # Should not have z-score columns added
        assert "pace_z" not in result_df.columns

    def test_multi_season_fit(
        self,
        normalizer: SeasonNormalizer,
        multi_season_df: pd.DataFrame,
    ) -> None:
        """fit() should handle multiple seasons."""
        normalizer.fit(multi_season_df)

        assert ("2022-23", "pace") in normalizer.stats
        assert ("2023-24", "pace") in normalizer.stats

        # Different seasons should have different means
        stats_2022 = normalizer.stats[("2022-23", "pace")]
        stats_2023 = normalizer.stats[("2023-24", "pace")]

        assert stats_2022.mean_value != stats_2023.mean_value

    def test_get_season_stats(
        self,
        normalizer: SeasonNormalizer,
        multi_season_df: pd.DataFrame,
    ) -> None:
        """get_season_stats() should return stats for specific season."""
        normalizer.fit(multi_season_df)

        stats = normalizer.get_season_stats("2023-24")

        assert "pace" in stats
        assert "offensive_rating" in stats
        assert isinstance(stats["pace"], MetricStats)

    def test_save_and_load_preserves_state(
        self,
        normalizer: SeasonNormalizer,
        sample_stats_df: pd.DataFrame,
        tmp_path,
    ) -> None:
        """Save and load should preserve normalizer state."""
        normalizer.fit(sample_stats_df)

        save_path = tmp_path / "normalizer.json"
        normalizer.save(save_path)

        # Load into new normalizer
        new_normalizer = SeasonNormalizer()
        new_normalizer.load(save_path)

        assert new_normalizer.fitted_
        assert new_normalizer.metrics == normalizer.metrics

        # Stats should match
        old_stats = normalizer.stats[("2023-24", "pace")]
        new_stats = new_normalizer.stats[("2023-24", "pace")]

        assert new_stats.mean_value == old_stats.mean_value
        assert new_stats.std_value == old_stats.std_value

    def test_save_without_fit_raises_error(
        self,
        normalizer: SeasonNormalizer,
        tmp_path,
    ) -> None:
        """save() without fit() should raise error."""
        with pytest.raises(ValueError, match="not been fitted"):
            normalizer.save(tmp_path / "normalizer.json")

    def test_zero_std_handled(self, normalizer: SeasonNormalizer) -> None:
        """Zero std should be replaced with 1.0 to prevent division by zero."""
        # Create data with constant values (std = 0)
        df = pd.DataFrame(
            {
                "season_id": ["2023-24"] * 10,
                "pace": [100.0] * 10,  # All same value
            }
        )

        normalizer.fit(df)

        stats = normalizer.stats[("2023-24", "pace")]
        assert stats.std_value == 1.0  # Should be 1.0, not 0.0

    def test_inplace_transform(
        self,
        normalizer: SeasonNormalizer,
        sample_stats_df: pd.DataFrame,
    ) -> None:
        """inplace=True should modify original DataFrame."""
        normalizer.fit(sample_stats_df)

        original_id = id(sample_stats_df)
        result_df = normalizer.transform(
            sample_stats_df, season="2023-24", inplace=True
        )

        # Should be same object
        assert id(result_df) == original_id
        assert "pace_z" in sample_stats_df.columns

    def test_transform_creates_copy_by_default(
        self,
        normalizer: SeasonNormalizer,
        sample_stats_df: pd.DataFrame,
    ) -> None:
        """transform() should create copy by default."""
        normalizer.fit(sample_stats_df)

        result_df = normalizer.transform(sample_stats_df, season="2023-24")

        # Should be different object
        assert id(result_df) != id(sample_stats_df)
        assert "pace_z" not in sample_stats_df.columns
        assert "pace_z" in result_df.columns

    def test_custom_metrics_list(self) -> None:
        """Custom metrics list should be used."""
        custom_metrics = ["pace", "efg_pct"]
        normalizer = SeasonNormalizer(metrics=custom_metrics)

        df = pd.DataFrame(
            {
                "season_id": ["2023-24"] * 10,
                "pace": np.random.normal(100, 5, 10),
                "efg_pct": np.random.normal(0.52, 0.05, 10),
                "offensive_rating": np.random.normal(110, 8, 10),  # Not in custom list
            }
        )

        normalizer.fit(df)

        assert ("2023-24", "pace") in normalizer.stats
        assert ("2023-24", "efg_pct") in normalizer.stats
        assert ("2023-24", "offensive_rating") not in normalizer.stats

    def test_normalize_by_season_convenience(self) -> None:
        """normalize_by_season() convenience function should work."""
        df = pd.DataFrame(
            {
                "season_id": ["2023-24"] * 100,
                "pace": np.random.normal(100, 5, 100),
            }
        )

        result = normalize_by_season(df, season="2023-24", metrics=["pace"])

        assert "pace_z" in result.columns
        assert result["pace_z"].mean() == pytest.approx(0, abs=0.01)

    def test_missing_column_skipped(
        self,
        normalizer: SeasonNormalizer,
    ) -> None:
        """Missing metric columns should be skipped without error."""
        df = pd.DataFrame(
            {
                "season_id": ["2023-24"] * 10,
                "pace": np.random.normal(100, 5, 10),
                # Missing other default metrics
            }
        )

        # Should not raise
        normalizer.fit(df)

        assert ("2023-24", "pace") in normalizer.stats
        assert ("2023-24", "offensive_rating") not in normalizer.stats
