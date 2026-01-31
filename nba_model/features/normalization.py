"""Season normalization for cross-season statistical comparability.

This module implements z-score normalization by season to handle
pace and space evolution in the NBA. Features are normalized against
season-specific means and standard deviations.

Formula: z = (x - μ_season) / σ_season

Example:
    >>> from nba_model.features.normalization import SeasonNormalizer
    >>> normalizer = SeasonNormalizer()
    >>> normalizer.fit(games_df)
    >>> normalized_df = normalizer.transform(df, season="2023-24")
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from nba_model.logging import get_logger
from nba_model.types import SeasonId

if TYPE_CHECKING:
    import pandas as pd
    from sqlalchemy.orm import Session

logger = get_logger(__name__)


# Metrics to normalize by default (per Phase 3 spec)
DEFAULT_METRICS_TO_NORMALIZE: list[str] = [
    "pace",
    "offensive_rating",
    "defensive_rating",
    "efg_pct",
    "tov_pct",
    "orb_pct",
    "ft_rate",
    "fg3a_rate",
    "points_per_game",
]

# Extended list including game-level metrics
ALL_NORMALIZABLE_METRICS: list[str] = [
    "pace",
    "offensive_rating",
    "defensive_rating",
    "efg_pct",
    "tov_pct",
    "orb_pct",
    "ft_rate",
    "fg3a_rate",
    "points_per_game",
    "rebounds_per_game",
    "assists_per_game",
    "steals_per_game",
    "blocks_per_game",
    "turnovers_per_game",
]


@dataclass
class MetricStats:
    """Statistics for a single metric within a season."""

    metric_name: str
    season_id: SeasonId
    mean_value: float
    std_value: float
    min_value: float
    max_value: float


class SeasonNormalizer:
    """Z-score normalizer by season for cross-season comparability.

    Calculates and applies z-score normalization using season-specific
    statistics. This accounts for league-wide changes in pace, three-point
    shooting, etc. over time.

    Attributes:
        metrics: List of metric names to normalize.
        stats: Dict mapping (season_id, metric_name) to MetricStats.
        fitted_: Whether the normalizer has been fitted.

    Example:
        >>> normalizer = SeasonNormalizer()
        >>> normalizer.fit(game_stats_df)
        >>> normalized = normalizer.transform(df, season="2023-24")
    """

    def __init__(
        self,
        metrics: list[str] | None = None,
    ) -> None:
        """Initialize normalizer.

        Args:
            metrics: List of metric names to normalize. Uses defaults if None.
        """
        self.metrics = metrics or DEFAULT_METRICS_TO_NORMALIZE
        self.stats: dict[tuple[SeasonId, str], MetricStats] = {}
        self.fitted_ = False

    def fit(
        self,
        data_df: pd.DataFrame,
        season_column: str = "season_id",
    ) -> SeasonNormalizer:
        """Calculate mean and std for each metric by season.

        Args:
            data_df: DataFrame with season and metric columns.
            season_column: Name of season identifier column.

        Returns:
            Self for method chaining.
        """
        if season_column not in data_df.columns:
            raise ValueError(f"Column '{season_column}' not found in DataFrame")

        seasons = data_df[season_column].unique()
        logger.info("Fitting normalizer for {} seasons", len(seasons))

        for season in seasons:
            season_data = data_df[data_df[season_column] == season]

            for metric in self.metrics:
                if metric not in season_data.columns:
                    logger.debug("Metric '{}' not found, skipping", metric)
                    continue

                values = season_data[metric].dropna()
                if len(values) == 0:
                    continue

                stats = MetricStats(
                    metric_name=metric,
                    season_id=str(season),
                    mean_value=float(values.mean()),
                    std_value=float(values.std()) if len(values) > 1 else 1.0,
                    min_value=float(values.min()),
                    max_value=float(values.max()),
                )

                # Prevent division by zero
                if stats.std_value == 0:
                    stats.std_value = 1.0

                self.stats[(str(season), metric)] = stats
                logger.debug(
                    "Season {}, {}: mean={:.3f}, std={:.3f}",
                    season,
                    metric,
                    stats.mean_value,
                    stats.std_value,
                )

        self.fitted_ = True
        logger.info(
            "Fitted normalizer with {} metric-season combinations", len(self.stats)
        )
        return self

    def fit_from_game_stats(
        self,
        game_stats_df: pd.DataFrame,
        games_df: pd.DataFrame,
    ) -> SeasonNormalizer:
        """Fit from game stats DataFrame joined with games.

        Args:
            game_stats_df: DataFrame with team game statistics.
            games_df: DataFrame with game_id and season_id.

        Returns:
            Self for method chaining.
        """

        # Merge to get season_id
        merged = game_stats_df.merge(
            games_df[["game_id", "season_id"]],
            on="game_id",
            how="left",
        )

        return self.fit(merged, season_column="season_id")

    def transform(
        self,
        data_df: pd.DataFrame,
        season: SeasonId,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """Apply z-score normalization using stored stats.

        Args:
            data_df: DataFrame with metric columns.
            season: Season ID for normalization stats.
            inplace: If True, modify DataFrame in place.

        Returns:
            DataFrame with normalized metric columns (suffixed with '_z').

        Raises:
            ValueError: If normalizer not fitted or season not found.
        """
        if not self.fitted_:
            raise ValueError("Normalizer has not been fitted. Call fit() first.")

        result = data_df if inplace else data_df.copy()

        # Verify season exists in fit data before transforming any metrics
        season_keys = [k for k in self.stats.keys() if k[0] == str(season)]
        if not season_keys:
            raise ValueError(
                f"Season '{season}' not found in fit data. "
                f"Available seasons: {sorted(set(k[0] for k in self.stats.keys()))}"
            )

        for metric in self.metrics:
            if metric not in result.columns:
                continue

            key = (str(season), metric)
            if key not in self.stats:
                logger.warning(
                    "No stats for season '{}', metric '{}'. Skipping.",
                    season,
                    metric,
                )
                continue

            stats = self.stats[key]
            z_column = f"{metric}_z"
            result[z_column] = (result[metric] - stats.mean_value) / stats.std_value

        return result

    def transform_value(
        self,
        value: float,
        metric: str,
        season: SeasonId,
    ) -> float:
        """Transform a single value.

        Args:
            value: Raw metric value.
            metric: Metric name.
            season: Season ID.

        Returns:
            Normalized z-score value.

        Raises:
            ValueError: If normalizer not fitted or metric/season not found.
        """
        if not self.fitted_:
            raise ValueError("Normalizer has not been fitted. Call fit() first.")

        key = (str(season), metric)
        if key not in self.stats:
            raise ValueError(f"No stats for season '{season}', metric '{metric}'")

        stats = self.stats[key]
        return (value - stats.mean_value) / stats.std_value

    def inverse_transform_value(
        self,
        z_value: float,
        metric: str,
        season: SeasonId,
    ) -> float:
        """Inverse transform a z-score back to original scale.

        Args:
            z_value: Normalized z-score value.
            metric: Metric name.
            season: Season ID.

        Returns:
            Original scale value.
        """
        if not self.fitted_:
            raise ValueError("Normalizer has not been fitted. Call fit() first.")

        key = (str(season), metric)
        if key not in self.stats:
            raise ValueError(f"No stats for season '{season}', metric '{metric}'")

        stats = self.stats[key]
        return z_value * stats.std_value + stats.mean_value

    def get_season_stats(self, season: SeasonId) -> dict[str, MetricStats]:
        """Get all stats for a season.

        Args:
            season: Season ID.

        Returns:
            Dict mapping metric name to MetricStats.
        """
        result = {}
        for (s, metric), stats in self.stats.items():
            if s == str(season):
                result[metric] = stats
        return result

    def save_stats(
        self,
        session: Session,
    ) -> int:
        """Persist normalization stats to season_stats table.

        Args:
            session: SQLAlchemy session.

        Returns:
            Number of records saved.
        """
        from nba_model.data.models import SeasonStats

        count = 0
        for (season_id, metric_name), stats in self.stats.items():
            # Check for existing record
            existing = (
                session.query(SeasonStats)
                .filter_by(season_id=season_id, metric_name=metric_name)
                .first()
            )

            if existing:
                existing.mean_value = stats.mean_value
                existing.std_value = stats.std_value
                existing.min_value = stats.min_value
                existing.max_value = stats.max_value
            else:
                record = SeasonStats(
                    season_id=season_id,
                    metric_name=metric_name,
                    mean_value=stats.mean_value,
                    std_value=stats.std_value,
                    min_value=stats.min_value,
                    max_value=stats.max_value,
                )
                session.add(record)
            count += 1

        session.commit()
        logger.info("Saved {} normalization stats to database", count)
        return count

    def load_stats(
        self,
        session: Session,
        season: SeasonId | None = None,
    ) -> SeasonNormalizer:
        """Load normalization stats from database.

        Args:
            session: SQLAlchemy session.
            season: Optional season to filter by. Loads all if None.

        Returns:
            Self for method chaining.
        """
        from nba_model.data.models import SeasonStats

        query = session.query(SeasonStats)
        if season is not None:
            query = query.filter_by(season_id=str(season))

        records = query.all()
        logger.info("Loading {} normalization stats from database", len(records))

        self.stats.clear()
        for record in records:
            stats = MetricStats(
                metric_name=record.metric_name,
                season_id=record.season_id,
                mean_value=record.mean_value,
                std_value=record.std_value,
                min_value=record.min_value,
                max_value=record.max_value,
            )
            self.stats[(record.season_id, record.metric_name)] = stats

        self.fitted_ = len(self.stats) > 0
        return self

    def save(self, path: Path) -> None:
        """Save normalizer state to JSON file.

        Args:
            path: Path to save file.
        """
        if not self.fitted_:
            raise ValueError("Normalizer has not been fitted. Call fit() first.")

        state = {
            "metrics": self.metrics,
            "stats": {
                f"{season}|{metric}": {
                    "metric_name": s.metric_name,
                    "season_id": s.season_id,
                    "mean_value": s.mean_value,
                    "std_value": s.std_value,
                    "min_value": s.min_value,
                    "max_value": s.max_value,
                }
                for (season, metric), s in self.stats.items()
            },
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

        logger.info("Saved normalizer to {}", path)

    def load(self, path: Path) -> SeasonNormalizer:
        """Load normalizer state from JSON file.

        Args:
            path: Path to saved state file.

        Returns:
            Self for method chaining.
        """
        with open(path) as f:
            state = json.load(f)

        self.metrics = state["metrics"]
        self.stats = {}

        for key, s in state["stats"].items():
            season, metric = key.split("|", 1)
            self.stats[(season, metric)] = MetricStats(
                metric_name=s["metric_name"],
                season_id=s["season_id"],
                mean_value=s["mean_value"],
                std_value=s["std_value"],
                min_value=s["min_value"],
                max_value=s["max_value"],
            )

        self.fitted_ = True
        logger.info("Loaded normalizer from {}", path)
        return self


def normalize_by_season(
    data_df: pd.DataFrame,
    season: SeasonId,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Convenience function to fit and transform in one step.

    Note: This fits on the provided data. For proper usage, fit on
    historical data and transform on new data.

    Args:
        data_df: DataFrame with metric and season_id columns.
        season: Season to normalize for.
        metrics: List of metrics to normalize.

    Returns:
        DataFrame with normalized columns.
    """
    normalizer = SeasonNormalizer(metrics=metrics)
    normalizer.fit(data_df)
    return normalizer.transform(data_df, season)


__all__ = [
    "ALL_NORMALIZABLE_METRICS",
    "DEFAULT_METRICS_TO_NORMALIZE",
    "MetricStats",
    "SeasonNormalizer",
    "normalize_by_season",
]
