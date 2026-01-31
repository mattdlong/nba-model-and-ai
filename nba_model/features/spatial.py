"""Convex Hull spacing calculator for lineup floor analysis.

This module implements spatial analysis of shot distributions to measure
lineup floor spacing. Uses convex hulls to quantify the area covered
by a lineup's shot locations.

Court Coordinate System:
    - LOC_X range: approximately -250 to 250 (tenths of feet from basket center)
    - LOC_Y range: approximately -50 to 900
    - Basket is at (0, 0)
    - Positive X is to the right of basket (from shooter perspective)
    - Positive Y is away from basket toward halfcourt

Example:
    >>> from nba_model.features.spatial import SpacingCalculator
    >>> calculator = SpacingCalculator()
    >>> metrics = calculator.calculate_lineup_spacing([1, 2, 3, 4, 5], shots_df)
    >>> print(metrics['hull_area'])
    550.0
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.stats import gaussian_kde

from nba_model.logging import get_logger
from nba_model.types import PlayerId, SpacingMetrics

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)

# Court dimensions in tenths of feet (NBA API units)
COURT_WIDTH = 500  # -250 to 250
COURT_LENGTH = 940  # -47 to 893 (full court)
HALF_COURT_LENGTH = 470  # Approximately

# Corner three zones (approximate coordinates)
LEFT_CORNER_X_MIN = -250
LEFT_CORNER_X_MAX = -220
RIGHT_CORNER_X_MIN = 220
RIGHT_CORNER_X_MAX = 250
CORNER_Y_MAX = 90  # Within this Y range for corner

# Minimum shots required for calculations
MIN_SHOTS_FOR_HULL = 4  # Need at least 4 non-collinear points
MIN_SHOTS_FOR_KDE = 10
MIN_SHOTS_FOR_LINEUP = 20  # Default minimum for lineup spacing

# Grid resolution for density maps
GRID_RESOLUTION = 50


@dataclass
class LineupSpacingResult:
    """Container for lineup spacing calculation results."""

    hull_area: float
    centroid_x: float
    centroid_y: float
    avg_distance: float
    corner_density: float
    shot_count: int
    lineup_hash: str


class SpacingCalculator:
    """Convex hull spacing calculator for lineup floor geometry analysis.

    Calculates floor spacing metrics by analyzing shot locations for player
    lineups. Uses convex hulls to measure the area covered by shots and
    KDE for density analysis.

    Attributes:
        min_shots: Minimum shots required for lineup calculation.
        grid_resolution: Resolution for KDE density maps.

    Example:
        >>> calculator = SpacingCalculator(min_shots=20)
        >>> metrics = calculator.calculate_lineup_spacing(player_ids, shots_df)
        >>> print(f"Hull area: {metrics['hull_area']:.1f} sq units")
    """

    def __init__(
        self,
        min_shots: int = MIN_SHOTS_FOR_LINEUP,
        grid_resolution: int = GRID_RESOLUTION,
    ) -> None:
        """Initialize spacing calculator.

        Args:
            min_shots: Minimum shots required for lineup spacing calculation.
            grid_resolution: Grid resolution for KDE density maps.
        """
        self.min_shots = min_shots
        self.grid_resolution = grid_resolution

    @staticmethod
    def compute_lineup_hash(player_ids: list[PlayerId]) -> str:
        """Compute deterministic hash for a lineup.

        Sorts player IDs to ensure same hash regardless of order.

        Args:
            player_ids: List of 5 player IDs.

        Returns:
            SHA256 hash string (first 16 chars).
        """
        sorted_ids = sorted(player_ids)
        id_string = ",".join(str(pid) for pid in sorted_ids)
        return hashlib.sha256(id_string.encode()).hexdigest()[:16]

    def _get_player_shots(
        self,
        player_ids: list[PlayerId],
        shots_df: pd.DataFrame,
    ) -> np.ndarray:
        """Get shot locations for a set of players.

        Args:
            player_ids: List of player IDs.
            shots_df: DataFrame with columns player_id, loc_x, loc_y.

        Returns:
            Array of shape (n_shots, 2) with [loc_x, loc_y] coordinates.
        """
        player_set = set(player_ids)
        mask = shots_df["player_id"].isin(player_set)
        player_shots = shots_df.loc[mask, ["loc_x", "loc_y"]].values
        return player_shots.astype(np.float64)

    def _calculate_hull_area(self, points: np.ndarray) -> float:
        """Calculate convex hull area from points.

        Handles edge cases:
        - Fewer than 3 points: returns 0
        - Collinear points: returns 0
        - Uses QhullError handling for degenerate cases

        Args:
            points: Array of shape (n_points, 2).

        Returns:
            Convex hull area in square units.
        """
        if len(points) < 3:
            return 0.0

        try:
            hull = ConvexHull(points)
            return float(hull.volume)  # In 2D, volume is area
        except QhullError:
            # Degenerate case (collinear points)
            logger.debug("Collinear points detected, hull area = 0")
            return 0.0

    def _calculate_centroid(self, points: np.ndarray) -> tuple[float, float]:
        """Calculate centroid of shot distribution.

        Args:
            points: Array of shape (n_points, 2).

        Returns:
            Tuple of (centroid_x, centroid_y).
        """
        if len(points) == 0:
            return 0.0, 0.0
        return float(np.mean(points[:, 0])), float(np.mean(points[:, 1]))

    def _calculate_avg_distance(self, points: np.ndarray) -> float:
        """Calculate average distance from basket.

        Basket is at (0, 0) in NBA API coordinates.

        Args:
            points: Array of shape (n_points, 2).

        Returns:
            Average Euclidean distance from basket.
        """
        if len(points) == 0:
            return 0.0
        distances = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
        return float(np.mean(distances))

    def _calculate_corner_density(self, points: np.ndarray) -> float:
        """Calculate proportion of shots from corner three zones.

        Corner zones defined as:
        - Left corner: x < -220, y < 90
        - Right corner: x > 220, y < 90

        Args:
            points: Array of shape (n_points, 2).

        Returns:
            Proportion of shots in corner zones (0 to 1).
        """
        if len(points) == 0:
            return 0.0

        left_corner = (points[:, 0] < LEFT_CORNER_X_MAX) & (points[:, 1] < CORNER_Y_MAX)
        right_corner = (points[:, 0] > RIGHT_CORNER_X_MIN) & (
            points[:, 1] < CORNER_Y_MAX
        )

        corner_shots = np.sum(left_corner | right_corner)
        return float(corner_shots / len(points))

    def calculate_lineup_spacing(
        self,
        player_ids: list[PlayerId],
        shots_df: pd.DataFrame,
    ) -> SpacingMetrics:
        """Calculate spacing metrics for a 5-player lineup.

        Aggregates shot locations for all players in the lineup and
        computes convex hull and distribution metrics.

        Args:
            player_ids: List of 5 player IDs.
            shots_df: DataFrame with columns player_id, loc_x, loc_y.

        Returns:
            SpacingMetrics TypedDict with keys:
                - hull_area: Convex hull area in square units
                - centroid_x: X coordinate of shot distribution centroid
                - centroid_y: Y coordinate of shot distribution centroid
                - avg_distance: Average distance from basket
                - corner_density: Proportion of corner three shots
                - shot_count: Number of shots used in calculation
        """
        # Get shots for lineup
        shots = self._get_player_shots(player_ids, shots_df)
        shot_count = len(shots)

        if shot_count < MIN_SHOTS_FOR_HULL:
            logger.debug(
                "Insufficient shots ({}) for lineup {}, returning zeros",
                shot_count,
                player_ids,
            )
            return SpacingMetrics(
                hull_area=0.0,
                centroid_x=0.0,
                centroid_y=0.0,
                avg_distance=0.0,
                corner_density=0.0,
                shot_count=shot_count,
            )

        # Calculate metrics
        hull_area = self._calculate_hull_area(shots)
        centroid_x, centroid_y = self._calculate_centroid(shots)
        avg_distance = self._calculate_avg_distance(shots)
        corner_density = self._calculate_corner_density(shots)

        return SpacingMetrics(
            hull_area=round(hull_area, 2),
            centroid_x=round(centroid_x, 2),
            centroid_y=round(centroid_y, 2),
            avg_distance=round(avg_distance, 2),
            corner_density=round(corner_density, 4),
            shot_count=shot_count,
        )

    def calculate_player_shot_density(
        self,
        player_id: PlayerId,
        shots_df: pd.DataFrame,
    ) -> np.ndarray:
        """Create KDE-based shot density map for a player.

        Generates a 2D probability density map of shot locations
        using Gaussian kernel density estimation.

        Args:
            player_id: Player ID.
            shots_df: DataFrame with columns player_id, loc_x, loc_y.

        Returns:
            2D array of shape (grid_resolution, grid_resolution) with
            density values. Returns zeros if insufficient shots.
        """
        # Get player shots
        mask = shots_df["player_id"] == player_id
        player_shots = shots_df.loc[mask, ["loc_x", "loc_y"]].values.astype(np.float64)

        if len(player_shots) < MIN_SHOTS_FOR_KDE:
            logger.debug(
                "Insufficient shots ({}) for KDE for player {}",
                len(player_shots),
                player_id,
            )
            return np.zeros((self.grid_resolution, self.grid_resolution))

        # Create grid
        x_min, x_max = -250, 250
        y_min, y_max = -50, 400  # Focus on half court
        xx, yy = np.mgrid[
            x_min : x_max : complex(0, self.grid_resolution),
            y_min : y_max : complex(0, self.grid_resolution),
        ]
        positions = np.vstack([xx.ravel(), yy.ravel()])

        try:
            # Fit KDE
            kernel = gaussian_kde(player_shots.T)
            density = np.reshape(kernel(positions).T, xx.shape)
            return density
        except (np.linalg.LinAlgError, ValueError) as e:
            logger.debug("KDE failed for player {}: {}", player_id, e)
            return np.zeros((self.grid_resolution, self.grid_resolution))

    def calculate_lineup_gravity_overlap(
        self,
        player_ids: list[PlayerId],
        shots_df: pd.DataFrame,
    ) -> float:
        """Calculate overlapping gravity zones between teammates.

        High overlap indicates poor spacing where players shoot from
        similar locations. Low overlap indicates good spacing variety.

        Uses KDE overlap as metric: sum of min(density_i, density_j)
        for each pair of players.

        Args:
            player_ids: List of player IDs.
            shots_df: DataFrame with shot data.

        Returns:
            Overlap score normalized to [0, 1]. Higher = more overlap.
        """
        # Calculate density maps for each player
        densities = []
        for player_id in player_ids:
            density = self.calculate_player_shot_density(player_id, shots_df)
            if np.any(density > 0):
                densities.append(density)

        if len(densities) < 2:
            return 0.0

        # Calculate pairwise overlap
        total_overlap = 0.0
        n_pairs = 0

        for i in range(len(densities)):
            for j in range(i + 1, len(densities)):
                # Overlap is integral of min(f, g)
                overlap = np.sum(np.minimum(densities[i], densities[j]))
                total_overlap += overlap
                n_pairs += 1

        if n_pairs == 0:
            return 0.0

        # Normalize by average self-overlap
        avg_self_overlap = sum(np.sum(d) for d in densities) / len(densities)
        if avg_self_overlap == 0:
            return 0.0

        normalized_overlap = (total_overlap / n_pairs) / avg_self_overlap
        return float(min(normalized_overlap, 1.0))

    def calculate_all_lineup_spacing(
        self,
        lineups: list[list[PlayerId]],
        shots_df: pd.DataFrame,
    ) -> dict[str, SpacingMetrics]:
        """Calculate spacing metrics for multiple lineups.

        Args:
            lineups: List of 5-player lineups.
            shots_df: DataFrame with shot data.

        Returns:
            Dict mapping lineup_hash to SpacingMetrics.
        """
        results: dict[str, SpacingMetrics] = {}

        for lineup in lineups:
            lineup_hash = self.compute_lineup_hash(lineup)
            metrics = self.calculate_lineup_spacing(lineup, shots_df)
            results[lineup_hash] = metrics

        logger.info("Calculated spacing for {} lineups", len(results))
        return results


def calculate_lineup_spacing(
    player_ids: list[PlayerId],
    shots_df: pd.DataFrame,
    min_shots: int = MIN_SHOTS_FOR_LINEUP,
) -> SpacingMetrics:
    """Convenience function to calculate spacing for a single lineup.

    Args:
        player_ids: List of 5 player IDs.
        shots_df: DataFrame with shot data.
        min_shots: Minimum shots required.

    Returns:
        SpacingMetrics for the lineup.
    """
    calculator = SpacingCalculator(min_shots=min_shots)
    return calculator.calculate_lineup_spacing(player_ids, shots_df)


__all__ = [
    "MIN_SHOTS_FOR_HULL",
    "MIN_SHOTS_FOR_LINEUP",
    "LineupSpacingResult",
    "SpacingCalculator",
    "calculate_lineup_spacing",
]
