"""Two-Tower Fusion architecture for NBA game prediction.

This module implements the fusion model that combines static context features
with dynamic sequence and graph representations to produce multi-task outputs
for game prediction.

Architecture:
    - Tower A (Context): Static features via 2-layer MLP
    - Tower B (Dynamic): Concatenated Transformer + GNN outputs via 2-layer MLP
    - Fusion Layer: Combined tower outputs via 2-layer MLP
    - Multi-task Heads: win_prob (binary), margin (regression), total (regression)

Example:
    >>> from nba_model.models.fusion import TwoTowerFusion, ContextFeatureBuilder
    >>> model = TwoTowerFusion(context_dim=32, transformer_dim=128, gnn_dim=128)
    >>> context = torch.randn(4, 32)
    >>> transformer_out = torch.randn(4, 128)
    >>> gnn_out = torch.randn(4, 128)
    >>> outputs = model(context, transformer_out, gnn_out)
    >>> print(outputs['win_prob'].shape)  # (4, 1)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from nba_model.logging import get_logger
from nba_model.types import GameId, TeamId

if TYPE_CHECKING:
    import pandas as pd
    from sqlalchemy.orm import Session

logger = get_logger(__name__)

# =============================================================================
# Constants
# =============================================================================

DEFAULT_CONTEXT_DIM: int = 32
DEFAULT_TRANSFORMER_DIM: int = 128
DEFAULT_GNN_DIM: int = 128
DEFAULT_HIDDEN_DIM: int = 256
DEFAULT_DROPOUT: float = 0.2

# Context feature indices
CONTEXT_FEATURES: list[str] = [
    # Team stats (home)
    "home_off_rating_z",
    "home_def_rating_z",
    "home_pace_z",
    "home_efg_z",
    "home_tov_z",
    "home_orb_z",
    "home_ft_rate_z",
    # Team stats (away)
    "away_off_rating_z",
    "away_def_rating_z",
    "away_pace_z",
    "away_efg_z",
    "away_tov_z",
    "away_orb_z",
    "away_ft_rate_z",
    # Rest and fatigue
    "home_rest_days",
    "away_rest_days",
    "rest_diff",
    "home_back_to_back",
    "away_back_to_back",
    "home_travel_miles",
    "away_travel_miles",
    # RAPM aggregates
    "home_rapm_sum",
    "away_rapm_sum",
    "rapm_diff",
    "home_orapm_sum",
    "home_drapm_sum",
    "away_orapm_sum",
    "away_drapm_sum",
    # Spacing
    "home_spacing_area",
    "away_spacing_area",
    # Derived
    "is_playoff",
    "is_neutral_site",
]


# =============================================================================
# Data Containers
# =============================================================================


@dataclass
class FusionOutput:
    """Container for fusion model outputs.

    Attributes:
        win_prob: Home win probability (0-1).
        margin: Predicted point margin (home - away).
        total: Predicted total points.
        fusion_embedding: Intermediate fusion layer embedding.
    """

    win_prob: torch.Tensor
    margin: torch.Tensor
    total: torch.Tensor
    fusion_embedding: torch.Tensor


@dataclass
class GameContext:
    """Container for game context features.

    Attributes:
        game_id: NBA game ID.
        home_team_id: Home team ID.
        away_team_id: Away team ID.
        features: Context feature vector.
    """

    game_id: GameId
    home_team_id: TeamId
    away_team_id: TeamId
    features: torch.Tensor


# =============================================================================
# Two-Tower Fusion Model
# =============================================================================


class TwoTowerFusion(nn.Module):
    """Two-Tower fusion architecture for NBA game prediction.

    Combines static context features (Tower A) with dynamic sequence and graph
    representations (Tower B) to produce multi-task prediction outputs.

    Architecture:
        Tower A: context_dim -> 256 -> 256 (ReLU, Dropout)
        Tower B: (transformer_dim + gnn_dim) -> 256 -> 256 (ReLU, Dropout)
        Fusion: 512 -> 256 -> 128 (ReLU, Dropout)
        Heads: Linear projections to 1 output each

    Attributes:
        context_dim: Dimension of static context features.
        transformer_dim: Dimension of Transformer output.
        gnn_dim: Dimension of GNN output.
        hidden_dim: Hidden layer dimension.
        dropout: Dropout probability.

    Example:
        >>> model = TwoTowerFusion()
        >>> context = torch.randn(2, 32)
        >>> transformer_out = torch.randn(2, 128)
        >>> gnn_out = torch.randn(2, 128)
        >>> outputs = model(context, transformer_out, gnn_out)
        >>> print(outputs['win_prob'].shape)  # (2, 1)
    """

    def __init__(
        self,
        context_dim: int = DEFAULT_CONTEXT_DIM,
        transformer_dim: int = DEFAULT_TRANSFORMER_DIM,
        gnn_dim: int = DEFAULT_GNN_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        """Initialize TwoTowerFusion.

        Args:
            context_dim: Dimension of static context features.
            transformer_dim: Dimension of Transformer output.
            gnn_dim: Dimension of GNN output.
            hidden_dim: Hidden layer dimension.
            dropout: Dropout probability.
        """
        super().__init__()
        self.context_dim = context_dim
        self.transformer_dim = transformer_dim
        self.gnn_dim = gnn_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout

        # Tower A: Context features
        self.context_tower = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Tower B: Dynamic features (Transformer + GNN)
        dynamic_input_dim = transformer_dim + gnn_dim
        self.dynamic_tower = nn.Sequential(
            nn.Linear(dynamic_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Fusion layers
        fusion_input_dim = hidden_dim * 2  # Concatenated tower outputs
        fusion_hidden_dim = hidden_dim // 2  # 128
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.ReLU(),
        )

        # Multi-task prediction heads
        self.win_head = nn.Linear(fusion_hidden_dim, 1)
        self.margin_head = nn.Linear(fusion_hidden_dim, 1)
        self.total_head = nn.Linear(fusion_hidden_dim, 1)

        # Initialize weights
        self._init_weights()

        logger.debug(
            "Initialized TwoTowerFusion: context={}, dynamic={}, hidden={}",
            context_dim,
            dynamic_input_dim,
            hidden_dim,
        )

    def _init_weights(self) -> None:
        """Initialize model weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        context: torch.Tensor,
        transformer_out: torch.Tensor,
        gnn_out: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through the fusion model.

        Args:
            context: Static context features of shape (batch, context_dim).
            transformer_out: Transformer sequence output of shape (batch, transformer_dim).
            gnn_out: GNN graph output of shape (batch, gnn_dim).

        Returns:
            Dictionary with keys:
                - 'win_prob': Home win probability (batch, 1), sigmoid applied.
                - 'margin': Predicted margin (batch, 1).
                - 'total': Predicted total points (batch, 1).
                - 'fusion_embedding': Fusion layer output (batch, 128).
        """
        # Tower A: Process context features
        context_out = self.context_tower(context)  # (batch, hidden_dim)

        # Tower B: Process dynamic features
        dynamic_input = torch.cat([transformer_out, gnn_out], dim=-1)
        dynamic_out = self.dynamic_tower(dynamic_input)  # (batch, hidden_dim)

        # Fusion: Combine towers
        fusion_input = torch.cat([context_out, dynamic_out], dim=-1)
        fusion_out = self.fusion(fusion_input)  # (batch, 128)

        # Multi-task heads
        win_logits = self.win_head(fusion_out)  # (batch, 1)
        margin = self.margin_head(fusion_out)  # (batch, 1)
        total = self.total_head(fusion_out)  # (batch, 1)

        # Apply sigmoid to win probability
        win_prob = torch.sigmoid(win_logits)

        return {
            "win_prob": win_prob,
            "margin": margin,
            "total": total,
            "fusion_embedding": fusion_out,
        }

    def get_tower_outputs(
        self,
        context: torch.Tensor,
        transformer_out: torch.Tensor,
        gnn_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get individual tower outputs for interpretability.

        Args:
            context: Static context features.
            transformer_out: Transformer sequence output.
            gnn_out: GNN graph output.

        Returns:
            Tuple of (context_tower_output, dynamic_tower_output).
        """
        context_out = self.context_tower(context)
        dynamic_input = torch.cat([transformer_out, gnn_out], dim=-1)
        dynamic_out = self.dynamic_tower(dynamic_input)
        return context_out, dynamic_out


# =============================================================================
# Context Feature Builder
# =============================================================================


class ContextFeatureBuilder:
    """Build static context feature vectors for Tower A.

    Aggregates team statistics, fatigue indicators, RAPM sums, and spacing
    metrics into a fixed-dimension feature vector for game prediction.

    Features Categories (32 total):
        - Team efficiency (14): ORtg, DRtg, pace, eFG%, TOV%, ORB%, FT rate × 2
        - Fatigue (6): rest days (×2), B2B flags (×2), travel miles (×2)
        - RAPM (7): team sums, diffs
        - Spacing (2): hull areas
        - Other (3): rest diff, playoff flag, neutral site

    Attributes:
        feature_names: List of feature names in order.

    Example:
        >>> builder = ContextFeatureBuilder()
        >>> features = builder.build(game_id="0022300123", db_session=session)
        >>> print(features.shape)  # (32,)
    """

    def __init__(self) -> None:
        """Initialize ContextFeatureBuilder."""
        self.feature_names = CONTEXT_FEATURES

    @property
    def feature_dim(self) -> int:
        """Return the number of context features."""
        return len(self.feature_names)

    def build(
        self,
        game_id: GameId,
        db_session: Session,
    ) -> torch.Tensor:
        """Build context feature vector for a game.

        Args:
            game_id: NBA game ID.
            db_session: SQLAlchemy database session.

        Returns:
            Context feature tensor of shape (32,).
        """
        # This is a placeholder implementation
        # Full implementation would query the database for all features
        logger.debug("Building context features for game {}", game_id)

        # Return zeros for now - full implementation pending
        return torch.zeros(self.feature_dim, dtype=torch.float32)

    def build_from_dict(self, features: dict[str, float]) -> torch.Tensor:
        """Build context feature vector from a dictionary.

        Args:
            features: Dictionary mapping feature names to values.

        Returns:
            Context feature tensor of shape (32,).
        """
        tensor = torch.zeros(self.feature_dim, dtype=torch.float32)

        for i, name in enumerate(self.feature_names):
            if name in features:
                tensor[i] = float(features[name])

        return tensor

    def build_from_dataframe(
        self,
        df: pd.DataFrame,
        game_id: GameId,
    ) -> torch.Tensor:
        """Build context feature vector from a DataFrame row.

        Args:
            df: DataFrame with context features.
            game_id: Game ID to look up.

        Returns:
            Context feature tensor of shape (32,).
        """
        import pandas as pd

        if "game_id" in df.columns:
            row = df[df["game_id"] == game_id]
        else:
            row = df.loc[[game_id]]

        if row.empty:
            logger.warning("No context data for game {}, using zeros", game_id)
            return torch.zeros(self.feature_dim, dtype=torch.float32)

        row = row.iloc[0]
        return self.build_from_dict(row.to_dict())

    def normalize_features(
        self,
        features: torch.Tensor,
        stats: dict[str, tuple[float, float]],
    ) -> torch.Tensor:
        """Normalize features using provided mean/std statistics.

        Args:
            features: Raw feature tensor of shape (..., feature_dim).
            stats: Dictionary mapping feature names to (mean, std) tuples.

        Returns:
            Normalized feature tensor of same shape.
        """
        normalized = features.clone()

        for i, name in enumerate(self.feature_names):
            if name in stats:
                mean, std = stats[name]
                if std > 0:
                    normalized[..., i] = (features[..., i] - mean) / std

        return normalized


# =============================================================================
# Multi-Task Loss
# =============================================================================


class MultiTaskLoss(nn.Module):
    """Multi-task loss function for fusion model.

    Combines:
        - Binary cross-entropy for win probability
        - Huber loss for margin prediction
        - Huber loss for total points prediction

    Loss weights can be fixed or learned.

    Attributes:
        win_weight: Weight for win probability loss.
        margin_weight: Weight for margin loss.
        total_weight: Weight for total points loss.
        huber_delta: Delta parameter for Huber loss.

    Example:
        >>> loss_fn = MultiTaskLoss()
        >>> outputs = model(context, transformer_out, gnn_out)
        >>> labels = {'win': win_labels, 'margin': margin_labels, 'total': total_labels}
        >>> loss, components = loss_fn(outputs, labels)
    """

    def __init__(
        self,
        win_weight: float = 1.0,
        margin_weight: float = 1.0,
        total_weight: float = 1.0,
        huber_delta: float = 1.0,
        learnable_weights: bool = False,
    ) -> None:
        """Initialize MultiTaskLoss.

        Args:
            win_weight: Weight for win probability loss.
            margin_weight: Weight for margin prediction loss.
            total_weight: Weight for total points loss.
            huber_delta: Delta parameter for Huber loss.
            learnable_weights: If True, weights are learnable log-variances.
        """
        super().__init__()
        self.huber_delta = huber_delta

        if learnable_weights:
            # Learnable task weights using uncertainty weighting
            # log_var = log(sigma^2), weight = 1 / (2 * sigma^2)
            self.log_var_win = nn.Parameter(torch.zeros(1))
            self.log_var_margin = nn.Parameter(torch.zeros(1))
            self.log_var_total = nn.Parameter(torch.zeros(1))
            self.learnable = True
        else:
            self.register_buffer("win_weight", torch.tensor(win_weight))
            self.register_buffer("margin_weight", torch.tensor(margin_weight))
            self.register_buffer("total_weight", torch.tensor(total_weight))
            self.learnable = False

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute multi-task loss.

        Args:
            outputs: Model outputs with keys 'win_prob', 'margin', 'total'.
            labels: Labels with keys 'win', 'margin', 'total'.

        Returns:
            Tuple of (total_loss, components_dict).
            components_dict contains individual loss values for logging.
        """
        # Binary cross-entropy for win probability
        # Clamp to avoid log(0)
        win_prob = outputs["win_prob"].clamp(1e-7, 1 - 1e-7)
        win_loss = F.binary_cross_entropy(
            win_prob,
            labels["win"].float().view_as(win_prob),
        )

        # Huber loss for margin
        margin_loss = F.huber_loss(
            outputs["margin"],
            labels["margin"].view_as(outputs["margin"]),
            delta=self.huber_delta,
        )

        # Huber loss for total
        total_loss = F.huber_loss(
            outputs["total"],
            labels["total"].view_as(outputs["total"]),
            delta=self.huber_delta,
        )

        # Compute weighted sum
        if self.learnable:
            # Uncertainty weighting: L = L_i / (2 * sigma^2) + log(sigma)
            precision_win = torch.exp(-self.log_var_win)
            precision_margin = torch.exp(-self.log_var_margin)
            precision_total = torch.exp(-self.log_var_total)

            combined_loss = (
                precision_win * win_loss
                + 0.5 * self.log_var_win
                + precision_margin * margin_loss
                + 0.5 * self.log_var_margin
                + precision_total * total_loss
                + 0.5 * self.log_var_total
            )
        else:
            combined_loss = (
                self.win_weight * win_loss
                + self.margin_weight * margin_loss
                + self.total_weight * total_loss
            )

        components = {
            "win_loss": win_loss.item(),
            "margin_loss": margin_loss.item(),
            "total_loss": total_loss.item(),
            "combined_loss": combined_loss.item(),
        }

        return combined_loss, components


# =============================================================================
# Utility Functions
# =============================================================================


def create_dummy_inputs(
    batch_size: int = 2,
    context_dim: int = DEFAULT_CONTEXT_DIM,
    transformer_dim: int = DEFAULT_TRANSFORMER_DIM,
    gnn_dim: int = DEFAULT_GNN_DIM,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create dummy inputs for testing the fusion model.

    Args:
        batch_size: Number of samples in batch.
        context_dim: Dimension of context features.
        transformer_dim: Dimension of Transformer output.
        gnn_dim: Dimension of GNN output.

    Returns:
        Tuple of (context, transformer_out, gnn_out) tensors.
    """
    context = torch.randn(batch_size, context_dim)
    transformer_out = torch.randn(batch_size, transformer_dim)
    gnn_out = torch.randn(batch_size, gnn_dim)
    return context, transformer_out, gnn_out


def create_dummy_labels(
    batch_size: int = 2,
) -> dict[str, torch.Tensor]:
    """Create dummy labels for testing loss computation.

    Args:
        batch_size: Number of samples in batch.

    Returns:
        Dictionary with 'win', 'margin', 'total' label tensors.
    """
    return {
        "win": torch.randint(0, 2, (batch_size, 1)).float(),
        "margin": torch.randn(batch_size, 1) * 10,  # ~10 point std
        "total": torch.randn(batch_size, 1) * 15 + 220,  # ~220 avg
    }
