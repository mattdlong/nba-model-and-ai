"""Graph Neural Network for NBA player interaction modeling.

This module implements a GATv2-based graph neural network that models player
interactions within lineup matchups. The graph structure captures both teammate
and opponent relationships.

Architecture Specifications:
    - Node features: 16 input dimensions
    - Hidden dimension: 64
    - Output dimension: 128
    - Attention heads: 4
    - GATv2 layers: 2
    - Dropout rate: 0.1

Graph Structure:
    - Nodes: 10 players (5 home, 5 away)
    - Teammate edges: Fully connected within each team (20 edges)
    - Opponent edges: Cross-team connections (25 edges for full connectivity)

Example:
    >>> from nba_model.models.gnn import PlayerInteractionGNN, LineupGraphBuilder
    >>> model = PlayerInteractionGNN()
    >>> builder = LineupGraphBuilder(player_features_df)
    >>> graph = builder.build_graph(home_lineup, away_lineup)
    >>> output = model(graph)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
import numpy as np

from nba_model.logging import get_logger
from nba_model.types import PlayerId, TeamId

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)

# =============================================================================
# Constants
# =============================================================================

DEFAULT_NODE_FEATURES: int = 16
DEFAULT_HIDDEN_DIM: int = 64
DEFAULT_OUTPUT_DIM: int = 128
DEFAULT_NUM_HEADS: int = 4
DEFAULT_NUM_LAYERS: int = 2
DEFAULT_DROPOUT: float = 0.1

# Node feature indices
FEATURE_ORAPM: int = 0
FEATURE_DRAPM: int = 1
FEATURE_TOTAL_RAPM: int = 2
FEATURE_HEIGHT: int = 3
FEATURE_WEIGHT: int = 4
FEATURE_POS_PG: int = 5
FEATURE_POS_SG: int = 6
FEATURE_POS_SF: int = 7
FEATURE_POS_PF: int = 8
FEATURE_POS_C: int = 9
FEATURE_FG_PCT: int = 10
FEATURE_FG3_PCT: int = 11
FEATURE_FT_PCT: int = 12
FEATURE_MPG: int = 13
FEATURE_USG_PCT: int = 14
FEATURE_IS_HOME: int = 15

# Number of players per team
PLAYERS_PER_TEAM: int = 5
TOTAL_PLAYERS: int = 10


# =============================================================================
# Data Containers
# =============================================================================


@dataclass
class PlayerFeatures:
    """Container for player feature data.

    Attributes:
        player_id: NBA player ID.
        orapm: Offensive RAPM coefficient.
        drapm: Defensive RAPM coefficient.
        height: Normalized height (z-score).
        weight: Normalized weight (z-score).
        position: Primary position (PG, SG, SF, PF, C).
        fg_pct: Field goal percentage.
        fg3_pct: Three-point percentage.
        ft_pct: Free throw percentage.
        mpg: Minutes per game.
        usg_pct: Usage percentage.
    """

    player_id: PlayerId
    orapm: float = 0.0
    drapm: float = 0.0
    height: float = 0.0
    weight: float = 0.0
    position: str = "SF"
    fg_pct: float = 0.45
    fg3_pct: float = 0.35
    ft_pct: float = 0.75
    mpg: float = 20.0
    usg_pct: float = 0.20

    def to_tensor(self, is_home: bool = True) -> torch.Tensor:
        """Convert to feature tensor.

        Args:
            is_home: Whether player is on home team.

        Returns:
            Tensor of shape (16,) with all features.
        """
        # Position one-hot encoding
        pos_encoding = [0.0] * 5
        pos_map = {"PG": 0, "SG": 1, "SF": 2, "PF": 3, "C": 4}
        pos_idx = pos_map.get(self.position.upper(), 2)  # Default to SF
        pos_encoding[pos_idx] = 1.0

        features = [
            self.orapm,
            self.drapm,
            self.orapm + self.drapm,  # Total RAPM
            self.height,
            self.weight,
            *pos_encoding,
            self.fg_pct,
            self.fg3_pct,
            self.ft_pct,
            self.mpg / 48.0,  # Normalize to [0, 1]
            self.usg_pct,
            1.0 if is_home else 0.0,
        ]

        return torch.tensor(features, dtype=torch.float32)


# =============================================================================
# GNN Model
# =============================================================================


class PlayerInteractionGNN(nn.Module):
    """GATv2-based GNN for modeling player interactions.

    Uses Graph Attention Network v2 (GATv2Conv) to learn player representations
    that capture teammate synergies and opponent matchups.

    Architecture:
        - Input projection layer
        - Multiple GATv2 layers with multi-head attention
        - Global pooling (mean + max concatenation)
        - Output projection layer

    Attributes:
        node_features: Input feature dimension per node.
        hidden_dim: Hidden layer dimension.
        output_dim: Output representation dimension.
        num_heads: Number of attention heads.
        num_layers: Number of GATv2 layers.
        dropout: Dropout probability.

    Example:
        >>> model = PlayerInteractionGNN(node_features=16, output_dim=128)
        >>> x = torch.randn(10, 16)  # 10 players, 16 features
        >>> edge_index = torch.tensor([[0,1,2], [1,2,0]])  # Some edges
        >>> data = Data(x=x, edge_index=edge_index)
        >>> output = model(data)  # (128,)
    """

    def __init__(
        self,
        node_features: int = DEFAULT_NODE_FEATURES,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        output_dim: int = DEFAULT_OUTPUT_DIM,
        num_heads: int = DEFAULT_NUM_HEADS,
        num_layers: int = DEFAULT_NUM_LAYERS,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        """Initialize PlayerInteractionGNN.

        Args:
            node_features: Number of input features per player node.
            hidden_dim: Hidden layer dimension.
            output_dim: Output representation dimension.
            num_heads: Number of attention heads in GATv2.
            num_layers: Number of GATv2 layers.
            dropout: Dropout probability.
        """
        super().__init__()
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        # Input projection
        self.input_proj = nn.Linear(node_features, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # GATv2 layers
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for i in range(num_layers):
            in_channels = hidden_dim if i == 0 else hidden_dim * num_heads
            out_channels = hidden_dim

            # Concat attention heads for all but last layer
            concat = i < num_layers - 1

            self.gat_layers.append(
                GATv2Conv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    heads=num_heads,
                    dropout=dropout,
                    concat=concat,
                    add_self_loops=True,
                    share_weights=False,
                )
            )

            # Layer norm after each GAT layer
            norm_dim = hidden_dim * num_heads if concat else hidden_dim
            self.layer_norms.append(nn.LayerNorm(norm_dim))

        # Output projection
        # After last GAT layer: hidden_dim (no concat)
        # After pooling: hidden_dim * 2 (mean + max)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.output_norm = nn.LayerNorm(output_dim)

        logger.debug(
            "Initialized PlayerInteractionGNN: nodes={}, hidden={}, out={}, heads={}",
            node_features,
            hidden_dim,
            output_dim,
            num_heads,
        )

    def forward(self, data: Data | Batch) -> torch.Tensor:
        """Forward pass through the GNN.

        Args:
            data: PyG Data or Batch object with:
                - x: Node features of shape (num_nodes, node_features).
                - edge_index: Edge connectivity of shape (2, num_edges).
                - batch: (Optional) Batch assignment for batched graphs.

        Returns:
            Graph-level representation of shape (batch_size, output_dim).
            If single graph, shape is (1, output_dim).
        """
        x = data.x
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else None

        # Input projection
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # GATv2 layers
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            x_residual = x
            x = gat(x, edge_index)
            x = norm(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # Residual connection if dimensions match
            if x.size(-1) == x_residual.size(-1):
                x = x + x_residual

        # Global pooling: combine mean and max
        if batch is None:
            # Single graph - create dummy batch
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x_mean = global_mean_pool(x, batch)  # (batch_size, hidden_dim)
        x_max = global_max_pool(x, batch)  # (batch_size, hidden_dim)
        x = torch.cat([x_mean, x_max], dim=-1)  # (batch_size, hidden_dim * 2)

        # Output projection
        x = self.output_proj(x)
        x = self.output_norm(x)

        return x

    def get_node_embeddings(self, data: Data) -> torch.Tensor:
        """Get node-level embeddings before pooling.

        Useful for interpreting which players contribute most.

        Args:
            data: PyG Data object with node features and edges.

        Returns:
            Node embeddings of shape (num_nodes, hidden_dim).
        """
        x = data.x
        edge_index = data.edge_index

        # Input projection
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = F.relu(x)

        # GATv2 layers
        for gat, norm in zip(self.gat_layers, self.layer_norms):
            x = gat(x, edge_index)
            x = norm(x)
            x = F.elu(x)

        return x


# =============================================================================
# Graph Builder
# =============================================================================


class LineupGraphBuilder:
    """Build PyG Data objects from lineup information.

    Constructs graphs representing player interactions for the GNN model.
    Handles player feature lookup, edge construction, and missing data.

    Attributes:
        player_features: DataFrame with player feature data indexed by player_id.
        include_opponent_edges: Whether to include cross-team edges.
        default_features: Default PlayerFeatures for unknown players.

    Example:
        >>> builder = LineupGraphBuilder(player_features_df)
        >>> home = [203507, 2544, 201566, 203076, 1628389]  # Lakers lineup
        >>> away = [201142, 201935, 203954, 1628369, 203507]  # Celtics lineup
        >>> graph = builder.build_graph(home, away)
    """

    def __init__(
        self,
        player_features_df: pd.DataFrame | None = None,
        include_opponent_edges: bool = True,
    ) -> None:
        """Initialize LineupGraphBuilder.

        Args:
            player_features_df: DataFrame with player features. Expected columns:
                - player_id (index or column)
                - orapm, drapm
                - height_inches, weight_lbs
                - position
                - fg_pct, fg3_pct, ft_pct
                - mpg, usg_pct
            include_opponent_edges: Whether to include edges between opponent
                players (for matchup modeling).
        """
        self.player_features_df = player_features_df
        self.include_opponent_edges = include_opponent_edges
        self._feature_cache: dict[PlayerId, PlayerFeatures] = {}

    def build_graph(
        self,
        home_lineup: list[PlayerId],
        away_lineup: list[PlayerId],
        chemistry_matrix: np.ndarray | None = None,
    ) -> Data:
        """Build a PyG Data object for a lineup matchup.

        Args:
            home_lineup: List of 5 home team player IDs.
            away_lineup: List of 5 away team player IDs.
            chemistry_matrix: Optional (10, 10) matrix of chemistry weights
                based on minutes played together.

        Returns:
            PyG Data object with node features and edge connectivity.

        Raises:
            ValueError: If lineups don't contain exactly 5 players each.
        """
        if len(home_lineup) != PLAYERS_PER_TEAM:
            raise ValueError(f"Home lineup must have {PLAYERS_PER_TEAM} players")
        if len(away_lineup) != PLAYERS_PER_TEAM:
            raise ValueError(f"Away lineup must have {PLAYERS_PER_TEAM} players")

        # Build node features
        # Nodes 0-4: home players, Nodes 5-9: away players
        node_features = []
        for player_id in home_lineup:
            features = self._get_player_features(player_id)
            node_features.append(features.to_tensor(is_home=True))

        for player_id in away_lineup:
            features = self._get_player_features(player_id)
            node_features.append(features.to_tensor(is_home=False))

        x = torch.stack(node_features)  # (10, 16)

        # Build edges
        edge_index = self._build_edges()

        # Create Data object
        data = Data(x=x, edge_index=edge_index)

        # Add edge attributes if chemistry matrix provided
        if chemistry_matrix is not None:
            edge_attr = self._build_edge_weights(edge_index, chemistry_matrix)
            data.edge_attr = edge_attr

        return data

    def _get_player_features(self, player_id: PlayerId) -> PlayerFeatures:
        """Look up features for a player.

        Args:
            player_id: NBA player ID.

        Returns:
            PlayerFeatures object with all feature values.
        """
        # Check cache first
        if player_id in self._feature_cache:
            return self._feature_cache[player_id]

        # Default features if no data available
        if self.player_features_df is None:
            return PlayerFeatures(player_id=player_id)

        # Look up in DataFrame
        df = self.player_features_df
        if "player_id" in df.columns:
            row = df[df["player_id"] == player_id]
        else:
            row = df[df.index == player_id]

        if row.empty:
            logger.debug("No features found for player {}, using defaults", player_id)
            return PlayerFeatures(player_id=player_id)

        row = row.iloc[0]

        # Extract features with defaults
        features = PlayerFeatures(
            player_id=player_id,
            orapm=float(row.get("orapm", 0.0)),
            drapm=float(row.get("drapm", 0.0)),
            height=self._normalize_height(row.get("height_inches", 79)),
            weight=self._normalize_weight(row.get("weight_lbs", 215)),
            position=str(row.get("position", "SF")),
            fg_pct=float(row.get("fg_pct", 0.45)),
            fg3_pct=float(row.get("fg3_pct", 0.35)),
            ft_pct=float(row.get("ft_pct", 0.75)),
            mpg=float(row.get("mpg", 20.0)),
            usg_pct=float(row.get("usg_pct", 0.20)),
        )

        # Cache for future lookups
        self._feature_cache[player_id] = features
        return features

    def _normalize_height(self, height_inches: float) -> float:
        """Normalize height to z-score.

        Args:
            height_inches: Player height in inches.

        Returns:
            Z-score normalized height (mean ~79", std ~3.5").
        """
        mean_height = 79.0  # NBA average ~6'7"
        std_height = 3.5
        return (height_inches - mean_height) / std_height

    def _normalize_weight(self, weight_lbs: float) -> float:
        """Normalize weight to z-score.

        Args:
            weight_lbs: Player weight in pounds.

        Returns:
            Z-score normalized weight (mean ~215 lbs, std ~25 lbs).
        """
        mean_weight = 215.0
        std_weight = 25.0
        return (weight_lbs - mean_weight) / std_weight

    def _build_edges(self) -> torch.Tensor:
        """Build edge index tensor for the lineup graph.

        Creates edges for:
        1. Teammate connections (fully connected within each team)
        2. Opponent connections (if enabled, full cross-team connectivity)

        Returns:
            Edge index tensor of shape (2, num_edges).
        """
        edges_src = []
        edges_dst = []

        # Home team edges (nodes 0-4, fully connected)
        for i in range(PLAYERS_PER_TEAM):
            for j in range(PLAYERS_PER_TEAM):
                if i != j:
                    edges_src.append(i)
                    edges_dst.append(j)

        # Away team edges (nodes 5-9, fully connected)
        for i in range(PLAYERS_PER_TEAM, TOTAL_PLAYERS):
            for j in range(PLAYERS_PER_TEAM, TOTAL_PLAYERS):
                if i != j:
                    edges_src.append(i)
                    edges_dst.append(j)

        # Opponent edges (cross-team)
        if self.include_opponent_edges:
            for i in range(PLAYERS_PER_TEAM):  # Home players
                for j in range(PLAYERS_PER_TEAM, TOTAL_PLAYERS):  # Away players
                    # Bidirectional
                    edges_src.extend([i, j])
                    edges_dst.extend([j, i])

        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        return edge_index

    def _build_edge_weights(
        self,
        edge_index: torch.Tensor,
        chemistry_matrix: np.ndarray,
    ) -> torch.Tensor:
        """Build edge weight tensor from chemistry matrix.

        Args:
            edge_index: Edge connectivity tensor of shape (2, num_edges).
            chemistry_matrix: (10, 10) matrix of chemistry scores.

        Returns:
            Edge weight tensor of shape (num_edges, 1).
        """
        num_edges = edge_index.size(1)
        weights = []

        for i in range(num_edges):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()
            weight = chemistry_matrix[src, dst]
            weights.append(weight)

        return torch.tensor(weights, dtype=torch.float32).unsqueeze(-1)

    def batch_graphs(self, graphs: list[Data]) -> Batch:
        """Batch multiple graphs into a single Batch object.

        Args:
            graphs: List of PyG Data objects.

        Returns:
            PyG Batch object with all graphs combined.
        """
        return Batch.from_data_list(graphs)


def create_empty_graph() -> Data:
    """Create an empty graph with default structure.

    Useful for games without lineup data.

    Returns:
        PyG Data object with 10 nodes and default features.
    """
    x = torch.zeros(TOTAL_PLAYERS, DEFAULT_NODE_FEATURES)
    edge_index = torch.tensor([[], []], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)
