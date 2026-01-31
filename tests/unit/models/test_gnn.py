"""Tests for PlayerInteractionGNN and LineupGraphBuilder.

Tests cover:
- Model initialization and forward pass
- Output shape verification for various graph sizes
- Edge construction patterns
- Graph batching
"""

from __future__ import annotations

import pytest
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data, Batch

from nba_model.models.gnn import (
    PlayerInteractionGNN,
    LineupGraphBuilder,
    PlayerFeatures,
    create_empty_graph,
    PLAYERS_PER_TEAM,
    TOTAL_PLAYERS,
)


class TestPlayerFeatures:
    """Tests for PlayerFeatures dataclass."""

    def test_default_values(self) -> None:
        """PlayerFeatures should have sensible defaults."""
        pf = PlayerFeatures(player_id=12345)
        assert pf.orapm == 0.0
        assert pf.drapm == 0.0
        assert pf.position == "SF"
        assert 0 <= pf.fg_pct <= 1

    def test_to_tensor_shape(self) -> None:
        """to_tensor should return 16-dimensional tensor."""
        pf = PlayerFeatures(player_id=12345)
        tensor = pf.to_tensor(is_home=True)
        assert tensor.shape == (16,)
        assert tensor.dtype == torch.float32

    def test_to_tensor_home_flag(self) -> None:
        """is_home flag should be reflected in tensor."""
        pf = PlayerFeatures(player_id=12345)
        home_tensor = pf.to_tensor(is_home=True)
        away_tensor = pf.to_tensor(is_home=False)

        # Last feature is home indicator
        assert home_tensor[-1] == 1.0
        assert away_tensor[-1] == 0.0

    def test_position_encoding(self) -> None:
        """Position should be one-hot encoded."""
        for pos in ["PG", "SG", "SF", "PF", "C"]:
            pf = PlayerFeatures(player_id=12345, position=pos)
            tensor = pf.to_tensor()
            # Check that exactly one position is 1.0
            pos_slice = tensor[5:10]
            assert pos_slice.sum() == 1.0


class TestPlayerInteractionGNN:
    """Tests for PlayerInteractionGNN model."""

    def test_initialization_default_params(self) -> None:
        """Model should initialize with default parameters."""
        model = PlayerInteractionGNN()
        assert model.node_features == 16
        assert model.hidden_dim == 64
        assert model.output_dim == 128

    def test_initialization_custom_params(self) -> None:
        """Model should accept custom parameters."""
        model = PlayerInteractionGNN(
            node_features=8, hidden_dim=32, output_dim=64, num_heads=2
        )
        assert model.node_features == 8
        assert model.output_dim == 64

    def test_forward_output_shape(
        self,
        gnn_model: PlayerInteractionGNN,
        sample_graph: Data,
    ) -> None:
        """Forward pass should produce correct output shape."""
        output = gnn_model(sample_graph)
        # Single graph should produce (1, output_dim)
        assert output.shape == (1, gnn_model.output_dim)

    def test_forward_batched_graphs(
        self,
        gnn_model: PlayerInteractionGNN,
        sample_graph: Data,
    ) -> None:
        """Forward pass should handle batched graphs."""
        # Create batch of 3 graphs
        batch = Batch.from_data_list([sample_graph, sample_graph, sample_graph])
        output = gnn_model(batch)
        assert output.shape == (3, gnn_model.output_dim)

    def test_gradient_flow(
        self,
        gnn_model: PlayerInteractionGNN,
        sample_graph: Data,
    ) -> None:
        """Gradients should flow through all layers."""
        output = gnn_model(sample_graph)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert gnn_model.input_proj.weight.grad is not None
        for gat in gnn_model.gat_layers:
            assert gat.lin_l.weight.grad is not None

    def test_eval_mode_deterministic(
        self,
        gnn_model: PlayerInteractionGNN,
        sample_graph: Data,
    ) -> None:
        """Model should be deterministic in eval mode."""
        gnn_model.eval()
        with torch.no_grad():
            out1 = gnn_model(sample_graph)
            out2 = gnn_model(sample_graph)
        assert torch.allclose(out1, out2)

    def test_get_node_embeddings(
        self,
        gnn_model: PlayerInteractionGNN,
        sample_graph: Data,
    ) -> None:
        """get_node_embeddings should return per-node representations."""
        embeddings = gnn_model.get_node_embeddings(sample_graph)
        assert embeddings.shape[0] == sample_graph.x.shape[0]  # Same num nodes


class TestLineupGraphBuilder:
    """Tests for LineupGraphBuilder class."""

    def test_initialization_without_features(self) -> None:
        """Builder should work without player features DataFrame."""
        builder = LineupGraphBuilder()
        assert builder.player_features_df is None

    def test_initialization_with_features(self) -> None:
        """Builder should accept player features DataFrame."""
        df = pd.DataFrame({
            "player_id": [1, 2, 3],
            "orapm": [1.0, 0.5, -0.5],
            "drapm": [0.5, 1.0, 0.0],
        })
        builder = LineupGraphBuilder(df)
        assert builder.player_features_df is not None

    def test_build_graph_output_type(self) -> None:
        """build_graph should return PyG Data object."""
        builder = LineupGraphBuilder()
        home = [1, 2, 3, 4, 5]
        away = [6, 7, 8, 9, 10]
        graph = builder.build_graph(home, away)
        assert isinstance(graph, Data)

    def test_build_graph_node_count(self) -> None:
        """Graph should have 10 nodes (5 per team)."""
        builder = LineupGraphBuilder()
        home = [1, 2, 3, 4, 5]
        away = [6, 7, 8, 9, 10]
        graph = builder.build_graph(home, away)
        assert graph.x.shape[0] == TOTAL_PLAYERS

    def test_build_graph_node_features(self) -> None:
        """Node features should have correct dimension."""
        builder = LineupGraphBuilder()
        home = [1, 2, 3, 4, 5]
        away = [6, 7, 8, 9, 10]
        graph = builder.build_graph(home, away)
        assert graph.x.shape[1] == 16  # DEFAULT_NODE_FEATURES

    def test_build_graph_edge_connectivity(self) -> None:
        """Graph should have teammate and opponent edges."""
        builder = LineupGraphBuilder(include_opponent_edges=True)
        home = [1, 2, 3, 4, 5]
        away = [6, 7, 8, 9, 10]
        graph = builder.build_graph(home, away)

        # Check edge_index shape
        assert graph.edge_index.shape[0] == 2
        assert graph.edge_index.shape[1] > 0

        # Should have teammate edges (4 edges per player = 40 total within-team)
        # Plus cross-team edges (25 * 2 = 50)

    def test_build_graph_invalid_lineup_size(self) -> None:
        """Should raise error for invalid lineup sizes."""
        builder = LineupGraphBuilder()

        with pytest.raises(ValueError, match="5 players"):
            builder.build_graph([1, 2, 3, 4], [6, 7, 8, 9, 10])

        with pytest.raises(ValueError, match="5 players"):
            builder.build_graph([1, 2, 3, 4, 5], [6, 7, 8, 9])

    def test_build_graph_with_chemistry(self) -> None:
        """Should use chemistry matrix for edge weights if provided."""
        builder = LineupGraphBuilder()
        home = [1, 2, 3, 4, 5]
        away = [6, 7, 8, 9, 10]
        chemistry = np.ones((10, 10))

        graph = builder.build_graph(home, away, chemistry_matrix=chemistry)
        assert graph.edge_attr is not None
        assert graph.edge_attr.shape[0] == graph.edge_index.shape[1]

    def test_batch_graphs(self) -> None:
        """batch_graphs should combine multiple graphs."""
        builder = LineupGraphBuilder()
        home = [1, 2, 3, 4, 5]
        away = [6, 7, 8, 9, 10]

        graphs = [builder.build_graph(home, away) for _ in range(3)]
        batch = builder.batch_graphs(graphs)

        assert isinstance(batch, Batch)
        assert batch.num_graphs == 3


class TestCreateEmptyGraph:
    """Tests for create_empty_graph function."""

    def test_creates_valid_graph(self) -> None:
        """Should create valid PyG Data object."""
        graph = create_empty_graph()
        assert isinstance(graph, Data)

    def test_correct_node_count(self) -> None:
        """Empty graph should have 10 nodes."""
        graph = create_empty_graph()
        assert graph.x.shape[0] == TOTAL_PLAYERS

    def test_correct_feature_dim(self) -> None:
        """Empty graph should have correct feature dimension."""
        graph = create_empty_graph()
        assert graph.x.shape[1] == 16

    def test_no_edges(self) -> None:
        """Empty graph should have no edges."""
        graph = create_empty_graph()
        assert graph.edge_index.shape[1] == 0
