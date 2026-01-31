"""Tests for TwoTowerFusion and related components.

Tests cover:
- Model initialization and forward pass
- Multi-task head output shapes
- Loss computation stability
- Context feature building
"""

from __future__ import annotations

import pytest
import torch
import pandas as pd

from nba_model.models.fusion import (
    TwoTowerFusion,
    ContextFeatureBuilder,
    MultiTaskLoss,
    FusionOutput,
    GameContext,
    CONTEXT_FEATURES,
    create_dummy_inputs,
    create_dummy_labels,
)


class TestTwoTowerFusion:
    """Tests for TwoTowerFusion model."""

    def test_initialization_default_params(self) -> None:
        """Model should initialize with default parameters."""
        model = TwoTowerFusion()
        assert model.context_dim == 32
        assert model.transformer_dim == 128
        assert model.gnn_dim == 128
        assert model.hidden_dim == 256

    def test_initialization_custom_params(self) -> None:
        """Model should accept custom parameters."""
        model = TwoTowerFusion(
            context_dim=16,
            transformer_dim=64,
            gnn_dim=64,
            hidden_dim=128,
        )
        assert model.context_dim == 16
        assert model.transformer_dim == 64

    def test_forward_output_types(
        self,
        fusion_model: TwoTowerFusion,
        sample_context: torch.Tensor,
        sample_transformer_output: torch.Tensor,
        sample_gnn_output: torch.Tensor,
    ) -> None:
        """Forward pass should return dictionary with correct keys."""
        # Adjust input sizes to match model
        context = torch.randn(2, 32)
        transformer_out = torch.randn(2, 64)
        gnn_out = torch.randn(2, 64)

        outputs = fusion_model(context, transformer_out, gnn_out)

        assert isinstance(outputs, dict)
        assert "win_prob" in outputs
        assert "margin" in outputs
        assert "total" in outputs
        assert "fusion_embedding" in outputs

    def test_forward_output_shapes(
        self,
        fusion_model: TwoTowerFusion,
    ) -> None:
        """Forward pass should produce correct output shapes."""
        batch_size = 4
        context = torch.randn(batch_size, 32)
        transformer_out = torch.randn(batch_size, 64)
        gnn_out = torch.randn(batch_size, 64)

        outputs = fusion_model(context, transformer_out, gnn_out)

        assert outputs["win_prob"].shape == (batch_size, 1)
        assert outputs["margin"].shape == (batch_size, 1)
        assert outputs["total"].shape == (batch_size, 1)

    def test_win_prob_range(
        self,
        fusion_model: TwoTowerFusion,
    ) -> None:
        """Win probability should be in [0, 1]."""
        context = torch.randn(10, 32)
        transformer_out = torch.randn(10, 64)
        gnn_out = torch.randn(10, 64)

        outputs = fusion_model(context, transformer_out, gnn_out)
        win_probs = outputs["win_prob"]

        assert (win_probs >= 0).all()
        assert (win_probs <= 1).all()

    def test_gradient_flow(
        self,
        fusion_model: TwoTowerFusion,
    ) -> None:
        """Gradients should flow through all towers."""
        context = torch.randn(2, 32)
        transformer_out = torch.randn(2, 64)
        gnn_out = torch.randn(2, 64)

        outputs = fusion_model(context, transformer_out, gnn_out)
        loss = outputs["win_prob"].sum() + outputs["margin"].sum() + outputs["total"].sum()
        loss.backward()

        # Check gradients in context tower
        assert fusion_model.context_tower[0].weight.grad is not None

        # Check gradients in dynamic tower
        assert fusion_model.dynamic_tower[0].weight.grad is not None

        # Check gradients in fusion
        assert fusion_model.fusion[0].weight.grad is not None

    def test_eval_mode_deterministic(
        self,
        fusion_model: TwoTowerFusion,
    ) -> None:
        """Model should be deterministic in eval mode."""
        fusion_model.eval()
        context = torch.randn(2, 32)
        transformer_out = torch.randn(2, 64)
        gnn_out = torch.randn(2, 64)

        with torch.no_grad():
            out1 = fusion_model(context, transformer_out, gnn_out)
            out2 = fusion_model(context, transformer_out, gnn_out)

        assert torch.allclose(out1["win_prob"], out2["win_prob"])
        assert torch.allclose(out1["margin"], out2["margin"])
        assert torch.allclose(out1["total"], out2["total"])

    def test_get_tower_outputs(
        self,
        fusion_model: TwoTowerFusion,
    ) -> None:
        """get_tower_outputs should return individual tower outputs."""
        context = torch.randn(2, 32)
        transformer_out = torch.randn(2, 64)
        gnn_out = torch.randn(2, 64)

        context_out, dynamic_out = fusion_model.get_tower_outputs(
            context, transformer_out, gnn_out
        )

        assert context_out.shape == (2, fusion_model.hidden_dim)
        assert dynamic_out.shape == (2, fusion_model.hidden_dim)


class TestMultiTaskLoss:
    """Tests for MultiTaskLoss function."""

    def test_initialization_default_weights(self) -> None:
        """Loss should initialize with equal weights."""
        loss_fn = MultiTaskLoss()
        assert loss_fn.huber_delta == 1.0
        assert not loss_fn.learnable

    def test_initialization_custom_weights(self) -> None:
        """Loss should accept custom weights."""
        loss_fn = MultiTaskLoss(win_weight=2.0, margin_weight=0.5, total_weight=0.5)
        assert loss_fn.win_weight.item() == 2.0

    def test_forward_output_type(self) -> None:
        """Forward should return (loss, components_dict)."""
        loss_fn = MultiTaskLoss()
        outputs = {
            "win_prob": torch.sigmoid(torch.randn(4, 1)),
            "margin": torch.randn(4, 1),
            "total": torch.randn(4, 1),
        }
        labels = {
            "win": torch.randint(0, 2, (4, 1)).float(),
            "margin": torch.randn(4, 1),
            "total": torch.randn(4, 1),
        }

        loss, components = loss_fn(outputs, labels)

        assert isinstance(loss, torch.Tensor)
        assert isinstance(components, dict)
        assert "win_loss" in components
        assert "margin_loss" in components
        assert "total_loss" in components

    def test_loss_is_positive(self) -> None:
        """All loss components should be non-negative."""
        loss_fn = MultiTaskLoss()
        outputs = {
            "win_prob": torch.sigmoid(torch.randn(4, 1)),
            "margin": torch.randn(4, 1),
            "total": torch.randn(4, 1),
        }
        labels = {
            "win": torch.randint(0, 2, (4, 1)).float(),
            "margin": torch.randn(4, 1),
            "total": torch.randn(4, 1),
        }

        loss, components = loss_fn(outputs, labels)

        assert loss.item() >= 0
        assert components["win_loss"] >= 0
        assert components["margin_loss"] >= 0
        assert components["total_loss"] >= 0

    def test_loss_gradient_flow(self) -> None:
        """Loss should allow gradient computation."""
        loss_fn = MultiTaskLoss()
        outputs = {
            "win_prob": torch.sigmoid(torch.randn(4, 1, requires_grad=True)),
            "margin": torch.randn(4, 1, requires_grad=True),
            "total": torch.randn(4, 1, requires_grad=True),
        }
        labels = {
            "win": torch.randint(0, 2, (4, 1)).float(),
            "margin": torch.randn(4, 1),
            "total": torch.randn(4, 1),
        }

        loss, _ = loss_fn(outputs, labels)
        loss.backward()

        # Gradients should exist for output tensors
        # (they have requires_grad=True via sigmoid input)

    def test_learnable_weights(self) -> None:
        """Learnable weights mode should create parameters."""
        loss_fn = MultiTaskLoss(learnable_weights=True)
        assert loss_fn.learnable
        assert hasattr(loss_fn, "log_var_win")
        assert isinstance(loss_fn.log_var_win, torch.nn.Parameter)


class TestContextFeatureBuilder:
    """Tests for ContextFeatureBuilder class."""

    def test_initialization(self) -> None:
        """Builder should initialize with feature names."""
        builder = ContextFeatureBuilder()
        assert len(builder.feature_names) > 0
        assert builder.feature_dim == len(CONTEXT_FEATURES)

    def test_build_from_dict(self) -> None:
        """Should build tensor from dictionary."""
        builder = ContextFeatureBuilder()
        features = {
            "home_off_rating_z": 1.0,
            "home_def_rating_z": -0.5,
            "home_pace_z": 0.2,
        }
        tensor = builder.build_from_dict(features)

        assert tensor.shape == (builder.feature_dim,)
        assert tensor[0] == 1.0  # home_off_rating_z

    def test_build_from_empty_dict(self) -> None:
        """Should return zeros for empty dict."""
        builder = ContextFeatureBuilder()
        tensor = builder.build_from_dict({})

        assert tensor.shape == (builder.feature_dim,)
        assert (tensor == 0).all()

    def test_normalize_features(self) -> None:
        """Should normalize features using provided stats."""
        builder = ContextFeatureBuilder()
        features = torch.ones(builder.feature_dim)
        stats = {
            "home_off_rating_z": (0.5, 0.5),  # mean=0.5, std=0.5
        }

        normalized = builder.normalize_features(features, stats)

        # (1 - 0.5) / 0.5 = 1.0
        assert normalized[0] == 1.0


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_create_dummy_inputs(self) -> None:
        """Should create inputs with correct shapes."""
        context, transformer_out, gnn_out = create_dummy_inputs(batch_size=4)

        assert context.shape == (4, 32)
        assert transformer_out.shape == (4, 128)
        assert gnn_out.shape == (4, 128)

    def test_create_dummy_labels(self) -> None:
        """Should create labels with correct keys and shapes."""
        labels = create_dummy_labels(batch_size=4)

        assert "win" in labels
        assert "margin" in labels
        assert "total" in labels
        assert labels["win"].shape == (4, 1)
        assert labels["margin"].shape == (4, 1)
        assert labels["total"].shape == (4, 1)
