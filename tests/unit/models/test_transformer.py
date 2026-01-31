"""Tests for GameFlowTransformer and EventTokenizer.

Tests cover:
- Model initialization and forward pass
- Output shape verification
- Attention mask handling
- Gradient flow through all layers
- EventTokenizer functionality
"""

from __future__ import annotations

import pytest
import torch
import pandas as pd

from nba_model.models.transformer import (
    GameFlowTransformer,
    EventTokenizer,
    PositionalEncoding,
    TokenizedSequence,
    collate_sequences,
    EVENT_VOCAB,
)


class TestPositionalEncoding:
    """Tests for PositionalEncoding module."""

    def test_output_shape_matches_input(self) -> None:
        """Positional encoding should preserve input shape."""
        pe = PositionalEncoding(d_model=128, max_len=50)
        x = torch.randn(2, 50, 128)
        output = pe(x)
        assert output.shape == x.shape

    def test_adds_positional_information(self) -> None:
        """Output should differ from input due to positional encoding."""
        pe = PositionalEncoding(d_model=128, max_len=50, dropout=0.0)
        x = torch.zeros(1, 50, 128)
        output = pe(x)
        # With zero input and no dropout, output should be the PE values
        assert not torch.allclose(output, x)

    def test_shorter_sequence(self) -> None:
        """Should handle sequences shorter than max_len."""
        pe = PositionalEncoding(d_model=128, max_len=100)
        x = torch.randn(2, 30, 128)  # Shorter than max_len
        output = pe(x)
        assert output.shape == x.shape


class TestGameFlowTransformer:
    """Tests for GameFlowTransformer model."""

    def test_initialization_default_params(self) -> None:
        """Model should initialize with default parameters."""
        model = GameFlowTransformer()
        assert model.d_model == 128
        assert model.vocab_size == len(EVENT_VOCAB)

    def test_initialization_custom_params(self) -> None:
        """Model should accept custom parameters."""
        model = GameFlowTransformer(
            vocab_size=20, d_model=64, nhead=2, num_layers=1
        )
        assert model.d_model == 64
        assert model.vocab_size == 20

    def test_forward_output_shape(
        self,
        transformer_model: GameFlowTransformer,
        sample_events: torch.Tensor,
        sample_times: torch.Tensor,
        sample_scores: torch.Tensor,
        sample_lineups: torch.Tensor,
    ) -> None:
        """Forward pass should produce correct output shape."""
        output = transformer_model(
            sample_events, sample_times, sample_scores, sample_lineups
        )
        batch_size = sample_events.shape[0]
        expected_dim = transformer_model.d_model
        assert output.shape == (batch_size, expected_dim)

    def test_forward_with_mask(
        self,
        transformer_model: GameFlowTransformer,
        sample_events: torch.Tensor,
        sample_times: torch.Tensor,
        sample_scores: torch.Tensor,
        sample_lineups: torch.Tensor,
    ) -> None:
        """Forward pass should handle attention mask correctly."""
        # Create mask where last 10 positions are masked
        mask = torch.zeros(2, 50, dtype=torch.bool)
        mask[:, 40:] = True

        output = transformer_model(
            sample_events, sample_times, sample_scores, sample_lineups, mask=mask
        )
        assert output.shape == (2, transformer_model.d_model)

    def test_forward_batch_size_one(
        self,
        transformer_model: GameFlowTransformer,
    ) -> None:
        """Should handle batch size of 1."""
        events = torch.randint(1, 14, (1, 50))
        times = torch.rand(1, 50, 1)
        scores = torch.randn(1, 50, 1)
        lineups = torch.zeros(1, 50, 10, dtype=torch.long)

        output = transformer_model(events, times, scores, lineups)
        assert output.shape == (1, transformer_model.d_model)

    def test_gradient_flow(
        self,
        transformer_model: GameFlowTransformer,
        sample_events: torch.Tensor,
        sample_times: torch.Tensor,
        sample_scores: torch.Tensor,
        sample_lineups: torch.Tensor,
    ) -> None:
        """Gradients should flow through all layers."""
        output = transformer_model(
            sample_events, sample_times, sample_scores, sample_lineups
        )
        loss = output.sum()
        loss.backward()

        # Check that gradients exist for key parameters
        assert transformer_model.event_embedding.weight.grad is not None
        assert transformer_model.input_projection.weight.grad is not None
        assert transformer_model.output_proj.weight.grad is not None

    def test_eval_mode_deterministic(
        self,
        transformer_model: GameFlowTransformer,
        sample_events: torch.Tensor,
        sample_times: torch.Tensor,
        sample_scores: torch.Tensor,
        sample_lineups: torch.Tensor,
    ) -> None:
        """Model should be deterministic in eval mode."""
        transformer_model.eval()
        with torch.no_grad():
            out1 = transformer_model(
                sample_events, sample_times, sample_scores, sample_lineups
            )
            out2 = transformer_model(
                sample_events, sample_times, sample_scores, sample_lineups
            )
        assert torch.allclose(out1, out2)


class TestEventTokenizer:
    """Tests for EventTokenizer class."""

    def test_initialization_default(self) -> None:
        """Tokenizer should initialize with default parameters."""
        tokenizer = EventTokenizer()
        assert tokenizer.max_seq_len == 50
        assert tokenizer.vocab_size == len(EVENT_VOCAB)
        assert tokenizer.pad_idx == 0

    def test_tokenize_game_output_types(
        self,
        sample_plays_df: pd.DataFrame,
    ) -> None:
        """Tokenization should produce correct tensor types."""
        tokenizer = EventTokenizer()
        tokens = tokenizer.tokenize_game(sample_plays_df)

        assert isinstance(tokens, TokenizedSequence)
        assert isinstance(tokens.events, torch.Tensor)
        assert isinstance(tokens.times, torch.Tensor)
        assert isinstance(tokens.scores, torch.Tensor)
        assert isinstance(tokens.lineups, torch.Tensor)
        assert isinstance(tokens.mask, torch.Tensor)

    def test_tokenize_game_output_shapes(
        self,
        sample_plays_df: pd.DataFrame,
    ) -> None:
        """Tokenization should produce tensors with correct shapes."""
        tokenizer = EventTokenizer(max_seq_len=50)
        tokens = tokenizer.tokenize_game(sample_plays_df)

        seq_len = len(sample_plays_df)
        assert tokens.events.shape == (seq_len,)
        assert tokens.times.shape == (seq_len, 1)
        assert tokens.scores.shape == (seq_len, 1)
        assert tokens.lineups.shape == (seq_len, 10)
        assert tokens.mask.shape == (seq_len,)

    def test_pad_sequence(
        self,
        sample_plays_df: pd.DataFrame,
    ) -> None:
        """Padding should extend sequence to target length."""
        tokenizer = EventTokenizer(max_seq_len=50)
        tokens = tokenizer.tokenize_game(sample_plays_df)
        padded = tokenizer.pad_sequence(tokens, target_len=50)

        assert padded.events.shape[0] == 50
        assert padded.times.shape[0] == 50
        # Padded positions should be masked
        assert padded.mask[10:].all()

    def test_event_mapping(self) -> None:
        """All event types should map to valid indices."""
        tokenizer = EventTokenizer()
        for event_type in range(1, 14):
            idx = tokenizer.event_to_idx.get(event_type, tokenizer.event_to_idx[14])
            assert 0 <= idx < tokenizer.vocab_size


class TestCollateSequences:
    """Tests for collate_sequences function."""

    def test_collate_multiple_sequences(
        self,
        sample_plays_df: pd.DataFrame,
    ) -> None:
        """Collation should batch multiple sequences."""
        tokenizer = EventTokenizer(max_seq_len=20)
        tokens1 = tokenizer.tokenize_game(sample_plays_df)
        tokens2 = tokenizer.tokenize_game(sample_plays_df)

        batched = collate_sequences([tokens1, tokens2])

        assert batched.events.shape[0] == 2  # Batch size
        assert batched.events.dim() == 2

    def test_collate_empty_raises(self) -> None:
        """Collating empty list should raise error."""
        with pytest.raises(ValueError, match="empty"):
            collate_sequences([])

    def test_collate_variable_lengths(self) -> None:
        """Collation should pad to longest sequence."""
        tokenizer = EventTokenizer()

        # Create sequences of different lengths
        df1 = pd.DataFrame({
            "event_type": [1, 2, 3],
            "period": [1, 1, 1],
            "pc_time": ["12:00", "11:00", "10:00"],
            "score_home": [0, 2, 4],
            "score_away": [0, 0, 0],
        })
        df2 = pd.DataFrame({
            "event_type": [1, 2, 3, 4, 5],
            "period": [1] * 5,
            "pc_time": ["12:00", "11:00", "10:00", "9:00", "8:00"],
            "score_home": [0, 2, 4, 4, 6],
            "score_away": [0, 0, 0, 0, 0],
        })

        tokens1 = tokenizer.tokenize_game(df1)
        tokens2 = tokenizer.tokenize_game(df2)

        batched = collate_sequences([tokens1, tokens2])
        assert batched.events.shape[1] == 5  # Padded to longest
