"""Transformer sequence model for NBA game flow encoding.

This module implements a Transformer encoder that processes tokenized play-by-play
event sequences to capture game flow dynamics. The model combines event type,
temporal, score, and lineup embeddings.

Architecture Specifications:
    - Embedding dimension: 128
    - Attention heads: 4
    - Encoder layers: 2
    - Maximum sequence length: 50 events
    - Dropout rate: 0.1

Example:
    >>> from nba_model.models.transformer import GameFlowTransformer, EventTokenizer
    >>> model = GameFlowTransformer(vocab_size=15)
    >>> tokenizer = EventTokenizer()
    >>> tokens = tokenizer.tokenize_game(plays_df, lineups_df)
    >>> output = model(tokens['events'], tokens['times'], tokens['scores'], tokens['lineups'])
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

from nba_model.logging import get_logger

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)

# =============================================================================
# Constants
# =============================================================================

DEFAULT_D_MODEL: int = 128
DEFAULT_NHEAD: int = 4
DEFAULT_NUM_LAYERS: int = 2
DEFAULT_MAX_SEQ_LEN: int = 50
DEFAULT_DROPOUT: float = 0.1
DEFAULT_PLAYER_EMBED_DIM: int = 16
DEFAULT_PLAYER_VOCAB_SIZE: int = 10000

# NBA event types from play-by-play EVENTMSGTYPE
EVENT_VOCAB: dict[int, str] = {
    0: "PAD",  # Padding token
    1: "MADE_SHOT",
    2: "MISSED_SHOT",
    3: "FREE_THROW",
    4: "REBOUND",
    5: "TURNOVER",
    6: "FOUL",
    7: "VIOLATION",
    8: "SUBSTITUTION",
    9: "TIMEOUT",
    10: "JUMP_BALL",
    11: "EJECTION",
    12: "PERIOD_START",
    13: "PERIOD_END",
    14: "UNKNOWN",
}

# Historical stats for normalization
SCORE_DIFF_STD: float = 8.5  # Historical std of score differentials


# =============================================================================
# Data Containers
# =============================================================================


@dataclass
class TokenizedSequence:
    """Container for tokenized game sequence data.

    Attributes:
        events: Tensor of event type indices (seq_len,).
        times: Tensor of normalized time remaining (seq_len, 1).
        scores: Tensor of normalized score differential (seq_len, 1).
        lineups: Tensor of lineup player IDs (seq_len, 10).
        mask: Attention mask for padding (seq_len,).
    """

    events: torch.Tensor
    times: torch.Tensor
    scores: torch.Tensor
    lineups: torch.Tensor
    mask: torch.Tensor

    def to(self, device: torch.device) -> TokenizedSequence:
        """Move all tensors to specified device."""
        return TokenizedSequence(
            events=self.events.to(device),
            times=self.times.to(device),
            scores=self.scores.to(device),
            lineups=self.lineups.to(device),
            mask=self.mask.to(device),
        )


# =============================================================================
# Positional Encoding
# =============================================================================


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer.

    Adds positional information to embeddings using sine and cosine functions
    of different frequencies, as described in "Attention Is All You Need".

    Attributes:
        d_model: Embedding dimension.
        max_len: Maximum sequence length.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int = DEFAULT_D_MODEL,
        max_len: int = DEFAULT_MAX_SEQ_LEN,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        """Initialize positional encoding.

        Args:
            d_model: Embedding dimension.
            max_len: Maximum sequence length.
            dropout: Dropout probability.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, but part of state_dict)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Tensor with positional encoding added, same shape as input.
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


# =============================================================================
# Transformer Model
# =============================================================================


class GameFlowTransformer(nn.Module):
    """Transformer Encoder for modeling NBA game flow sequences.

    Processes tokenized play-by-play event sequences to capture game dynamics.
    Combines event type embeddings with temporal, score differential, and
    lineup context embeddings.

    Architecture:
        - Event type embedding layer
        - Sinusoidal positional encoding
        - Time, score, and lineup projection layers
        - Transformer encoder with self-attention
        - Output projection to fixed dimension

    Attributes:
        vocab_size: Number of unique event types.
        d_model: Embedding/hidden dimension (default 128).
        nhead: Number of attention heads (default 4).
        num_layers: Number of encoder layers (default 2).
        max_seq_len: Maximum sequence length (default 50).
        dropout: Dropout probability (default 0.1).

    Example:
        >>> model = GameFlowTransformer(vocab_size=15, d_model=128)
        >>> events = torch.randint(0, 15, (2, 50))  # batch=2, seq_len=50
        >>> times = torch.rand(2, 50, 1)
        >>> scores = torch.randn(2, 50, 1)
        >>> lineups = torch.zeros(2, 50, 10, dtype=torch.long)
        >>> output = model(events, times, scores, lineups)
        >>> print(output.shape)  # (2, 128)
    """

    def __init__(
        self,
        vocab_size: int = len(EVENT_VOCAB),
        d_model: int = DEFAULT_D_MODEL,
        nhead: int = DEFAULT_NHEAD,
        num_layers: int = DEFAULT_NUM_LAYERS,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        dropout: float = DEFAULT_DROPOUT,
        player_vocab_size: int = DEFAULT_PLAYER_VOCAB_SIZE,
        player_embed_dim: int = DEFAULT_PLAYER_EMBED_DIM,
    ) -> None:
        """Initialize GameFlowTransformer.

        Args:
            vocab_size: Number of unique event types in vocabulary.
            d_model: Embedding and hidden dimension.
            nhead: Number of attention heads.
            num_layers: Number of Transformer encoder layers.
            max_seq_len: Maximum sequence length for positional encoding.
            dropout: Dropout probability for regularization.
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.player_vocab_size = player_vocab_size
        self.player_embed_dim = player_embed_dim

        # Event type embedding
        self.event_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Additional feature embeddings (project to d_model // 4 each)
        quarter_dim = d_model // 4
        self.time_embedding = nn.Linear(1, quarter_dim)
        self.score_embedding = nn.Linear(1, quarter_dim)
        self.player_embedding = nn.Embedding(
            player_vocab_size,
            player_embed_dim,
            padding_idx=0,
        )
        self.lineup_projection = nn.Linear(player_embed_dim * 2, d_model // 2)

        # Fusion layer to combine all embeddings
        combined_dim = d_model + quarter_dim + quarter_dim + (d_model // 2)
        self.input_projection = nn.Linear(combined_dim, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for better training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
        self.output_norm = nn.LayerNorm(d_model)

        # Initialize weights
        self._init_weights()

        logger.debug(
            "Initialized GameFlowTransformer: vocab={}, d_model={}, nhead={}, layers={}, player_vocab={}, player_dim={}",
            vocab_size,
            d_model,
            nhead,
            num_layers,
            player_vocab_size,
            player_embed_dim,
        )

    def _bucket_player_ids(self, player_ids: torch.Tensor) -> torch.Tensor:
        """Map raw player IDs into embedding buckets with padding support."""
        if self.player_vocab_size <= 1:
            raise ValueError("player_vocab_size must be > 1 for hashing")

        ids = player_ids.long()
        # Keep padding as 0; bucket non-zero IDs into [1, vocab_size - 1]
        bucketed = (ids % (self.player_vocab_size - 1)) + 1
        return torch.where(ids > 0, bucketed, torch.zeros_like(ids))

    def _init_weights(self) -> None:
        """Initialize model weights using Xavier initialization."""
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self,
        events: torch.Tensor,
        times: torch.Tensor,
        scores: torch.Tensor,
        lineups: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the Transformer.

        Args:
            events: Event type indices of shape (batch, seq_len).
            times: Normalized time remaining of shape (batch, seq_len, 1).
            scores: Normalized score differential of shape (batch, seq_len, 1).
            lineups: Player ID tensor of shape (batch, seq_len, 10).
            mask: Optional attention mask of shape (batch, seq_len).
                  True values indicate positions to mask (ignore).

        Returns:
            Sequence representation of shape (batch, d_model) via mean pooling
            over non-masked positions.
        """
        # Shape validation (unused but checked for correctness)
        _ = events.shape  # (batch, seq_len)

        # Embed event types
        event_emb = self.event_embedding(events)  # (batch, seq_len, d_model)

        # Embed additional features
        time_emb = self.time_embedding(times)  # (batch, seq_len, d_model // 4)
        score_emb = self.score_embedding(scores)  # (batch, seq_len, d_model // 4)
        lineup_ids = lineups.long()
        bucketed = self._bucket_player_ids(lineup_ids)
        player_emb = self.player_embedding(bucketed)  # (batch, seq_len, 10, player_dim)

        home_ids = lineup_ids[:, :, :5]
        away_ids = lineup_ids[:, :, 5:]

        home_mask = (home_ids > 0).float().unsqueeze(-1)
        away_mask = (away_ids > 0).float().unsqueeze(-1)

        home_sum = (player_emb[:, :, :5, :] * home_mask).sum(dim=2)
        away_sum = (player_emb[:, :, 5:, :] * away_mask).sum(dim=2)

        home_count = home_mask.sum(dim=2).clamp(min=1.0)
        away_count = away_mask.sum(dim=2).clamp(min=1.0)

        home_emb = home_sum / home_count
        away_emb = away_sum / away_count

        lineup_features = torch.cat([home_emb, away_emb], dim=-1)
        lineup_emb = self.lineup_projection(lineup_features)

        # Concatenate all embeddings
        combined = torch.cat([event_emb, time_emb, score_emb, lineup_emb], dim=-1)

        # Project to d_model dimension
        x = self.input_projection(combined)  # (batch, seq_len, d_model)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Create attention mask for Transformer (expects True for positions to attend)
        # PyTorch TransformerEncoder uses src_key_padding_mask where True = ignore
        src_key_padding_mask = mask if mask is not None else None

        # Pass through Transformer encoder
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Mean pooling over sequence (excluding masked positions)
        if mask is not None:
            # Invert mask: True = valid, False = masked
            valid_mask = ~mask
            valid_mask = valid_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            x = (x * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)  # (batch, d_model)

        # Final projection and normalization
        x = self.output_proj(x)
        x = self.output_norm(x)

        return x

    def get_attention_weights(
        self,
        events: torch.Tensor,
        times: torch.Tensor,
        scores: torch.Tensor,
        lineups: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Get attention weights from each layer for interpretability.

        Args:
            events: Event type indices of shape (batch, seq_len).
            times: Normalized time remaining of shape (batch, seq_len, 1).
            scores: Normalized score differential of shape (batch, seq_len, 1).
            lineups: Player ID tensor of shape (batch, seq_len, 10).
            mask: Optional attention mask of shape (batch, seq_len).

        Returns:
            List of attention weight tensors, one per layer.
            Each tensor has shape (batch, nhead, seq_len, seq_len).
        """
        # This is a simplified version - full implementation would need
        # to hook into the attention layers
        # For now, just return empty list as placeholder
        logger.warning("get_attention_weights not fully implemented")
        return []


# =============================================================================
# Event Tokenizer
# =============================================================================


class EventTokenizer:
    """Tokenizer for converting play-by-play data to model inputs.

    Converts raw play-by-play DataFrames into tensors suitable for the
    GameFlowTransformer model. Handles event type mapping, time normalization,
    score differential scaling, and lineup encoding.

    Attributes:
        event_vocab: Mapping from EVENTMSGTYPE to vocabulary index.
        max_seq_len: Maximum sequence length (longer sequences are truncated).
        score_std: Standard deviation for score differential normalization.

    Example:
        >>> tokenizer = EventTokenizer(max_seq_len=50)
        >>> tokens = tokenizer.tokenize_game(plays_df, stints_df)
        >>> print(tokens.events.shape)  # (seq_len,)
    """

    def __init__(
        self,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        score_std: float = SCORE_DIFF_STD,
    ) -> None:
        """Initialize EventTokenizer.

        Args:
            max_seq_len: Maximum sequence length. Longer sequences are truncated.
            score_std: Standard deviation for normalizing score differentials.
        """
        self.max_seq_len = max_seq_len
        self.score_std = score_std

        # Build vocabulary mapping: EVENTMSGTYPE -> vocab index
        self.event_to_idx: dict[int, int] = {}
        self.idx_to_event: dict[int, str] = {}

        for idx, (event_type, name) in enumerate(EVENT_VOCAB.items()):
            self.event_to_idx[event_type] = idx
            self.idx_to_event[idx] = name

        self.pad_idx = 0
        self.vocab_size = len(EVENT_VOCAB)

    def tokenize_game(
        self,
        plays_df: pd.DataFrame,
        stints_df: pd.DataFrame | None = None,
    ) -> TokenizedSequence:
        """Tokenize a game's play-by-play into model inputs.

        Args:
            plays_df: DataFrame with columns:
                - event_type (int): EVENTMSGTYPE value
                - period (int): Game period (1-4 or OT)
                - pc_time (str): Period clock "MM:SS"
                - score_home (int): Home team score
                - score_away (int): Away team score
            stints_df: Optional DataFrame with lineup information for each event.
                If not provided, lineup embeddings will be zeros.

        Returns:
            TokenizedSequence with all tensors ready for model input.
        """

        # Filter to relevant events and sort by time
        plays = plays_df.copy()
        if "event_num" in plays.columns:
            plays = plays.sort_values("event_num")

        # Truncate to max_seq_len
        if len(plays) > self.max_seq_len:
            # Sample evenly from the game to capture full flow
            indices = np.linspace(0, len(plays) - 1, self.max_seq_len, dtype=int)
            plays = plays.iloc[indices].reset_index(drop=True)

        seq_len = len(plays)

        # Tokenize events
        events = self._tokenize_events(plays)

        # Normalize times
        times = self._normalize_times(plays)

        # Normalize score differentials
        scores = self._normalize_scores(plays)

        # Encode lineups
        lineups = self._encode_lineups(plays, stints_df)

        # Create attention mask (all valid, no padding)
        mask = torch.zeros(seq_len, dtype=torch.bool)

        return TokenizedSequence(
            events=events,
            times=times,
            scores=scores,
            lineups=lineups,
            mask=mask,
        )

    def _tokenize_events(self, plays: pd.DataFrame) -> torch.Tensor:
        """Convert event types to vocabulary indices.

        Args:
            plays: DataFrame with 'event_type' column.

        Returns:
            Tensor of event indices (seq_len,).
        """
        events = []
        for _, row in plays.iterrows():
            event_type = int(row.get("event_type", 14))  # Default to UNKNOWN
            idx = self.event_to_idx.get(event_type, self.event_to_idx[14])
            events.append(idx)
        return torch.tensor(events, dtype=torch.long)

    def _normalize_times(self, plays: pd.DataFrame) -> torch.Tensor:
        """Normalize time remaining to [0, 1].

        Args:
            plays: DataFrame with 'period' and 'pc_time' columns.

        Returns:
            Tensor of normalized times (seq_len, 1).
        """
        times = []
        for _, row in plays.iterrows():
            period = int(row.get("period", 1))
            pc_time = str(row.get("pc_time", "12:00"))

            # Parse minutes:seconds
            try:
                parts = pc_time.split(":")
                minutes = int(parts[0])
                seconds = int(parts[1]) if len(parts) > 1 else 0
                time_in_period = minutes * 60 + seconds  # Seconds remaining in period
            except (ValueError, IndexError):
                time_in_period = 720  # Default to 12 minutes

            # Calculate total time remaining (assuming 12 min periods, 4 periods)
            # Period 1: 48-36 min remaining, Period 4: 12-0 min remaining
            periods_remaining = max(0, 4 - period)
            total_seconds = periods_remaining * 720 + time_in_period
            normalized = total_seconds / (48 * 60)  # Normalize to [0, 1]
            times.append(normalized)

        return torch.tensor(times, dtype=torch.float32).unsqueeze(-1)

    def _normalize_scores(self, plays: pd.DataFrame) -> torch.Tensor:
        """Normalize score differential by historical std.

        Args:
            plays: DataFrame with 'score_home' and 'score_away' columns.

        Returns:
            Tensor of normalized score differentials (seq_len, 1).
        """
        scores = []
        for _, row in plays.iterrows():
            home = int(row.get("score_home", 0))
            away = int(row.get("score_away", 0))
            diff = (home - away) / self.score_std  # Z-score normalization
            scores.append(diff)

        return torch.tensor(scores, dtype=torch.float32).unsqueeze(-1)

    def _encode_lineups(
        self,
        plays: pd.DataFrame,
        stints_df: pd.DataFrame | None,
    ) -> torch.Tensor:
        """Encode lineup as player ID tensor (home 5 + away 5).

        Output ordering is fixed:
            - Dims 0-4: Home player IDs
            - Dims 5-9: Away player IDs

        Args:
            plays: DataFrame of play events with period and pc_time columns.
            stints_df: DataFrame with lineup info (home_lineup, away_lineup,
                period, start_time, end_time columns).

        Returns:
            Tensor of lineup player IDs (seq_len, 10).
        """
        import json

        seq_len = len(plays)

        if stints_df is None or stints_df.empty:
            # Return zeros if no lineup data available
            return torch.zeros(seq_len, 10, dtype=torch.long)

        # Build lineup encodings for each play
        lineup_encodings = []

        # Pre-process stints for faster lookup
        # Group stints by period
        stints_by_period: dict[int, list[dict]] = {}
        for _, stint in stints_df.iterrows():
            period = int(stint.get("period", 1))
            if period not in stints_by_period:
                stints_by_period[period] = []

            # Parse lineups
            home_lineup = stint.get("home_lineup", "[]")
            away_lineup = stint.get("away_lineup", "[]")
            if isinstance(home_lineup, str):
                try:
                    home_lineup = json.loads(home_lineup)
                except json.JSONDecodeError:
                    home_lineup = []
            if isinstance(away_lineup, str):
                try:
                    away_lineup = json.loads(away_lineup)
                except json.JSONDecodeError:
                    away_lineup = []

            # Parse times
            start_time = self._parse_time_to_seconds(
                str(stint.get("start_time", "12:00"))
            )
            end_time = self._parse_time_to_seconds(str(stint.get("end_time", "0:00")))

            stints_by_period[period].append(
                {
                    "home_lineup": home_lineup,
                    "away_lineup": away_lineup,
                    "start_time": start_time,
                    "end_time": end_time,
                }
            )

        # Encode each play
        for _, play in plays.iterrows():
            period = int(play.get("period", 1))
            pc_time = str(play.get("pc_time", "12:00"))
            play_time = self._parse_time_to_seconds(pc_time)

            # Find matching stint
            encoding = torch.zeros(10, dtype=torch.long)

            if period in stints_by_period:
                for stint in stints_by_period[period]:
                    # Check if play falls within stint time range
                    # Note: start_time > end_time since clock counts down
                    if stint["end_time"] <= play_time <= stint["start_time"]:
                        home_lineup = [int(pid) for pid in stint["home_lineup"]][:5]
                        away_lineup = [int(pid) for pid in stint["away_lineup"]][:5]

                        if len(home_lineup) < 5:
                            home_lineup.extend([0] * (5 - len(home_lineup)))
                        if len(away_lineup) < 5:
                            away_lineup.extend([0] * (5 - len(away_lineup)))

                        encoding[:5] = torch.tensor(home_lineup, dtype=torch.long)
                        encoding[5:] = torch.tensor(away_lineup, dtype=torch.long)

                        break

            lineup_encodings.append(encoding)

        return torch.stack(lineup_encodings)

    def _parse_time_to_seconds(self, time_str: str) -> int:
        """Parse MM:SS time string to seconds.

        Args:
            time_str: Time string in MM:SS format.

        Returns:
            Total seconds.
        """
        try:
            parts = time_str.split(":")
            minutes = int(parts[0])
            seconds = int(parts[1]) if len(parts) > 1 else 0
            return minutes * 60 + seconds
        except (ValueError, IndexError):
            return 720  # Default to 12 minutes

    def pad_sequence(
        self,
        tokens: TokenizedSequence,
        target_len: int | None = None,
    ) -> TokenizedSequence:
        """Pad a tokenized sequence to target length.

        Args:
            tokens: TokenizedSequence to pad.
            target_len: Target sequence length. Defaults to max_seq_len.

        Returns:
            Padded TokenizedSequence.
        """
        if target_len is None:
            target_len = self.max_seq_len

        current_len = tokens.events.size(0)
        if current_len >= target_len:
            return tokens

        pad_len = target_len - current_len

        # Pad events with pad_idx (0)
        events = torch.cat(
            [
                tokens.events,
                torch.zeros(pad_len, dtype=torch.long),
            ]
        )

        # Pad times with zeros
        times = torch.cat(
            [
                tokens.times,
                torch.zeros(pad_len, 1),
            ]
        )

        # Pad scores with zeros
        scores = torch.cat(
            [
                tokens.scores,
                torch.zeros(pad_len, 1),
            ]
        )

        # Pad lineups with zeros
        lineups = torch.cat(
            [
                tokens.lineups,
                torch.zeros(pad_len, 10, dtype=torch.long),
            ]
        )

        # Update mask to mark padded positions
        mask = torch.cat(
            [
                tokens.mask,
                torch.ones(pad_len, dtype=torch.bool),  # True = masked
            ]
        )

        return TokenizedSequence(
            events=events,
            times=times,
            scores=scores,
            lineups=lineups,
            mask=mask,
        )


def collate_sequences(
    sequences: list[TokenizedSequence],
) -> TokenizedSequence:
    """Collate multiple TokenizedSequences into a batch.

    Args:
        sequences: List of TokenizedSequence objects.

    Returns:
        Batched TokenizedSequence with shape (batch, seq_len, ...).
    """
    if not sequences:
        raise ValueError("Cannot collate empty sequence list")

    # Find max length
    max_len = max(seq.events.size(0) for seq in sequences)

    # Create tokenizer for padding
    tokenizer = EventTokenizer(max_seq_len=max_len)

    # Pad all sequences
    padded = [tokenizer.pad_sequence(seq, max_len) for seq in sequences]

    # Stack into batch
    return TokenizedSequence(
        events=torch.stack([s.events for s in padded]),
        times=torch.stack([s.times for s in padded]),
        scores=torch.stack([s.scores for s in padded]),
        lineups=torch.stack([s.lineups for s in padded]),
        mask=torch.stack([s.mask for s in padded]),
    )
