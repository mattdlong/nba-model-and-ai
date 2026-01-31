"""PyTorch Dataset implementations for NBA game prediction.

This module provides Dataset classes that assemble complete training samples
from the database, including tokenized sequences, graph data, context features,
and labels.

Data Sources:
    - Play-by-play events (for Transformer)
    - Lineup information (for GNN graphs)
    - Team/player statistics (for context features)
    - Game outcomes (for labels)

Example:
    >>> from nba_model.models.dataset import NBADataset, nba_collate_fn
    >>> dataset = NBADataset.from_season("2023-24", db_session)
    >>> loader = DataLoader(dataset, batch_size=32, collate_fn=nba_collate_fn)
    >>> for batch in loader:
    ...     outputs = model(**batch)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
import numpy as np

from nba_model.logging import get_logger
from nba_model.types import GameId, SeasonId

if TYPE_CHECKING:
    import pandas as pd
    from sqlalchemy.orm import Session

logger = get_logger(__name__)

# =============================================================================
# Constants
# =============================================================================

DEFAULT_SEQ_LEN: int = 50
DEFAULT_CONTEXT_DIM: int = 32
DEFAULT_NODE_FEATURES: int = 16


# =============================================================================
# Data Containers
# =============================================================================


@dataclass
class GameSample:
    """Container for a single game's training data.

    Attributes:
        game_id: NBA game ID.
        events: Event type indices tensor (seq_len,).
        times: Normalized time tensor (seq_len, 1).
        scores: Normalized score differential tensor (seq_len, 1).
        lineups: Lineup encoding tensor (seq_len, 20).
        mask: Attention mask tensor (seq_len,).
        graph: PyG Data object for lineup graph.
        context: Context feature tensor (context_dim,).
        win_label: Binary win label (1 = home win).
        margin_label: Point margin (home - away).
        total_label: Total points scored.
    """

    game_id: GameId
    events: torch.Tensor
    times: torch.Tensor
    scores: torch.Tensor
    lineups: torch.Tensor
    mask: torch.Tensor
    graph: Data
    context: torch.Tensor
    win_label: float
    margin_label: float
    total_label: float


# =============================================================================
# NBA Dataset
# =============================================================================


class NBADataset(Dataset[GameSample]):
    """PyTorch Dataset for NBA game prediction.

    Assembles complete training samples from raw data, including:
    - Tokenized play-by-play sequences (for Transformer)
    - Lineup graphs (for GNN)
    - Context features (for Tower A)
    - Labels (win, margin, total)

    Attributes:
        game_ids: List of game IDs in the dataset.
        games_df: DataFrame with game information.
        plays_df: DataFrame with play-by-play events.
        context_df: DataFrame with context features.
        player_features_df: DataFrame with player features.
        seq_len: Maximum sequence length.

    Example:
        >>> dataset = NBADataset(games_df, plays_df, context_df, player_df)
        >>> sample = dataset[0]
        >>> print(sample.game_id, sample.win_label)
    """

    def __init__(
        self,
        games_df: pd.DataFrame,
        plays_df: pd.DataFrame | None = None,
        context_df: pd.DataFrame | None = None,
        player_features_df: pd.DataFrame | None = None,
        seq_len: int = DEFAULT_SEQ_LEN,
        context_dim: int = DEFAULT_CONTEXT_DIM,
    ) -> None:
        """Initialize NBADataset.

        Args:
            games_df: DataFrame with game info. Required columns:
                - game_id: NBA game ID
                - home_score: Home team final score
                - away_score: Away team final score
            plays_df: DataFrame with play-by-play events. Optional columns:
                - game_id: Game ID
                - event_type: EVENTMSGTYPE value
                - period: Game period
                - pc_time: Period clock
                - score_home, score_away: Running scores
            context_df: DataFrame with pre-computed context features.
            player_features_df: DataFrame with player features for GNN.
            seq_len: Maximum sequence length for Transformer.
            context_dim: Dimension of context feature vector.
        """
        import pandas as pd

        self.games_df = games_df
        self.plays_df = plays_df
        self.context_df = context_df
        self.player_features_df = player_features_df
        self.seq_len = seq_len
        self.context_dim = context_dim

        # Extract game IDs
        self.game_ids: list[GameId] = games_df["game_id"].tolist()

        # Index DataFrames by game_id for fast lookup
        if plays_df is not None and "game_id" in plays_df.columns:
            self._plays_grouped = plays_df.groupby("game_id")
        else:
            self._plays_grouped = None

        logger.info(
            "Initialized NBADataset with {} games, seq_len={}",
            len(self.game_ids),
            seq_len,
        )

    def __len__(self) -> int:
        """Return number of games in dataset."""
        return len(self.game_ids)

    def __getitem__(self, idx: int) -> GameSample:
        """Get a single game sample.

        Args:
            idx: Index into the dataset.

        Returns:
            GameSample with all data for the game.
        """
        game_id = self.game_ids[idx]
        game_row = self.games_df[self.games_df["game_id"] == game_id].iloc[0]

        # Get labels
        home_score = int(game_row.get("home_score", 0))
        away_score = int(game_row.get("away_score", 0))
        win_label = 1.0 if home_score > away_score else 0.0
        margin_label = float(home_score - away_score)
        total_label = float(home_score + away_score)

        # Get sequence data
        events, times, scores, lineups, mask = self._get_sequence_data(game_id)

        # Get graph data
        graph = self._get_graph_data(game_id, game_row)

        # Get context features
        context = self._get_context_features(game_id)

        return GameSample(
            game_id=game_id,
            events=events,
            times=times,
            scores=scores,
            lineups=lineups,
            mask=mask,
            graph=graph,
            context=context,
            win_label=win_label,
            margin_label=margin_label,
            total_label=total_label,
        )

    def _get_sequence_data(
        self,
        game_id: GameId,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get tokenized sequence data for a game.

        Args:
            game_id: NBA game ID.

        Returns:
            Tuple of (events, times, scores, lineups, mask) tensors.
        """
        from nba_model.models.transformer import EventTokenizer

        tokenizer = EventTokenizer(max_seq_len=self.seq_len)

        if self._plays_grouped is not None:
            try:
                plays = self._plays_grouped.get_group(game_id)
            except KeyError:
                plays = None
        else:
            plays = None

        if plays is None or plays.empty:
            # Return dummy sequence for games without play-by-play
            return self._create_dummy_sequence()

        # Tokenize
        tokens = tokenizer.tokenize_game(plays, stints_df=None)

        # Pad to seq_len
        tokens = tokenizer.pad_sequence(tokens, self.seq_len)

        return (
            tokens.events,
            tokens.times,
            tokens.scores,
            tokens.lineups,
            tokens.mask,
        )

    def _create_dummy_sequence(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create dummy sequence for games without play-by-play data.

        Returns:
            Tuple of zero-filled tensors with correct shapes.
        """
        events = torch.zeros(self.seq_len, dtype=torch.long)
        times = torch.zeros(self.seq_len, 1, dtype=torch.float32)
        scores = torch.zeros(self.seq_len, 1, dtype=torch.float32)
        lineups = torch.zeros(self.seq_len, 20, dtype=torch.float32)
        mask = torch.ones(self.seq_len, dtype=torch.bool)  # All masked
        return events, times, scores, lineups, mask

    def _get_graph_data(self, game_id: GameId, game_row: pd.Series) -> Data:
        """Get lineup graph for a game.

        Args:
            game_id: NBA game ID.
            game_row: Row from games DataFrame.

        Returns:
            PyG Data object with lineup graph.
        """
        from nba_model.models.gnn import LineupGraphBuilder, create_empty_graph

        # Try to get lineup from game data
        home_lineup = game_row.get("home_lineup", None)
        away_lineup = game_row.get("away_lineup", None)

        if home_lineup is None or away_lineup is None:
            return create_empty_graph()

        # Parse lineups if they're strings
        if isinstance(home_lineup, str):
            import json

            try:
                home_lineup = json.loads(home_lineup)
            except json.JSONDecodeError:
                return create_empty_graph()

        if isinstance(away_lineup, str):
            import json

            try:
                away_lineup = json.loads(away_lineup)
            except json.JSONDecodeError:
                return create_empty_graph()

        # Validate lineup sizes
        if len(home_lineup) != 5 or len(away_lineup) != 5:
            return create_empty_graph()

        # Build graph
        builder = LineupGraphBuilder(self.player_features_df)
        try:
            graph = builder.build_graph(home_lineup, away_lineup)
        except Exception as e:
            logger.debug("Failed to build graph for {}: {}", game_id, e)
            return create_empty_graph()

        return graph

    def _get_context_features(self, game_id: GameId) -> torch.Tensor:
        """Get context features for a game.

        Args:
            game_id: NBA game ID.

        Returns:
            Context feature tensor of shape (context_dim,).
        """
        if self.context_df is None:
            return torch.zeros(self.context_dim, dtype=torch.float32)

        if "game_id" in self.context_df.columns:
            row = self.context_df[self.context_df["game_id"] == game_id]
        else:
            row = self.context_df.loc[[game_id]] if game_id in self.context_df.index else None

        if row is None or row.empty:
            return torch.zeros(self.context_dim, dtype=torch.float32)

        # Extract numeric columns
        row = row.iloc[0]
        features = []
        for col in self.context_df.columns:
            if col != "game_id":
                try:
                    features.append(float(row[col]))
                except (ValueError, TypeError):
                    features.append(0.0)

        # Pad or truncate to context_dim
        while len(features) < self.context_dim:
            features.append(0.0)
        features = features[: self.context_dim]

        return torch.tensor(features, dtype=torch.float32)

    @classmethod
    def from_season(
        cls,
        season_id: SeasonId,
        db_session: Session,
        seq_len: int = DEFAULT_SEQ_LEN,
    ) -> NBADataset:
        """Create dataset from a season's data in the database.

        Args:
            season_id: Season ID (e.g., "2023-24").
            db_session: SQLAlchemy database session.
            seq_len: Maximum sequence length.

        Returns:
            NBADataset instance.
        """
        import pandas as pd

        from nba_model.data.models import Game, Play

        # Query games
        games = (
            db_session.query(Game)
            .filter(Game.season_id == season_id)
            .filter(Game.status == "completed")
            .all()
        )

        games_data = [
            {
                "game_id": g.game_id,
                "season_id": g.season_id,
                "home_team_id": g.home_team_id,
                "away_team_id": g.away_team_id,
                "home_score": g.home_score,
                "away_score": g.away_score,
            }
            for g in games
        ]
        games_df = pd.DataFrame(games_data)

        if games_df.empty:
            logger.warning("No completed games found for season {}", season_id)
            return cls(games_df, seq_len=seq_len)

        # Query plays
        game_ids = games_df["game_id"].tolist()
        plays = (
            db_session.query(Play)
            .filter(Play.game_id.in_(game_ids))
            .all()
        )

        plays_data = [
            {
                "game_id": p.game_id,
                "event_num": p.event_num,
                "period": p.period,
                "pc_time": p.pc_time,
                "event_type": p.event_type,
                "score_home": p.score_home,
                "score_away": p.score_away,
            }
            for p in plays
        ]
        plays_df = pd.DataFrame(plays_data) if plays_data else None

        logger.info(
            "Created dataset for {} with {} games, {} plays",
            season_id,
            len(games_df),
            len(plays_df) if plays_df is not None else 0,
        )

        return cls(games_df, plays_df, seq_len=seq_len)

    @classmethod
    def from_dataframes(
        cls,
        games_df: pd.DataFrame,
        plays_df: pd.DataFrame | None = None,
        context_df: pd.DataFrame | None = None,
        player_features_df: pd.DataFrame | None = None,
        seq_len: int = DEFAULT_SEQ_LEN,
    ) -> NBADataset:
        """Create dataset from DataFrames.

        Convenience factory method for creating datasets from pre-loaded data.

        Args:
            games_df: Games DataFrame.
            plays_df: Plays DataFrame.
            context_df: Context features DataFrame.
            player_features_df: Player features DataFrame.
            seq_len: Maximum sequence length.

        Returns:
            NBADataset instance.
        """
        return cls(
            games_df=games_df,
            plays_df=plays_df,
            context_df=context_df,
            player_features_df=player_features_df,
            seq_len=seq_len,
        )


# =============================================================================
# Collate Function
# =============================================================================


def nba_collate_fn(samples: list[GameSample]) -> dict:
    """Custom collate function for NBADataset.

    Handles batching of variable-length sequences and PyG graph data.

    Args:
        samples: List of GameSample objects.

    Returns:
        Dictionary with batched tensors ready for model input.
    """
    if not samples:
        raise ValueError("Cannot collate empty sample list")

    # Stack sequence tensors
    events = torch.stack([s.events for s in samples])
    times = torch.stack([s.times for s in samples])
    scores = torch.stack([s.scores for s in samples])
    lineups = torch.stack([s.lineups for s in samples])
    masks = torch.stack([s.mask for s in samples])

    # Batch graphs
    graphs = Batch.from_data_list([s.graph for s in samples])

    # Stack context features
    context = torch.stack([s.context for s in samples])

    # Stack labels
    win_labels = torch.tensor([s.win_label for s in samples], dtype=torch.float32)
    margin_labels = torch.tensor([s.margin_label for s in samples], dtype=torch.float32)
    total_labels = torch.tensor([s.total_label for s in samples], dtype=torch.float32)

    return {
        "game_ids": [s.game_id for s in samples],
        "events": events,
        "times": times,
        "scores": scores,
        "lineups": lineups,
        "mask": masks,
        "graphs": graphs,
        "context": context,
        "win_label": win_labels.unsqueeze(-1),
        "margin_label": margin_labels.unsqueeze(-1),
        "total_label": total_labels.unsqueeze(-1),
    }


# =============================================================================
# Data Splitting
# =============================================================================


def temporal_split(
    dataset: NBADataset,
    val_ratio: float = 0.2,
    games_df: pd.DataFrame | None = None,
) -> tuple[NBADataset, NBADataset]:
    """Split dataset temporally (earlier games for train, later for val).

    This prevents data leakage by ensuring all training games occur before
    validation games.

    Args:
        dataset: NBADataset to split.
        val_ratio: Fraction of games for validation (from end of season).
        games_df: DataFrame with game dates. If None, uses dataset.games_df.

    Returns:
        Tuple of (train_dataset, val_dataset).
    """
    import pandas as pd

    games_df = games_df if games_df is not None else dataset.games_df

    # Sort by date
    if "game_date" in games_df.columns:
        sorted_df = games_df.sort_values("game_date")
    else:
        sorted_df = games_df  # Assume already in order

    # Split by position
    n_val = int(len(sorted_df) * val_ratio)
    train_df = sorted_df.iloc[:-n_val] if n_val > 0 else sorted_df
    val_df = sorted_df.iloc[-n_val:] if n_val > 0 else pd.DataFrame(columns=sorted_df.columns)

    # Filter plays for each split
    train_ids = set(train_df["game_id"].tolist())
    val_ids = set(val_df["game_id"].tolist())

    train_plays = None
    val_plays = None
    if dataset.plays_df is not None:
        train_plays = dataset.plays_df[dataset.plays_df["game_id"].isin(train_ids)]
        val_plays = dataset.plays_df[dataset.plays_df["game_id"].isin(val_ids)]

    train_context = None
    val_context = None
    if dataset.context_df is not None:
        if "game_id" in dataset.context_df.columns:
            train_context = dataset.context_df[dataset.context_df["game_id"].isin(train_ids)]
            val_context = dataset.context_df[dataset.context_df["game_id"].isin(val_ids)]

    train_dataset = NBADataset(
        games_df=train_df,
        plays_df=train_plays,
        context_df=train_context,
        player_features_df=dataset.player_features_df,
        seq_len=dataset.seq_len,
        context_dim=dataset.context_dim,
    )

    val_dataset = NBADataset(
        games_df=val_df,
        plays_df=val_plays,
        context_df=val_context,
        player_features_df=dataset.player_features_df,
        seq_len=dataset.seq_len,
        context_dim=dataset.context_dim,
    )

    logger.info(
        "Split dataset: {} train, {} val",
        len(train_dataset),
        len(val_dataset),
    )

    return train_dataset, val_dataset


def create_data_loader(
    dataset: NBADataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader for the NBADataset.

    Args:
        dataset: NBADataset instance.
        batch_size: Batch size.
        shuffle: Whether to shuffle data.
        num_workers: Number of worker processes.

    Returns:
        DataLoader with custom collate function.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=nba_collate_fn,
    )
