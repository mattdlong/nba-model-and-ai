"""Production inference pipeline for NBA game predictions.

This module implements the core prediction pipeline that generates game predictions
by loading trained models, building features, and running inference. It supports
both single-game and batch predictions with injury adjustments.

Latency Requirements:
    - Single game prediction: < 5 seconds
    - Full day predictions (~15 games): < 2 minutes

Example:
    >>> from nba_model.predict import InferencePipeline
    >>> from nba_model.models import ModelRegistry
    >>> from nba_model.data import session_scope
    >>> with session_scope() as session:
    ...     pipeline = InferencePipeline(ModelRegistry(), session)
    ...     prediction = pipeline.predict_game("0022300123")
    ...     print(f"{prediction.matchup}: {prediction.home_win_prob:.1%}")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING, Protocol, TypedDict

import pandas as pd
import torch

from nba_model.logging import get_logger
from nba_model.types import GameId, TeamId

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from nba_model.models.registry import ModelRegistry
    from nba_model.predict.injuries import AdjustmentValuesDict


class GameInfoDict(TypedDict):
    """Type for game info dictionary."""

    game_date: date
    home_team: str
    away_team: str
    home_team_id: int
    away_team_id: int


class ContextFeatureBuilderProtocol(Protocol):
    """Protocol for context feature builder dependency injection."""

    def build(self, game_id: GameId, db_session: Session) -> torch.Tensor: ...


class InjuryAdjusterProtocol(Protocol):
    """Protocol for injury adjuster dependency injection."""

    def adjust_prediction_values(
        self,
        game_id: GameId,
        base_home_win_prob: float,
        base_margin: float,
        base_total: float,
        home_lineup: list[int] | None = None,
        away_lineup: list[int] | None = None,
    ) -> AdjustmentValuesDict: ...


class LineupGraphBuilderProtocol(Protocol):
    """Protocol for lineup graph builder dependency injection."""

    def build_graph(
        self, home_lineup: list[int], away_lineup: list[int]
    ) -> torch.Tensor: ...


logger = get_logger(__name__)

# =============================================================================
# Constants
# =============================================================================

DEFAULT_CONTEXT_DIM: int = 32
DEFAULT_TRANSFORMER_DIM: int = 128
DEFAULT_GNN_DIM: int = 128
DEFAULT_SEQ_LEN: int = 50

# Reasonable bounds for predictions
MIN_WIN_PROB: float = 0.01
MAX_WIN_PROB: float = 0.99
MIN_MARGIN: float = -35.0
MAX_MARGIN: float = 35.0
MIN_TOTAL: float = 175.0
MAX_TOTAL: float = 270.0


# =============================================================================
# Exceptions
# =============================================================================


class InferenceError(Exception):
    """Base exception for inference errors."""


class ModelLoadError(InferenceError):
    """Error loading models from registry."""


class GameNotFoundError(InferenceError):
    """Game not found in database."""


class FeatureExtractionError(InferenceError):
    """Error extracting features for a game."""


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class GamePrediction:
    """Container for game prediction output.

    Attributes:
        game_id: NBA game identifier.
        game_date: Date of the game.
        home_team: Home team abbreviation.
        away_team: Away team abbreviation.
        matchup: Formatted matchup string (e.g., "LAL @ BOS").
        home_win_prob: Raw model probability of home team winning.
        predicted_margin: Raw model predicted point margin (home - away).
        predicted_total: Raw model predicted total points.
        home_win_prob_adjusted: Injury-adjusted home win probability.
        predicted_margin_adjusted: Injury-adjusted predicted margin.
        predicted_total_adjusted: Injury-adjusted predicted total.
        confidence: Model confidence score (0-1).
        injury_uncertainty: Uncertainty due to GTD players (0-1).
        top_factors: List of (feature_name, importance) tuples.
        model_version: Version of model used for prediction.
        prediction_timestamp: When prediction was generated.
        home_lineup: Expected home starting lineup.
        away_lineup: Expected away starting lineup.
        inference_time_ms: Time to generate prediction in milliseconds.
    """

    game_id: GameId
    game_date: date
    home_team: str
    away_team: str
    matchup: str

    # Raw model outputs
    home_win_prob: float
    predicted_margin: float
    predicted_total: float

    # Injury-adjusted outputs
    home_win_prob_adjusted: float
    predicted_margin_adjusted: float
    predicted_total_adjusted: float

    # Uncertainty quantification
    confidence: float
    injury_uncertainty: float

    # Explainability
    top_factors: list[tuple[str, float]] = field(default_factory=list)

    # Metadata
    model_version: str = ""
    prediction_timestamp: datetime = field(default_factory=datetime.now)
    home_lineup: list[str] = field(default_factory=list)
    away_lineup: list[str] = field(default_factory=list)
    inference_time_ms: float = 0.0


@dataclass
class PredictionBatch:
    """Container for batch prediction results.

    Attributes:
        predictions: List of individual game predictions.
        prediction_date: Date for which predictions were generated.
        total_games: Number of games predicted.
        model_version: Model version used.
        total_time_ms: Total time for batch prediction.
    """

    predictions: list[GamePrediction]
    prediction_date: date
    total_games: int
    model_version: str
    total_time_ms: float


# =============================================================================
# Inference Pipeline
# =============================================================================


class InferencePipeline:
    """Production inference pipeline for NBA game predictions.

    Orchestrates the full prediction flow:
    1. Load models from ModelRegistry
    2. Build context features for Tower A
    3. Construct lineup graphs for GNN
    4. Generate sequence representations for Transformer
    5. Run fusion model
    6. Apply injury adjustments

    Attributes:
        registry: ModelRegistry instance for loading models.
        db_session: SQLAlchemy database session.
        model_version: Version of models to use.
        device: Device for model inference.

    Example:
        >>> pipeline = InferencePipeline(registry, session)
        >>> prediction = pipeline.predict_game("0022300123")
        >>> predictions = pipeline.predict_today()
    """

    def __init__(
        self,
        model_registry: ModelRegistry,
        db_session: Session,
        model_version: str = "latest",
        device: torch.device | None = None,
        context_feature_builder: ContextFeatureBuilderProtocol | None = None,
        injury_adjuster: InjuryAdjusterProtocol | None = None,
        lineup_graph_builder: LineupGraphBuilderProtocol | None = None,
    ) -> None:
        """Initialize InferencePipeline.

        Args:
            model_registry: ModelRegistry for loading trained models.
            db_session: SQLAlchemy database session.
            model_version: Model version to load ("latest" or specific version).
            device: Device for inference. If None, auto-detects.
            context_feature_builder: Optional injected ContextFeatureBuilder.
                If None, creates a default instance.
            injury_adjuster: Optional injected InjuryAdjuster.
                If None, creates a default instance.
            lineup_graph_builder: Optional injected LineupGraphBuilder.
                If None, creates a default instance when needed.
        """
        self.registry = model_registry
        self.db_session = db_session
        self.model_version = model_version

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Store injected dependencies (lazy-load defaults if None)
        self._context_feature_builder = context_feature_builder
        self._injury_adjuster = injury_adjuster
        self._lineup_graph_builder = lineup_graph_builder

        # Lazy-load models
        self._models_loaded = False
        self._transformer: torch.nn.Module | None = None
        self._gnn: torch.nn.Module | None = None
        self._fusion: torch.nn.Module | None = None
        self._actual_version: str = ""

        logger.debug(
            "Initialized InferencePipeline with version {} on {}",
            model_version,
            self.device,
        )

    def _ensure_models_loaded(self) -> None:
        """Ensure models are loaded before inference."""
        if self._models_loaded:
            return

        try:
            self._load_models()
        except Exception as e:
            raise ModelLoadError(f"Failed to load models: {e}") from e

    def _load_models(self) -> None:
        """Load models from registry."""
        from nba_model.models import (
            GameFlowTransformer,
            PlayerInteractionGNN,
            TwoTowerFusion,
        )

        logger.info("Loading models version: {}", self.model_version)

        # Load model weights
        weights = self.registry.load_model(self.model_version)
        metadata = self.registry.load_metadata(self.model_version)

        if metadata:
            self._actual_version = metadata.version
        else:
            self._actual_version = self.model_version

        # Initialize models
        self._transformer = GameFlowTransformer()
        self._gnn = PlayerInteractionGNN()
        self._fusion = TwoTowerFusion()

        # Load weights if available
        if "transformer" in weights:
            self._transformer.load_state_dict(weights["transformer"])
        if "gnn" in weights:
            self._gnn.load_state_dict(weights["gnn"])
        if "fusion" in weights:
            self._fusion.load_state_dict(weights["fusion"])

        # Move to device and set eval mode
        self._transformer = self._transformer.to(self.device).eval()
        self._gnn = self._gnn.to(self.device).eval()
        self._fusion = self._fusion.to(self.device).eval()

        self._models_loaded = True
        logger.info("Models loaded successfully (version: {})", self._actual_version)

    def predict_game(
        self,
        game_id: GameId,
        apply_injury_adjustment: bool = True,
    ) -> GamePrediction:
        """Generate full prediction for a single game.

        Args:
            game_id: NBA game identifier.
            apply_injury_adjustment: Whether to apply injury adjustments.

        Returns:
            GamePrediction with full prediction details.

        Raises:
            GameNotFoundError: If game not found in database.
            InferenceError: If prediction fails.
        """
        start_time = time.perf_counter()

        # Ensure models are loaded
        self._ensure_models_loaded()

        # Get game info
        game_info = self._get_game_info(game_id)

        # Build features
        context_features = self._build_context_features(game_id)
        transformer_out = self._get_transformer_output(game_id)
        gnn_out = self._get_gnn_output(game_id)

        # Ensure fusion model is loaded
        assert self._fusion is not None

        # Run inference
        with torch.no_grad():
            outputs = self._fusion(
                context_features.unsqueeze(0).to(self.device),
                transformer_out.unsqueeze(0).to(self.device),
                gnn_out.unsqueeze(0).to(self.device),
            )

        # Extract predictions
        home_win_prob = float(outputs["win_prob"].squeeze().cpu())
        predicted_margin = float(outputs["margin"].squeeze().cpu())
        predicted_total = float(outputs["total"].squeeze().cpu())

        # Clamp to reasonable bounds
        home_win_prob = max(MIN_WIN_PROB, min(MAX_WIN_PROB, home_win_prob))
        predicted_margin = max(MIN_MARGIN, min(MAX_MARGIN, predicted_margin))
        predicted_total = max(MIN_TOTAL, min(MAX_TOTAL, predicted_total))

        # Calculate confidence based on probability distance from 0.5
        confidence = abs(home_win_prob - 0.5) * 2  # 0-1 scale

        # Get lineups
        home_lineup, away_lineup = self._get_expected_lineups(game_id)

        # Initialize adjusted values (will be updated by injury adjuster)
        home_win_prob_adjusted = home_win_prob
        predicted_margin_adjusted = predicted_margin
        predicted_total_adjusted = predicted_total
        injury_uncertainty = 0.0

        # Apply injury adjustment if requested
        if apply_injury_adjustment:
            try:
                adjuster = self._get_injury_adjuster()
                # Get lineup IDs for two-scenario model scoring
                home_lineup_ids, away_lineup_ids = self._get_expected_lineup_ids(
                    game_id
                )
                adjustment = adjuster.adjust_prediction_values(
                    game_id=game_id,
                    base_home_win_prob=home_win_prob,
                    base_margin=predicted_margin,
                    base_total=predicted_total,
                    home_lineup=home_lineup_ids,
                    away_lineup=away_lineup_ids,
                )
                home_win_prob_adjusted = adjustment["home_win_prob_adjusted"]
                predicted_margin_adjusted = adjustment["predicted_margin_adjusted"]
                predicted_total_adjusted = adjustment["predicted_total_adjusted"]
                injury_uncertainty = adjustment["injury_uncertainty"]
            except Exception as e:
                logger.warning("Injury adjustment failed for {}: {}", game_id, e)

        # Get top contributing factors
        top_factors = self._get_top_factors(context_features)

        inference_time_ms = (time.perf_counter() - start_time) * 1000

        return GamePrediction(
            game_id=game_id,
            game_date=game_info["game_date"],
            home_team=game_info["home_team"],
            away_team=game_info["away_team"],
            matchup=f"{game_info['away_team']} @ {game_info['home_team']}",
            home_win_prob=home_win_prob,
            predicted_margin=predicted_margin,
            predicted_total=predicted_total,
            home_win_prob_adjusted=home_win_prob_adjusted,
            predicted_margin_adjusted=predicted_margin_adjusted,
            predicted_total_adjusted=predicted_total_adjusted,
            confidence=confidence,
            injury_uncertainty=injury_uncertainty,
            top_factors=top_factors,
            model_version=self._actual_version,
            prediction_timestamp=datetime.now(),
            home_lineup=home_lineup,
            away_lineup=away_lineup,
            inference_time_ms=inference_time_ms,
        )

    def predict_today(self) -> list[GamePrediction]:
        """Generate predictions for all games scheduled for today.

        Returns:
            List of GamePrediction objects for today's games.
        """
        return self.predict_date(date.today())

    def predict_date(self, target_date: date) -> list[GamePrediction]:
        """Generate predictions for all games on a specific date.

        Args:
            target_date: Date to generate predictions for.

        Returns:
            List of GamePrediction objects.
        """
        start_time = time.perf_counter()

        # Get games for the date
        game_ids = self._get_games_for_date(target_date)

        if not game_ids:
            logger.info("No games found for {}", target_date)
            return []

        logger.info(
            "Generating predictions for {} games on {}", len(game_ids), target_date
        )

        predictions = []
        for game_id in game_ids:
            try:
                prediction = self.predict_game(game_id)
                predictions.append(prediction)
            except Exception as e:
                logger.error("Failed to predict game {}: {}", game_id, e)

        total_time_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "Generated {} predictions in {:.1f}ms",
            len(predictions),
            total_time_ms,
        )

        return predictions

    def _get_game_info(self, game_id: GameId) -> GameInfoDict:
        """Get basic game information from database."""
        from nba_model.data.models import Game, Team

        game = self.db_session.query(Game).filter(Game.game_id == game_id).first()

        if game is None:
            raise GameNotFoundError(f"Game {game_id} not found")

        # Get team abbreviations
        home_team = (
            self.db_session.query(Team)
            .filter(Team.team_id == game.home_team_id)
            .first()
        )
        away_team = (
            self.db_session.query(Team)
            .filter(Team.team_id == game.away_team_id)
            .first()
        )

        return {
            "game_date": game.game_date,
            "home_team": home_team.abbreviation if home_team else "UNK",
            "away_team": away_team.abbreviation if away_team else "UNK",
            "home_team_id": game.home_team_id,
            "away_team_id": game.away_team_id,
        }

    def _get_games_for_date(self, target_date: date) -> list[GameId]:
        """Get all game IDs for a specific date."""
        from nba_model.data.models import Game

        games = (
            self.db_session.query(Game.game_id)
            .filter(Game.game_date == target_date)
            .all()
        )

        return [g[0] for g in games]

    def _get_context_feature_builder(self) -> ContextFeatureBuilderProtocol:
        """Get or create the context feature builder."""
        if self._context_feature_builder is None:
            from nba_model.models.fusion import ContextFeatureBuilder

            self._context_feature_builder = ContextFeatureBuilder()
        return self._context_feature_builder

    def _get_injury_adjuster(self) -> InjuryAdjusterProtocol:
        """Get or create the injury adjuster with lineup scorer callback."""
        if self._injury_adjuster is not None:
            return self._injury_adjuster

        from nba_model.predict.injuries import InjuryAdjuster

        # Create lineup scorer callback for two-scenario model scoring
        adjuster: InjuryAdjusterProtocol = InjuryAdjuster(
            self.db_session,
            lineup_scorer=self._score_lineup,
        )
        self._injury_adjuster = adjuster
        return adjuster

    def _score_lineup(
        self,
        game_id: GameId,
        home_lineup: list[int],
        away_lineup: list[int],
    ) -> tuple[float, float, float]:
        """Score a lineup configuration using the fusion model.

        This callback is provided to InjuryAdjuster for true two-scenario
        model scoring.

        Args:
            game_id: Game identifier.
            home_lineup: Home team player IDs.
            away_lineup: Away team player IDs.

        Returns:
            Tuple of (home_win_prob, margin, total).
        """
        # Ensure models are loaded (this method is called during predict_game)
        assert self._gnn is not None
        assert self._fusion is not None

        # Build context features (these don't change between lineup scenarios)
        context_features = self._build_context_features(game_id)

        # Build player features DataFrame
        player_features_df = self._build_player_features_df()

        # Build GNN output for the specific lineup
        try:
            if len(home_lineup) >= 4 and len(away_lineup) >= 4:
                # Pad lineup to 5 if needed (with zeros for missing players)
                padded_home = home_lineup + [0] * (5 - len(home_lineup))
                padded_away = away_lineup + [0] * (5 - len(away_lineup))

                builder = self._get_lineup_graph_builder(player_features_df)
                graph = builder.build_graph(padded_home[:5], padded_away[:5])

                with torch.no_grad():
                    graph = graph.to(self.device)
                    gnn_out = self._gnn(graph).squeeze().cpu()
            else:
                gnn_out = torch.zeros(DEFAULT_GNN_DIM)
        except Exception as e:
            logger.debug("Lineup scoring GNN failed: {}", e)
            gnn_out = torch.zeros(DEFAULT_GNN_DIM)

        # Transformer output (zeros for pre-game)
        transformer_out = torch.zeros(DEFAULT_TRANSFORMER_DIM)

        # Run fusion model
        with torch.no_grad():
            outputs = self._fusion(
                context_features.unsqueeze(0).to(self.device),
                transformer_out.unsqueeze(0).to(self.device),
                gnn_out.unsqueeze(0).to(self.device),
            )

        home_win_prob = float(outputs["win_prob"].squeeze().cpu())
        margin = float(outputs["margin"].squeeze().cpu())
        total = float(outputs["total"].squeeze().cpu())

        # Clamp to reasonable bounds
        home_win_prob = max(MIN_WIN_PROB, min(MAX_WIN_PROB, home_win_prob))
        margin = max(MIN_MARGIN, min(MAX_MARGIN, margin))
        total = max(MIN_TOTAL, min(MAX_TOTAL, total))

        return home_win_prob, margin, total

    def _get_lineup_graph_builder(
        self, player_features_df: pd.DataFrame
    ) -> LineupGraphBuilderProtocol:
        """Get or create the lineup graph builder.

        Args:
            player_features_df: Player features DataFrame for building graphs.

        Returns:
            LineupGraphBuilder instance.
        """
        if self._lineup_graph_builder is None:
            from nba_model.models.gnn import LineupGraphBuilder

            self._lineup_graph_builder = LineupGraphBuilder(player_features_df)
        return self._lineup_graph_builder

    def _build_context_features(self, game_id: GameId) -> torch.Tensor:
        """Build context feature vector for Tower A."""
        builder = self._get_context_feature_builder()
        features = builder.build(game_id, self.db_session)
        return features

    def _get_transformer_output(self, game_id: GameId) -> torch.Tensor:
        """Get Transformer output for a game.

        For pre-game predictions, we use zeros since we don't have
        play-by-play data yet.
        """
        # For pre-game predictions, use zeros
        # In-game predictions would tokenize actual plays
        return torch.zeros(DEFAULT_TRANSFORMER_DIM)

    def _get_gnn_output(self, game_id: GameId) -> torch.Tensor:
        """Get GNN output for lineup graph."""
        try:
            # Ensure GNN model is loaded (called after _ensure_models_loaded)
            assert self._gnn is not None

            # Get expected lineups
            home_lineup, away_lineup = self._get_expected_lineup_ids(game_id)

            if len(home_lineup) == 5 and len(away_lineup) == 5:
                # Build player features DataFrame
                player_features_df = self._build_player_features_df()

                # Use injected or default lineup graph builder
                builder = self._get_lineup_graph_builder(player_features_df)
                graph = builder.build_graph(home_lineup, away_lineup)

                # Run GNN
                with torch.no_grad():
                    graph = graph.to(self.device)
                    gnn_out: torch.Tensor = self._gnn(graph)
                    return gnn_out.squeeze().cpu()

        except Exception as e:
            logger.debug("GNN inference failed, using zeros: {}", e)

        # Fallback to zeros
        return torch.zeros(DEFAULT_GNN_DIM)

    def _get_expected_lineups(self, game_id: GameId) -> tuple[list[str], list[str]]:
        """Get expected starting lineup names."""
        from nba_model.data.models import Player

        home_ids, away_ids = self._get_expected_lineup_ids(game_id)

        home_names = []
        for player_id in home_ids:
            player = (
                self.db_session.query(Player)
                .filter(Player.player_id == player_id)
                .first()
            )
            home_names.append(player.full_name if player else f"Player {player_id}")

        away_names = []
        for player_id in away_ids:
            player = (
                self.db_session.query(Player)
                .filter(Player.player_id == player_id)
                .first()
            )
            away_names.append(player.full_name if player else f"Player {player_id}")

        return home_names, away_names

    def _get_expected_lineup_ids(
        self,
        game_id: GameId,
    ) -> tuple[list[int], list[int]]:
        """Get expected starting lineup player IDs.

        Uses most recent game's starting lineup as proxy for expected lineup.
        """
        import json

        from nba_model.data.models import Game, Stint

        # Get game info
        game = self.db_session.query(Game).filter(Game.game_id == game_id).first()

        if game is None:
            return [], []

        # Try to get stints from this game first
        first_stint = (
            self.db_session.query(Stint)
            .filter(Stint.game_id == game_id)
            .order_by(Stint.period, Stint.start_time.desc())
            .first()
        )

        if first_stint:
            try:
                home = (
                    json.loads(first_stint.home_lineup)
                    if isinstance(first_stint.home_lineup, str)
                    else first_stint.home_lineup
                )
                away = (
                    json.loads(first_stint.away_lineup)
                    if isinstance(first_stint.away_lineup, str)
                    else first_stint.away_lineup
                )
                return home, away
            except (json.JSONDecodeError, TypeError):
                pass

        # Fallback: use most recent game's lineup for each team
        home_lineup = self._get_recent_lineup(game.home_team_id, game.game_date)
        away_lineup = self._get_recent_lineup(game.away_team_id, game.game_date)

        return home_lineup, away_lineup

    def _get_recent_lineup(self, team_id: TeamId, before_date: date) -> list[int]:
        """Get most recent starting lineup for a team."""
        import json

        from sqlalchemy import or_

        from nba_model.data.models import Game, Stint

        # Find most recent game for this team
        recent_game = (
            self.db_session.query(Game)
            .filter(
                or_(
                    Game.home_team_id == team_id,
                    Game.away_team_id == team_id,
                )
            )
            .filter(Game.game_date < before_date)
            .order_by(Game.game_date.desc())
            .first()
        )

        if recent_game is None:
            return []

        # Get first stint
        stint = (
            self.db_session.query(Stint)
            .filter(Stint.game_id == recent_game.game_id)
            .order_by(Stint.period, Stint.start_time.desc())
            .first()
        )

        if stint is None:
            return []

        try:
            is_home = recent_game.home_team_id == team_id
            lineup_str = stint.home_lineup if is_home else stint.away_lineup
            lineup = (
                json.loads(lineup_str) if isinstance(lineup_str, str) else lineup_str
            )
            return lineup if isinstance(lineup, list) else []
        except (json.JSONDecodeError, TypeError):
            return []

    def _build_player_features_df(self) -> pd.DataFrame:
        """Build player features DataFrame for GNN."""
        from nba_model.data.models import Player, PlayerRAPM

        # Query player data
        players = self.db_session.query(Player).all()

        data = []
        for player in players:
            # Get most recent RAPM
            rapm = (
                self.db_session.query(PlayerRAPM)
                .filter(PlayerRAPM.player_id == player.player_id)
                .order_by(PlayerRAPM.calculation_date.desc())
                .first()
            )

            data.append(
                {
                    "player_id": player.player_id,
                    "height_inches": player.height_inches or 78,  # Default 6'6"
                    "weight_lbs": player.weight_lbs or 210,
                    "orapm": rapm.orapm if rapm else 0.0,
                    "drapm": rapm.drapm if rapm else 0.0,
                    "rapm": rapm.rapm if rapm else 0.0,
                }
            )

        return pd.DataFrame(data).set_index("player_id")

    def _get_top_factors(
        self,
        context_features: torch.Tensor,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Get top contributing factors from context features."""
        from nba_model.models.fusion import CONTEXT_FEATURES

        # Use absolute feature values as simple importance proxy
        values = context_features.abs().numpy()

        # Get indices of top-k features
        top_indices = values.argsort()[-top_k:][::-1]

        factors = []
        for idx in top_indices:
            if idx < len(CONTEXT_FEATURES):
                name = CONTEXT_FEATURES[idx]
                value = float(context_features[idx])
                factors.append((name, value))

        return factors


# =============================================================================
# Utility Functions
# =============================================================================


def create_mock_pipeline(
    db_session: Session,
    model_version: str = "latest",
) -> InferencePipeline:
    """Create an inference pipeline with mock models for testing.

    Args:
        db_session: Database session.
        model_version: Model version string.

    Returns:
        InferencePipeline with mock models.
    """
    from nba_model.models import ModelRegistry

    registry = ModelRegistry()
    return InferencePipeline(registry, db_session, model_version)
