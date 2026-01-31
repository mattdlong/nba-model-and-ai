"""Training pipeline for NBA prediction models.

This module implements the training loop for the full fusion model, including
multi-task loss optimization, early stopping, checkpointing, and metrics logging.

Training Protocol:
    - Optimizer: AdamW with weight decay (1e-5)
    - Learning rate: 1e-4 with cosine annealing or reduce-on-plateau
    - Gradient clipping: Max norm 1.0
    - Early stopping: Patience 10 epochs
    - Checkpointing: Save best model by validation loss

Example:
    >>> from nba_model.models.trainer import FusionTrainer
    >>> trainer = FusionTrainer(transformer, gnn, fusion)
    >>> history = trainer.fit(train_loader, val_loader, epochs=50)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from nba_model.config import get_settings
from nba_model.logging import get_logger
from nba_model.models.fusion import MultiTaskLoss

if TYPE_CHECKING:
    from nba_model.models.transformer import GameFlowTransformer
    from nba_model.models.gnn import PlayerInteractionGNN
    from nba_model.models.fusion import TwoTowerFusion

logger = get_logger(__name__)

# =============================================================================
# Constants
# =============================================================================

DEFAULT_LEARNING_RATE: float = 1e-4
DEFAULT_WEIGHT_DECAY: float = 1e-5
DEFAULT_EPOCHS: int = 50
DEFAULT_PATIENCE: int = 10
DEFAULT_GRADIENT_CLIP: float = 1.0
DEFAULT_BATCH_SIZE: int = 32


# =============================================================================
# Data Containers
# =============================================================================


@dataclass
class TrainingConfig:
    """Configuration for model training.

    Attributes:
        learning_rate: Learning rate for optimizer.
        weight_decay: Weight decay for regularization.
        epochs: Maximum number of training epochs.
        patience: Early stopping patience.
        gradient_clip: Maximum gradient norm for clipping.
        batch_size: Training batch size.
        scheduler: Learning rate scheduler type ('plateau' or 'cosine').
        checkpoint_dir: Directory for saving model checkpoints.
    """

    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    epochs: int = DEFAULT_EPOCHS
    patience: int = DEFAULT_PATIENCE
    gradient_clip: float = DEFAULT_GRADIENT_CLIP
    batch_size: int = DEFAULT_BATCH_SIZE
    scheduler: str = "plateau"
    checkpoint_dir: Path = field(default_factory=lambda: Path("data/models"))

    @classmethod
    def from_settings(cls) -> TrainingConfig:
        """Create config from application settings."""
        settings = get_settings()
        return cls(
            learning_rate=settings.learning_rate,
            batch_size=settings.batch_size,
            checkpoint_dir=settings.model_dir_obj,
        )


@dataclass
class EpochMetrics:
    """Metrics for a single training epoch.

    Attributes:
        epoch: Epoch number (0-indexed).
        train_loss: Average training loss.
        val_loss: Average validation loss.
        win_accuracy: Win prediction accuracy.
        margin_mae: Mean absolute error for margin.
        total_mae: Mean absolute error for total.
        brier_score: Probability calibration score.
        log_loss: Cross-entropy loss.
        learning_rate: Current learning rate.
    """

    epoch: int
    train_loss: float
    val_loss: float
    win_accuracy: float = 0.0
    margin_mae: float = 0.0
    total_mae: float = 0.0
    brier_score: float = 0.0
    log_loss: float = 0.0
    learning_rate: float = 0.0


@dataclass
class TrainingHistory:
    """Container for training history.

    Attributes:
        epochs: List of epoch metrics.
        best_epoch: Epoch with best validation loss.
        best_val_loss: Best validation loss achieved.
        total_time: Total training time in seconds.
    """

    epochs: list[EpochMetrics] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    total_time: float = 0.0

    def add_epoch(self, metrics: EpochMetrics) -> None:
        """Add epoch metrics to history."""
        self.epochs.append(metrics)
        if metrics.val_loss < self.best_val_loss:
            self.best_val_loss = metrics.val_loss
            self.best_epoch = metrics.epoch


# =============================================================================
# Trainer
# =============================================================================


class FusionTrainer:
    """Trainer for the full fusion model pipeline.

    Coordinates training of Transformer, GNN, and Fusion models together
    with multi-task optimization and proper validation.

    Attributes:
        transformer: GameFlowTransformer model.
        gnn: PlayerInteractionGNN model.
        fusion: TwoTowerFusion model.
        config: Training configuration.
        device: Device to train on (CPU or CUDA).

    Example:
        >>> trainer = FusionTrainer(transformer, gnn, fusion)
        >>> history = trainer.fit(train_loader, val_loader, epochs=50)
        >>> print(f"Best epoch: {history.best_epoch}")
    """

    def __init__(
        self,
        transformer: GameFlowTransformer,
        gnn: PlayerInteractionGNN,
        fusion: TwoTowerFusion,
        config: TrainingConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Initialize FusionTrainer.

        Args:
            transformer: GameFlowTransformer model instance.
            gnn: PlayerInteractionGNN model instance.
            fusion: TwoTowerFusion model instance.
            config: Training configuration. If None, uses defaults.
            device: Device to train on. If None, auto-detects.
        """
        self.transformer = transformer
        self.gnn = gnn
        self.fusion = fusion
        self.config = config or TrainingConfig()

        # Detect device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Move models to device
        self.transformer.to(self.device)
        self.gnn.to(self.device)
        self.fusion.to(self.device)

        # Initialize loss function
        self.loss_fn = MultiTaskLoss(learnable_weights=False)

        # Initialize optimizer with all model parameters
        self.optimizer = AdamW(
            list(self.transformer.parameters())
            + list(self.gnn.parameters())
            + list(self.fusion.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Initialize scheduler
        if self.config.scheduler == "plateau":
            self.scheduler: ReduceLROnPlateau | CosineAnnealingLR = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=5,
                verbose=True,
            )
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
            )

        # Training state
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

        logger.info(
            "Initialized FusionTrainer on {}, lr={}, batch_size={}",
            self.device,
            self.config.learning_rate,
            self.config.batch_size,
        )

    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        """Train for one epoch.

        Args:
            dataloader: Training data loader.

        Returns:
            Dictionary of average loss components.
        """
        self.transformer.train()
        self.gnn.train()
        self.fusion.train()

        total_loss = 0.0
        total_win_loss = 0.0
        total_margin_loss = 0.0
        total_total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self._forward_batch(batch)
            labels = self._extract_labels(batch)

            # Compute loss
            loss, components = self.loss_fn(outputs, labels)

            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.transformer.parameters())
                + list(self.gnn.parameters())
                + list(self.fusion.parameters()),
                self.config.gradient_clip,
            )
            self.optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            total_win_loss += components["win_loss"]
            total_margin_loss += components["margin_loss"]
            total_total_loss += components["total_loss"]
            num_batches += 1

        return {
            "loss": total_loss / max(num_batches, 1),
            "win_loss": total_win_loss / max(num_batches, 1),
            "margin_loss": total_margin_loss / max(num_batches, 1),
            "total_loss": total_total_loss / max(num_batches, 1),
        }

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """Evaluate on validation set.

        Args:
            dataloader: Validation data loader.

        Returns:
            Dictionary of evaluation metrics.
        """
        self.transformer.eval()
        self.gnn.eval()
        self.fusion.eval()

        total_loss = 0.0
        all_win_probs: list[float] = []
        all_win_labels: list[int] = []
        all_margins: list[float] = []
        all_margin_labels: list[float] = []
        all_totals: list[float] = []
        all_total_labels: list[float] = []
        num_batches = 0

        for batch in dataloader:
            # Forward pass
            outputs = self._forward_batch(batch)
            labels = self._extract_labels(batch)

            # Compute loss
            loss, _ = self.loss_fn(outputs, labels)
            total_loss += loss.item()
            num_batches += 1

            # Collect predictions for metrics
            win_probs = outputs["win_prob"].cpu().numpy().flatten()
            margins = outputs["margin"].cpu().numpy().flatten()
            totals = outputs["total"].cpu().numpy().flatten()

            win_labels = labels["win"].cpu().numpy().flatten()
            margin_labels = labels["margin"].cpu().numpy().flatten()
            total_labels = labels["total"].cpu().numpy().flatten()

            all_win_probs.extend(win_probs.tolist())
            all_win_labels.extend(win_labels.astype(int).tolist())
            all_margins.extend(margins.tolist())
            all_margin_labels.extend(margin_labels.tolist())
            all_totals.extend(totals.tolist())
            all_total_labels.extend(total_labels.tolist())

        # Calculate metrics
        metrics = self._calculate_metrics(
            all_win_probs,
            all_win_labels,
            all_margins,
            all_margin_labels,
            all_totals,
            all_total_labels,
        )
        metrics["loss"] = total_loss / max(num_batches, 1)

        return metrics

    def _forward_batch(self, batch: dict) -> dict[str, torch.Tensor]:
        """Forward pass for a batch of data.

        Args:
            batch: Dictionary containing batch data.

        Returns:
            Model outputs dictionary.
        """
        # Extract and move data to device
        context = batch["context"].to(self.device)

        # Process sequence data through Transformer
        events = batch["events"].to(self.device)
        times = batch["times"].to(self.device)
        scores = batch["scores"].to(self.device)
        lineups = batch["lineups"].to(self.device)
        mask = batch.get("mask")
        if mask is not None:
            mask = mask.to(self.device)

        transformer_out = self.transformer(events, times, scores, lineups, mask)

        # Process graph data through GNN
        graphs = batch["graphs"]
        if isinstance(graphs, list):
            graphs = Batch.from_data_list(graphs)
        graphs = graphs.to(self.device)
        gnn_out = self.gnn(graphs)

        # Fusion
        outputs = self.fusion(context, transformer_out, gnn_out)

        return outputs

    def _extract_labels(self, batch: dict) -> dict[str, torch.Tensor]:
        """Extract labels from batch.

        Args:
            batch: Dictionary containing batch data.

        Returns:
            Labels dictionary with 'win', 'margin', 'total'.
        """
        return {
            "win": batch["win_label"].to(self.device),
            "margin": batch["margin_label"].to(self.device),
            "total": batch["total_label"].to(self.device),
        }

    def _calculate_metrics(
        self,
        win_probs: list[float],
        win_labels: list[int],
        margins: list[float],
        margin_labels: list[float],
        totals: list[float],
        total_labels: list[float],
    ) -> dict[str, float]:
        """Calculate evaluation metrics.

        Args:
            win_probs: Predicted win probabilities.
            win_labels: Actual win outcomes.
            margins: Predicted margins.
            margin_labels: Actual margins.
            totals: Predicted totals.
            total_labels: Actual totals.

        Returns:
            Dictionary of metrics.
        """
        import numpy as np

        win_probs_arr = np.array(win_probs)
        win_labels_arr = np.array(win_labels)
        margins_arr = np.array(margins)
        margin_labels_arr = np.array(margin_labels)
        totals_arr = np.array(totals)
        total_labels_arr = np.array(total_labels)

        # Win accuracy
        predictions = (win_probs_arr > 0.5).astype(int)
        accuracy = np.mean(predictions == win_labels_arr)

        # MAE for margin and total
        margin_mae = np.mean(np.abs(margins_arr - margin_labels_arr))
        total_mae = np.mean(np.abs(totals_arr - total_labels_arr))

        # Brier score (lower is better)
        brier_score = np.mean((win_probs_arr - win_labels_arr) ** 2)

        # Log loss (cross-entropy)
        eps = 1e-7
        win_probs_clipped = np.clip(win_probs_arr, eps, 1 - eps)
        log_loss = -np.mean(
            win_labels_arr * np.log(win_probs_clipped)
            + (1 - win_labels_arr) * np.log(1 - win_probs_clipped)
        )

        return {
            "accuracy": float(accuracy),
            "margin_mae": float(margin_mae),
            "total_mae": float(total_mae),
            "brier_score": float(brier_score),
            "log_loss": float(log_loss),
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int | None = None,
        checkpoint_dir: Path | None = None,
    ) -> TrainingHistory:
        """Full training loop with early stopping.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            epochs: Maximum epochs. Uses config default if None.
            checkpoint_dir: Directory for checkpoints. Uses config default if None.

        Returns:
            TrainingHistory with all epoch metrics.
        """
        import time

        epochs = epochs or self.config.epochs
        checkpoint_dir = checkpoint_dir or self.config.checkpoint_dir
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        history = TrainingHistory()
        start_time = time.time()

        logger.info("Starting training for {} epochs", epochs)

        for epoch in range(epochs):
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader)

            # Validate
            val_metrics = self.evaluate(val_loader)

            # Update scheduler
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_metrics["loss"])
            else:
                self.scheduler.step()

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Create epoch metrics
            epoch_metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_metrics["loss"],
                val_loss=val_metrics["loss"],
                win_accuracy=val_metrics["accuracy"],
                margin_mae=val_metrics["margin_mae"],
                total_mae=val_metrics["total_mae"],
                brier_score=val_metrics["brier_score"],
                log_loss=val_metrics["log_loss"],
                learning_rate=current_lr,
            )
            history.add_epoch(epoch_metrics)

            # Check for improvement
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.epochs_without_improvement = 0
                # Save checkpoint
                self._save_checkpoint(checkpoint_dir / "best_model.pt")
                logger.info(
                    "Epoch {}: New best val_loss={:.4f}, saving checkpoint",
                    epoch,
                    val_metrics["loss"],
                )
            else:
                self.epochs_without_improvement += 1

            # Log progress
            epoch_time = time.time() - epoch_start
            logger.info(
                "Epoch {}/{}: train_loss={:.4f}, val_loss={:.4f}, "
                "accuracy={:.3f}, brier={:.4f}, time={:.1f}s",
                epoch + 1,
                epochs,
                train_metrics["loss"],
                val_metrics["loss"],
                val_metrics["accuracy"],
                val_metrics["brier_score"],
                epoch_time,
            )

            # Early stopping
            if self.epochs_without_improvement >= self.config.patience:
                logger.info(
                    "Early stopping at epoch {} (no improvement for {} epochs)",
                    epoch,
                    self.config.patience,
                )
                break

        history.total_time = time.time() - start_time
        logger.info(
            "Training complete: best_epoch={}, best_val_loss={:.4f}, time={:.1f}s",
            history.best_epoch,
            history.best_val_loss,
            history.total_time,
        )

        return history

    def _save_checkpoint(self, path: Path) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint.
        """
        checkpoint = {
            "transformer_state_dict": self.transformer.state_dict(),
            "gnn_state_dict": self.gnn.state_dict(),
            "fusion_state_dict": self.fusion.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "timestamp": datetime.now().isoformat(),
        }
        torch.save(checkpoint, path)
        logger.debug("Saved checkpoint to {}", path)

    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.transformer.load_state_dict(checkpoint["transformer_state_dict"])
        self.gnn.load_state_dict(checkpoint["gnn_state_dict"])
        self.fusion.load_state_dict(checkpoint["fusion_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        logger.info("Loaded checkpoint from {}", path)

    def save_models(self, output_dir: Path) -> None:
        """Save individual model weights.

        Args:
            output_dir: Directory to save models.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.transformer.state_dict(), output_dir / "transformer.pt")
        torch.save(self.gnn.state_dict(), output_dir / "gnn.pt")
        torch.save(self.fusion.state_dict(), output_dir / "fusion.pt")

        logger.info("Saved models to {}", output_dir)

    def load_models(self, input_dir: Path) -> None:
        """Load individual model weights.

        Args:
            input_dir: Directory containing model files.
        """
        self.transformer.load_state_dict(
            torch.load(input_dir / "transformer.pt", map_location=self.device)
        )
        self.gnn.load_state_dict(
            torch.load(input_dir / "gnn.pt", map_location=self.device)
        )
        self.fusion.load_state_dict(
            torch.load(input_dir / "fusion.pt", map_location=self.device)
        )

        logger.info("Loaded models from {}", input_dir)
