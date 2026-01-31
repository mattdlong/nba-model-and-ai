"""Base collector class for NBA data collection.

This module provides the abstract base class that all collectors inherit from,
providing common functionality for progress logging and error handling.

Example:
    >>> class MyCollector(BaseCollector):
    ...     def collect_game(self, game_id: str) -> list[Model]:
    ...         # Implementation here
    ...         pass
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from nba_model.data.api import NBAApiClient


class BaseCollector(ABC):
    """Base class for all data collectors.

    Provides common functionality:
    - API client and session injection
    - Progress logging
    - Error handling with configurable strategies
    - Checkpointing for resumable collection

    Attributes:
        api: NBA API client instance.
        session: SQLAlchemy database session.
        logger: Logger instance for this collector.
        _last_checkpoint: Last successfully processed identifier.
    """

    def __init__(
        self,
        api_client: NBAApiClient,
        db_session: Session | None = None,
    ) -> None:
        """Initialize collector.

        Args:
            api_client: NBA API client instance.
            db_session: Optional SQLAlchemy session for DB operations.
        """
        self.api = api_client
        self.session = db_session
        self.logger = logging.getLogger(self.__class__.__name__)
        self._last_checkpoint: str | None = None

    @abstractmethod
    def collect_game(self, game_id: str) -> Any:
        """Collect data for a single game.

        Args:
            game_id: NBA game ID string.

        Returns:
            Collected data (type depends on collector).
        """
        pass

    def collect(
        self,
        season_range: list[str],
        resume_from: str | None = None,
    ) -> Any:
        """Collect data for a range of seasons.

        This is the primary collection method that supports checkpointing
        and resumption from a specific point.

        Args:
            season_range: List of season strings (e.g., ["2022-23", "2023-24"]).
            resume_from: Optional identifier to resume from (e.g., game_id).

        Returns:
            Collected data (type depends on collector).

        Note:
            Subclasses should override this method to implement
            season-based collection logic.
        """
        self.logger.warning(
            f"collect() not implemented for {self.__class__.__name__}. "
            "Use collector-specific methods instead."
        )
        return []

    def get_last_checkpoint(self) -> str | None:
        """Get the last successfully processed identifier.

        Returns:
            The last checkpoint identifier (e.g., game_id) or None if
            no checkpoint has been set.
        """
        return self._last_checkpoint

    def set_checkpoint(self, checkpoint: str) -> None:
        """Set the current checkpoint.

        Args:
            checkpoint: Identifier of the last successfully processed item.
        """
        self._last_checkpoint = checkpoint
        self.logger.debug(f"Checkpoint set: {checkpoint}")

    def _log_progress(
        self,
        current: int,
        total: int,
        item_id: str,
    ) -> None:
        """Log collection progress.

        Args:
            current: Current item number (1-indexed).
            total: Total number of items.
            item_id: ID of current item being processed.
        """
        pct = (current / total * 100) if total > 0 else 0
        self.logger.info(f"Progress: {current}/{total} ({pct:.1f}%) - {item_id}")

    def _handle_error(
        self,
        error: Exception,
        item_id: str,
        on_error: Literal["raise", "skip", "log"],
    ) -> None:
        """Handle collection error based on strategy.

        Args:
            error: The exception that occurred.
            item_id: ID of the item that failed.
            on_error: Error handling strategy:
                - "raise": Re-raise the exception
                - "skip": Silently skip the item
                - "log": Log the error and continue
        """
        if on_error == "raise":
            raise error
        elif on_error == "log":
            self.logger.error(f"Error collecting {item_id}: {error}")
        # "skip" does nothing - just continues
