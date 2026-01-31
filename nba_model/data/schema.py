"""SQLAlchemy base class and mixins for database models.

This module provides the declarative base and shared mixins used by all
SQLAlchemy model classes in the application.

Example:
    >>> from nba_model.data.schema import Base, TimestampMixin
    >>> class MyModel(TimestampMixin, Base):
    ...     __tablename__ = "my_table"
    ...     id: Mapped[int] = mapped_column(primary_key=True)
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import event, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all models.

    All ORM models should inherit from this base class to ensure
    consistent metadata and table creation.
    """

    pass


class TimestampMixin:
    """Mixin that adds created_at and updated_at timestamp columns.

    These columns are automatically set on insert and update operations.

    Attributes:
        created_at: Timestamp when record was created.
        updated_at: Timestamp when record was last updated.
    """

    created_at: Mapped[datetime] = mapped_column(
        default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


def _set_updated_at(
    mapper: Any,
    connection: Any,
    target: Any,
) -> None:
    """Event listener to update updated_at on modification."""
    if hasattr(target, "updated_at"):
        target.updated_at = datetime.now()


# Register the event listener for before_update events
# This ensures updated_at is set even when using session.execute()
event.listen(Base, "before_update", _set_updated_at, propagate=True)
