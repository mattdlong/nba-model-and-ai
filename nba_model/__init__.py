"""NBA Quantitative Trading Strategy.

A Python CLI application integrating Transformer models, Graph Neural Networks,
and a Two-Tower fusion architecture to predict NBA game outcomes and generate
betting signals.

Example:
    >>> from nba_model.config import get_settings
    >>> settings = get_settings()
    >>> print(settings.db_path)
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "NBA Model Team"

# Public API exports
from nba_model.config import Settings, get_settings

__all__ = [
    "Settings",
    "__author__",
    "__version__",
    "get_settings",
]
