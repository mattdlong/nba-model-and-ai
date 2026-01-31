"""Event parser for play-by-play text extraction.

This module implements regex-based parsing of play-by-play event descriptions
to extract structured features such as turnover type, shot context, and
shot clock usage.

NBA play-by-play descriptions follow consistent patterns that allow for
reliable extraction of game context information.

Example:
    >>> from nba_model.features.parsing import EventParser
    >>> parser = EventParser()
    >>> context = parser.parse_shot_context("LeBron James Driving Layup")
    >>> print(context['shot_type'])
    'driving'
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from nba_model.logging import get_logger

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)


class TurnoverType(Enum):
    """Turnover classification."""

    UNFORCED = "unforced"
    FORCED = "forced"
    UNKNOWN = "unknown"


class ShotType(Enum):
    """Shot type classification."""

    DRIVING = "driving"
    PULLUP = "pullup"
    STEPBACK = "stepback"
    CATCH_SHOOT = "catch_shoot"
    FLOATING = "floating"
    FADEAWAY = "fadeaway"
    HOOK = "hook"
    TURNAROUND = "turnaround"
    TIP = "tip"
    DUNK = "dunk"
    LAYUP = "layup"
    OTHER = "other"


class ShotClockCategory(Enum):
    """Shot clock usage category."""

    EARLY = "early"  # < 8 seconds used
    MID = "mid"  # 8-16 seconds used
    LATE = "late"  # > 16 seconds used
    UNKNOWN = "unknown"


@dataclass
class ShotContext:
    """Container for parsed shot context information."""

    shot_type: ShotType
    is_transition: bool
    is_contested: bool
    is_assisted: bool
    shot_clock_category: ShotClockCategory


@dataclass
class TurnoverContext:
    """Container for parsed turnover information."""

    turnover_type: TurnoverType
    is_steal: bool
    handler_error: bool


# Regex patterns for play-by-play parsing
# Patterns are case-insensitive


class EventPatterns:
    """Compiled regex patterns for event parsing."""

    # Turnover patterns
    BAD_PASS = re.compile(r"Bad Pass", re.IGNORECASE)
    LOST_BALL = re.compile(r"Lost Ball", re.IGNORECASE)
    OUT_OF_BOUNDS = re.compile(r"Out of Bounds", re.IGNORECASE)
    TRAVELING = re.compile(r"Traveling|Travel Violation", re.IGNORECASE)
    OFFENSIVE_FOUL = re.compile(r"Offensive Foul|Charge", re.IGNORECASE)
    DOUBLE_DRIBBLE = re.compile(r"Double Dribble|Discontinue Dribble", re.IGNORECASE)
    SHOT_CLOCK = re.compile(r"Shot Clock", re.IGNORECASE)
    STEAL = re.compile(r"STEAL|STL", re.IGNORECASE)

    # Shot type patterns
    DRIVING = re.compile(r"Driving|Drive", re.IGNORECASE)
    PULLUP = re.compile(r"Pull[-\s]?[Uu]p", re.IGNORECASE)
    STEP_BACK = re.compile(r"Step[-\s]?[Bb]ack", re.IGNORECASE)
    CATCH_SHOOT = re.compile(r"Catch and Shoot|C&S", re.IGNORECASE)
    FLOATING = re.compile(r"Floating|Floater|Float", re.IGNORECASE)
    FADEAWAY = re.compile(r"Fade[-\s]?[Aa]way|Fade", re.IGNORECASE)
    HOOK = re.compile(r"Hook Shot|Hook", re.IGNORECASE)
    TURNAROUND = re.compile(r"Turn[-\s]?[Aa]round", re.IGNORECASE)
    TIP = re.compile(r"Tip Shot|Tip[-\s]?[Ii]n|Putback", re.IGNORECASE)
    DUNK = re.compile(r"Dunk|Slam", re.IGNORECASE)
    LAYUP = re.compile(r"Layup|Lay[-\s]?[Uu]p|Finger Roll", re.IGNORECASE)
    JUMP_SHOT = re.compile(r"Jump Shot|Jumper", re.IGNORECASE)

    # Context patterns
    TRANSITION = re.compile(r"Fast Break|Transition|In Transition", re.IGNORECASE)
    CONTESTED = re.compile(r"Contested", re.IGNORECASE)
    ASSISTED = re.compile(r"AST|Assist", re.IGNORECASE)
    BLOCK = re.compile(r"BLOCK|BLK", re.IGNORECASE)
    THREE_POINT = re.compile(r"3PT|Three Point|3-Point", re.IGNORECASE)


class EventParser:
    """Parser for extracting structured data from play-by-play descriptions.

    Uses regex patterns to identify turnover types, shot contexts, and
    other game events from NBA play-by-play text.

    Example:
        >>> parser = EventParser()
        >>> turnover = parser.parse_turnover_type("Bad Pass Turnover")
        >>> print(turnover)
        TurnoverType.UNFORCED
    """

    def __init__(self) -> None:
        """Initialize event parser with compiled patterns."""
        self.patterns = EventPatterns()

    def parse_turnover_type(self, description: str | None) -> TurnoverType:
        """Classify turnover as unforced or forced.

        Unforced turnovers: bad passes, traveling, offensive fouls
        Forced turnovers: lost balls, steals involved

        Args:
            description: Play-by-play description text.

        Returns:
            TurnoverType enum value.
        """
        if not description:
            return TurnoverType.UNKNOWN

        # Unforced turnovers
        if self.patterns.BAD_PASS.search(description):
            return TurnoverType.UNFORCED
        if self.patterns.TRAVELING.search(description):
            return TurnoverType.UNFORCED
        if self.patterns.OFFENSIVE_FOUL.search(description):
            return TurnoverType.UNFORCED
        if self.patterns.DOUBLE_DRIBBLE.search(description):
            return TurnoverType.UNFORCED
        if self.patterns.SHOT_CLOCK.search(description):
            return TurnoverType.UNFORCED
        if self.patterns.OUT_OF_BOUNDS.search(description):
            # Out of bounds can be either, default to unforced
            return TurnoverType.UNFORCED

        # Forced turnovers
        if self.patterns.LOST_BALL.search(description):
            return TurnoverType.FORCED
        if self.patterns.STEAL.search(description):
            return TurnoverType.FORCED

        return TurnoverType.UNKNOWN

    def parse_turnover_context(self, description: str | None) -> TurnoverContext:
        """Parse detailed turnover context.

        Args:
            description: Play-by-play description text.

        Returns:
            TurnoverContext with type, steal, and handler error info.
        """
        if not description:
            return TurnoverContext(
                turnover_type=TurnoverType.UNKNOWN,
                is_steal=False,
                handler_error=False,
            )

        turnover_type = self.parse_turnover_type(description)
        is_steal = bool(self.patterns.STEAL.search(description))

        # Handler error = bad pass or lost ball by ball handler
        handler_error = bool(
            self.patterns.BAD_PASS.search(description)
            or self.patterns.LOST_BALL.search(description)
        )

        return TurnoverContext(
            turnover_type=turnover_type,
            is_steal=is_steal,
            handler_error=handler_error,
        )

    def parse_shot_type(self, description: str | None) -> ShotType:
        """Extract shot type from description.

        Priority order handles overlapping patterns (e.g., "Driving Layup"
        classified as driving, not layup).

        Args:
            description: Play-by-play description text.

        Returns:
            ShotType enum value.
        """
        if not description:
            return ShotType.OTHER

        # Check patterns in priority order
        if self.patterns.TIP.search(description):
            return ShotType.TIP
        if self.patterns.DUNK.search(description):
            return ShotType.DUNK
        if self.patterns.DRIVING.search(description):
            return ShotType.DRIVING
        if self.patterns.PULLUP.search(description):
            return ShotType.PULLUP
        if self.patterns.STEP_BACK.search(description):
            return ShotType.STEPBACK
        if self.patterns.CATCH_SHOOT.search(description):
            return ShotType.CATCH_SHOOT
        if self.patterns.FLOATING.search(description):
            return ShotType.FLOATING
        if self.patterns.FADEAWAY.search(description):
            return ShotType.FADEAWAY
        if self.patterns.HOOK.search(description):
            return ShotType.HOOK
        if self.patterns.TURNAROUND.search(description):
            return ShotType.TURNAROUND
        if self.patterns.LAYUP.search(description):
            return ShotType.LAYUP

        return ShotType.OTHER

    def parse_shot_context(self, description: str | None) -> ShotContext:
        """Extract full shot context from description.

        Args:
            description: Play-by-play description text.

        Returns:
            ShotContext with shot type, transition, contested flags.
        """
        if not description:
            return ShotContext(
                shot_type=ShotType.OTHER,
                is_transition=False,
                is_contested=False,
                is_assisted=False,
                shot_clock_category=ShotClockCategory.UNKNOWN,
            )

        shot_type = self.parse_shot_type(description)
        is_transition = bool(self.patterns.TRANSITION.search(description))
        is_contested = bool(self.patterns.CONTESTED.search(description))
        is_assisted = bool(self.patterns.ASSISTED.search(description))

        return ShotContext(
            shot_type=shot_type,
            is_transition=is_transition,
            is_contested=is_contested,
            is_assisted=is_assisted,
            shot_clock_category=ShotClockCategory.UNKNOWN,  # Set separately
        )

    def _parse_time_string(self, time_str: str | None) -> int | None:
        """Parse time string to seconds.

        Args:
            time_str: Time in "MM:SS" format.

        Returns:
            Total seconds or None if invalid.
        """
        if not time_str or not isinstance(time_str, str):
            return None

        try:
            parts = time_str.strip().split(":")
            if len(parts) == 2:
                minutes = int(parts[0])
                seconds = int(parts[1])
                return minutes * 60 + seconds
        except (ValueError, IndexError):
            pass

        return None

    def calculate_shot_clock_usage(
        self,
        plays_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Calculate shot clock usage for each play.

        Estimates time used on possession before shot by tracking time
        between possession start (rebound, inbound, turnover recovery)
        and shot attempt.

        Args:
            plays_df: DataFrame with columns:
                - pc_time: Period clock time ("MM:SS")
                - event_type: NBA event type code
                - period: Game period

        Returns:
            DataFrame with additional column 'shot_clock_category'.
        """

        result_df = plays_df.copy()
        result_df["shot_clock_category"] = ShotClockCategory.UNKNOWN.value

        # Event type codes (from NBA API)
        shot_made = 1
        shot_miss = 2
        free_throw = 3
        rebound = 4
        turnover = 5

        shot_events = {shot_made, shot_miss}
        # Possession starts on: rebounds, turnovers (recovery), and inbounds
        # after free throws (when opponent gets ball after made FT)
        possession_start_events = {rebound, turnover, free_throw}

        # Process each period separately
        for period in result_df["period"].unique():
            period_mask = result_df["period"] == period
            period_plays = result_df[period_mask].copy()

            possession_start_time: int | None = None

            for idx in period_plays.index:
                row = result_df.loc[idx]
                event_type = row.get("event_type")
                pc_time = self._parse_time_string(row.get("pc_time"))

                if pc_time is None:
                    continue

                # Track possession start
                if event_type in possession_start_events:
                    possession_start_time = pc_time
                elif event_type in shot_events:
                    # Calculate time used
                    if possession_start_time is not None:
                        time_used = possession_start_time - pc_time

                        if time_used < 0:
                            # Handle period transitions
                            time_used = 0

                        # Categorize
                        if time_used < 8:
                            category = ShotClockCategory.EARLY
                        elif time_used <= 16:
                            category = ShotClockCategory.MID
                        else:
                            category = ShotClockCategory.LATE

                        result_df.at[idx, "shot_clock_category"] = category.value

                    # Reset possession tracking on makes
                    if event_type == shot_made:
                        possession_start_time = pc_time

        return result_df

    def parse_all_descriptions(
        self,
        plays_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Parse all descriptions in a play-by-play DataFrame.

        Adds columns for turnover type and shot context.

        Args:
            plays_df: DataFrame with columns:
                - home_description, away_description, neutral_description

        Returns:
            DataFrame with additional parsed columns.
        """
        result_df = plays_df.copy()

        # Combine descriptions
        def combine_descriptions(row) -> str:
            parts = []
            for col in ["home_description", "away_description", "neutral_description"]:
                if row.get(col):
                    parts.append(str(row[col]))
            return " | ".join(parts)

        result_df["combined_description"] = result_df.apply(
            combine_descriptions, axis=1
        )

        # Parse turnovers
        result_df["turnover_type"] = result_df["combined_description"].apply(
            lambda x: self.parse_turnover_type(x).value
        )

        # Parse shot context
        def extract_shot_type(desc: str) -> str:
            return self.parse_shot_type(desc).value

        def extract_is_transition(desc: str) -> bool:
            return bool(self.patterns.TRANSITION.search(desc)) if desc else False

        def extract_is_contested(desc: str) -> bool:
            return bool(self.patterns.CONTESTED.search(desc)) if desc else False

        result_df["shot_type"] = result_df["combined_description"].apply(
            extract_shot_type
        )
        result_df["is_transition"] = result_df["combined_description"].apply(
            extract_is_transition
        )
        result_df["is_contested"] = result_df["combined_description"].apply(
            extract_is_contested
        )

        return result_df


def parse_turnover_type(description: str | None) -> TurnoverType:
    """Convenience function to parse turnover type.

    Args:
        description: Play-by-play description text.

    Returns:
        TurnoverType enum value.
    """
    parser = EventParser()
    return parser.parse_turnover_type(description)


def parse_shot_context(description: str | None) -> ShotContext:
    """Convenience function to parse shot context.

    Args:
        description: Play-by-play description text.

    Returns:
        ShotContext dataclass.
    """
    parser = EventParser()
    return parser.parse_shot_context(description)


__all__ = [
    "EventParser",
    "EventPatterns",
    "ShotClockCategory",
    "ShotContext",
    "ShotType",
    "TurnoverContext",
    "TurnoverType",
    "parse_shot_context",
    "parse_turnover_type",
]
