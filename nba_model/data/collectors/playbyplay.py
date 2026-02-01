"""Play-by-play collector for NBA game events.

This module provides the PlayByPlayCollector class for fetching
play-by-play data from the NBA API.

Example:
    >>> from nba_model.data.collectors import PlayByPlayCollector
    >>> collector = PlayByPlayCollector(api_client, session)
    >>> plays = collector.collect_game("0022300001")
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pandas as pd

from nba_model.data.collectors.base import BaseCollector
from nba_model.data.models import Play

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from nba_model.data.api import NBAApiClient


# =============================================================================
# Event Type Constants
# =============================================================================

EVENT_TYPES: dict[int, str] = {
    1: "FIELD_GOAL_MADE",
    2: "FIELD_GOAL_MISSED",
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
    18: "INSTANT_REPLAY",
    20: "STOPPAGE",
}


class PlayByPlayCollector(BaseCollector):
    """Collects play-by-play event data for games.

    Each game has ~300-500 events with event types, descriptions,
    timestamps, and player references.
    """

    def __init__(
        self,
        api_client: NBAApiClient,
        db_session: Session | None = None,
    ) -> None:
        """Initialize play-by-play collector.

        Args:
            api_client: NBA API client instance.
            db_session: Optional SQLAlchemy session.
        """
        super().__init__(api_client, db_session)

    def collect(
        self,
        season_range: list[str],
        resume_from: str | None = None,
    ) -> dict[str, list[Play]]:
        """Collect play-by-play for all games in a season range.

        Note: This requires game IDs which should be collected first
        using GamesCollector. Use collect_games() with specific game IDs.

        Args:
            season_range: List of season strings.
            resume_from: Optional game_id to resume from.

        Returns:
            Dict mapping game_id to list of Play instances.
        """
        self.logger.warning(
            "collect() for PlayByPlayCollector requires game IDs. "
            "Use collect_games() with specific game IDs instead."
        )
        return {}

    def collect_game(self, game_id: str) -> list[Play]:
        """Collect play-by-play for a single game.

        Args:
            game_id: NBA game ID.

        Returns:
            List of Play model instances (deduplicated by event_num).
        """
        self.logger.debug(f"Collecting play-by-play for game {game_id}")

        try:
            df = self.api.get_play_by_play(game_id=game_id)

            if df.empty:
                self.logger.warning(f"No play-by-play data for game {game_id}")
                return []

            # Deduplicate plays by event_num (unique key is game_id + event_num)
            plays_by_event: dict[int, Play] = {}
            for _, row in df.iterrows():
                play = self._transform_play(row, game_id)
                if play and play.event_num not in plays_by_event:
                    plays_by_event[play.event_num] = play

            plays = list(plays_by_event.values())
            self.logger.debug(f"Collected {len(plays)} plays for game {game_id}")
            return plays

        except Exception as e:
            self.logger.error(f"Error collecting play-by-play for {game_id}: {e}")
            raise

    def collect_games(
        self,
        game_ids: list[str],
        on_error: Literal["raise", "skip", "log"] = "log",
        resume_from: str | None = None,
    ) -> dict[str, list[Play]]:
        """Collect play-by-play for multiple games.

        Args:
            game_ids: List of game IDs.
            on_error: Error handling strategy:
                - "raise": Stop on first error
                - "skip": Skip failed games silently
                - "log": Log errors and continue
            resume_from: Optional game_id to resume from.

        Returns:
            Dict mapping game_id to list of Play instances.
        """
        self.logger.info(f"Collecting play-by-play for {len(game_ids)} games")

        results: dict[str, list[Play]] = {}

        # Handle resume_from
        start_collecting = resume_from is None
        if resume_from and resume_from in game_ids:
            start_idx = game_ids.index(resume_from)
            game_ids = game_ids[start_idx:]
            start_collecting = True

        for i, game_id in enumerate(game_ids, 1):
            self._log_progress(i, len(game_ids), game_id)

            try:
                plays = self.collect_game(game_id)
                results[game_id] = plays
                # Set checkpoint after successful collection
                self.set_checkpoint(game_id)
            except Exception as e:
                self._handle_error(e, game_id, on_error)
                if on_error == "raise":
                    raise

        self.logger.info(
            f"Collected play-by-play for {len(results)}/{len(game_ids)} games"
        )
        return results

    def _transform_play(self, row: pd.Series, game_id: str) -> Play | None:
        """Transform API response row to Play model.

        Args:
            row: DataFrame row from PlayByPlayV2.
            game_id: NBA game ID.

        Returns:
            Play model instance or None.
        """
        try:
            # Extract player IDs
            player1_id, player2_id, player3_id = self._extract_player_ids(row)

            # Parse time
            pc_time = self._parse_time(row.get("PCTIMESTRING", ""))

            # Extract event number
            event_num = int(row.get("EVENTNUM", 0))

            # Parse score
            score_str = row.get("SCORE", "")
            score_home, score_away = self._parse_score(score_str)

            return Play(
                game_id=game_id,
                event_num=event_num,
                period=int(row.get("PERIOD", 1)),
                pc_time=pc_time,
                wc_time=str(row.get("WCTIMESTRING", "")) or None,
                event_type=int(row.get("EVENTMSGTYPE", 0)),
                event_action=self._safe_int(row.get("EVENTMSGACTIONTYPE")),
                home_description=str(row.get("HOMEDESCRIPTION", "")) or None,
                away_description=str(row.get("VISITORDESCRIPTION", "")) or None,
                neutral_description=str(row.get("NEUTRALDESCRIPTION", "")) or None,
                score_home=score_home,
                score_away=score_away,
                player1_id=player1_id,
                player2_id=player2_id,
                player3_id=player3_id,
                team_id=self._safe_int(row.get("PLAYER1_TEAM_ID")),
            )

        except Exception as e:
            self.logger.error(f"Error transforming play: {e}")
            return None

    def _parse_time(self, pctimestring: str | None) -> str | None:
        """Parse time string to consistent format.

        Args:
            pctimestring: Time string from API (e.g., "11:45").

        Returns:
            Normalized time string or None.
        """
        if not pctimestring or pd.isna(pctimestring):
            return None
        return str(pctimestring).strip()

    def _parse_score(self, score_str: str | None) -> tuple[int | None, int | None]:
        """Parse score string to home and away scores.

        Args:
            score_str: Score string (e.g., "100 - 95").

        Returns:
            Tuple of (score_home, score_away) or (None, None).
        """
        if not score_str or pd.isna(score_str):
            return None, None

        try:
            # Format: "AWAY - HOME" (visitor score first)
            parts = str(score_str).split(" - ")
            if len(parts) == 2:
                return int(parts[1]), int(parts[0])
        except (ValueError, IndexError):
            pass

        return None, None

    def _is_team_id(self, id_value: int | None) -> bool:
        """Check if an ID is a team ID rather than a player ID.

        NBA team IDs are in the range 1610612737-1610612766 (30 teams).
        Player IDs are outside this range.

        Args:
            id_value: ID to check.

        Returns:
            True if this is a team ID, False otherwise.
        """
        if id_value is None:
            return False
        # NBA team IDs range from 1610612737 (ATL) to 1610612766 (WAS)
        return 1610612737 <= id_value <= 1610612766

    def _extract_player_ids(
        self,
        row: pd.Series,
        player_name_to_id: dict[str, int] | None = None,
    ) -> tuple[int | None, int | None, int | None]:
        """Extract player IDs from event data.

        First tries to get player IDs from the explicit PLAYER_ID fields.
        Falls back to extracting from event descriptions if not present.
        Filters out team IDs that are sometimes placed in player ID fields
        (e.g., for team rebounds).

        Args:
            row: DataFrame row.
            player_name_to_id: Optional mapping of player names to IDs for fallback.

        Returns:
            Tuple of (player1_id, player2_id, player3_id).
        """
        player1_id = self._safe_int(row.get("PLAYER1_ID"))
        player2_id = self._safe_int(row.get("PLAYER2_ID"))
        player3_id = self._safe_int(row.get("PLAYER3_ID"))

        # Filter out team IDs - these are incorrectly placed in player fields
        # for events like team rebounds
        if self._is_team_id(player1_id):
            player1_id = None
        if self._is_team_id(player2_id):
            player2_id = None
        if self._is_team_id(player3_id):
            player3_id = None

        # Fallback to description parsing when explicit IDs are missing
        if player_name_to_id and player1_id is None:
            # Try home description first, then away, then neutral
            for desc_field in ["HOMEDESCRIPTION", "VISITORDESCRIPTION", "NEUTRALDESCRIPTION"]:
                description = row.get(desc_field)
                if description and not pd.isna(description):
                    extracted_ids = self.extract_player_references_from_description(
                        str(description), player_name_to_id
                    )
                    if extracted_ids:
                        # Assign extracted IDs to missing slots
                        if player1_id is None and len(extracted_ids) >= 1:
                            player1_id = extracted_ids[0]
                        if player2_id is None and len(extracted_ids) >= 2:
                            player2_id = extracted_ids[1]
                        if player3_id is None and len(extracted_ids) >= 3:
                            player3_id = extracted_ids[2]
                        break

        return player1_id, player2_id, player3_id

    def extract_player_references_from_description(
        self,
        description: str | None,
        player_name_to_id: dict[str, int] | None = None,
    ) -> list[int]:
        """Extract player IDs from event descriptions.

        Parses descriptions like:
        - "Curry 25' 3PT Shot: Made" -> ["Curry"]
        - "LeBron Driving Layup" -> ["LeBron"]
        - "Curry 6' Floating Jump Shot (Assist: Thompson)" -> ["Curry", "Thompson"]
        - "MISS Curry 3PT" -> ["Curry"]

        Args:
            description: Event description string.
            player_name_to_id: Optional mapping of player names to IDs.

        Returns:
            List of extracted player names (or IDs if mapping provided).
        """
        if not description:
            return []

        # Known patterns for player mentions
        # 1. Player name at start: "Curry 25' 3PT"
        # 2. After MISS: "MISS Curry 3PT"
        # 3. After Assist: "(Assist: Thompson)"
        # 4. After Block: "(Block: Green)"
        # 5. After Steal: "(Steal: Curry)"
        # 6. SUB patterns: "SUB: Curry FOR Thompson"

        player_names: list[str] = []

        # Pattern: Assist/Block/Steal: Name
        import re

        assist_match = re.search(r"\((?:Assist|AST):\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\)", description)
        if assist_match:
            player_names.append(assist_match.group(1).strip())

        block_match = re.search(r"\((?:Block|BLK):\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\)", description)
        if block_match:
            player_names.append(block_match.group(1).strip())

        steal_match = re.search(r"\((?:Steal|STL):\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\)", description)
        if steal_match:
            player_names.append(steal_match.group(1).strip())

        # Pattern: SUB: Name FOR Name
        sub_match = re.search(r"SUB:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+FOR\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", description)
        if sub_match:
            player_names.append(sub_match.group(1).strip())
            player_names.append(sub_match.group(2).strip())

        # Pattern: Starting name (e.g., "Curry 25' 3PT")
        # Skip common prefixes like MISS, REBOUND
        clean_desc = description.strip()
        for prefix in ["MISS ", "MISSED ", "REBOUND "]:
            if clean_desc.upper().startswith(prefix):
                clean_desc = clean_desc[len(prefix):]
                break

        # First word that looks like a name
        first_word_match = re.match(r"^([A-Z][a-z]+(?:\.\s[A-Z][a-z]+)?)\s", clean_desc)
        if first_word_match:
            name = first_word_match.group(1)
            # Skip common non-name words
            if name.upper() not in ["FREE", "TURNOVER", "FOUL", "JUMP", "OFFENSIVE", "DEFENSIVE"]:
                player_names.append(name)

        # Convert to player IDs if mapping provided
        if player_name_to_id:
            player_ids = []
            for name in player_names:
                # Try exact match first
                if name in player_name_to_id:
                    player_ids.append(player_name_to_id[name])
                else:
                    # Try case-insensitive match
                    for known_name, pid in player_name_to_id.items():
                        if known_name.lower() == name.lower():
                            player_ids.append(pid)
                            break
            return player_ids

        # Return names as placeholder (caller can resolve to IDs)
        return []  # Return empty if no ID mapping - names aren't IDs

    def _safe_int(self, value: object) -> int | None:
        """Safely convert value to int.

        Args:
            value: Value to convert.

        Returns:
            Integer or None if invalid/missing.
        """
        if value is None or pd.isna(value):
            return None
        try:
            int_val = int(value)
            # NBA API sometimes uses 0 for missing player IDs
            return int_val if int_val != 0 else None
        except (ValueError, TypeError):
            return None
