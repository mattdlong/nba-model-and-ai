"""Stint derivation logic from play-by-play data.

This module derives lineup stints from play-by-play substitution events.
A stint is a continuous period where the 5-player lineup remains unchanged.

Stint boundaries occur at:
- Substitution events (EVENTMSGTYPE = 8)
- Period start/end events
- Game start/end

Example:
    >>> from nba_model.data.stints import StintDeriver
    >>> deriver = StintDeriver()
    >>> stints = deriver.derive_stints(plays, "0022300001")
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nba_model.data.models import Play, Stint

logger = logging.getLogger(__name__)


# Event type constants from play-by-play data
EVENT_FIELD_GOAL_MADE = 1
EVENT_FIELD_GOAL_MISSED = 2
EVENT_FREE_THROW = 3
EVENT_REBOUND = 4
EVENT_TURNOVER = 5
EVENT_FOUL = 6
EVENT_VIOLATION = 7
EVENT_SUBSTITUTION = 8
EVENT_TIMEOUT = 9
EVENT_JUMP_BALL = 10
EVENT_EJECTION = 11
EVENT_PERIOD_START = 12
EVENT_PERIOD_END = 13

# Minutes per regulation period
PERIOD_MINUTES = 12
# Minutes per overtime period
OT_MINUTES = 5


@dataclass
class LineupChange:
    """Represents a lineup change event.

    Attributes:
        event_num: Play event number.
        period: Game period (1-4, 5+ for OT).
        pc_time: Period clock time string "MM:SS".
        team_id: Team making the substitution.
        player_in: Player ID entering the game.
        player_out: Player ID leaving the game.
        game_seconds: Total seconds from game start.
    """

    event_num: int
    period: int
    pc_time: str
    team_id: int
    player_in: int
    player_out: int
    game_seconds: int = 0


@dataclass
class StintData:
    """Internal representation of a stint before conversion to model.

    Attributes:
        game_id: Game ID.
        team_id: Team ID.
        lineup: Sorted list of 5 player IDs.
        start_event_num: Event number at stint start.
        end_event_num: Event number at stint end.
        start_time: Start time in game seconds.
        end_time: End time in game seconds.
        home_points: Points scored by home team during stint.
        away_points: Points scored by away team during stint.
        possessions: Estimated possessions during stint.
    """

    game_id: str
    team_id: int
    lineup: list[int]
    start_event_num: int
    end_event_num: int
    start_time: int
    end_time: int
    home_points: int = 0
    away_points: int = 0
    possessions: float = 0.0


class StintDeriver:
    """Derives lineup stints from play-by-play substitution events.

    Tracks lineup changes through substitutions and period boundaries
    to create stint records with outcomes.
    """

    def __init__(self) -> None:
        """Initialize stint deriver."""
        self.logger = logging.getLogger(self.__class__.__name__)

    def derive_stints(
        self,
        plays: list,
        game_id: str,
        home_team_id: int | None = None,
        away_team_id: int | None = None,
    ) -> list:
        """Derive all stints for a game from play-by-play data.

        Args:
            plays: List of Play objects for the game (sorted by event_num).
            game_id: Game ID for the stints.
            home_team_id: Home team ID (optional, inferred if not provided).
            away_team_id: Away team ID (optional, inferred if not provided).

        Returns:
            List of Stint objects.
        """
        from nba_model.data.models import Stint

        if not plays:
            self.logger.warning(f"No plays provided for game {game_id}")
            return []

        # Sort plays by event_num
        plays = sorted(plays, key=lambda p: p.event_num)

        # Infer team IDs if not provided
        if home_team_id is None or away_team_id is None:
            home_team_id, away_team_id = self._infer_team_ids(plays)

        if home_team_id is None or away_team_id is None:
            self.logger.error(f"Could not determine team IDs for game {game_id}")
            return []

        self.logger.debug(
            f"Deriving stints for {game_id}: "
            f"home={home_team_id}, away={away_team_id}, plays={len(plays)}"
        )

        # Get starting lineups
        try:
            home_lineup, away_lineup = self._get_starting_lineups(
                plays, home_team_id, away_team_id
            )
        except ValueError as e:
            self.logger.error(f"Could not determine starting lineups: {e}")
            return []

        if len(home_lineup) != 5 or len(away_lineup) != 5:
            self.logger.warning(
                f"Incomplete starting lineups: home={len(home_lineup)}, "
                f"away={len(away_lineup)}"
            )
            # Try to continue with incomplete lineups
            if len(home_lineup) < 5 or len(away_lineup) < 5:
                return []

        # Track lineup changes through game
        lineup_changes = self._track_substitutions(
            plays, home_lineup.copy(), away_lineup.copy(), home_team_id, away_team_id
        )

        # Build stints from lineup changes
        stint_data_list = self._build_stints(
            plays,
            home_lineup,
            away_lineup,
            lineup_changes,
            game_id,
            home_team_id,
            away_team_id,
        )

        # Convert to Stint model objects
        stints = []
        for sd in stint_data_list:
            stint = Stint(
                game_id=sd.game_id,
                team_id=sd.team_id,
                lineup_json=self._lineup_to_json(sd.lineup),
                start_time=sd.start_time,
                end_time=sd.end_time,
                home_points=sd.home_points,
                away_points=sd.away_points,
                possessions=sd.possessions,
            )
            stints.append(stint)

        self.logger.debug(f"Derived {len(stints)} stints for game {game_id}")
        return stints

    def _infer_team_ids(self, plays: list) -> tuple[int | None, int | None]:
        """Infer home and away team IDs from plays.

        Uses the first non-zero team IDs found. Assumes home team
        is more common in home descriptions.

        Args:
            plays: List of Play objects.

        Returns:
            Tuple of (home_team_id, away_team_id).
        """
        team_ids: set[int] = set()

        for play in plays:
            if play.player1_team_id and play.player1_team_id != 0:
                team_ids.add(play.player1_team_id)

            if len(team_ids) >= 2:
                break

        if len(team_ids) < 2:
            return None, None

        # Return sorted to be consistent (lower ID first as "home")
        sorted_ids = sorted(team_ids)
        return sorted_ids[0], sorted_ids[1]

    def _get_starting_lineups(
        self,
        plays: list,
        home_team_id: int,
        away_team_id: int,
    ) -> tuple[list[int], list[int]]:
        """Determine starting lineups from period 1 start events.

        Uses first 5 distinct player IDs for each team from period 1.

        Args:
            plays: List of Play objects.
            home_team_id: Home team ID.
            away_team_id: Away team ID.

        Returns:
            Tuple of (home_lineup, away_lineup) as sorted player ID lists.

        Raises:
            ValueError: If starting lineups cannot be determined.
        """
        home_players: list[int] = []
        away_players: list[int] = []

        # Look at period 1 plays until we have 5 players per team
        for play in plays:
            if play.period != 1:
                continue

            # Check player1
            if play.player1_id and play.player1_id != 0:
                if play.player1_team_id == home_team_id:
                    if play.player1_id not in home_players:
                        home_players.append(play.player1_id)
                elif play.player1_team_id == away_team_id:
                    if play.player1_id not in away_players:
                        away_players.append(play.player1_id)

            # Check player2 (usually for assists, steals, etc.)
            if play.player2_id and play.player2_id != 0:
                # player2 doesn't have team_id, need to infer
                pass

            if len(home_players) >= 5 and len(away_players) >= 5:
                break

        if len(home_players) < 5 or len(away_players) < 5:
            # Try to use jump ball event for starters
            for play in plays:
                if play.event_type == EVENT_JUMP_BALL and play.period == 1:
                    # Jump ball involves players from both teams
                    if play.player1_id and play.player1_id != 0:
                        if play.player1_team_id == home_team_id:
                            if play.player1_id not in home_players:
                                home_players.append(play.player1_id)
                        elif play.player1_team_id == away_team_id:
                            if play.player1_id not in away_players:
                                away_players.append(play.player1_id)

        return sorted(home_players[:5]), sorted(away_players[:5])

    def _track_substitutions(
        self,
        plays: list,
        home_lineup: list[int],
        away_lineup: list[int],
        home_team_id: int,
        away_team_id: int,
    ) -> list[LineupChange]:
        """Track all lineup changes through the game.

        Parses substitution events to track who enters/exits.

        Args:
            plays: List of Play objects.
            home_lineup: Starting home lineup.
            away_lineup: Starting away lineup.
            home_team_id: Home team ID.
            away_team_id: Away team ID.

        Returns:
            List of LineupChange objects with timestamps.
        """
        changes: list[LineupChange] = []
        current_period = 1

        for play in plays:
            # Track period changes
            if play.event_type == EVENT_PERIOD_START:
                current_period = play.period

            # Track substitutions
            if play.event_type == EVENT_SUBSTITUTION:
                # player1 = player entering, player2 = player leaving
                player_in = play.player1_id
                player_out = play.player2_id
                team_id = play.player1_team_id

                if player_in and player_out and team_id:
                    game_seconds = self._parse_time_to_seconds(
                        play.period, play.pc_time_string or "0:00"
                    )

                    change = LineupChange(
                        event_num=play.event_num,
                        period=play.period,
                        pc_time=play.pc_time_string or "0:00",
                        team_id=team_id,
                        player_in=player_in,
                        player_out=player_out,
                        game_seconds=game_seconds,
                    )
                    changes.append(change)

        return changes

    def _build_stints(
        self,
        plays: list,
        home_lineup: list[int],
        away_lineup: list[int],
        lineup_changes: list[LineupChange],
        game_id: str,
        home_team_id: int,
        away_team_id: int,
    ) -> list[StintData]:
        """Build stint data from lineup changes.

        Args:
            plays: List of Play objects.
            home_lineup: Starting home lineup.
            away_lineup: Starting away lineup.
            lineup_changes: List of LineupChange objects.
            game_id: Game ID.
            home_team_id: Home team ID.
            away_team_id: Away team ID.

        Returns:
            List of StintData objects.
        """
        stints: list[StintData] = []

        # Current lineups (mutable)
        current_home = home_lineup.copy()
        current_away = away_lineup.copy()

        # Track stint boundaries
        home_stint_start = 0  # Game seconds
        away_stint_start = 0
        home_start_event = plays[0].event_num if plays else 0
        away_start_event = plays[0].event_num if plays else 0

        # Group changes by game_seconds to handle simultaneous subs
        changes_by_time: dict[int, list[LineupChange]] = {}
        for change in lineup_changes:
            if change.game_seconds not in changes_by_time:
                changes_by_time[change.game_seconds] = []
            changes_by_time[change.game_seconds].append(change)

        # Process changes in time order
        for game_seconds in sorted(changes_by_time.keys()):
            time_changes = changes_by_time[game_seconds]

            # Process home team changes
            home_changes = [c for c in time_changes if c.team_id == home_team_id]
            if home_changes:
                # End current home stint
                end_event = home_changes[0].event_num
                stint = self._create_stint_data(
                    plays,
                    game_id,
                    home_team_id,
                    current_home,
                    home_start_event,
                    end_event,
                    home_stint_start,
                    game_seconds,
                )
                if stint:
                    stints.append(stint)

                # Apply all home substitutions
                for change in home_changes:
                    if change.player_out in current_home:
                        current_home.remove(change.player_out)
                    if change.player_in not in current_home:
                        current_home.append(change.player_in)
                current_home = sorted(current_home)

                # Start new stint
                home_stint_start = game_seconds
                home_start_event = end_event

            # Process away team changes
            away_changes = [c for c in time_changes if c.team_id == away_team_id]
            if away_changes:
                # End current away stint
                end_event = away_changes[0].event_num
                stint = self._create_stint_data(
                    plays,
                    game_id,
                    away_team_id,
                    current_away,
                    away_start_event,
                    end_event,
                    away_stint_start,
                    game_seconds,
                )
                if stint:
                    stints.append(stint)

                # Apply all away substitutions
                for change in away_changes:
                    if change.player_out in current_away:
                        current_away.remove(change.player_out)
                    if change.player_in not in current_away:
                        current_away.append(change.player_in)
                current_away = sorted(current_away)

                # Start new stint
                away_stint_start = game_seconds
                away_start_event = end_event

        # Final stints at game end
        if plays:
            last_play = plays[-1]
            game_end_seconds = self._parse_time_to_seconds(
                last_play.period, last_play.pc_time_string or "0:00"
            )
            end_event = last_play.event_num

            # Final home stint
            home_stint = self._create_stint_data(
                plays,
                game_id,
                home_team_id,
                current_home,
                home_start_event,
                end_event,
                home_stint_start,
                game_end_seconds,
            )
            if home_stint:
                stints.append(home_stint)

            # Final away stint
            away_stint = self._create_stint_data(
                plays,
                game_id,
                away_team_id,
                current_away,
                away_start_event,
                end_event,
                away_stint_start,
                game_end_seconds,
            )
            if away_stint:
                stints.append(away_stint)

        return stints

    def _create_stint_data(
        self,
        plays: list,
        game_id: str,
        team_id: int,
        lineup: list[int],
        start_event: int,
        end_event: int,
        start_time: int,
        end_time: int,
    ) -> StintData | None:
        """Create a stint data object.

        Args:
            plays: List of Play objects.
            game_id: Game ID.
            team_id: Team ID.
            lineup: List of 5 player IDs.
            start_event: Starting event number.
            end_event: Ending event number.
            start_time: Start time in game seconds.
            end_time: End time in game seconds.

        Returns:
            StintData object or None if invalid.
        """
        # Skip zero-length stints
        if start_time >= end_time:
            return None

        # Skip incomplete lineups
        if len(lineup) != 5:
            return None

        # Calculate outcomes
        home_pts, away_pts, poss = self._calculate_stint_outcomes(
            plays, start_event, end_event
        )

        return StintData(
            game_id=game_id,
            team_id=team_id,
            lineup=sorted(lineup),
            start_event_num=start_event,
            end_event_num=end_event,
            start_time=start_time,
            end_time=end_time,
            home_points=home_pts,
            away_points=away_pts,
            possessions=poss,
        )

    def _calculate_stint_outcomes(
        self,
        plays: list,
        stint_start: int,
        stint_end: int,
    ) -> tuple[int, int, float]:
        """Calculate points and possessions for a stint.

        Args:
            plays: List of Play objects.
            stint_start: Starting event number.
            stint_end: Ending event number.

        Returns:
            Tuple of (home_points, away_points, possessions).
        """
        home_points = 0
        away_points = 0
        possessions = 0.0

        for play in plays:
            if play.event_num < stint_start or play.event_num > stint_end:
                continue

            # Track scoring from score changes
            if play.score_home is not None and play.score_away is not None:
                # Score tracking is cumulative, need previous score
                # For simplicity, count made shots
                pass

            # Count made field goals
            if play.event_type == EVENT_FIELD_GOAL_MADE:
                # Determine points (2 or 3)
                pts = 2
                desc = (play.home_description or "") + (play.visitor_description or "")
                if "3PT" in desc.upper():
                    pts = 3

                # Determine which team scored
                # home_description non-null = home team scored
                if play.home_description:
                    home_points += pts
                else:
                    away_points += pts

            # Count made free throws
            if play.event_type == EVENT_FREE_THROW:
                desc = (play.home_description or "") + (play.visitor_description or "")
                if "MISS" not in desc.upper():
                    if play.home_description:
                        home_points += 1
                    else:
                        away_points += 1

        # Estimate possessions
        possessions = self._estimate_possessions(plays, stint_start, stint_end)

        return home_points, away_points, possessions

    def _estimate_possessions(
        self,
        plays: list,
        stint_start: int,
        stint_end: int,
    ) -> float:
        """Estimate possessions using event counting heuristics.

        Possession ends on:
        - Made field goal (not and-1)
        - Defensive rebound
        - Turnover
        - Made final free throw

        Args:
            plays: List of Play objects.
            stint_start: Starting event number.
            stint_end: Ending event number.

        Returns:
            Estimated possession count.
        """
        fga = 0
        fgm = 0
        turnovers = 0
        ft_attempts = 0
        offensive_rebounds = 0

        for play in plays:
            if play.event_num < stint_start or play.event_num > stint_end:
                continue

            if play.event_type == EVENT_FIELD_GOAL_MADE:
                fga += 1
                fgm += 1
            elif play.event_type == EVENT_FIELD_GOAL_MISSED:
                fga += 1
            elif play.event_type == EVENT_TURNOVER:
                turnovers += 1
            elif play.event_type == EVENT_FREE_THROW:
                ft_attempts += 1
            elif play.event_type == EVENT_REBOUND:
                # Check if offensive rebound
                desc = (play.home_description or "") + (play.visitor_description or "")
                if "OFF" in desc.upper():
                    offensive_rebounds += 1

        # Possession estimate formula
        # Possessions = FGA + 0.44 * FTA + TOV - OREB
        possessions = fga + 0.44 * ft_attempts + turnovers - offensive_rebounds

        return max(0.0, possessions)

    def _parse_time_to_seconds(self, period: int, pc_time: str) -> int:
        """Convert period and clock time to total game seconds elapsed.

        Args:
            period: Period number (1-4, or 5+ for OT).
            pc_time: Clock time string "MM:SS" or "M:SS".

        Returns:
            Total seconds from game start (0 = start of period 1).
        """
        try:
            parts = pc_time.split(":")
            if len(parts) == 2:
                minutes = int(parts[0])
                seconds = int(parts[1])
            else:
                minutes = 0
                seconds = 0
        except (ValueError, AttributeError):
            minutes = 0
            seconds = 0

        # Calculate seconds remaining in period
        clock_seconds = minutes * 60 + seconds

        if period <= 4:
            # Regulation periods (12 minutes each)
            period_length = PERIOD_MINUTES * 60
            seconds_before_period = (period - 1) * period_length
            seconds_into_period = period_length - clock_seconds
        else:
            # Overtime periods (5 minutes each)
            regulation_seconds = 4 * PERIOD_MINUTES * 60
            ot_period = period - 4
            period_length = OT_MINUTES * 60
            seconds_before_period = regulation_seconds + (ot_period - 1) * period_length
            seconds_into_period = period_length - clock_seconds

        return seconds_before_period + seconds_into_period

    def _lineup_to_json(self, player_ids: list[int]) -> str:
        """Convert sorted player ID list to JSON string.

        Args:
            player_ids: List of player IDs.

        Returns:
            JSON string of sorted player IDs.
        """
        return json.dumps(sorted(player_ids))
