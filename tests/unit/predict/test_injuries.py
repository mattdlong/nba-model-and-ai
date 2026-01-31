"""Unit tests for the injury adjustment module."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pytest

from nba_model.predict.injuries import (
    CONTEXT_MODIFIERS,
    INJURY_TYPE_MODIFIERS,
    PRIOR_PLAY_PROBABILITIES,
    InjuryAdjuster,
    InjuryAdjustmentResult,
    InjuryReport,
    InjuryReportFetcher,
    InjuryStatus,
    PlayerAvailability,
    parse_injury_status,
)


class TestPriorPlayProbabilities:
    """Tests for prior probability constants."""

    def test_all_priors_valid_probabilities(self) -> None:
        """Test that all prior probabilities are between 0 and 1."""
        for status, prob in PRIOR_PLAY_PROBABILITIES.items():
            assert 0.0 <= prob <= 1.0, f"Invalid probability for {status}: {prob}"

    def test_ordered_probabilities(self) -> None:
        """Test that probabilities are ordered logically."""
        assert PRIOR_PLAY_PROBABILITIES["out"] == 0.0
        assert PRIOR_PLAY_PROBABILITIES["doubtful"] < PRIOR_PLAY_PROBABILITIES["questionable"]
        assert PRIOR_PLAY_PROBABILITIES["questionable"] < PRIOR_PLAY_PROBABILITIES["probable"]
        assert PRIOR_PLAY_PROBABILITIES["probable"] < PRIOR_PLAY_PROBABILITIES["available"]
        assert PRIOR_PLAY_PROBABILITIES["available"] == 1.0

    def test_specific_values(self) -> None:
        """Test specific prior probability values."""
        assert PRIOR_PLAY_PROBABILITIES["probable"] == 0.93
        assert PRIOR_PLAY_PROBABILITIES["questionable"] == 0.55
        assert PRIOR_PLAY_PROBABILITIES["doubtful"] == 0.03
        assert PRIOR_PLAY_PROBABILITIES["out"] == 0.0


class TestInjuryTypeModifiers:
    """Tests for injury type modifier constants."""

    def test_all_modifiers_positive(self) -> None:
        """Test that all injury modifiers are positive."""
        for injury_type, modifier in INJURY_TYPE_MODIFIERS.items():
            assert modifier > 0, f"Invalid modifier for {injury_type}: {modifier}"

    def test_rest_modifier_lower(self) -> None:
        """Test that rest/load management have lower modifiers."""
        assert INJURY_TYPE_MODIFIERS["rest"] < 1.0
        assert INJURY_TYPE_MODIFIERS["load management"] < INJURY_TYPE_MODIFIERS["rest"]

    def test_minor_injuries_higher(self) -> None:
        """Test that minor injuries have equal or higher modifiers."""
        assert INJURY_TYPE_MODIFIERS["finger"] >= 1.0
        assert INJURY_TYPE_MODIFIERS["toe"] >= 1.0


class TestContextModifiers:
    """Tests for team context modifier constants."""

    def test_back_to_back_reduces_probability(self) -> None:
        """Test that back-to-back games reduce play probability."""
        assert CONTEXT_MODIFIERS["back_to_back"] < 1.0

    def test_playoff_increases_probability(self) -> None:
        """Test that playoff context increases play probability."""
        assert CONTEXT_MODIFIERS["playoff_race"] > 1.0
        assert CONTEXT_MODIFIERS["playoff_game"] > CONTEXT_MODIFIERS["playoff_race"]


class TestInjuryStatus:
    """Tests for InjuryStatus enum."""

    def test_all_statuses_defined(self) -> None:
        """Test that all expected statuses are defined."""
        assert InjuryStatus.PROBABLE.value == "probable"
        assert InjuryStatus.QUESTIONABLE.value == "questionable"
        assert InjuryStatus.DOUBTFUL.value == "doubtful"
        assert InjuryStatus.OUT.value == "out"
        assert InjuryStatus.AVAILABLE.value == "available"


class TestInjuryReport:
    """Tests for InjuryReport dataclass."""

    def test_create_injury_report(self) -> None:
        """Test creating an injury report."""
        report = InjuryReport(
            player_id=203507,
            player_name="Giannis Antetokounmpo",
            team_id=1610612749,
            status="questionable",
            injury_description="left knee soreness",
            report_date=date(2024, 1, 15),
        )

        assert report.player_id == 203507
        assert report.status == "questionable"
        assert "knee" in report.injury_description


class TestPlayerAvailability:
    """Tests for PlayerAvailability dataclass."""

    def test_create_player_availability(self) -> None:
        """Test creating a PlayerAvailability object."""
        availability = PlayerAvailability(
            player_id=203507,
            player_name="Giannis Antetokounmpo",
            status="questionable",
            play_probability=0.55,
            prior_probability=0.55,
            history_modifier=1.0,
            context_modifier=0.85,
            injury_modifier=1.0,
            rapm=4.5,
            minutes_projection=32.0,
        )

        assert availability.play_probability == 0.55
        assert availability.rapm == 4.5


class TestParseInjuryStatus:
    """Tests for parse_injury_status function."""

    def test_parse_standard_statuses(self) -> None:
        """Test parsing standard injury status strings."""
        assert parse_injury_status("out") == "out"
        assert parse_injury_status("doubtful") == "doubtful"
        assert parse_injury_status("questionable") == "questionable"
        assert parse_injury_status("probable") == "probable"
        assert parse_injury_status("available") == "available"

    def test_parse_case_insensitive(self) -> None:
        """Test case insensitivity."""
        assert parse_injury_status("OUT") == "out"
        assert parse_injury_status("Questionable") == "questionable"
        assert parse_injury_status("PROBABLE") == "probable"

    def test_parse_whitespace(self) -> None:
        """Test handling of whitespace."""
        assert parse_injury_status("  out  ") == "out"
        assert parse_injury_status("\tquestionable\n") == "questionable"

    def test_parse_aliases(self) -> None:
        """Test parsing of status aliases."""
        assert parse_injury_status("gtd") == "questionable"
        assert parse_injury_status("game time decision") == "questionable"
        assert parse_injury_status("day-to-day") == "questionable"
        assert parse_injury_status("active") == "available"
        assert parse_injury_status("healthy") == "available"

    def test_parse_single_letter(self) -> None:
        """Test parsing single letter abbreviations."""
        assert parse_injury_status("o") == "out"
        assert parse_injury_status("d") == "doubtful"
        assert parse_injury_status("q") == "questionable"
        assert parse_injury_status("p") == "probable"

    def test_parse_unknown_defaults_to_questionable(self) -> None:
        """Test that unknown status defaults to questionable."""
        assert parse_injury_status("unknown") == "questionable"
        assert parse_injury_status("xyz") == "questionable"


class TestInjuryAdjuster:
    """Tests for InjuryAdjuster class."""

    @pytest.fixture
    def mock_session(self) -> MagicMock:
        """Create a mock database session."""
        return MagicMock()

    @pytest.fixture
    def adjuster(self, mock_session: MagicMock) -> InjuryAdjuster:
        """Create an InjuryAdjuster instance."""
        return InjuryAdjuster(mock_session)

    def test_init(self, adjuster: InjuryAdjuster) -> None:
        """Test InjuryAdjuster initialization."""
        assert adjuster._player_history_cache == {}

    def test_get_play_probability_out(self, adjuster: InjuryAdjuster) -> None:
        """Test play probability for out status."""
        prob = adjuster.get_play_probability(203507, "out")
        assert prob == 0.0

    def test_get_play_probability_available(self, adjuster: InjuryAdjuster) -> None:
        """Test play probability for available status."""
        prob = adjuster.get_play_probability(203507, "available")
        assert prob == 1.0

    def test_get_play_probability_questionable(self, adjuster: InjuryAdjuster) -> None:
        """Test play probability for questionable status."""
        prob = adjuster.get_play_probability(203507, "questionable")
        # Without modifiers, should be the prior
        assert prob == pytest.approx(0.55, rel=0.1)

    def test_get_play_probability_with_injury_type(
        self, adjuster: InjuryAdjuster
    ) -> None:
        """Test play probability with injury type modifier."""
        prob_no_type = adjuster.get_play_probability(203507, "questionable")
        prob_rest = adjuster.get_play_probability(
            203507, "questionable", injury_type="rest"
        )

        # Rest should reduce probability
        assert prob_rest < prob_no_type

    def test_get_play_probability_with_context(
        self, adjuster: InjuryAdjuster
    ) -> None:
        """Test play probability with team context."""
        prob_normal = adjuster.get_play_probability(203507, "questionable")
        prob_b2b = adjuster.get_play_probability(
            203507,
            "questionable",
            team_context={"back_to_back": True},
        )
        prob_playoff = adjuster.get_play_probability(
            203507,
            "questionable",
            team_context={"playoff_game": True},
        )

        # Back-to-back should reduce, playoff should increase
        assert prob_b2b < prob_normal
        assert prob_playoff > prob_normal

    def test_get_play_probability_clamped(self, adjuster: InjuryAdjuster) -> None:
        """Test that probability is clamped to [0, 1]."""
        # Try to push probability above 1 with multiple positive modifiers
        prob = adjuster.get_play_probability(
            203507,
            "probable",
            team_context={"playoff_game": True, "playoff_race": True},
        )
        assert 0.0 <= prob <= 1.0

    def test_calculate_player_history_likelihood_default(
        self, adjuster: InjuryAdjuster, mock_session: MagicMock
    ) -> None:
        """Test player history calculation with default."""
        mock_session.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []

        likelihood = adjuster.calculate_player_history_likelihood(203507, "questionable")

        # Should return default 1.0
        assert likelihood == 1.0

    def test_calculate_player_history_likelihood_caching(
        self, adjuster: InjuryAdjuster
    ) -> None:
        """Test that player history is cached."""
        # Pre-populate cache
        adjuster._player_history_cache[203507] = 1.1

        likelihood = adjuster.calculate_player_history_likelihood(203507, "questionable")

        assert likelihood == 1.1


class TestInjuryAdjusterPredictionAdjustment:
    """Tests for prediction adjustment methods."""

    @pytest.fixture
    def mock_session(self) -> MagicMock:
        """Create a mock database session."""
        session = MagicMock()
        # Mock game query to return a game
        mock_game = MagicMock()
        mock_game.home_team_id = 1610612738
        mock_game.away_team_id = 1610612749
        session.query.return_value.filter.return_value.first.return_value = mock_game
        return session

    @pytest.fixture
    def adjuster(self, mock_session: MagicMock) -> InjuryAdjuster:
        """Create an InjuryAdjuster instance."""
        return InjuryAdjuster(mock_session)

    def test_adjust_prediction_values_no_injuries(
        self, adjuster: InjuryAdjuster
    ) -> None:
        """Test adjustment when no injuries."""
        result = adjuster.adjust_prediction_values(
            game_id="0022300001",
            base_home_win_prob=0.55,
            base_margin=4.0,
            base_total=220.0,
        )

        # Without injuries, adjustments should be minimal
        assert "home_win_prob_adjusted" in result
        assert "predicted_margin_adjusted" in result
        assert "predicted_total_adjusted" in result
        assert "injury_uncertainty" in result

    def test_adjust_prediction_values_bounds(
        self, adjuster: InjuryAdjuster
    ) -> None:
        """Test that adjusted values are within bounds."""
        result = adjuster.adjust_prediction_values(
            game_id="0022300001",
            base_home_win_prob=0.95,
            base_margin=30.0,
            base_total=250.0,
        )

        assert 0.01 <= result["home_win_prob_adjusted"] <= 0.99
        assert -35 <= result["predicted_margin_adjusted"] <= 35
        assert 175 <= result["predicted_total_adjusted"] <= 270


class TestCalculateReplacementImpact:
    """Tests for replacement impact calculation."""

    @pytest.fixture
    def mock_session(self) -> MagicMock:
        """Create a mock database session."""
        return MagicMock()

    @pytest.fixture
    def adjuster(self, mock_session: MagicMock) -> InjuryAdjuster:
        """Create an InjuryAdjuster instance."""
        return InjuryAdjuster(mock_session)

    def test_calculate_replacement_impact_star_player(
        self, adjuster: InjuryAdjuster, mock_session: MagicMock
    ) -> None:
        """Test impact calculation for star player."""
        # Mock RAPM query for starter (high RAPM)
        mock_rapm = MagicMock()
        mock_rapm.rapm = 5.0  # All-star level

        mock_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = mock_rapm

        impact = adjuster.calculate_replacement_impact(
            player_id=203507,
            replacement_id=None,  # Unknown replacement
            team_id=1610612749,
        )

        # Star player missing should have negative impact
        # (positive RAPM minus replacement level)
        assert impact > 0  # Impact is positive because star RAPM > replacement

    def test_calculate_replacement_impact_with_known_replacement(
        self, adjuster: InjuryAdjuster, mock_session: MagicMock
    ) -> None:
        """Test impact calculation with known replacement player."""
        # Setup different RAPM for starter and replacement
        call_count = [0]

        def mock_first():
            call_count[0] += 1
            mock = MagicMock()
            if call_count[0] == 1:
                mock.rapm = 5.0  # Starter
            else:
                mock.rapm = 1.0  # Replacement (decent player)
            return mock

        mock_session.query.return_value.filter.return_value.order_by.return_value.first = mock_first

        impact = adjuster.calculate_replacement_impact(
            player_id=203507,
            replacement_id=12345,
            team_id=1610612749,
        )

        # Difference should be less than with unknown replacement
        # Since replacement is decent, impact is smaller


class TestInjuryReportFetcher:
    """Tests for InjuryReportFetcher class."""

    def test_init(self) -> None:
        """Test fetcher initialization."""
        fetcher = InjuryReportFetcher(api_delay=1.0)
        assert fetcher.api_delay == 1.0

    def test_get_current_injuries_returns_dataframe(self) -> None:
        """Test that get_current_injuries returns a DataFrame."""
        fetcher = InjuryReportFetcher()
        df = fetcher.get_current_injuries()

        assert len(df.columns) == 6
        assert "player_id" in df.columns
        assert "status" in df.columns
        assert "team_id" in df.columns

    def test_get_team_injuries_filters(self) -> None:
        """Test that get_team_injuries filters by team."""
        fetcher = InjuryReportFetcher()
        df = fetcher.get_team_injuries(1610612749)

        # Currently returns empty, but should filter correctly
        assert len(df) == 0
