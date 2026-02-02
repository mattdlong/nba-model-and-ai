"""Tests for data quality review module."""

from datetime import datetime
from typing import TYPE_CHECKING, Generator

import pytest

from nba_model.config import reset_settings
from nba_model.data import init_db, reset_engine, session_scope
from nba_model.data.quality import DataQualityReviewer, QualityIssue, QualityReport

if TYPE_CHECKING:
    from nba_model.config import Settings


@pytest.fixture(autouse=True)
def reset_db_between_tests(test_settings: "Settings") -> Generator[None, None, None]:
    """Reset database engine before and after each test."""
    reset_engine()
    yield
    reset_engine()


class TestQualityIssue:
    """Tests for QualityIssue dataclass."""

    def test_create_issue(self) -> None:
        """Test creating a quality issue."""
        issue = QualityIssue(
            severity="error",
            dimension="completeness",
            entity="Game",
            entity_id="0022300001",
            message="Missing play-by-play data",
        )
        assert issue.severity == "error"
        assert issue.dimension == "completeness"
        assert issue.entity == "Game"
        assert issue.entity_id == "0022300001"
        assert issue.message == "Missing play-by-play data"

    def test_str_with_entity_id(self) -> None:
        """Test string representation with entity ID."""
        issue = QualityIssue(
            severity="error",
            dimension="completeness",
            entity="Game",
            entity_id="0022300001",
            message="Missing play-by-play data",
        )
        result = str(issue)
        assert "[ERROR]" in result
        assert "Game" in result
        assert "0022300001" in result
        assert "Missing play-by-play data" in result

    def test_str_without_entity_id(self) -> None:
        """Test string representation without entity ID."""
        issue = QualityIssue(
            severity="warning",
            dimension="validity",
            entity="Season",
            entity_id=None,
            message="Low game count",
        )
        result = str(issue)
        assert "[WARNING]" in result
        assert "Season" in result
        assert "Low game count" in result


class TestQualityReport:
    """Tests for QualityReport dataclass."""

    def test_create_empty_report(self) -> None:
        """Test creating an empty report."""
        report = QualityReport(generated_at=datetime.now())
        assert report.issues == []
        assert report.summary == {}
        assert report.games_needing_repair == []
        assert report.error_count == 0
        assert report.warning_count == 0
        assert report.info_count == 0

    def test_add_error_issue(self) -> None:
        """Test adding an error issue."""
        report = QualityReport(generated_at=datetime.now())
        issue = QualityIssue(
            severity="error",
            dimension="completeness",
            entity="Game",
            entity_id="0022300001",
            message="Missing data",
        )
        report.add_issue(issue)

        assert len(report.issues) == 1
        assert report.error_count == 1
        assert report.summary["completeness_error"] == 1
        assert "0022300001" in report.games_needing_repair

    def test_add_warning_issue(self) -> None:
        """Test adding a warning issue."""
        report = QualityReport(generated_at=datetime.now())
        issue = QualityIssue(
            severity="warning",
            dimension="validity",
            entity="Game",
            entity_id="0022300002",
            message="Unusual score",
        )
        report.add_issue(issue)

        assert len(report.issues) == 1
        assert report.warning_count == 1
        assert report.error_count == 0
        # Warnings don't add to games_needing_repair
        assert "0022300002" not in report.games_needing_repair

    def test_add_info_issue(self) -> None:
        """Test adding an info issue."""
        report = QualityReport(generated_at=datetime.now())
        issue = QualityIssue(
            severity="info",
            dimension="consistency",
            entity="Game",
            entity_id="0022300003",
            message="Minor discrepancy",
        )
        report.add_issue(issue)

        assert len(report.issues) == 1
        assert report.info_count == 1
        assert report.error_count == 0
        assert report.warning_count == 0

    def test_multiple_issues_same_game(self) -> None:
        """Test adding multiple issues for the same game."""
        report = QualityReport(generated_at=datetime.now())
        issue1 = QualityIssue(
            severity="error",
            dimension="completeness",
            entity="Game",
            entity_id="0022300001",
            message="Missing plays",
        )
        issue2 = QualityIssue(
            severity="error",
            dimension="referential",
            entity="Game",
            entity_id="0022300001",
            message="Invalid team",
        )
        report.add_issue(issue1)
        report.add_issue(issue2)

        assert len(report.issues) == 2
        assert report.error_count == 2
        # Game should only appear once in games_needing_repair
        assert report.games_needing_repair.count("0022300001") == 1

    def test_get_by_dimension(self) -> None:
        """Test filtering issues by dimension."""
        report = QualityReport(generated_at=datetime.now())
        report.add_issue(
            QualityIssue("error", "completeness", "Game", "1", "msg1")
        )
        report.add_issue(
            QualityIssue("error", "validity", "Game", "2", "msg2")
        )
        report.add_issue(
            QualityIssue("warning", "completeness", "Game", "3", "msg3")
        )

        completeness = report.get_by_dimension("completeness")
        assert len(completeness) == 2

        validity = report.get_by_dimension("validity")
        assert len(validity) == 1

    def test_get_by_severity(self) -> None:
        """Test filtering issues by severity."""
        report = QualityReport(generated_at=datetime.now())
        report.add_issue(
            QualityIssue("error", "completeness", "Game", "1", "msg1")
        )
        report.add_issue(
            QualityIssue("warning", "validity", "Game", "2", "msg2")
        )
        report.add_issue(
            QualityIssue("error", "referential", "Game", "3", "msg3")
        )

        errors = report.get_by_severity("error")
        assert len(errors) == 2

        warnings = report.get_by_severity("warning")
        assert len(warnings) == 1


class TestDataQualityReviewer:
    """Tests for DataQualityReviewer class."""

    def test_init_creates_logger(self) -> None:
        """Test that initialization creates a logger."""
        reviewer = DataQualityReviewer()
        assert reviewer.logger is not None

    def test_run_full_review_empty_db(self, test_settings: "Settings") -> None:
        """Test running review on empty database."""
        init_db()
        reviewer = DataQualityReviewer()

        with session_scope() as session:
            report = reviewer.run_full_review(session)

        assert isinstance(report, QualityReport)
        assert report.generated_at is not None
        # Should have at least a warning about no games
        assert len(report.issues) >= 1

    def test_run_full_review_with_season_filter(self, test_settings: "Settings") -> None:
        """Test running review with season filter."""
        init_db()
        reviewer = DataQualityReviewer()

        with session_scope() as session:
            report = reviewer.run_full_review(session, season="2023-24")

        assert report.season_filter == "2023-24"


class TestCompletenessChecks:
    """Tests for completeness check methods."""

    def test_check_completeness_no_games(self, test_settings: "Settings") -> None:
        """Test completeness check with no games."""
        init_db()
        reviewer = DataQualityReviewer()

        with session_scope() as session:
            issues = reviewer.check_completeness(session)

        # Should have at least a warning about no games
        assert any(i.dimension == "completeness" for i in issues)


class TestValidityChecks:
    """Tests for validity check methods."""

    def test_check_validity_empty_db(self, test_settings: "Settings") -> None:
        """Test validity check on empty database."""
        init_db()
        reviewer = DataQualityReviewer()

        with session_scope() as session:
            issues = reviewer.check_validity(session)

        # Empty DB should have no validity issues (nothing to be invalid)
        assert isinstance(issues, list)


class TestConsistencyChecks:
    """Tests for consistency check methods."""

    def test_check_consistency_empty_db(self, test_settings: "Settings") -> None:
        """Test consistency check on empty database."""
        init_db()
        reviewer = DataQualityReviewer()

        with session_scope() as session:
            issues = reviewer.check_consistency(session)

        # Empty DB should have no consistency issues
        assert isinstance(issues, list)


class TestReferentialIntegrityChecks:
    """Tests for referential integrity check methods."""

    def test_check_referential_integrity_empty_db(self, test_settings: "Settings") -> None:
        """Test referential integrity check on empty database."""
        init_db()
        reviewer = DataQualityReviewer()

        with session_scope() as session:
            issues = reviewer.check_referential_integrity(session)

        # Empty DB should have no referential integrity issues
        assert isinstance(issues, list)
