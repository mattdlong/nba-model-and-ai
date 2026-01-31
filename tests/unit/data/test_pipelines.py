"""Tests for ETL pipeline orchestration.

Tests the CollectionPipeline class for orchestrating
data collection with checkpointing and batch processing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from nba_model.data.pipelines import (
    BatchResult,
    CollectionPipeline,
    PipelineResult,
    PipelineStatus,
)


class TestPipelineStatus:
    """Tests for PipelineStatus enum."""

    def test_pending_status(self) -> None:
        """Should have PENDING status."""
        assert PipelineStatus.PENDING.value == "pending"

    def test_running_status(self) -> None:
        """Should have RUNNING status."""
        assert PipelineStatus.RUNNING.value == "running"

    def test_completed_status(self) -> None:
        """Should have COMPLETED status."""
        assert PipelineStatus.COMPLETED.value == "completed"

    def test_failed_status(self) -> None:
        """Should have FAILED status."""
        assert PipelineStatus.FAILED.value == "failed"

    def test_paused_status(self) -> None:
        """Should have PAUSED status."""
        assert PipelineStatus.PAUSED.value == "paused"


class TestPipelineResult:
    """Tests for PipelineResult dataclass."""

    def test_create_with_status(self) -> None:
        """Should create result with status."""
        result = PipelineResult(status=PipelineStatus.RUNNING)

        assert result.status == PipelineStatus.RUNNING
        assert result.seasons_processed == []
        assert result.games_processed == 0
        assert result.plays_collected == 0
        assert result.shots_collected == 0
        assert result.stints_derived == 0
        assert result.errors == []
        assert result.duration_seconds == 0.0

    def test_create_full_result(self) -> None:
        """Should create result with all fields."""
        result = PipelineResult(
            status=PipelineStatus.COMPLETED,
            seasons_processed=["2022-23", "2023-24"],
            games_processed=100,
            plays_collected=50000,
            shots_collected=10000,
            stints_derived=2000,
            errors=["Error 1"],
            duration_seconds=3600.5,
        )

        assert result.status == PipelineStatus.COMPLETED
        assert len(result.seasons_processed) == 2
        assert result.games_processed == 100
        assert result.plays_collected == 50000
        assert result.shots_collected == 10000
        assert result.stints_derived == 2000
        assert len(result.errors) == 1
        assert result.duration_seconds == 3600.5


class TestBatchResult:
    """Tests for BatchResult dataclass."""

    def test_default_values(self) -> None:
        """Should have empty defaults."""
        result = BatchResult()

        assert result.game_ids == []
        assert result.plays == []
        assert result.shots == []
        assert result.game_stats == []
        assert result.player_game_stats == []
        assert result.stints == []
        assert result.errors == []

    def test_create_with_data(self) -> None:
        """Should create result with data."""
        result = BatchResult(
            game_ids=["0022300001", "0022300002"],
            plays=[{"id": 1}],
            shots=[{"id": 1}],
            game_stats=[{"id": 1}],
            player_game_stats=[{"id": 1}],
            stints=[{"id": 1}],
            errors=[("0022300003", "API error")],
        )

        assert len(result.game_ids) == 2
        assert len(result.plays) == 1
        assert len(result.shots) == 1
        assert len(result.errors) == 1

    def test_errors_are_tuples(self) -> None:
        """Errors should be (game_id, message) tuples."""
        result = BatchResult(
            errors=[("0022300001", "Connection timeout")],
        )

        game_id, message = result.errors[0]
        assert game_id == "0022300001"
        assert message == "Connection timeout"


class TestCollectionPipeline:
    """Tests for CollectionPipeline class."""

    @pytest.fixture
    def mock_session(self) -> MagicMock:
        """Create a mock database session."""
        return MagicMock()

    @pytest.fixture
    def mock_api_client(self) -> MagicMock:
        """Create a mock API client."""
        return MagicMock()

    @pytest.fixture
    def mock_checkpoint_manager(self) -> MagicMock:
        """Create a mock checkpoint manager."""
        return MagicMock()

    @pytest.fixture
    def pipeline(
        self,
        mock_session: MagicMock,
        mock_api_client: MagicMock,
        mock_checkpoint_manager: MagicMock,
    ) -> CollectionPipeline:
        """Create a pipeline with mocked dependencies."""
        with patch("nba_model.data.collectors.GamesCollector"):
            with patch("nba_model.data.collectors.PlayersCollector"):
                with patch("nba_model.data.collectors.PlayByPlayCollector"):
                    with patch("nba_model.data.collectors.ShotsCollector"):
                        with patch("nba_model.data.collectors.BoxScoreCollector"):
                            return CollectionPipeline(
                                session=mock_session,
                                api_client=mock_api_client,
                                checkpoint_manager=mock_checkpoint_manager,
                                batch_size=10,
                            )

    def test_init_sets_batch_size(
        self,
        mock_session: MagicMock,
        mock_api_client: MagicMock,
        mock_checkpoint_manager: MagicMock,
    ) -> None:
        """Should set batch size on init."""
        with patch("nba_model.data.collectors.GamesCollector"):
            with patch("nba_model.data.collectors.PlayersCollector"):
                with patch("nba_model.data.collectors.PlayByPlayCollector"):
                    with patch("nba_model.data.collectors.ShotsCollector"):
                        with patch("nba_model.data.collectors.BoxScoreCollector"):
                            pipeline = CollectionPipeline(
                                session=mock_session,
                                api_client=mock_api_client,
                                checkpoint_manager=mock_checkpoint_manager,
                                batch_size=25,
                            )

                            assert pipeline.batch_size == 25

    def test_init_default_batch_size(
        self,
        mock_session: MagicMock,
        mock_api_client: MagicMock,
        mock_checkpoint_manager: MagicMock,
    ) -> None:
        """Should default batch size to 50."""
        with patch("nba_model.data.collectors.GamesCollector"):
            with patch("nba_model.data.collectors.PlayersCollector"):
                with patch("nba_model.data.collectors.PlayByPlayCollector"):
                    with patch("nba_model.data.collectors.ShotsCollector"):
                        with patch("nba_model.data.collectors.BoxScoreCollector"):
                            pipeline = CollectionPipeline(
                                session=mock_session,
                                api_client=mock_api_client,
                                checkpoint_manager=mock_checkpoint_manager,
                            )

                            assert pipeline.batch_size == 50


class TestGetCurrentSeason:
    """Tests for _get_current_season method."""

    @pytest.fixture
    def mock_session(self) -> MagicMock:
        """Create a mock database session."""
        return MagicMock()

    @pytest.fixture
    def mock_api_client(self) -> MagicMock:
        """Create a mock API client."""
        return MagicMock()

    @pytest.fixture
    def mock_checkpoint_manager(self) -> MagicMock:
        """Create a mock checkpoint manager."""
        return MagicMock()

    @pytest.fixture
    def pipeline(
        self,
        mock_session: MagicMock,
        mock_api_client: MagicMock,
        mock_checkpoint_manager: MagicMock,
    ) -> CollectionPipeline:
        """Create a pipeline with mocked dependencies."""
        with patch("nba_model.data.collectors.GamesCollector"):
            with patch("nba_model.data.collectors.PlayersCollector"):
                with patch("nba_model.data.collectors.PlayByPlayCollector"):
                    with patch("nba_model.data.collectors.ShotsCollector"):
                        with patch("nba_model.data.collectors.BoxScoreCollector"):
                            return CollectionPipeline(
                                session=mock_session,
                                api_client=mock_api_client,
                                checkpoint_manager=mock_checkpoint_manager,
                            )

    def test_october_starts_new_season(self, pipeline: CollectionPipeline) -> None:
        """October should be in new season."""
        with patch("nba_model.data.pipelines.date") as mock_date:
            mock_date.today.return_value = date(2024, 10, 15)
            mock_date.side_effect = lambda *args, **kwargs: date(*args, **kwargs)

            season = pipeline._get_current_season()
            assert season == "2024-25"

    def test_january_in_current_season(self, pipeline: CollectionPipeline) -> None:
        """January should be in current season (started previous year)."""
        with patch("nba_model.data.pipelines.date") as mock_date:
            mock_date.today.return_value = date(2024, 1, 15)
            mock_date.side_effect = lambda *args, **kwargs: date(*args, **kwargs)

            season = pipeline._get_current_season()
            assert season == "2023-24"

    def test_june_in_current_season(self, pipeline: CollectionPipeline) -> None:
        """June should still be in current season (playoffs)."""
        with patch("nba_model.data.pipelines.date") as mock_date:
            mock_date.today.return_value = date(2024, 6, 15)
            mock_date.side_effect = lambda *args, **kwargs: date(*args, **kwargs)

            season = pipeline._get_current_season()
            assert season == "2023-24"


class TestRepairGames:
    """Tests for repair_games method."""

    @pytest.fixture
    def mock_session(self) -> MagicMock:
        """Create a mock database session."""
        session = MagicMock()
        # Configure query chain for delete operations
        query_mock = MagicMock()
        query_mock.filter.return_value = query_mock
        query_mock.delete.return_value = 0
        session.query.return_value = query_mock
        return session

    @pytest.fixture
    def mock_api_client(self) -> MagicMock:
        """Create a mock API client."""
        return MagicMock()

    @pytest.fixture
    def mock_checkpoint_manager(self) -> MagicMock:
        """Create a mock checkpoint manager."""
        return MagicMock()

    def test_repair_returns_result(
        self,
        mock_session: MagicMock,
        mock_api_client: MagicMock,
        mock_checkpoint_manager: MagicMock,
    ) -> None:
        """Should return PipelineResult."""
        with patch("nba_model.data.collectors.GamesCollector"):
            with patch("nba_model.data.collectors.PlayersCollector"):
                with patch("nba_model.data.collectors.PlayByPlayCollector") as mock_pbp:
                    with patch("nba_model.data.collectors.ShotsCollector") as mock_shots:
                        with patch(
                            "nba_model.data.collectors.BoxScoreCollector"
                        ) as mock_box:
                            # Configure mocks
                            mock_pbp.return_value.collect_game.return_value = []
                            mock_shots.return_value.collect_game.return_value = []
                            mock_box.return_value.collect_game.return_value = ([], [])

                            pipeline = CollectionPipeline(
                                session=mock_session,
                                api_client=mock_api_client,
                                checkpoint_manager=mock_checkpoint_manager,
                            )

                            result = pipeline.repair_games(["0022300001"])

                            assert isinstance(result, PipelineResult)
                            assert result.status == PipelineStatus.COMPLETED
                            assert result.duration_seconds >= 0


class TestFullHistoricalLoad:
    """Tests for full_historical_load method."""

    @pytest.fixture
    def mock_session(self) -> MagicMock:
        """Create a mock database session."""
        session = MagicMock()
        # Configure query for Season lookup
        query_mock = MagicMock()
        query_mock.filter.return_value = query_mock
        query_mock.first.return_value = None  # No existing season
        session.query.return_value = query_mock
        return session

    @pytest.fixture
    def mock_api_client(self) -> MagicMock:
        """Create a mock API client."""
        return MagicMock()

    @pytest.fixture
    def mock_checkpoint_manager(self) -> MagicMock:
        """Create a mock checkpoint manager."""
        manager = MagicMock()
        manager.load.return_value = None  # No existing checkpoint
        return manager

    def test_returns_pipeline_result(
        self,
        mock_session: MagicMock,
        mock_api_client: MagicMock,
        mock_checkpoint_manager: MagicMock,
    ) -> None:
        """Should return PipelineResult."""
        with patch("nba_model.data.collectors.GamesCollector") as mock_games:
            with patch("nba_model.data.collectors.PlayersCollector") as mock_players:
                with patch("nba_model.data.collectors.PlayByPlayCollector"):
                    with patch("nba_model.data.collectors.ShotsCollector"):
                        with patch("nba_model.data.collectors.BoxScoreCollector"):
                            # No games found
                            mock_games.return_value.collect_season.return_value = []
                            mock_players.return_value.collect_teams.return_value = []
                            mock_players.return_value.collect_rosters.return_value = (
                                [],
                                [],
                            )

                            pipeline = CollectionPipeline(
                                session=mock_session,
                                api_client=mock_api_client,
                                checkpoint_manager=mock_checkpoint_manager,
                            )

                            result = pipeline.full_historical_load(["2023-24"])

                            assert isinstance(result, PipelineResult)
                            assert result.status == PipelineStatus.COMPLETED

    def test_saves_checkpoint(
        self,
        mock_session: MagicMock,
        mock_api_client: MagicMock,
        mock_checkpoint_manager: MagicMock,
    ) -> None:
        """Should save checkpoint during execution."""
        with patch("nba_model.data.collectors.GamesCollector") as mock_games:
            with patch("nba_model.data.collectors.PlayersCollector") as mock_players:
                with patch("nba_model.data.collectors.PlayByPlayCollector"):
                    with patch("nba_model.data.collectors.ShotsCollector"):
                        with patch("nba_model.data.collectors.BoxScoreCollector"):
                            mock_games.return_value.collect_season.return_value = []
                            mock_players.return_value.collect_teams.return_value = []
                            mock_players.return_value.collect_rosters.return_value = (
                                [],
                                [],
                            )

                            pipeline = CollectionPipeline(
                                session=mock_session,
                                api_client=mock_api_client,
                                checkpoint_manager=mock_checkpoint_manager,
                            )

                            pipeline.full_historical_load(["2023-24"])

                            # Should have called save at least once
                            assert mock_checkpoint_manager.save.called


class TestIncrementalUpdate:
    """Tests for incremental_update method."""

    @pytest.fixture
    def mock_session(self) -> MagicMock:
        """Create a mock database session."""
        session = MagicMock()
        # Configure query chain
        query_mock = MagicMock()
        query_mock.filter.return_value = query_mock
        query_mock.order_by.return_value = query_mock
        query_mock.first.return_value = None  # No existing games
        query_mock.all.return_value = []
        session.query.return_value = query_mock
        return session

    @pytest.fixture
    def mock_api_client(self) -> MagicMock:
        """Create a mock API client."""
        return MagicMock()

    @pytest.fixture
    def mock_checkpoint_manager(self) -> MagicMock:
        """Create a mock checkpoint manager."""
        return MagicMock()

    def test_returns_completed_when_no_new_games(
        self,
        mock_session: MagicMock,
        mock_api_client: MagicMock,
        mock_checkpoint_manager: MagicMock,
    ) -> None:
        """Should return completed status when no new games."""
        with patch("nba_model.data.collectors.GamesCollector") as mock_games:
            with patch("nba_model.data.collectors.PlayersCollector"):
                with patch("nba_model.data.collectors.PlayByPlayCollector"):
                    with patch("nba_model.data.collectors.ShotsCollector"):
                        with patch("nba_model.data.collectors.BoxScoreCollector"):
                            # No new games from API
                            mock_games.return_value.collect_date_range.return_value = []

                            pipeline = CollectionPipeline(
                                session=mock_session,
                                api_client=mock_api_client,
                                checkpoint_manager=mock_checkpoint_manager,
                            )

                            result = pipeline.incremental_update()

                            assert result.status == PipelineStatus.COMPLETED
                            assert result.games_processed == 0
