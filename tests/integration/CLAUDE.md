# Integration Tests

## Purpose

Cross-module tests that verify multiple components work together correctly. Tests data flow through the system, not isolated units.

## Files

| File | Purpose |
|------|---------|
| `__init__.py` | Package marker |
| `test_data_pipeline.py` | Data collection flow (Phase 2) |
| `test_feature_pipeline.py` | Feature engineering flow (Phase 3) |
| `test_training_pipeline.py` | Model training flow (Phase 4) |
| `test_backtest_pipeline.py` | Backtesting flow (Phase 5) |
| `test_monitor_pipeline.py` | Monitoring flow (Phase 6) |
| `test_prediction_pipeline.py` | Inference flow (Phase 7) |

## Patterns

Integration tests should:
1. Use in-memory SQLite database (not production)
2. Mock external APIs (NBA API, etc.)
3. Test realistic data flows
4. Verify cross-module interactions

```python
@pytest.mark.integration
class TestDataPipeline:
    """Integration tests for data collection pipeline."""

    def test_collect_and_store_games(
        self,
        mock_nba_api: Mock,
        db_session: Session,
    ) -> None:
        """Collected games should be stored in database."""
        # Setup mock API response
        mock_nba_api.get_games.return_value = sample_games

        # Run pipeline
        pipeline = DataPipeline(api=mock_nba_api, db=db_session)
        result = pipeline.collect_season("2023-24")

        # Verify storage
        stored = db_session.query(Game).all()
        assert len(stored) == len(sample_games)
```

## Running

```bash
# All integration tests
pytest tests/integration/ -m integration

# Specific pipeline
pytest tests/integration/test_data_pipeline.py
```

## Anti-Patterns

- Never call real external APIs
- Never use production database paths
- Never skip cleanup in teardown
- Never test implementation details (test behavior)
