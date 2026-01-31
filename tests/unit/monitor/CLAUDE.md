# Tests for monitor module

## Status

**Phase 6 - Complete**

## Test Files

| File | Coverage | Tests For |
|------|----------|-----------|
| `test_drift.py` | DriftDetector, ConceptDriftDetector | KS tests, PSI, accuracy/Brier |
| `test_triggers.py` | RetrainingTrigger | Scheduled, drift, performance, data triggers |
| `test_versioning.py` | ModelVersionManager | Version CRUD, comparison, promotion, rollback |

## Test Categories

### Drift Detection Tests
- KS test returns valid statistics for identical/shifted distributions
- PSI calculation returns expected values (> 0.2 for heavy shifts)
- check_drift aggregates results correctly
- Empty dataframe handling tests
- Missing feature column handling

### Trigger Tests
- Each trigger type activates at correct thresholds
- Priority assignment logic
- Multiple trigger combination scenarios
- Dict and TriggerContext input handling

### Versioning Tests
- Version string generation follows semantic versioning
- Metadata correctly written and read (includes created_at)
- compare_versions uses correct metric priorities
- compare_versions with live inference on test_data (win_prob column)
- compare_versions handles empty test_data gracefully
- Promote/rollback update status correctly
- Lineage traverses parent chain
- Version listing sorted by creation date descending

## Integration Tests

See `tests/integration/test_monitor_pipeline.py` for:
- Drift pipeline end-to-end tests
- Trigger pipeline integration tests
- Version lifecycle tests
- Complete monitoring cycle test

## Running Tests

```bash
# Run all monitor tests
pytest tests/unit/monitor/ -v

# Run with coverage
pytest tests/unit/monitor/ --cov=nba_model.monitor

# Run specific test file
pytest tests/unit/monitor/test_drift.py -v

# Run integration tests
pytest tests/integration/test_monitor_pipeline.py -v
```
