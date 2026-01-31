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
- PSI calculation returns expected values
- check_drift aggregates results correctly
- Edge case handling for empty/missing data

### Trigger Tests
- Each trigger type activates at correct thresholds
- Priority assignment logic
- Multiple trigger combination scenarios
- Dict and TriggerContext input handling

### Versioning Tests
- Version string generation follows semantic versioning
- Metadata correctly written and read
- compare_versions uses correct metric priorities
- Promote/rollback update status correctly
- Lineage traverses parent chain

## Running Tests

```bash
# Run all monitor tests
pytest tests/unit/monitor/ -v

# Run with coverage
pytest tests/unit/monitor/ --cov=nba_model.monitor

# Run specific test file
pytest tests/unit/monitor/test_drift.py -v
```
