# Tests for predict

## Status

✅ **Complete** - Unit tests for Phase 7 prediction module.

## Test Files

| File | Tests | Coverage |
|------|-------|----------|
| `test_inference.py` | InferencePipeline, GamePrediction | ~85% |
| `test_injuries.py` | InjuryAdjuster, Bayesian priors | ~90% |
| `test_signals.py` | SignalGenerator, BettingSignal | ~90% |

## Test Categories

### Inference Tests (`test_inference.py`)

- **GamePrediction**: Dataclass creation and default values
- **PredictionBatch**: Batch result container
- **InferencePipeline**: Initialization, lazy loading, model errors
- **Bounds**: Win probability, margin, and total bounds
- **Helpers**: Feature building, lineup extraction

### Injury Tests (`test_injuries.py`)

- **Constants**: Prior probabilities, modifiers
- **InjuryStatus**: Enum values
- **parse_injury_status**: Status parsing with aliases
- **InjuryAdjuster**: Bayesian probability calculation
- **Replacement Impact**: RAPM-based impact calculation

### Signal Tests (`test_signals.py`)

- **MarketOdds**: Odds container creation
- **BettingSignal**: Signal dataclass
- **Odds Conversion**: American ↔ Decimal conversion
- **SignalGenerator**: Signal generation with edge filter
- **Confidence**: Classification thresholds

## Running Tests

```bash
# Run predict unit tests
pytest tests/unit/predict/ -v

# With coverage
pytest tests/unit/predict/ --cov=nba_model.predict

# Skip slow tests
pytest tests/unit/predict/ -m "not slow"
```

## Key Fixtures

From `conftest.py`:
- `sample_prediction_result` - Sample prediction dict
- `sample_betting_signal` - Sample signal dict
- `frozen_today` - Fixed date for deterministic tests

## Test Patterns

```python
class TestInjuryAdjuster:
    @pytest.fixture
    def mock_session(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def adjuster(self, mock_session) -> InjuryAdjuster:
        return InjuryAdjuster(mock_session)

    def test_out_returns_zero(self, adjuster):
        prob = adjuster.get_play_probability(12345, "out")
        assert prob == 0.0
```

## Anti-Patterns

- ❌ Never test with real NBA API (use mocks)
- ❌ Never use real models in unit tests (mock registry)
- ❌ Never share state between tests (use fixtures)
