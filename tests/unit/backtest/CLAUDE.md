# Tests for backtest

## Status

✅ **Phase 5 - Complete**

## Test Files

| File | Tests | Module |
|------|-------|--------|
| `test_devig.py` | Devigging methods | `nba_model/backtest/devig.py` |
| `test_kelly.py` | Kelly criterion sizing | `nba_model/backtest/kelly.py` |
| `test_metrics.py` | Performance metrics | `nba_model/backtest/metrics.py` |
| `test_engine.py` | Walk-forward validation | `nba_model/backtest/engine.py` |

## Coverage

Target: 80% minimum coverage for new code.

Run tests:
```bash
pytest tests/unit/backtest/ -v --cov=nba_model.backtest
```

## Key Test Cases

### Devigging
- Multiplicative devig produces probabilities summing to 1
- Power method converges within tolerance
- Shin's method produces valid probabilities
- Edge calculation returns correct sign/magnitude

### Kelly
- Full Kelly matches formula: f* = (bp - q) / b
- Fractional Kelly applies multiplier correctly
- Bet size capped at max_bet_pct
- Zero bet when edge below threshold
- Zero bet when Kelly fraction negative

### Metrics
- Sharpe ratio calculation
- Max drawdown calculation
- Win rate, ROI calculations
- CLV calculation with closing odds

### Engine
- Fold generation maintains chronological ordering
- Train/val sets are disjoint
- Minimum training size respected
- Backtest runs end-to-end with mock trainer
